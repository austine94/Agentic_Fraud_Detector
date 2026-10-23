"""Run one deterministic evidence-acquisition comparator on validation only.

Threshold bands are explicit inputs selected during validation development. This
script intentionally contains no terminal payment-action logic.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean

from src.agentic_controller import AgenticControllerRuntime
from src.comparison_policies import (
    FullEvidencePolicy,
    SimpleThresholdCascadePolicy,
    StructuredRulePolicy,
    TransactionOnlyPolicy,
)
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


def _rows(connection: sqlite3.Connection, split: str, modulus: int):
    return connection.execute(
        """SELECT c.case_id,c.type,c.amount,c.oldbalance_org,c.oldbalance_dest,c.is_fraud,p.*,
          n.prior_unique_origins AS destination_prior_unique_origins,n.recent_new_origins AS destination_recent_new_origins
          FROM cases c JOIN behavior.profiles p USING(case_id)
          LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
          JOIN iid.assignments a USING(case_id)
          WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0) ORDER BY c.case_id""",
        (split, modulus),
    )


def _parse_bands(value: str) -> dict[str, tuple[float, float]]:
    try:
        raw = json.loads(value)
        return {name: (float(band[0]), float(band[1])) for name, band in raw.items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as error:
        raise argparse.ArgumentTypeError("bands must be JSON like '{\"initial\":[0.1,0.9],...}'") from error


def _policy(args: argparse.Namespace):
    if args.policy == "transaction_only":
        return TransactionOnlyPolicy(), {"kind": "transaction_only"}
    if args.policy == "full_evidence":
        return FullEvidencePolicy(), {"kind": "full_evidence"}
    if args.policy == "simple_cascade":
        if args.continue_low is None or args.continue_high is None:
            raise SystemExit("simple_cascade requires --continue-low and --continue-high")
        return SimpleThresholdCascadePolicy(args.continue_low, args.continue_high), {
            "kind": "simple_cascade", "continue_band": [args.continue_low, args.continue_high]
        }
    if args.structured_bands is None:
        raise SystemExit("structured_rule requires --structured-bands")
    return StructuredRulePolicy(_parse_bands(args.structured_bands)), {
        "kind": "structured_rule", "continuation_bands": json.loads(args.structured_bands)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only deterministic comparator evaluation.")
    parser.add_argument("--policy", choices=("transaction_only", "full_evidence", "simple_cascade", "structured_rule"), required=True)
    parser.add_argument("--continue-low", type=float); parser.add_argument("--continue-high", type=float)
    parser.add_argument("--structured-bands")
    parser.add_argument("--limit", type=int, default=50); parser.add_argument("--positive-limit", type=int); parser.add_argument("--negative-limit", type=int)
    parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--full-reference-report", type=Path,
                        help="reuse full-evidence traces from an earlier paired report with the same cohort")
    parser.add_argument("--label", help="display label for this frozen policy configuration")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite")); parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite")); parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()

    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database); connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(_rows(connection, "train", 45)); labels = [int(row["is_fraud"]) for row in train]
        positive_weight = (len(labels) - sum(labels)) / sum(labels)
        scorer = fit_state_risk_scorers([safe_evidence_from_row(row) for row in train], labels, seed=20260924,
                                        sample_weights=[positive_weight if label else 1.0 for label in labels])
        case_ids = []; positive_ids = []; negative_ids = []
        validation_rows = list(_rows(connection, "validation", 1))
        initial_rows = [{"initial_evidence": safe_evidence_from_row(row)["initial_evidence"]} for row in validation_rows]
        for row, score in zip(validation_rows, scorer.predict_risks(initial_rows), strict=True):
            if score >= args.entry_gate:
                case_id = int(row["case_id"])
                if args.positive_limit is None or args.negative_limit is None:
                    case_ids.append(case_id)
                    if len(case_ids) == args.limit:
                        break
                elif int(row["is_fraud"]) and len(positive_ids) < args.positive_limit:
                    positive_ids.append(case_id)
                elif not int(row["is_fraud"]) and len(negative_ids) < args.negative_limit:
                    negative_ids.append(case_id)
                if args.positive_limit is not None and args.negative_limit is not None and len(positive_ids) == args.positive_limit and len(negative_ids) == args.negative_limit:
                    case_ids = sorted(positive_ids + negative_ids)
                    break
    finally:
        connection.close()

    policy, policy_spec = _policy(args)
    runtime = AgenticControllerRuntime(EvidenceService(args.database, args.profiles, split_database=args.split_database), scorer,
                                       entry_gate=args.entry_gate, budget=11.25)
    reused_full = {}
    if args.full_reference_report is not None:
        reference = json.loads(args.full_reference_report.read_text(encoding="utf-8"))
        reused_full = {int(pair["case_id"]): pair["full_evidence"] for pair in reference["pairs"]}
        if set(reused_full) != set(case_ids):
            raise RuntimeError("full reference report does not match this selected cohort")
    pairs = []
    for case_id in case_ids:
        controller = runtime.run(case_id, policy).as_dict()
        full_evidence = reused_full.get(case_id) or runtime.run(case_id, FullEvidencePolicy()).as_dict()
        pairs.append({"case_id": case_id, "controller": controller, "full_evidence": full_evidence,
                      "risk_delta": abs(controller["risk_trajectory"][-1]["risk"] - full_evidence["risk_trajectory"][-1]["risk"])})
    traces = [pair["controller"] for pair in pairs]
    actions = Counter(step["proposed_action"] for trace in traces for step in trace["controller_steps"])
    report = {
        "purpose": "validation-only deterministic acquisition-policy trace; no terminal payment decisions or test access",
        "policy": policy_spec, "goal_profile": args.label or policy_spec["kind"], "entry_gate": args.entry_gate, "cases": len(traces), "case_ids": case_ids,
        "cohort": {"positive_limit": args.positive_limit, "negative_limit": args.negative_limit},
        "action_mix": actions, "mean_cost": mean(trace["total_cost"] for trace in traces) if traces else 0.0,
        "mean_stopping_depth": mean(sum(step["proposed_action"] != "stop" for step in trace["controller_steps"]) for trace in traces) if traces else 0.0,
        "mean_full_evidence_cost": mean(pair["full_evidence"]["total_cost"] for pair in pairs) if pairs else 0.0,
        "mean_absolute_risk_delta": mean(pair["risk_delta"] for pair in pairs) if pairs else 0.0,
        "invalid_actions": sum(not step["valid"] for trace in traces for step in trace["controller_steps"]), "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"pairs", "case_ids"}}, indent=2, default=dict))


if __name__ == "__main__":
    main()
