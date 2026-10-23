"""Bounded real-model development run for the LLM evidence controller."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


def rows(connection: sqlite3.Connection, split: str, negative_modulus: int):
    return connection.execute("""
        SELECT c.case_id, c.type, c.amount, c.oldbalance_org, c.oldbalance_dest, c.is_fraud, p.*,
               n.prior_unique_origins AS destination_prior_unique_origins,
               n.recent_new_origins AS destination_recent_new_origins
        FROM cases c JOIN behavior.profiles p USING(case_id)
        LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
        JOIN iid.assignments a USING(case_id)
        WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0)
        ORDER BY c.case_id
    """, (split, negative_modulus))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded validation development loop with one local controller model.")
    parser.add_argument("--model", default="smollm:360m")
    parser.add_argument("--goal-profile", choices=("evidence_first", "balanced", "cost_conscious"), default="balanced")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--output", type=Path, default=Path("results/development/smollm_controller_sample.json"))
    args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", 45))
        examples = [safe_evidence_from_row(row) for row in train]
        labels = [int(row["is_fraud"]) for row in train]
        positive_weight = (len(labels) - sum(labels)) / sum(labels)
        scorer = fit_state_risk_scorers(examples, labels, seed=20260924,
                                        sample_weights=[positive_weight if label else 1.0 for label in labels])
        case_ids: list[int] = []
        for row in rows(connection, "validation", 1):
            initial = safe_evidence_from_row(row)["initial_evidence"]
            if scorer.predict_risk({"initial_evidence": initial}) >= args.entry_gate:
                case_ids.append(int(row["case_id"]))
                if len(case_ids) == args.limit:
                    break
    finally:
        connection.close()
    runtime = AgenticControllerRuntime(EvidenceService(args.database, args.profiles, split_database=args.split_database),
                                       scorer, entry_gate=args.entry_gate, budget=11.25)
    traces = [runtime.run(case_id, OllamaControllerPolicy(args.model, goal_profile=args.goal_profile)).as_dict() for case_id in case_ids]
    actions = Counter(step["proposed_action"] for trace in traces for step in trace["controller_steps"])
    report = {"purpose": "bounded validation prompt-development run; not a model comparison or final evaluation",
              "model": args.model, "goal_profile": args.goal_profile, "entry_gate": args.entry_gate, "cases": len(traces), "case_ids": case_ids,
              "statuses": Counter(trace["status"] for trace in traces), "action_mix": actions,
              "invalid_actions": sum(not step["valid"] for trace in traces for step in trace["controller_steps"]), "traces": traces}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"traces", "case_ids"}}, indent=2, default=dict))


if __name__ == "__main__":
    main()
