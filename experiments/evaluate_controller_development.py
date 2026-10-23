"""Paired, bounded validation evaluation for a governed LLM controller."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from statistics import mean
from pathlib import Path

from src.agentic_controller import AgenticControllerRuntime, ControllerDecision, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


class FullEvidencePolicy:
    def __init__(self) -> None: self.actions = iter(("acquire_network", "acquire_origin", "acquire_destination", "stop"))
    def choose(self, request): return ControllerDecision(next(self.actions))


def rows(connection, split, modulus):
    return connection.execute("""SELECT c.case_id,c.type,c.amount,c.oldbalance_org,c.oldbalance_dest,c.is_fraud,p.*,
      n.prior_unique_origins AS destination_prior_unique_origins,n.recent_new_origins AS destination_recent_new_origins
      FROM cases c JOIN behavior.profiles p USING(case_id)
      LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
      JOIN iid.assignments a USING(case_id) WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0) ORDER BY c.case_id""", (split, modulus))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bounded controller-versus-full-evidence development evaluation.")
    parser.add_argument("--model", default="qwen3:4b"); parser.add_argument("--goal-profile", default="balanced")
    parser.add_argument("--limit", type=int, default=50); parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--positive-limit", type=int); parser.add_argument("--negative-limit", type=int)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite")); parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite")); parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/qwen3_4b_paired_50.json")); args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database); connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", 45)); labels = [int(row["is_fraud"]) for row in train]
        weight = (len(labels) - sum(labels)) / sum(labels)
        scorer = fit_state_risk_scorers([safe_evidence_from_row(row) for row in train], labels, seed=20260924,
                                        sample_weights=[weight if label else 1.0 for label in labels])
        case_ids = []; positive_ids = []; negative_ids = []
        for row in rows(connection, "validation", 1):
            if scorer.predict_risk({"initial_evidence": safe_evidence_from_row(row)["initial_evidence"]}) >= args.entry_gate:
                case_id = int(row["case_id"])
                if args.positive_limit is None or args.negative_limit is None:
                    case_ids.append(case_id)
                    if len(case_ids) == args.limit: break
                elif int(row["is_fraud"]) and len(positive_ids) < args.positive_limit:
                    positive_ids.append(case_id)
                elif not int(row["is_fraud"]) and len(negative_ids) < args.negative_limit:
                    negative_ids.append(case_id)
                if args.positive_limit is not None and args.negative_limit is not None and len(positive_ids) == args.positive_limit and len(negative_ids) == args.negative_limit:
                    case_ids = sorted(positive_ids + negative_ids); break
    finally: connection.close()
    service = EvidenceService(args.database, args.profiles, split_database=args.split_database)
    runtime = AgenticControllerRuntime(service, scorer, entry_gate=args.entry_gate, budget=11.25)
    pairs = []
    for case_id in case_ids:
        controller = runtime.run(case_id, OllamaControllerPolicy(args.model, goal_profile=args.goal_profile)).as_dict()
        full = runtime.run(case_id, FullEvidencePolicy()).as_dict()
        pairs.append({"case_id": case_id, "controller": controller, "full_evidence": full,
                      "risk_delta": abs(controller["risk_trajectory"][-1]["risk"] - full["risk_trajectory"][-1]["risk"])})
    actions = Counter(step["proposed_action"] for pair in pairs for step in pair["controller"]["controller_steps"])
    report = {"purpose": "bounded validation controller development; no terminal decisions or test access", "model": args.model,
              "goal_profile": args.goal_profile, "entry_gate": args.entry_gate, "cases": len(pairs), "case_ids": case_ids,
              "cohort": {"positive_limit": args.positive_limit, "negative_limit": args.negative_limit},
              "controller_action_mix": actions, "invalid_actions": sum(not step["valid"] for pair in pairs for step in pair["controller"]["controller_steps"]),
              "mean_controller_cost": mean(pair["controller"]["total_cost"] for pair in pairs), "mean_full_evidence_cost": mean(pair["full_evidence"]["total_cost"] for pair in pairs),
              "mean_cost_reduction": mean(1 - pair["controller"]["total_cost"] / pair["full_evidence"]["total_cost"] for pair in pairs),
              "mean_absolute_risk_delta": mean(pair["risk_delta"] for pair in pairs), "max_absolute_risk_delta": max(pair["risk_delta"] for pair in pairs), "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"pairs", "case_ids"}}, indent=2, default=dict))


if __name__ == "__main__": main()
