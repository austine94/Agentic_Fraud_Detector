"""Select the sole transaction-only threshold for entering the LLM system."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from src.iid_split import attach_assignment
from src.risk_scoring import feature_vector_from_evidence, fit_state_risk_scorers, safe_evidence_from_row
from src.threshold_selection import precision_frontier, select_precision_threshold, select_recall_threshold, weighted_binary_metrics


def rows(connection: sqlite3.Connection, split: str, negative_modulus: int):
    return connection.execute("""
        SELECT c.type, c.amount, c.oldbalance_org, c.oldbalance_dest, c.is_fraud, p.*,
               n.prior_unique_origins AS destination_prior_unique_origins,
               n.recent_new_origins AS destination_recent_new_origins
        FROM cases c JOIN behavior.profiles p USING(case_id)
        LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
        JOIN iid.assignments a USING(case_id)
        WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0)
        ORDER BY c.case_id
    """, (split, negative_modulus))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only selection of the sole LLM entry threshold.")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--train-negative-modulus", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--precision-target", type=float, default=.90)
    parser.add_argument("--recall-target", type=float, default=.80)
    parser.add_argument("--selection-objective", choices=("precision", "recall"), default="recall")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", args.train_negative_modulus))
        train_examples = [safe_evidence_from_row(row) for row in train]
        train_labels = [int(row["is_fraud"]) for row in train]
        positive_weight = (len(train_labels) - sum(train_labels)) / sum(train_labels)
        scorer = fit_state_risk_scorers(train_examples, train_labels, seed=20260924,
                                        sample_weights=[positive_weight if label else 1.0 for label in train_labels],
                                        states=((),))
        cursor = rows(connection, "validation", 1)
        labels: list[int] = []; scores: list[float] = []
        while batch := cursor.fetchmany(args.batch_size):
            evidence = [safe_evidence_from_row(row) for row in batch]
            matrix = np.vstack([feature_vector_from_evidence({"initial_evidence": item["initial_evidence"]}) for item in evidence])
            scores.extend(scorer.models[()].predict_proba(matrix)[:, 1].tolist())
            labels.extend(int(row["is_fraud"]) for row in batch)
    finally:
        connection.close()
    weights = [1.0] * len(labels)
    gate = select_precision_threshold(labels, scores, weights, target_precision=args.precision_target)
    recall_gate = select_recall_threshold(labels, scores, weights, target_recall=args.recall_target)
    selected_gate = gate if args.selection_objective == "precision" else recall_gate
    recall_gate_record = dict(recall_gate.__dict__)
    recall_gate_record["target_recall"] = recall_gate_record.pop("target_precision")
    selected_gate_record = dict(selected_gate.__dict__)
    if args.selection_objective == "recall":
        selected_gate_record["target_recall"] = selected_gate_record.pop("target_precision")
    report = {
        "purpose": "sole transaction-only LLM-entry gate; no terminal payment decision is selected here",
        "status": "validation_gate_selected_and_frozen_for_orchestration_comparison", "score_state": "initial_evidence",
        "selection_objective": args.selection_objective, "precision_target": args.precision_target,
        "selected_gate": selected_gate_record,
        "recall_target_exploration": {"target": args.recall_target, "highest_gate_meeting_target": recall_gate_record,
                                       "metrics": weighted_binary_metrics(labels, scores, weights, threshold=recall_gate.threshold)},
        "validation_metrics_at_gate": weighted_binary_metrics(labels, scores, weights, threshold=selected_gate.threshold),
        "precision_frontier": precision_frontier(labels, scores, weights, targets=(.80, .85, .90, .95, .99)),
        "training_cases": len(train), "validation_cases": len(labels), "test_access": "not queried",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
