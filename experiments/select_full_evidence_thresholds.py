"""Compute full-evidence terminal thresholds from train and validation only."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from src.iid_split import attach_assignment
from src.risk_scoring import STATE_TOOLSETS, feature_vector_from_evidence, fit_state_risk_scorers, safe_evidence_from_row
from src.threshold_selection import precision_frontier, select_terminal_thresholds, weighted_binary_metrics


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
    parser = argparse.ArgumentParser(description="Validation-only threshold selection for the full-evidence model.")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--train-negative-modulus", type=int, default=45)
    parser.add_argument("--validation-negative-modulus", type=int, default=1,
                        help="Use 1 for complete validation selection; larger values sample negatives with restoring weights.")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--flag-precision-target", type=float, default=.95)
    parser.add_argument("--escalation-precision-target", type=float, default=.999)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", args.train_negative_modulus))
        examples = [safe_evidence_from_row(row) for row in train]
        labels = [int(row["is_fraud"]) for row in train]
        # Balance the sampled train loss for ranking, matching the established
        # evidence screen.  Validation weighting remains population-faithful.
        positive_weight = (len(labels) - sum(labels)) / sum(labels)
        train_weights = [positive_weight if label else 1.0 for label in labels]
        scorer = fit_state_risk_scorers(examples, labels, seed=20260924, sample_weights=train_weights)
        cursor = rows(connection, "validation", args.validation_negative_modulus)
        validation_labels: list[int] = []; validation_scores: list[float] = []
        while batch := cursor.fetchmany(args.batch_size):
            evidence = [safe_evidence_from_row(row) for row in batch]
            matrix = np.vstack([feature_vector_from_evidence(item) for item in evidence])
            validation_scores.extend(scorer.models[STATE_TOOLSETS[-1]].predict_proba(matrix)[:, 1].tolist())
            validation_labels.extend(int(row["is_fraud"]) for row in batch)
    finally:
        connection.close()
    validation_weights = [1.0 if label else float(args.validation_negative_modulus) for label in validation_labels]
    thresholds, selection = select_terminal_thresholds(
        validation_labels, validation_scores, validation_weights,
        flag_precision=args.flag_precision_target, escalation_precision=args.escalation_precision_target,
    )
    report = {
        "status": "validation_thresholds_selected_and_frozen_for_policy_comparison", "impact_scheme": "operational",
        "model": {"family": "HistGradientBoostingClassifier", "calibration": "sigmoid_cv3", "seed": 20260924},
        "sampling": {"train_negative_modulus": args.train_negative_modulus, "validation_negative_modulus": args.validation_negative_modulus,
                     "training_loss": "class_balanced_sample", "validation_weights_restore_negative_sampling_prevalence": True},
        "training_cases": len(train), "validation_cases": len(validation_labels),
        "threshold_selection": {"actionable_precision_target": args.flag_precision_target,
                                "escalation_precision_target": args.escalation_precision_target, "selected": selection},
        "terminal_thresholds": {"flag": thresholds.flag, "escalate": thresholds.escalate},
        "precision_frontier": precision_frontier(validation_labels, validation_scores, validation_weights,
                                                   targets=(.90, .95, .98, .99, .999)),
        "validation_metrics": weighted_binary_metrics(validation_labels, validation_scores, validation_weights, threshold=thresholds.flag),
        "test_access": "not queried",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
