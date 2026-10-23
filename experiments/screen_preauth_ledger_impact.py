"""Validation-only screen for coarse counterfactual ledger-impact tools."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.iid_split import attach_assignment
from src.preauth_behavior import feature_vector as behavior_vector
from src.preauth_ledger import SCHEMES, features_from_row


TOOL_SETS = {
    "initial": ((), ()),
    "destination_network": (("destination_network",), ()),
    "origin_impact": ((), ("origin_impact",)),
    "destination_impact": ((), ("destination_impact",)),
    "both_ledger_impacts": ((), ("origin_impact", "destination_impact", "combined_impact")),
    "behavior_plus_ledger_impacts": (("destination_network",), ("origin_impact", "destination_impact", "combined_impact")),
}


def attach_profiles(connection: sqlite3.Connection, path: Path) -> None:
    connection.execute("ATTACH DATABASE ? AS behavior", (str(path),))


def rows(connection: sqlite3.Connection, split: str, negative_modulus: int):
    select = """SELECT c.case_id, c.type, c.amount, c.oldbalance_org, c.oldbalance_dest, c.is_fraud, p.*, 
                  n.prior_unique_origins AS destination_prior_unique_origins,
                  n.recent_new_origins AS destination_recent_new_origins,
                  t.total AS origin_last_transfer_incoming_total
           FROM cases c JOIN behavior.profiles p USING(case_id)
           LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
           LEFT JOIN behavior.transfer_incoming_step t ON t.account=c.name_orig AND t.step=c.step-p.origin_steps_since_transfer_incoming
    """
    return connection.execute(select + " JOIN iid.assignments a USING(case_id) WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0) ORDER BY c.case_id", (split, negative_modulus))


def matrix(connection: sqlite3.Connection, split: str, behavior_tools: tuple[str, ...], impact_tools: tuple[str, ...], scheme: str, modulus: int):
    vectors, labels = [], []
    for row in rows(connection, split, modulus):
        vectors.append(np.asarray(list(behavior_vector(row, behavior_tools)) + features_from_row(row, impact_tools, scheme), dtype=np.float32))
        labels.append(row["is_fraud"])
    return np.vstack(vectors), np.asarray(labels, dtype=np.int8)


def metric_summary(labels: np.ndarray, scores: np.ndarray, modulus: int) -> dict[str, float]:
    weights = np.where(labels == 1, 1.0, float(modulus))
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    recall_99 = recall[precision >= .99]
    return {"pr_auc_prevalence_adjusted": float(average_precision_score(labels, scores, sample_weight=weights)),
            "roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights)),
            "recall_at_99pct_precision": float(np.max(recall_99)) if recall_99.size else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-set", choices=sorted(TOOL_SETS), required=True)
    parser.add_argument("--impact-scheme", choices=sorted(SCHEMES), default="fine")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--train-negative-modulus", type=int, default=45)
    parser.add_argument("--validation-negative-modulus", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    behavior_tools, impact_tools = TOOL_SETS[args.tool_set]
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        split_protocol = "frozen stratified IID 50/10/40"
        attach_assignment(connection, args.split_database)
        metadata = connection.execute("SELECT value FROM iid.metadata WHERE key='protocol'").fetchone()
        if metadata is not None:
            try:
                split_protocol = str(json.loads(str(metadata[0])))
            except json.JSONDecodeError:
                split_protocol = str(metadata[0])
        attach_profiles(connection, args.profiles)
        x_train, y_train = matrix(connection, "train", behavior_tools, impact_tools, args.impact_scheme, args.train_negative_modulus)
        positive_weight = (len(y_train) - int(y_train.sum())) / int(y_train.sum())
        model = HistGradientBoostingClassifier(learning_rate=.1, max_iter=args.max_iter, max_leaf_nodes=31, l2_regularization=1., random_state=20260926)
        model.fit(x_train, y_train, sample_weight=np.where(y_train == 1, positive_weight, 1.0))
        x_validation, y_validation = matrix(connection, "validation", behavior_tools, impact_tools, args.impact_scheme, args.validation_negative_modulus)
        scores = model.predict_proba(x_validation)[:, 1]
    finally:
        connection.close()
    report = {"purpose": "Development-only screen of coarse counterfactual ledger impact; observed post-state is never queried.",
              "tool_set": args.tool_set, "behavior_tools": list(behavior_tools), "impact_tools": list(impact_tools),
              "impact_scheme": args.impact_scheme,
              "split_protocol": split_protocol,
              "training_cases": int(len(y_train)), "validation_cases": int(len(y_validation)),
              "negative_sampling": {"train": args.train_negative_modulus, "validation": args.validation_negative_modulus},
              "model": {"family": "HistGradientBoostingClassifier", "max_iter": args.max_iter},
              "validation_metrics": metric_summary(y_validation, scores, args.validation_negative_modulus), "status": "validation only; test not queried"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
