"""Offline label-based evaluation of completed controller traces.

Labels are joined only here, after the controller episode is complete; they
never enter the controller prompt, scorer features, gateway, or trace state.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    result = {"pr_auc": float(average_precision_score(labels, scores)), "roc_auc": float(roc_auc_score(labels, scores))}
    for target in (.80, .90, .95):
        eligible = recall[precision >= target]
        result[f"recall_at_{int(target * 100)}pct_precision"] = float(eligible.max()) if eligible.size else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stopped-state versus full-evidence scores on a routed validation cohort.")
    parser.add_argument("paired_report", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.paired_report.read_text())
    pairs = report["pairs"]; ids = [pair["case_id"] for pair in pairs]
    connection = sqlite3.connect(args.database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(args.split_database),))
        labels_by_id = dict(connection.execute("""SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id)
            WHERE a.split='validation' AND c.case_id IN (%s)""" % ",".join("?" * len(ids)), ids))
    finally: connection.close()
    if len(labels_by_id) != len(ids): raise RuntimeError("paired report includes a non-validation or missing case")
    labels = np.asarray([labels_by_id[case_id] for case_id in ids], dtype=np.int8)
    controller = np.asarray([pair["controller"]["risk_trajectory"][-1]["risk"] for pair in pairs])
    full = np.asarray([pair["full_evidence"]["risk_trajectory"][-1]["risk"] for pair in pairs])
    outcome = {"purpose": "offline routed-cohort development metric; labels were not available to the controller",
               "cases": len(ids), "positive_cases": int(labels.sum()), "controller": summary(labels, controller),
               "full_evidence": summary(labels, full)}
    output = args.output or args.paired_report.with_name(args.paired_report.stem + "_outcomes.json")
    output.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__": main()
