"""Fit train-only state scorers and run governed full-evidence validation traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.investigation_runtime import InvestigationRuntime, TerminalThresholds
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row
from src.threshold_selection import select_terminal_thresholds, weighted_binary_metrics


def training_rows(connection: sqlite3.Connection, negative_modulus: int):
    return connection.execute("""
        SELECT c.type, c.amount, c.oldbalance_org, c.oldbalance_dest, c.is_fraud, p.*,
               n.prior_unique_origins AS destination_prior_unique_origins,
               n.recent_new_origins AS destination_recent_new_origins
        FROM cases c JOIN behavior.profiles p USING(case_id)
        LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
        JOIN iid.assignments a USING(case_id)
        WHERE a.split='train' AND (c.is_fraud=1 OR c.case_id % ?=0)
        ORDER BY c.case_id
    """, (negative_modulus,))


def validation_cases(connection: sqlite3.Connection, limit: int | None):
    query = "SELECT c.case_id, c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) WHERE a.split='validation' ORDER BY c.case_id"
    if limit is not None:
        query += " LIMIT ?"
        return connection.execute(query, (limit,))
    return connection.execute(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only governed full-evidence reference run.")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--train-negative-modulus", type=int, default=45)
    parser.add_argument("--validation-negative-modulus", type=int, default=20)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        rows = list(training_rows(connection, args.train_negative_modulus))
        examples = [safe_evidence_from_row(row) for row in rows]
        labels = [int(row["is_fraud"]) for row in rows]
        validation = [(int(row["case_id"]), int(row["is_fraud"])) for row in validation_cases(connection, args.validation_limit)]
    finally:
        connection.close()
    positive_weight = (len(labels) - sum(labels)) / sum(labels)
    scorer = fit_state_risk_scorers(examples, labels, seed=20260924,
                                    sample_weights=[positive_weight if label else 1.0 for label in labels])
    service = EvidenceService(args.database, args.profiles, split_database=args.split_database)
    # Scores do not depend on thresholds; use a provisional valid value to acquire the full path once.
    runtime = InvestigationRuntime(service, scorer, TerminalThresholds(0.0, 1.0), budget=11.25)
    traces = [runtime.run_full_evidence(case_id) for case_id, _ in validation]
    scores = [trace.risk_trajectory[-1]["risk"] for trace in traces]
    labels = [label for _, label in validation]
    weights = [1.0 if label else float(args.validation_negative_modulus) for label in labels]
    thresholds, selection = select_terminal_thresholds(labels, scores, weights)
    for trace in traces:
        risk = trace.risk_trajectory[-1]["risk"]
        trace.terminal_action = "escalate" if risk >= thresholds.escalate else "flag" if risk >= thresholds.flag else "approve"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for trace in traces:
            destination.write(json.dumps(trace.as_dict(), sort_keys=True) + "\n")
    report = {
        "status": "validation_only_thresholds_selected_not_yet_frozen", "training_cases": len(rows),
        "validation_cases": len(validation), "trace_output": str(args.output), "impact_scheme": "operational",
        "threshold_selection": {"actionable_precision_target": .99, "escalation_precision_target": .999,
                                "negative_sampling_weight": args.validation_negative_modulus, "selected": selection},
        "terminal_thresholds": {"flag": thresholds.flag, "escalate": thresholds.escalate},
        "validation_metrics": weighted_binary_metrics(labels, scores, weights, threshold=thresholds.flag),
        "test_access": "not queried",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
