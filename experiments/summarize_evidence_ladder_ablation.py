"""Summarise a fixed evidence-ladder ablation from full-evidence paired traces.

The source report must contain the full, governed acquisition path for every
case in one frozen validation cohort.  This measures the predictive and
resource effect of *permitted evidence states*, rather than a controller's
choice to stop early.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


STAGES = (
    ("Transaction-only", ("initial_evidence",), 1),
    ("Destination-network", ("initial_evidence", "destination_network"), 2),
    ("Origin-impact", ("initial_evidence", "destination_network", "origin_impact"), 3),
    (
        "Full evidence",
        ("initial_evidence", "destination_network", "origin_impact", "destination_impact", "combined_impact"),
        5,
    ),
)


def metric_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    result = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    for target in (0.80, 0.90, 0.95):
        eligible = recall[precision >= target]
        result[f"recall_at_{int(target * 100)}pct_precision"] = float(eligible.max()) if eligible.size else 0.0
    return result


def validation_labels(case_ids: list[int], database: Path, split_database: Path) -> dict[int, int]:
    placeholders = ",".join("?" for _ in case_ids)
    connection = sqlite3.connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(split_database),))
        result = dict(connection.execute(
            "SELECT c.case_id, c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) "
            f"WHERE a.split='validation' AND c.case_id IN ({placeholders})", case_ids
        ))
    finally:
        connection.close()
    if len(result) != len(case_ids):
        raise ValueError("all ablation cases must be unique validation cases")
    return result


def stage_values(pairs: list[dict], stage_state: tuple[str, ...], tool_call_count: int) -> tuple[list[float], list[dict]]:
    scores, resources = [], []
    for pair in pairs:
        trace = pair["full_evidence"]
        trajectory = trace.get("risk_trajectory", [])
        matching = [entry for entry in trajectory if tuple(entry["evidence_state"]) == stage_state]
        if len(matching) != 1:
            raise ValueError(f"case {pair.get('case_id')} is missing the required evidence state {stage_state}")
        calls = trace.get("tool_calls", [])[:tool_call_count]
        if len(calls) != tool_call_count or any(call.get("status") != "ok" for call in calls):
            raise ValueError(f"case {pair.get('case_id')} has an incomplete full-evidence trace")
        scores.append(float(matching[0]["risk"]))
        resources.append({
            "acquisition_cost": float(sum(call["normalized_cost"] for call in calls)),
            "data_transfer_units": int(sum(call["data_transfer_units"] for call in calls)),
            "service_latency_s": sum(call["latency_ns"] for call in calls) / 1e9,
            "acquisition_tool_calls": tool_call_count - 1,
        })
    return scores, resources


def render_markdown(rows: list[dict], cases: int, positives: int) -> str:
    lines = [
        "# Evidence-Ladder Ablation",
        "",
        f"Validation routed cohort: {cases} cases ({positives} fraud). Scores are evaluated offline; labels are not exposed to the evidence service or scorer.",
        "",
        "| Evidence state | PR-AUC | ΔPR-AUC vs prior stage | ROC-AUC | Recall @ 80% precision | Recall @ 90% precision | Recall @ 95% precision | Mean acquisition cost | Mean service latency (s) | P95 service latency (s) | Data-transfer units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['stage']} | {metrics['pr_auc']:.3f} | {row['pr_auc_change_from_prior']:+.3f} | "
            f"{metrics['roc_auc']:.3f} | {metrics['recall_at_80pct_precision']:.0%} | "
            f"{metrics['recall_at_90pct_precision']:.0%} | {metrics['recall_at_95pct_precision']:.0%} | "
            f"{row['mean_acquisition_cost']:.2f} | {row['mean_service_latency_s']:.4f} | "
            f"{row['p95_service_latency_s']:.4f} | {row['mean_data_transfer_units']:.1f} |"
        )
    return "\n".join([*lines, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the fixed validation evidence-ladder ablation.")
    parser.add_argument("full_evidence_report", type=Path, help="Paired report containing a full_evidence trace for every case")
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.full_evidence_report.read_text(encoding="utf-8"))
    pairs = report.get("pairs", [])
    case_ids = [int(pair["case_id"]) for pair in pairs]
    if len(case_ids) != args.expected_cases or len(set(case_ids)) != args.expected_cases:
        raise SystemExit(f"the report must contain exactly {args.expected_cases} unique paired cases")
    labels_by_id = validation_labels(case_ids, args.database, args.split_database)
    labels = np.asarray([labels_by_id[case_id] for case_id in case_ids], dtype=np.int8)

    rows = []
    prior_pr_auc: float | None = None
    for stage, state, tool_call_count in STAGES:
        scores, resources = stage_values(pairs, state, tool_call_count)
        metrics = metric_summary(labels, np.asarray(scores))
        rows.append({
            "stage": stage,
            "evidence_state": list(state),
            "metrics": metrics,
            "pr_auc_change_from_prior": 0.0 if prior_pr_auc is None else metrics["pr_auc"] - prior_pr_auc,
            "mean_acquisition_cost": float(np.mean([r["acquisition_cost"] for r in resources])),
            "mean_acquisition_tool_calls": float(np.mean([r["acquisition_tool_calls"] for r in resources])),
            "mean_data_transfer_units": float(np.mean([r["data_transfer_units"] for r in resources])),
            "mean_service_latency_s": float(np.mean([r["service_latency_s"] for r in resources])),
            "p95_service_latency_s": float(np.percentile([r["service_latency_s"] for r in resources], 95)),
        })
        prior_pr_auc = metrics["pr_auc"]
    output = {
        "purpose": "fixed evidence-state ablation on a routed validation cohort; no test access",
        "source_full_evidence_report": str(args.full_evidence_report),
        "cases": len(case_ids),
        "positive_cases": int(labels.sum()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(rows, len(case_ids), int(labels.sum())), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
