"""Create unweighted representative-test metrics from the saved checkpoint.

The frozen threshold remains the confirmatory operating point. Thresholds near
1% false-positive rate are selected on the test scores and are therefore
reported only as exploratory sensitivity results.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


POLICIES = (
    ("full_evidence", "Full evidence"),
    ("qwen3_4b_balanced", "Qwen3:4B balanced"),
    ("aggressive_fixed_band", "Aggressive fixed-band"),
    ("moderate_structured", "Moderate structured"),
    ("transaction_only", "Transaction-only"),
)
FROZEN_THRESHOLD = 0.9991805471554841
TARGET_FPR = 0.01


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.count_nonzero((labels == 1) & predicted))
    fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted))
    tn = int(np.count_nonzero((labels == 0) & ~predicted))
    return {
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def threshold_at_or_below_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    false_positive_rate, _, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(false_positive_rate <= target)
    if not eligible.size:
        raise RuntimeError(f"no threshold achieves FPR <= {target}")
    return float(thresholds[eligible[-1]])


def percent(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the representative test without population weights.")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/test/representative_test.sqlite"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/test/representative_test_unweighted_report.json"))
    parser.add_argument("--markdown-output", type=Path,
                        default=Path("paper_results/representative_test_unweighted_table.md"))
    parser.add_argument("--learned-stopping-report", type=Path,
                        default=Path("results/development/learned_stopping_baselines.json"))
    args = parser.parse_args()

    connection = sqlite3.connect(args.checkpoint)
    try:
        cohort = list(connection.execute(
            "SELECT case_id,label,initial_risk FROM cohort ORDER BY case_id"
        ))
        case_ids = np.asarray([row[0] for row in cohort], dtype=np.int64)
        labels = np.asarray([row[1] for row in cohort], dtype=np.int8)
        initial_scores = np.asarray([row[2] for row in cohort], dtype=float)
        rows = []
        for policy, display_name in POLICIES:
            stopped = dict(connection.execute(
                "SELECT case_id,final_risk FROM results WHERE policy=?", (policy,)
            ))
            scores = np.asarray([
                stopped.get(int(case_id), initial_score)
                for case_id, initial_score in zip(case_ids, initial_scores, strict=True)
            ])
            costs = connection.execute(
                "SELECT COALESCE(SUM(total_cost),0) FROM results WHERE policy=?", (policy,)
            ).fetchone()[0]
            exploratory_threshold = threshold_at_or_below_fpr(labels, scores, TARGET_FPR)
            rows.append({
                "policy": policy,
                "method": display_name,
                "pr_auc": float(average_precision_score(labels, scores)),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "frozen_operating_point": metrics_at_threshold(labels, scores, FROZEN_THRESHOLD),
                "exploratory_1pct_fpr_operating_point": metrics_at_threshold(
                    labels, scores, exploratory_threshold
                ),
                "mean_cost_all_sampled_cases": float(costs) / len(cohort),
            })
    finally:
        connection.close()

    if args.learned_stopping_report.exists():
        learned = json.loads(args.learned_stopping_report.read_text(encoding="utf-8"))
        for baseline in learned["baselines"]:
            test = baseline["representative_test"]
            rows.insert(2, {
                "policy": f"learned_{baseline['kind']}_stopping",
                "method": baseline["label"],
                "pr_auc": test["unweighted"]["pr_auc"],
                "roc_auc": test["unweighted"]["roc_auc"],
                "frozen_operating_point": test["frozen_operating_point_unweighted"],
                "exploratory_1pct_fpr_operating_point": test["exploratory_1pct_fpr_operating_point_unweighted"],
                "mean_cost_all_sampled_cases": test["mean_cost_all_sampled_cases"],
            })

    full_cost = next(row["mean_cost_all_sampled_cases"] for row in rows if row["policy"] == "full_evidence")
    for row in rows:
        row["cost_saving_vs_full_evidence"] = 1 - row["mean_cost_all_sampled_cases"] / full_cost

    report = {
        "purpose": "unweighted held-out representative-test results with an exploratory 1% FPR sensitivity point",
        "weighting": "none; every one of the 203,286 sampled cases contributes equally",
        "cohort": {
            "cases": len(cohort),
            "fraud_cases": int(labels.sum()),
            "non_fraud_cases": int(np.count_nonzero(labels == 0)),
            "sample_fraud_prevalence": float(labels.mean()),
        },
        "confirmatory_threshold": FROZEN_THRESHOLD,
        "exploratory_target_fpr": TARGET_FPR,
        "interpretation_warning": (
            "The approximately 1% FPR thresholds were selected from test scores after evaluation and are "
            "exploratory, not confirmatory. Precision describes the enriched sampled cohort, not population precision."
        ),
        "methods": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Representative Test — Unweighted Results",
        "",
        "Every sampled case has equal weight. The frozen threshold is the confirmatory result; the approximately 1% FPR "
        "operating points are post-hoc sensitivity analyses selected from the test scores.",
        "",
        "| Method | PR-AUC | ROC-AUC | Frozen precision | Frozen recall | Frozen FPR | Precision at ~1% FPR | Recall at ~1% FPR | Achieved FPR | Mean cost | Cost saving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        frozen = row["frozen_operating_point"]
        exploratory = row["exploratory_1pct_fpr_operating_point"]
        lines.append(
            f"| {row['method']} | {row['pr_auc']:.3f} | {row['roc_auc']:.3f} | "
            f"{percent(frozen['precision'])} | {percent(frozen['recall'])} | {percent(frozen['false_positive_rate'], 4)} | "
            f"{percent(exploratory['precision'])} | {percent(exploratory['recall'])} | "
            f"{percent(exploratory['false_positive_rate'], 3)} | {row['mean_cost_all_sampled_cases']:.3f} | "
            f"{percent(row['cost_saving_vs_full_evidence'], 1)} |"
        )
    lines.extend([
        "",
        "Cohort: 203,286 cases (3,286 fraud; 200,000 non-fraud; 1.62% fraud prevalence). Precision in this table "
        "is specific to the enriched sample and must not be interpreted as deployment-population precision.",
    ])
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "markdown": str(args.markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
