"""Plot validation ROC curves from completed paired controller reports."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


COMPARATOR_REPORTS = (
    ("Qwen3:4B", "qwen3_4b_balanced_500_stratified.json", "controller"),
    ("Full Evidence", "qwen3_4b_balanced_500_stratified.json", "full_evidence"),
    ("No Additional Evidence", "comparison_transaction_only_500_stratified.json", "controller"),
    ("Rule-Based Cost Focus", "comparison_aggressive_fixed_band_500_stratified.json", "controller"),
    ("Rule-Based Evidence Focus", "comparison_moderate_structured_500_stratified.json", "controller"),
    ("CART", "learned_cart_stopping_500_stratified.json", "controller"),
    ("Gradient-Boosting", "learned_boosted_stopping_500_stratified.json", "controller"),
)

MODEL_REPORTS = (
    ("Qwen3:4B", "qwen3_4b_balanced_500_stratified.json", "controller"),
    ("SmolLM 360M", "smollm_360m_balanced_500_stratified.json", "controller"),
    ("Qwen 0.5B", "qwen_0_5b_balanced_500_stratified.json", "controller"),
    ("Gemma4 E2B", "cajina_gemma4_e2b_q4_k_s_v01_balanced_500_stratified.json", "controller"),
    ("Llama 3.2 1B", "llama3_2_1b_balanced_500_stratified.json", "controller"),
)

COLORS = ("#0072B2", "#000000", "#E69F00", "#56B4E9", "#D55E00", "#CC79A7", "#009E73")
LINESTYLES = ("-", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1)), ":")


def load_curves(
    specifications: tuple[tuple[str, str, str], ...],
    reports_dir: Path,
    database: Path,
    split_database: Path,
) -> tuple[list[tuple[str, list[float], list[float], float]], int, int]:
    reports = []
    reference_ids: list[int] | None = None
    for label, filename, trace_key in specifications:
        report = json.loads((reports_dir / filename).read_text(encoding="utf-8"))
        case_ids = [int(case_id) for case_id in report["case_ids"]]
        if reference_ids is None:
            reference_ids = case_ids
        elif case_ids != reference_ids:
            raise ValueError(f"{filename} does not use the shared validation cohort")
        reports.append((label, report, trace_key))

    assert reference_ids is not None
    placeholders = ",".join("?" for _ in reference_ids)
    connection = sqlite3.connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(split_database),))
        labels_by_id = dict(
            connection.execute(
                f"""SELECT c.case_id,c.is_fraud
                    FROM cases c JOIN iid.assignments a USING(case_id)
                    WHERE a.split='validation' AND c.case_id IN ({placeholders})""",
                reference_ids,
            )
        )
    finally:
        connection.close()
    if len(labels_by_id) != len(reference_ids):
        raise ValueError("the report cohort contains missing or non-validation cases")

    labels = [int(labels_by_id[case_id]) for case_id in reference_ids]
    curves = []
    for label, report, trace_key in reports:
        pair_by_id = {int(pair["case_id"]): pair for pair in report["pairs"]}
        scores = [
            float(pair_by_id[case_id][trace_key]["risk_trajectory"][-1]["risk"])
            for case_id in reference_ids
        ]
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        curves.append(
            (
                label,
                false_positive_rate.tolist(),
                true_positive_rate.tolist(),
                float(auc(false_positive_rate, true_positive_rate)),
            )
        )
    return curves, sum(labels), len(labels) - sum(labels)


def draw_plot(
    curves: list[tuple[str, list[float], list[float], float]],
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
        }
    )
    figure, axis = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for index, (label, false_positive_rate, true_positive_rate, roc_auc) in enumerate(curves):
        width = 2.6 if index == 0 else 2.0
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            linewidth=width,
            label=f"{label} (AUC = {roc_auc:.3f})",
            zorder=5 - index,
        )

    axis.plot([0, 1], [0, 1], color="#777777", linewidth=1.2, linestyle=(0, (4, 4)), label="Chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True, framealpha=0.97, edgecolor="#B8B8B8")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC curves for the shared 500-case validation cohort.")
    parser.add_argument("--reports-dir", type=Path, default=Path("results/development"))
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results"))
    args = parser.parse_args()

    comparator_curves, positives, negatives = load_curves(
        COMPARATOR_REPORTS, args.reports_dir, args.database, args.split_database
    )
    model_curves, model_positives, model_negatives = load_curves(
        MODEL_REPORTS, args.reports_dir, args.database, args.split_database
    )
    if (positives, negatives) != (model_positives, model_negatives):
        raise ValueError("the comparator and cross-model cohorts do not match")

    draw_plot(
        comparator_curves,
        args.output_dir / "roc_qwen_vs_comparators_validation",
    )
    draw_plot(
        model_curves,
        args.output_dir / "roc_five_slms_validation",
    )

    for heading, curves in (("comparators", comparator_curves), ("models", model_curves)):
        print(heading)
        for label, _, _, roc_auc in curves:
            print(f"  {label}: {roc_auc:.6f}")


if __name__ == "__main__":
    main()
