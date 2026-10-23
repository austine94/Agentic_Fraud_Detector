"""Plot episode and single-call latency distributions from the frozen benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


MODELS = (
    ("qwen3:4b", "Qwen3:4B"),
    ("smollm:360m", "SmolLM 360M"),
    ("qwen:0.5b", "Qwen 0.5B"),
    ("cajina/gemma4_e2b-q4_k_s:v01", "Gemma 4 E2B"),
    ("llama3.2:1b", "Llama 3.2 1B"),
)
COLOURS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")


def samples(connection: sqlite3.Connection, model: str) -> dict[str, list[float]]:
    result = {"end_to_end": [], "controller": [], "classifier": []}
    query = (
        "SELECT classifier_calls_ns,controller_calls_ns,end_to_end_ns "
        "FROM episodes WHERE model=? ORDER BY case_id"
    )
    for classifier_json, controller_json, end_to_end_ns in connection.execute(query, (model,)):
        result["end_to_end"].append(int(end_to_end_ns) / 1e6)
        result["classifier"].extend(int(value) / 1e6 for value in json.loads(classifier_json))
        result["controller"].extend(int(value) / 1e6 for value in json.loads(controller_json))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency box plots from the completed benchmark checkpoint.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/benchmarks/controller_latency_500.sqlite"))
    parser.add_argument("--png-output", type=Path,
                        default=Path("paper_results/controller_latency_component_boxplots.png"))
    parser.add_argument("--pdf-output", type=Path,
                        default=Path("paper_results/controller_latency_component_boxplots.pdf"))
    parser.add_argument("--end-to-end-png-output", type=Path,
                        default=Path("paper_results/controller_end_to_end_latency_boxplot.png"))
    parser.add_argument("--end-to-end-pdf-output", type=Path,
                        default=Path("paper_results/controller_end_to_end_latency_boxplot.pdf"))
    args = parser.parse_args()

    cache = args.png_output.parent / ".latency-plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    from matplotlib import pyplot as plt

    connection = sqlite3.connect(args.checkpoint)
    try:
        by_model = [samples(connection, model) for model, _ in MODELS]
    finally:
        connection.close()
    if any(len(model_samples["end_to_end"]) != 500 for model_samples in by_model):
        raise RuntimeError("the checkpoint must contain 500 completed episodes for every model")

    panels = (
        ("end_to_end", "End-to-end episode", "Latency per episode (ms)"),
        ("controller", "SLM controller", "Latency per call (ms)"),
        ("classifier", "Trained classifier", "Latency per call (ms)"),
    )
    labels = [display for _, display in MODELS]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for axis, (key, title, ylabel) in zip(axes, panels):
        boxes = axis.boxplot(
            [model_samples[key] for model_samples in by_model],
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops={"color": "black", "linewidth": 1.3},
        )
        for box, colour in zip(boxes["boxes"], COLOURS):
            box.set_facecolor(colour)
            box.set_edgecolor(colour)
            box.set_alpha(0.82)
        for item_name in ("whiskers", "caps"):
            for item, colour in zip(boxes[item_name], [colour for colour in COLOURS for _ in (0, 1)]):
                item.set_color(colour)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Latency distributions on the 500-case validation cohort", fontsize=14)
    figure.tight_layout()
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png_output, dpi=300, bbox_inches="tight")
    figure.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    boxes = axis.boxplot(
        [model_samples["end_to_end"] for model_samples in by_model],
        patch_artist=True,
        showfliers=False,
        widths=0.58,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for box, colour in zip(boxes["boxes"], COLOURS):
        box.set_facecolor(colour)
        box.set_edgecolor(colour)
        box.set_alpha(0.82)
    for item_name in ("whiskers", "caps"):
        for item, colour in zip(boxes[item_name], [colour for colour in COLOURS for _ in (0, 1)]):
            item.set_color(colour)
    axis.set_title("End-to-end latency on the 500-case validation cohort")
    axis.set_ylabel("Latency per episode (ms)")
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.end_to_end_png_output, dpi=300, bbox_inches="tight")
    figure.savefig(args.end_to_end_pdf_output, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Component PNG: {args.png_output}\nComponent PDF: {args.pdf_output}\n"
        f"End-to-end PNG: {args.end_to_end_png_output}\nEnd-to-end PDF: {args.end_to_end_pdf_output}"
    )


if __name__ == "__main__":
    main()
