"""Measure classifier, controller, and end-to-end latency on 500 cases.

Run this benchmark only on an otherwise idle machine. It executes the five
balanced SLM controllers serially on the shared routed validation cohort.
One excluded warm-up episode is run before each model. Successful episodes
are checkpointed after every case so a failed run can be inspected or resumed,
although a single uninterrupted run is preferable for comparable timings.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from experiments.evaluate_controller_development import rows
from experiments.evaluate_qwen_boundary_repeatability import require_ollama_model
from experiments.run_terminal_classifier_ablation import DEFAULT_REPORTS, DISPLAY_NAMES, load_and_validate_reports
from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


SEED = 20260924
ENTRY_GATE = 0.533566727121687
QUANTILES = (50, 99)


@dataclass
class TimedRiskScorer:
    scorer: Any
    call_latencies_ns: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.call_latencies_ns.clear()

    def predict_risk(self, evidence: Mapping[str, Mapping[str, Any]]) -> float:
        started = perf_counter_ns()
        result = float(self.scorer.predict_risk(evidence))
        self.call_latencies_ns.append(perf_counter_ns() - started)
        return result


def fit_classifier(database: Path, profiles: Path, split_database: Path):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(profiles),))
        train = list(rows(connection, "train", 45))
        labels = [int(row["is_fraud"]) for row in train]
        weight = (len(labels) - sum(labels)) / sum(labels)
        return fit_state_risk_scorers(
            [safe_evidence_from_row(row) for row in train],
            labels,
            seed=SEED,
            sample_weights=[weight if label else 1.0 for label in labels],
        )
    finally:
        connection.close()


def initialise_checkpoint(connection: sqlite3.Connection, contract: dict) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
             model TEXT NOT NULL, case_id INTEGER NOT NULL, status TEXT NOT NULL,
             classifier_calls_ns TEXT NOT NULL, controller_calls_ns TEXT NOT NULL,
             evidence_service_ns INTEGER NOT NULL, end_to_end_ns INTEGER NOT NULL,
             PRIMARY KEY(model,case_id)) WITHOUT ROWID"""
    )
    encoded = json.dumps(contract, sort_keys=True)
    saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if saved is None:
        connection.execute("INSERT INTO metadata(key,value) VALUES ('contract',?)", (encoded,))
    elif saved[0] != encoded:
        raise RuntimeError("checkpoint belongs to a different latency benchmark contract")
    connection.commit()


def percentile(values: list[float], quantile: int) -> float:
    return float(np.percentile(values, quantile))


def quantile_pair(values_ns: list[int]) -> dict[str, float]:
    values_ms = [value / 1e6 for value in values_ns]
    return {f"p{quantile}_ms": percentile(values_ms, quantile) for quantile in QUANTILES}


def load_samples(connection: sqlite3.Connection, model: str) -> dict[str, list[int]]:
    samples = {
        "classifier_episode": [],
        "controller_episode": [],
        "end_to_end_episode": [],
        "classifier_call": [],
        "controller_call": [],
    }
    query = (
        "SELECT classifier_calls_ns,controller_calls_ns,end_to_end_ns "
        "FROM episodes WHERE model=? ORDER BY case_id"
    )
    for classifier_json, controller_json, end_to_end_ns in connection.execute(query, (model,)):
        classifier_calls = [int(value) for value in json.loads(classifier_json)]
        controller_calls = [int(value) for value in json.loads(controller_json)]
        samples["classifier_episode"].append(sum(classifier_calls))
        samples["controller_episode"].append(sum(controller_calls))
        samples["end_to_end_episode"].append(int(end_to_end_ns))
        samples["classifier_call"].extend(classifier_calls)
        samples["controller_call"].extend(controller_calls)
    return samples


def summarise_model(connection: sqlite3.Connection, model: str) -> tuple[dict, dict[str, list[int]]]:
    samples = load_samples(connection, model)
    return {
        "model": model,
        "display_name": DISPLAY_NAMES.get(model, model),
        "episodes": len(samples["end_to_end_episode"]),
        "episode_latency": {
            "classifier": quantile_pair(samples["classifier_episode"]),
            "controller": quantile_pair(samples["controller_episode"]),
            "end_to_end": quantile_pair(samples["end_to_end_episode"]),
        },
        "per_call_latency": {
            "classifier": quantile_pair(samples["classifier_call"]),
            "controller": quantile_pair(samples["controller_call"]),
        },
    }, samples


def render_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "# Latency Quantiles on the 500-Case Validation Cohort",
        "",
        "## Episode-level latency",
        "",
        "Each classifier/controller value is the sum of that component's calls within one episode. End-to-end is measured wall-clock time around the complete governed episode. All values are milliseconds.",
        "",
        "| Model | Classifier P50 | Classifier P99 | Controller P50 | Controller P99 | End-to-end P50 | End-to-end P99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        episode = row["episode_latency"]
        lines.append(
            f"| {row['display_name']} | {episode['classifier']['p50_ms']:.2f} | "
            f"{episode['classifier']['p99_ms']:.2f} | {episode['controller']['p50_ms']:.2f} | "
            f"{episode['controller']['p99_ms']:.2f} | {episode['end_to_end']['p50_ms']:.2f} | "
            f"{episode['end_to_end']['p99_ms']:.2f} |"
        )
    lines.extend([
        "",
        "## Per-stage latency",
        "",
        "Each observation is one trained-classifier call or one SLM-controller call. All values are milliseconds.",
        "",
        "| Model | Classifier call P50 | Classifier call P99 | SLM call P50 | SLM call P99 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        stage = row["per_call_latency"]
        lines.append(
            f"| {row['display_name']} | {stage['classifier']['p50_ms']:.2f} | "
            f"{stage['classifier']['p99_ms']:.2f} | {stage['controller']['p50_ms']:.2f} | "
            f"{stage['controller']['p99_ms']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_boxplot(summary_rows: list[dict], samples_by_model: list[dict[str, list[int]]], output: Path) -> None:
    cache = output.parent / ".latency-plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    from matplotlib import pyplot as plt

    labels = [row["display_name"] for row in summary_rows]
    series = (
        ("end_to_end_episode", "End-to-end", "#4C78A8"),
        ("classifier_episode", "Classifier", "#F58518"),
        ("controller_episode", "Controller", "#54A24B"),
    )
    positions = np.arange(len(labels), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for offset, (key, display, colour) in zip((-width, 0.0, width), series):
        boxes = axis.boxplot(
            [[value / 1e6 for value in samples[key]] for samples in samples_by_model],
            positions=positions + offset,
            widths=width * 0.88,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"facecolor": colour, "edgecolor": colour, "alpha": 0.82},
            whiskerprops={"color": colour},
            capprops={"color": colour},
        )
        boxes["boxes"][0].set_label(display)
    axis.set_xticks(positions, labels)
    axis.set_yscale("log")
    axis.set_ylabel("Latency per episode (ms, log scale)")
    axis.set_title("Latency across the 500-case validation cohort")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled five-model latency benchmark.")
    parser.add_argument("reports", nargs="*", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/benchmarks/controller_latency_500.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/controller_latency_500.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/controller_latency_500.md"))
    parser.add_argument("--figure-output", type=Path, default=Path("paper_results/controller_latency_500_boxplot.png"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reports, case_ids = load_and_validate_reports(args.reports, args.expected_cases)
    models = [report["model"] for report in reports]
    contract = {
        "models": models,
        "case_ids": case_ids,
        "goal_profile": "balanced",
        "seed": SEED,
        "entry_gate": ENTRY_GATE,
        "timing": "perf_counter_ns",
        "execution": "serial, one excluded warm-up episode per model",
    }
    if args.dry_run:
        print(json.dumps({**contract, "episodes": len(models) * len(case_ids)}, indent=2))
        return

    scorer = TimedRiskScorer(fit_classifier(args.database, args.profiles, args.split_database))
    service = EvidenceService(args.database, args.profiles, split_database=args.split_database)
    runtime = AgenticControllerRuntime(service, scorer, entry_gate=ENTRY_GATE, budget=11.25)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = sqlite3.connect(args.checkpoint)
    initialise_checkpoint(checkpoint, contract)
    try:
        for model in models:
            require_ollama_model(args.endpoint, model)
            policy = OllamaControllerPolicy(
                model,
                endpoint=args.endpoint,
                seed=SEED,
                temperature=0.0,
                timeout_seconds=args.timeout_seconds,
                goal_profile="balanced",
            )
            completed = {int(row[0]) for row in checkpoint.execute(
                "SELECT case_id FROM episodes WHERE model=?", (model,)
            )}
            remaining = [case_id for case_id in case_ids if case_id not in completed]
            if not remaining:
                print(f"{model}: 500/500 already complete", flush=True)
                continue

            scorer.reset()
            warm_trace = runtime.run(remaining[0], policy)
            if warm_trace.status != "stopped":
                raise RuntimeError(f"{model} warm-up ended with {warm_trace.status}")
            print(f"{model}: warm-up complete; measuring {len(remaining)} episodes", flush=True)

            for index, case_id in enumerate(remaining, start=1):
                scorer.reset()
                started = perf_counter_ns()
                trace = runtime.run(case_id, policy)
                end_to_end_ns = perf_counter_ns() - started
                if trace.status != "stopped":
                    raise RuntimeError(f"{model} case {case_id} ended with {trace.status}")
                controller_calls_ns = [
                    int(step["controller_latency_ns"])
                    for step in trace.controller_steps
                    if step.get("decision_source", "policy") == "policy"
                ]
                checkpoint.execute(
                    "INSERT INTO episodes VALUES (?,?,?,?,?,?,?)",
                    (
                        model,
                        case_id,
                        trace.status,
                        json.dumps(scorer.call_latencies_ns, separators=(",", ":")),
                        json.dumps(controller_calls_ns, separators=(",", ":")),
                        int(trace.total_latency_ns),
                        int(end_to_end_ns),
                    ),
                )
                checkpoint.commit()
                if index % 25 == 0 or index == len(remaining):
                    print(f"{model}: saved {len(completed) + index}/500", flush=True)

        summaries = [summarise_model(checkpoint, model) for model in models]
        summary_rows = [item[0] for item in summaries]
        samples_by_model = [item[1] for item in summaries]
        output = {
            "purpose": "controlled latency benchmark on the shared 500-case routed validation cohort",
            "contract": contract,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            "units": "milliseconds",
            "rows": summary_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary_rows), encoding="utf-8")
        render_boxplot(summary_rows, samples_by_model, args.figure_output)
    finally:
        checkpoint.close()

    print(f"JSON: {args.output}\nTables: {args.markdown_output}\nFigure: {args.figure_output}", flush=True)


if __name__ == "__main__":
    main()
