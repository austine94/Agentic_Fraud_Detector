"""Run a terminal risk-classifier ablation on frozen controller traces.

Each local model receives only the final evidence selected by its existing
balanced controller trace and emits one fraud probability.  The reference row
uses the frozen risk-classifier score from the Qwen3:4B trace.  Fraud labels
are joined only for offline evaluation after inference.

Progress is checkpointed in SQLite after every successful model/case call, so
rerunning the same command resumes an interrupted experiment.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from experiments.evaluate_qwen_boundary_repeatability import require_ollama_model
from experiments.evaluate_qwen_repeatability import labels_for


SEED = 20260924
PRECISION_TARGETS = (0.80, 0.90, 0.95, 0.99)
PROMPT_VERSION = "terminal_fraud_probability_v1"
DEFAULT_REPORTS = (
    Path("results/development/qwen3_4b_balanced_500_stratified.json"),
    Path("results/development/smollm_360m_balanced_500_stratified.json"),
    Path("results/development/qwen_0_5b_balanced_500_stratified.json"),
    Path("results/development/cajina_gemma4_e2b_q4_k_s_v01_balanced_500_stratified.json"),
    Path("results/development/llama3_2_1b_balanced_500_stratified.json"),
)
DISPLAY_NAMES = {
    "qwen3:4b": "Qwen3:4B",
    "smollm:360m": "SmolLM 360M",
    "qwen:0.5b": "Qwen 0.5B",
    "cajina/gemma4_e2b-q4_k_s:v01": "Gemma 4 E2B",
    "llama3.2:1b": "Llama 3.2 1B",
}


def load_and_validate_reports(paths: list[Path], expected_cases: int) -> tuple[list[dict], list[int]]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(reports) != 5:
        raise ValueError("the ablation requires exactly five balanced model reports")
    reference_ids: list[int] | None = None
    seen_models: set[str] = set()
    for report in reports:
        model = report.get("model")
        if not isinstance(model, str) or model in seen_models:
            raise ValueError("every source report must name a distinct model")
        if report.get("goal_profile") != "balanced":
            raise ValueError(f"{model} is not a balanced controller report")
        pairs = report.get("pairs", [])
        case_ids = [int(pair["case_id"]) for pair in pairs]
        if len(case_ids) != expected_cases or len(set(case_ids)) != expected_cases:
            raise ValueError(f"{model} must contain exactly {expected_cases} unique cases")
        if any(pair.get("controller", {}).get("status") != "stopped" for pair in pairs):
            raise ValueError(f"{model} contains a non-stopped controller trace")
        if reference_ids is None:
            reference_ids = case_ids
        elif case_ids != reference_ids:
            raise ValueError(f"{model} does not use the same ordered cohort as the first report")
        seen_models.add(model)
    return reports, reference_ids or []


def terminal_prompt(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the label-free, classifier-score-free Ollama request payload."""
    schema = {
        "type": "object",
        "properties": {
            "fraud_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["fraud_probability"],
        "additionalProperties": False,
    }
    system = (
        "You are the terminal fraud-risk decision component for a payment investigation. "
        "Using only the supplied acquired evidence, estimate the probability that the transaction is fraudulent. "
        "A probability of 0 means certainly not fraud and 1 means certainly fraud. "
        "Return only the required JSON object, without explanation."
    )
    return {
        "stream": False,
        "format": schema,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Acquired evidence:\n" + json.dumps(evidence, sort_keys=True)},
        ],
        # Small models may spell out very small probabilities with many zeros;
        # leave enough tokens to close the otherwise schema-valid JSON object.
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": 64},
    }


def predict_probability(
    endpoint: str,
    model: str,
    evidence: dict[str, dict[str, Any]],
    timeout_seconds: float,
    max_attempts: int,
) -> tuple[float, int, str]:
    payload = terminal_prompt(evidence) | {"model": model}
    encoded = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    last_raw: str | None = None
    for _ in range(max_attempts):
        started = perf_counter_ns()
        try:
            request = urllib.request.Request(endpoint, data=encoded, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    response_payload = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                response = subprocess.run(
                    ["curl", "--fail", "--silent", "--show-error", "--max-time", str(timeout_seconds),
                     "-H", "Content-Type: application/json", "--data-binary", "@-", endpoint],
                    input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                )
                try:
                    response_payload = json.loads(response.stdout)
                except json.JSONDecodeError as parse_error:
                    raise RuntimeError(f"invalid Ollama response after urllib failure: {error}") from parse_error
            raw = str(response_payload["message"]["content"])
            last_raw = raw
            parsed = json.loads(raw)
            probability = float(parsed["fraud_probability"])
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"fraud_probability is outside [0,1]: {probability}")
            return probability, perf_counter_ns() - started, raw
        except (KeyError, TypeError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
            last_error = error
    raise RuntimeError(
        f"{model} failed to return a valid probability after {max_attempts} attempts: {last_error}; "
        f"last raw response={last_raw!r}"
    )


def initialise_checkpoint(connection: sqlite3.Connection, contract: dict) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
             model TEXT NOT NULL, case_id INTEGER NOT NULL, probability REAL NOT NULL,
             latency_ns INTEGER NOT NULL, raw_response TEXT NOT NULL,
             PRIMARY KEY(model, case_id)) WITHOUT ROWID"""
    )
    saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    encoded = json.dumps(contract, sort_keys=True)
    if saved is None:
        connection.execute("INSERT INTO metadata(key,value) VALUES ('contract',?)", (encoded,))
    elif saved[0] != encoded:
        raise RuntimeError("checkpoint belongs to a different terminal-ablation contract")
    connection.commit()


def metric_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    result = {"pr_auc": float(average_precision_score(labels, scores))}
    for target in PRECISION_TARGETS:
        eligible = recall[precision >= target]
        result[f"recall_at_{int(target * 100)}pct_precision"] = float(eligible.max()) if eligible.size else 0.0
    return result


def render_markdown(output: dict) -> str:
    lines = [
        "# Terminal Risk-Classifier Ablation",
        "",
        f"Shared routed validation cohort: {output['cases']} cases ({output['positive_cases']} fraud and "
        f"{output['cases'] - output['positive_cases']} non-fraud).",
        "",
        "| Method | PR-AUC | Recall @ 80% precision | Recall @ 90% precision | Recall @ 95% precision | Recall @ 99% precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output["rows"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['method']} | {metrics['pr_auc']:.3f} | "
            f"{metrics['recall_at_80pct_precision']:.0%} | {metrics['recall_at_90pct_precision']:.0%} | "
            f"{metrics['recall_at_95pct_precision']:.0%} | {metrics['recall_at_99pct_precision']:.0%} |"
        )
    lines.extend([
        "",
        "The reference uses the frozen stopped-state risk-classifier score from the Qwen3:4B balanced trace. "
        "Each ablated row uses one probability emitted by that row's controller model from its own final acquired evidence; "
        "classifier scores and fraud labels are not included in the terminal model prompt.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    reports: list[dict],
    case_ids: list[int],
    labels_by_id: dict[int, int],
    checkpoint: sqlite3.Connection,
    output_path: Path,
    markdown_path: Path,
) -> None:
    labels = np.asarray([labels_by_id[case_id] for case_id in case_ids], dtype=np.int8)
    qwen = next(report for report in reports if report["model"] == "qwen3:4b")
    reference_scores = np.asarray([
        float(pair["controller"]["risk_trajectory"][-1]["risk"]) for pair in qwen["pairs"]
    ])
    rows = [{
        "method": "Qwen3:4B controller + risk classifier",
        "model": "qwen3:4b",
        "terminal_decision": "frozen_risk_classifier",
        "metrics": metric_summary(labels, reference_scores),
    }]
    per_case: list[dict] = []
    for report in reports:
        model = report["model"]
        saved = dict(checkpoint.execute(
            "SELECT case_id,probability FROM predictions WHERE model=?", (model,)
        ))
        if len(saved) != len(case_ids):
            raise RuntimeError(f"cannot summarise incomplete predictions for {model}: {len(saved)}/{len(case_ids)}")
        scores = np.asarray([saved[case_id] for case_id in case_ids])
        display = DISPLAY_NAMES.get(model, model)
        rows.append({
            "method": f"{display} controller + {display} decision",
            "model": model,
            "terminal_decision": "same_slm",
            "metrics": metric_summary(labels, scores),
        })
        per_case.extend({"model": model, "case_id": case_id, "label": labels_by_id[case_id],
                         "fraud_probability": saved[case_id]} for case_id in case_ids)
    output = {
        "purpose": "terminal risk-classifier ablation on shared balanced-controller validation traces; no test access",
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "cases": len(case_ids),
        "positive_cases": int(labels.sum()),
        "negative_cases": int(len(labels) - labels.sum()),
        "precision_targets": list(PRECISION_TARGETS),
        "case_ids": case_ids,
        "rows": rows,
        "slm_predictions": per_case,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run terminal risk-classifier ablation on frozen balanced traces.")
    parser.add_argument("reports", nargs="*", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/development/terminal_classifier_ablation_500.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/terminal_classifier_ablation_500.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/terminal_classifier_ablation_500.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.expected_cases < 1 or args.timeout_seconds <= 0 or args.max_attempts < 1:
        raise SystemExit("case count, timeout, and maximum attempts must be positive")

    reports, case_ids = load_and_validate_reports(args.reports, args.expected_cases)
    contract = {
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "models": [report["model"] for report in reports],
        "case_ids": case_ids,
    }
    if args.dry_run:
        print(json.dumps({**contract, "calls": len(reports) * len(case_ids), "test_access": "not queried"}, indent=2))
        return

    for report in reports:
        require_ollama_model(args.endpoint, report["model"])
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = sqlite3.connect(args.checkpoint)
    initialise_checkpoint(checkpoint, contract)
    try:
        for report in reports:
            model = report["model"]
            completed = {row[0] for row in checkpoint.execute(
                "SELECT case_id FROM predictions WHERE model=?", (model,)
            )}
            pairs = [pair for pair in report["pairs"] if int(pair["case_id"]) not in completed]
            print(f"{model}: {len(completed)}/{len(case_ids)} complete; running {len(pairs)}", flush=True)
            for index, pair in enumerate(pairs, start=1):
                case_id = int(pair["case_id"])
                probability, latency_ns, raw = predict_probability(
                    args.endpoint, model, pair["controller"]["evidence"], args.timeout_seconds, args.max_attempts
                )
                checkpoint.execute(
                    "INSERT INTO predictions(model,case_id,probability,latency_ns,raw_response) VALUES (?,?,?,?,?)",
                    (model, case_id, probability, latency_ns, raw),
                )
                checkpoint.commit()
                if index % 50 == 0 or index == len(pairs):
                    print(f"{model}: saved {len(completed) + index}/{len(case_ids)}", flush=True)

        labels_by_id = labels_for(case_ids, args.database, args.split_database)
        if sum(labels_by_id.values()) != 100 or len(labels_by_id) - sum(labels_by_id.values()) != 400:
            raise RuntimeError("expected the frozen 500-case cohort to contain 100 fraud and 400 non-fraud cases")
        write_outputs(reports, case_ids, labels_by_id, checkpoint, args.output, args.markdown_output)
    finally:
        checkpoint.close()
    print(f"Results: {args.output}\nTable: {args.markdown_output}", flush=True)


if __name__ == "__main__":
    main()
