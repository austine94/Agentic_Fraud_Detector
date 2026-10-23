"""Replace the trained risk classifier with each balanced controller SLM.

The governed control loop is unchanged: a risk score is produced from the
current evidence, the balanced controller receives that score and decides
whether to stop or acquire the next permitted evidence layer, and the loop
repeats.  The ablation changes only the source of the risk score: the same SLM
used by the controller produces it from evidence instead of the fitted
classifier.

Completed model/case episodes are checkpointed in SQLite and are safe to
resume.  Labels are joined only after every inference episode is complete.
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
from typing import Any, Mapping

import numpy as np

from experiments.evaluate_qwen_boundary_repeatability import require_ollama_model
from experiments.evaluate_qwen_repeatability import labels_for
from experiments.run_terminal_classifier_ablation import (
    DEFAULT_REPORTS,
    DISPLAY_NAMES,
    PRECISION_TARGETS,
    load_and_validate_reports,
    metric_summary,
)
from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService


SEED = 20260924
ENTRY_GATE = 0.533566727121687
PROMPT_VERSION = "slm_risk_scorer_v1"


def risk_prompt(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "fraud_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["fraud_probability"],
        "additionalProperties": False,
    }
    return {
        "stream": False,
        "format": schema,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the fraud-risk scoring component in a sequential payment investigation. "
                    "Using only the currently acquired evidence, estimate the probability that the transaction is fraudulent. "
                    "A probability of 0 means certainly not fraud and 1 means certainly fraud. "
                    "Do not request evidence or explain the score. Return only the required JSON object."
                ),
            },
            {"role": "user", "content": "Current acquired evidence:\n" + json.dumps(evidence, sort_keys=True)},
        ],
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": 64},
    }


class OllamaRiskScorer:
    """Evidence-only SLM scorer implementing the runtime's RiskScorer protocol."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str,
        timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.call_records: list[dict[str, Any]] = []

    def reset_records(self) -> None:
        self.call_records.clear()

    def predict_risk(self, evidence: Mapping[str, Mapping[str, Any]]) -> float:
        payload = risk_prompt(evidence) | {"model": self.model}
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        last_raw: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = perf_counter_ns()
            try:
                request = urllib.request.Request(
                    self.endpoint, data=encoded, headers={"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                        response_payload = json.loads(response.read())
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                    response = subprocess.run(
                        ["curl", "--fail", "--silent", "--show-error", "--max-time", str(self.timeout_seconds),
                         "-H", "Content-Type: application/json", "--data-binary", "@-", self.endpoint],
                        input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                    )
                    try:
                        response_payload = json.loads(response.stdout)
                    except json.JSONDecodeError as parse_error:
                        raise RuntimeError(f"invalid Ollama response after urllib failure: {error}") from parse_error
                raw = str(response_payload["message"]["content"])
                last_raw = raw
                probability = float(json.loads(raw)["fraud_probability"])
                if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"fraud_probability is outside [0,1]: {probability}")
                self.call_records.append({
                    "probability": probability,
                    "latency_ns": perf_counter_ns() - started,
                    "attempts": attempt,
                    "raw_response": raw,
                    "evidence_state": list(evidence),
                })
                return probability
            except (KeyError, TypeError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
                last_error = error
        raise RuntimeError(
            f"{self.model} risk scorer failed after {self.max_attempts} attempts: {last_error}; "
            f"last raw response={last_raw!r}"
        )


def initialise_checkpoint(connection: sqlite3.Connection, contract: dict) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
             model TEXT NOT NULL, case_id INTEGER NOT NULL, status TEXT NOT NULL,
             final_risk REAL NOT NULL, risk_calls INTEGER NOT NULL, controller_calls INTEGER NOT NULL,
             trace_json TEXT NOT NULL, PRIMARY KEY(model,case_id)) WITHOUT ROWID"""
    )
    encoded = json.dumps(contract, sort_keys=True)
    saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if saved is None:
        connection.execute("INSERT INTO metadata(key,value) VALUES ('contract',?)", (encoded,))
    elif saved[0] != encoded:
        raise RuntimeError("checkpoint belongs to a different SLM risk-classifier ablation contract")
    connection.commit()


def compact_trace(trace: dict, risk_calls: list[dict]) -> str:
    return json.dumps({
        "status": trace["status"],
        "risk_trajectory": trace["risk_trajectory"],
        "controller_steps": [{
            "proposed_action": step["proposed_action"],
            "valid": step["valid"],
            "controller_latency_ns": step["controller_latency_ns"],
            "decision_source": step.get("decision_source", "policy"),
            "raw_response": step.get("raw_response"),
        } for step in trace["controller_steps"]],
        "risk_scorer_calls": risk_calls,
        "tool_calls": trace["tool_calls"],
        "total_cost": trace["total_cost"],
    }, separators=(",", ":"))


def render_markdown(output: dict) -> str:
    lines = [
        "# SLM Risk-Classifier Ablation",
        "",
        f"Shared routed validation cohort: {output['cases']} cases ({output['positive_cases']} fraud and "
        f"{output['negative_cases']} non-fraud).",
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
        "The reference is the existing Qwen3:4B balanced controller with the frozen trained risk classifier. "
        "In every ablated row, the named SLM replaces that classifier at the initial gate and after each evidence acquisition; "
        "the same model then receives its own risk score through the unchanged balanced controller prompt.",
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
        "method": "Qwen3:4B controller + trained risk classifier",
        "model": "qwen3:4b",
        "risk_scorer": "trained_classifier",
        "metrics": metric_summary(labels, reference_scores),
    }]
    per_case: list[dict] = []
    for report in reports:
        model = report["model"]
        saved = {int(row[0]): row[1:] for row in checkpoint.execute(
            "SELECT case_id,status,final_risk,risk_calls,controller_calls FROM episodes WHERE model=?", (model,)
        )}
        if len(saved) != len(case_ids):
            raise RuntimeError(f"cannot summarise incomplete episodes for {model}: {len(saved)}/{len(case_ids)}")
        scores = np.asarray([float(saved[case_id][1]) for case_id in case_ids])
        display = DISPLAY_NAMES.get(model, model)
        rows.append({
            "method": f"{display} controller + {display} risk scorer",
            "model": model,
            "risk_scorer": "same_slm",
            "metrics": metric_summary(labels, scores),
            "statuses": {status: sum(values[0] == status for values in saved.values())
                         for status in sorted({values[0] for values in saved.values()})},
            "risk_scorer_calls": int(sum(values[2] for values in saved.values())),
            "controller_calls": int(sum(values[3] for values in saved.values())),
        })
        per_case.extend({
            "model": model,
            "case_id": case_id,
            "label": labels_by_id[case_id],
            "status": saved[case_id][0],
            "final_risk": saved[case_id][1],
            "risk_calls": saved[case_id][2],
            "controller_calls": saved[case_id][3],
        } for case_id in case_ids)
    output = {
        "purpose": "ablate the trained classifier with the same SLM inside the sequential balanced control loop; no test access",
        "prompt_version": PROMPT_VERSION,
        "controller_goal_profile": "balanced",
        "seed": SEED,
        "entry_gate": ENTRY_GATE,
        "cases": len(case_ids),
        "positive_cases": int(labels.sum()),
        "negative_cases": int(len(labels) - labels.sum()),
        "precision_targets": list(PRECISION_TARGETS),
        "case_ids": case_ids,
        "rows": rows,
        "episodes": per_case,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace the trained sequential risk classifier with each controller SLM.")
    parser.add_argument("reports", nargs="*", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/development/slm_risk_classifier_ablation_500.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/slm_risk_classifier_ablation_500.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/slm_risk_classifier_ablation_500.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.expected_cases < 1 or args.timeout_seconds <= 0 or args.max_attempts < 1:
        raise SystemExit("case count, timeout, and maximum attempts must be positive")

    reports, case_ids = load_and_validate_reports(args.reports, args.expected_cases)
    contract = {
        "prompt_version": PROMPT_VERSION,
        "controller_goal_profile": "balanced",
        "seed": SEED,
        "entry_gate": ENTRY_GATE,
        "models": [report["model"] for report in reports],
        "case_ids": case_ids,
    }
    if args.dry_run:
        print(json.dumps({
            **contract,
            "maximum_risk_scorer_calls": len(reports) * len(case_ids) * 4,
            "maximum_controller_calls": len(reports) * len(case_ids) * 4,
            "test_access": "not queried",
        }, indent=2))
        return

    for report in reports:
        require_ollama_model(args.endpoint, report["model"])
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = sqlite3.connect(args.checkpoint)
    initialise_checkpoint(checkpoint, contract)
    service = EvidenceService(args.database, args.profiles, split_database=args.split_database)
    try:
        for report in reports:
            model = report["model"]
            completed = {int(row[0]) for row in checkpoint.execute(
                "SELECT case_id FROM episodes WHERE model=?", (model,)
            )}
            remaining = [case_id for case_id in case_ids if case_id not in completed]
            print(f"{model}: {len(completed)}/{len(case_ids)} complete; running {len(remaining)}", flush=True)
            scorer = OllamaRiskScorer(
                model, endpoint=args.endpoint, timeout_seconds=args.timeout_seconds, max_attempts=args.max_attempts
            )
            policy = OllamaControllerPolicy(
                model, endpoint=args.endpoint, seed=SEED, temperature=0.0,
                timeout_seconds=args.timeout_seconds, goal_profile="balanced",
            )
            runtime = AgenticControllerRuntime(
                service, scorer, entry_gate=ENTRY_GATE, budget=11.25, auto_stop_at_full_evidence=True,
            )
            for index, case_id in enumerate(remaining, start=1):
                scorer.reset_records()
                trace = runtime.run(case_id, policy).as_dict()
                if not trace["risk_trajectory"]:
                    raise RuntimeError(f"{model} case {case_id} produced no risk score")
                if trace["status"] not in {"below_entry_gate", "stopped"}:
                    raise RuntimeError(f"{model} case {case_id} ended with {trace['status']}; episode not checkpointed")
                checkpoint.execute(
                    """INSERT INTO episodes
                       (model,case_id,status,final_risk,risk_calls,controller_calls,trace_json)
                       VALUES (?,?,?,?,?,?,?)""",
                    (model, case_id, trace["status"], float(trace["risk_trajectory"][-1]["risk"]),
                     len(scorer.call_records), len(trace["controller_steps"]),
                     compact_trace(trace, scorer.call_records)),
                )
                checkpoint.commit()
                if index % 25 == 0 or index == len(remaining):
                    print(f"{model}: saved {len(completed) + index}/{len(case_ids)}", flush=True)

        labels_by_id = labels_for(case_ids, args.database, args.split_database)
        if sum(labels_by_id.values()) != 100 or len(labels_by_id) - sum(labels_by_id.values()) != 400:
            raise RuntimeError("expected the frozen cohort to contain 100 fraud and 400 non-fraud cases")
        write_outputs(reports, case_ids, labels_by_id, checkpoint, args.output, args.markdown_output)
    finally:
        checkpoint.close()
    print(f"Results: {args.output}\nTable: {args.markdown_output}", flush=True)


if __name__ == "__main__":
    main()
