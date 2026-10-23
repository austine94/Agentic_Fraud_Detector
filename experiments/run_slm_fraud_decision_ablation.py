"""Replace the trained risk classifier with categorical SLM decisions.

At every evidence state the same local model used by the balanced controller
returns exactly ``fraud`` or ``not_fraud``.  That categorical assessment is
then supplied to the evidence controller, which may stop or request the next
governed evidence layer.  No probability is requested from or exposed by the
SLM.  Internally, the runtime adapter maps the two labels to 1/0 solely to
preserve the existing entry-gate and trace interfaces.

Completed model/case episodes are checkpointed in SQLite and can be resumed.
Fraud labels are joined only after inference for offline evaluation.
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
from sklearn.metrics import precision_recall_curve

from experiments.evaluate_qwen_boundary_repeatability import require_ollama_model
from experiments.evaluate_qwen_repeatability import labels_for
from experiments.run_terminal_classifier_ablation import DEFAULT_REPORTS, DISPLAY_NAMES, load_and_validate_reports
from experiments.summarize_controller_scenarios import FULL_EVIDENCE_THRESHOLD
from src.agentic_controller import (
    GOAL_PROFILES,
    AgenticControllerRuntime,
    ControllerDecision,
    ControllerRequest,
)
from src.evidence_service import EvidenceService


SEED = 20260924
ENTRY_GATE = 0.533566727121687
PROMPT_VERSION = "slm_categorical_fraud_decision_v1"
DECISIONS = ("fraud", "not_fraud")


def categorical_decision_prompt(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": list(DECISIONS)}},
        "required": ["decision"],
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
                    "You are the fraud-decision component in a sequential payment investigation. "
                    "Using only the currently acquired evidence, decide whether the transaction is fraud or not fraud. "
                    "Do not estimate a probability, request evidence, or explain the answer. "
                    "Return only the required JSON object."
                ),
            },
            {"role": "user", "content": "Current acquired evidence:\n" + json.dumps(evidence, sort_keys=True)},
        ],
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": 24},
    }


def categorical_controller_prompt(request: ControllerRequest) -> dict[str, Any]:
    assessment = "fraud" if request.risk == 1.0 else "not_fraud"
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(request.allowed_actions)}},
        "required": ["action"],
        "additionalProperties": False,
    }
    state = {
        "evidence": request.evidence,
        "fraud_decision": assessment,
        "allowed_actions": request.allowed_actions,
        "remaining_budget": request.remaining_budget,
    }
    system = (
        "You are a cost-aware evidence acquisition controller. A separate categorical SLM decision component has assessed "
        "the current transaction as fraud or not_fraud. You do not classify fraud and cannot invent actions. "
        "Choose exactly one action from allowed_actions. Never repeat the request or explain your choice. "
        "acquire_network costs 1 and adds cheap recipient-network history. acquire_origin costs 5 and adds "
        "sender-ledger impact evidence. acquire_destination costs 5.25 and adds recipient plus combined-impact evidence. "
        "A case reaches you only after the entry gate. At the first gate, type and amount alone are insufficient, so choose "
        "acquire_network when it is allowed. " + GOAL_PROFILES["balanced"]
    )
    return {
        "stream": False,
        "format": schema,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Controller state:\n" + json.dumps(state, sort_keys=True) +
             "\nChoose one allowed action now."},
        ],
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": 24},
    }


def ollama_json_call(
    endpoint: str,
    model: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, int]:
    encoded = json.dumps(payload | {"model": model}).encode("utf-8")
    started = perf_counter_ns()
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
    return json.loads(raw), raw, perf_counter_ns() - started


class OllamaCategoricalDecisionScorer:
    """Categorical fraud decision adapted to the runtime's numeric interface."""

    def __init__(self, model: str, *, endpoint: str, timeout_seconds: float, max_attempts: int) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.call_records: list[dict[str, Any]] = []

    def reset_records(self) -> None:
        self.call_records.clear()

    def predict_risk(self, evidence: Mapping[str, Mapping[str, Any]]) -> float:
        last_error: Exception | None = None
        last_raw: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                parsed, raw, latency_ns = ollama_json_call(
                    self.endpoint, self.model, categorical_decision_prompt(evidence), self.timeout_seconds
                )
                last_raw = raw
                decision = str(parsed["decision"])
                if decision not in DECISIONS:
                    raise ValueError(f"invalid categorical decision: {decision!r}")
                self.call_records.append({
                    "decision": decision,
                    "latency_ns": latency_ns,
                    "attempts": attempt,
                    "raw_response": raw,
                    "evidence_state": list(evidence),
                })
                return 1.0 if decision == "fraud" else 0.0
            except (KeyError, TypeError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError,
                    json.JSONDecodeError) as error:
                last_error = error
        raise RuntimeError(
            f"{self.model} categorical decision failed after {self.max_attempts} attempts: {last_error}; "
            f"last raw response={last_raw!r}"
        )


class OllamaCategoricalControllerPolicy:
    """Balanced evidence controller receiving a categorical fraud assessment."""

    def __init__(self, model: str, *, endpoint: str, timeout_seconds: float, max_attempts: int) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        last_error: Exception | None = None
        last_raw: str | None = None
        for _ in range(self.max_attempts):
            try:
                parsed, raw, _ = ollama_json_call(
                    self.endpoint, self.model, categorical_controller_prompt(request), self.timeout_seconds
                )
                last_raw = raw
                return ControllerDecision(action=str(parsed["action"]), raw_response=raw)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError,
                    json.JSONDecodeError) as error:
                last_error = error
        return ControllerDecision(
            action="__categorical_controller_error__",
            raw_response=f"{last_error}; last raw response={last_raw!r}",
        )


def categorical_metrics(labels: list[int], decisions: list[str]) -> dict[str, float | int]:
    if len(labels) != len(decisions) or any(decision not in DECISIONS for decision in decisions):
        raise ValueError("labels and categorical decisions must be aligned and valid")
    predicted = [decision == "fraud" for decision in decisions]
    tp = sum(label == 1 and prediction for label, prediction in zip(labels, predicted))
    fp = sum(label == 0 and prediction for label, prediction in zip(labels, predicted))
    fn = sum(label == 1 and not prediction for label, prediction in zip(labels, predicted))
    tn = sum(label == 0 and not prediction for label, prediction in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "fraud_decisions": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def classifier_operating_point(
    labels: list[int], scores: list[float], target_precision: float
) -> dict[str, float | int]:
    """Select the nearest empirical precision point, breaking ties by recall."""
    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(label_array, score_array)
    distances = np.abs(precision[:-1] - target_precision)
    nearest = np.flatnonzero(distances == distances.min())
    maximum_recall = recall[nearest].max()
    index = int(nearest[recall[nearest] == maximum_recall][0])
    decisions = ["fraud" if score >= thresholds[index] else "not_fraud" for score in scores]
    return categorical_metrics(labels, decisions) | {
        "target_precision": target_precision,
        "threshold": float(thresholds[index]),
    }


def initialise_checkpoint(connection: sqlite3.Connection, contract: dict) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
             model TEXT NOT NULL, case_id INTEGER NOT NULL, status TEXT NOT NULL,
             final_decision TEXT NOT NULL CHECK(final_decision IN ('fraud','not_fraud')),
             decision_calls INTEGER NOT NULL, controller_calls INTEGER NOT NULL,
             trace_json TEXT NOT NULL, PRIMARY KEY(model,case_id)) WITHOUT ROWID"""
    )
    encoded = json.dumps(contract, sort_keys=True)
    saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if saved is None:
        connection.execute("INSERT INTO metadata(key,value) VALUES ('contract',?)", (encoded,))
    elif saved[0] != encoded:
        raise RuntimeError("checkpoint belongs to a different categorical-ablation contract")
    connection.commit()


def compact_trace(trace: dict, decision_calls: list[dict]) -> str:
    return json.dumps({
        "status": trace["status"],
        "categorical_decision_trajectory": decision_calls,
        "controller_steps": [{
            "proposed_action": step["proposed_action"],
            "valid": step["valid"],
            "controller_latency_ns": step["controller_latency_ns"],
            "raw_response": step.get("raw_response"),
        } for step in trace["controller_steps"]],
        "tool_calls": trace["tool_calls"],
        "total_cost": trace["total_cost"],
    }, separators=(",", ":"))


def render_markdown(output: dict) -> str:
    lines = [
        "# Categorical SLM Fraud-Decision Ablation",
        "",
        f"Shared routed validation cohort: {output['cases']} cases ({output['positive_cases']} fraud and "
        f"{output['negative_cases']} non-fraud).",
        "",
        "| Method | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in output["rows"]:
        metrics = row["metrics"]
        precision = "—" if metrics["fraud_decisions"] == 0 else f"{metrics['precision']:.1%}"
        lines.append(
            f"| {row['method']} | {precision} | {metrics['recall']:.1%} | "
            f"{metrics['f1']:.3f} |"
        )
    lines.extend([
        "",
        "The two trained-classifier reference rows select the nearest available empirical operating point to the stated "
        "precision on this validation cohort, breaking ties by recall. Each ablated row uses only categorical `fraud`/`not_fraud` SLM decisions "
        "throughout the sequential evidence loop; no SLM probability is requested.",
        "",
    ])
    return "\n".join(lines)


def render_latex(output: dict) -> str:
    row_end = r"\\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        rf"\textbf{{Configuration}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1}} {row_end}",
        r"\midrule",
    ]
    for row in output["rows"]:
        metrics = row["metrics"]
        method = row["method"]
        if row["decision_source"] == "trained_classifier_at_target_precision":
            target = int(round(float(metrics["target_precision"]) * 100))
            method = rf"\textbf{{Qwen3:4B + classifier ($\approx${target}\%)}}"
        else:
            method = method.replace(" controller + ", " + ").replace(" categorical decision", " categorical")
        precision = "--" if metrics["fraud_decisions"] == 0 else f"{100 * float(metrics['precision']):.1f}\\%"
        lines.append(
            f"{method} & {precision} & {100 * float(metrics['recall']):.1f}\\% & "
            f"{float(metrics['f1']):.3f} {row_end}"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Categorical risk-classifier ablation on 500 shared routed validation cases (100 fraud and 400 non-fraud). The reference rows use the Qwen3:4B balanced controller with the trained classifier at the nearest available operating points to 95\% and 99\% precision. Ablated rows replace the classifier at every evidence state with a categorical decision from the same SLM used by the controller. A dash denotes undefined precision because no cases were classified as fraud.}",
        r"\label{table:categorical_classifier_ablation}",
        r"\end{table}",
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
    latex_path: Path | None = None,
) -> None:
    labels = [labels_by_id[case_id] for case_id in case_ids]
    qwen = next(report for report in reports if report["model"] == "qwen3:4b")
    reference_scores = [
        float(pair["controller"]["risk_trajectory"][-1]["risk"]) for pair in qwen["pairs"]
    ]
    rows = [{
        "method": f"Qwen3:4B controller + trained classifier (≈{int(target * 100)}% precision)",
        "model": "qwen3:4b",
        "decision_source": "trained_classifier_at_target_precision",
        "metrics": classifier_operating_point(labels, reference_scores, target),
    } for target in (0.95, 0.99)]
    episodes: list[dict] = []
    for report in reports:
        model = report["model"]
        saved = {int(row[0]): row[1:] for row in checkpoint.execute(
            "SELECT case_id,status,final_decision,decision_calls,controller_calls FROM episodes WHERE model=?",
            (model,),
        )}
        if len(saved) != len(case_ids):
            raise RuntimeError(f"cannot summarise incomplete episodes for {model}: {len(saved)}/{len(case_ids)}")
        decisions = [str(saved[case_id][1]) for case_id in case_ids]
        display = DISPLAY_NAMES.get(model, model)
        rows.append({
            "method": f"{display} controller + {display} categorical decision",
            "model": model,
            "decision_source": "same_slm_categorical",
            "metrics": categorical_metrics(labels, decisions),
            "statuses": {status: sum(values[0] == status for values in saved.values())
                         for status in sorted({values[0] for values in saved.values()})},
            "decision_calls": int(sum(values[2] for values in saved.values())),
            "controller_calls": int(sum(values[3] for values in saved.values())),
        })
        episodes.extend({
            "model": model,
            "case_id": case_id,
            "label": labels_by_id[case_id],
            "status": saved[case_id][0],
            "final_decision": saved[case_id][1],
            "decision_calls": saved[case_id][2],
            "controller_calls": saved[case_id][3],
        } for case_id in case_ids)
    output = {
        "purpose": "ablate the trained classifier with categorical decisions from the same SLM in the sequential balanced loop; no test access",
        "prompt_version": PROMPT_VERSION,
        "controller_goal_profile": "balanced",
        "seed": SEED,
        "entry_gate_mapping": {"fraud": 1, "not_fraud": 0, "threshold": ENTRY_GATE},
        "reference_threshold": FULL_EVIDENCE_THRESHOLD,
        "reference_operating_targets": [0.95, 0.99],
        "cases": len(case_ids),
        "positive_cases": sum(labels),
        "negative_cases": len(labels) - sum(labels),
        "case_ids": case_ids,
        "rows": rows,
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(output), encoding="utf-8")
    if latex_path is not None:
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(render_latex(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequential categorical SLM fraud-decision ablation.")
    parser.add_argument("reports", nargs="*", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--case-budget-per-model", type=int,
                        help="maximum unfinished cases per model for a checkpointed pilot/resume run")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/development/slm_fraud_decision_ablation_500.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/slm_fraud_decision_ablation_500.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/slm_fraud_decision_ablation_500.md"))
    parser.add_argument("--latex-output", type=Path, default=Path("paper_results/slm_fraud_decision_ablation_500.tex"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.expected_cases < 1 or args.timeout_seconds <= 0 or args.max_attempts < 1 or
            (args.case_budget_per_model is not None and args.case_budget_per_model < 1)):
        raise SystemExit("case count, timeout, attempts, and optional case budget must be positive")

    reports, case_ids = load_and_validate_reports(args.reports, args.expected_cases)
    contract = {
        "prompt_version": PROMPT_VERSION,
        "controller_goal_profile": "balanced",
        "seed": SEED,
        "entry_gate_mapping": {"fraud": 1, "not_fraud": 0, "threshold": ENTRY_GATE},
        "reference_threshold": FULL_EVIDENCE_THRESHOLD,
        "models": [report["model"] for report in reports],
        "case_ids": case_ids,
    }
    if args.dry_run:
        print(json.dumps({
            **contract,
            "maximum_decision_calls": len(reports) * len(case_ids) * 4,
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
            if args.case_budget_per_model is not None:
                remaining = remaining[:args.case_budget_per_model]
            print(f"{model}: {len(completed)}/{len(case_ids)} complete; running {len(remaining)}", flush=True)
            scorer = OllamaCategoricalDecisionScorer(
                model, endpoint=args.endpoint, timeout_seconds=args.timeout_seconds, max_attempts=args.max_attempts
            )
            policy = OllamaCategoricalControllerPolicy(
                model, endpoint=args.endpoint, timeout_seconds=args.timeout_seconds, max_attempts=args.max_attempts
            )
            runtime = AgenticControllerRuntime(
                service, scorer, entry_gate=ENTRY_GATE, budget=11.25, auto_stop_at_full_evidence=True,
            )
            for index, case_id in enumerate(remaining, start=1):
                scorer.reset_records()
                trace = runtime.run(case_id, policy).as_dict()
                if not scorer.call_records:
                    raise RuntimeError(f"{model} case {case_id} produced no categorical decision")
                if trace["status"] not in {"below_entry_gate", "stopped"}:
                    raise RuntimeError(f"{model} case {case_id} ended with {trace['status']}; episode not checkpointed")
                final_decision = str(scorer.call_records[-1]["decision"])
                checkpoint.execute(
                    """INSERT INTO episodes
                       (model,case_id,status,final_decision,decision_calls,controller_calls,trace_json)
                       VALUES (?,?,?,?,?,?,?)""",
                    (model, case_id, trace["status"], final_decision, len(scorer.call_records),
                     len(trace["controller_steps"]), compact_trace(trace, scorer.call_records)),
                )
                checkpoint.commit()
                if index % 25 == 0 or index == len(remaining):
                    print(f"{model}: saved {len(completed) + index}/{len(case_ids)}", flush=True)

        counts = dict(checkpoint.execute("SELECT model,COUNT(*) FROM episodes GROUP BY model"))
        if all(counts.get(report["model"], 0) == len(case_ids) for report in reports):
            labels_by_id = labels_for(case_ids, args.database, args.split_database)
            if sum(labels_by_id.values()) != 100 or len(labels_by_id) - sum(labels_by_id.values()) != 400:
                raise RuntimeError("expected the frozen cohort to contain 100 fraud and 400 non-fraud cases")
            write_outputs(
                reports, case_ids, labels_by_id, checkpoint, args.output, args.markdown_output, args.latex_output
            )
            print(f"Results: {args.output}\nMarkdown table: {args.markdown_output}\nLaTeX table: {args.latex_output}", flush=True)
        else:
            print(f"Pilot/checkpoint run complete; resume without --case-budget-per-model: {args.checkpoint}", flush=True)
    finally:
        checkpoint.close()


if __name__ == "__main__":
    main()
