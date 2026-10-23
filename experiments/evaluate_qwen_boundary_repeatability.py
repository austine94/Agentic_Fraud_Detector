"""Repeat Qwen-balanced on empirically difficult acquisition-boundary cases.

The fixed cohort is selected from a completed validation-only controller report.
At the origin and destination acquisition gates, cases are ranked by the
absolute risk gap to the closest case that produced the opposite action.  A
round-robin selection across gate/action strata retains both sides of both
observed action transitions.  Fraud labels are not used for selection; they
are attached only afterward as audit metadata and are never passed to the
controller, scorer, prompt, or evidence service.

This is an empirical risk/action-boundary test.  It does not claim access to
the language model's token probabilities or its complete decision boundary.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.evaluate_qwen_repeatability import labels_for, rows, sequence, summarise
from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


GATES = (
    ("origin_gate", "acquire_origin"),
    ("destination_gate", "acquire_destination"),
)


def require_ollama_model(endpoint: str, model: str, timeout_seconds: float = 5.0) -> None:
    """Fail before a long run when the local server or requested model is absent."""
    tags_endpoint = endpoint.rsplit("/", 1)[0] + "/tags"
    try:
        with urllib.request.urlopen(tags_endpoint, timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        try:
            response = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--max-time", str(timeout_seconds), tags_endpoint],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            payload = json.loads(response.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as fallback_error:
            raise RuntimeError(
                f"Ollama is not reachable at {tags_endpoint}; start Ollama before running this experiment "
                f"({error}; curl fallback: {fallback_error})"
            ) from fallback_error
    installed = {
        str(item.get(key))
        for item in payload.get("models", [])
        for key in ("name", "model")
        if item.get(key)
    }
    if model not in installed:
        raise RuntimeError(f"Ollama model {model!r} is not installed; available models: {sorted(installed)}")


def boundary_candidates(source: dict) -> list[dict]:
    """Rank cases by proximity to an opposite action at the same risk gate."""
    observations: dict[str, list[dict]] = {gate: [] for gate, _ in GATES}
    for pair in source.get("pairs", []):
        case_id = int(pair["case_id"])
        for step in pair["controller"].get("controller_steps", []):
            if not step.get("valid", False):
                continue
            allowed = tuple(step["request"]["allowed_actions"])
            action = str(step["proposed_action"])
            for gate, acquisition_action in GATES:
                if set(allowed) == {"stop", acquisition_action} and action in {"stop", acquisition_action}:
                    observations[gate].append({
                        "case_id": case_id,
                        "gate": gate,
                        "action": action,
                        "risk": float(step["request"]["risk"]),
                    })

    candidates = []
    for gate, acquisition_action in GATES:
        values = observations[gate]
        actions = ("stop", acquisition_action)
        by_action = {action: [value for value in values if value["action"] == action] for action in actions}
        if any(not by_action[action] for action in actions):
            raise ValueError(f"source report must contain both stop and {acquisition_action} at {gate}")
        for action in actions:
            opposite_action = acquisition_action if action == "stop" else "stop"
            opposite = by_action[opposite_action]
            for value in by_action[action]:
                nearest = min(opposite, key=lambda row: (abs(value["risk"] - row["risk"]), row["case_id"]))
                candidates.append({
                    **value,
                    "opposite_action": opposite_action,
                    "nearest_opposite_case_id": nearest["case_id"],
                    "nearest_opposite_risk": nearest["risk"],
                    "boundary_distance": abs(value["risk"] - nearest["risk"]),
                })
    return candidates


def select_boundary_cohort(source: dict, case_count: int) -> list[dict]:
    """Select unique cases round-robin across the two sides of both gates."""
    if case_count < 4:
        raise ValueError("case_count must be at least four to cover every gate/action stratum")
    candidates = boundary_candidates(source)
    strata = [
        (gate, action)
        for gate, acquisition_action in GATES
        for action in ("stop", acquisition_action)
    ]
    queues = {
        stratum: sorted(
            (row for row in candidates if (row["gate"], row["action"]) == stratum),
            key=lambda row: (row["boundary_distance"], row["risk"], row["case_id"]),
        )
        for stratum in strata
    }
    offsets = {stratum: 0 for stratum in strata}
    selected: list[dict] = []
    selected_ids: set[int] = set()
    while len(selected) < case_count:
        made_progress = False
        for stratum in strata:
            queue = queues[stratum]
            while offsets[stratum] < len(queue) and queue[offsets[stratum]]["case_id"] in selected_ids:
                offsets[stratum] += 1
            if offsets[stratum] >= len(queue):
                continue
            row = dict(queue[offsets[stratum]])
            offsets[stratum] += 1
            row["selection_rank"] = len(selected) + 1
            selected.append(row)
            selected_ids.add(row["case_id"])
            made_progress = True
            if len(selected) == case_count:
                break
        if not made_progress:
            raise ValueError(f"source report contains only {len(selected)} unique boundary candidates")
    return selected


def enrich_summary(summary: dict, observations: list[dict], elapsed_seconds: float) -> dict:
    steps = [step for observation in observations for step in observation["steps"]]
    invalid_actions = sum(not step["valid"] for step in steps)
    invalid_limit_cases = sum(observation["status"] == "invalid_action_limit" for observation in observations)
    return {
        **summary,
        "episodes": len(observations),
        "controller_actions": len(steps),
        "valid_controller_actions": sum(bool(step["valid"]) for step in steps),
        "initial_network_adherence_count": sum(
            bool(observation["action_sequence"])
            and observation["action_sequence"][0] == "acquire_network"
            for observation in observations
        ),
        "action_rejections": invalid_actions,
        "retry_decisions": invalid_actions - invalid_limit_cases,
        "service_failure_cases": sum(observation["status"] == "service_unavailable" for observation in observations),
        "elapsed_seconds": elapsed_seconds,
        "controller_inference_seconds": float(sum(observation["controller_latency_seconds"] for observation in observations)),
        "evidence_service_seconds": float(sum(observation["service_latency_seconds"] for observation in observations)),
    }


def render_markdown(output: dict) -> str:
    metrics = output["summary"]
    lines = [
        "# Qwen3:4B Boundary-Case Repeatability Check",
        "",
        f"{output['repeats']} repeated rollouts on {len(output['cohort'])} fixed validation cases selected near empirical risk/action transitions.",
        "Fraud labels were not used for cohort selection. Seed, prompt, schema, scorer, budget, and decoding were held fixed.",
        "",
        "| Reliability measure | Count | Rate / value |",
        "|---|---:|---:|",
        f"| Evaluated boundary cases | {len(output['cohort'])} | — |",
        f"| Repeated episodes | {metrics['episodes']:,} | — |",
        f"| Valid controller actions | {metrics['valid_controller_actions']:,}/{metrics['controller_actions']:,} | {metrics['valid_controller_actions']/metrics['controller_actions']:.1%} |",
        f"| Stopped episodes | {round(metrics['stopped_episode_rate_mean'] * metrics['episodes']):,}/{metrics['episodes']:,} | {metrics['stopped_episode_rate_mean']:.1%} |",
        f"| Initial-network adherence | {metrics['initial_network_adherence_count']:,}/{metrics['episodes']:,} | {metrics['initial_network_adherence_count']/metrics['episodes']:.1%} |",
        f"| Action rejections | {metrics['action_rejections']} | — |",
        f"| Retry decisions | {metrics['retry_decisions']} | — |",
        f"| Service failures | {metrics['service_failure_cases']} | — |",
        f"| Mean sequence agreement | — | {metrics['mean_action_sequence_agreement']:.1%} |",
        f"| Minimum sequence agreement | — | {metrics['minimum_action_sequence_agreement']:.1%} |",
        f"| Mean unique sequences per case | — | {metrics['mean_unique_action_sequences']:.2f} |",
        f"| Mean per-case cost SD | — | {metrics['mean_per_case_cost_sd']:.3f} |",
        f"| Mean per-case stopped-risk SD | — | {metrics['mean_per_case_stopped_risk_sd']:.5f} |",
        f"| Wall-clock runtime | — | {metrics['elapsed_seconds']/60:.1f} min |",
        "",
        "| Case | Gate | Reference action | Boundary distance | Modal sequence | Agreement | Unique sequences |",
        "|---:|---|---|---:|---|---:|---:|",
    ]
    selection = {row["case_id"]: row for row in output["cohort"]}
    for row in output["per_case"]:
        selected = selection[row["case_id"]]
        lines.append(
            f"| {row['case_id']} | {selected['gate']} | {selected['reference_action']} | "
            f"{selected['boundary_distance']:.6f} | {' → '.join(row['modal_action_sequence'])} | "
            f"{row['action_sequence_agreement']:.1%} | {row['unique_action_sequences']} |"
        )
    return "\n".join([*lines, ""])


def render_latex(output: dict) -> str:
    metrics = output["summary"]
    cases = len(output["cohort"])
    stopped = round(metrics["stopped_episode_rate_mean"] * metrics["episodes"])
    row_end = r"\\"
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"\textbf{Reliability measure} & \textbf{Count} & \textbf{Rate / value} \\",
        r"\midrule",
        f"Evaluated boundary cases & {cases} & -- {row_end}",
        f"Repeated episodes & {metrics['episodes']:,} & -- {row_end}",
        f"Valid controller actions & {metrics['valid_controller_actions']:,}/{metrics['controller_actions']:,} & {100*metrics['valid_controller_actions']/metrics['controller_actions']:.1f}\\% {row_end}",
        f"Stopped episodes & {stopped:,}/{metrics['episodes']:,} & {100*metrics['stopped_episode_rate_mean']:.1f}\\% {row_end}",
        f"Initial-network adherence & {metrics['initial_network_adherence_count']:,}/{metrics['episodes']:,} & {100*metrics['initial_network_adherence_count']/metrics['episodes']:.1f}\\% {row_end}",
        f"Action rejections & {metrics['action_rejections']} & -- {row_end}",
        f"Retry decisions & {metrics['retry_decisions']} & -- {row_end}",
        f"Service failures & {metrics['service_failure_cases']} & -- {row_end}",
        f"Mean sequence agreement & -- & {100*metrics['mean_action_sequence_agreement']:.1f}\\% {row_end}",
        f"Minimum sequence agreement & -- & {100*metrics['minimum_action_sequence_agreement']:.1f}\\% {row_end}",
        f"Mean unique sequences per case & -- & {metrics['mean_unique_action_sequences']:.2f} {row_end}",
        f"Mean per-case cost SD & -- & {metrics['mean_per_case_cost_sd']:.3f} {row_end}",
        f"Mean stopped-risk SD & -- & {metrics['mean_per_case_stopped_risk_sd']:.5f} {row_end}",
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Reliability of the Qwen3:4B balanced controller over {output['repeats']} repeated rollouts of {cases} cases selected near empirical evidence-acquisition boundaries.}}",
        r"\label{table:boundary_reliability}",
        r"\end{table}",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat Qwen-balanced on empirical validation action-boundary cases.")
    parser.add_argument("--cohort-source", type=Path, default=Path("results/development/qwen3_4b_balanced_500_stratified.json"))
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--goal-profile", default="balanced")
    parser.add_argument("--case-count", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260924, help="Held fixed across every rollout")
    parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/qwen3_4b_boundary_repeatability_25x50.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/qwen3_4b_boundary_repeatability_25x50.md"))
    parser.add_argument("--latex-output", type=Path, default=Path("paper_results/qwen3_4b_boundary_repeatability_25x50.tex"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the frozen cohort without calling Ollama")
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least two")

    source = json.loads(args.cohort_source.read_text(encoding="utf-8"))
    if source.get("goal_profile") != args.goal_profile:
        raise ValueError("cohort source and requested goal profile do not match")
    selected = select_boundary_cohort(source, args.case_count)
    case_ids = sorted(row["case_id"] for row in selected)
    labels_by_id = labels_for(case_ids, args.database, args.split_database)
    source_pairs = {int(pair["case_id"]): pair for pair in source["pairs"]}
    if args.dry_run:
        print(json.dumps({
            "case_count": len(selected),
            "selection_uses_labels": False,
            "strata": dict(Counter(f"{row['gate']}:{row['action']}" for row in selected)),
            "case_ids": case_ids,
            "estimated_episodes": len(selected) * args.repeats,
        }, indent=2))
        return
    try:
        require_ollama_model(args.endpoint, args.model)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", 45))
    finally:
        connection.close()
    labels = [int(row["is_fraud"]) for row in train]
    positive_weight = (len(labels) - sum(labels)) / sum(labels)
    scorer = fit_state_risk_scorers(
        [safe_evidence_from_row(row) for row in train],
        labels,
        seed=20260924,
        sample_weights=[positive_weight if label else 1.0 for label in labels],
    )
    runtime = AgenticControllerRuntime(
        EvidenceService(args.database, args.profiles, split_database=args.split_database),
        scorer,
        entry_gate=args.entry_gate,
        budget=11.25,
    )

    observations = []
    started = perf_counter()
    for repeat in range(args.repeats):
        policy = OllamaControllerPolicy(
            args.model,
            endpoint=args.endpoint,
            goal_profile=args.goal_profile,
            seed=args.seed,
        )
        for case_id in case_ids:
            trace = runtime.run(case_id, policy).as_dict()
            if any(step["proposed_action"] == "__transport_error__" for step in trace["controller_steps"]):
                raise RuntimeError(
                    f"Ollama transport failed during repeat {repeat}, case {case_id}; no result files were written"
                )
            observations.append({
                "repeat": repeat,
                "case_id": case_id,
                "status": trace["status"],
                "action_sequence": list(sequence(trace)),
                "cost": trace["total_cost"],
                "stopped_risk": trace["risk_trajectory"][-1]["risk"],
                "controller_latency_seconds": sum(step["controller_latency_ns"] for step in trace["controller_steps"]) / 1e9,
                "service_latency_seconds": trace["total_latency_ns"] / 1e9,
                "steps": [
                    {"proposed_action": step["proposed_action"], "valid": step["valid"]}
                    for step in trace["controller_steps"]
                ],
            })
        if repeat == 0 or (repeat + 1) % 5 == 0 or repeat + 1 == args.repeats:
            elapsed = perf_counter() - started
            print(
                f"completed repeat {repeat + 1}/{args.repeats} "
                f"({len(observations)}/{args.repeats * len(case_ids)} episodes, {elapsed / 60:.1f} min)",
                flush=True,
            )
    elapsed_seconds = perf_counter() - started
    summary, per_case = summarise(observations, case_ids, args.repeats)
    summary = enrich_summary(summary, observations, elapsed_seconds)

    selection_by_id = {row["case_id"]: row for row in selected}
    cohort = []
    for case_id in case_ids:
        row = selection_by_id[case_id]
        pair = source_pairs[case_id]
        cohort.append({
            "case_id": case_id,
            "label": labels_by_id[case_id],
            "gate": row["gate"],
            "reference_action": row["action"],
            "reference_risk": row["risk"],
            "opposite_action": row["opposite_action"],
            "nearest_opposite_case_id": row["nearest_opposite_case_id"],
            "nearest_opposite_risk": row["nearest_opposite_risk"],
            "boundary_distance": row["boundary_distance"],
            "selection_rank": row["selection_rank"],
            "reference_action_sequence": list(sequence(pair["controller"])),
        })

    output = {
        "purpose": "fixed-seed Qwen-balanced repeatability on empirical validation risk/action-boundary cases; no test access",
        "boundary_definition": (
            "At each post-network and post-origin gate, absolute calibrated-risk distance to the nearest case "
            "with the opposite stop/acquire action; unique cases selected round-robin across gate/action strata"
        ),
        "selection_uses_labels": False,
        "model": args.model,
        "goal_profile": args.goal_profile,
        "seed": args.seed,
        "temperature": 0.0,
        "repeats": args.repeats,
        "cohort": cohort,
        "summary": summary,
        "per_case": per_case,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(output), encoding="utf-8")
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(render_latex(output), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
