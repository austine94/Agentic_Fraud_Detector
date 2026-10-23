"""Compare controller reliability on one shared empirical boundary cohort.

The 25-case cohort is selected once from the primary Qwen3:4B balanced
validation trace using the label-independent rule implemented by
``evaluate_qwen_boundary_repeatability``.  Every candidate model then receives
the same cases, prompt, schema, scorer, budget, seed, and decoding settings.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from time import perf_counter

from experiments.evaluate_qwen_boundary_repeatability import (
    enrich_summary,
    require_ollama_model,
    select_boundary_cohort,
)
from experiments.evaluate_qwen_repeatability import labels_for, rows, sequence, summarise
from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


DEFAULT_MODELS = (
    "qwen3:4b",
    "smollm:360m",
    "qwen:0.5b",
    "cajina/gemma4_e2b-q4_k_s:v01",
    "llama3.2:1b",
)
DISPLAY_NAMES = {
    "qwen3:4b": "Qwen3:4B",
    "smollm:360m": "SmolLM 360M",
    "qwen:0.5b": "Qwen 0.5B",
    "cajina/gemma4_e2b-q4_k_s:v01": "Gemma 4 E2B",
    "llama3.2:1b": "Llama 3.2 1B",
}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def count_text(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,}"


def render_markdown(output: dict) -> str:
    lines = [
        "# Cross-Model Boundary-Case Reliability",
        "",
        f"{output['repeats']} repeated fixed-seed rollouts for each model on the same {len(output['cohort'])} validation cases near empirical risk/action transitions.",
        "The cohort was selected once from the primary Qwen3:4B trace without using fraud labels.",
        "",
        "| Model | Valid actions | Action rejections | Prompt adherence |",
        "|---|---:|---:|---:|",
    ]
    for result in output["model_results"]:
        metrics = result["summary"]
        lines.append(
            f"| {result['display_name']} | {count_text(metrics['valid_controller_actions'], metrics['controller_actions'])} "
            f"({metrics['valid_controller_actions']/metrics['controller_actions']:.1%}) | "
            f"{metrics['action_rejections']} | "
            f"{count_text(metrics['initial_network_adherence_count'], metrics['episodes'])} "
            f"({metrics['initial_network_adherence_count']/metrics['episodes']:.1%}) |"
        )
    lines.extend([
        "",
        f"Total wall-clock runtime: {output['elapsed_seconds']/60:.1f} minutes.",
    ])
    return "\n".join([*lines, ""])


def render_latex(output: dict) -> str:
    row_end = r"\\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \shortstack{\textbf{Valid}\\\textbf{actions}} & \shortstack{\textbf{Action}\\\textbf{rejections}} & \shortstack{\textbf{Prompt}\\\textbf{adherence}} \\",
        r"\midrule",
    ]
    for result in output["model_results"]:
        metrics = result["summary"]
        name = result["display_name"].replace("_", r"\_")
        if result["model"] == "qwen3:4b":
            name = rf"\textbf{{{name}}}"
        lines.append(
            f"{name} & {count_text(metrics['valid_controller_actions'], metrics['controller_actions'])} & "
            f"{metrics['action_rejections']} & "
            f"{count_text(metrics['initial_network_adherence_count'], metrics['episodes'])} "
            f"({100*metrics['initial_network_adherence_count']/metrics['episodes']:.1f}\\%) {row_end}"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Cross-model reliability over {output['repeats']} repeated rollouts of {len(output['cohort'])} shared cases selected near empirical evidence-acquisition boundaries. Prompt adherence denotes acquisition of the initial network evidence required by the balanced objective.}}",
        r"\label{table:cross_model_boundary_reliability}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local controllers on one shared validation boundary cohort.")
    parser.add_argument("--cohort-source", type=Path, default=Path("results/development/qwen3_4b_balanced_500_stratified.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--goal-profile", default="balanced")
    parser.add_argument("--case-count", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260924, help="Held fixed across every rollout")
    parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/development/cross_model_boundary_repeatability_25x25.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/cross_model_boundary_repeatability_25x25.md"))
    parser.add_argument("--latex-output", type=Path, default=Path("paper_results/cross_model_boundary_repeatability_25x25.tex"))
    parser.add_argument("--dry-run", action="store_true", help="Validate the shared cohort and print the run size")
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least two")
    if len(set(args.models)) != len(args.models):
        raise SystemExit("--models must not contain duplicates")

    source = json.loads(args.cohort_source.read_text(encoding="utf-8"))
    if source.get("goal_profile") != args.goal_profile:
        raise ValueError("cohort source and requested goal profile do not match")
    selected = select_boundary_cohort(source, args.case_count)
    case_ids = sorted(row["case_id"] for row in selected)
    labels_by_id = labels_for(case_ids, args.database, args.split_database)
    if args.dry_run:
        print(json.dumps({
            "models": args.models,
            "case_count": len(case_ids),
            "repeats_per_model": args.repeats,
            "episodes_per_model": len(case_ids) * args.repeats,
            "total_episodes": len(args.models) * len(case_ids) * args.repeats,
            "selection_uses_labels": False,
            "case_ids": case_ids,
        }, indent=2))
        return
    for model in args.models:
        try:
            require_ollama_model(args.endpoint, model)
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
    training_labels = [int(row["is_fraud"]) for row in train]
    positive_weight = (len(training_labels) - sum(training_labels)) / sum(training_labels)
    scorer = fit_state_risk_scorers(
        [safe_evidence_from_row(row) for row in train],
        training_labels,
        seed=20260924,
        sample_weights=[positive_weight if label else 1.0 for label in training_labels],
    )
    runtime = AgenticControllerRuntime(
        EvidenceService(args.database, args.profiles, split_database=args.split_database),
        scorer,
        entry_gate=args.entry_gate,
        budget=11.25,
    )

    run_started = perf_counter()
    model_results = []
    all_observations = []
    for model_index, model in enumerate(args.models, start=1):
        model_started = perf_counter()
        observations = []
        print(f"starting {display_name(model)} ({model_index}/{len(args.models)})", flush=True)
        for repeat in range(args.repeats):
            policy = OllamaControllerPolicy(
                model,
                endpoint=args.endpoint,
                goal_profile=args.goal_profile,
                seed=args.seed,
            )
            for case_id in case_ids:
                trace = runtime.run(case_id, policy).as_dict()
                if any(step["proposed_action"] == "__transport_error__" for step in trace["controller_steps"]):
                    raise RuntimeError(
                        f"Ollama transport failed for {model}, repeat {repeat}, case {case_id}; no result files were written"
                    )
                observation = {
                    "model": model,
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
                }
                observations.append(observation)
                all_observations.append(observation)
            if repeat == 0 or (repeat + 1) % 5 == 0 or repeat + 1 == args.repeats:
                print(
                    f"{display_name(model)}: repeat {repeat + 1}/{args.repeats} "
                    f"({len(observations)}/{args.repeats * len(case_ids)} episodes, "
                    f"{(perf_counter() - model_started) / 60:.1f} min)",
                    flush=True,
                )
        summary, per_case = summarise(observations, case_ids, args.repeats)
        summary = enrich_summary(summary, observations, perf_counter() - model_started)
        model_results.append({
            "model": model,
            "display_name": display_name(model),
            "summary": summary,
            "per_case": per_case,
        })

    selection_by_id = {row["case_id"]: row for row in selected}
    cohort = [{
        "case_id": case_id,
        "label": labels_by_id[case_id],
        "gate": selection_by_id[case_id]["gate"],
        "reference_action": selection_by_id[case_id]["action"],
        "reference_risk": selection_by_id[case_id]["risk"],
        "boundary_distance": selection_by_id[case_id]["boundary_distance"],
        "selection_rank": selection_by_id[case_id]["selection_rank"],
    } for case_id in case_ids]
    output = {
        "purpose": "fixed-seed cross-model repeatability on a shared empirical validation boundary cohort; no test access",
        "boundary_source_model": source.get("model"),
        "selection_uses_labels": False,
        "goal_profile": args.goal_profile,
        "seed": args.seed,
        "temperature": 0.0,
        "repeats": args.repeats,
        "elapsed_seconds": perf_counter() - run_started,
        "cohort": cohort,
        "model_results": model_results,
        "observations": all_observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(output), encoding="utf-8")
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(render_latex(output), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "models": args.models,
        "episodes": len(all_observations),
        "elapsed_minutes": output["elapsed_seconds"] / 60,
    }, indent=2))


if __name__ == "__main__":
    main()
