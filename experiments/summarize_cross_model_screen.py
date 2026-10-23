"""Summarise the frozen, balanced-only local-controller comparison.

Each input is a paired output from :mod:`evaluate_controller_development`.
The script rejects reports that are not the same balanced routed validation
cohort, so a table cannot silently compare models on different cases.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from experiments.summarize_controller_scenarios import FULL_EVIDENCE_THRESHOLD, pr


ACTION_NAMES = ("acquire_network", "acquire_origin", "acquire_destination")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def labels_for(case_ids: list[int], database: Path, split_database: Path) -> dict[int, int]:
    placeholders = ",".join("?" for _ in case_ids)
    connection = sqlite3.connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(split_database),))
        rows = connection.execute(
            "SELECT c.case_id, c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) "
            f"WHERE a.split='validation' AND c.case_id IN ({placeholders})",
            case_ids,
        )
        result = dict(rows)
    finally:
        connection.close()
    if len(result) != len(case_ids):
        raise ValueError("every paired case must belong to the validation split")
    return result


def validate_reports(reports: list[dict], expected_cases: int) -> list[int]:
    if not reports:
        raise ValueError("at least one paired report is required")
    reference_ids: list[int] | None = None
    seen_models: set[str] = set()
    for report in reports:
        if report.get("goal_profile") != "balanced":
            raise ValueError(f"{report.get('model', 'unnamed report')} is not a balanced run")
        model = report.get("model")
        if not isinstance(model, str) or model in seen_models:
            raise ValueError("each report must name a distinct model")
        seen_models.add(model)
        ids = [int(pair["case_id"]) for pair in report.get("pairs", [])]
        if len(ids) != expected_cases or len(set(ids)) != expected_cases:
            raise ValueError(f"{model} must contain exactly {expected_cases} unique paired cases")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{model} does not use the same ordered routed cohort as the first report")
    return reference_ids or []


def summarise(report: dict, labels_by_id: dict[int, int]) -> dict:
    pairs = report["pairs"]
    labels = np.array([labels_by_id[int(pair["case_id"])] for pair in pairs])
    stopped_scores = np.array([pair["controller"]["risk_trajectory"][-1]["risk"] for pair in pairs])
    full_scores = np.array([pair["full_evidence"]["risk_trajectory"][-1]["risk"] for pair in pairs])
    precision, recall = pr(labels, stopped_scores, FULL_EVIDENCE_THRESHOLD)
    full_precision, full_recall = pr(labels, full_scores, FULL_EVIDENCE_THRESHOLD)

    traces = [pair["controller"] for pair in pairs]
    steps = [step for trace in traces for step in trace["controller_steps"]]
    costs = [trace["total_cost"] for trace in traces]
    decision_latencies_s = [step["controller_latency_ns"] / 1e9 for step in steps]
    episode_decision_s = [sum(step["controller_latency_ns"] for step in trace["controller_steps"]) / 1e9 for trace in traces]
    episode_service_s = [trace["total_latency_ns"] / 1e9 for trace in traces]
    episode_end_to_end_s = [decision + service for decision, service in zip(episode_decision_s, episode_service_s)]
    invalid_steps = [step for step in steps if not step["valid"]]
    invalid_limit_cases = sum(trace["status"] == "invalid_action_limit" for trace in traces)
    budget_exceeded_cases = sum(trace["status"] == "budget_exceeded" for trace in traces)
    stopped_cases = sum(trace["status"] == "stopped" for trace in traces)
    nonterminal_gateway_rejections = len(invalid_steps) - invalid_limit_cases
    action_counts = {
        action: sum(step["valid"] and step["proposed_action"] == action for step in steps)
        for action in ACTION_NAMES
    }
    acquisition_depths = [sum(step["valid"] and step["proposed_action"] in ACTION_NAMES for step in trace["controller_steps"]) for trace in traces]
    first_actions = [trace["controller_steps"][0]["proposed_action"] for trace in traces if trace["controller_steps"]]

    return {
        "model": report["model"],
        "cases": len(pairs),
        "positives": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, stopped_scores)),
        "full_evidence_pr_auc": float(average_precision_score(labels, full_scores)),
        "pr_auc_change": float(average_precision_score(labels, stopped_scores) - average_precision_score(labels, full_scores)),
        "precision_at_fixed_threshold": precision,
        "recall_at_fixed_threshold": recall,
        "precision_change_at_fixed_threshold_pp": (precision - full_precision) * 100,
        "recall_change_at_fixed_threshold_pp": (recall - full_recall) * 100,
        "cost_saving_mean_pct": float(np.mean(1 - np.array(costs) / 11.25) * 100),
        "cost_saving_sd_pct": float(np.std(1 - np.array(costs) / 11.25, ddof=1) * 100),
        "mean_acquisition_cost": float(np.mean(costs)),
        "mean_stopping_depth": float(np.mean(acquisition_depths)),
        "stopping_depth_sd": float(np.std(acquisition_depths, ddof=1)),
        "action_mix_per_case": {action: action_counts[action] / len(pairs) for action in ACTION_NAMES},
        "initial_network_acquisition_rate": sum(action == "acquire_network" for action in first_actions) / len(pairs),
        "premature_initial_stop_rate": sum(action == "stop" for action in first_actions) / len(pairs),
        "decision_latency_mean_s": float(np.mean(decision_latencies_s)),
        "decision_latency_p95_s": percentile(decision_latencies_s, 95),
        "episode_decision_latency_mean_s": float(np.mean(episode_decision_s)),
        "episode_decision_latency_p95_s": percentile(episode_decision_s, 95),
        "episode_service_latency_mean_s": float(np.mean(episode_service_s)),
        "episode_service_latency_p95_s": percentile(episode_service_s, 95),
        "episode_end_to_end_latency_mean_s": float(np.mean(episode_end_to_end_s)),
        "episode_end_to_end_latency_p95_s": percentile(episode_end_to_end_s, 95),
        "valid_action_rate": sum(step["valid"] for step in steps) / len(steps),
        "invalid_action_attempts": len(invalid_steps),
        "gateway_rejections": len(invalid_steps),
        "retry_decisions_after_rejection": nonterminal_gateway_rejections,
        "episodes_with_invalid_action": sum(any(not step["valid"] for step in trace["controller_steps"]) for trace in traces),
        "stopped_episode_rate": stopped_cases / len(pairs),
        "invalid_action_limit_cases": invalid_limit_cases,
        "budget_exceeded_cases": budget_exceeded_cases,
        "service_failure_cases": sum(trace["status"] not in {"stopped", "invalid_action_limit", "budget_exceeded"} for trace in traces),
    }


def render_markdown(rows: list[dict]) -> str:
    """Render the three paper-ready views without hiding the raw JSON output."""
    performance = [
        "# Balanced Cross-Model Controller Screen",
        "",
        "## Predictive and cost outcomes",
        "",
        "| Model | Cost saving (mean ± SD) | PR-AUC | ΔPR-AUC vs full evidence | Recall change at fixed threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    operations = [
        "",
        "## Latency and acquisition outcomes",
        "",
        "| Model | Mean acquisition cost | Stopping depth (mean ± SD) | Decision latency mean / P95 (s) | Episode end-to-end latency mean / P95 (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    reliability = [
        "",
        "## Execution reliability outcomes",
        "",
        "| Model | Valid actions | Gateway rejections | Retry decisions | Invalid-limit cases | Service-failure cases | Stopped episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        performance.append(
            f"| {row['model']} | {row['cost_saving_mean_pct']:.1f}% ± {row['cost_saving_sd_pct']:.1f}% | "
            f"{row['pr_auc']:.3f} | {row['pr_auc_change']:+.3f} | {row['recall_change_at_fixed_threshold_pp']:+.1f} pp |"
        )
        operations.append(
            f"| {row['model']} | {row['mean_acquisition_cost']:.2f} | "
            f"{row['mean_stopping_depth']:.2f} ± {row['stopping_depth_sd']:.2f} | "
            f"{row['decision_latency_mean_s']:.3f} / {row['decision_latency_p95_s']:.3f} | "
            f"{row['episode_end_to_end_latency_mean_s']:.3f} / {row['episode_end_to_end_latency_p95_s']:.3f} |"
        )
        reliability.append(
            f"| {row['model']} | {row['valid_action_rate']:.1%} | {row['gateway_rejections']} | "
            f"{row['retry_decisions_after_rejection']} | {row['invalid_action_limit_cases']} | "
            f"{row['service_failure_cases']} | {row['stopped_episode_rate']:.1%} |"
        )
    adherence = [
        "",
        "## Balanced-prompt adherence",
        "",
        "The prompt requires `acquire_network` at the initial gate; schema-valid `stop` is therefore reported separately from gateway validity.",
        "",
        "| Model | Initial network-acquisition rate | Premature initial-stop rate |",
        "|---|---:|---:|",
    ]
    for row in rows:
        adherence.append(
            f"| {row['model']} | {row['initial_network_acquisition_rate']:.1%} | {row['premature_initial_stop_rate']:.1%} |"
        )
    return "\n".join([*performance, *operations, *reliability, *adherence, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the balanced-only cross-model controller screen.")
    parser.add_argument("reports", nargs="+", type=Path, help="Paired balanced controller reports, one per model")
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, help="Optional paper-ready three-table Markdown view")
    args = parser.parse_args()
    if args.expected_cases < 1:
        raise SystemExit("--expected-cases must be positive")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    case_ids = validate_reports(reports, args.expected_cases)
    labels_by_id = labels_for(case_ids, args.database, args.split_database)
    rows = [summarise(report, labels_by_id) for report in reports]
    output = {
        "purpose": "balanced-only cross-model validation screen; no test access",
        "goal_profile": "balanced",
        "evaluation_threshold": FULL_EVIDENCE_THRESHOLD,
        "cohort_case_ids": case_ids,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
