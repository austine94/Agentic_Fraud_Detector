"""Repeat the frozen Qwen-balanced controller on a fixed strong validation cohort.

The cohort is selected once from an existing routed, paired validation report:
the ten fraud cases with the highest full-evidence score and the ten
non-fraud cases with the lowest.  Labels select the validation-analysis cohort
only; they are never supplied to the controller, prompt, or evidence service.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


def rows(connection: sqlite3.Connection, split: str, modulus: int):
    return connection.execute(
        """SELECT c.case_id,c.type,c.amount,c.oldbalance_org,c.oldbalance_dest,c.is_fraud,p.*,
          n.prior_unique_origins AS destination_prior_unique_origins,n.recent_new_origins AS destination_recent_new_origins
          FROM cases c JOIN behavior.profiles p USING(case_id)
          LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
          JOIN iid.assignments a USING(case_id)
          WHERE a.split=? AND (c.is_fraud=1 OR c.case_id % ?=0) ORDER BY c.case_id""",
        (split, modulus),
    )


def labels_for(case_ids: list[int], database: Path, split_database: Path) -> dict[int, int]:
    placeholders = ",".join("?" for _ in case_ids)
    connection = sqlite3.connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(split_database),))
        result = dict(connection.execute(
            "SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) "
            f"WHERE a.split='validation' AND c.case_id IN ({placeholders})", case_ids
        ))
    finally:
        connection.close()
    if len(result) != len(case_ids):
        raise ValueError("the cohort source contains a non-validation case")
    return result


def strong_cohort(source: dict, labels_by_id: dict[int, int], per_class: int) -> tuple[list[int], dict[int, float]]:
    pairs = source.get("pairs", [])
    scores = {
        int(pair["case_id"]): float(pair["full_evidence"]["risk_trajectory"][-1]["risk"])
        for pair in pairs
    }
    if set(scores) != set(labels_by_id):
        raise ValueError("cohort source pairs and validation labels do not match")
    fraud = sorted((case_id for case_id, label in labels_by_id.items() if label == 1), key=lambda case_id: (-scores[case_id], case_id))
    nonfraud = sorted((case_id for case_id, label in labels_by_id.items() if label == 0), key=lambda case_id: (scores[case_id], case_id))
    if len(fraud) < per_class or len(nonfraud) < per_class:
        raise ValueError("source report does not contain enough cases in each class")
    return sorted(fraud[:per_class] + nonfraud[:per_class]), scores


def sequence(trace: dict) -> tuple[str, ...]:
    return tuple(step["proposed_action"] for step in trace["controller_steps"])


def summarise(observations: list[dict], case_ids: list[int], repeats: int) -> tuple[dict, list[dict]]:
    by_case: dict[int, list[dict]] = {case_id: [] for case_id in case_ids}
    by_repeat: dict[int, list[dict]] = {repeat: [] for repeat in range(repeats)}
    for observation in observations:
        by_case[observation["case_id"]].append(observation)
        by_repeat[observation["repeat"]].append(observation)
    per_case = []
    for case_id, values in by_case.items():
        sequences = [tuple(value["action_sequence"]) for value in values]
        counts = Counter(sequences)
        modal_sequence, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        costs = np.asarray([value["cost"] for value in values])
        risks = np.asarray([value["stopped_risk"] for value in values])
        per_case.append({
            "case_id": case_id,
            "modal_action_sequence": list(modal_sequence),
            "action_sequence_agreement": modal_count / repeats,
            "unique_action_sequences": len(counts),
            "mean_cost": float(costs.mean()),
            "cost_sd": float(costs.std(ddof=1)),
            "mean_stopped_risk": float(risks.mean()),
            "stopped_risk_sd": float(risks.std(ddof=1)),
        })
    rollout_costs = np.asarray([np.mean([v["cost"] for v in values]) for values in by_repeat.values()])
    rollout_risks = np.asarray([np.mean([v["stopped_risk"] for v in values]) for values in by_repeat.values()])
    valid_action_rates = []
    invalid_counts = []
    rejection_counts = []
    retry_counts = []
    stopped_rates = []
    for values in by_repeat.values():
        steps = [step for value in values for step in value["steps"]]
        invalid = sum(not step["valid"] for step in steps)
        invalid_limits = sum(value["status"] == "invalid_action_limit" for value in values)
        valid_action_rates.append(sum(step["valid"] for step in steps) / len(steps))
        invalid_counts.append(invalid)
        rejection_counts.append(invalid)
        retry_counts.append(invalid - invalid_limits)
        stopped_rates.append(sum(value["status"] == "stopped" for value in values) / len(values))
    summary = {
        "rollout_mean_cost": float(rollout_costs.mean()),
        "rollout_mean_cost_sd": float(rollout_costs.std(ddof=1)),
        "rollout_mean_stopped_risk": float(rollout_risks.mean()),
        "rollout_mean_stopped_risk_sd": float(rollout_risks.std(ddof=1)),
        "mean_action_sequence_agreement": float(np.mean([row["action_sequence_agreement"] for row in per_case])),
        "minimum_action_sequence_agreement": float(np.min([row["action_sequence_agreement"] for row in per_case])),
        "mean_unique_action_sequences": float(np.mean([row["unique_action_sequences"] for row in per_case])),
        "mean_per_case_cost_sd": float(np.mean([row["cost_sd"] for row in per_case])),
        "maximum_per_case_cost_sd": float(np.max([row["cost_sd"] for row in per_case])),
        "mean_per_case_stopped_risk_sd": float(np.mean([row["stopped_risk_sd"] for row in per_case])),
        "maximum_per_case_stopped_risk_sd": float(np.max([row["stopped_risk_sd"] for row in per_case])),
        "valid_action_rate_mean": float(np.mean(valid_action_rates)),
        "valid_action_rate_sd": float(np.std(valid_action_rates, ddof=1)),
        "gateway_rejections_mean": float(np.mean(rejection_counts)),
        "gateway_rejections_sd": float(np.std(rejection_counts, ddof=1)),
        "retry_decisions_mean": float(np.mean(retry_counts)),
        "retry_decisions_sd": float(np.std(retry_counts, ddof=1)),
        "stopped_episode_rate_mean": float(np.mean(stopped_rates)),
        "stopped_episode_rate_sd": float(np.std(stopped_rates, ddof=1)),
    }
    return summary, sorted(per_case, key=lambda row: row["case_id"])


def render_markdown(output: dict) -> str:
    metrics = output["summary"]
    lines = [
        "# Qwen3:4B Balanced Repeatability Check",
        "",
        f"{output['repeats']} repeated rollouts on {len(output['cohort'])} fixed strong validation cases; seed, prompt, schema, scorer, budget, and decoding are held fixed.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean action-sequence agreement | {metrics['mean_action_sequence_agreement']:.1%} |",
        f"| Minimum action-sequence agreement | {metrics['minimum_action_sequence_agreement']:.1%} |",
        f"| Mean unique sequences per case | {metrics['mean_unique_action_sequences']:.2f} |",
        f"| Cohort mean cost across rollouts | {metrics['rollout_mean_cost']:.3f} ± {metrics['rollout_mean_cost_sd']:.3f} |",
        f"| Cohort mean stopped risk across rollouts | {metrics['rollout_mean_stopped_risk']:.5f} ± {metrics['rollout_mean_stopped_risk_sd']:.5f} |",
        f"| Mean per-case cost SD | {metrics['mean_per_case_cost_sd']:.3f} |",
        f"| Mean per-case stopped-risk SD | {metrics['mean_per_case_stopped_risk_sd']:.5f} |",
        f"| Valid action rate | {metrics['valid_action_rate_mean']:.1%} ± {metrics['valid_action_rate_sd']:.1%} |",
        f"| Gateway rejections per rollout | {metrics['gateway_rejections_mean']:.2f} ± {metrics['gateway_rejections_sd']:.2f} |",
        f"| Retry decisions per rollout | {metrics['retry_decisions_mean']:.2f} ± {metrics['retry_decisions_sd']:.2f} |",
        f"| Stopped-episode rate | {metrics['stopped_episode_rate_mean']:.1%} ± {metrics['stopped_episode_rate_sd']:.1%} |",
        "",
        "| Case | Modal action sequence | Agreement | Unique sequences | Cost mean ± SD | Stopped risk mean ± SD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in output["per_case"]:
        lines.append(
            f"| {row['case_id']} | {' → '.join(row['modal_action_sequence'])} | "
            f"{row['action_sequence_agreement']:.1%} | {row['unique_action_sequences']} | "
            f"{row['mean_cost']:.3f} ± {row['cost_sd']:.3f} | {row['mean_stopped_risk']:.5f} ± {row['stopped_risk_sd']:.5f} |"
        )
    return "\n".join([*lines, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the frozen Qwen-balanced controller on a fixed validation cohort.")
    parser.add_argument("--cohort-source", type=Path, default=Path("results/development/qwen3_4b_balanced_500_stratified.json"))
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cases-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924, help="Held fixed across every rollout")
    parser.add_argument("--entry-gate", type=float, default=.533566727121687)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.cases_per_class < 1:
        raise SystemExit("--repeats must be at least 2 and --cases-per-class must be positive")
    source = json.loads(args.cohort_source.read_text(encoding="utf-8"))
    source_ids = [int(pair["case_id"]) for pair in source.get("pairs", [])]
    labels_by_id = labels_for(source_ids, args.database, args.split_database)
    case_ids, reference_scores = strong_cohort(source, labels_by_id, args.cases_per_class)

    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(rows(connection, "train", 45))
    finally:
        connection.close()
    labels = [int(row["is_fraud"]) for row in train]
    positive_weight = (len(labels) - sum(labels)) / sum(labels)
    scorer = fit_state_risk_scorers([safe_evidence_from_row(row) for row in train], labels, seed=20260924,
                                    sample_weights=[positive_weight if label else 1.0 for label in labels])
    runtime = AgenticControllerRuntime(EvidenceService(args.database, args.profiles, split_database=args.split_database), scorer,
                                       entry_gate=args.entry_gate, budget=11.25)
    observations = []
    for repeat in range(args.repeats):
        policy = OllamaControllerPolicy(args.model, goal_profile="balanced", seed=args.seed)
        for case_id in case_ids:
            trace = runtime.run(case_id, policy).as_dict()
            observations.append({
                "repeat": repeat,
                "case_id": case_id,
                "status": trace["status"],
                "action_sequence": list(sequence(trace)),
                "cost": trace["total_cost"],
                "stopped_risk": trace["risk_trajectory"][-1]["risk"],
                "reference_full_evidence_risk": reference_scores[case_id],
                "steps": [{"proposed_action": step["proposed_action"], "valid": step["valid"]} for step in trace["controller_steps"]],
            })
    summary, per_case = summarise(observations, case_ids, args.repeats)
    output = {
        "purpose": "fixed-seed Qwen-balanced repeatability check on a strong validation cohort; no test access",
        "model": args.model,
        "goal_profile": "balanced",
        "seed": args.seed,
        "repeats": args.repeats,
        "cohort_selection": "ten highest-confidence fraud and ten lowest-confidence non-fraud cases from the routed source report",
        "cohort": [{"case_id": case_id, "label": labels_by_id[case_id], "reference_full_evidence_risk": reference_scores[case_id]} for case_id in case_ids],
        "summary": summary,
        "per_case": per_case,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
