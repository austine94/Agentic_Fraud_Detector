"""Validation timing pilot for the frozen representative test evaluation.

The final cohort contract is all 3,286 test fraud cases plus a deterministic
sample of 250,000 test non-fraud cases.  This pilot deliberately uses the
validation split to estimate routing and runtime without opening the test set.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.comparison_policies import (
    FullEvidencePolicy,
    SimpleThresholdCascadePolicy,
    StructuredRulePolicy,
    TransactionOnlyPolicy,
)
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


ENTRY_GATE = 0.533566727121687
TARGET_FRAUD_CASES = 3_286
TARGET_NON_FRAUD_CASES = 200_000
SEED = 20260924


def training_rows(connection: sqlite3.Connection):
    return connection.execute(
        """SELECT c.case_id,c.type,c.amount,c.oldbalance_org,c.oldbalance_dest,c.is_fraud,p.*,
          n.prior_unique_origins AS destination_prior_unique_origins,
          n.recent_new_origins AS destination_recent_new_origins
          FROM cases c JOIN behavior.profiles p USING(case_id)
          LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
          JOIN iid.assignments a USING(case_id)
          WHERE a.split='train' AND (c.is_fraud=1 OR c.case_id % 45=0)
          ORDER BY c.case_id"""
    )


def validation_gate_pilot(
    connection: sqlite3.Connection,
    scorer,
    *,
    batch_size: int,
    positive_limit: int,
    negative_limit: int,
) -> tuple[dict[int, int], dict[int, int], list[int], float]:
    """Batch-score validation and reservoir-sample routed cases by label."""
    totals = Counter()
    routed = Counter()
    samples: dict[int, list[int]] = {0: [], 1: []}
    limits = {0: negative_limit, 1: positive_limit}
    rng = random.Random(SEED)
    cursor = connection.execute(
        """SELECT c.case_id,c.type,c.amount,c.is_fraud
           FROM cases c JOIN iid.assignments a USING(case_id)
           WHERE a.split='validation' ORDER BY c.case_id"""
    )
    started = perf_counter()
    while batch := cursor.fetchmany(batch_size):
        evidence = [{"initial_evidence": {"type": str(row["type"]), "amount": float(row["amount"])}}
                    for row in batch]
        scores = scorer.predict_risks(evidence)
        for row, score in zip(batch, scores, strict=True):
            label = int(row["is_fraud"])
            totals[label] += 1
            if score < ENTRY_GATE:
                continue
            routed[label] += 1
            sample = samples[label]
            if len(sample) < limits[label]:
                sample.append(int(row["case_id"]))
            else:
                replacement = rng.randrange(routed[label])
                if replacement < limits[label]:
                    sample[replacement] = int(row["case_id"])
    return dict(totals), dict(routed), sorted(samples[0] + samples[1]), perf_counter() - started


def policies(model: str):
    return {
        "qwen3_4b_balanced": OllamaControllerPolicy(model, goal_profile="balanced"),
        "transaction_only": TransactionOnlyPolicy(),
        "full_evidence": FullEvidencePolicy(),
        "aggressive_fixed_band": SimpleThresholdCascadePolicy(.7, 1.0),
        "moderate_structured": StructuredRulePolicy({
            "initial": (.75, 1.0), "network": (0.0, 1.0), "origin": (0.0, 1.0),
        }),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time a validation-only pilot and project the representative test runtime."
    )
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--pilot-positive-cases", type=int, default=10)
    parser.add_argument("--pilot-negative-cases", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot/representative_test_runtime.json"))
    args = parser.parse_args()
    if args.pilot_positive_cases <= 0 or args.pilot_negative_cases <= 0 or args.batch_size <= 0:
        raise SystemExit("pilot case counts and batch size must be positive")

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        train = list(training_rows(connection))
        labels = [int(row["is_fraud"]) for row in train]
        positive_weight = (len(labels) - sum(labels)) / sum(labels)
        fit_started = perf_counter()
        scorer = fit_state_risk_scorers(
            [safe_evidence_from_row(row) for row in train], labels, seed=SEED,
            sample_weights=[positive_weight if label else 1.0 for label in labels],
        )
        fit_seconds = perf_counter() - fit_started
        totals, routed, case_ids, batch_gate_seconds = validation_gate_pilot(
            connection, scorer, batch_size=args.batch_size,
            positive_limit=args.pilot_positive_cases, negative_limit=args.pilot_negative_cases,
        )
        pilot_labels = dict(connection.execute(
            "SELECT case_id,is_fraud FROM cases WHERE case_id IN (%s)" % ",".join("?" * len(case_ids)),
            case_ids,
        ))
    finally:
        connection.close()

    runtime = AgenticControllerRuntime(
        EvidenceService(args.database, args.profiles, split_database=args.split_database),
        scorer, entry_gate=ENTRY_GATE, budget=11.25, auto_stop_at_full_evidence=True,
    )
    policy_reports = {}
    for name, policy in policies(args.model).items():
        elapsed_by_label: dict[int, list[float]] = {0: [], 1: []}
        controller_by_label: dict[int, list[float]] = {0: [], 1: []}
        statuses = Counter()
        failure_examples = []
        costs = []
        for case_id in case_ids:
            started = perf_counter()
            trace = runtime.run(case_id, policy)
            elapsed_by_label[pilot_labels[case_id]].append(perf_counter() - started)
            controller_by_label[pilot_labels[case_id]].append(
                sum(step["controller_latency_ns"] for step in trace.controller_steps) / 1e9
            )
            statuses[trace.status] += 1
            if trace.status not in {"stopped", "below_entry_gate"} and len(failure_examples) < 3:
                last_step = trace.controller_steps[-1] if trace.controller_steps else {}
                failure_examples.append({
                    "case_id": case_id,
                    "status": trace.status,
                    "raw_response": last_step.get("raw_response"),
                })
            costs.append(trace.total_cost)
        means = {label: mean(values) for label, values in elapsed_by_label.items() if values}
        controller_means = {label: mean(values) for label, values in controller_by_label.items() if values}
        projected_routed = {
            label: (TARGET_FRAUD_CASES if label else TARGET_NON_FRAUD_CASES) * routed[label] / totals[label]
            for label in (0, 1)
        }
        projected_seconds = sum(projected_routed[label] * means[label] for label in (0, 1))
        projected_controller_seconds = sum(
            projected_routed[label] * controller_means[label] for label in (0, 1)
        )
        valid_for_projection = not failure_examples
        policy_reports[name] = {
            "pilot_cases": len(case_ids),
            "pilot_cases_by_label": {str(label): len(elapsed_by_label[label]) for label in (0, 1)},
            "statuses": dict(statuses),
            "valid_for_projection": valid_for_projection,
            "failure_examples": failure_examples,
            "mean_episode_wall_seconds_by_label": {str(k): v for k, v in means.items()},
            "mean_controller_seconds_by_label": {str(k): v for k, v in controller_means.items()},
            "mean_evidence_cost": mean(costs),
            "projected_routed_cases": {str(k): v for k, v in projected_routed.items()},
            "projected_episode_wall_hours": projected_seconds / 3600 if valid_for_projection else None,
            "projected_controller_hours": projected_controller_seconds / 3600 if valid_for_projection else None,
        }

    target_rows = TARGET_FRAUD_CASES + TARGET_NON_FRAUD_CASES
    gate_throughput = sum(totals.values()) / batch_gate_seconds
    projected_target_gate_seconds = target_rows / gate_throughput
    projected_all_policies_hours = sum(
        policy["projected_episode_wall_hours"] for policy in policy_reports.values()
        if policy["projected_episode_wall_hours"] is not None
    )
    report = {
        "purpose": "validation-only runtime pilot for the frozen representative test; test split not queried",
        "representative_test_contract": {
            "test_fraud_cases": TARGET_FRAUD_CASES,
            "sampled_test_non_fraud_cases": TARGET_NON_FRAUD_CASES,
            "total_cases": target_rows,
            "negative_sampling": "deterministic reservoir sample with seed 20260924",
            "labels_available_only_for_cohort_construction_and_offline_evaluation": True,
        },
        "execution_contract": {
            "initial_gate": ENTRY_GATE,
            "initial_gate_scoring": f"batched in chunks of {args.batch_size}",
            "final_evidence_stop": "runtime-forced; no model call",
            "model": args.model,
            "goal_profile": "balanced",
        },
        "scorer_fit_seconds": fit_seconds,
        "validation_batch_gate_seconds": batch_gate_seconds,
        "validation_gate_throughput_rows_per_second": gate_throughput,
        "projected_representative_gate_seconds": projected_target_gate_seconds,
        "validation_counts": {"total": totals, "routed": routed},
        "validation_route_rates": {str(label): routed[label] / totals[label] for label in (0, 1)},
        "policies": policy_reports,
        "projected_all_policies_serial_hours": projected_all_policies_hours,
        "projected_end_to_end_serial_hours_including_fit_and_gate": (
            projected_all_policies_hours + (fit_seconds + projected_target_gate_seconds) / 3600
        ),
        "test_access": "not queried",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
