"""Benchmark causal behaviour and counterfactual-impact acquisition layers.

Local SQLite latency measures the prototype implementation.  The accompanying
normalized costs are an explicit operational scenario: a cached behavioural
feature-store lookup costs one unit, while each synchronous live-ledger read
costs five.  They must not be mistaken for production latency measurements.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from src.iid_split import attach_assignment
from src.preauth_ledger import combined_impact_profile, destination_impact_profile, origin_impact_profile


OPERATIONAL_COSTS = {
    "initial_evidence": 0.0,
    "destination_network": 1.0,
    "origin_impact": 5.0,
    "destination_impact": 5.0,
    "combined_impact": 0.25,
}


def attach_profiles(connection: sqlite3.Connection, path: Path) -> None:
    connection.execute("ATTACH DATABASE ? AS behavior", (str(path),))


def sample_case_ids(connection: sqlite3.Connection, per_type: int) -> list[int]:
    ids: list[int] = []
    types = [row[0] for row in connection.execute(
        "SELECT DISTINCT c.type FROM cases c JOIN iid.assignments a USING(case_id) WHERE a.split='validation' ORDER BY c.type"
    )]
    for transaction_type in types:
        candidates = [row[0] for row in connection.execute(
            """SELECT c.case_id FROM cases c JOIN iid.assignments a USING(case_id)
               WHERE a.split='validation' AND c.type=? ORDER BY c.case_id""", (transaction_type,)
        )]
        positions = [round(index * (len(candidates) - 1) / (per_type - 1)) for index in range(per_type)]
        ids.extend(candidates[position] for position in positions)
    return ids


def summary(durations: list[int]) -> dict[str, float | int]:
    values = np.asarray(durations, dtype=np.int64)
    return {"calls": int(values.size), "median_ms": float(median(values) / 1e6),
            "p95_ms": float(np.percentile(values, 95) / 1e6), "mean_ms": float(np.mean(values) / 1e6)}


def timed(durations: dict[str, list[int]], stage: str, operation):
    started = perf_counter_ns(); result = operation(); durations[stage].append(perf_counter_ns() - started); return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--per-type", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/preauth_layer_latency.json"))
    args = parser.parse_args()
    connection = sqlite3.connect(args.database); connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, args.split_database); attach_profiles(connection, args.profiles)
        case_ids = sample_case_ids(connection, args.per_type)
        durations: dict[str, list[int]] = defaultdict(list)
        for case_id in case_ids:
            initial = timed(durations, "initial_evidence", lambda: connection.execute(
                "SELECT type, amount FROM cases WHERE case_id=?", (case_id,)).fetchone())
            network = timed(durations, "destination_network", lambda: connection.execute(
                """SELECT p.destination_prior_count, p.destination_prior_total, p.destination_recent_count,
                          n.prior_unique_origins, n.recent_new_origins
                   FROM cases c JOIN behavior.profiles p USING(case_id)
                   LEFT JOIN behavior.destination_network_history n ON n.account=c.name_dest AND n.step=c.step
                   WHERE c.case_id=?""", (case_id,)).fetchone())
            if initial is None or network is None:
                raise RuntimeError(f"Missing evidence for case {case_id}")
            origin = timed(durations, "origin_impact", lambda: origin_impact_profile(
                initial["type"], initial["amount"], connection.execute(
                    "SELECT oldbalance_org FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]))
            destination = timed(durations, "destination_impact", lambda: destination_impact_profile(
                initial["type"], initial["amount"], connection.execute(
                    "SELECT oldbalance_dest FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]))
            timed(durations, "combined_impact", lambda: combined_impact_profile(origin, destination))
    finally:
        connection.close()
    stages = {name: summary(values) for name, values in durations.items()}
    cumulative = []
    elapsed = cost = 0.0
    for stage in ("initial_evidence", "destination_network", "origin_impact", "destination_impact", "combined_impact"):
        elapsed += float(stages[stage]["median_ms"]); cost += OPERATIONAL_COSTS[stage]
        cumulative.append({"through_stage": stage, "median_local_latency_ms": elapsed, "normalized_operational_cost": cost})
    report = {
        "purpose": "IID-validation prototype acquisition benchmark; labels and observed post-transaction fields excluded.",
        "sampling": "deterministic evenly spaced validation cases by transaction type; labels excluded",
        "total_cases": len(case_ids), "per_type": args.per_type,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "stage_latency": stages, "operational_cost_model": {
            "units": OPERATIONAL_COSTS,
            "meaning": "Predeclared scenario units: cached behaviour lookup=1; each live ledger read=5; local simulation=.25.",
        },
        "cumulative_layers": cumulative,
        "status": "prototype latency is local-only; freeze or sensitivity-test operational units before policy evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
