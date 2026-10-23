"""Select deterministic acquisition bands from validation full-evidence traces.

The script replays already-computed safe risk trajectories. Labels are joined
only for validation policy selection; no labels enter a controller episode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np

from experiments.summarize_controller_scenarios import FULL_EVIDENCE_THRESHOLD, pr


def metrics(labels: np.ndarray, stopped: np.ndarray, full: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    precision, recall = pr(labels, stopped, FULL_EVIDENCE_THRESHOLD)
    full_precision, full_recall = pr(labels, full, FULL_EVIDENCE_THRESHOLD)
    return {
        "cost_saving_pct": float(np.mean(1 - costs / 11.25) * 100),
        "precision_change_pp": (precision - full_precision) * 100,
        "recall_change_pp": (recall - full_recall) * 100,
        "mean_stopping_depth": float(np.mean(np.select([costs == 0, costs == 1, costs == 6], [0, 1, 2], default=3))),
    }


def replay(risks: np.ndarray, bands: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Replay a one-band-per-state stop/continue policy over 4 risk states."""
    depth = np.zeros(len(risks), dtype=np.int8)
    active = np.ones(len(risks), dtype=bool)
    for stage, (low, high) in enumerate(bands):
        advance = active & (risks[:, stage] >= low) & (risks[:, stage] <= high)
        depth[advance] = stage + 1
        active = advance
    stopped = risks[np.arange(len(risks)), depth]
    costs = np.choose(depth, [0.0, 1.0, 6.0, 11.25])
    return stopped, costs


def objective(result: dict[str, float], max_recall_loss: float, target_cost_saving: float | None) -> tuple[int, float, float, float]:
    feasible = result["recall_change_pp"] >= -max_recall_loss and result["precision_change_pp"] >= 0
    # Among feasible candidates maximise savings; otherwise retain performance.
    if feasible:
        if target_cost_saving is not None:
            return (1, -abs(result["cost_saving_pct"] - target_cost_saving), result["recall_change_pp"], result["precision_change_pp"])
        return (1, result["cost_saving_pct"], result["recall_change_pp"], result["precision_change_pp"])
    return (0, result["recall_change_pp"], result["precision_change_pp"], result["cost_saving_pct"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only band selection for deterministic comparison policies.")
    parser.add_argument("--full-report", type=Path, default=Path("results/development/comparison_full_evidence_500_stratified.json"))
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite")); parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--max-recall-loss-pp", type=float, default=1.0)
    parser.add_argument("--target-cost-saving-pct", type=float,
                        help="select the feasible policy nearest this saving level instead of the most economical one")
    parser.add_argument("--grid-step", type=float, default=.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.grid_step <= 1 or args.max_recall_loss_pp < 0 or (args.target_cost_saving_pct is not None and not 0 <= args.target_cost_saving_pct <= 100):
        raise SystemExit("grid step must be in (0,1] and recall loss must be non-negative")

    report = json.loads(args.full_report.read_text(encoding="utf-8")); pairs = report["pairs"]
    ids = [pair["case_id"] for pair in pairs]
    connection = sqlite3.connect(args.database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(args.split_database),))
        labels_by_id = dict(connection.execute("SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) WHERE a.split='validation' AND c.case_id IN (%s)" % ",".join("?" * len(ids)), ids))
    finally:
        connection.close()
    labels = np.asarray([labels_by_id[case_id] for case_id in ids], dtype=np.int8)
    risks = np.asarray([[state["risk"] for state in pair["controller"]["risk_trajectory"]] for pair in pairs], dtype=float)
    full = risks[:, -1]
    values = np.unique(np.append(np.arange(0, 1 + args.grid_step / 2, args.grid_step), 1.0))
    bands = [(float(low), float(high)) for low, high in itertools.combinations_with_replacement(values, 2)]

    simple = []
    for band in bands:
        stopped, costs = replay(risks, (band, band, band))
        result = metrics(labels, stopped, full, costs)
        simple.append({"band": band, **result})
    chosen_simple = max(simple, key=lambda item: objective(item, args.max_recall_loss_pp, args.target_cost_saving_pct))

    # Coordinate ascent starts at the full-evidence policy and makes each state
    # independently as economical as possible while retaining the global target.
    structured = [(0.0, 1.0)] * 3
    for _ in range(4):
        changed = False
        for stage in range(3):
            candidates = []
            for band in bands:
                proposal = tuple(band if index == stage else structured[index] for index in range(3))
                stopped, costs = replay(risks, proposal)
                candidates.append({"bands": proposal, **metrics(labels, stopped, full, costs)})
            best = max(candidates, key=lambda item: objective(item, args.max_recall_loss_pp, args.target_cost_saving_pct))
            if best["bands"] != tuple(structured):
                structured = list(best["bands"]); changed = True
        if not changed:
            break
    stopped, costs = replay(risks, tuple(structured)); structured_result = {"bands": tuple(structured), **metrics(labels, stopped, full, costs)}
    output = {
        "purpose": "validation-only deterministic-policy selection from safe full-evidence risk trajectories",
        "selection_constraint": {"max_recall_loss_pp": args.max_recall_loss_pp, "minimum_precision_change_pp": 0.0,
                                 "target_cost_saving_pct": args.target_cost_saving_pct},
        "operating_threshold": FULL_EVIDENCE_THRESHOLD, "grid_step": args.grid_step,
        "simple_cascade": chosen_simple, "structured_rule": structured_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
