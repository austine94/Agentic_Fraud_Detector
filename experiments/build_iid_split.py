"""Create the frozen stratified 50/10/40 IID PaySim split.

The assignment is held separately from the evidence database, so the source
data and its chronological account-history index remain unchanged.  It is
stratified by the fraud label solely to preserve enough positives in each
partition; labels are never exposed as model features.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


SPLIT_FRACTIONS = (("train", 0.50), ("validation", 0.10), ("test", 0.40))


def allocation(total: int) -> dict[str, int]:
    """Return exact integer counts, assigning rounding remainder to test."""
    train = int(total * 0.50)
    validation = int(total * 0.10)
    return {"train": train, "validation": validation, "test": total - train - validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic stratified random split assignment.")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing frozen assignment: {args.output}")

    source = sqlite3.connect(args.database)
    source.row_factory = sqlite3.Row
    try:
        identifiers = {label: [] for label in (0, 1)}
        for row in source.execute("SELECT case_id, is_fraud FROM cases ORDER BY case_id"):
            identifiers[int(row["is_fraud"])].append(int(row["case_id"]))
    finally:
        source.close()

    output = sqlite3.connect(args.output)
    try:
        output.execute("CREATE TABLE assignments (case_id INTEGER PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'test'))) WITHOUT ROWID")
        output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        rng = np.random.default_rng(args.seed)
        summary: dict[str, dict[str, int]] = {name: {"cases": 0, "fraud_cases": 0} for name, _ in SPLIT_FRACTIONS}
        for label, case_ids in identifiers.items():
            rng.shuffle(case_ids)
            counts = allocation(len(case_ids))
            position = 0
            for split, _ in SPLIT_FRACTIONS:
                selected = case_ids[position:position + counts[split]]
                output.executemany("INSERT INTO assignments(case_id, split) VALUES (?, ?)", ((case_id, split) for case_id in selected))
                summary[split]["cases"] += len(selected)
                summary[split]["fraud_cases"] += len(selected) * label
                position += len(selected)
        metadata = {
            "protocol": "stratified_iid_random_split",
            "seed": args.seed,
            "fractions": {name: fraction for name, fraction in SPLIT_FRACTIONS},
            "summary": summary,
        }
        output.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", ((key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()))
        output.commit()
    finally:
        output.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
