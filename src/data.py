"""Raw PaySim validation utilities for the pre-authorisation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = frozenset(
    {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
)


@dataclass(frozen=True)
class DatasetSummary:
    """Audit information needed before split boundaries are frozen."""

    rows: int
    fraud_rows: int
    fraud_rate: float
    min_step: int
    max_step: int
    transaction_types: dict[str, int]
    unique_origin_accounts: int
    unique_destination_accounts: int
    sha256: str


def file_sha256(path: Path) -> str:
    """Return the digest of the unmodified raw CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_summarize(path: Path) -> DatasetSummary:
    """Validate required PaySim fields and return a lightweight audit summary."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Dataset has no header row.")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        rows = fraud_rows = 0
        min_step: int | None = None
        max_step: int | None = None
        types: Counter[str] = Counter()
        origins: set[str] = set()
        destinations: set[str] = set()

        for case_id, row in enumerate(reader):
            try:
                step = int(row["step"])
                fraud = int(row["isFraud"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid step or isFraud value at input row {case_id + 2}") from error
            if fraud not in (0, 1):
                raise ValueError(f"isFraud must be binary at input row {case_id + 2}")

            rows += 1
            fraud_rows += fraud
            min_step = step if min_step is None else min(min_step, step)
            max_step = step if max_step is None else max(max_step, step)
            types[row["type"]] += 1
            origins.add(row["nameOrig"])
            destinations.add(row["nameDest"])

    if rows == 0:
        raise ValueError("Dataset contains no transaction rows.")
    assert min_step is not None and max_step is not None
    return DatasetSummary(
        rows=rows,
        fraud_rows=fraud_rows,
        fraud_rate=fraud_rows / rows,
        min_step=min_step,
        max_step=max_step,
        transaction_types=dict(sorted(types.items())),
        unique_origin_accounts=len(origins),
        unique_destination_accounts=len(destinations),
        sha256=file_sha256(path),
    )


def chronological_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Add stable IDs and sort for storage; profiles must exclude same-step rows."""
    numbered = [dict(row, case_id=str(case_id)) for case_id, row in enumerate(rows)]
    return sorted(numbered, key=lambda row: (int(row["step"]), int(row["case_id"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a raw PaySim CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to the unmodified PaySim CSV")
    args = parser.parse_args()
    print(json.dumps(asdict(validate_and_summarize(args.csv_path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
