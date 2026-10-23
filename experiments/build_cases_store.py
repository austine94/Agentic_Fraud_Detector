"""CLI for building the private PaySim case store."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cases_store import build_cases_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/PS_20174392719_1491204439457_log.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evidence.sqlite"))
    args = parser.parse_args()
    build_cases_store(args.csv, args.output)


if __name__ == "__main__":
    main()
