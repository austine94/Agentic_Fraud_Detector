"""Access helpers for the frozen IID partition assignment."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")


def attach_assignment(connection: sqlite3.Connection, path: Path) -> None:
    """Attach the immutable assignment database under the stable ``iid`` name."""
    if not path.is_file():
        raise FileNotFoundError(f"IID split assignment not found: {path}")
    connection.execute("ATTACH DATABASE ? AS iid", (str(path),))


def validate_split_name(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown IID split: {split}")
