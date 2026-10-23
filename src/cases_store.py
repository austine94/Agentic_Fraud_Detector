"""Build the private raw-case store used by pre-authorisation tools."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


def build_cases_store(csv_path: Path, database_path: Path, batch_size: int = 20_000) -> None:
    """Ingest the verified raw CSV without deriving features or exposing labels."""
    if database_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing case store: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript("""
        CREATE TABLE cases (
          case_id INTEGER PRIMARY KEY, step INTEGER NOT NULL, type TEXT NOT NULL,
          amount REAL NOT NULL, name_orig TEXT NOT NULL, oldbalance_org REAL NOT NULL,
          newbalance_orig REAL NOT NULL, name_dest TEXT NOT NULL, oldbalance_dest REAL NOT NULL,
          newbalance_dest REAL NOT NULL, is_fraud INTEGER NOT NULL CHECK(is_fraud IN (0,1)),
          is_flagged_fraud INTEGER NOT NULL CHECK(is_flagged_fraud IN (0,1))
        );
        """)
        insert = "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        batch = []
        with csv_path.open("r", encoding="utf-8", newline="") as source:
            for case_id, row in enumerate(csv.DictReader(source)):
                batch.append((case_id, int(row["step"]), row["type"], float(row["amount"]), row["nameOrig"],
                              float(row["oldbalanceOrg"]), float(row["newbalanceOrig"]), row["nameDest"],
                              float(row["oldbalanceDest"]), float(row["newbalanceDest"]), int(row["isFraud"]),
                              int(row["isFlaggedFraud"])))
                if len(batch) == batch_size:
                    connection.executemany(insert, batch); batch.clear()
            if batch:
                connection.executemany(insert, batch)
        connection.executescript("""
        CREATE INDEX cases_origin_chronology ON cases(name_orig, step, case_id);
        CREATE INDEX cases_destination_chronology ON cases(name_dest, step, case_id);
        CREATE INDEX cases_pair_chronology ON cases(name_orig, name_dest, step, case_id);
        """)
        connection.commit()
    except Exception:
        connection.rollback(); raise
    finally:
        connection.close()
