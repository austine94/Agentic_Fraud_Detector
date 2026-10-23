"""Build the strict-prior behavioural store in restartable SQLite stages."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def attach(connection: sqlite3.Connection, source: Path) -> None:
    escaped = str(source.resolve()).replace("'", "''")
    connection.execute(f"ATTACH DATABASE '{escaped}' AS source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("init", "origin", "destination", "pair", "network", "transfer", "join", "verify"))
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/preauth_behavior.sqlite"))
    args = parser.parse_args()
    if args.stage == "init":
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        connection = sqlite3.connect(args.output)
        connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=FILE;")
        connection.close()
        print("initialised")
        return
    connection = sqlite3.connect(args.output)
    try:
        attach(connection, args.database)
        if args.stage == "origin":
            connection.executescript("""
            CREATE TABLE origin_history AS
            WITH origin_step AS (
              SELECT name_orig AS account, step, COUNT(*) AS n, SUM(amount) AS total, SUM(amount*amount) AS sum_squared,
                     SUM(type='TRANSFER') AS transfer_n, SUM(type='CASH_OUT') AS cashout_n
              FROM source.cases GROUP BY name_orig, step
            )
            SELECT account, step,
              COALESCE(SUM(n) OVER prior,0) AS prior_count, COALESCE(SUM(total) OVER prior,0) AS prior_total,
              COALESCE(SUM(sum_squared) OVER prior,0) AS prior_sum_squared,
              COALESCE(SUM(transfer_n) OVER prior,0) AS prior_transfer_count,
              COALESCE(SUM(cashout_n) OVER prior,0) AS prior_cashout_count, MAX(step) OVER prior AS last_step,
              COALESCE(SUM(n) OVER recent,0) AS recent_count, COALESCE(SUM(total) OVER recent,0) AS recent_total
            FROM origin_step
            WINDOW prior AS (PARTITION BY account ORDER BY step ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   recent AS (PARTITION BY account ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING);
            CREATE UNIQUE INDEX origin_history_key ON origin_history(account, step);
            """)
        elif args.stage == "destination":
            connection.executescript("""
            CREATE TABLE destination_history AS
            WITH destination_step AS (
              SELECT name_dest AS account, step, COUNT(*) AS n, SUM(amount) AS total, SUM(type='TRANSFER') AS transfer_n
              FROM source.cases GROUP BY name_dest, step
            )
            SELECT account, step,
              COALESCE(SUM(n) OVER prior,0) AS prior_count, COALESCE(SUM(total) OVER prior,0) AS prior_total,
              COALESCE(SUM(transfer_n) OVER prior,0) AS prior_transfer_count, MAX(step) OVER prior AS last_step,
              MAX(CASE WHEN transfer_n > 0 THEN step END) OVER prior AS last_transfer_step,
              COALESCE(SUM(n) OVER recent,0) AS recent_count, COALESCE(SUM(total) OVER recent,0) AS recent_total
            FROM destination_step
            WINDOW prior AS (PARTITION BY account ORDER BY step ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   recent AS (PARTITION BY account ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING);
            CREATE UNIQUE INDEX destination_history_key ON destination_history(account, step);
            """)
        elif args.stage == "pair":
            connection.executescript("""
            CREATE TABLE pair_history AS
            WITH pair_step AS (
              SELECT name_orig AS origin, name_dest AS destination, step, COUNT(*) AS n, SUM(amount) AS total
              FROM source.cases GROUP BY name_orig, name_dest, step
            )
            SELECT origin, destination, step, COALESCE(SUM(n) OVER prior,0) AS prior_count,
                   COALESCE(SUM(total) OVER prior,0) AS prior_total, MAX(step) OVER prior AS last_step
            FROM pair_step
            WINDOW prior AS (PARTITION BY origin, destination ORDER BY step ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
            CREATE UNIQUE INDEX pair_history_key ON pair_history(origin, destination, step);
            """)
        elif args.stage == "network":
            connection.executescript("""
            CREATE TABLE destination_network_history AS
            WITH first_pair AS (
              SELECT name_dest AS account, name_orig AS origin, MIN(step) AS first_step
              FROM source.cases GROUP BY name_dest, name_orig
            ), new_origin_step AS (
              SELECT account, first_step AS step, COUNT(*) AS n FROM first_pair GROUP BY account, first_step
            )
            SELECT account, step, COALESCE(SUM(n) OVER prior,0) AS prior_unique_origins,
                   COALESCE(SUM(n) OVER recent,0) AS recent_new_origins
            FROM new_origin_step
            WINDOW prior AS (PARTITION BY account ORDER BY step ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   recent AS (PARTITION BY account ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING);
            CREATE UNIQUE INDEX destination_network_history_key ON destination_network_history(account, step);
            """)
        elif args.stage == "transfer":
            connection.executescript("""
            CREATE TABLE transfer_incoming_step AS
            SELECT name_dest AS account, step, SUM(amount) AS total
            FROM source.cases WHERE type='TRANSFER' GROUP BY name_dest, step;
            CREATE UNIQUE INDEX transfer_incoming_step_key ON transfer_incoming_step(account, step);
            """)
        elif args.stage == "join":
            connection.executescript("""
            CREATE TABLE profiles AS
            SELECT c.case_id,
              oh.prior_count AS origin_prior_count, oh.prior_total AS origin_prior_total, oh.prior_sum_squared AS origin_prior_sum_squared,
              oh.prior_transfer_count AS origin_prior_transfer_count, oh.prior_cashout_count AS origin_prior_cashout_count,
              oh.recent_count AS origin_recent_count, oh.recent_total AS origin_recent_total,
              CASE WHEN oh.last_step IS NULL THEN NULL ELSE c.step-oh.last_step END AS origin_steps_since_last,
              ph.prior_count AS pair_prior_count, ph.prior_total AS pair_prior_total,
              CASE WHEN ph.last_step IS NULL THEN NULL ELSE c.step-ph.last_step END AS pair_steps_since_last,
              dh.prior_count AS destination_prior_count, dh.prior_total AS destination_prior_total,
              dh.prior_transfer_count AS destination_prior_transfer_count, dh.recent_count AS destination_recent_count,
              dh.recent_total AS destination_recent_total,
              CASE WHEN dh.last_step IS NULL THEN NULL ELSE c.step-dh.last_step END AS destination_steps_since_last,
              ih.prior_count AS origin_incoming_prior_count, ih.recent_count AS origin_incoming_recent_count,
              ih.recent_total AS origin_incoming_recent_total,
              CASE WHEN ih.last_step IS NULL THEN NULL ELSE c.step-ih.last_step END AS origin_steps_since_incoming,
              CASE WHEN ih.last_transfer_step IS NULL THEN NULL ELSE c.step-ih.last_transfer_step END AS origin_steps_since_transfer_incoming
            FROM source.cases c
            LEFT JOIN origin_history oh ON oh.account=c.name_orig AND oh.step=c.step
            LEFT JOIN pair_history ph ON ph.origin=c.name_orig AND ph.destination=c.name_dest AND ph.step=c.step
            LEFT JOIN destination_history dh ON dh.account=c.name_dest AND dh.step=c.step
            LEFT JOIN destination_history ih ON ih.account=c.name_orig AND ih.step=c.step;
            CREATE UNIQUE INDEX profiles_case_id ON profiles(case_id);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES
              ('availability','strictly earlier step only; same-step rows excluded'),
              ('excluded','is_fraud,is_flagged_fraud,current post-transaction balances,account identifiers'),
              ('recent_window_steps','24');
            """)
        elif args.stage == "verify":
            cases = connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
            nulls = connection.execute("SELECT COUNT(*) FROM profiles WHERE origin_prior_count IS NULL").fetchone()[0]
            print(f"profiles={cases} no_origin_history={nulls}")
            return
        connection.commit()
        print(f"completed {args.stage}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
