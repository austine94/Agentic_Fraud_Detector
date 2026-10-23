"""Resumable held-out representative-test evaluation.

Nothing in this module reads the test split unless ``--authorize-test`` is
passed. Progress is stored in SQLite, so Ctrl-C is safe and rerunning the same
command resumes at the next unfinished policy/case pair.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.representative_test import (
    ENTRY_GATE,
    SEED,
    TARGET_FRAUD_CASES,
    TARGET_NON_FRAUD_CASES,
    policies,
    training_rows,
)
from src.agentic_controller import AgenticControllerRuntime
from src.evidence_service import EvidenceService
from src.iid_split import attach_assignment
from src.risk_scoring import fit_state_risk_scorers, safe_evidence_from_row


POLICY_NAMES = (
    "qwen3_4b_balanced",
    "transaction_only",
    "full_evidence",
    "aggressive_fixed_band",
    "moderate_structured",
)
EVALUATION_THRESHOLD = 0.9991805471554841


def initialise_checkpoint(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cohort (
             case_id INTEGER PRIMARY KEY, label INTEGER NOT NULL CHECK(label IN (0,1)),
             initial_risk REAL, routed INTEGER CHECK(routed IN (0,1))) WITHOUT ROWID"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS results (
             policy TEXT NOT NULL, case_id INTEGER NOT NULL, status TEXT NOT NULL,
             final_risk REAL NOT NULL, total_cost REAL NOT NULL, wall_seconds REAL NOT NULL,
             controller_seconds REAL NOT NULL, action_sequence TEXT NOT NULL,
             trace_json TEXT NOT NULL, PRIMARY KEY(policy,case_id),
             FOREIGN KEY(case_id) REFERENCES cohort(case_id)) WITHOUT ROWID"""
    )
    connection.commit()


def metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def set_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (key, json.dumps(value, sort_keys=True))
    )


def frozen_contract(model: str) -> dict:
    return {
        "split": "test",
        "fraud_cases": TARGET_FRAUD_CASES,
        "sampled_non_fraud_cases": TARGET_NON_FRAUD_CASES,
        "sampling_seed": SEED,
        "entry_gate": ENTRY_GATE,
        "evaluation_threshold": EVALUATION_THRESHOLD,
        "model": model,
        "goal_profile": "balanced",
        "final_evidence_stop": "runtime_forced",
        "policies": list(POLICY_NAMES),
    }


def ensure_contract(connection: sqlite3.Connection, model: str) -> None:
    expected = frozen_contract(model)
    saved = metadata(connection, "contract")
    if saved is None:
        set_metadata(connection, "contract", expected)
        connection.commit()
    elif json.loads(saved) != expected:
        raise RuntimeError("checkpoint belongs to a different frozen experiment contract")


def build_cohort(input_connection: sqlite3.Connection, checkpoint: sqlite3.Connection) -> None:
    existing = checkpoint.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    if existing:
        counts = dict(checkpoint.execute("SELECT label,COUNT(*) FROM cohort GROUP BY label"))
        if counts != {0: TARGET_NON_FRAUD_CASES, 1: TARGET_FRAUD_CASES}:
            raise RuntimeError(f"incomplete or incompatible saved cohort: {counts}")
        return

    rng = random.Random(SEED)
    positives: list[int] = []
    negatives: list[int] = []
    negative_seen = 0
    cursor = input_connection.execute(
        """SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id)
           WHERE a.split='test' ORDER BY c.case_id"""
    )
    for case_id, label in cursor:
        if label:
            positives.append(int(case_id))
            continue
        negative_seen += 1
        if len(negatives) < TARGET_NON_FRAUD_CASES:
            negatives.append(int(case_id))
        else:
            replacement = rng.randrange(negative_seen)
            if replacement < TARGET_NON_FRAUD_CASES:
                negatives[replacement] = int(case_id)
    if len(positives) != TARGET_FRAUD_CASES or len(negatives) != TARGET_NON_FRAUD_CASES:
        raise RuntimeError(
            f"unexpected test composition: fraud={len(positives)}, sampled_non_fraud={len(negatives)}"
        )
    with checkpoint:
        checkpoint.executemany(
            "INSERT INTO cohort(case_id,label) VALUES (?,?)",
            sorted([(case_id, 1) for case_id in positives] + [(case_id, 0) for case_id in negatives]),
        )
        set_metadata(checkpoint, "test_non_fraud_population", negative_seen)


def fit_scorer(input_connection: sqlite3.Connection):
    train = list(training_rows(input_connection))
    labels = [int(row["is_fraud"]) for row in train]
    positive_weight = (len(labels) - sum(labels)) / sum(labels)
    return fit_state_risk_scorers(
        [safe_evidence_from_row(row) for row in train], labels, seed=SEED,
        sample_weights=[positive_weight if label else 1.0 for label in labels],
    )


def score_initial_batches(
    input_connection: sqlite3.Connection, checkpoint: sqlite3.Connection, scorer, batch_size: int
) -> None:
    missing = [row[0] for row in checkpoint.execute("SELECT case_id FROM cohort WHERE initial_risk IS NULL")]
    if not missing:
        return
    input_connection.execute("CREATE TEMP TABLE selected_cases(case_id INTEGER PRIMARY KEY) WITHOUT ROWID")
    input_connection.executemany("INSERT INTO selected_cases(case_id) VALUES (?)", ((case_id,) for case_id in missing))
    cursor = input_connection.execute(
        """SELECT c.case_id,c.type,c.amount FROM cases c JOIN selected_cases s USING(case_id)
           ORDER BY c.case_id"""
    )
    while batch := cursor.fetchmany(batch_size):
        evidence = [{"initial_evidence": {"type": str(row["type"]), "amount": float(row["amount"])}}
                    for row in batch]
        scores = scorer.predict_risks(evidence)
        with checkpoint:
            checkpoint.executemany(
                "UPDATE cohort SET initial_risk=?,routed=? WHERE case_id=?",
                ((float(score), int(score >= ENTRY_GATE), int(row["case_id"]))
                 for row, score in zip(batch, scores, strict=True)),
            )


def compact_trace(trace, *, retain_raw_response: bool) -> str:
    steps = []
    for step in trace.controller_steps:
        saved = {
            "proposed_action": step["proposed_action"],
            "valid": step["valid"],
            "controller_latency_ns": step["controller_latency_ns"],
            "decision_source": step.get("decision_source", "policy"),
        }
        if retain_raw_response:
            saved["raw_response"] = step.get("raw_response")
        steps.append(saved)
    return json.dumps({
        "status": trace.status,
        "risk_trajectory": trace.risk_trajectory,
        "controller_steps": steps,
        "tool_calls": trace.tool_calls,
    }, separators=(",", ":"))


def run_policy(
    checkpoint: sqlite3.Connection,
    runtime: AgenticControllerRuntime,
    name: str,
    policy,
    *,
    checkpoint_every: int,
    case_budget: int | None,
) -> int:
    completed = {row[0] for row in checkpoint.execute("SELECT case_id FROM results WHERE policy=?", (name,))}
    routed_ids = [row[0] for row in checkpoint.execute("SELECT case_id FROM cohort WHERE routed=1 ORDER BY case_id")]
    remaining = [case_id for case_id in routed_ids if case_id not in completed]
    if case_budget is not None:
        remaining = remaining[:case_budget]
    print(f"{name}: {len(completed)}/{len(routed_ids)} already complete; running {len(remaining)}")
    written = 0
    try:
        for case_id in remaining:
            started = perf_counter()
            trace = runtime.run(case_id, policy)
            wall_seconds = perf_counter() - started
            if trace.status != "stopped":
                checkpoint.commit()
                raise RuntimeError(
                    f"{name} case {case_id} ended with {trace.status}; it was not checkpointed and will retry on resume"
                )
            controller_seconds = sum(step["controller_latency_ns"] for step in trace.controller_steps) / 1e9
            actions = [step["proposed_action"] for step in trace.controller_steps]
            final_risk = float(trace.risk_trajectory[-1]["risk"])
            checkpoint.execute(
                """INSERT OR REPLACE INTO results
                   (policy,case_id,status,final_risk,total_cost,wall_seconds,controller_seconds,
                    action_sequence,trace_json) VALUES (?,?,?,?,?,?,?,?,?)""",
                (name, case_id, trace.status, final_risk, trace.total_cost, wall_seconds,
                 controller_seconds, json.dumps(actions),
                 compact_trace(trace, retain_raw_response=name == "qwen3_4b_balanced")),
            )
            written += 1
            if written % checkpoint_every == 0:
                checkpoint.commit()
            if written % 100 == 0:
                print(f"{name}: saved {len(completed) + written}/{len(routed_ids)}")
    except KeyboardInterrupt:
        checkpoint.commit()
        print(f"\nPaused safely after saving {written} new {name} cases. Rerun the same command to resume.")
        return 130
    checkpoint.commit()
    return 0


def write_report(checkpoint: sqlite3.Connection, output: Path) -> None:
    cohort = list(checkpoint.execute("SELECT case_id,label,initial_risk,routed FROM cohort ORDER BY case_id"))
    ids = np.asarray([row[0] for row in cohort], dtype=np.int64)
    labels = np.asarray([row[1] for row in cohort], dtype=np.int8)
    initial = np.asarray([row[2] for row in cohort], dtype=float)
    routed_count = sum(row[3] for row in cohort)
    negative_population = json.loads(metadata(checkpoint, "test_non_fraud_population") or "null")
    weights = np.where(labels == 1, 1.0, negative_population / TARGET_NON_FRAUD_CASES)
    rows = []
    for name in POLICY_NAMES:
        results = {row[0]: row[1:] for row in checkpoint.execute(
            "SELECT case_id,status,final_risk,total_cost,wall_seconds,controller_seconds FROM results WHERE policy=?",
            (name,),
        )}
        complete = len(results) == routed_count
        row = {"policy": name, "routed_completed": len(results), "routed_expected": routed_count,
               "complete": complete}
        if complete:
            scores = np.asarray([results.get(int(case_id), (None, risk))[1] for case_id, risk in zip(ids, initial)], dtype=float)
            predicted = scores >= EVALUATION_THRESHOLD
            tp = float(weights[(labels == 1) & predicted].sum())
            fp = float(weights[(labels == 0) & predicted].sum())
            fn = float(weights[(labels == 1) & ~predicted].sum())
            tn = float(weights[(labels == 0) & ~predicted].sum())
            row.update({
                "pr_auc": float(average_precision_score(labels, scores, sample_weight=weights)),
                "roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights)),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                "weighted_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "mean_cost_all_cases": sum(value[2] for value in results.values()) / len(cohort),
                "episode_wall_hours": sum(value[3] for value in results.values()) / 3600,
                "controller_hours": sum(value[4] for value in results.values()) / 3600,
                "statuses": dict(Counter(value[0] for value in results.values())),
            })
        rows.append(row)
    report = {
        "purpose": "held-out representative test with population-weighted non-fraud sampling",
        "contract": json.loads(metadata(checkpoint, "contract") or "{}"),
        "cohort_cases": len(cohort),
        "routed_cases": routed_count,
        "non_fraud_population_weight": negative_population / TARGET_NON_FRAUD_CASES,
        "methods": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume the frozen representative held-out test.")
    parser.add_argument("--authorize-test", action="store_true", help="required to open the held-out test split")
    parser.add_argument("--dry-run", action="store_true", help="print the contract without reading any dataset")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--policies", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--case-budget", type=int, help="maximum new routed cases per policy in this invocation")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/test/representative_test.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("results/test/representative_test_report.json"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_every <= 0 or (args.case_budget is not None and args.case_budget <= 0):
        raise SystemExit("batch size, checkpoint interval, and optional case budget must be positive")
    if args.dry_run:
        print(json.dumps({
            "contract": frozen_contract(args.model),
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
            "resume": "rerun the identical command; completed policy/case pairs are skipped",
            "test_access": "not queried",
        }, indent=2))
        return 0
    if not args.authorize_test:
        raise SystemExit("refusing to open the test split without --authorize-test; use --dry-run to inspect safely")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = sqlite3.connect(args.checkpoint)
    checkpoint.execute("PRAGMA foreign_keys=ON")
    initialise_checkpoint(checkpoint)
    ensure_contract(checkpoint, args.model)
    input_connection = sqlite3.connect(args.database)
    input_connection.row_factory = sqlite3.Row
    try:
        attach_assignment(input_connection, args.split_database)
        input_connection.execute("ATTACH DATABASE ? AS behavior", (str(args.profiles),))
        build_cohort(input_connection, checkpoint)
        scorer = fit_scorer(input_connection)
        score_initial_batches(input_connection, checkpoint, scorer, args.batch_size)
    finally:
        input_connection.close()

    runtime = AgenticControllerRuntime(
        EvidenceService(
            args.database, args.profiles, split_database=args.split_database,
            allowed_splits=frozenset({"test"}),
        ),
        scorer, entry_gate=ENTRY_GATE, budget=11.25, auto_stop_at_full_evidence=True,
    )
    available = policies(args.model)
    for name in args.policies:
        status = run_policy(
            checkpoint, runtime, name, available[name], checkpoint_every=args.checkpoint_every,
            case_budget=args.case_budget,
        )
        write_report(checkpoint, args.report)
        if status:
            checkpoint.close()
            return status
    checkpoint.close()
    print(f"Checkpoint: {args.checkpoint}\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
