"""Tests for representative-test checkpoint and resume behaviour."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.run_representative_test import initialise_checkpoint, run_policy


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, case_id, policy):
        self.calls.append(case_id)
        return SimpleNamespace(
            status="stopped",
            risk_trajectory=[{"risk": case_id / 10}],
            controller_steps=[{
                "proposed_action": "stop", "valid": True, "controller_latency_ns": 10,
                "decision_source": "policy", "raw_response": None,
            }],
            tool_calls=[],
            total_cost=0.0,
        )


class RepresentativeTestCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.connection = sqlite3.connect(Path(temporary.name) / "checkpoint.sqlite")
        self.addCleanup(self.connection.close)
        initialise_checkpoint(self.connection)
        self.connection.executemany(
            "INSERT INTO cohort(case_id,label,initial_risk,routed) VALUES (?,?,?,1)",
            [(1, 0, .1), (2, 0, .2), (3, 1, .3)],
        )
        self.connection.commit()

    def test_second_invocation_skips_checkpointed_cases(self) -> None:
        runtime = FakeRuntime()
        status = run_policy(
            self.connection, runtime, "transaction_only", object(), checkpoint_every=1, case_budget=2,
        )
        self.assertEqual(status, 0)
        self.assertEqual(runtime.calls, [1, 2])

        resumed = FakeRuntime()
        status = run_policy(
            self.connection, resumed, "transaction_only", object(), checkpoint_every=1, case_budget=None,
        )
        self.assertEqual(status, 0)
        self.assertEqual(resumed.calls, [3])
        rows = list(self.connection.execute(
            "SELECT case_id,action_sequence FROM results WHERE policy='transaction_only' ORDER BY case_id"
        ))
        self.assertEqual([row[0] for row in rows], [1, 2, 3])
        self.assertTrue(all(json.loads(row[1]) == ["stop"] for row in rows))


if __name__ == "__main__":
    unittest.main()
