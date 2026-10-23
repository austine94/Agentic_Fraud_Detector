"""Tests for the constrained controller and policy gateway."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.agentic_controller import AgenticControllerRuntime, ControllerDecision
from src.evidence_service import EvidenceService


class CountScorer:
    def predict_risk(self, evidence): return len(evidence) / 10


class SequencePolicy:
    def __init__(self, actions): self.actions = iter(actions); self.calls = 0
    def choose(self, request): self.calls += 1; return ControllerDecision(next(self.actions))


class AgenticControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); root = Path(temp.name)
        cases, profiles = root / "cases.sqlite", root / "profiles.sqlite"
        db = sqlite3.connect(cases)
        db.execute("""CREATE TABLE cases (case_id INTEGER PRIMARY KEY, step INTEGER, type TEXT, amount REAL, name_orig TEXT,
          oldbalance_org REAL, newbalance_orig REAL, name_dest TEXT, oldbalance_dest REAL, newbalance_dest REAL, is_fraud INTEGER, is_flagged_fraud INTEGER)""")
        db.execute("INSERT INTO cases VALUES (1, 2, 'TRANSFER', 100, 'C1', 120, 20, 'M1', 50, 150, 0, 0)"); db.commit(); db.close()
        db = sqlite3.connect(profiles)
        db.execute("""CREATE TABLE profiles (case_id INTEGER, destination_prior_count INTEGER, destination_prior_total REAL,
          destination_recent_count INTEGER, destination_recent_total REAL, destination_steps_since_last INTEGER, destination_prior_transfer_count INTEGER)""")
        db.execute("INSERT INTO profiles VALUES (1, 0, 0, 0, 0, NULL, 0)")
        db.execute("CREATE TABLE destination_network_history (account TEXT, step INTEGER, prior_unique_origins INTEGER, recent_new_origins INTEGER)")
        db.commit(); db.close(); self.service = EvidenceService(cases, profiles)

    def test_gateway_enforces_path_and_records_complete_episode(self) -> None:
        trace = AgenticControllerRuntime(self.service, CountScorer(), entry_gate=.05, budget=11.25).run(
            1, SequencePolicy(["acquire_network", "acquire_origin", "acquire_destination", "stop"]))
        self.assertEqual(trace.status, "stopped")
        self.assertEqual([step["proposed_action"] for step in trace.controller_steps],
                         ["acquire_network", "acquire_origin", "acquire_destination", "stop"])
        self.assertEqual(trace.total_cost, 11.25)
        self.assertEqual(len(trace.risk_trajectory), 4)

    def test_invalid_model_actions_are_retried_then_halted(self) -> None:
        trace = AgenticControllerRuntime(self.service, CountScorer(), entry_gate=.05, budget=11.25, max_invalid_retries=1).run(
            1, SequencePolicy(["approve", "approve"]))
        self.assertEqual(trace.status, "invalid_action_limit")
        self.assertEqual(len(trace.tool_calls), 1)

    def test_below_gate_case_never_calls_controller_policy(self) -> None:
        trace = AgenticControllerRuntime(self.service, CountScorer(), entry_gate=.2, budget=11.25).run(
            1, SequencePolicy(["acquire_network"]))
        self.assertEqual(trace.status, "below_entry_gate")
        self.assertEqual(trace.controller_steps, [])

    def test_full_evidence_stop_can_be_forced_without_calling_policy(self) -> None:
        policy = SequencePolicy(["acquire_network", "acquire_origin", "acquire_destination"])
        trace = AgenticControllerRuntime(
            self.service, CountScorer(), entry_gate=.05, budget=11.25, auto_stop_at_full_evidence=True
        ).run(1, policy)
        self.assertEqual(trace.status, "stopped")
        self.assertEqual(policy.calls, 3)
        self.assertEqual([step["proposed_action"] for step in trace.controller_steps],
                         ["acquire_network", "acquire_origin", "acquire_destination", "stop"])
        self.assertEqual(trace.controller_steps[-1]["decision_source"], "runtime_forced")
        self.assertEqual(trace.controller_steps[-1]["controller_latency_ns"], 0)
