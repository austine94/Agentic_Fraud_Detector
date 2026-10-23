"""Tests for the deterministic governed investigation runtime."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.evidence_service import EvidenceService
from src.investigation_runtime import InvestigationRuntime, TerminalThresholds


class EvidenceCountScorer:
    def predict_risk(self, evidence):
        return min(1.0, len(evidence) / 10)


class InvestigationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(); root = Path(self.tempdir.name)
        cases, profiles = root / "cases.sqlite", root / "profiles.sqlite"
        connection = sqlite3.connect(cases)
        connection.execute("""CREATE TABLE cases (case_id INTEGER PRIMARY KEY, step INTEGER, type TEXT, amount REAL,
            name_orig TEXT, oldbalance_org REAL, newbalance_orig REAL, name_dest TEXT, oldbalance_dest REAL,
            newbalance_dest REAL, is_fraud INTEGER, is_flagged_fraud INTEGER)""")
        connection.execute("INSERT INTO cases VALUES (1, 12, 'TRANSFER', 100, 'C1', 120, 20, 'M1', 50, 150, 1, 0)")
        connection.commit(); connection.close()
        connection = sqlite3.connect(profiles)
        connection.execute("""CREATE TABLE profiles (case_id INTEGER, destination_prior_count INTEGER,
            destination_prior_total REAL, destination_recent_count INTEGER, destination_recent_total REAL,
            destination_steps_since_last INTEGER, destination_prior_transfer_count INTEGER)""")
        connection.execute("INSERT INTO profiles VALUES (1, 3, 250, 1, 100, 1, 2)")
        connection.execute("CREATE TABLE destination_network_history (account TEXT, step INTEGER, prior_unique_origins INTEGER, recent_new_origins INTEGER)")
        connection.execute("INSERT INTO destination_network_history VALUES ('M1', 12, 2, 1)")
        connection.commit(); connection.close()
        self.service = EvidenceService(cases, profiles)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_full_evidence_path_is_ordered_metered_and_terminal(self) -> None:
        runtime = InvestigationRuntime(self.service, EvidenceCountScorer(), TerminalThresholds(.3, .5), budget=11.25)
        trace = runtime.run_full_evidence(1)
        self.assertEqual(trace.status, "completed")
        self.assertEqual([call["tool"] for call in trace.tool_calls], [
            "initial_evidence", "destination_network", "origin_impact", "destination_impact", "combined_impact",
        ])
        self.assertEqual(trace.terminal_action, "escalate")
        self.assertEqual(trace.total_cost, 11.25)
        self.assertEqual(len(trace.risk_trajectory), 5)
        self.assertNotIn("newbalance_orig", str(trace.as_dict()))

    def test_gateway_rejects_out_of_order_and_over_budget_actions(self) -> None:
        runtime = InvestigationRuntime(self.service, EvidenceCountScorer(), TerminalThresholds(.3, .5), budget=1.0)
        trace = runtime._start(1)
        rejected = runtime.acquire(trace, "origin_impact")
        self.assertEqual(rejected.status, "action_not_allowed")
        allowed = runtime.acquire(trace, "destination_network")
        self.assertEqual(allowed.status, "ok")
        budgeted = runtime.acquire(trace, "origin_impact")
        self.assertEqual(budgeted.status, "budget_exceeded")

    def test_transaction_only_makes_no_evidence_acquisitions(self) -> None:
        runtime = InvestigationRuntime(self.service, EvidenceCountScorer(), TerminalThresholds(.05, .5), budget=11.25)
        trace = runtime.run_transaction_only(1)
        self.assertEqual(trace.status, "completed")
        self.assertEqual([call["tool"] for call in trace.tool_calls], ["initial_evidence"])
        self.assertEqual(trace.total_cost, 0.0)
        self.assertEqual(trace.terminal_action, "flag")
