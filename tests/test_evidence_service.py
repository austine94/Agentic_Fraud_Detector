"""Tests for the governed local evidence-service adapters."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.evidence_service import EvidenceService, PROHIBITED_OUTPUT_FIELDS


class EvidenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.cases = root / "cases.sqlite"
        self.profiles = root / "profiles.sqlite"
        self.splits = root / "splits.sqlite"
        connection = sqlite3.connect(self.cases)
        connection.execute("""CREATE TABLE cases (
            case_id INTEGER PRIMARY KEY, step INTEGER, type TEXT, amount REAL,
            name_orig TEXT, oldbalance_org REAL, newbalance_orig REAL,
            name_dest TEXT, oldbalance_dest REAL, newbalance_dest REAL,
            is_fraud INTEGER, is_flagged_fraud INTEGER)""")
        connection.execute("INSERT INTO cases VALUES (1, 12, 'TRANSFER', 100, 'C1', 120, 20, 'M1', 50, 150, 1, 0)")
        connection.commit(); connection.close()
        connection = sqlite3.connect(self.profiles)
        connection.execute("""CREATE TABLE profiles (
            case_id INTEGER, destination_prior_count INTEGER, destination_prior_total REAL,
            destination_recent_count INTEGER, destination_recent_total REAL,
            destination_steps_since_last INTEGER, destination_prior_transfer_count INTEGER)""")
        connection.execute("INSERT INTO profiles VALUES (1, 3, 250, 1, 100, 1, 2)")
        connection.execute("CREATE TABLE destination_network_history (account TEXT, step INTEGER, prior_unique_origins INTEGER, recent_new_origins INTEGER)")
        connection.execute("INSERT INTO destination_network_history VALUES ('M1', 12, 2, 1)")
        connection.commit(); connection.close()
        connection = sqlite3.connect(self.splits)
        connection.execute("CREATE TABLE assignments (case_id INTEGER PRIMARY KEY, split TEXT)")
        connection.execute("INSERT INTO assignments VALUES (1, 'validation')")
        connection.commit(); connection.close()
        self.service = EvidenceService(self.cases, self.profiles)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_responses_are_sanitised_and_metered(self) -> None:
        for tool in ("destination_network", "origin_impact", "destination_impact"):
            response = self.service.acquire(1, tool)
            self.assertEqual(response.status, "ok")
            self.assertGreater(response.normalized_cost, 0)
            self.assertGreater(response.data_transfer_units, 0)
            self.assertFalse(set(response.summary).intersection(PROHIBITED_OUTPUT_FIELDS))

    def test_combined_impact_requires_both_ledgers(self) -> None:
        blocked = self.service.acquire(1, "combined_impact")
        self.assertEqual(blocked.status, "precondition_failed")
        self.assertEqual(blocked.normalized_cost, 0)
        allowed = self.service.acquire(1, "combined_impact", frozenset({"origin_impact", "destination_impact"}))
        self.assertEqual(allowed.status, "ok")
        self.assertEqual(set(allowed.summary), {"high_liquidity_pressure", "high_destination_relative_increase"})

    def test_post_transaction_mutation_cannot_change_impact_summary(self) -> None:
        before = self.service.acquire(1, "origin_impact").summary
        connection = sqlite3.connect(self.cases)
        connection.execute("UPDATE cases SET newbalance_orig=999999, newbalance_dest=0 WHERE case_id=1")
        connection.commit(); connection.close()
        self.assertEqual(before, self.service.acquire(1, "origin_impact").summary)

    def test_test_split_is_blocked_when_a_split_guard_is_configured(self) -> None:
        guarded = EvidenceService(self.cases, self.profiles, split_database=self.splits)
        self.assertEqual(guarded.acquire(1, "origin_impact").status, "ok")
        connection = sqlite3.connect(self.splits)
        connection.execute("UPDATE assignments SET split='test' WHERE case_id=1")
        connection.commit(); connection.close()
        response = guarded.acquire(1, "origin_impact")
        self.assertEqual(response.status, "split_not_authorized")
        self.assertEqual(response.summary, {})
