"""Tests for the strict pre-authorisation feature contract."""

from __future__ import annotations

import unittest

from src.preauth_behavior import TOOL_NAMES, feature_vector, initial_features


ROW = {
    "type": "CASH_OUT", "amount": 100.0,
    "origin_prior_count": 4, "origin_prior_total": 200.0, "origin_prior_sum_squared": 15000.0,
    "origin_prior_transfer_count": 2, "origin_prior_cashout_count": 1, "origin_recent_count": 2,
    "origin_recent_total": 150.0, "origin_steps_since_last": 2,
    "pair_prior_count": 0, "pair_prior_total": 0.0, "pair_steps_since_last": None,
    "destination_prior_count": 3, "destination_prior_total": 250.0, "destination_prior_transfer_count": 2,
    "destination_recent_count": 1, "destination_recent_total": 100.0, "destination_steps_since_last": 1,
    "destination_prior_unique_origins": 2, "destination_recent_new_origins": 1,
    "origin_incoming_prior_count": 2, "origin_incoming_recent_count": 1, "origin_incoming_recent_total": 120.0,
    "origin_steps_since_incoming": 1, "origin_steps_since_transfer_incoming": 1, "origin_last_transfer_incoming_total": 100.0,
}


class PreauthorisationFeaturesTests(unittest.TestCase):
    def test_initial_evidence_has_no_balance_or_time_field(self) -> None:
        self.assertEqual(len(initial_features(ROW)), 6)

    def test_each_tool_adds_only_its_declared_summary(self) -> None:
        base = len(feature_vector(ROW, ()))
        self.assertEqual(len(feature_vector(ROW, ("origin_velocity",))), base + 11)
        self.assertEqual(len(feature_vector(ROW, ("relationship",))), base + 4)
        self.assertEqual(len(feature_vector(ROW, ("destination_network",))), base + 9)
        self.assertEqual(len(feature_vector(ROW, ("flow_sequence",))), base + 8)
        self.assertGreater(len(feature_vector(ROW, TOOL_NAMES)), base)

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feature_vector(ROW, ("post_transaction_balance",))
