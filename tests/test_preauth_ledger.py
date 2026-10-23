"""Tests for counterfactual pre-authorisation impact bands."""

from __future__ import annotations

import unittest

from src.preauth_ledger import features_from_row, origin_impact_profile


class PreauthorisationLedgerTests(unittest.TestCase):
    def test_near_full_use_is_a_broad_band_not_an_exact_equality_feature(self) -> None:
        for scheme in ("fine", "operational", "coarse"):
            exact = origin_impact_profile("TRANSFER", 100.0, 100.0, scheme)
            nearby = origin_impact_profile("TRANSFER", 100.0, 110.0, scheme)
            self.assertEqual(exact["origin_burden_band"], nearby["origin_burden_band"])
        # Operational/coarse modes also deliberately merge exact depletion and
        # a positive residual, preventing a hidden equality boundary.
        self.assertEqual(
            origin_impact_profile("TRANSFER", 100.0, 100.0, "operational")["origin_residual_band"],
            origin_impact_profile("TRANSFER", 100.0, 110.0, "operational")["origin_residual_band"],
        )

    def test_features_read_only_pre_transaction_fields(self) -> None:
        row = {"type": "TRANSFER", "amount": 100.0, "oldbalance_org": 120.0, "oldbalance_dest": 50.0,
               "newbalance_orig": 0.0, "newbalance_dest": 999999.0}
        before = features_from_row(row, ("origin_impact", "destination_impact", "combined_impact"))
        row["newbalance_orig"] = 888888.0
        row["newbalance_dest"] = 0.0
        self.assertEqual(before, features_from_row(row, ("origin_impact", "destination_impact", "combined_impact")))
