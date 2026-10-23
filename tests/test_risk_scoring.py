"""Tests for state-specific safe feature construction and scoring."""

from __future__ import annotations

import unittest

import numpy as np

from src.risk_scoring import StateRiskScorer, evidence_state, feature_vector_from_evidence, fit_state_risk_scorers, safe_evidence_from_row


EVIDENCE = {
    "initial_evidence": {"type": "TRANSFER", "amount": 100.0},
    "destination_network": {
        "destination_is_new": 0, "destination_prior_count": 3, "destination_prior_total": 250.0,
        "destination_recent_count": 1, "destination_recent_total": 100.0, "destination_steps_since_last": 1,
        "destination_prior_transfer_count": 2, "destination_prior_unique_origins": 2, "destination_recent_new_origins": 1,
    },
    "origin_impact": {"origin_debit_required": 1, "origin_burden_band": 2, "origin_residual_band": 0, "origin_balance_order": 2},
    "destination_impact": {"destination_credit_expected": 1, "destination_increase_band": 2, "destination_balance_order": 1, "destination_projected_balance_order": 2},
    "combined_impact": {"high_liquidity_pressure": 1, "high_destination_relative_increase": 0},
}


class FixedModel:
    def predict_proba(self, matrix):
        return np.tile(np.array([.2, .8]), (len(matrix), 1))


class RiskScoringTests(unittest.TestCase):
    def test_vectors_expand_only_for_acquired_safe_summaries(self) -> None:
        initial = feature_vector_from_evidence({"initial_evidence": EVIDENCE["initial_evidence"]})
        full = feature_vector_from_evidence(EVIDENCE)
        self.assertEqual(len(initial), 6)
        self.assertEqual(len(full), 25)
        self.assertEqual(evidence_state(EVIDENCE), ("destination_network", "origin_impact", "destination_impact", "combined_impact"))

    def test_scorer_selects_model_by_evidence_state(self) -> None:
        scorer = StateRiskScorer({(): FixedModel()})
        self.assertEqual(scorer.predict_risk({"initial_evidence": EVIDENCE["initial_evidence"]}), .8)
        with self.assertRaises(ValueError):
            scorer.predict_risk(EVIDENCE)

    def test_training_row_is_reduced_to_adapter_equivalent_evidence(self) -> None:
        row = {
            "type": "TRANSFER", "amount": 100.0, "oldbalance_org": 120.0, "oldbalance_dest": 50.0,
            "newbalance_orig": 20.0, "newbalance_dest": 150.0, "is_fraud": 1,
            **EVIDENCE["destination_network"],
        }
        safe = safe_evidence_from_row(row)
        self.assertEqual(safe, EVIDENCE)
        self.assertNotIn("is_fraud", str(safe))
        self.assertNotIn("newbalance", str(safe))

    def test_fitted_models_return_calibrated_probabilities(self) -> None:
        examples = []
        for index in range(8):
            example = {name: dict(summary) for name, summary in EVIDENCE.items()}
            example["initial_evidence"]["amount"] = float(20 + index * 30)
            example["origin_impact"]["origin_burden_band"] = index % 3
            examples.append(example)
        scorer = fit_state_risk_scorers(examples, [0, 0, 0, 0, 1, 1, 1, 1], seed=4)
        risk = scorer.predict_risk(EVIDENCE)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
