"""Unit tests for deterministic acquisition-only comparison policies."""

from __future__ import annotations

import unittest

from src.agentic_controller import ControllerRequest
from src.comparison_policies import FullEvidencePolicy, SimpleThresholdCascadePolicy, StructuredRulePolicy, TransactionOnlyPolicy


def request(risk: float, evidence: dict, allowed=("stop", "acquire_network")) -> ControllerRequest:
    return ControllerRequest(evidence, risk, allowed, 11.25)


class ComparisonPoliciesTests(unittest.TestCase):
    def test_transaction_only_stops(self) -> None:
        self.assertEqual(TransactionOnlyPolicy().choose(request(.9, {"initial_evidence": {}})).action, "stop")

    def test_full_evidence_takes_only_permitted_next_action(self) -> None:
        self.assertEqual(FullEvidencePolicy().choose(request(.1, {"initial_evidence": {}})).action, "acquire_network")
        decision = FullEvidencePolicy().choose(
            request(.1, {"initial_evidence": {}, "destination_network": {}}, ("stop", "acquire_origin"))
        )
        self.assertEqual(decision.action, "acquire_origin")

    def test_simple_cascade_uses_one_fixed_band(self) -> None:
        policy = SimpleThresholdCascadePolicy(.2, .8)
        self.assertEqual(policy.choose(request(.5, {"initial_evidence": {}})).action, "acquire_network")
        self.assertEqual(policy.choose(request(.9, {"initial_evidence": {}})).action, "stop")

    def test_structured_policy_uses_state_specific_bands(self) -> None:
        policy = StructuredRulePolicy({"initial": (.2, .8), "network": (.4, .6), "origin": (.3, .7)})
        self.assertEqual(policy.choose(request(.3, {"initial_evidence": {}})).action, "acquire_network")
        decision = policy.choose(
            request(.3, {"initial_evidence": {}, "destination_network": {}}, ("stop", "acquire_origin"))
        )
        self.assertEqual(decision.action, "stop")
