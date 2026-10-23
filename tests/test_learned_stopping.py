"""Tests for cost-sensitive learned stopping helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.agentic_controller import ControllerRequest
from src.learned_stopping import LearnedStoppingPolicy, trajectory_features


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features):
        return np.repeat(self.value, len(features))


class LearnedStoppingTests(unittest.TestCase):
    def test_features_use_only_observed_risks(self) -> None:
        early = trajectory_features([.2, .9, .8, .7], 0)
        changed_future = trajectory_features([.2, .1, .1, .1], 0)
        np.testing.assert_array_equal(early, changed_future)

    def test_runtime_policy_compares_predicted_value_with_frozen_threshold(self) -> None:
        evidence = {"initial_evidence": {"type": "TRANSFER", "amount": 100.0}}
        request = ControllerRequest(evidence, .7, ("stop", "acquire_network"), 11.25)
        continuing = LearnedStoppingPolicy([ConstantRegressor(.2)] * 3, value_threshold=.1)
        stopping = LearnedStoppingPolicy([ConstantRegressor(.05)] * 3, value_threshold=.1)
        self.assertEqual(continuing.choose(request).action, "acquire_network")
        self.assertEqual(stopping.choose(request).action, "stop")


if __name__ == "__main__":
    unittest.main()
