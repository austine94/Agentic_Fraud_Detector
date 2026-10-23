"""Tests for reproducible validation operating-point selection."""

from __future__ import annotations

import unittest

from src.threshold_selection import precision_frontier, select_precision_threshold, select_recall_threshold, select_terminal_thresholds, weighted_binary_metrics


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_lowest_threshold_that_meets_precision_target(self) -> None:
        point = select_precision_threshold([1, 0, 1, 0], [.9, .8, .7, .1], [1, 1, 1, 1], target_precision=.99)
        self.assertEqual(point.threshold, .9)
        self.assertEqual(point.status, "selected")

    def test_nested_thresholds_and_weighted_metrics(self) -> None:
        thresholds, details = select_terminal_thresholds([1, 0, 1, 0], [.9, .8, .7, .1], [1, 20, 1, 20])
        self.assertLessEqual(thresholds.flag, thresholds.escalate)
        self.assertEqual(set(details), {"flag", "escalate"})
        metrics = weighted_binary_metrics([1, 0, 1, 0], [.9, .8, .7, .1], [1, 20, 1, 20], threshold=thresholds.flag)
        self.assertEqual(set(metrics), {"pr_auc_prevalence_adjusted", "roc_auc_prevalence_adjusted", "precision", "recall", "false_positive_rate", "routed_case_rate"})

    def test_frontier_reports_feasible_and_unfeasible_targets(self) -> None:
        frontier = precision_frontier([1, 0, 1, 0], [.9, .8, .7, .1], [1, 1, 1, 1], targets=(.5, .99))
        self.assertEqual(frontier["targets"]["precision_0.5"]["status"], "selected")
        self.assertEqual(frontier["targets"]["precision_0.99"]["threshold"], .9)

    def test_selects_highest_threshold_that_meets_recall_target(self) -> None:
        point = select_recall_threshold([1, 0, 1, 0], [.9, .8, .7, .1], [1, 1, 1, 1], target_recall=1.0)
        self.assertEqual(point.threshold, .7)
        self.assertEqual(point.achieved_precision, 2 / 3)
