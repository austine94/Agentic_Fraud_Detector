"""Formatting tests for the cross-model boundary reliability report."""

from __future__ import annotations

import unittest

from experiments.evaluate_cross_model_boundary_repeatability import render_latex


class CrossModelBoundaryReportTests(unittest.TestCase):
    def test_latex_contains_model_counts_and_row_terminator(self) -> None:
        summary = {
            "episodes": 1250,
            "controller_actions": 3500,
            "valid_controller_actions": 3500,
            "stopped_episode_rate_mean": 1.0,
            "initial_network_adherence_count": 1250,
            "mean_action_sequence_agreement": 1.0,
            "mean_unique_action_sequences": 1.0,
            "action_rejections": 0,
        }
        output = {
            "repeats": 50,
            "cohort": [{} for _ in range(25)],
            "model_results": [{"model": "qwen3:4b", "display_name": "Qwen3:4B", "summary": summary}],
        }
        latex = render_latex(output)
        self.assertIn(r"\textbf{Qwen3:4B} & 3,500/3,500 & 0", latex)
        self.assertIn("1,250/1,250 (100.0\\%) \\\\", latex)


if __name__ == "__main__":
    unittest.main()
