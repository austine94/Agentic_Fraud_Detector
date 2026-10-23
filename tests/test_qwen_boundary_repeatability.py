"""Unit tests for empirical action-boundary cohort selection."""

from __future__ import annotations

import unittest

from experiments.evaluate_qwen_boundary_repeatability import boundary_candidates, render_latex, select_boundary_cohort


def step(risk: float, action: str, acquisition_action: str) -> dict:
    return {
        "request": {"risk": risk, "allowed_actions": ["stop", acquisition_action]},
        "proposed_action": action,
        "valid": True,
    }


def source_report() -> dict:
    pairs = []
    for case_id in range(1, 41):
        steps = []
        if case_id <= 20:
            action = "stop" if case_id % 2 else "acquire_origin"
            steps.append(step(0.40 + case_id / 1000, action, "acquire_origin"))
        else:
            action = "stop" if case_id % 2 else "acquire_destination"
            steps.append(step(0.60 + case_id / 1000, action, "acquire_destination"))
        pairs.append({"case_id": case_id, "controller": {"controller_steps": steps}})
    return {"pairs": pairs}


class BoundarySelectionTests(unittest.TestCase):
    def test_candidates_have_opposite_action_neighbours(self) -> None:
        candidates = boundary_candidates(source_report())
        self.assertEqual(len(candidates), 40)
        self.assertTrue(all(row["boundary_distance"] > 0 for row in candidates))
        self.assertTrue(all(row["case_id"] != row["nearest_opposite_case_id"] for row in candidates))

    def test_selection_is_unique_and_covers_every_stratum(self) -> None:
        selected = select_boundary_cohort(source_report(), 25)
        self.assertEqual(len(selected), 25)
        self.assertEqual(len({row["case_id"] for row in selected}), 25)
        strata = {(row["gate"], row["action"]) for row in selected}
        self.assertEqual(strata, {
            ("origin_gate", "stop"),
            ("origin_gate", "acquire_origin"),
            ("destination_gate", "stop"),
            ("destination_gate", "acquire_destination"),
        })

    def test_missing_action_side_is_rejected(self) -> None:
        source = source_report()
        for pair in source["pairs"]:
            for controller_step in pair["controller"]["controller_steps"]:
                if "acquire_origin" in controller_step["request"]["allowed_actions"]:
                    controller_step["proposed_action"] = "stop"
        with self.assertRaisesRegex(ValueError, "both stop and acquire_origin"):
            boundary_candidates(source)

    def test_latex_rows_have_double_backslash_terminators(self) -> None:
        summary = {
            "episodes": 8,
            "valid_controller_actions": 24,
            "controller_actions": 24,
            "stopped_episode_rate_mean": 1.0,
            "initial_network_adherence_count": 8,
            "action_rejections": 0,
            "retry_decisions": 0,
            "service_failure_cases": 0,
            "mean_action_sequence_agreement": 1.0,
            "minimum_action_sequence_agreement": 1.0,
            "mean_unique_action_sequences": 1.0,
            "mean_per_case_cost_sd": 0.0,
            "mean_per_case_stopped_risk_sd": 0.0,
        }
        latex = render_latex({"summary": summary, "cohort": [{}, {}, {}, {}], "repeats": 2})
        self.assertIn(r"Repeated episodes & 8 & -- \\", latex)
        self.assertIn(r"Valid controller actions & 24/24 & 100.0\% \\", latex)


if __name__ == "__main__":
    unittest.main()
