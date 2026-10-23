import unittest

from src.agentic_controller import ControllerRequest
from experiments.run_slm_fraud_decision_ablation import (
    categorical_controller_prompt,
    categorical_decision_prompt,
    categorical_metrics,
    classifier_operating_point,
    render_latex,
)


class SlmFraudDecisionAblationTests(unittest.TestCase):
    def test_decision_prompt_requests_only_a_category(self) -> None:
        payload = categorical_decision_prompt({"initial_evidence": {"type": "TRANSFER", "amount": 100.0}})
        rendered = str(payload)
        self.assertIn("TRANSFER", rendered)
        self.assertIn("not_fraud", rendered)
        self.assertNotIn("fraud_probability", rendered)
        self.assertNotIn("is_fraud", rendered)

    def test_controller_receives_category_not_probability(self) -> None:
        request = ControllerRequest({"initial_evidence": {}}, 1.0, ("stop", "acquire_network"), 11.25)
        rendered = str(categorical_controller_prompt(request))
        self.assertIn("fraud_decision", rendered)
        self.assertNotIn("fraud_probability", rendered)
        self.assertNotIn('"risk"', rendered)

    def test_categorical_metrics(self) -> None:
        metrics = categorical_metrics([1, 1, 0, 0], ["fraud", "not_fraud", "fraud", "not_fraud"])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["fraud_decisions"], 2)

    def test_classifier_operating_point_selects_nearest_precision(self) -> None:
        metrics = classifier_operating_point(
            [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], target_precision=0.95
        )
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["threshold"], 0.8)

    def test_latex_marks_undefined_precision(self) -> None:
        output = {"rows": [{
            "method": "Gemma categorical",
            "decision_source": "same_slm_categorical",
            "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fraud_decisions": 0},
        }]}
        latex = render_latex(output)
        self.assertIn("Gemma categorical & -- & 0.0\\% & 0.000", latex)
        self.assertIn(r"\label{table:categorical_classifier_ablation}", latex)


if __name__ == "__main__":
    unittest.main()
