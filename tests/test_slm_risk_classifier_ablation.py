import unittest

from experiments.run_slm_risk_classifier_ablation import risk_prompt


class SlmRiskClassifierAblationTests(unittest.TestCase):
    def test_risk_prompt_has_evidence_but_no_classifier_score_or_label(self) -> None:
        payload = risk_prompt({"initial_evidence": {"type": "TRANSFER", "amount": 100.0}})
        rendered = str(payload)
        self.assertIn("TRANSFER", rendered)
        self.assertNotIn("risk_trajectory", rendered)
        self.assertNotIn("is_fraud", rendered)
        self.assertNotIn('"risk"', rendered)


if __name__ == "__main__":
    unittest.main()
