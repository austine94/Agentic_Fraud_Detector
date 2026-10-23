import unittest

import numpy as np

from experiments.run_terminal_classifier_ablation import metric_summary, terminal_prompt


class TerminalClassifierAblationTests(unittest.TestCase):
    def test_prompt_contains_evidence_but_no_classifier_risk(self) -> None:
        payload = terminal_prompt({"initial_evidence": {"type": "TRANSFER", "amount": 100.0}})
        rendered = str(payload)
        self.assertIn("TRANSFER", rendered)
        self.assertNotIn("risk_trajectory", rendered)
        self.assertNotIn('"risk"', rendered)

    def test_metrics_include_99_percent_precision_recall(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        metrics = metric_summary(labels, scores)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["recall_at_99pct_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
