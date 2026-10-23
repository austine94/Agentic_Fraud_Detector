"""Validation-only operating-threshold selection and weighted metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.investigation_runtime import TerminalThresholds


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    target_precision: float
    achieved_precision: float
    achieved_recall: float
    status: str


def select_precision_threshold(labels: Iterable[int], scores: Iterable[float], weights: Iterable[float], *, target_precision: float) -> OperatingPoint:
    """Maximise recall subject to precision, breaking ties toward lower score."""
    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(scores), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(w):
        raise ValueError("labels, scores, and weights must be non-empty and aligned")
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    candidates = np.unique(p)
    candidates.sort()
    for threshold in candidates:
        predicted = p >= threshold
        true_positive = float(w[(predicted) & (y == 1)].sum())
        false_positive = float(w[(predicted) & (y == 0)].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
        recall = true_positive / float(w[y == 1].sum())
        if precision >= target_precision:
            return OperatingPoint(float(threshold), target_precision, precision, recall, "selected")
    # Scores are probabilities.  A threshold of one is the least-actionable
    # representable terminal setting while preserving the threshold contract.
    return OperatingPoint(1.0, target_precision, 1.0, 0.0, "no_feasible_threshold")


def select_recall_threshold(labels: Iterable[int], scores: Iterable[float], weights: Iterable[float], *, target_recall: float) -> OperatingPoint:
    """Choose the highest threshold meeting a recall target, limiting LLM load."""
    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(scores), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(w):
        raise ValueError("labels, scores, and weights must be non-empty and aligned")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    positives = float(w[y == 1].sum())
    for threshold in np.unique(p)[::-1]:
        selected = p >= threshold
        true_positive = float(w[selected & (y == 1)].sum())
        false_positive = float(w[selected & (y == 0)].sum())
        recall = true_positive / positives
        if recall >= target_recall:
            precision = true_positive / (true_positive + false_positive)
            return OperatingPoint(float(threshold), target_recall, precision, recall, "selected")
    return OperatingPoint(0.0, target_recall, float(y.mean()), 1.0, "all_cases_required")


def select_terminal_thresholds(labels: Iterable[int], scores: Iterable[float], weights: Iterable[float], *, flag_precision: float = .99,
                               escalation_precision: float = .999) -> tuple[TerminalThresholds, dict[str, dict[str, float | str]]]:
    """Select nested flag/escalation thresholds from validation evidence only."""
    labels, scores, weights = list(labels), list(scores), list(weights)
    flag = select_precision_threshold(labels, scores, weights, target_precision=flag_precision)
    escalation = select_precision_threshold(labels, scores, weights, target_precision=escalation_precision)
    if escalation.threshold < flag.threshold:
        escalation = OperatingPoint(flag.threshold, escalation.target_precision, flag.achieved_precision, flag.achieved_recall, "raised_to_flag_threshold")
    thresholds = TerminalThresholds(flag=flag.threshold, escalate=escalation.threshold)
    return thresholds, {"flag": asdict(flag), "escalate": asdict(escalation)}


def weighted_binary_metrics(labels: Iterable[int], scores: Iterable[float], weights: Iterable[float], *, threshold: float) -> dict[str, float]:
    """Report prevalence-adjusted ranking and operating-point metrics."""
    y, p, w = np.asarray(list(labels), dtype=np.int8), np.asarray(list(scores), dtype=float), np.asarray(list(weights), dtype=float)
    predicted = p >= threshold
    tp = float(w[predicted & (y == 1)].sum()); fp = float(w[predicted & (y == 0)].sum())
    fn = float(w[~predicted & (y == 1)].sum()); tn = float(w[~predicted & (y == 0)].sum())
    return {
        "pr_auc_prevalence_adjusted": float(average_precision_score(y, p, sample_weight=w)),
        "roc_auc_prevalence_adjusted": float(roc_auc_score(y, p, sample_weight=w)),
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "routed_case_rate": float(np.average(predicted, weights=w)),
    }


def precision_frontier(labels: Iterable[int], scores: Iterable[float], weights: Iterable[float], *, targets: Iterable[float]) -> dict[str, object]:
    """Summarise feasible operating points without changing the selected contract."""
    y, p, w = list(labels), list(scores), list(weights)
    points = {f"precision_{target:g}": select_precision_threshold(y, p, w, target_precision=target).__dict__ for target in targets}
    candidates = np.unique(np.asarray(p, dtype=float)); candidates.sort()
    best_precision = -1.0; best_threshold = 1.0; best_recall = 0.0
    y_array, p_array, w_array = np.asarray(y), np.asarray(p), np.asarray(w)
    positives = float(w_array[y_array == 1].sum())
    for threshold in candidates:
        selected = p_array >= threshold
        tp = float(w_array[selected & (y_array == 1)].sum()); fp = float(w_array[selected & (y_array == 0)].sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / positives if positives else 0.0
        if precision > best_precision or (precision == best_precision and recall > best_recall):
            best_precision, best_threshold, best_recall = precision, float(threshold), recall
    return {"targets": points, "maximum_observed_precision": best_precision,
            "maximum_precision_threshold": best_threshold, "recall_at_maximum_precision": best_recall}
