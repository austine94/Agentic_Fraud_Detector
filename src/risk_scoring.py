"""State-specific calibrated fraud-risk scorers.

Feature construction starts with the exact safe summaries emitted by
``EvidenceService``.  This keeps the training/inference boundary explicit:
raw balances may be private inputs to impact synthesis, but they never enter a
classifier vector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from src.preauth_behavior import initial_features, tool_features
from src.preauth_ledger import combined_impact_profile, destination_impact_profile, origin_impact_profile


STATE_TOOLSETS = (
    (),
    ("destination_network",),
    ("destination_network", "origin_impact"),
    ("destination_network", "origin_impact", "destination_impact"),
    ("destination_network", "origin_impact", "destination_impact", "combined_impact"),
)
ORIGIN_FIELDS = ("origin_debit_required", "origin_burden_band", "origin_residual_band", "origin_balance_order")
DESTINATION_FIELDS = ("destination_credit_expected", "destination_increase_band", "destination_balance_order", "destination_projected_balance_order")
COMBINED_FIELDS = ("high_liquidity_pressure", "high_destination_relative_increase")


def evidence_state(evidence: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the frozen, order-preserving evidence state signature."""
    return tuple(tool for tool in STATE_TOOLSETS[-1] if tool in evidence)


def feature_vector_from_evidence(evidence: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Convert an adapter-safe evidence state into one numeric model vector."""
    if "initial_evidence" not in evidence:
        raise ValueError("initial evidence is required before scoring")
    state = evidence_state(evidence)
    if state not in STATE_TOOLSETS:
        raise ValueError(f"invalid evidence state: {state}")
    initial = evidence["initial_evidence"]
    values = list(initial_features(initial))
    if "destination_network" in evidence:
        network_row = dict(initial) | dict(evidence["destination_network"])
        values.extend(tool_features(network_row, "destination_network"))
    for tool, fields in (("origin_impact", ORIGIN_FIELDS), ("destination_impact", DESTINATION_FIELDS), ("combined_impact", COMBINED_FIELDS)):
        if tool in evidence:
            summary = evidence[tool]
            missing = set(fields).difference(summary)
            if missing:
                raise ValueError(f"{tool} summary missing fields: {sorted(missing)}")
            values.extend(float(summary[field]) for field in fields)
    return np.asarray(values, dtype=np.float32)


def safe_evidence_from_row(row: Mapping[str, Any], *, impact_scheme: str = "operational") -> dict[str, dict[str, Any]]:
    """Build adapter-equivalent evidence for *training* from a private row.

    The returned object has the same public shape as a completed full-evidence
    trace.  Callers retain labels separately; no label or raw balance is placed
    in the resulting evidence object.
    """
    initial = {"type": str(row["type"]), "amount": float(row["amount"])}
    network = {
        "destination_is_new": int(not row["destination_prior_count"]),
        "destination_prior_count": int(row["destination_prior_count"] or 0),
        "destination_prior_total": float(row["destination_prior_total"] or 0),
        "destination_recent_count": int(row["destination_recent_count"] or 0),
        "destination_recent_total": float(row["destination_recent_total"] or 0),
        "destination_steps_since_last": int(row["destination_steps_since_last"])
        if row["destination_steps_since_last"] is not None else -1,
        "destination_prior_transfer_count": int(row["destination_prior_transfer_count"] or 0),
        "destination_prior_unique_origins": int(row["destination_prior_unique_origins"] or 0),
        "destination_recent_new_origins": int(row["destination_recent_new_origins"] or 0),
    }
    origin = origin_impact_profile(initial["type"], initial["amount"], float(row["oldbalance_org"]), impact_scheme)
    destination = destination_impact_profile(initial["type"], initial["amount"], float(row["oldbalance_dest"]), impact_scheme)
    return {
        "initial_evidence": initial, "destination_network": network,
        "origin_impact": origin, "destination_impact": destination,
        "combined_impact": combined_impact_profile(origin, destination),
    }


@dataclass(frozen=True)
class StateRiskScorer:
    """A fitted calibrated model for every permitted acquisition state."""

    models: Mapping[tuple[str, ...], Any]

    def predict_risk(self, evidence: Mapping[str, Mapping[str, Any]]) -> float:
        state = evidence_state(evidence)
        model = self.models.get(state)
        if model is None:
            raise ValueError(f"no fitted scorer for evidence state: {state}")
        vector = feature_vector_from_evidence(evidence).reshape(1, -1)
        return float(model.predict_proba(vector)[0, 1])

    def predict_risks(self, evidence_rows: Iterable[Mapping[str, Mapping[str, Any]]]) -> np.ndarray:
        """Score a same-state batch without changing the fitted model or calibration."""
        rows = list(evidence_rows)
        if not rows:
            return np.asarray([], dtype=float)
        state = evidence_state(rows[0])
        if any(evidence_state(row) != state for row in rows):
            raise ValueError("batch scoring requires one shared evidence state")
        model = self.models.get(state)
        if model is None:
            raise ValueError(f"no fitted scorer for evidence state: {state}")
        matrix = np.vstack([feature_vector_from_evidence(row) for row in rows])
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def fit_state_risk_scorers(examples: Iterable[Mapping[str, Mapping[str, Any]]], labels: Iterable[int], *, seed: int,
                           sample_weights: Iterable[float] | None = None,
                           states: Iterable[tuple[str, ...]] = STATE_TOOLSETS) -> StateRiskScorer:
    """Fit calibrated models from train-only adapter-equivalent evidence states."""
    examples = list(examples)
    labels_array = np.asarray(list(labels), dtype=np.int8)
    if not examples or len(examples) != len(labels_array):
        raise ValueError("examples and labels must be non-empty and equally sized")
    counts = Counter(labels_array.tolist())
    if set(counts) != {0, 1} or min(counts.values()) < 3:
        raise ValueError("at least three examples of each class are required for calibration")
    weights_array = np.ones(len(labels_array), dtype=float) if sample_weights is None else np.asarray(list(sample_weights), dtype=float)
    if len(weights_array) != len(labels_array) or np.any(weights_array <= 0):
        raise ValueError("sample weights must be positive and aligned with examples")
    models: dict[tuple[str, ...], Any] = {}
    calibration_folds = min(3, min(counts.values()))
    for state in states:
        if state not in STATE_TOOLSETS:
            raise ValueError(f"unknown scorer state: {state}")
        matrices = []
        for example in examples:
            permitted = {"initial_evidence", *state}
            matrices.append(feature_vector_from_evidence({key: value for key, value in example.items() if key in permitted}))
        estimator = HistGradientBoostingClassifier(
            learning_rate=.1, max_iter=100, max_leaf_nodes=31, l2_regularization=1., random_state=seed,
        )
        model = CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_folds)
        model.fit(np.vstack(matrices), labels_array, sample_weight=weights_array)
        models[state] = model
    return StateRiskScorer(models)
