"""Train-only, non-LLM policies for sequential evidence stopping.

The policy observes only governed evidence and risks available at the current
gate. Stage-specific regressors predict how much the risk score would change if the
remaining evidence were acquired. A validation-selected threshold converts
that predicted value of information into a stop/continue decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.agentic_controller import ControllerDecision, ControllerRequest
from src.comparison_policies import _next_acquisition, state_name
from src.risk_scoring import feature_vector_from_evidence


STAGES = ("initial", "network", "origin")
DEPTH_COSTS = np.asarray((0.0, 1.0, 6.0, 11.25), dtype=float)


def trajectory_features(risks: Sequence[float], stage: int) -> np.ndarray:
    """Return fixed, gate-local features from risks observed so far."""
    observed = np.asarray(risks[: stage + 1], dtype=float)
    if observed.shape != (stage + 1,):
        raise ValueError("risk trajectory is shorter than the requested stage")
    current = float(observed[-1])
    clipped = float(np.clip(current, 1e-6, 1 - 1e-6))
    features = [
        *observed.tolist(),
        current - 0.5,
        abs(current - 0.5),
        float(np.log(clipped / (1 - clipped))),
    ]
    if stage:
        deltas = np.diff(observed)
        features.extend(deltas.tolist())
        features.extend(np.abs(deltas).tolist())
    return np.asarray(features, dtype=float)


def feature_matrix(risks: np.ndarray, stage: int) -> np.ndarray:
    if risks.ndim != 2 or risks.shape[1] != 4:
        raise ValueError("risks must have shape (n_cases, 4)")
    return np.vstack([trajectory_features(row, stage) for row in risks])


@dataclass
class LearnedStoppingPolicy:
    """Runtime adapter for three fitted value-of-information regressors."""

    models: Sequence[Any]
    value_threshold: float
    risk_history: list[float] | None = None

    def __post_init__(self) -> None:
        if len(self.models) != 3 or self.value_threshold < 0:
            raise ValueError("three stage models and a non-negative value threshold are required")

    def reset(self) -> None:
        self.risk_history = []

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        stage_name = state_name(request)
        if stage_name not in STAGES:
            return ControllerDecision("stop")
        if self.risk_history is None or len(self.risk_history) != STAGES.index(stage_name):
            self.risk_history = []
        self.risk_history.append(float(request.risk))
        stage = STAGES.index(stage_name)
        features = np.concatenate((
            feature_vector_from_evidence(request.evidence),
            trajectory_features(self.risk_history, stage),
        )).reshape(1, -1)
        predicted_value = max(0.0, float(self.models[stage].predict(features)[0]))
        should_continue = predicted_value > self.value_threshold
        return ControllerDecision(_next_acquisition(request) if should_continue else "stop")
