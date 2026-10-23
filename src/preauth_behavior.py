"""Pre-authorisation, label-free behavioural feature contract.

Every profile value is computed from transactions at strictly earlier PaySim
steps.  Same-step rows are deliberately excluded: their file order is not a
credible event-time ordering.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TYPE_INDEX = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
TOOL_NAMES = ("origin_velocity", "relationship", "destination_network", "flow_sequence")


def _log(value: Any) -> float:
    return float(np.log1p(max(float(value or 0.0), 0.0)))


def _recency(value: Any) -> float:
    """Encode missing history separately from elapsed whole steps."""
    return -1.0 if value is None else _log(value)


def initial_features(row: Any) -> list[float]:
    """Only current transaction fields available before authorisation."""
    return [_log(row["amount"])] + [float(row["type"] == kind) for kind in TYPE_INDEX]


def tool_features(row: Any, tool: str) -> list[float]:
    """Return only the pre-authorisation profile fields for one tool family."""
    if tool == "origin_velocity":
        count = float(row["origin_prior_count"] or 0)
        total = float(row["origin_prior_total"] or 0)
        sum_squared = float(row["origin_prior_sum_squared"] or 0)
        mean = total / count if count else 0.0
        variance = max(0.0, sum_squared / count - mean * mean) if count else 0.0
        return [
            float(count > 0), _log(count), _log(total), _log(variance),
            _log(row["origin_recent_count"]), _log(row["origin_recent_total"]),
            _recency(row["origin_steps_since_last"]),
            _log(row["origin_prior_transfer_count"]), _log(row["origin_prior_cashout_count"]),
            (float(row["amount"]) - mean) / (variance ** 0.5 + 1.0),
            float(row["amount"]) / max(mean, 1.0),
        ]
    if tool == "relationship":
        return [
            float((row["pair_prior_count"] or 0) == 0), _log(row["pair_prior_count"]),
            _log(row["pair_prior_total"]), _recency(row["pair_steps_since_last"]),
        ]
    if tool == "destination_network":
        return [
            float((row["destination_prior_count"] or 0) == 0), _log(row["destination_prior_count"]),
            _log(row["destination_prior_total"]), _log(row["destination_recent_count"]),
            _log(row["destination_recent_total"]), _recency(row["destination_steps_since_last"]),
            _log(row["destination_prior_transfer_count"]), _log(row["destination_prior_unique_origins"]),
            _log(row["destination_recent_new_origins"]),
        ]
    if tool == "flow_sequence":
        return [
            float((row["origin_incoming_prior_count"] or 0) == 0),
            _log(row["origin_incoming_prior_count"]), _log(row["origin_incoming_recent_count"]),
            _log(row["origin_incoming_recent_total"]),
            _recency(row["origin_steps_since_incoming"]),
            _recency(row["origin_steps_since_transfer_incoming"]),
            _log(row["origin_last_transfer_incoming_total"]),
            float(row["amount"]) / max(float(row["origin_last_transfer_incoming_total"] or 0.0), 1.0),
        ]
    raise ValueError(f"Unknown behaviour tool: {tool}")


def feature_vector(row: Any, tools: tuple[str, ...]) -> np.ndarray:
    """Compose current evidence with only explicitly acquired tool families."""
    unknown = set(tools).difference(TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown behaviour tools: {sorted(unknown)}")
    features = initial_features(row)
    for tool in tools:
        features.extend(tool_features(row, tool))
    return np.asarray(features, dtype=np.float32)
