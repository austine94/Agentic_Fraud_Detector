"""Coarse pre-authorisation ledger-impact simulation.

The simulator consumes only current *pre*-transaction balances, type, and
amount.  It intentionally returns operational bands rather than raw balances
or observed PaySim post-transaction fields.
"""

from __future__ import annotations

from typing import Any


DEBIT_ORIGIN_TYPES = frozenset(("CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"))
CREDIT_DESTINATION_TYPES = frozenset(("CASH_OUT", "PAYMENT", "TRANSFER"))

# Fixed, label-independent observation-fidelity scenarios.  ``fine`` retains
# the original development bands.  ``operational`` deliberately merges exact
# depletion with up-to-25%-residual cases.  ``coarse`` exposes only broad
# resource-pressure categories.  No scheme has a boundary at amount/balance=1.
SCHEMES = {
    "fine": {
        "burden": (0.10, 0.25, 0.50, 0.75, 1.25, 2.0),
        "residual": (0.0, 0.10, 0.25, 0.50, 0.75),
        "balance": (10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0),
        "increase": (0.10, 0.25, 0.50, 1.0, 2.0, 10.0),
    },
    "operational": {
        "burden": (0.25, 0.75, 1.50),
        "residual": (0.25, 0.75),
        "balance": (100.0, 10_000.0, 1_000_000.0),
        "increase": (0.25, 1.50, 10.0),
    },
    "coarse": {
        "burden": (0.50, 1.50),
        "residual": (0.50,),
        "balance": (10_000.0,),
        "increase": (0.50, 2.0),
    },
}


def _band(value: float, edges: tuple[float, ...]) -> int:
    """Return a fixed ordinal band; edges are policy semantics, not fitted."""
    return next((index for index, edge in enumerate(edges) if value <= edge), len(edges))


def _order_of_magnitude(value: float, edges: tuple[float, ...]) -> int:
    if value <= 0:
        return 0
    return _band(value, edges) + 1


def origin_impact_profile(transaction_type: str, amount: float, balance_before: float, scheme: str = "fine") -> dict[str, int]:
    """Forecast coarse sender liquidity if this transaction is authorised."""
    debit = transaction_type in DEBIT_ORIGIN_TYPES
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown impact scheme: {scheme}")
    edges = SCHEMES[scheme]
    projected_after = balance_before - amount if debit else balance_before + amount
    burden = amount / max(balance_before, 1.0)
    residual_fraction = projected_after / max(balance_before, 1.0)
    return {
        "origin_debit_required": int(debit),
        "origin_burden_band": _band(burden, edges["burden"]),
        "origin_residual_band": _band(residual_fraction, edges["residual"]),
        "origin_balance_order": _order_of_magnitude(balance_before, edges["balance"]),
    }


def destination_impact_profile(transaction_type: str, amount: float, balance_before: float, scheme: str = "fine") -> dict[str, int]:
    """Forecast coarse recipient exposure if this transaction is authorised."""
    credit = transaction_type in CREDIT_DESTINATION_TYPES
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown impact scheme: {scheme}")
    edges = SCHEMES[scheme]
    projected_after = balance_before + amount if credit else balance_before
    increase = amount / max(balance_before, 1.0)
    return {
        "destination_credit_expected": int(credit),
        "destination_increase_band": _band(increase, edges["increase"]),
        "destination_balance_order": _order_of_magnitude(balance_before, edges["balance"]),
        "destination_projected_balance_order": _order_of_magnitude(projected_after, edges["balance"]),
    }


def combined_impact_profile(origin: dict[str, int], destination: dict[str, int]) -> dict[str, int]:
    """Return a coarse cross-ledger impact class after both lookups."""
    return {
        "high_liquidity_pressure": int(origin["origin_residual_band"] <= 1),
        "high_destination_relative_increase": int(destination["destination_increase_band"] >= 4),
    }


def profile_from_row(row: Any, side: str, scheme: str = "fine") -> dict[str, int]:
    if side == "origin":
        return origin_impact_profile(str(row["type"]), float(row["amount"]), float(row["oldbalance_org"]), scheme)
    if side == "destination":
        return destination_impact_profile(str(row["type"]), float(row["amount"]), float(row["oldbalance_dest"]), scheme)
    raise ValueError(f"Unknown ledger side: {side}")


def features_from_row(row: Any, tools: tuple[str, ...], scheme: str = "fine") -> list[float]:
    """Expose only coarse simulated impact outputs for acquired tools."""
    valid = {"origin_impact", "destination_impact", "combined_impact"}
    if unknown := set(tools).difference(valid):
        raise ValueError(f"Unknown impact tools: {sorted(unknown)}")
    origin = profile_from_row(row, "origin", scheme) if {"origin_impact", "combined_impact"}.intersection(tools) else None
    destination = profile_from_row(row, "destination", scheme) if {"destination_impact", "combined_impact"}.intersection(tools) else None
    features: list[float] = []
    if "origin_impact" in tools:
        features.extend(float(value) for value in origin.values())  # type: ignore[union-attr]
    if "destination_impact" in tools:
        features.extend(float(value) for value in destination.values())  # type: ignore[union-attr]
    if "combined_impact" in tools:
        if origin is None:
            origin = profile_from_row(row, "origin", scheme)
        if destination is None:
            destination = profile_from_row(row, "destination", scheme)
        features.extend(float(value) for value in combined_impact_profile(origin, destination).values())
    return features
