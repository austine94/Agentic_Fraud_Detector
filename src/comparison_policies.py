"""Deterministic evidence-acquisition baselines for controller comparison.

These policies deliberately share the constrained controller action contract.
They do not make payment decisions, access stores, or bypass the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.agentic_controller import ControllerDecision, ControllerRequest


def _next_acquisition(request: ControllerRequest) -> str:
    """Return the only permitted acquisition action, otherwise stop."""
    return next((action for action in request.allowed_actions if action != "stop"), "stop")


def state_name(request: ControllerRequest) -> str:
    """A stable, safe state label for deterministic policy specifications."""
    evidence = request.evidence
    if "origin_impact" in evidence:
        return "origin"
    if "destination_network" in evidence:
        return "network"
    return "initial"


class TransactionOnlyPolicy:
    """Never acquire extra evidence after the shared entry gate."""

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        return ControllerDecision("stop")


class FullEvidencePolicy:
    """Acquire every permitted stage, then stop."""

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        return ControllerDecision(_next_acquisition(request))


@dataclass(frozen=True)
class SimpleThresholdCascadePolicy:
    """Continue only while a score lies inside one fixed ambiguity band."""

    continue_low: float
    continue_high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.continue_low <= self.continue_high <= 1.0:
            raise ValueError("continuation band must satisfy 0 <= low <= high <= 1")

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        if self.continue_low <= request.risk <= self.continue_high:
            return ControllerDecision(_next_acquisition(request))
        return ControllerDecision("stop")


@dataclass(frozen=True)
class StructuredRulePolicy:
    """Per-state, validation-selected ambiguity bands without an LLM."""

    continuation_bands: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        expected = {"initial", "network", "origin"}
        if set(self.continuation_bands) != expected:
            raise ValueError("structured policy needs exactly initial, network, and origin bands")
        for low, high in self.continuation_bands.values():
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("each continuation band must satisfy 0 <= low <= high <= 1")

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        low, high = self.continuation_bands[state_name(request)]
        if low <= request.risk <= high:
            return ControllerDecision(_next_acquisition(request))
        return ControllerDecision("stop")
