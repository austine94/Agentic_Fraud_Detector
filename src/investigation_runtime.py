"""Deterministic, governed investigation runtime for Phase 2.

Policy code can choose to stop or request an action, but this runtime owns all
case lookup, action validation, metering, scoring, and terminal decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from src.evidence_service import EvidenceResponse, EvidenceService, TOOL_COSTS, TOOL_ORDER


class RiskScorer(Protocol):
    """A state-specific calibrated scorer supplied by the modelling layer."""

    def predict_risk(self, evidence: Mapping[str, Mapping[str, Any]]) -> float: ...


@dataclass(frozen=True)
class TerminalThresholds:
    """Validation-selected operating thresholds; no defaults are silently frozen."""

    flag: float
    escalate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.flag <= self.escalate <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= flag <= escalate <= 1")


def terminal_action_for_risk(risk: float, thresholds: TerminalThresholds) -> str:
    if risk >= thresholds.escalate:
        return "escalate"
    if risk >= thresholds.flag:
        return "flag"
    return "approve"


@dataclass
class InvestigationTrace:
    case_id: int
    status: str
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_trajectory: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    terminal_action: str | None = None
    total_cost: float = 0.0
    total_transfer_units: int = 0
    total_latency_ns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "status": self.status, "evidence": self.evidence,
            "risk_trajectory": self.risk_trajectory, "tool_calls": self.tool_calls,
            "terminal_action": self.terminal_action, "total_cost": self.total_cost,
            "total_transfer_units": self.total_transfer_units, "total_latency_ns": self.total_latency_ns,
        }


class InvestigationRuntime:
    """Runs a policy while enforcing the experiment's action and budget contract."""

    def __init__(self, service: EvidenceService, scorer: RiskScorer, thresholds: TerminalThresholds, *, budget: float) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.service, self.scorer, self.thresholds, self.budget = service, scorer, thresholds, budget

    def run_full_evidence(self, case_id: int) -> InvestigationTrace:
        """Reference policy: take every stage possible, then classify deterministically."""
        trace = self._start(case_id)
        if trace.status != "ok":
            return trace
        for action in TOOL_ORDER:
            response = self.acquire(trace, action)
            if response.status != "ok":
                trace.status = response.status
                return trace
        trace.terminal_action = terminal_action_for_risk(trace.risk_trajectory[-1]["risk"], self.thresholds)
        trace.status = "completed"
        return trace

    def run_transaction_only(self, case_id: int) -> InvestigationTrace:
        """Lower-bound policy: classify using type and amount with no acquisition."""
        trace = self._start(case_id)
        if trace.status != "ok":
            return trace
        trace.terminal_action = terminal_action_for_risk(trace.risk_trajectory[-1]["risk"], self.thresholds)
        trace.status = "completed"
        return trace

    def _start(self, case_id: int) -> InvestigationTrace:
        initial = self.service.initial_evidence(case_id)
        trace = InvestigationTrace(case_id=case_id, status="ok")
        self._record(trace, initial)
        if initial.status != "ok":
            trace.status = initial.status
            return trace
        self._score(trace)
        return trace

    def acquire(self, trace: InvestigationTrace, action: str) -> EvidenceResponse:
        """The only tool gateway exposed to acquisition policies."""
        if trace.status != "ok":
            raise RuntimeError("cannot acquire evidence for an inactive investigation")
        if action not in TOOL_ORDER:
            raise ValueError(f"action is not an evidence tool: {action}")
        acquired_count = len(set(trace.evidence).difference({"initial_evidence"}))
        expected = TOOL_ORDER[acquired_count] if acquired_count < len(TOOL_ORDER) else None
        # Rejected gateway attempts are traced but do not advance the state.
        if action != expected:
            response = EvidenceResponse(action, "action_not_allowed", {}, 0.0, 0, 0)
            self._record(trace, response)
            return response
        if trace.total_cost + TOOL_COSTS[action] > self.budget:
            response = EvidenceResponse(action, "budget_exceeded", {}, 0.0, 0, 0)
            self._record(trace, response)
            return response
        response = self.service.acquire(trace.case_id, action, frozenset(trace.evidence).difference({"initial_evidence"}))
        self._record(trace, response)
        if response.status == "ok":
            self._score(trace)
        return response

    def _record(self, trace: InvestigationTrace, response: EvidenceResponse) -> None:
        trace.tool_calls.append(response.as_trace_dict())
        trace.total_cost += response.normalized_cost
        trace.total_transfer_units += response.data_transfer_units
        trace.total_latency_ns += response.latency_ns
        if response.status == "ok":
            trace.evidence[response.tool] = dict(response.summary)

    def _score(self, trace: InvestigationTrace) -> None:
        risk = float(self.scorer.predict_risk(trace.evidence))
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk scorer must return a probability in [0, 1]")
        trace.risk_trajectory.append({"evidence_state": list(trace.evidence), "risk": risk})
