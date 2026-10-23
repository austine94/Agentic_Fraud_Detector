"""Constrained LLM evidence-orchestration controller.

This module deliberately has no payment-decision action.  A controller can
only stop or ask the governed gateway for the next permitted evidence stage.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from time import perf_counter_ns
from typing import Any, Mapping, Protocol

from src.evidence_service import EvidenceResponse, EvidenceService
from src.investigation_runtime import RiskScorer


ACTIONS = ("stop", "acquire_network", "acquire_origin", "acquire_destination")
ACTION_TOOLS = {
    "acquire_network": ("destination_network",),
    "acquire_origin": ("origin_impact",),
    "acquire_destination": ("destination_impact", "combined_impact"),
}
ACTION_COSTS = {"acquire_network": 1.0, "acquire_origin": 5.0, "acquire_destination": 5.25}
GOAL_PROFILES = {
    "evidence_first": (
        "This is the full-evidence reference setting. At every gate where an acquisition action is available, choose that "
        "acquisition action. Do not choose stop until no acquisition action remains."
    ),
    "balanced": (
        "Your goal is to balance informative evidence against cost. At later gates, stop when the latest evidence has "
        "materially resolved or stabilised the risk; otherwise acquire the next permitted evidence."
    ),
    "cost_conscious": (
        "Your goal is to minimise evidence cost while retaining useful context. Prefer the cheap network context; request a "
        "ledger stage only when the available evidence remains genuinely ambiguous after the prior stage."
    ),
}


@dataclass(frozen=True)
class ControllerRequest:
    evidence: Mapping[str, Mapping[str, Any]]
    risk: float
    allowed_actions: tuple[str, ...]
    remaining_budget: float

    def as_prompt_payload(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "risk": self.risk, "allowed_actions": self.allowed_actions,
                "remaining_budget": self.remaining_budget}


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    raw_response: str | None = None


class ControllerPolicy(Protocol):
    def choose(self, request: ControllerRequest) -> ControllerDecision: ...


class OllamaControllerPolicy:
    """Local Ollama policy constrained to the frozen action schema."""

    def __init__(self, model: str, *, endpoint: str = "http://127.0.0.1:11434/api/chat", seed: int = 20260924,
                 temperature: float = 0.0, timeout_seconds: float = 30.0, goal_profile: str = "balanced") -> None:
        if goal_profile not in GOAL_PROFILES:
            raise ValueError(f"unknown controller goal profile: {goal_profile}")
        self.model, self.endpoint, self.seed = model, endpoint, seed
        self.temperature, self.timeout_seconds = temperature, timeout_seconds
        self.goal_profile = goal_profile

    def choose(self, request: ControllerRequest) -> ControllerDecision:
        system = (
            "You are a cost-aware evidence acquisition controller. You do not classify fraud and cannot invent actions. "
            "Choose exactly one action from allowed_actions. Never repeat the request or explain your choice. "
            "acquire_network costs 1 and adds cheap recipient-network history. acquire_origin costs 5 and adds "
            "sender-ledger impact evidence. acquire_destination costs 5.25 and adds recipient plus combined-impact evidence. "
            "A case reaches you only after the entry gate. At the first gate, type and amount alone are insufficient, so choose "
            "acquire_network when it is allowed. " +
            GOAL_PROFILES[self.goal_profile]
        )
        action_schema = {"type": "object", "properties": {"action": {"type": "string", "enum": list(request.allowed_actions)}},
                         "required": ["action"], "additionalProperties": False}
        examples = [
            {"role": "user", "content": "Controller state:\n{\"evidence\":{\"initial_evidence\":{\"type\":\"TRANSFER\",\"amount\":5000}},\"risk\":0.62,\"allowed_actions\":[\"stop\",\"acquire_network\"],\"remaining_budget\":11.25}\nChoose one allowed action now."},
            {"role": "assistant", "content": "{\"action\":\"acquire_network\"}"},
            {"role": "user", "content": "Controller state:\n{\"evidence\":{\"initial_evidence\":{\"type\":\"PAYMENT\",\"amount\":20},\"destination_network\":{\"destination_is_new\":0,\"destination_prior_count\":80}},\"risk\":0.03,\"allowed_actions\":[\"stop\",\"acquire_origin\"],\"remaining_budget\":10.25}\nChoose one allowed action now."},
            {"role": "assistant", "content": "{\"action\":\"stop\"}"},
        ]
        payload = {"model": self.model, "stream": False, "format": action_schema, "think": False,
                   "messages": [{"role": "system", "content": system}, *examples,
                                {"role": "user", "content": "Controller state:\n" + json.dumps(request.as_prompt_payload(), sort_keys=True) +
                                 "\nChoose one allowed action now."}],
                   "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": 16}}
        encoded = json.dumps(payload).encode("utf-8")
        call = urllib.request.Request(self.endpoint, data=encoded, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(call, timeout=self.timeout_seconds) as response:
                content = json.loads(response.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            # The desktop sandbox can deny Python's loopback socket capability
            # while allowing the system curl client to reach a local Ollama
            # daemon.  Keep the normal stdlib path first; this is only a local
            # transport fallback and uses no shell interpolation.
            try:
                response = subprocess.run(
                    ["curl", "--fail", "--silent", "--show-error", "--max-time", str(self.timeout_seconds),
                     "-H", "Content-Type: application/json", "--data-binary", "@-", self.endpoint],
                    input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                )
                content = json.loads(response.stdout)["message"]["content"]
            except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as fallback_error:
                return ControllerDecision(action="__transport_error__", raw_response=f"{error}; curl fallback: {fallback_error}")
        try:
            parsed = json.loads(content)
            return ControllerDecision(action=str(parsed["action"]), raw_response=content)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            return ControllerDecision(action="__parse_error__", raw_response=f"{content}\n{error}")


@dataclass
class ControllerTrace:
    case_id: int
    status: str
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_trajectory: list[dict[str, Any]] = field(default_factory=list)
    controller_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency_ns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgenticControllerRuntime:
    """Gateway-owned episode runner; policy code has no direct tool access."""

    def __init__(self, service: EvidenceService, scorer: RiskScorer, *, entry_gate: float, budget: float,
                 max_invalid_retries: int = 2, auto_stop_at_full_evidence: bool = False) -> None:
        if not 0 <= entry_gate <= 1 or budget < 0 or max_invalid_retries < 0:
            raise ValueError("entry gate, budget, and max_invalid_retries must be valid")
        self.service, self.scorer = service, scorer
        self.entry_gate, self.budget, self.max_invalid_retries = entry_gate, budget, max_invalid_retries
        self.auto_stop_at_full_evidence = auto_stop_at_full_evidence

    def run(self, case_id: int, policy: ControllerPolicy) -> ControllerTrace:
        initial = self.service.initial_evidence(case_id)
        trace = ControllerTrace(case_id, initial.status)
        self._record_response(trace, initial)
        if initial.status != "ok":
            return trace
        self._score(trace)
        if trace.risk_trajectory[-1]["risk"] < self.entry_gate:
            trace.status = "below_entry_gate"
            return trace
        invalid_retries = 0
        while True:
            allowed = self._allowed_actions(trace)
            request = ControllerRequest(trace.evidence, trace.risk_trajectory[-1]["risk"], allowed, self.budget - trace.total_cost)
            if self.auto_stop_at_full_evidence and allowed == ("stop",):
                trace.controller_steps.append({"request": request.as_prompt_payload(), "proposed_action": "stop",
                                               "raw_response": None, "valid": True, "controller_latency_ns": 0,
                                               "decision_source": "runtime_forced"})
                trace.status = "stopped"
                return trace
            started = perf_counter_ns(); decision = policy.choose(request); decision_latency = perf_counter_ns() - started
            valid = decision.action in allowed
            trace.controller_steps.append({"request": request.as_prompt_payload(), "proposed_action": decision.action,
                                           "raw_response": decision.raw_response, "valid": valid,
                                           "controller_latency_ns": decision_latency, "decision_source": "policy"})
            if not valid:
                invalid_retries += 1
                if invalid_retries > self.max_invalid_retries:
                    trace.status = "invalid_action_limit"
                    return trace
                continue
            if decision.action == "stop":
                trace.status = "stopped"
                return trace
            invalid_retries = 0
            if self.budget - trace.total_cost < ACTION_COSTS[decision.action]:
                trace.status = "budget_exceeded"
                return trace
            for tool in ACTION_TOOLS[decision.action]:
                response = self.service.acquire(trace.case_id, tool, frozenset(trace.evidence).difference({"initial_evidence"}))
                self._record_response(trace, response)
                if response.status != "ok":
                    trace.status = response.status
                    return trace
            self._score(trace)

    def _allowed_actions(self, trace: ControllerTrace) -> tuple[str, ...]:
        acquired = set(trace.evidence).difference({"initial_evidence"})
        if not acquired:
            return ("stop", "acquire_network")
        if acquired == {"destination_network"}:
            return ("stop", "acquire_origin")
        if acquired == {"destination_network", "origin_impact"}:
            return ("stop", "acquire_destination")
        return ("stop",)

    def _record_response(self, trace: ControllerTrace, response: EvidenceResponse) -> None:
        trace.tool_calls.append(response.as_trace_dict())
        trace.total_cost += response.normalized_cost
        trace.total_latency_ns += response.latency_ns
        if response.status == "ok":
            trace.evidence[response.tool] = dict(response.summary)

    def _score(self, trace: ControllerTrace) -> None:
        risk = float(self.scorer.predict_risk(trace.evidence))
        if not 0 <= risk <= 1:
            raise ValueError("risk scorer must return a probability in [0, 1]")
        trace.risk_trajectory.append({"evidence_state": list(trace.evidence), "risk": risk})
