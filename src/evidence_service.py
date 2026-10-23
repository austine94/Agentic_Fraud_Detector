"""Governed local adapters for pre-authorisation evidence acquisition.

The case identifier is a private lookup key.  Responses intentionally contain
only declared summaries: no account identifiers, labels, raw balances, or
observed post-transaction values can cross this boundary.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

from src.preauth_ledger import combined_impact_profile, destination_impact_profile, origin_impact_profile


TOOL_ORDER = ("destination_network", "origin_impact", "destination_impact", "combined_impact")
TOOL_COSTS = {
    "destination_network": 1.0,
    "origin_impact": 5.0,
    "destination_impact": 5.0,
    "combined_impact": 0.25,
}
# A transfer unit is one scalar in a sanitised response, not a raw database row.
TOOL_TRANSFER_UNITS = {
    "destination_network": 9,
    "origin_impact": 4,
    "destination_impact": 4,
    "combined_impact": 2,
}
PROHIBITED_OUTPUT_FIELDS = frozenset({
    "case_id", "step", "name_orig", "name_dest", "is_fraud", "is_flagged_fraud",
    "oldbalance_org", "oldbalance_dest", "newbalance_orig", "newbalance_dest",
    "current_balance_delta", "observed_post_transaction_state",
})


@dataclass(frozen=True)
class EvidenceResponse:
    """Auditable result of one governed acquisition call."""

    tool: str
    status: str
    summary: Mapping[str, Any]
    normalized_cost: float
    data_transfer_units: int
    latency_ns: int

    def as_trace_dict(self) -> dict[str, Any]:
        """Return the safe, serialisable representation for an investigation trace."""
        return {
            "tool": self.tool,
            "status": self.status,
            "summary": dict(self.summary),
            "normalized_cost": self.normalized_cost,
            "data_transfer_units": self.data_transfer_units,
            "latency_ns": self.latency_ns,
        }


class EvidenceService:
    """Local service simulation implementing the frozen evidence boundary."""

    def __init__(self, cases_database: Path, profiles_database: Path, *, impact_scheme: str = "operational",
                 split_database: Path | None = None, allowed_splits: frozenset[str] = frozenset({"train", "validation"})) -> None:
        if impact_scheme not in {"fine", "operational", "coarse"}:
            raise ValueError(f"Unknown impact scheme: {impact_scheme}")
        self.cases_database = Path(cases_database)
        self.profiles_database = Path(profiles_database)
        self.impact_scheme = impact_scheme
        self.split_database = Path(split_database) if split_database is not None else None
        self.allowed_splits = allowed_splits

    def acquire(self, case_id: int, tool: str, acquired_tools: frozenset[str] = frozenset()) -> EvidenceResponse:
        """Acquire a permitted summary, enforcing tool order at the boundary."""
        if tool not in TOOL_ORDER:
            raise ValueError(f"Unknown evidence tool: {tool}")
        if tool == "combined_impact" and not {"origin_impact", "destination_impact"}.issubset(acquired_tools):
            return self._response(tool, "precondition_failed", {}, 0)
        started = perf_counter_ns()
        try:
            if not self._case_is_allowed(case_id):
                return self._response(tool, "split_not_authorized", {}, perf_counter_ns() - started)
            summary = self._summary(case_id, tool)
        except sqlite3.Error:
            # The caller receives no implementation detail or private record data.
            return self._response(tool, "service_unavailable", {}, perf_counter_ns() - started)
        if summary is None:
            return self._response(tool, "case_not_found", {}, perf_counter_ns() - started)
        self._assert_safe_summary(summary)
        return self._response(tool, "ok", summary, perf_counter_ns() - started)

    def initial_evidence(self, case_id: int) -> EvidenceResponse:
        """Return the sole current-transaction fields permitted to the runtime."""
        started = perf_counter_ns()
        try:
            if not self._case_is_allowed(case_id):
                return self._initial_response("split_not_authorized", {}, perf_counter_ns() - started)
            connection = self._connection()
            try:
                row = connection.execute("SELECT type, amount FROM cases WHERE case_id=?", (case_id,)).fetchone()
            finally:
                connection.close()
        except sqlite3.Error:
            return self._initial_response("service_unavailable", {}, perf_counter_ns() - started)
        if row is None:
            return self._initial_response("case_not_found", {}, perf_counter_ns() - started)
        summary = {"type": str(row["type"]), "amount": float(row["amount"])}
        self._assert_safe_summary(summary)
        return self._initial_response("ok", summary, perf_counter_ns() - started)

    def _initial_response(self, status: str, summary: Mapping[str, Any], latency_ns: int) -> EvidenceResponse:
        return EvidenceResponse("initial_evidence", status, summary, 0.0, 2 if status == "ok" else 0, latency_ns)

    def _response(self, tool: str, status: str, summary: Mapping[str, Any], latency_ns: int) -> EvidenceResponse:
        successful = status == "ok"
        return EvidenceResponse(
            tool=tool,
            status=status,
            summary=summary,
            normalized_cost=TOOL_COSTS[tool] if successful else 0.0,
            data_transfer_units=TOOL_TRANSFER_UNITS[tool] if successful else 0,
            latency_ns=latency_ns,
        )

    def _connection(self) -> sqlite3.Connection:
        if not self.cases_database.is_file() or not self.profiles_database.is_file():
            raise sqlite3.OperationalError("required evidence store unavailable")
        connection = sqlite3.connect(f"file:{self.cases_database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("ATTACH DATABASE ? AS behavior", (f"file:{self.profiles_database}?mode=ro",))
        return connection

    def _case_is_allowed(self, case_id: int) -> bool:
        """Keep held-out cases inaccessible while validation development is active."""
        if self.split_database is None:
            return True
        if not self.split_database.is_file():
            raise sqlite3.OperationalError("split assignment unavailable")
        connection = sqlite3.connect(f"file:{self.split_database}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT split FROM assignments WHERE case_id=?", (case_id,)).fetchone()
            return row is not None and row[0] in self.allowed_splits
        finally:
            connection.close()

    def _summary(self, case_id: int, tool: str) -> dict[str, int | float] | None:
        connection = self._connection()
        try:
            if tool == "destination_network":
                row = connection.execute("""
                    SELECT p.destination_prior_count, p.destination_prior_total,
                           p.destination_recent_count, p.destination_recent_total,
                           p.destination_steps_since_last, p.destination_prior_transfer_count,
                           n.prior_unique_origins, n.recent_new_origins
                    FROM cases c JOIN behavior.profiles p USING(case_id)
                    LEFT JOIN behavior.destination_network_history n
                      ON n.account=c.name_dest AND n.step=c.step
                    WHERE c.case_id=?
                """, (case_id,)).fetchone()
                if row is None:
                    return None
                return {
                    "destination_is_new": int(not row["destination_prior_count"]),
                    "destination_prior_count": int(row["destination_prior_count"] or 0),
                    "destination_prior_total": float(row["destination_prior_total"] or 0),
                    "destination_recent_count": int(row["destination_recent_count"] or 0),
                    "destination_recent_total": float(row["destination_recent_total"] or 0),
                    "destination_steps_since_last": int(row["destination_steps_since_last"])
                    if row["destination_steps_since_last"] is not None else -1,
                    "destination_prior_transfer_count": int(row["destination_prior_transfer_count"] or 0),
                    "destination_prior_unique_origins": int(row["prior_unique_origins"] or 0),
                    "destination_recent_new_origins": int(row["recent_new_origins"] or 0),
                }
            row = connection.execute(
                "SELECT type, amount, oldbalance_org, oldbalance_dest FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
            if row is None:
                return None
            if tool == "origin_impact":
                return origin_impact_profile(row["type"], row["amount"], row["oldbalance_org"], self.impact_scheme)
            if tool == "destination_impact":
                return destination_impact_profile(row["type"], row["amount"], row["oldbalance_dest"], self.impact_scheme)
            origin = origin_impact_profile(row["type"], row["amount"], row["oldbalance_org"], self.impact_scheme)
            destination = destination_impact_profile(row["type"], row["amount"], row["oldbalance_dest"], self.impact_scheme)
            return combined_impact_profile(origin, destination)
        finally:
            connection.close()

    @staticmethod
    def _assert_safe_summary(summary: Mapping[str, Any]) -> None:
        forbidden = set(summary).intersection(PROHIBITED_OUTPUT_FIELDS)
        if forbidden:
            raise RuntimeError(f"Prohibited fields in evidence response: {sorted(forbidden)}")
