"""Small real-Ollama integration smoke test for the governed controller."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from src.agentic_controller import AgenticControllerRuntime, OllamaControllerPolicy
from src.evidence_service import EvidenceService


class FixedHighRiskScorer:
    """Forces entry to exercise orchestration; never use for evaluation."""

    def predict_risk(self, evidence) -> float:
        return .9


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few governed local-Ollama controller episodes.")
    parser.add_argument("--model", default="smollm:360m")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("results/smoke/smollm_controller.json"))
    args = parser.parse_args()
    connection = sqlite3.connect(args.split_database)
    try:
        case_ids = [row[0] for row in connection.execute(
            "SELECT case_id FROM assignments WHERE split='validation' ORDER BY case_id LIMIT ?", (args.limit,)
        )]
    finally:
        connection.close()
    service = EvidenceService(args.database, args.profiles, split_database=args.split_database)
    policy = OllamaControllerPolicy(args.model)
    runtime = AgenticControllerRuntime(service, FixedHighRiskScorer(), entry_gate=.533566727121687, budget=11.25)
    traces = [runtime.run(case_id, policy).as_dict() for case_id in case_ids]
    invalid = sum(not step["valid"] for trace in traces for step in trace["controller_steps"])
    report = {"purpose": "integration smoke only; fixed scorer forces controller entry", "model": args.model,
              "cases": len(traces), "statuses": [trace["status"] for trace in traces],
              "controller_steps": sum(len(trace["controller_steps"]) for trace in traces), "invalid_actions": invalid,
              "retry_limit_exceeded": sum(trace["status"] == "invalid_action_limit" for trace in traces), "traces": traces}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "traces"}, indent=2))


if __name__ == "__main__":
    main()
