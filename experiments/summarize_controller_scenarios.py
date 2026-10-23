"""Create a compact, shared-threshold table for controller goal scenarios."""
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path
import numpy as np

FULL_EVIDENCE_THRESHOLD = .9991805471554841

def pr(labels, scores, threshold):
    predicted = scores >= threshold; tp = int(np.sum(predicted & (labels == 1))); fp = int(np.sum(predicted & (labels == 0))); fn = int(np.sum(~predicted & (labels == 1)))
    return (tp / (tp + fp) if tp + fp else 1., tp / (tp + fn) if tp + fn else 0.)

def main():
    p = argparse.ArgumentParser(); p.add_argument("reports", nargs="+", type=Path); p.add_argument("--database", type=Path, default=Path("data/evidence.sqlite")); p.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite")); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    loaded = [json.loads(path.read_text()) for path in a.reports]; ids = sorted({x["case_id"] for r in loaded for x in r["pairs"]})
    con = sqlite3.connect(a.database)
    try:
        con.execute("ATTACH DATABASE ? AS iid", (str(a.split_database),)); labels_by_id = dict(con.execute("SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) WHERE a.split='validation' AND c.case_id IN (%s)" % ",".join("?" * len(ids)), ids))
    finally: con.close()
    rows = []
    for report in loaded:
        pairs = report["pairs"]; labels = np.array([labels_by_id[x["case_id"]] for x in pairs]); stopped = np.array([x["controller"]["risk_trajectory"][-1]["risk"] for x in pairs]); full = np.array([x["full_evidence"]["risk_trajectory"][-1]["risk"] for x in pairs])
        cp, cr = pr(labels, stopped, FULL_EVIDENCE_THRESHOLD); fp, fr = pr(labels, full, FULL_EVIDENCE_THRESHOLD)
        costs = np.array([x["controller"]["total_cost"] for x in pairs]); depth = np.array([sum(s["proposed_action"].startswith("acquire_") for s in x["controller"]["controller_steps"]) for x in pairs]); latency = np.array([sum(s["controller_latency_ns"] for s in x["controller"]["controller_steps"]) / 1e9 for x in pairs])
        steps = [s for x in pairs for s in x["controller"]["controller_steps"]]; action = lambda name: float(sum(s["proposed_action"] == name for s in steps) / len(pairs))
        scenario = report.get("goal_profile", report.get("policy", {}).get("kind", "unnamed_policy"))
        rows.append({"scenario": scenario, "cases": len(pairs), "positives": int(labels.sum()), "cost_saving_mean_pct": float(np.mean(1-costs/11.25)*100), "cost_saving_sd_pct": float(np.std(1-costs/11.25, ddof=1)*100), "precision_change_at_fixed_threshold_pp": (cp-fp)*100, "recall_change_at_fixed_threshold_pp": (cr-fr)*100, "mean_stopping_depth": float(depth.mean()), "stopping_depth_sd": float(depth.std(ddof=1)), "network_rate": action("acquire_network"), "origin_avoidance_rate": 1-action("acquire_origin"), "destination_avoidance_rate": 1-action("acquire_destination"), "controller_latency_mean_s": float(latency.mean()), "controller_latency_p95_s": float(np.percentile(latency,95)), "valid_action_rate": float(sum(s["valid"] for s in steps)/len(steps)), "invalid_actions": int(sum(not s["valid"] for s in steps)), "gateway_violations": int(sum(x["controller"]["status"] == "invalid_action_limit" for x in pairs))})
    a.output.write_text(json.dumps({"evaluation_threshold": FULL_EVIDENCE_THRESHOLD, "rows": rows}, indent=2)+"\n"); print(json.dumps(rows, indent=2))
if __name__ == "__main__": main()
