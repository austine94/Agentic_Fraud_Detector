"""Train and evaluate conventional learned evidence-stopping baselines.

Models are trained only on train-split governed evidence and risk trajectories.
Their value-of-information threshold is selected on the existing 500-case validation cohort, after which the frozen
policy is replayed over the saved representative-test full-evidence traces.
No test label or outcome participates in fitting or policy selection.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeRegressor

from experiments.representative_test import ENTRY_GATE, SEED, training_rows
from src.iid_split import attach_assignment
from src.learned_stopping import (
    DEPTH_COSTS,
    feature_matrix,
)
from src.risk_scoring import (
    STATE_TOOLSETS,
    feature_vector_from_evidence,
    fit_state_risk_scorers,
    safe_evidence_from_row,
)


MODEL_LABELS = {
    "cart": "Learned CART stopping",
    "boosted": "Learned boosted stopping",
}
STAGE_STATES = (STATE_TOOLSETS[0], STATE_TOOLSETS[1], STATE_TOOLSETS[2])
FROZEN_THRESHOLD = 0.9991805471554841


def four_state_risks(scorer, examples: list[dict]) -> np.ndarray:
    columns = []
    for state in (STATE_TOOLSETS[0], STATE_TOOLSETS[1], STATE_TOOLSETS[2], STATE_TOOLSETS[4]):
        permitted = {"initial_evidence", *state}
        rows = [{key: value for key, value in example.items() if key in permitted} for example in examples]
        columns.append(scorer.predict_risks(rows))
    return np.column_stack(columns)


def policy_feature_matrices(examples: list[dict], risks: np.ndarray) -> list[np.ndarray]:
    matrices = []
    for stage, state in enumerate(STAGE_STATES):
        permitted = {"initial_evidence", *state}
        evidence_matrix = np.vstack([
            feature_vector_from_evidence({key: value for key, value in example.items() if key in permitted})
            for example in examples
        ])
        matrices.append(np.column_stack((evidence_matrix, feature_matrix(risks, stage))))
    return matrices


def build_train_cache(database: Path, profiles: Path, split_database: Path, cache: Path):
    if cache.exists():
        saved = np.load(cache)
        if all(key in saved for key in ("x0", "x1", "x2")):
            return (
                np.asarray(saved["risks"], dtype=float),
                np.asarray(saved["labels"], dtype=np.int8),
                [np.asarray(saved[f"x{stage}"], dtype=float) for stage in range(3)],
            )
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        attach_assignment(connection, split_database)
        connection.execute("ATTACH DATABASE ? AS behavior", (str(profiles),))
        rows = list(training_rows(connection))
    finally:
        connection.close()
    labels = np.asarray([int(row["is_fraud"]) for row in rows], dtype=np.int8)
    examples = [safe_evidence_from_row(row) for row in rows]
    positive_weight = (len(labels) - int(labels.sum())) / int(labels.sum())
    scorer = fit_state_risk_scorers(
        examples,
        labels,
        seed=SEED,
        sample_weights=[positive_weight if label else 1.0 for label in labels],
        states=(STATE_TOOLSETS[0], STATE_TOOLSETS[1], STATE_TOOLSETS[2], STATE_TOOLSETS[4]),
    )
    risks = four_state_risks(scorer, examples)
    routed = risks[:, 0] >= ENTRY_GATE
    risks, labels = risks[routed], labels[routed]
    examples = [example for example, keep in zip(examples, routed, strict=True) if keep]
    matrices = policy_feature_matrices(examples, risks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, risks=risks, labels=labels, x0=matrices[0], x1=matrices[1], x2=matrices[2])
    return risks, labels, matrices


def load_validation(report_path: Path, database: Path, split_database: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pairs = report["pairs"]
    case_ids = [int(pair["case_id"]) for pair in pairs]
    connection = sqlite3.connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS iid", (str(split_database),))
        placeholders = ",".join("?" for _ in case_ids)
        labels_by_id = dict(connection.execute(
            f"SELECT c.case_id,c.is_fraud FROM cases c JOIN iid.assignments a USING(case_id) "
            f"WHERE a.split='validation' AND c.case_id IN ({placeholders})",
            case_ids,
        ))
    finally:
        connection.close()
    labels = np.asarray([labels_by_id[case_id] for case_id in case_ids], dtype=np.int8)
    risks = np.asarray([
        [float(state["risk"]) for state in pair["full_evidence"]["risk_trajectory"]]
        for pair in pairs
    ])
    if risks.shape != (len(pairs), 4):
        raise ValueError("validation source must contain four-state full-evidence trajectories")
    examples = [pair["full_evidence"]["evidence"] for pair in pairs]
    return report, case_ids, labels, risks, policy_feature_matrices(examples, risks)


def make_estimator(kind: str):
    if kind == "cart":
        return DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, random_state=SEED)
    if kind == "boosted":
        return HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            max_depth=4,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=SEED,
        )
    raise ValueError(f"unknown estimator kind: {kind}")


def fit_models(kind: str, train_risks: np.ndarray, train_features: list[np.ndarray]):
    models = []
    for stage in range(3):
        model = make_estimator(kind)
        target = np.abs(train_risks[:, -1] - train_risks[:, stage])
        model.fit(train_features[stage], target)
        models.append(model)
    return models


def replay_value_models(models, features: list[np.ndarray], risks: np.ndarray, threshold: float):
    depth = np.zeros(len(risks), dtype=np.int8)
    active = np.ones(len(risks), dtype=bool)
    for stage, model in enumerate(models):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        predicted_value = np.maximum(0.0, np.asarray(model.predict(features[stage][indices]), dtype=float))
        chosen = indices[predicted_value > threshold]
        depth[chosen] = stage + 1
        active[:] = False
        active[chosen] = True
    return risks[np.arange(len(risks)), depth], DEPTH_COSTS[depth], depth


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, sample_weight=None) -> dict[str, float]:
    result = {
        "pr_auc": float(average_precision_score(labels, scores, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
    }
    if sample_weight is None:
        precision, recall, _ = precision_recall_curve(labels, scores)
        for target in (0.80, 0.90, 0.95):
            eligible = recall[precision >= target]
            result[f"recall_at_{int(target * 100)}pct_precision"] = float(eligible.max()) if eligible.size else 0.0
    return result


def operating_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, weights: np.ndarray) -> dict:
    predicted = scores >= threshold
    tp = float(weights[(labels == 1) & predicted].sum())
    fp = float(weights[(labels == 0) & predicted].sum())
    fn = float(weights[(labels == 1) & ~predicted].sum())
    tn = float(weights[(labels == 0) & ~predicted].sum())
    return {
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    false_positive_rate, _, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(false_positive_rate <= target)
    return float(thresholds[eligible[-1]])


def select_model(
    kind: str,
    train_risks: np.ndarray,
    train_features: list[np.ndarray],
    validation_risks: np.ndarray,
    validation_features: list[np.ndarray],
    validation_labels: np.ndarray,
    max_pr_auc_loss: float,
):
    full_pr_auc = average_precision_score(validation_labels, validation_risks[:, -1])
    models = fit_models(kind, train_risks, train_features)
    predictions = np.concatenate([
        np.maximum(0.0, np.asarray(model.predict(validation_features[stage]), dtype=float))
        for stage, model in enumerate(models)
    ])
    positive = predictions[predictions > 0]
    thresholds = np.unique(np.concatenate(([-1e-12, 0.0], np.quantile(positive, np.linspace(0, 1, 101)))))
    candidates = []
    for threshold in thresholds:
        scores, costs, depths = replay_value_models(models, validation_features, validation_risks, float(threshold))
        metrics = ranking_metrics(validation_labels, scores)
        candidates.append({
            "value_threshold": float(threshold),
            "models": models,
            "scores": scores,
            "costs": costs,
            "depths": depths,
            "metrics": metrics,
            "mean_cost": float(costs.mean()),
            "cost_saving": float(1 - costs.mean() / DEPTH_COSTS[-1]),
            "eligible": metrics["pr_auc"] >= full_pr_auc - max_pr_auc_loss,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda candidate: (candidate["cost_saving"], candidate["metrics"]["pr_auc"]))
    else:
        selected = max(candidates, key=lambda candidate: (candidate["metrics"]["pr_auc"], candidate["cost_saving"]))
    return selected, candidates


def controller_trace(risk_row: np.ndarray, depth: int) -> dict:
    states = (
        ["initial_evidence"],
        ["initial_evidence", "destination_network"],
        ["initial_evidence", "destination_network", "origin_impact"],
        ["initial_evidence", "destination_network", "origin_impact", "destination_impact", "combined_impact"],
    )
    actions = ("acquire_network", "acquire_origin", "acquire_destination")
    trajectory = [{"evidence_state": states[index], "risk": float(risk_row[index])} for index in range(depth + 1)]
    steps = [
        {"proposed_action": actions[index], "valid": True, "controller_latency_ns": 0, "decision_source": "learned_policy"}
        for index in range(depth)
    ]
    steps.append({"proposed_action": "stop", "valid": True, "controller_latency_ns": 0, "decision_source": "learned_policy"})
    return {"status": "stopped", "risk_trajectory": trajectory, "controller_steps": steps, "total_cost": float(DEPTH_COSTS[depth])}


def write_validation_report(path: Path, kind: str, source: dict, case_ids: list[int], risks: np.ndarray, selected: dict) -> None:
    pairs = []
    for index, (case_id, depth) in enumerate(zip(case_ids, selected["depths"], strict=True)):
        pairs.append({
            "case_id": case_id,
            "controller": controller_trace(risks[index], int(depth)),
            "full_evidence": source["pairs"][index]["full_evidence"],
        })
    report = {
        "purpose": "validation evaluation of a train-fitted non-LLM learned stopping policy",
        "policy": {"kind": f"learned_{kind}_stopping", "value_threshold": selected["value_threshold"]},
        "goal_profile": MODEL_LABELS[kind],
        "entry_gate": ENTRY_GATE,
        "cases": len(case_ids),
        "case_ids": case_ids,
        "mean_cost": selected["mean_cost"],
        "pairs": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def evidence_from_tool_calls(trace: dict) -> dict:
    return {call["tool"]: call["summary"] for call in trace["tool_calls"] if call["status"] == "ok"}


def load_test(checkpoint_path: Path):
    connection = sqlite3.connect(checkpoint_path)
    try:
        cohort = list(connection.execute("SELECT case_id,label,initial_risk,routed FROM cohort ORDER BY case_id"))
        full = dict(connection.execute(
            "SELECT case_id,trace_json FROM results WHERE policy='full_evidence' ORDER BY case_id"
        ))
        population = json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key='test_non_fraud_population'"
        ).fetchone()[0])
    finally:
        connection.close()
    routed_ids = [int(row[0]) for row in cohort if row[3]]
    if set(routed_ids) != set(full):
        raise ValueError("representative checkpoint lacks complete full-evidence routed traces")
    traces = [json.loads(full[case_id]) for case_id in routed_ids]
    routed_risks = np.asarray([[float(state["risk"]) for state in trace["risk_trajectory"]] for trace in traces])
    examples = [evidence_from_tool_calls(trace) for trace in traces]
    return cohort, routed_ids, routed_risks, policy_feature_matrices(examples, routed_risks), int(population)


def test_summary(cohort, routed_ids, routed_risks, routed_features, negative_population: int, models, threshold: float) -> dict:
    routed_scores, routed_costs, routed_depths = replay_value_models(models, routed_features, routed_risks, threshold)
    score_by_id = dict(zip(routed_ids, routed_scores, strict=True))
    labels = np.asarray([row[1] for row in cohort], dtype=np.int8)
    scores = np.asarray([score_by_id.get(int(row[0]), float(row[2])) for row in cohort])
    weights = np.where(labels == 1, 1.0, negative_population / int(np.count_nonzero(labels == 0)))
    unit_weights = np.ones(len(labels), dtype=float)
    exploratory_threshold = threshold_at_fpr(labels, scores, 0.01)
    return {
        "cases": len(cohort),
        "routed_cases": len(routed_ids),
        "unweighted": ranking_metrics(labels, scores),
        "population_weighted": ranking_metrics(labels, scores, sample_weight=weights),
        "frozen_operating_point_unweighted": operating_metrics(labels, scores, FROZEN_THRESHOLD, unit_weights),
        "frozen_operating_point_population_weighted": operating_metrics(labels, scores, FROZEN_THRESHOLD, weights),
        "exploratory_1pct_fpr_operating_point_unweighted": operating_metrics(
            labels, scores, exploratory_threshold, unit_weights
        ),
        "mean_cost_all_sampled_cases": float(routed_costs.sum() / len(cohort)),
        "mean_cost_routed_cases": float(routed_costs.mean()),
        "cost_saving_vs_full_evidence": float(1 - routed_costs.mean() / DEPTH_COSTS[-1]),
        "routed_stopping_depth_counts": {str(depth): int(np.count_nonzero(routed_depths == depth)) for depth in range(4)},
    }


def render_markdown(baselines: list[dict]) -> str:
    lines = [
        "# Learned Non-LLM Stopping Baselines",
        "",
        "Policies were fitted on train-split trajectories, selected on the 500-case validation cohort, and then frozen before deterministic replay on the representative held-out checkpoint.",
        "",
        "## Shared 500-case validation cohort",
        "",
        "| Method | Selected value threshold | PR-AUC | ROC-AUC | Recall @ 80% precision | Recall @ 90% precision | Recall @ 95% precision | Mean cost | Cost saving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline in baselines:
        val = baseline["validation"]
        metrics = val["metrics"]
        lines.append(
            f"| {baseline['label']} | {baseline['value_threshold']:.4g} | {metrics['pr_auc']:.3f} | {metrics['roc_auc']:.3f} | "
            f"{metrics['recall_at_80pct_precision']:.0%} | {metrics['recall_at_90pct_precision']:.0%} | "
            f"{metrics['recall_at_95pct_precision']:.0%} | {val['mean_cost']:.3f} | {val['cost_saving']:.1%} |"
        )
    lines.extend([
        "",
        "## Representative held-out sample",
        "",
        "| Method | Unweighted PR-AUC | Unweighted ROC-AUC | Population-weighted PR-AUC | Population-weighted ROC-AUC | Mean cost (all cases) | Routed-case cost saving |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for baseline in baselines:
        test = baseline["representative_test"]
        lines.append(
            f"| {baseline['label']} | {test['unweighted']['pr_auc']:.3f} | {test['unweighted']['roc_auc']:.3f} | "
            f"{test['population_weighted']['pr_auc']:.3f} | {test['population_weighted']['roc_auc']:.3f} | "
            f"{test['mean_cost_all_sampled_cases']:.3f} | {test['cost_saving_vs_full_evidence']:.1%} |"
        )
    return "\n".join([*lines, ""])


def render_weighted_test_markdown(baselines: list[dict], representative_report: Path) -> str:
    report = json.loads(representative_report.read_text(encoding="utf-8"))
    existing = {row["policy"]: row for row in report["methods"]}
    learned = {baseline["kind"]: baseline for baseline in baselines}
    full_cost = existing["full_evidence"]["mean_cost_all_cases"]
    rows = []
    for policy, label in (
        ("full_evidence", "Full evidence"),
        ("qwen3_4b_balanced", "Qwen3:4B balanced"),
    ):
        row = existing[policy]
        rows.append((label, row["pr_auc"], row["roc_auc"], row["precision"], row["recall"],
                     row["mean_cost_all_cases"], 1 - row["mean_cost_all_cases"] / full_cost))
    for kind in ("boosted", "cart"):
        baseline = learned[kind]
        test = baseline["representative_test"]
        frozen = test["frozen_operating_point_population_weighted"]
        rows.append((baseline["label"], test["population_weighted"]["pr_auc"],
                     test["population_weighted"]["roc_auc"], frozen["precision"], frozen["recall"],
                     test["mean_cost_all_sampled_cases"], test["cost_saving_vs_full_evidence"]))
    for policy, label in (
        ("aggressive_fixed_band", "Aggressive fixed-band"),
        ("moderate_structured", "Moderate structured"),
        ("transaction_only", "Transaction-only"),
    ):
        row = existing[policy]
        precision = None if policy == "transaction_only" and row["precision"] == 0 else row["precision"]
        rows.append((label, row["pr_auc"], row["roc_auc"], precision, row["recall"],
                     row["mean_cost_all_cases"], 1 - row["mean_cost_all_cases"] / full_cost))
    lines = [
        "# Representative Test — Population-Weighted Results",
        "",
        "All 3,286 test fraud cases and 200,000 seeded test non-fraud cases are included; non-fraud cases receive population weight 12.70882.",
        "",
        "| Method | PR-AUC | ROC-AUC | Frozen precision | Frozen recall | Mean cost | Cost saving vs full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, pr_auc, roc_auc, precision, recall, cost, saving in rows:
        precision_text = "—" if precision is None else f"{precision:.2%}"
        lines.append(
            f"| {label} | {pr_auc:.3f} | {roc_auc:.3f} | {precision_text} | {recall:.2%} | {cost:.3f} | {saving:.1%} |"
        )
    return "\n".join([*lines, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and replay non-LLM learned stopping baselines.")
    parser.add_argument("--database", type=Path, default=Path("data/evidence.sqlite"))
    parser.add_argument("--profiles", type=Path, default=Path("data/preauth_behavior.sqlite"))
    parser.add_argument("--split-database", type=Path, default=Path("data/iid_split.sqlite"))
    parser.add_argument("--validation-report", type=Path, default=Path("results/development/qwen3_4b_balanced_500_stratified.json"))
    parser.add_argument("--test-checkpoint", type=Path, default=Path("results/test/representative_test.sqlite"))
    parser.add_argument("--train-cache", type=Path, default=Path("results/development/learned_stopping_train_trajectories.npz"))
    parser.add_argument("--model-output", type=Path, default=Path("results/development/learned_stopping_models.joblib"))
    parser.add_argument("--output", type=Path, default=Path("results/development/learned_stopping_baselines.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("paper_results/learned_stopping_baselines.md"))
    parser.add_argument("--representative-report", type=Path,
                        default=Path("results/test/representative_test_report.json"))
    parser.add_argument("--weighted-markdown-output", type=Path,
                        default=Path("paper_results/representative_test_weighted_table.md"))
    parser.add_argument("--max-validation-pr-auc-loss", type=float, default=0.01,
                        help="largest permitted absolute validation PR-AUC loss versus full evidence")
    args = parser.parse_args()

    train_risks, train_labels, train_features = build_train_cache(args.database, args.profiles, args.split_database, args.train_cache)
    source, case_ids, validation_labels, validation_risks, validation_features = load_validation(
        args.validation_report, args.database, args.split_database
    )
    cohort, routed_ids, test_risks, test_features, negative_population = load_test(args.test_checkpoint)
    baselines = []
    frozen_models = {}
    for kind in MODEL_LABELS:
        selected, candidates = select_model(
            kind,
            train_risks,
            train_features,
            validation_risks,
            validation_features,
            validation_labels,
            args.max_validation_pr_auc_loss,
        )
        frozen_models[kind] = {
            "models": selected["models"],
            "value_threshold": selected["value_threshold"],
        }
        validation_path = args.output.parent / f"learned_{kind}_stopping_500_stratified.json"
        write_validation_report(validation_path, kind, source, case_ids, validation_risks, selected)
        baselines.append({
            "kind": kind,
            "label": MODEL_LABELS[kind],
            "value_threshold": selected["value_threshold"],
            "selection_constraint": {"max_pr_auc_loss_vs_full_evidence": args.max_validation_pr_auc_loss},
            "validation_report": str(validation_path),
            "validation": {
                "metrics": selected["metrics"],
                "mean_cost": selected["mean_cost"],
                "cost_saving": selected["cost_saving"],
                "stopping_depth_counts": {str(depth): int(np.count_nonzero(selected["depths"] == depth)) for depth in range(4)},
            },
            "validation_candidates": [
                {
                    "value_threshold": item["value_threshold"],
                    "metrics": item["metrics"],
                    "mean_cost": item["mean_cost"],
                    "cost_saving": item["cost_saving"],
                    "eligible": item["eligible"],
                }
                for item in candidates
            ],
            "representative_test": test_summary(
                cohort, routed_ids, test_risks, test_features, negative_population,
                selected["models"], selected["value_threshold"]
            ),
        })

    report = {
        "purpose": "train-fitted, validation-selected non-LLM adaptive acquisition baselines",
        "protocol": {
            "training_split": "train",
            "selection_split": "validation 500-case routed cohort",
            "final_evaluation": "saved representative held-out test trajectories",
            "features": "current and previously observed calibrated risk scores only",
            "learning_target": "absolute change from current risk to full-evidence risk",
            "stopping_rule": "stop when predicted value of additional evidence is below a validation-selected threshold",
            "test_outcomes_used_for_fitting_or_selection": False,
        },
        "training_routed_cases": len(train_labels),
        "training_routed_fraud_cases": int(train_labels.sum()),
        "baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"protocol": report["protocol"], "models": frozen_models}, args.model_output)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(baselines), encoding="utf-8")
    args.weighted_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.weighted_markdown_output.write_text(
        render_weighted_test_markdown(baselines, args.representative_report), encoding="utf-8"
    )
    print(json.dumps({
        "training_routed_cases": len(train_labels),
        "baselines": [
            {
                "kind": baseline["kind"],
                "value_threshold": baseline["value_threshold"],
                "validation": baseline["validation"],
                "representative_test": baseline["representative_test"],
            }
            for baseline in baselines
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
