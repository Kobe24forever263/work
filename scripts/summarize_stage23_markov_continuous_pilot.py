#!/usr/bin/env python3
"""Freeze the three-seed Stage 23 optimization pilot gate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "stage23_markov_continuous"
OUTPUT = BASE / "pilot" / "stage23_markov_continuous_pilot_gate.json"
RESUME_GATE = BASE / "resume_equivalence" / "stage23_resume_equivalence_gate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows = []
    for index in range(1, 4):
        path = BASE / "pilot" / f"seed_{index:02d}" / "stage23_ppo.summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        weight = Path(summary["final_weight"])
        evaluation = summary["evaluation"]
        rule = summary["rule_baseline"]
        rows.append({
            "seed_index": index,
            "experiment_run_seed": summary["experiment_run_seed"],
            "passed": summary["passed"],
            "updates": summary["updates"],
            "parameter_delta_l2": summary["parameter_delta_l2"],
            "success_rate": evaluation["success_rate"],
            "rule_success_rate": rule["success_rate"],
            "mean_reward": evaluation["mean_reward"],
            "rule_mean_reward": rule["mean_reward"],
            "successful_throughput_tasks_per_hour": evaluation[
                "successful_throughput_tasks_per_hour"],
            "rule_successful_throughput_tasks_per_hour": rule[
                "successful_throughput_tasks_per_hour"],
            "cost_reference_mode_agreement": evaluation[
                "cost_reference_mode_agreement"],
            "transport_modes": evaluation["transport_modes"],
            "illegal_action_count": evaluation["illegal_action_count"],
            "resource_leak_count": evaluation["resource_leak_count"],
            "normalizer_source": summary["reward_normalization"]["source"],
            "warm_start_source": summary["warm_start_source"],
            "weight": str(weight),
            "weight_sha256": sha256(weight),
        })
    resume = json.loads(RESUME_GATE.read_text(encoding="utf-8"))
    checks = {
        "all_three_runs_present_and_passed": (
            len(rows) == 3 and all(row["passed"] for row in rows)),
        "all_runs_have_50_updates": all(row["updates"] == 50 for row in rows),
        "all_parameters_changed": all(
            row["parameter_delta_l2"] > 0 for row in rows),
        "all_tasks_safe_and_resolved": all(
            row["illegal_action_count"] == 0 and
            row["resource_leak_count"] == 0 for row in rows),
        "all_runs_select_three_modes": all(
            set(row["transport_modes"]) == {
                "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
            for row in rows),
        "all_runs_use_corrected_normalizer": all(
            row["normalizer_source"] ==
            "reverse_variable_discounted_return_to_go_std_v2"
            for row in rows),
        "training_weights_are_unique": len({
            row["weight_sha256"] for row in rows}) == 3,
        "exact_resume_equivalence_passes": bool(resume["passed"]),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "warehouse_stage23_pilot_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 23,
        "gate": "THREE_SEED_OPTIMIZATION_PILOT",
        "claim_boundary": (
            "Optimization stability, safety and provenance only. The pilot "
            "is underpowered and is not a formal performance comparison."),
        "rows": rows,
        "descriptive_summary": {
            "mean_policy_success_rate": statistics.mean(
                row["success_rate"] for row in rows),
            "mean_rule_success_rate": statistics.mean(
                row["rule_success_rate"] for row in rows),
            "mean_policy_reward": statistics.mean(
                row["mean_reward"] for row in rows),
            "mean_rule_reward": statistics.mean(
                row["rule_mean_reward"] for row in rows),
            "mean_cost_reference_mode_agreement": statistics.mean(
                row["cost_reference_mode_agreement"] for row in rows),
        },
        "checks": checks,
        "passed": passed,
        "next_gate": (
            "MANUAL_TEN_SEED_FORMAL_TRAINING" if passed else
            "REPAIR_PILOT_OR_RESUME_GATE"),
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
