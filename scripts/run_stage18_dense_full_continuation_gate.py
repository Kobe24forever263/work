#!/usr/bin/env python3
"""Decide whether ten DENSE Full seeds may proceed to DENSE ablations."""

from __future__ import annotations

import json
from pathlib import Path
import statistics


WORK_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = []
    for seed_index in range(1, 11):
        path = (WORK_ROOT / "results" / "stage18_dense" / "full_context_v2" /
                f"seed_{seed_index:02d}" / "stage18_ppo.summary.json")
        if not path.exists():
            raise FileNotFoundError(f"missing DENSE Full seed summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        policy, rule = summary["evaluation"], summary["rule_baseline"]
        rows.append({
            "seed_index": seed_index, "summary": str(path),
            "passed": summary["passed"],
            "arrival_profile": summary["arrival_profile"],
            "reward_delta": policy["mean_reward"] - rule["mean_reward"],
            "success_delta_pp": 100.0 * (
                policy["success_rate"] - rule["success_rate"]),
            "throughput_delta": policy["throughput_tasks_per_hour"] -
            rule["throughput_tasks_per_hour"],
            "illegal_action_count": policy["illegal_action_count"],
            "resource_leak_count": policy["resource_leak_count"],
            "transport_modes": policy["transport_modes"],
        })
    reward = [row["reward_delta"] for row in rows]
    success = [row["success_delta_pp"] for row in rows]
    assertions = {
        "all_runs_are_dense": all(row["arrival_profile"] == "DENSE"
                                  for row in rows),
        "all_ten_training_runs_passed": all(row["passed"] for row in rows),
        "all_ten_have_zero_illegal_actions": all(
            row["illegal_action_count"] == 0 for row in rows),
        "all_ten_have_zero_resource_leaks": all(
            row["resource_leak_count"] == 0 for row in rows),
        "all_ten_expose_all_transport_modes": all(
            set(row["transport_modes"]) == {
                "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"} for row in rows),
        "at_least_eight_of_ten_reward_deltas_positive": sum(
            value > 0 for value in reward) >= 8,
        "mean_reward_delta_positive": statistics.mean(reward) > 0,
        "mean_success_delta_not_negative": statistics.mean(success) >= 0,
    }
    report = {
        "stage": 18, "arrival_profile": "DENSE",
        "gate": "DENSE_FULL_TEN_SEED_ABLATION_CONTINUATION",
        "claim_boundary": "Validation continuation gate; not locked test.",
        "runs": rows,
        "aggregate": {
            "reward_delta_mean": statistics.mean(reward),
            "reward_delta_stdev": statistics.stdev(reward),
            "success_delta_pp_mean": statistics.mean(success),
            "success_delta_pp_stdev": statistics.stdev(success),
            "throughput_delta_mean": statistics.mean(
                row["throughput_delta"] for row in rows),
        },
        "assertions": assertions, "passed": all(assertions.values()),
    }
    output = (WORK_ROOT / "results" / "stage18_dense" /
              "stage18_dense_full_ten_seed_continuation_gate.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": report["aggregate"],
                      "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
