#!/usr/bin/env python3
"""Verify that Stage 16 raw rows independently reproduce saved aggregates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "warehouse_stage16_evaluation_v1"
METHODS = ("policy", "rule_baseline", "risk_aware_rule_baseline")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else 0.0


def close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-9,
                        abs_tol=1e-9)


def recompute(raw_method: dict) -> dict:
    episodes = raw_method["episodes"]
    tasks = raw_method["tasks"]
    waiting = [row["waiting_time"] for row in tasks]
    flow = [row["flow_time"] for row in tasks]
    completed = sum(row["task_result"] == "COMPLETED" for row in tasks)
    modes = Counter(row["transport_mode"] for row in tasks)
    total_time = sum(row["simulated_time"] for row in episodes)
    total_distance = sum(
        row["distance_per_task"] * row["task_count"] for row in episodes)
    return {
        "episode_count": len(episodes),
        "task_count": len(tasks),
        "completed": completed,
        "failed": len(tasks) - completed,
        "success_rate": completed / max(len(tasks), 1),
        "mean_episode_reward": float(np.mean(
            [row["reward"] for row in episodes])),
        "throughput_tasks_per_hour": (
            len(tasks) / max(total_time, 1e-9) * 3600),
        "successful_throughput_tasks_per_hour": (
            completed / max(total_time, 1e-9) * 3600),
        "mean_waiting_time": float(np.mean(waiting)),
        "p95_waiting_time": percentile(waiting, 95),
        "mean_flow_time": float(np.mean(flow)),
        "p95_flow_time": percentile(flow, 95),
        "distance_per_task": total_distance / max(len(tasks), 1),
        "transport_modes": dict(modes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    raw_path = Path(summary["raw_results"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "summary_schema_matches": (
            summary.get("schema_version") == SCHEMA_VERSION),
        "raw_schema_matches": raw.get("schema_version") == SCHEMA_VERSION,
        "execution_mode_matches": (
            summary["execution_mode"] == raw["execution_mode"]),
    }
    comparisons = {}
    for scene, raw_scene in raw["scenarios"].items():
        summary_scene = summary["results"][scene]
        fingerprints_by_method = {}
        for method in METHODS:
            raw_method = raw_scene[method]
            expected = summary_scene[method]
            actual = recompute(raw_method)
            method_checks = {}
            for key, value in actual.items():
                expected_value = expected[key]
                method_checks[key] = (
                    value == expected_value if isinstance(value, (dict, int))
                    else close(value, expected_value))
            flattened_count = sum(
                len(episode["tasks"]) for episode in raw_method["episodes"])
            method_checks["flat_and_nested_task_rows_match"] = (
                flattened_count == len(raw_method["tasks"]))
            method_checks["episode_task_counts_match"] = all(
                len(episode["tasks"]) == episode["task_count"]
                for episode in raw_method["episodes"])
            method_checks["episode_metrics_recompute"] = all(
                close(np.mean([task["waiting_time"]
                               for task in episode["tasks"]]),
                      episode["mean_waiting_time"]) and
                close(percentile([task["waiting_time"]
                                  for task in episode["tasks"]], 95),
                      episode["p95_waiting_time"]) and
                close(np.mean([task["flow_time"]
                               for task in episode["tasks"]]),
                      episode["mean_flow_time"]) and
                close(percentile([task["flow_time"]
                                  for task in episode["tasks"]], 95),
                      episode["p95_flow_time"])
                for episode in raw_method["episodes"])
            fingerprints_by_method[method] = {
                episode["seed"]: episode["task_stream_fingerprint"]
                for episode in raw_method["episodes"]}
            comparisons[f"{scene}.{method}"] = {
                "recomputed": actual,
                "checks": method_checks,
            }
            checks[f"{scene}.{method}.all_aggregates_recomputed"] = all(
                method_checks.values())
        checks[f"{scene}.paired_task_streams_identical"] = (
            fingerprints_by_method["policy"] ==
            fingerprints_by_method["rule_baseline"] ==
            fingerprints_by_method["risk_aware_rule_baseline"])

    report = {
        "stage": 16,
        "gate": "RAW_RESULT_SCHEMA_RECOMPUTATION",
        "summary": str(args.summary.resolve()),
        "raw_results": str(raw_path.resolve()),
        "schema_version": SCHEMA_VERSION,
        "comparisons": comparisons,
        "assertions": checks,
        "passed": all(checks.values()),
    }
    output = args.output or args.summary.with_suffix(".schema_gate.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "assertion_count": len(checks),
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
