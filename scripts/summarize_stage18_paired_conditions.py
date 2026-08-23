#!/usr/bin/env python3
"""Summarize paired Stage 18 validation results across ten training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import math


WORK_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mean_reward", "success_rate", "throughput_tasks_per_hour",
           "mean_simulated_time", "cost_reference_mode_agreement")


def percentile_interval(values: list[float], seed: int,
                        repetitions: int = 20000) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    means = sorted(statistics.mean(values[rng.randrange(count)]
                                   for _ in range(count))
                   for _ in range(repetitions))
    return [means[int(0.025 * repetitions)],
            means[int(0.975 * repetitions)]]


def load(profile_root: str, condition: str, seed_index: int) -> dict:
    path = (WORK_ROOT / "results" / profile_root / condition /
            f"seed_{seed_index:02d}" / "stage18_ppo.summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    rule = summary["rule_baseline"]
    scale = max(abs(float(rule["mean_reward"])), 10.0)
    relative_score = 100.0 * math.exp(
        (float(evaluation["mean_reward"]) - float(rule["mean_reward"])) /
        scale)
    return {
        "seed_index": seed_index,
        "summary": str(path),
        "passed": summary["passed"],
        "illegal_action_count": evaluation["illegal_action_count"],
        "resource_leak_count": evaluation["resource_leak_count"],
        "transport_modes": evaluation["transport_modes"],
        "positive_relative_score": relative_score,
        "rule_baseline_score": 100.0,
        **{metric: evaluation[metric] for metric in METRICS},
    }


def aggregate(rows: list[dict], bootstrap_seed: int) -> dict:
    result = {}
    for offset, metric in enumerate(METRICS):
        values = [row[metric] for row in rows]
        result[metric] = {
            "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values),
            "bootstrap_95_interval": percentile_interval(
                values, bootstrap_seed + offset),
        }
    result["transport_mode_totals"] = {
        mode: sum(row["transport_modes"].get(mode, 0) for row in rows)
        for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("comparison")
    parser.add_argument("--arrival-profile", choices=("MEDIUM", "DENSE"),
                        default="MEDIUM")
    args = parser.parse_args()
    profile_root = ("stage18" if args.arrival_profile == "MEDIUM" else
                    "stage18_dense")
    reference = [load(profile_root, args.reference, seed)
                 for seed in range(1, 11)]
    comparison = [load(profile_root, args.comparison, seed)
                  for seed in range(1, 11)]
    paired = {}
    for offset, metric in enumerate(METRICS):
        values = [reference[i][metric] - comparison[i][metric]
                  for i in range(10)]
        interval = percentile_interval(values, 44000100 + offset)
        paired[metric] = {
            "direction": "reference_minus_comparison",
            "per_seed": values,
            "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values),
            "bootstrap_95_interval": interval,
            "interval_excludes_zero": interval[0] > 0 or interval[1] < 0,
        }
    report = {
        "schema_version": "warehouse_stage18_paired_validation_v1",
        "arrival_profile": args.arrival_profile,
        "claim_boundary": (
            "Ten paired training seeds on the shared validation set; not the "
            "locked test-set result."),
        "reference_condition": args.reference,
        "comparison_condition": args.comparison,
        "training_seed_count": 10,
        "reference": aggregate(reference, 44000000),
        "comparison": aggregate(comparison, 44000020),
        "paired_reference_minus_comparison": paired,
        "quality": {
            "all_runs_passed": all(row["passed"] for row in reference + comparison),
            "illegal_action_count": sum(row["illegal_action_count"]
                                        for row in reference + comparison),
            "resource_leak_count": sum(row["resource_leak_count"]
                                       for row in reference + comparison),
        },
    }
    output = (WORK_ROOT / "results" / profile_root /
              f"paired_{args.reference}_vs_{args.comparison}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "quality": report["quality"],
                      "paired": paired}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
