#!/usr/bin/env python3
"""Summarize one paired Stage 24 non-formal optimizer pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results" / "stage24_rolling_optimizer" / "pilot" /
    "stage24_optimizer_smoke_20ep_seed72111000.json")
DEFAULT_OUTPUT = (
    ROOT / "results" / "stage24_rolling_optimizer" / "pilot" /
    "stage24_optimizer_paired_pilot_summary.json")
METRICS = {
    "episode_p95_waiting_time": ("p95_waiting_time", "lower"),
    "success_rate": ("success_rate", "higher"),
    "successful_throughput_tasks_per_hour": (
        "successful_throughput_tasks_per_hour", "higher"),
    "mean_waiting_time": ("mean_waiting_time", "lower"),
    "p95_flow_time": ("p95_flow_time", "lower"),
    "distance_per_task": ("distance_per_task", "lower"),
    "episode_reward": ("reward", "higher"),
}


def paired_bootstrap(
        optimizer: np.ndarray, baseline: np.ndarray, *, seed: int,
        samples: int) -> dict:
    differences = optimizer - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "optimizer": float(optimizer.mean()),
        "time_greedy": float(baseline.mean()),
        "optimizer_minus_time_greedy": float(differences.mean()),
        "paired_percentile_bootstrap_95ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "optimizer_better_stream_count": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-seed", type=int, default=72130000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("pilot gate did not pass")
    raw = payload["raw_results"]
    optimizer_episodes = raw["optimizer"]["episodes"]
    baseline_episodes = raw["time_greedy"]["episodes"]
    if len(optimizer_episodes) != len(baseline_episodes):
        raise RuntimeError("paired episode counts differ")
    if [row["seed"] for row in optimizer_episodes] != [
            row["seed"] for row in baseline_episodes]:
        raise RuntimeError("paired episode seeds differ")

    comparisons = {}
    for offset, (label, (field, direction)) in enumerate(METRICS.items()):
        optimizer = np.asarray(
            [row[field] for row in optimizer_episodes], dtype=np.float64)
        baseline = np.asarray(
            [row[field] for row in baseline_episodes], dtype=np.float64)
        row = paired_bootstrap(
            optimizer, baseline, seed=args.bootstrap_seed + offset,
            samples=args.bootstrap_samples)
        differences = optimizer - baseline
        row.update({
            "direction": direction,
            "optimizer_better_stream_count": int(np.sum(
                differences < 0.0 if direction == "lower" else
                differences > 0.0)),
            "tie_stream_count": int(np.sum(differences == 0.0)),
            "stream_count": len(differences),
        })
        comparisons[label] = row

    primary = comparisons["episode_p95_waiting_time"]
    latency = payload["optimizer_diagnostic_summary"]
    assertions = {
        "source_pilot_passed": payload["passed"] is True,
        "episode_pairing_complete": len(optimizer_episodes) == 20,
        "task_fingerprint_matched": payload["assertions"][
            "task_fingerprints_identical"],
        "handover_fingerprint_matched": payload["assertions"][
            "handover_fingerprints_identical"],
        "zero_illegal_actions": (
            raw["optimizer"]["illegal_action_count"] == 0),
        "zero_resource_leaks": (
            raw["optimizer"]["resource_leak_count"] == 0),
        "budget_compliance_at_least_99pct": (
            latency["budget_compliance_rate"] >= 0.99),
        "gap_semantics_complete": (
            latency["gap_reporting_complete_rate"] == 1.0),
    }
    report = {
        "schema_version": "warehouse_stage24_optimizer_paired_pilot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 24,
        "status": "PILOT_ONLY",
        "claim_boundary": (
            "Development evidence only. This pilot may select the compute "
            "budget but cannot support a formal performance claim."),
        "source": str(args.input.resolve()),
        "visibility_mode": payload["optimizer_config"]["visibility_mode"],
        "seed_range": payload["seed_range"],
        "episode_count": len(optimizer_episodes),
        "tasks_per_episode": payload["tasks_per_episode"],
        "bootstrap": {
            "method": "paired episode resampling",
            "seed": args.bootstrap_seed,
            "samples": args.bootstrap_samples,
        },
        "primary_endpoint": {
            "name": "mean episode-level P95 waiting time",
            **primary,
            "pilot_direction_supported": (
                primary["paired_percentile_bootstrap_95ci"][1] < 0.0),
        },
        "comparisons": comparisons,
        "compute": latency,
        "transport_modes": {
            "optimizer": raw["optimizer"]["transport_modes"],
            "time_greedy": raw["time_greedy"]["transport_modes"],
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
        "formal_test_status": "blocked_pending_visibility_decision",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
