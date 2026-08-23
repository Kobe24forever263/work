#!/usr/bin/env python3
"""Quality-check and summarize the Stage 19 crossed locked test."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/stage19_mixed_recovery/locked_test_v1"
OUTPUT = ROOT / "results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json"
PROFILES = ("MEDIUM", "DENSE", "BURST")
PHASES = ("NORMAL", "DENSE", "BURST", "RECOVERY")
METRICS = (
    "success_rate",
    "reward",
    "successful_throughput_tasks_per_hour",
    "mean_waiting_time",
    "p95_waiting_time",
    "mean_flow_time",
    "distance_per_task",
)
BOOTSTRAP_SEED = 57000000
BOOTSTRAP_SAMPLES = 10000


def percentile_interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def episode_metrics(result: dict) -> dict[str, np.ndarray]:
    episodes = result["episodes"]
    return {
        "success_rate": np.asarray([row["success_rate"] for row in episodes]),
        "reward": np.asarray([row["reward"] for row in episodes]),
        "successful_throughput_tasks_per_hour": np.asarray([
            row["successful_throughput_tasks_per_hour"] for row in episodes]),
        "mean_waiting_time": np.asarray([
            row["mean_waiting_time"] for row in episodes]),
        "p95_waiting_time": np.asarray([
            row["p95_waiting_time"] for row in episodes]),
        "mean_flow_time": np.asarray([
            row["mean_flow_time"] for row in episodes]),
        "distance_per_task": np.asarray([
            row["distance_per_task"] for row in episodes]),
    }


def episode_adaptation(result: dict) -> dict[str, np.ndarray]:
    phase_cdc = {phase: [] for phase in PHASES}
    recovery_delta = []
    for episode in result["episodes"]:
        for phase in PHASES:
            phase_cdc[phase].append(
                episode["transport_modes_by_phase"].get(phase, {}).get(
                    "CAR_DOG_CAR", 0) / 20.0)
        recovery = sorted(
            (row for row in episode["tasks"] if row["phase"] == "RECOVERY"),
            key=lambda row: row["arrival_time"])
        early = np.mean([row["waiting_time"] for row in recovery[:10]])
        late = np.mean([row["waiting_time"] for row in recovery[10:]])
        recovery_delta.append(late - early)
    return {
        **{f"cdc_share_{phase.lower()}": np.asarray(values)
           for phase, values in phase_cdc.items()},
        "cdc_share_burst_minus_normal": (
            np.asarray(phase_cdc["BURST"]) - np.asarray(phase_cdc["NORMAL"])),
        "cdc_share_recovery_minus_burst": (
            np.asarray(phase_cdc["RECOVERY"]) - np.asarray(phase_cdc["BURST"])),
        "recovery_wait_late_minus_early": np.asarray(recovery_delta),
    }


def crossed_bootstrap_difference(policy: np.ndarray, baseline: np.ndarray,
                                 rng: np.random.Generator) -> dict:
    training_count, test_count = policy.shape
    samples = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        training = rng.integers(0, training_count, training_count)
        tests = rng.integers(0, test_count, test_count)
        samples[index] = (policy[np.ix_(training, tests)].mean() -
                          baseline[tests].mean())
    estimate = float(policy.mean() - baseline.mean())
    return {
        "estimate": estimate,
        "ci95": percentile_interval(samples),
        "direction_consistency_across_training_seeds": float(np.mean(
            np.sign(policy.mean(axis=1) - baseline.mean()) == np.sign(estimate))),
    }


def crossed_bootstrap_one_sample(values: np.ndarray,
                                 rng: np.random.Generator) -> dict:
    training_count, test_count = values.shape
    samples = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        training = rng.integers(0, training_count, training_count)
        tests = rng.integers(0, test_count, test_count)
        samples[index] = values[np.ix_(training, tests)].mean()
    return {"estimate": float(values.mean()), "ci95": percentile_interval(samples)}


def crossed_bootstrap_profiles(left: np.ndarray, right: np.ndarray,
                               rng: np.random.Generator) -> dict:
    left_training, test_count = left.shape
    right_training = right.shape[0]
    samples = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        left_rows = rng.integers(0, left_training, left_training)
        right_rows = rng.integers(0, right_training, right_training)
        tests = rng.integers(0, test_count, test_count)
        samples[index] = (
            left[np.ix_(left_rows, tests)].mean() -
            right[np.ix_(right_rows, tests)].mean())
    return {
        "estimate": float(left.mean() - right.mean()),
        "ci95": percentile_interval(samples),
        "interpretation": "exploratory; left minus right",
    }


def load_inputs():
    policies = {}
    raw_rows = {}
    for profile in PROFILES:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
            (INPUT / profile.lower()).glob("seed_*.json"))]
        if len(rows) != 10:
            raise RuntimeError(f"{profile}: expected 10 model files, got {len(rows)}")
        raw_rows[profile] = rows
        policies[profile] = {
            "metrics": {metric: np.stack([
                episode_metrics(row["result"])[metric] for row in rows])
                for metric in METRICS},
            "adaptation": {metric: np.stack([
                episode_adaptation(row["result"])[metric] for row in rows])
                for metric in episode_adaptation(rows[0]["result"])},
        }
    baseline_payload = json.loads(
        (INPUT / "baselines.json").read_text(encoding="utf-8"))
    baselines = {}
    for name in ("time_greedy_rule", "single_dog_only"):
        result = baseline_payload[name]["result"]
        baselines[name] = {
            "result": result,
            "metrics": episode_metrics(result),
            "adaptation": episode_adaptation(result),
        }
    return raw_rows, policies, baselines


def data_quality(raw_rows: dict, baselines: dict) -> dict:
    failures = []
    task_keys = set()
    duplicate_task_keys = 0
    policy_episode_count = 0
    policy_task_count = 0
    illegal_actions = 0
    resource_leaks = 0
    unresolved = 0
    task_fingerprints = {}
    method_failure_reasons = {}
    for profile, rows in raw_rows.items():
        reasons = Counter()
        for row in rows:
            result = row["result"]
            policy_episode_count += len(result["episodes"])
            policy_task_count += result["task_count"]
            illegal_actions += result["illegal_action_count"]
            resource_leaks += result["resource_leak_count"]
            reasons.update(result["failure_reasons"])
            for episode in result["episodes"]:
                eval_seed = episode["seed"]
                task_fingerprints.setdefault(eval_seed, set()).add(
                    episode["task_stream_fingerprint"])
                unresolved += episode.get("unresolved_at_terminal", 0)
                if episode["task_count"] != 80 or len(episode["tasks"]) != 80:
                    failures.append(
                        f"{profile}/train{row['training_seed_index']}/eval{eval_seed}: "
                        "not exactly 80 accounted tasks")
                if set(episode["transport_modes_by_phase"]) != set(PHASES):
                    failures.append(
                        f"{profile}/train{row['training_seed_index']}/eval{eval_seed}: "
                        "phase coverage mismatch")
                for task in episode["tasks"]:
                    key = (profile, row["training_seed_index"], eval_seed,
                           task["task_id"])
                    if key in task_keys:
                        duplicate_task_keys += 1
                    task_keys.add(key)
        method_failure_reasons[profile] = dict(reasons)
    baseline_task_count = 0
    baseline_task_keys = set()
    duplicate_baseline_task_keys = 0
    for name, item in baselines.items():
        result = item["result"]
        baseline_task_count += result["task_count"]
        illegal_actions += result["illegal_action_count"]
        resource_leaks += result["resource_leak_count"]
        method_failure_reasons[name] = result["failure_reasons"]
        for episode in result["episodes"]:
            task_fingerprints.setdefault(episode["seed"], set()).add(
                episode["task_stream_fingerprint"])
            if episode["task_count"] != 80 or len(episode["tasks"]) != 80:
                failures.append(f"{name}/eval{episode['seed']}: incomplete accounting")
            for task in episode["tasks"]:
                key = (name, episode["seed"], task["task_id"])
                if key in baseline_task_keys:
                    duplicate_baseline_task_keys += 1
                baseline_task_keys.add(key)
    fingerprint_mismatches = sum(len(values) != 1
                                 for values in task_fingerprints.values())
    if policy_episode_count != 3000:
        failures.append(f"policy episode count {policy_episode_count} != 3000")
    if policy_task_count != 240000:
        failures.append(f"policy task count {policy_task_count} != 240000")
    if baseline_task_count != 16000:
        failures.append(f"baseline task count {baseline_task_count} != 16000")
    if duplicate_task_keys:
        failures.append(f"duplicate policy task keys: {duplicate_task_keys}")
    if duplicate_baseline_task_keys:
        failures.append(
            f"duplicate baseline task keys: {duplicate_baseline_task_keys}")
    if illegal_actions:
        failures.append(f"illegal actions: {illegal_actions}")
    if resource_leaks:
        failures.append(f"resource leaks: {resource_leaks}")
    if fingerprint_mismatches:
        failures.append(f"task-stream fingerprint mismatches: {fingerprint_mismatches}")
    return {
        "intended_grain": "(method, training_seed_if_PPO, test_seed, task_id)",
        "policy_model_files": sum(len(rows) for rows in raw_rows.values()),
        "policy_episodes": policy_episode_count,
        "policy_tasks": policy_task_count,
        "baseline_episodes": 200,
        "baseline_tasks": baseline_task_count,
        "unique_policy_task_keys": len(task_keys),
        "duplicate_policy_task_keys": duplicate_task_keys,
        "unique_baseline_task_keys": len(baseline_task_keys),
        "duplicate_baseline_task_keys": duplicate_baseline_task_keys,
        "test_seed_count": len(task_fingerprints),
        "task_stream_fingerprint_mismatches": fingerprint_mismatches,
        "illegal_action_count": illegal_actions,
        "resource_leak_count": resource_leaks,
        "policy_terminal_unresolved_count": unresolved,
        "failure_reasons_by_method": method_failure_reasons,
        "severity": "PASS" if not failures else "HIGH",
        "safe_for_formal_analysis": not failures,
        "failures": failures,
    }


def main() -> int:
    raw_rows, policies, baselines = load_inputs()
    quality = data_quality(raw_rows, baselines)
    if not quality["safe_for_formal_analysis"]:
        raise RuntimeError(f"Stage 19 data quality failed: {quality['failures']}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    profile_summary = {}
    comparisons = {}
    for profile in PROFILES:
        profile_summary[profile] = {
            "metrics": {
                metric: {
                    "mean": float(values.mean()),
                    "training_seed_sd": float(values.mean(axis=1).std(ddof=1)),
                }
                for metric, values in policies[profile]["metrics"].items()
            },
            "adaptation": {
                metric: crossed_bootstrap_one_sample(values, rng)
                for metric, values in policies[profile]["adaptation"].items()
            },
        }
        comparisons[profile] = {}
        for baseline_name, baseline in baselines.items():
            comparisons[profile][baseline_name] = {
                metric: crossed_bootstrap_difference(
                    policies[profile]["metrics"][metric],
                    baseline["metrics"][metric], rng)
                for metric in METRICS
            }
    baseline_summary = {
        name: {
            "metrics": {metric: float(values.mean())
                        for metric, values in item["metrics"].items()},
            "adaptation": {metric: float(values.mean())
                           for metric, values in item["adaptation"].items()},
        }
        for name, item in baselines.items()
    }
    profile_pairwise = {}
    for left, right in (("MEDIUM", "DENSE"), ("MEDIUM", "BURST"),
                        ("DENSE", "BURST")):
        profile_pairwise[f"{left}_minus_{right}"] = {
            metric: crossed_bootstrap_profiles(
                policies[left]["metrics"][metric],
                policies[right]["metrics"][metric], rng)
            for metric in METRICS
        }
    payload = {
        "schema_version": "warehouse_stage19_mixed_recovery_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Fresh locked-test evidence. Profile comparisons are exploratory; "
            "PPO-vs-baseline comparisons use crossed training/test-seed bootstrap."),
        "protocol": {
            "test_seed_range": [56000000, 56000099],
            "training_seeds_per_profile": 10,
            "episodes_per_model": 100,
            "tasks_per_episode": 80,
            "phases": list(PHASES),
            "state_reset_at_phase_boundary": False,
            "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_design": "crossed training-seed by shared test-seed",
        },
        "data_quality": quality,
        "profiles": profile_summary,
        "baselines": baseline_summary,
        "comparisons": comparisons,
        "profile_pairwise_exploratory": profile_pairwise,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "quality_passed": quality["safe_for_formal_analysis"],
        "policy_tasks": quality["policy_tasks"],
        "baseline_tasks": quality["baseline_tasks"],
        "profiles": {
            profile: {
                metric: round(profile_summary[profile]["metrics"][metric]["mean"], 6)
                for metric in ("success_rate", "reward",
                               "successful_throughput_tasks_per_hour",
                               "mean_waiting_time")}
            for profile in PROFILES},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
