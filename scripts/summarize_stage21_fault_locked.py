#!/usr/bin/env python3
"""Audit and summarize the Stage 21 all-weight locked fault campaign."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage21_faults import (  # noqa: E402
    CONTROL, FAULT_SCENARIOS, NAVIGATION_FAILURE)


CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage21_fault_robustness.yaml")
CAMPAIGN = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_campaign" / "locked_v1")
OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_locked_summary.json")
EPISODE_METRICS = (
    "success_rate", "reward", "successful_throughput_tasks_per_hour",
    "mean_waiting_time", "p95_waiting_time", "mean_flow_time",
    "p95_flow_time", "distance_per_task")


def load_job(path: Path, *, method: str, fault: str,
             episodes: int, seed_start: int, weight_index=None) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row["result"]
    if not (
            row.get("schema_version") ==
            "warehouse_stage21_fault_locked_job_v1" and
            row.get("method") == method and
            row.get("fault_scenario") == fault and
            row.get("seed_start") == seed_start and
            row.get("episode_count") == episodes and
            row.get("training_seed_index") == weight_index and
            len(result.get("episodes", [])) == episodes and
            result.get("task_count") == episodes * 80 and
            result.get("completed", 0) + result.get("failed", 0) ==
            episodes * 80 and
            result.get("illegal_action_count") == 0 and
            result.get("resource_leak_count") == 0):
        raise RuntimeError(f"invalid locked result: {path}")
    return result


def episode_values(result: dict, metric: str) -> np.ndarray:
    return np.asarray(
        [episode[metric] for episode in result["episodes"]],
        dtype=np.float64)


def recovery_values(result: dict) -> np.ndarray:
    values = []
    for episode in result["episodes"]:
        schedule = episode.get("fault_schedule") or {}
        completions = sorted(
            task["finished_at"] for task in episode["tasks"]
            if task["task_result"] == "COMPLETED")
        samples = []
        for window in schedule.get("windows", []):
            end = float(window["end_s"])
            later = [value for value in completions if value >= end]
            if later:
                samples.append(later[0] - end)
        values.append(float(np.mean(samples)) if samples else np.nan)
    return np.asarray(values, dtype=np.float64)


def interval(samples: np.ndarray) -> list[float]:
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def crossed_difference(policy: np.ndarray, comparator: np.ndarray,
                       rng: np.random.Generator, samples: int) -> dict:
    if policy.ndim != 2 or comparator.shape != (policy.shape[1],):
        raise ValueError("crossed comparison shape mismatch")
    differences = policy - comparator[None, :]
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        train = rng.integers(0, policy.shape[0], policy.shape[0])
        test = rng.integers(0, policy.shape[1], policy.shape[1])
        draws[index] = differences[np.ix_(train, test)].mean()
    return {
        "difference": float(differences.mean()),
        "crossed_bootstrap_95ci": interval(draws),
        "training_seed_count": policy.shape[0],
        "test_seed_count": policy.shape[1],
    }


def crossed_paired(policy_a: np.ndarray, policy_b: np.ndarray,
                   rng: np.random.Generator, samples: int) -> dict:
    if policy_a.shape != policy_b.shape or policy_a.ndim != 2:
        raise ValueError("paired policy matrices must match")
    differences = policy_a - policy_b
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        train = rng.integers(0, differences.shape[0], differences.shape[0])
        test = rng.integers(0, differences.shape[1], differences.shape[1])
        draws[index] = differences[np.ix_(train, test)].mean()
    return {
        "difference": float(differences.mean()),
        "crossed_bootstrap_95ci": interval(draws),
        "training_seed_count": differences.shape[0],
        "test_seed_count": differences.shape[1],
    }


def paired_vector(a: np.ndarray, b: np.ndarray,
                  rng: np.random.Generator, samples: int) -> dict:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired baseline vectors must match")
    differences = a - b
    indices = rng.integers(0, len(differences), (samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "paired_bootstrap_95ci": interval(draws),
        "test_seed_count": len(differences),
    }


def absolute_summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": None, "std": None, "sample_count": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "sample_count": int(len(finite)),
    }


def event_count(result: dict, event_name: str) -> int:
    return sum(
        event.get("event") == event_name
        for episode in result["episodes"]
        for event in episode.get("fault_events", []))


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed_cfg = protocol["locked_test"]
    seed_start = int(seed_cfg["seed_start"])
    episodes = int(seed_cfg["seed_count"])
    bootstrap = protocol["bootstrap"]
    bootstrap_samples = int(bootstrap["samples"])
    rng = np.random.default_rng(int(bootstrap["seed"]))

    policies = {}
    baselines = {}
    for fault in FAULT_SCENARIOS:
        folder = CAMPAIGN / fault.lower()
        policies[fault] = [
            load_job(
                folder / f"ppo_seed_{index:02d}.json",
                method="PPO", fault=fault, episodes=episodes,
                seed_start=seed_start, weight_index=index)
            for index in range(1, 11)]
        baselines[fault] = {
            method: load_job(
                folder / f"{method.lower()}.json",
                method=method, fault=fault, episodes=episodes,
                seed_start=seed_start)
            for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY")}

    task_fingerprints = {
        result["task_fingerprint"]
        for fault in FAULT_SCENARIOS
        for result in (
            policies[fault] + list(baselines[fault].values()))}
    per_fault_shared_faults = {
        fault: len({
            result["fault_fingerprint"] for result in (
                policies[fault] + list(baselines[fault].values()))}) == 1
        for fault in FAULT_SCENARIOS}

    absolute = {}
    comparisons = {}
    degradation = {}
    for fault in FAULT_SCENARIOS:
        absolute[fault] = {}
        comparisons[fault] = {}
        policy_matrices = {
            metric: np.stack([
                episode_values(result, metric)
                for result in policies[fault]])
            for metric in EPISODE_METRICS}
        policy_recovery = np.stack([
            recovery_values(result) for result in policies[fault]])
        absolute[fault]["PPO"] = {
            metric: absolute_summary(values)
            for metric, values in policy_matrices.items()}
        absolute[fault]["PPO"]["recovery_time_s"] = absolute_summary(
            policy_recovery)

        for method, result in baselines[fault].items():
            absolute[fault][method] = {
                metric: absolute_summary(episode_values(result, metric))
                for metric in EPISODE_METRICS}
            absolute[fault][method]["recovery_time_s"] = absolute_summary(
                recovery_values(result))
            comparisons[fault][f"PPO_MINUS_{method}"] = {
                metric: crossed_difference(
                    policy_matrices[metric], episode_values(result, metric),
                    rng, bootstrap_samples)
                for metric in EPISODE_METRICS}
            if fault != CONTROL:
                recovery = recovery_values(result)
                valid = np.isfinite(recovery) & np.all(
                    np.isfinite(policy_recovery), axis=0)
                if valid.any():
                    comparisons[fault][f"PPO_MINUS_{method}"][
                        "recovery_time_s"] = crossed_difference(
                            policy_recovery[:, valid], recovery[valid],
                            rng, bootstrap_samples)

        degradation[fault] = {}
        if fault == CONTROL:
            continue
        for metric in EPISODE_METRICS:
            fault_matrix = policy_matrices[metric]
            control_matrix = np.stack([
                episode_values(result, metric)
                for result in policies[CONTROL]])
            degradation[fault].setdefault("PPO", {})[metric] = \
                crossed_paired(
                    fault_matrix, control_matrix, rng, bootstrap_samples)
        for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"):
            degradation[fault][method] = {
                metric: paired_vector(
                    episode_values(baselines[fault][method], metric),
                    episode_values(baselines[CONTROL][method], metric),
                    rng, bootstrap_samples)
                for metric in EPISODE_METRICS}

    failure_events = {
        fault: {
            "PPO": sum(event_count(result, "TASK_NAVIGATION_FAILURE")
                       for result in policies[fault]),
            **{
                method: event_count(result, "TASK_NAVIGATION_FAILURE")
                for method, result in baselines[fault].items()},
        } for fault in FAULT_SCENARIOS}
    window_starts = {
        fault: {
            "PPO": sum(event_count(result, "FAULT_WINDOW_START")
                       for result in policies[fault]),
            **{
                method: event_count(result, "FAULT_WINDOW_START")
                for method, result in baselines[fault].items()},
        } for fault in FAULT_SCENARIOS}
    assertions = {
        "all_40_policy_jobs_loaded": sum(
            len(items) for items in policies.values()) == 40,
        "all_8_baseline_jobs_loaded": sum(
            len(items) for items in baselines.values()) == 8,
        "all_methods_share_one_task_stream_family":
            len(task_fingerprints) == 1,
        "fault_schedule_shared_within_each_scenario":
            all(per_fault_shared_faults.values()),
        "control_has_no_fault_events": (
            failure_events[CONTROL]["PPO"] == 0 and
            window_starts[CONTROL]["PPO"] == 0 and
            all(failure_events[CONTROL][method] == 0 and
                window_starts[CONTROL][method] == 0
                for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"))),
        "navigation_failure_activated_for_all_methods": (
            failure_events[NAVIGATION_FAILURE]["PPO"] > 0 and
            all(failure_events[NAVIGATION_FAILURE][method] > 0
                for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"))),
        "all_outage_windows_observed": all(
            window_starts[fault]["PPO"] > 0 and
            all(window_starts[fault][method] > 0
                for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"))
            for fault in ("STAIR_OUTAGE", "ROBOT_OUTAGE")),
    }
    payload = {
        "schema_version": "warehouse_stage21_fault_locked_summary_v1",
        "stage": 21,
        "gate": "FAULT_ROBUSTNESS_ALL_WEIGHT_LOCKED_V1",
        "claim_boundary": protocol["claim_boundary"],
        "protocol": {
            "training_seed_count": 10,
            "test_seed_count": episodes,
            "test_seed_start": seed_start,
            "bootstrap_seed": int(bootstrap["seed"]),
            "bootstrap_samples": bootstrap_samples,
            "statistics": (
                "crossed training-seed x shared-test-seed bootstrap for PPO; "
                "paired test-seed bootstrap for baselines"),
        },
        "absolute": absolute,
        "comparisons": comparisons,
        "degradation_from_control": degradation,
        "fault_event_counts": failure_events,
        "fault_window_start_counts": window_starts,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "passed": payload["passed"],
        "assertions": assertions}, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
