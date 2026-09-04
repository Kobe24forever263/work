#!/usr/bin/env python3
"""Audit and summarize the run-once Stage 23 held-out campaign."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results" / "stage23_markov_continuous" / "locked_test_v1")
CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_stage23_locked_test.yaml")
FREEZE = (
    ROOT / "results" / "stage23_markov_continuous" / "freeze" /
    "stage23_policy_freeze_manifest.json")
OUT = (
    ROOT / "results" / "stage23_markov_continuous" /
    "stage23_locked_test_summary.json")

METRICS = {
    "mean_episode_p95_waiting_time": ("p95_waiting_time", "lower"),
    "success_rate": ("success_rate", "higher"),
    "successful_throughput_tasks_per_hour": (
        "successful_throughput_tasks_per_hour", "higher"),
    "mean_waiting_time": ("mean_waiting_time", "lower"),
    "mean_flow_time": ("mean_flow_time", "lower"),
    "mean_episode_p95_flow_time": ("p95_flow_time", "lower"),
    "distance_per_task": ("distance_per_task", "lower"),
    "mean_episode_reward": ("reward", "higher"),
}
SECONDARY = (
    "mean_waiting_time", "mean_flow_time", "mean_episode_p95_flow_time",
    "distance_per_task", "mean_episode_reward",
)
BASELINE_KEYS = {
    "TIME_GREEDY_RULE": "time_greedy_rule",
    "SINGLE_DOG_ONLY": "single_dog_only",
}
PHASES = {"NORMAL", "DENSE", "BURST", "RECOVERY"}


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def episode_vectors(result: dict, expected_seeds: list[int],
                    reference_task: dict[int, str],
                    reference_handover: dict[int, str],
                    label: str) -> tuple[dict[str, list[float]], list[str]]:
    failures = []
    episodes = {int(row["seed"]): row for row in result.get("episodes", [])}
    if sorted(episodes) != expected_seeds:
        failures.append(f"{label}: locked seed set mismatch")
    vectors = {metric: [] for metric in METRICS}
    for seed in expected_seeds:
        episode = episodes.get(seed)
        if episode is None:
            continue
        tasks = episode.get("tasks", [])
        if episode.get("task_stream_fingerprint") != reference_task[seed]:
            failures.append(f"{label}: task fingerprint mismatch at {seed}")
        if (episode.get("handover_potential_fingerprint") !=
                reference_handover[seed]):
            failures.append(
                f"{label}: handover fingerprint mismatch at {seed}")
        if episode.get("task_count") != 80 or len(tasks) != 80:
            failures.append(f"{label}: expected 80 task rows at {seed}")
        if len({task.get("task_id") for task in tasks}) != 80:
            failures.append(f"{label}: duplicate task id at {seed}")
        if not all(task.get("task_result") in {"COMPLETED", "FAILED"}
                   for task in tasks):
            failures.append(f"{label}: missing terminal task result at {seed}")
        if (episode.get("completed", 0) + episode.get("failed", 0) != 80 or
                episode.get("illegal_actions") != 0 or
                episode.get("resource_leak") is not False):
            failures.append(f"{label}: safety/completeness at {seed}")
        if {task.get("phase") for task in tasks} != PHASES:
            failures.append(f"{label}: phase coverage at {seed}")
        for metric, (source_key, _) in METRICS.items():
            value = episode.get(source_key)
            if not finite(value):
                failures.append(f"{label}: non-finite {metric} at {seed}")
                value = math.nan
            vectors[metric].append(float(value))
    return vectors, failures


def pooled_task_p95(result: dict) -> float:
    waiting = [
        float(task["waiting_time"])
        for episode in result["episodes"] for task in episode["tasks"]]
    return float(np.percentile(waiting, 95))


def point_summary(values: np.ndarray) -> dict:
    flattened = values.reshape(-1)
    return {
        "mean": float(flattened.mean()),
        "training_seed_sd_of_test_seed_means": (
            float(values.mean(axis=1).std(ddof=1))
            if values.shape[0] > 1 else 0.0),
        "test_stream_count_per_training_seed": int(values.shape[1]),
        "training_seed_count": int(values.shape[0]),
    }


def crossed_bootstrap(
        policy: dict[str, np.ndarray], baselines: dict[str, dict[str, np.ndarray]],
        *, samples: int, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    train_count, test_count = next(iter(policy.values())).shape
    train_indices = rng.integers(
        0, train_count, size=(samples, train_count))
    shared_test_indices = rng.integers(
        0, test_count, size=(samples, test_count))
    policy_draws = {}
    for metric, values in policy.items():
        train_resampled = values[train_indices].mean(axis=1)
        policy_draws[metric] = np.take_along_axis(
            train_resampled, shared_test_indices, axis=1).mean(axis=1)
    baseline_draws = {
        name: {
            metric: values[shared_test_indices].mean(axis=1)
            for metric, values in metrics.items()}
        for name, metrics in baselines.items()}
    return policy_draws, baseline_draws


def comparison(point_policy: float, point_baseline: float,
               policy_draws: np.ndarray, baseline_draws: np.ndarray,
               direction: str) -> dict:
    delta = policy_draws - baseline_draws
    point_delta = point_policy - point_baseline
    if direction == "lower":
        probability_better = float(np.mean(delta < 0))
        superiority = float(np.percentile(delta, 97.5)) < 0
    else:
        probability_better = float(np.mean(delta > 0))
        superiority = float(np.percentile(delta, 2.5)) > 0
    nonpositive = (np.count_nonzero(delta <= 0) + 1) / (len(delta) + 1)
    nonnegative = (np.count_nonzero(delta >= 0) + 1) / (len(delta) + 1)
    return {
        "policy_minus_baseline": point_delta,
        "crossed_bootstrap_95ci": [
            float(np.percentile(delta, 2.5)),
            float(np.percentile(delta, 97.5))],
        "direction_better": direction,
        "probability_policy_better": probability_better,
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(
            nonpositive, nonnegative)),
        "superiority_supported_by_95ci": superiority,
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * raw[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    locked = protocol["locked_test"]
    seed_start = int(locked["seed_start"])
    test_count = int(locked["seed_count"])
    expected_seeds = list(range(seed_start, seed_start + test_count))
    failures: list[str] = []

    status_path = RESULT_ROOT / "campaign.status.json"
    baseline_path = RESULT_ROOT / "baselines.json"
    if not status_path.exists() or not baseline_path.exists():
        raise FileNotFoundError("locked campaign is not complete")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    baselines_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    rule_episodes = baselines_payload["time_greedy_rule"]["episodes"]
    reference_task = {
        int(row["seed"]): row["task_stream_fingerprint"]
        for row in rule_episodes}
    reference_handover = {
        int(row["seed"]): row["handover_potential_fingerprint"]
        for row in rule_episodes}
    if sorted(reference_task) != expected_seeds:
        failures.append("baseline reference seed set mismatch")

    policy_paths = sorted((RESULT_ROOT / "policy").glob("seed_*.json"))
    policy_rows = []
    policy_vectors = []
    pooled_policy_p95 = []
    mode_counts = Counter()
    mode_by_phase = Counter()
    for path in policy_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        vectors, row_failures = episode_vectors(
            result, expected_seeds, reference_task, reference_handover,
            path.name)
        failures.extend(row_failures)
        policy_vectors.append(vectors)
        pooled_policy_p95.append(pooled_task_p95(result))
        mode_counts.update(result.get("transport_modes", {}))
        for phase, counts in result.get("phase_results", {}).items():
            for mode, count in counts.get("transport_modes", {}).items():
                mode_by_phase[(phase, mode)] += int(count)
        policy_rows.append({
            "training_seed_index": payload["training_seed_index"],
            "selected_checkpoint_update":
            payload["selected_checkpoint_update"],
            "weight_sha256": payload["weight_sha256"],
            "pooled_task_p95_waiting_time": pooled_policy_p95[-1],
        })

    baseline_vectors = {}
    pooled_baseline_p95 = {}
    for name, key in BASELINE_KEYS.items():
        result = baselines_payload[key]
        vectors, row_failures = episode_vectors(
            result, expected_seeds, reference_task, reference_handover, name)
        failures.extend(row_failures)
        baseline_vectors[name] = {
            metric: np.asarray(values, dtype=np.float64)
            for metric, values in vectors.items()}
        pooled_baseline_p95[name] = pooled_task_p95(result)

    policy_arrays = {
        metric: np.asarray(
            [row[metric] for row in policy_vectors], dtype=np.float64)
        for metric in METRICS}
    policy_points = {
        metric: point_summary(values)
        for metric, values in policy_arrays.items()}
    baseline_points = {
        name: {
            metric: {"mean": float(values.mean())}
            for metric, values in metrics.items()}
        for name, metrics in baseline_vectors.items()}

    stats = protocol["statistics"]
    policy_draws, baseline_draws = crossed_bootstrap(
        policy_arrays, baseline_vectors,
        samples=int(stats["bootstrap_samples"]),
        seed=int(stats["bootstrap_seed"]))
    comparisons = {}
    for baseline_name in BASELINE_KEYS:
        rows = {
            metric: comparison(
                policy_points[metric]["mean"],
                baseline_points[baseline_name][metric]["mean"],
                policy_draws[metric], baseline_draws[baseline_name][metric],
                direction)
            for metric, (_, direction) in METRICS.items()}
        adjusted = holm_adjust({
            metric: rows[metric]["two_sided_bootstrap_p"]
            for metric in SECONDARY})
        for metric, value in adjusted.items():
            rows[metric]["holm_adjusted_p_within_secondary_family"] = value
        comparisons[f"STAGE23_POLICY_vs_{baseline_name}"] = rows

    total_modes = max(sum(mode_counts.values()), 1)
    phase_modes = {}
    for phase in sorted(PHASES):
        count = sum(
            mode_by_phase[(phase, mode)]
            for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"))
        phase_modes[phase] = {
            "count": count,
            "shares": {
                mode: mode_by_phase[(phase, mode)] / max(count, 1)
                for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")},
        }

    primary_key = stats["primary_estimand"]["id"]
    qa = {
        "campaign_completed": status.get("state") == "COMPLETED",
        "freeze_gate_passed": freeze.get("hard_gate", {}).get("passed") is True,
        "analysis_protocol_hash_matches_freeze":
        freeze.get("analysis_protocol", {}).get("sha256") == sha256(CONFIG),
        "all_10_frozen_policy_files_present": len(policy_paths) == 10,
        "all_100_shared_locked_seeds_present":
        len(reference_task) == test_count,
        "all_task_fingerprints_match": not any(
            "task fingerprint mismatch" in item for item in failures),
        "all_handover_fingerprints_match": not any(
            "handover fingerprint mismatch" in item for item in failures),
        "no_data_quality_failure": not failures,
        "primary_endpoint_predeclared":
        primary_key == "mean_episode_p95_waiting_time",
        "noninferiority_claim_disabled_without_margin":
        stats["noninferiority_claim_allowed"] is False,
    }
    report = {
        "schema_version": "warehouse_stage23_locked_summary_v1",
        "stage": 23,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Run-once held-out logical-simulator comparison of all ten "
            "frozen training seeds. Guardrails are descriptive because no "
            "noninferiority margins were preregistered."),
        "analysis_protocol": str(CONFIG),
        "analysis_protocol_sha256": sha256(CONFIG),
        "freeze_manifest": str(FREEZE),
        "test_seed_range": [expected_seeds[0], expected_seeds[-1]],
        "training_seed_count": len(policy_paths),
        "test_stream_count": test_count,
        "tasks_per_stream": 80,
        "bootstrap": {
            "design": stats["design"],
            "samples": int(stats["bootstrap_samples"]),
            "seed": int(stats["bootstrap_seed"]),
        },
        "metric_definitions": {
            metric: {"source_episode_field": source, "direction": direction}
            for metric, (source, direction) in METRICS.items()},
        "estimand_note": (
            "Primary P95 is computed within each 80-task stream and then "
            "averaged across shared streams and training seeds. Pooled-task "
            "P95 is a sensitivity estimate, not 80000 independent samples."),
        "quality_assurance": qa,
        "quality_failures": failures,
        "frozen_policy_rows": policy_rows,
        "policy_point_estimates": policy_points,
        "baseline_point_estimates": baseline_points,
        "pooled_task_p95_waiting_time_sensitivity": {
            "policy_mean_across_training_seeds":
            float(np.mean(pooled_policy_p95)),
            "policy_training_seed_sd":
            float(np.std(pooled_policy_p95, ddof=1)),
            "baselines": pooled_baseline_p95,
        },
        "policy_transport_modes": {
            "counts": dict(mode_counts),
            "shares": {
                mode: count / total_modes
                for mode, count in mode_counts.items()},
            "by_phase": phase_modes,
        },
        "comparisons": comparisons,
        "primary_endpoint": {
            "id": primary_key,
            "comparisons": {
                name: rows[primary_key]
                for name, rows in comparisons.items()},
        },
        "guardrail_interpretation": {
            "mode": stats["guardrail_inference"],
            "noninferiority_claim_allowed": False,
            "reason": stats["noninferiority_reason"],
        },
        "passed": all(qa.values()),
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "passed": report["passed"],
        "quality_assurance": qa,
        "policy_point_estimates": policy_points,
        "baseline_point_estimates": baseline_points,
        "primary_endpoint": report["primary_endpoint"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
