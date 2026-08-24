#!/usr/bin/env python3
"""Audit and summarize the Stage 20 held-out mixed-curriculum campaign."""

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
    ROOT / "results" / "stage20_mixed_curriculum" / "locked_test_v1")
CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage20_mixed_curriculum.yaml")
FREEZE = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json")
OUT = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_locked_test_summary.json")

METRICS = {
    "success_rate": ("success_rate", "higher"),
    "reward": ("reward", "higher"),
    "mean_episode_successful_throughput_tasks_per_hour": (
        "successful_throughput_tasks_per_hour", "higher"),
    "mean_waiting_time": ("mean_waiting_time", "lower"),
    "mean_episode_p95_waiting_time": ("p95_waiting_time", "lower"),
    "mean_flow_time": ("mean_flow_time", "lower"),
    "mean_episode_p95_flow_time": ("p95_flow_time", "lower"),
    "distance_per_task": ("distance_per_task", "lower"),
}
RUN_LEVEL_METRICS = {
    "success_rate": "success_rate",
    "reward": "mean_episode_reward",
    "successful_throughput_tasks_per_hour": (
        "successful_throughput_tasks_per_hour"),
    "mean_waiting_time": "mean_waiting_time",
    "pooled_task_p95_waiting_time": "p95_waiting_time",
    "mean_flow_time": "mean_flow_time",
    "pooled_task_p95_flow_time": "p95_flow_time",
    "distance_per_task": "distance_per_task",
}
COHORT_PATHS = {
    "STAGE20_MIXED_CURRICULUM": (
        RESULT_ROOT / "stage20_mixed_curriculum" / "mixed_curriculum"),
    "STAGE19_MEDIUM": (
        RESULT_ROOT / "stage19_fixed_profile" / "medium"),
    "STAGE19_DENSE": (
        RESULT_ROOT / "stage19_fixed_profile" / "dense"),
    "STAGE19_BURST": (
        RESULT_ROOT / "stage19_fixed_profile" / "burst"),
}


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def episode_vector(episode: dict, metric: str) -> float:
    source_key = METRICS[metric][0]
    return float(episode[source_key])


def run_level_values(result: dict) -> dict[str, float]:
    return {
        metric: float(result[source_key])
        for metric, source_key in RUN_LEVEL_METRICS.items()}


def infer_phase_order(tasks: list[dict]) -> tuple[str, ...]:
    first_arrival: dict[str, float] = {}
    for task in tasks:
        phase = task["phase"]
        first_arrival[phase] = min(
            first_arrival.get(phase, float("inf")), task["arrival_time"])
    return tuple(sorted(first_arrival, key=first_arrival.get))


def audit_and_extract(path: Path, reference_fingerprints: dict[int, str],
                      expected_seeds: list[int]) -> tuple[dict, list[str]]:
    failures: list[str] = []
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row["result"]
    episodes = result["episodes"]
    if len(episodes) != len(expected_seeds):
        failures.append(f"{path}: episode count {len(episodes)}")
    episode_by_seed = {int(episode["seed"]): episode for episode in episodes}
    if sorted(episode_by_seed) != expected_seeds:
        failures.append(f"{path}: test seed set mismatch")
    vectors = {metric: [] for metric in METRICS}
    phase_orders = Counter()
    mode_counts = Counter()
    phase_mode_counts = Counter()
    for seed in expected_seeds:
        episode = episode_by_seed.get(seed)
        if episode is None:
            continue
        tasks = episode.get("tasks", [])
        fingerprints_match = (
            episode.get("task_stream_fingerprint") ==
            reference_fingerprints[seed])
        if not fingerprints_match:
            failures.append(f"{path}: fingerprint mismatch at seed {seed}")
        if episode.get("task_count") != 80 or len(tasks) != 80:
            failures.append(f"{path}: seed {seed} does not contain 80 tasks")
        task_ids = [task.get("task_id") for task in tasks]
        if len(set(task_ids)) != 80:
            failures.append(f"{path}: duplicate task id at seed {seed}")
        if (episode.get("completed", 0) + episode.get("failed", 0) != 80 or
                episode.get("illegal_actions") != 0 or
                episode.get("resource_leak") is not False):
            failures.append(f"{path}: safety/completeness at seed {seed}")
        order = infer_phase_order(tasks)
        phase_orders["->".join(order)] += 1
        if set(order) != {"NORMAL", "DENSE", "BURST", "RECOVERY"}:
            failures.append(f"{path}: phase coverage at seed {seed}: {order}")
        elif order.index("BURST") > order.index("RECOVERY"):
            failures.append(f"{path}: BURST after RECOVERY at seed {seed}")
        mode_counts.update(episode.get("transport_modes", {}))
        for phase, counts in episode.get("transport_modes_by_phase", {}).items():
            for mode, count in counts.items():
                phase_mode_counts[(phase, mode)] += count
        for metric in METRICS:
            value = episode_vector(episode, metric)
            if not finite(value):
                failures.append(f"{path}: non-finite {metric} at seed {seed}")
            vectors[metric].append(value)
    return {
        "metadata": {
            "source_group": row["source_group"],
            "training_cohort": row["training_cohort"],
            "training_seed_index": row["training_seed_index"],
            "training_gate_accepted": row["training_gate_accepted"],
            "weight_sha256": row["weight_sha256"],
        },
        "vectors": vectors,
        "run_level": run_level_values(result),
        "phase_orders": dict(phase_orders),
        "transport_modes": dict(mode_counts),
        "transport_modes_by_phase": {
            phase: {
                mode: phase_mode_counts[(phase, mode)]
                for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")
                if phase_mode_counts[(phase, mode)]
            }
            for phase in ("NORMAL", "DENSE", "BURST", "RECOVERY")
        },
    }, failures


def baseline_extract(result: dict, reference_fingerprints: dict[int, str],
                     expected_seeds: list[int], name: str) -> tuple[dict, list[str]]:
    failures = []
    episodes = {int(row["seed"]): row for row in result["episodes"]}
    if sorted(episodes) != expected_seeds:
        failures.append(f"{name}: test seed set mismatch")
    vectors = {metric: [] for metric in METRICS}
    for seed in expected_seeds:
        episode = episodes.get(seed)
        if episode is None:
            continue
        if episode.get("task_stream_fingerprint") != reference_fingerprints[seed]:
            failures.append(f"{name}: fingerprint mismatch at seed {seed}")
        if (episode.get("task_count") != 80 or
                len(episode.get("tasks", [])) != 80 or
                episode.get("illegal_actions") != 0 or
                episode.get("resource_leak") is not False):
            failures.append(f"{name}: safety/completeness at seed {seed}")
        for metric in METRICS:
            value = episode_vector(episode, metric)
            if not finite(value):
                failures.append(f"{name}: non-finite {metric} at seed {seed}")
            vectors[metric].append(value)
    return {
        "vectors": vectors,
        "run_level": run_level_values(result),
    }, failures


def cohort_summary(array_by_metric: dict[str, np.ndarray]) -> dict:
    result = {}
    for metric, values in array_by_metric.items():
        flattened = values.reshape(-1)
        result[metric] = {
            "mean": float(flattened.mean()),
            "training_seed_sd_of_test_seed_means": float(
                values.mean(axis=1).std(ddof=1)) if values.shape[0] > 1 else 0.0,
            "test_episode_count_per_training_seed": int(values.shape[1]),
        }
    return result


def run_level_summary(rows: list[dict]) -> dict:
    result = {}
    for metric in RUN_LEVEL_METRICS:
        values = np.asarray(
            [row["run_level"][metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean_across_training_seeds": float(values.mean()),
            "training_seed_sd": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0),
        }
    return result


def mode_summary(rows: list[dict]) -> dict:
    total = Counter()
    by_phase = Counter()
    for row in rows:
        total.update(row["transport_modes"])
        for phase, counts in row["transport_modes_by_phase"].items():
            for mode, count in counts.items():
                by_phase[(phase, mode)] += count
    denominator = max(sum(total.values()), 1)
    phases = {}
    for phase in ("NORMAL", "DENSE", "BURST", "RECOVERY"):
        phase_total = sum(
            by_phase[(phase, mode)]
            for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"))
        phases[phase] = {
            "count": phase_total,
            "shares": {
                mode: by_phase[(phase, mode)] / max(phase_total, 1)
                for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")},
        }
    return {
        "counts": dict(total),
        "shares": {mode: count / denominator for mode, count in total.items()},
        "by_phase": phases,
    }


def bootstrap_distributions(
        cohorts: dict[str, dict[str, np.ndarray]],
        baselines: dict[str, dict[str, np.ndarray]], *, samples: int,
        seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    test_count = next(iter(next(iter(cohorts.values())).values())).shape[1]
    shared_test_indices = rng.integers(
        0, test_count, size=(samples, test_count))
    cohort_draws: dict[str, dict[str, np.ndarray]] = {}
    for name, metrics in cohorts.items():
        train_count = next(iter(metrics.values())).shape[0]
        train_indices = rng.integers(
            0, train_count, size=(samples, train_count))
        cohort_draws[name] = {}
        for metric, values in metrics.items():
            train_resampled = values[train_indices].mean(axis=1)
            cohort_draws[name][metric] = np.take_along_axis(
                train_resampled, shared_test_indices, axis=1).mean(axis=1)
    baseline_draws: dict[str, dict[str, np.ndarray]] = {}
    for name, metrics in baselines.items():
        baseline_draws[name] = {
            metric: values[shared_test_indices].mean(axis=1)
            for metric, values in metrics.items()}
    return cohort_draws, baseline_draws


def comparison(point_a: float, point_b: float, draws_a: np.ndarray,
               draws_b: np.ndarray, direction: str) -> dict:
    delta = draws_a - draws_b
    point_delta = point_a - point_b
    probability_better = (
        float(np.mean(delta > 0)) if direction == "higher" else
        float(np.mean(delta < 0)))
    return {
        "a_minus_b": point_delta,
        "crossed_bootstrap_95ci": [
            float(np.percentile(delta, 2.5)),
            float(np.percentile(delta, 97.5))],
        "direction_better": direction,
        "probability_a_better": probability_better,
    }


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    locked = protocol["locked_test"]
    seed_start = int(locked["seed_start"])
    test_count = int(locked["seed_count"])
    expected_seeds = list(range(seed_start, seed_start + test_count))
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed_start"])
    bootstrap_samples = int(protocol["statistics"]["samples"])
    failures: list[str] = []

    baseline_payload = json.loads(
        (RESULT_ROOT / "baselines.json").read_text(encoding="utf-8"))
    rule_episodes = baseline_payload["time_greedy_rule"]["episodes"]
    reference_fingerprints = {
        int(episode["seed"]): episode["task_stream_fingerprint"]
        for episode in rule_episodes}
    if sorted(reference_fingerprints) != expected_seeds:
        failures.append("reference task seed set mismatch")

    cohort_runs: dict[str, list[dict]] = {name: [] for name in COHORT_PATHS}
    phase_orders = Counter()
    for cohort, directory in COHORT_PATHS.items():
        paths = sorted(directory.glob("seed_*.json"))
        if len(paths) != 10:
            failures.append(f"{cohort}: expected 10 files, found {len(paths)}")
        for path in paths:
            extracted, row_failures = audit_and_extract(
                path, reference_fingerprints, expected_seeds)
            failures.extend(row_failures)
            cohort_runs[cohort].append(extracted)
            phase_orders.update(extracted["phase_orders"])

    baseline_rows = {}
    for key in ("time_greedy_rule", "single_dog_only"):
        extracted, row_failures = baseline_extract(
            baseline_payload[key], reference_fingerprints,
            expected_seeds, key)
        failures.extend(row_failures)
        baseline_rows[key.upper()] = extracted

    cohort_arrays = {
        cohort: {
            metric: np.asarray(
                [run["vectors"][metric] for run in runs],
                dtype=np.float64)
            for metric in METRICS}
        for cohort, runs in cohort_runs.items()}
    baseline_arrays = {
        name: {
            metric: np.asarray(row["vectors"][metric], dtype=np.float64)
            for metric in METRICS}
        for name, row in baseline_rows.items()}
    cohort_points = {
        name: cohort_summary(metrics)
        for name, metrics in cohort_arrays.items()}
    baseline_points = {
        name: {
            metric: {"mean": float(values.mean())}
            for metric, values in metrics.items()}
        for name, metrics in baseline_arrays.items()}
    cohort_run_level_points = {
        name: run_level_summary(rows)
        for name, rows in cohort_runs.items()}
    baseline_run_level_points = {
        name: {
            metric: {"value": row["run_level"][metric]}
            for metric in RUN_LEVEL_METRICS}
        for name, row in baseline_rows.items()}
    cohort_mode_summaries = {
        name: mode_summary(rows) for name, rows in cohort_runs.items()}

    cohort_draws, baseline_draws = bootstrap_distributions(
        cohort_arrays, baseline_arrays, samples=bootstrap_samples,
        seed=bootstrap_seed)
    primary = "STAGE20_MIXED_CURRICULUM"
    comparisons = {}
    comparator_names = [
        "STAGE19_MEDIUM", "STAGE19_DENSE", "STAGE19_BURST",
        "TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"]
    for comparator in comparator_names:
        is_cohort = comparator in cohort_arrays
        points = cohort_points if is_cohort else baseline_points
        draws = cohort_draws if is_cohort else baseline_draws
        comparisons[f"{primary}_vs_{comparator}"] = {
            metric: comparison(
                cohort_points[primary][metric]["mean"],
                points[comparator][metric]["mean"],
                cohort_draws[primary][metric], draws[comparator][metric],
                direction)
            for metric, (_, direction) in METRICS.items()}

    all_stage20_rows = cohort_runs[primary]
    accepted_labels = [
        row["metadata"]["training_gate_accepted"]
        for row in all_stage20_rows]
    qa = {
        "campaign_completed": json.loads(
            (RESULT_ROOT / "campaign.status.json").read_text(
                encoding="utf-8"))["state"] == "COMPLETED",
        "freeze_gate_passed": freeze["hard_gate"]["passed"] is True,
        "all_40_policy_files_present": sum(
            len(runs) for runs in cohort_runs.values()) == 40,
        "all_10_stage20_seeds_retained": len(all_stage20_rows) == 10,
        "stage20_gate_labels_are_7_and_3": (
            sum(accepted_labels) == 7 and len(accepted_labels) -
            sum(accepted_labels) == 3),
        "all_100_shared_test_seeds_present": (
            len(reference_fingerprints) == 100),
        "multiple_random_phase_orders_observed": len(phase_orders) >= 2,
        "burst_precedes_recovery_in_every_observed_order": all(
            order.split("->").index("BURST") <
            order.split("->").index("RECOVERY")
            for order in phase_orders),
        "no_data_quality_failure": not failures,
    }
    report = {
        "schema_version": "warehouse_stage20_locked_summary_v1",
        "stage": 20,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Held-out descriptive and crossed-bootstrap comparison. All ten "
            "pre-registered Stage 20 training seeds are retained regardless "
            "of their training gate label."),
        "data_as_of": "2026-08-24 Asia/Shanghai",
        "test_seed_range": [expected_seeds[0], expected_seeds[-1]],
        "test_episode_count": test_count,
        "tasks_per_episode": 80,
        "bootstrap": {
            "design": "crossed_training_seed_by_shared_test_seed",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "episode_level_metric_definitions": {
            metric: {"source_episode_field": source, "direction": direction}
            for metric, (source, direction) in METRICS.items()},
        "estimand_note": (
            "Crossed bootstrap operates on episode-level metrics. Therefore "
            "tail metrics are explicitly named mean_episode_p95, and the "
            "run-level table separately reports pooled-task P95. Throughput "
            "is likewise separated into mean-episode and aggregate run-level "
            "estimands."),
        "quality_assurance": qa,
        "quality_failures": failures,
        "phase_order_counts_across_policy_runs": dict(phase_orders),
        "cohort_point_estimates": cohort_points,
        "baseline_point_estimates": baseline_points,
        "cohort_run_level_point_estimates": cohort_run_level_points,
        "baseline_run_level_point_estimates": baseline_run_level_points,
        "cohort_transport_mode_summaries": cohort_mode_summaries,
        "primary_comparisons": comparisons,
        "passed": all(qa.values()),
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "passed": report["passed"],
        "quality_assurance": qa,
        "stage20": cohort_points[primary],
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
