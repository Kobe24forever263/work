#!/usr/bin/env python3
"""Summarize locked Stage 18 tests with paired hierarchical statistics."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


WORK_ROOT = Path(__file__).resolve().parents[1]
ROOT = WORK_ROOT / "results" / "stage18_locked_test"
PROFILES = ("MEDIUM", "DENSE", "BURST")
CONDITIONS = (
    "full_context_v2", "no_queue_resource", "no_handover_cues",
    "no_persistent_position")
METRICS = ("reward", "success_rate", "throughput_tasks_per_hour")
BOOTSTRAP_SEED = 44000000
BOOTSTRAP_SAMPLES = 10000


def path(profile: str, condition: str, seed: int) -> Path:
    return (ROOT / f"trained_{profile.lower()}" / condition /
            f"tested_{profile.lower()}" / f"seed_{seed:02d}.json")


def episode_values(result: dict, side: str, metric: str) -> np.ndarray:
    episodes = result[side]["episodes"]
    if metric == "reward":
        return np.asarray([row["reward"] for row in episodes], dtype=float)
    if metric == "success_rate":
        return np.asarray([
            row["completed"] / max(row["completed"] + row["failed"], 1)
            for row in episodes], dtype=float)
    if metric == "throughput_tasks_per_hour":
        return np.asarray([
            (row["completed"] + row["failed"]) /
            max(row["simulated_time"], 1e-9) * 3600.0
            for row in episodes], dtype=float)
    raise ValueError(metric)


def exact_sign_flip_p(seed_deltas: np.ndarray) -> float:
    observed = abs(float(seed_deltas.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(seed_deltas)):
        values.append(abs(float(np.mean(seed_deltas * signs))))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def paired_stats(episode_deltas: np.ndarray, rng: np.random.Generator) -> dict:
    seed_means = episode_deltas.mean(axis=1)
    bootstrap = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    seed_count, episode_count = episode_deltas.shape
    for index in range(BOOTSTRAP_SAMPLES):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_episodes = rng.integers(
            0, episode_count, size=(seed_count, episode_count))
        rows = episode_deltas[sampled_seeds]
        bootstrap[index] = rows[
            np.arange(seed_count)[:, None], sampled_episodes].mean()
    mean = float(seed_means.mean())
    stdev = float(seed_means.std(ddof=1))
    return {
        "training_seed_count": seed_count,
        "test_episode_count_per_seed": episode_count,
        "mean_delta": mean,
        "sample_stdev_across_training_seeds": stdev,
        "hierarchical_bootstrap_95ci": [
            float(np.quantile(bootstrap, .025)),
            float(np.quantile(bootstrap, .975))],
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(seed_means),
        "cohen_dz": mean / stdev if stdev > 0 else (
            math.inf if mean > 0 else -math.inf if mean < 0 else 0.0),
        "positive_training_seed_count": int(np.sum(seed_means > 0)),
        "seed_mean_deltas": seed_means.tolist(),
    }


def load_matched(profile: str, condition: str) -> list[dict]:
    rows = []
    for seed in range(1, 11):
        source = path(profile, condition, seed)
        if not source.exists():
            raise FileNotFoundError(source)
        rows.append(json.loads(source.read_text(encoding="utf-8")))
    return rows


def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    for profile in PROFILES:
        full = load_matched(profile, "full_context_v2")
        for metric in METRICS:
            deltas = np.stack([
                episode_values(row, "policy", metric) -
                episode_values(row, "rule_baseline", metric)
                for row in full])
            comparisons.append({
                "profile": profile,
                "comparison": "full_context_v2_minus_rule_baseline",
                "metric": metric,
                **paired_stats(deltas, rng),
            })
        for condition in CONDITIONS[1:]:
            ablation = load_matched(profile, condition)
            for metric in METRICS:
                deltas = np.stack([
                    episode_values(full[index], "policy", metric) -
                    episode_values(ablation[index], "policy", metric)
                    for index in range(10)])
                comparisons.append({
                    "profile": profile,
                    "comparison": f"full_context_v2_minus_{condition}",
                    "metric": metric,
                    **paired_stats(deltas, rng),
                })
    report = {
        "schema_version": "warehouse_stage18_locked_statistics_v1",
        "claim_boundary": "Locked test; paired across training and test seeds.",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "method": {
            "uncertainty": (
                "Hierarchical bootstrap over 10 independent training seeds "
                "and 100 paired test episodes per seed."),
            "significance": (
                "Exact two-sided sign-flip randomization test over the 10 "
                "training-seed mean paired differences."),
            "effect_size": "Paired standardized mean difference (Cohen dz).",
        },
        "comparisons": comparisons,
    }
    output = ROOT / "stage18_locked_test_statistics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    csv_path = ROOT / "stage18_locked_test_statistics.csv"
    fields = [
        "profile", "comparison", "metric", "mean_delta",
        "sample_stdev_across_training_seeds", "ci_low", "ci_high",
        "exact_two_sided_sign_flip_p", "cohen_dz",
        "positive_training_seed_count"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({
                **{key: row[key] for key in fields
                   if key not in {"ci_low", "ci_high"}},
                "ci_low": row["hierarchical_bootstrap_95ci"][0],
                "ci_high": row["hierarchical_bootstrap_95ci"][1],
            })
    print(json.dumps({
        "output": str(output), "csv": str(csv_path),
        "comparison_count": len(comparisons)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
