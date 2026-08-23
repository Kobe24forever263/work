#!/usr/bin/env python3
"""Summarize Stage18 v2 with crossed pairing and corrected metrics."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
ROOT = WORK_ROOT / "results" / "stage18_locked_test_v2"
SEED_CONFIG = (WORK_ROOT / "src" / "warehouse_bringup" / "config" /
               "experiment_seeds_stage18_v2.yaml")
PROFILES = ("MEDIUM", "DENSE", "BURST")
CONDITIONS = (
    "full_context_v2", "no_queue_resource", "no_handover_cues",
    "no_persistent_position")
METRICS = (
    "reward", "success_rate", "resolved_throughput_tasks_per_hour",
    "successful_throughput_tasks_per_hour")
DOG_CONTROL_METRICS = (
    "terminal_adjusted_reward", "success_rate",
    "resolved_throughput_tasks_per_hour",
    "successful_throughput_tasks_per_hour")


def result_path(training_profile: str, condition: str,
                evaluation_profile: str, seed: int) -> Path:
    return (ROOT / f"trained_{training_profile.lower()}" / condition /
            f"tested_{evaluation_profile.lower()}" /
            f"seed_{seed:02d}.json")


def load_results(training_profile: str, condition: str,
                 evaluation_profile: str) -> list[dict]:
    rows = []
    for seed in range(1, 11):
        source = result_path(
            training_profile, condition, evaluation_profile, seed)
        if not source.exists():
            raise FileNotFoundError(source)
        row = json.loads(source.read_text(encoding="utf-8"))
        if row.get("schema_version") != "warehouse_stage18_locked_test_v2":
            raise ValueError(f"unexpected schema: {source}")
        if (row.get("evaluation_protocol", {}).get("handover_sampling") !=
                "TASK_KEYED_COMMON_RANDOM_NUMBERS"):
            raise ValueError(f"unpaired handover sampling: {source}")
        rows.append(row)
    return rows


def load_single_dog_baseline(evaluation_profile: str) -> dict:
    source = (ROOT / "baselines" / "single_dog_only" /
              f"tested_{evaluation_profile.lower()}.json")
    if not source.exists():
        raise FileNotFoundError(source)
    row = json.loads(source.read_text(encoding="utf-8"))
    if (row.get("schema_version") !=
            "warehouse_stage18_single_dog_baseline_v1"):
        raise ValueError(f"unexpected single-dog schema: {source}")
    result = row.get("single_dog_only", {})
    if (result.get("allowed_transport_modes") != ["SINGLE_DOG"] or
            set(result.get("transport_modes", {})) - {"SINGLE_DOG"}):
        raise ValueError(f"single-dog baseline exposed another mode: {source}")
    return row


def side_components(results: list[dict], side: str,
                    metric: str) -> tuple[str, np.ndarray, np.ndarray | None]:
    numerator = []
    denominator = []
    expected_seeds = None
    for result in results:
        episodes = result[side]["episodes"]
        episode_seeds = [int(row["seed"]) for row in episodes]
        if expected_seeds is None:
            expected_seeds = episode_seeds
        elif episode_seeds != expected_seeds:
            raise ValueError("shared test-seed order differs across runs")
        if metric in {"reward", "terminal_adjusted_reward"}:
            numerator.append([
                float(row.get(metric, row["reward"])) for row in episodes])
            continue
        if metric == "success_rate":
            numerator.append([float(row["completed"]) for row in episodes])
            denominator.append([
                float(row.get(
                    "scheduled_tasks", row["completed"] + row["failed"]))
                for row in episodes])
            continue
        if metric == "resolved_throughput_tasks_per_hour":
            numerator.append([
                float(row["completed"] + row["failed"])
                for row in episodes])
            denominator.append([
                float(row["simulated_time"]) for row in episodes])
            continue
        if metric == "successful_throughput_tasks_per_hour":
            numerator.append([float(row["completed"]) for row in episodes])
            denominator.append([
                float(row["simulated_time"]) for row in episodes])
            continue
        raise ValueError(metric)
    if metric in {"reward", "terminal_adjusted_reward"}:
        return "mean", np.asarray(numerator, dtype=float), None
    scale = 1.0 if metric == "success_rate" else 3600.0
    return (f"ratio:{scale}", np.asarray(numerator, dtype=float),
            np.asarray(denominator, dtype=float))


def select(matrix: np.ndarray, seed_indices: np.ndarray | None,
           test_indices: np.ndarray | None) -> np.ndarray:
    if seed_indices is None:
        seed_indices = np.arange(matrix.shape[0])
    if test_indices is None:
        test_indices = np.arange(matrix.shape[1])
    return matrix[np.ix_(seed_indices, test_indices)]


def estimate(kind: str, numerator: np.ndarray,
             denominator: np.ndarray | None,
             seed_indices: np.ndarray | None = None,
             test_indices: np.ndarray | None = None) -> float:
    chosen_numerator = select(numerator, seed_indices, test_indices)
    if kind == "mean":
        return float(chosen_numerator.mean())
    if denominator is None:
        raise ValueError("ratio estimator requires a denominator")
    scale = float(kind.split(":", 1)[1])
    chosen_denominator = select(denominator, seed_indices, test_indices)
    return (float(chosen_numerator.sum()) /
            max(float(chosen_denominator.sum()), 1e-12) * scale)


def exact_sign_flip_p(seed_deltas: np.ndarray) -> float:
    observed = abs(float(seed_deltas.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(seed_deltas)):
        values.append(abs(float(np.mean(seed_deltas * signs))))
    return (sum(value >= observed - 1e-12 for value in values) /
            len(values))


def paired_stats(left: list[dict], left_side: str, right: list[dict],
                 right_side: str, metric: str,
                 rng: np.random.Generator, samples: int) -> dict:
    left_kind, left_num, left_den = side_components(left, left_side, metric)
    right_kind, right_num, right_den = side_components(
        right, right_side, metric)
    if left_kind != right_kind or left_num.shape != right_num.shape:
        raise ValueError("paired result shapes or estimands differ")
    seed_count, test_count = left_num.shape
    left_value = estimate(left_kind, left_num, left_den)
    right_value = estimate(right_kind, right_num, right_den)
    seed_deltas = np.asarray([
        estimate(left_kind, left_num, left_den,
                 np.asarray([seed]), None) -
        estimate(right_kind, right_num, right_den,
                 np.asarray([seed]), None)
        for seed in range(seed_count)], dtype=float)
    bootstrap = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        # The same sampled test-seed indices are used for every selected
        # training seed.  This preserves the crossed, shared-scenario design.
        sampled_tests = rng.integers(0, test_count, size=test_count)
        bootstrap[index] = (
            estimate(left_kind, left_num, left_den,
                     sampled_seeds, sampled_tests) -
            estimate(right_kind, right_num, right_den,
                     sampled_seeds, sampled_tests))
    mean = left_value - right_value
    stdev = float(seed_deltas.std(ddof=1))
    return {
        "training_seed_count": seed_count,
        "shared_test_seed_count": test_count,
        "left_estimate": left_value,
        "right_estimate": right_value,
        "mean_delta": mean,
        "sample_stdev_across_training_seeds": stdev,
        "crossed_bootstrap_95ci": [
            float(np.quantile(bootstrap, .025)),
            float(np.quantile(bootstrap, .975))],
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(seed_deltas),
        "cohen_dz": mean / stdev if stdev > 0 else (
            math.inf if mean > 0 else -math.inf if mean < 0 else 0.0),
        "positive_training_seed_count": int(np.sum(seed_deltas > 0)),
        "seed_mean_deltas": seed_deltas.tolist(),
    }


def adjusted_pvalues(values: list[float], method: str) -> list[float]:
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (count - rank) * values[index])
            adjusted[index] = min(running, 1.0)
        return adjusted
    if method == "bh":
        running = 1.0
        for reverse_rank in range(count - 1, -1, -1):
            index = order[reverse_rank]
            running = min(
                running, values[index] * count / (reverse_rank + 1))
            adjusted[index] = min(running, 1.0)
        return adjusted
    raise ValueError(method)


def add_multiplicity(rows: list[dict]) -> None:
    raw = [row["exact_two_sided_sign_flip_p"] for row in rows]
    for row, value in zip(rows, adjusted_pvalues(raw, "holm")):
        row["global_holm_adjusted_p"] = value
    for row, value in zip(rows, adjusted_pvalues(raw, "bh")):
        row["global_bh_fdr_q"] = value
    families = sorted({row["hypothesis_family"] for row in rows
                       if row["hypothesis_family"] != "diagnostic"})
    for family in families:
        indices = [index for index, row in enumerate(rows)
                   if row["hypothesis_family"] == family]
        family_raw = [raw[index] for index in indices]
        holm = adjusted_pvalues(family_raw, "holm")
        bh = adjusted_pvalues(family_raw, "bh")
        for position, index in enumerate(indices):
            rows[index]["family_holm_adjusted_p"] = holm[position]
            rows[index]["family_bh_fdr_q"] = bh[position]
            rows[index]["family_hypothesis_count"] = len(indices)
    for row in rows:
        if row["hypothesis_family"] == "diagnostic":
            row["family_holm_adjusted_p"] = None
            row["family_bh_fdr_q"] = None
            row["family_hypothesis_count"] = 0


def primary_family(comparison: str, metric: str) -> str:
    if metric == "resolved_throughput_tasks_per_hour":
        return "diagnostic"
    if comparison == "full_context_v2_minus_rule_baseline":
        return "primary_full_vs_rule"
    if comparison == "full_context_v2_minus_single_dog_only":
        return "primary_multiagent_vs_single_dog"
    return f"exploratory_{comparison}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "profile", "training_profile", "evaluation_profile", "comparison",
        "metric", "hypothesis_family", "left_estimate", "right_estimate",
        "mean_delta", "ci_low", "ci_high",
        "exact_two_sided_sign_flip_p", "family_holm_adjusted_p",
        "family_bh_fdr_q", "global_holm_adjusted_p", "global_bh_fdr_q",
        "positive_training_seed_count"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            interval = row["crossed_bootstrap_95ci"]
            writer.writerow({
                "profile": row.get("profile", ""),
                "training_profile": row.get("training_profile", ""),
                "evaluation_profile": row.get("evaluation_profile", ""),
                "comparison": row["comparison"],
                "metric": row["metric"],
                "hypothesis_family": row["hypothesis_family"],
                "left_estimate": row["left_estimate"],
                "right_estimate": row["right_estimate"],
                "mean_delta": row["mean_delta"],
                "ci_low": interval[0],
                "ci_high": interval[1],
                "exact_two_sided_sign_flip_p": (
                    row["exact_two_sided_sign_flip_p"]),
                "family_holm_adjusted_p": row["family_holm_adjusted_p"],
                "family_bh_fdr_q": row["family_bh_fdr_q"],
                "global_holm_adjusted_p": row["global_holm_adjusted_p"],
                "global_bh_fdr_q": row["global_bh_fdr_q"],
                "positive_training_seed_count": (
                    row["positive_training_seed_count"]),
            })


def main() -> int:
    protocol = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    samples = int(protocol["statistics"]["bootstrap_samples"])
    rng = np.random.default_rng(
        int(protocol["statistics"]["bootstrap_seed_start"]))
    matched = []
    for profile in PROFILES:
        full = load_results(profile, "full_context_v2", profile)
        for metric in METRICS:
            comparison = "full_context_v2_minus_rule_baseline"
            matched.append({
                "profile": profile,
                "comparison": comparison,
                "metric": metric,
                "hypothesis_family": primary_family(comparison, metric),
                **paired_stats(full, "policy", full, "rule_baseline",
                               metric, rng, samples),
            })
        dog_only = load_single_dog_baseline(profile)
        # The blank control has no training randomness.  Replicate its same
        # shared-seed episode vector across the ten learned-policy seeds so
        # the crossed comparison keeps the correct 10x100 design.
        dog_rows = [dog_only] * len(full)
        for metric in DOG_CONTROL_METRICS:
            comparison = "full_context_v2_minus_single_dog_only"
            matched.append({
                "profile": profile,
                "comparison": comparison,
                "metric": metric,
                "hypothesis_family": primary_family(comparison, metric),
                **paired_stats(full, "policy", dog_rows,
                               "single_dog_only", metric, rng, samples),
            })
        for condition in CONDITIONS[1:]:
            ablation = load_results(profile, condition, profile)
            comparison = f"full_context_v2_minus_{condition}"
            for metric in METRICS:
                matched.append({
                    "profile": profile,
                    "comparison": comparison,
                    "metric": metric,
                    "hypothesis_family": primary_family(comparison, metric),
                    **paired_stats(full, "policy", ablation, "policy",
                                   metric, rng, samples),
                })
    add_multiplicity(matched)

    cross_load = []
    for training_profile in PROFILES:
        for evaluation_profile in PROFILES:
            full = load_results(
                training_profile, "full_context_v2", evaluation_profile)
            for metric in METRICS:
                comparison = "full_context_v2_minus_rule_baseline"
                cross_load.append({
                    "training_profile": training_profile,
                    "evaluation_profile": evaluation_profile,
                    "comparison": comparison,
                    "metric": metric,
                    "hypothesis_family": (
                        "cross_load_full_vs_rule" if metric !=
                        "resolved_throughput_tasks_per_hour" else
                        "diagnostic"),
                    **paired_stats(full, "policy", full, "rule_baseline",
                                   metric, rng, samples),
                })
    add_multiplicity(cross_load)

    report = {
        "schema_version": "warehouse_stage18_locked_statistics_v2",
        "claim_boundary": (
            "Fresh corrected test namespace; no training or checkpoint "
            "selection used these seeds."),
        "test_seed_start": int(protocol["test"]["seed_start"]),
        "test_seed_count": int(protocol["test"]["seed_count"]),
        "bootstrap_seed": int(
            protocol["statistics"]["bootstrap_seed_start"]),
        "bootstrap_samples": samples,
        "method": {
            "randomness_pairing": "TASK_KEYED common random numbers",
            "uncertainty": (
                "Two-way crossed paired bootstrap over 10 training seeds "
                "and the same 100 test-seed indices."),
            "throughput": {
                "resolved": "(completed+failed)/total simulated time*3600",
                "successful": "completed/total simulated time*3600",
            },
            "significance": (
                "Exact two-sided sign-flip test over training-seed paired "
                "differences."),
            "multiplicity": (
                "Holm familywise adjustment plus BH-FDR, reported both "
                "within declared families and globally."),
            "single_dog_control": (
                "Four-Go2 fleet with SINGLE_DOG as the only permitted "
                "transport mode; no Carter and no handover."),
        },
        "matched_load_comparisons": matched,
        "cross_load_comparisons": cross_load,
    }
    output = ROOT / "stage18_locked_test_v2_statistics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    matched_csv = ROOT / "stage18_locked_test_v2_statistics.csv"
    cross_csv = ROOT / "stage18_locked_test_v2_cross_load_statistics.csv"
    write_csv(matched_csv, matched)
    write_csv(cross_csv, cross_load)
    print(json.dumps({
        "output": str(output),
        "matched_csv": str(matched_csv),
        "cross_load_csv": str(cross_csv),
        "matched_comparison_count": len(matched),
        "cross_load_comparison_count": len(cross_load),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
