#!/usr/bin/env python3
"""Audit and summarize the frozen Stage 26 recurrent locked test."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (ROOT / "results" / "stage26_recurrent_causal" /
               "locked_test_v1")
SCHEMA = "warehouse_stage26_recurrent_locked_test_v1"
BOOTSTRAP_SEED = 73460000
BOOTSTRAP_SAMPLES = 10000
METRICS = (
    "mean_reward", "success_rate",
    "successful_throughput_tasks_per_hour", "mean_simulated_time")


def load_results() -> list[dict]:
    rows = []
    expected_test_seeds = None
    checkpoint_hashes = set()
    training_seeds = set()
    for index in range(1, 11):
        path = RESULT_ROOT / f"seed_{index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != SCHEMA or not row.get("passed"):
            raise ValueError(f"invalid locked result: {path}")
        if row.get("training_seed_index") != index:
            raise ValueError(f"training seed index mismatch: {path}")
        seeds = [int(item["seed"]) for item in row["policy"]["episodes"]]
        baseline_seeds = [
            int(item["seed"]) for item in row["rule_baseline"]["episodes"]]
        if seeds != baseline_seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"test-seed pairing/uniqueness failed: {path}")
        if expected_test_seeds is None:
            expected_test_seeds = seeds
        elif seeds != expected_test_seeds:
            raise ValueError(f"shared test-seed order differs: {path}")
        checkpoint_hashes.add(row["checkpoint_sha256"])
        training_seeds.add(int(row["training_base_seed"]))
        rows.append(row)
    if len(checkpoint_hashes) != 10 or len(training_seeds) != 10:
        raise ValueError("training checkpoints/seeds are not unique")
    return rows


def components(results: list[dict], side: str,
               metric: str) -> tuple[str, np.ndarray, np.ndarray | None]:
    numerator, denominator = [], []
    for result in results:
        episodes = result[side]["episodes"]
        if metric == "mean_reward":
            numerator.append([float(row["reward"]) for row in episodes])
        elif metric == "mean_simulated_time":
            numerator.append([
                float(row["simulated_time"]) for row in episodes])
        elif metric == "success_rate":
            numerator.append([float(row["completed"]) for row in episodes])
            denominator.append([
                float(row["scheduled_tasks"]) for row in episodes])
        elif metric == "successful_throughput_tasks_per_hour":
            numerator.append([float(row["completed"]) for row in episodes])
            denominator.append([
                float(row["simulated_time"]) for row in episodes])
        else:
            raise ValueError(metric)
    if metric in {"mean_reward", "mean_simulated_time"}:
        return "mean", np.asarray(numerator), None
    scale = 1.0 if metric == "success_rate" else 3600.0
    return f"ratio:{scale}", np.asarray(numerator), np.asarray(denominator)


def select(matrix: np.ndarray, seed_indices: np.ndarray | None,
           test_indices: np.ndarray | None) -> np.ndarray:
    seeds = (np.arange(matrix.shape[0]) if seed_indices is None
             else seed_indices)
    tests = (np.arange(matrix.shape[1]) if test_indices is None
             else test_indices)
    return matrix[np.ix_(seeds, tests)]


def estimate(kind: str, numerator: np.ndarray,
             denominator: np.ndarray | None,
             seed_indices: np.ndarray | None = None,
             test_indices: np.ndarray | None = None) -> float:
    chosen = select(numerator, seed_indices, test_indices)
    if kind == "mean":
        return float(chosen.mean())
    if denominator is None:
        raise ValueError("ratio estimator lacks denominator")
    scale = float(kind.split(":", 1)[1])
    return float(chosen.sum() / max(
        select(denominator, seed_indices, test_indices).sum(), 1e-12) * scale)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * signs)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def paired_stats(results: list[dict], metric: str,
                 rng: np.random.Generator) -> dict:
    kind, policy_num, policy_den = components(results, "policy", metric)
    other_kind, rule_num, rule_den = components(
        results, "rule_baseline", metric)
    if kind != other_kind or policy_num.shape != rule_num.shape:
        raise ValueError(f"unpaired metric arrays for {metric}")
    train_count, test_count = policy_num.shape
    policy_estimate = estimate(kind, policy_num, policy_den)
    rule_estimate = estimate(kind, rule_num, rule_den)
    seed_deltas = np.asarray([
        estimate(kind, policy_num, policy_den, np.asarray([index]), None) -
        estimate(kind, rule_num, rule_den, np.asarray([index]), None)
        for index in range(train_count)])
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        train_indices = rng.integers(0, train_count, size=train_count)
        test_indices = rng.integers(0, test_count, size=test_count)
        draws[index] = (
            estimate(kind, policy_num, policy_den,
                     train_indices, test_indices) -
            estimate(kind, rule_num, rule_den,
                     train_indices, test_indices))
    mean_delta = policy_estimate - rule_estimate
    stdev = float(seed_deltas.std(ddof=1))
    return {
        "metric": metric,
        "direction": (
            "lower_is_better" if metric == "mean_simulated_time" else
            "higher_is_better"),
        "training_seed_count": train_count,
        "shared_locked_test_seed_count": test_count,
        "policy_estimate": policy_estimate,
        "rule_baseline_estimate": rule_estimate,
        "mean_delta_policy_minus_rule": mean_delta,
        "sample_stdev_of_training_seed_deltas": stdev,
        "crossed_bootstrap_95ci": [
            float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975))],
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(seed_deltas),
        "cohen_dz": mean_delta / stdev if stdev > 0 else (
            math.inf if mean_delta > 0 else
            -math.inf if mean_delta < 0 else 0.0),
        "positive_training_seed_count": int(np.sum(seed_deltas > 0)),
        "negative_training_seed_count": int(np.sum(seed_deltas < 0)),
        "training_seed_deltas": seed_deltas.tolist(),
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def seed_level_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        row = {
            "training_seed_index": result["training_seed_index"],
            "training_base_seed": result["training_base_seed"],
        }
        for side in ("policy", "rule_baseline"):
            value = result[side]
            row.update({
                f"{side}_success_rate": value["success_rate"],
                f"{side}_mean_reward": value["mean_reward"],
                f"{side}_successful_throughput_tasks_per_hour":
                    value["successful_throughput_tasks_per_hour"],
                f"{side}_mean_simulated_time":
                    value["mean_simulated_time"],
                f"{side}_illegal_action_count":
                    value["illegal_action_count"],
                f"{side}_resource_leak_count":
                    value["resource_leak_count"],
            })
        for metric in METRICS:
            row[f"delta_{metric}"] = (
                row[f"policy_{metric}"] - row[f"rule_baseline_{metric}"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality_checks(results: list[dict]) -> dict:
    policy_episodes = [
        episode for result in results for episode in result["policy"]["episodes"]]
    baseline_reference = results[0]["rule_baseline"]["episodes"]
    baseline_replicas_equal = all(
        result["rule_baseline"]["episodes"] == baseline_reference
        for result in results[1:])
    checks = {
        "ten_unique_training_seeds": len(results) == 10,
        "one_hundred_shared_unique_test_seeds": all(
            len(result["policy"]["episodes"]) == 100 for result in results),
        "policy_episode_row_count_is_1000": len(policy_episodes) == 1000,
        "baseline_replicas_are_identical": baseline_replicas_equal,
        "all_source_gates_passed": all(result["passed"] for result in results),
        "no_policy_illegal_actions": all(
            result["policy"]["illegal_action_count"] == 0
            for result in results),
        "no_policy_resource_leaks": all(
            result["policy"]["resource_leak_count"] == 0
            for result in results),
        "all_policy_tasks_resolved": all(
            result["policy"]["unresolved"] == 0 for result in results),
        "valid_episode_counts_and_times": all(
            episode["scheduled_tasks"] > 0 and
            episode["completed"] >= 0 and episode["failed"] >= 0 and
            episode["completed"] + episode["failed"] ==
            episode["scheduled_tasks"] and episode["simulated_time"] > 0
            for episode in policy_episodes),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "grain": {
            "independent_training_seeds": 10,
            "shared_locked_test_seeds": 100,
            "policy_episode_rows": 1000,
            "independent_rule_baseline_episode_rows": 100,
            "warning": (
                "The baseline's 100 shared episodes are replicated across "
                "training seeds for paired computation and are not 1000 "
                "independent baseline observations."),
        },
    }


def markdown_report(statistics: list[dict], quality: dict) -> str:
    labels = {
        "mean_reward": "平均回报",
        "success_rate": "成功率",
        "successful_throughput_tasks_per_hour": "成功吞吐量（任务/小时）",
        "mean_simulated_time": "平均仿真时长（秒）",
    }
    lines = [
        "# Stage 26 循环策略正式锁定测试汇总", "",
        "## 结论边界", "",
        "十个训练种子在同一组 100 个未见 locked-test 任务流上评估。"
        "权重在测试前已按 SHA-256 冻结，结果不得用于再次挑选权重。", "",
        "## 数据质量", "",
        f"- 数据质量门禁：{'通过' if quality['passed'] else '未通过'}",
        "- 独立训练种子：10",
        "- 共享 locked-test 种子：100",
        "- 策略逐 episode 记录：1000 条",
        "- 规则基线独立任务流：100 条（跨训练种子复用，不按 1000 条计）",
        "", "## 策略相对规则基线", "",
        "| 指标 | 策略 | 规则基线 | 差值（策略−基线） | 95% CI | p（sign-flip） | Holm p | Cohen dz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        low, high = row["crossed_bootstrap_95ci"]
        lines.append(
            f"| {labels[row['metric']]} | {row['policy_estimate']:.4f} | "
            f"{row['rule_baseline_estimate']:.4f} | "
            f"{row['mean_delta_policy_minus_rule']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | "
            f"{row['exact_two_sided_sign_flip_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} | {row['cohen_dz']:+.3f} |")
    lines.extend([
        "", "## 统计方法", "",
        "- 置信区间：训练种子与共享测试种子的双向 crossed bootstrap，10,000 次。",
        "- 显著性：对十个训练种子的配对均值差做精确双侧 sign-flip 检验。",
        "- 多重比较：四项指标使用 Holm 校正。",
        "- 效应量：训练种子配对差值的 Cohen dz。",
        "- 平均仿真时长为越低越好；其余三项指标为越高越好。",
        "", "## 使用限制", "",
        "本报告只比较 Stage 26 循环策略与同任务流规则基线。"
        "若要形成完整论文结论，仍需与前馈策略、单狗空白对照和消融条件在同一 locked-test 协议下比较。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    results = load_results()
    quality = quality_checks(results)
    if not quality["passed"]:
        raise RuntimeError("Stage 26 locked data quality gate failed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    statistics = [paired_stats(results, metric, rng) for metric in METRICS]
    adjusted = holm_adjust([
        row["exact_two_sided_sign_flip_p"] for row in statistics])
    for row, value in zip(statistics, adjusted):
        row["holm_adjusted_p"] = value
    report = {
        "schema_version": "warehouse_stage26_recurrent_locked_statistics_v1",
        "stage": 26,
        "claim_boundary": (
            "Frozen ten-seed cohort on 100 shared unseen locked-test seeds; "
            "no post-test checkpoint selection."),
        "quality": quality,
        "method": {
            "bootstrap": "two-way crossed paired bootstrap",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "significance": "exact two-sided training-seed sign-flip test",
            "multiplicity": "Holm correction across four reported metrics",
            "effect_size": "paired standardized mean difference (Cohen dz)",
        },
        "comparisons": statistics,
    }
    json_path = RESULT_ROOT / "stage26_locked_statistics.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    seed_rows = seed_level_rows(results)
    write_csv(RESULT_ROOT / "stage26_locked_seed_metrics.csv", seed_rows)
    stat_rows = []
    for row in statistics:
        low, high = row["crossed_bootstrap_95ci"]
        stat_rows.append({
            "metric": row["metric"],
            "direction": row["direction"],
            "policy_estimate": row["policy_estimate"],
            "rule_baseline_estimate": row["rule_baseline_estimate"],
            "mean_delta_policy_minus_rule":
                row["mean_delta_policy_minus_rule"],
            "ci_low": low, "ci_high": high,
            "exact_two_sided_sign_flip_p":
                row["exact_two_sided_sign_flip_p"],
            "holm_adjusted_p": row["holm_adjusted_p"],
            "cohen_dz": row["cohen_dz"],
            "positive_training_seed_count":
                row["positive_training_seed_count"],
        })
    write_csv(RESULT_ROOT / "stage26_locked_statistics.csv", stat_rows)
    (RESULT_ROOT / "stage26_locked_quality_review.md").write_text(
        markdown_report(statistics, quality), encoding="utf-8")
    print(json.dumps({
        "quality_passed": quality["passed"],
        "json": str(json_path),
        "seed_csv": str(RESULT_ROOT / "stage26_locked_seed_metrics.csv"),
        "statistics_csv": str(RESULT_ROOT / "stage26_locked_statistics.csv"),
        "report": str(RESULT_ROOT / "stage26_locked_quality_review.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
