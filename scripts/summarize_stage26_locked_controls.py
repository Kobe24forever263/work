#!/usr/bin/env python3
"""Compare Stage 26 recurrent policies with locked control conditions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from summarize_stage26_recurrent_locked_test import (
    BOOTSTRAP_SEED, METRICS, RESULT_ROOT, holm_adjust, load_results,
    paired_stats)


CONTROL_ROOT = RESULT_ROOT / "controls"


def load_controls(full: list[dict]) -> tuple[list[dict], list[dict], dict]:
    memory_rows = []
    checks = {
        "ten_memory_ablation_results": True,
        "memory_ablation_task_streams_match_full": True,
        "memory_ablation_zero_illegal_actions": True,
        "memory_ablation_zero_resource_leaks": True,
    }
    for index, full_row in enumerate(full, start=1):
        path = (CONTROL_ROOT / "memory_reset_each_decision" /
                f"seed_{index:02d}.json")
        if not path.is_file():
            checks["ten_memory_ablation_results"] = False
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        control = row["control"]
        if not row.get("passed") or row.get("condition") != \
                "RESET_GRU_MEMORY_EACH_DECISION":
            raise ValueError(f"invalid memory control: {path}")
        full_seeds = [item["seed"] for item in full_row["policy"]["episodes"]]
        control_seeds = [item["seed"] for item in control["episodes"]]
        checks["memory_ablation_task_streams_match_full"] &= (
            full_seeds == control_seeds)
        checks["memory_ablation_zero_illegal_actions"] &= (
            control["illegal_action_count"] == 0)
        checks["memory_ablation_zero_resource_leaks"] &= (
            control["resource_leak_count"] == 0)
        memory_rows.append({
            "policy": full_row["policy"],
            "rule_baseline": control,
        })

    dog_path = CONTROL_ROOT / "single_dog_only.json"
    if not dog_path.is_file():
        raise FileNotFoundError(dog_path)
    dog_row = json.loads(dog_path.read_text(encoding="utf-8"))
    dog = dog_row["control"]
    checks.update({
        "single_dog_result_passed": dog_row.get("passed") is True,
        "single_dog_has_100_unique_test_seeds": (
            len(dog["episodes"]) == 100 and
            len({item["seed"] for item in dog["episodes"]}) == 100),
        "single_dog_only_uses_single_dog_mode": (
            not (set(dog["transport_modes"]) - {"SINGLE_DOG"})),
        "single_dog_zero_illegal_actions": dog["illegal_action_count"] == 0,
        "single_dog_zero_resource_leaks": dog["resource_leak_count"] == 0,
        "single_dog_tasks_are_all_accounted_for": (
            dog["completed"] + dog["failed"] + dog["unresolved"] ==
            dog["scheduled_task_count"]),
    })
    dog_rows = [{
        "policy": row["policy"],
        "rule_baseline": dog,
    } for row in full]
    quality = {
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(checks.values()),
        "limitations": [
            "The single-dog control has 100 independent task streams and is "
            "replicated only for pairing with ten policy training seeds.",
            "RESET_GRU_MEMORY_EACH_DECISION is an inference-time intervention "
            "on the same trained weights, not a separately trained network.",
            "Stage 25 has only one complete long-run seed and its summary did "
            "not pass; it is excluded from confirmatory statistics.",
        ],
    }
    return memory_rows, dog_rows, quality


def rename(row: dict, comparison: str) -> dict:
    return {
        **row,
        "comparison": comparison,
        "left_estimate": row.pop("policy_estimate"),
        "right_estimate": row.pop("rule_baseline_estimate"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "comparison", "metric", "direction", "left_estimate",
        "right_estimate", "mean_delta_policy_minus_rule", "ci_low",
        "ci_high", "exact_two_sided_sign_flip_p", "holm_adjusted_p",
        "cohen_dz", "positive_training_seed_count",
        "negative_training_seed_count"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            low, high = row["crossed_bootstrap_95ci"]
            writer.writerow({
                "comparison": row["comparison"],
                "metric": row["metric"],
                "direction": row["direction"],
                "left_estimate": row["left_estimate"],
                "right_estimate": row["right_estimate"],
                "mean_delta_policy_minus_rule":
                    row["mean_delta_policy_minus_rule"],
                "ci_low": low, "ci_high": high,
                "exact_two_sided_sign_flip_p":
                    row["exact_two_sided_sign_flip_p"],
                "holm_adjusted_p": row["holm_adjusted_p"],
                "cohen_dz": row["cohen_dz"],
                "positive_training_seed_count":
                    row["positive_training_seed_count"],
                "negative_training_seed_count":
                    row["negative_training_seed_count"],
            })


def table(lines: list[str], title: str, rows: list[dict]) -> None:
    lines.extend([
        f"### {title}", "",
        "| 指标 | 完整循环策略 | 对照 | 差值 | 95% CI | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    labels = {
        "mean_reward": "平均回报",
        "success_rate": "成功率",
        "successful_throughput_tasks_per_hour": "成功吞吐量",
        "mean_simulated_time": "平均仿真时长",
    }
    for row in rows:
        low, high = row["crossed_bootstrap_95ci"]
        lines.append(
            f"| {labels[row['metric']]} | {row['left_estimate']:.4f} | "
            f"{row['right_estimate']:.4f} | "
            f"{row['mean_delta_policy_minus_rule']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | "
            f"{row['holm_adjusted_p']:.4f} |")
    lines.append("")


def main() -> int:
    full = load_results()
    memory, dog, quality = load_controls(full)
    if not quality["passed"]:
        raise RuntimeError("locked controls data-quality gate failed")
    original = json.loads(
        (RESULT_ROOT / "stage26_locked_statistics.json").read_text(
            encoding="utf-8"))
    rule_rows = []
    for source in original["comparisons"]:
        row = dict(source)
        row["comparison"] = "RECURRENT_MINUS_RULE_BASELINE"
        row["left_estimate"] = row.pop("policy_estimate")
        row["right_estimate"] = row.pop("rule_baseline_estimate")
        rule_rows.append(row)

    rng = np.random.default_rng(BOOTSTRAP_SEED + 26)
    groups = {
        "RECURRENT_MINUS_RULE_BASELINE": rule_rows,
        "RECURRENT_MINUS_SINGLE_DOG_ONLY": [
            rename(paired_stats(dog, metric, rng),
                   "RECURRENT_MINUS_SINGLE_DOG_ONLY")
            for metric in METRICS],
        "RECURRENT_MINUS_RESET_MEMORY_EACH_DECISION": [
            rename(paired_stats(memory, metric, rng),
                   "RECURRENT_MINUS_RESET_MEMORY_EACH_DECISION")
            for metric in METRICS],
    }
    for rows in groups.values():
        adjusted = holm_adjust([
            row["exact_two_sided_sign_flip_p"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["holm_adjusted_p"] = value
    all_rows = [row for rows in groups.values() for row in rows]
    report = {
        "schema_version": "warehouse_stage26_locked_controls_statistics_v1",
        "stage": 26,
        "claim_boundary": (
            "Frozen ten-seed recurrent cohort on 100 shared locked task "
            "streams. Memory removal is an inference intervention; Stage 25 "
            "is excluded from confirmatory claims."),
        "quality": quality,
        "method": {
            "uncertainty": "two-way crossed paired bootstrap, 10000 draws",
            "significance": "exact two-sided training-seed sign-flip test",
            "multiplicity": "Holm correction within each four-metric family",
        },
        "comparisons": groups,
    }
    json_path = RESULT_ROOT / "stage26_locked_control_statistics.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    csv_path = RESULT_ROOT / "stage26_locked_control_statistics.csv"
    write_csv(csv_path, all_rows)

    lines = [
        "# Stage 26 正式 locked controls 汇总", "",
        "完整循环策略、规则基线、四 Go2 单狗运输空白对照及每步清空 GRU 记忆干预均使用同一组 100 个 locked-test 任务流。", "",
        f"数据质量门禁：{'通过' if quality['passed'] else '未通过'}。", "",
    ]
    table(lines, "相对规则基线", groups["RECURRENT_MINUS_RULE_BASELINE"])
    table(lines, "相对单狗空白对照", groups["RECURRENT_MINUS_SINGLE_DOG_ONLY"])
    table(lines, "相对每步清空记忆", groups[
        "RECURRENT_MINUS_RESET_MEMORY_EACH_DECISION"])
    lines.extend([
        "## 解释边界", "",
        "- 单狗对照保留四只 Go2，但禁用 Carter 和交接；未完成任务计入有效失败。",
        "- 记忆消融使用相同训练权重，仅在推理时每个决策清空 GRU 状态，因此不能替代重新训练的架构消融。",
        "- Stage 25 只有一个完整长训练种子且其训练摘要未通过，不进入正式显著性表。",
        "",
    ])
    report_path = RESULT_ROOT / "stage26_locked_control_review.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "quality_passed": quality["passed"],
        "json": str(json_path), "csv": str(csv_path),
        "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
