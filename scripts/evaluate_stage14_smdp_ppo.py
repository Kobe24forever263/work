#!/usr/bin/env python3
"""Evaluate a Stage 14 SMDP-PPO checkpoint against the rule baseline."""

import argparse
import json
import sys
from pathlib import Path

# NumPy must load before torch on the Apple Silicon ROS/Pixi environment.
import numpy as np  # noqa: F401
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import (
    baseline_relative_score, evaluate, load_checkpoint)


def compact(result):
    return {key: value for key, value in result.items()
            if key != "episodes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--execution-mode", choices=(
        "AUTO", "SERIAL", "CONCURRENT", "MIXED"), default="AUTO")
    parser.add_argument("--arrival-profile", choices=(
        "AUTO", "MEDIUM", "DENSE", "BURST"), default="AUTO")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20262801)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(args.weights, device)
    mode = (checkpoint["execution_mode"] if args.execution_mode == "AUTO"
            else args.execution_mode)
    profile = (checkpoint["arrival_profile"]
               if args.arrival_profile == "AUTO"
               else args.arrival_profile)
    observation_variant = checkpoint.get(
        "observation_variant", "FULL_CONTEXT_V2")
    seeds = list(range(args.seed, args.seed + args.episodes))
    policy = evaluate(
        model, seeds, mode, profile, device,
        observation_variant=observation_variant)
    rule = evaluate(
        model, seeds, mode, profile, device, use_rule=True,
        observation_variant=observation_variant)
    transport_modes = {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
    context_matrix = policy["selected_mode_by_cost_reference"]
    context_match_rates = {
        reference_mode: (
            selected_counts.get(reference_mode, 0) /
            max(sum(selected_counts.values()), 1))
        for reference_mode, selected_counts in context_matrix.items()
    }
    assertions = {
        "policy_resolved_all_tasks": policy["task_count"] == args.episodes * 20,
        "policy_actions_all_legal": policy["illegal_action_count"] == 0,
        "policy_has_no_resource_leak": policy["resource_leak_count"] == 0,
        "rule_has_no_resource_leak": rule["resource_leak_count"] == 0,
        "mode_conditioned_checkpoint_shape_matches": (
            checkpoint["model_metadata"]["state_width"] == 284 and
            checkpoint["model_metadata"]["action_width"] in {28, 30}),
        "policy_selects_all_three_modes": (
            set(policy["transport_modes"]) == transport_modes),
        "cost_reference_covers_all_three_modes": (
            set(context_matrix) == transport_modes),
        "policy_switches_by_context": all(
            context_match_rates.get(mode_name, 0.0) >= .20
            for mode_name in transport_modes),
    }
    comparison = {
        "success_rate_delta": policy["success_rate"] - rule["success_rate"],
        "mean_reward_delta": policy["mean_reward"] - rule["mean_reward"],
        "throughput_delta_tasks_per_hour": (
            policy["throughput_tasks_per_hour"] -
            rule["throughput_tasks_per_hour"]),
        "mean_simulated_time_delta": (
            policy["mean_simulated_time"] - rule["mean_simulated_time"]),
    }
    positive_relative_score = baseline_relative_score(
        policy["mean_reward"], rule["mean_reward"])
    output = args.output or args.weights.with_suffix(".evaluation.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_suffix(".md")
    summary = {
        "stage": 14,
        "gate": "SMDP_PPO_HELDOUT_EVALUATION",
        "weights": str(args.weights.resolve()),
        "execution_mode": mode,
        "arrival_profile": profile,
        "observation_variant": observation_variant,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "policy": compact(policy),
        "rule_baseline": compact(rule),
        "comparison": comparison,
        "positive_relative_score": positive_relative_score,
        "contextual_mode_acceptance": {
            "selected_mode_by_cost_reference": context_matrix,
            "diagonal_match_rates": context_match_rates,
            "minimum_diagonal_rate": .20,
            "note": (
                "This is conditional behavior, not a target frequency for "
                "any transport mode."),
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report.write_text(
        "# Stage 14 SMDP-PPO留出集评估\n\n"
        f"- 权重：`{summary['weights']}`\n"
        f"- 执行模式：`{mode}`\n"
        f"- 到达负载：`{profile}`\n"
        f"- 留出episode：{args.episodes}\n\n"
        "| 指标 | 学习策略 | 规则基线 |\n|---|---:|---:|\n"
        f"| 成功率 | {policy['success_rate']:.2%} | {rule['success_rate']:.2%} |\n"
        f"| 原始episode平均回报 | {policy['mean_reward']:.4f} | {rule['mean_reward']:.4f} |\n"
        f"| 吞吐(任务/小时) | {policy['throughput_tasks_per_hour']:.3f} | {rule['throughput_tasks_per_hour']:.3f} |\n"
        f"| 平均仿真时间(s) | {policy['mean_simulated_time']:.3f} | {rule['mean_simulated_time']:.3f} |\n"
        f"| 非法动作 | {policy['illegal_action_count']} | {rule['illegal_action_count']} |\n"
        f"| 资源泄漏 | {policy['resource_leak_count']} | {rule['resource_leak_count']} |\n\n"
        "## 按环境选择方案\n\n"
        "以下数据按当前任务与车队条件下的最低代价参考模式分组，检查同一策略是否随"
        "环境切换；它不是模式配额。\n\n"
        f"- 条件选择矩阵：`{json.dumps(context_matrix, ensure_ascii=False)}`\n"
        f"- 各参考模式对角一致率：`{json.dumps(context_match_rates, ensure_ascii=False)}`\n\n"
        "该报告比较的是逻辑仿真时间，不等同于RViz墙钟时间。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
