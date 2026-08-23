#!/usr/bin/env python3
"""Paired Stage 15 comparison of independently trained SERIAL/CONCURRENT PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# NumPy must load before torch in the Apple Silicon Pixi environment.
import numpy as np  # noqa: F401
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "scripts"))
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from run_stage15_stress_evaluation import (  # noqa: E402
    SCENARIOS,
    compact,
    paired_bootstrap,
    run_policy,
)
from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402


METRIC_LABELS = {
    "success_rate": "成功率",
    "reward": "原始episode回报",
    "throughput_tasks_per_hour": "已解决吞吐(任务/h)",
    "successful_throughput_tasks_per_hour": "成功吞吐(任务/h)",
    "mean_waiting_time": "平均等待(s)",
    "p95_waiting_time": "P95等待(s)",
    "mean_flow_time": "平均流转(s)",
    "p95_flow_time": "P95流转(s)",
    "distance_per_task": "每任务总里程(m)",
}


def comparison_with_clear_names(concurrent, serial, seed):
    raw = paired_bootstrap(concurrent, serial, seed)
    return {
        key: {
            "concurrent_minus_serial": value["policy_minus_rule"],
            "paired_bootstrap_95ci": value["paired_bootstrap_95ci"],
        }
        for key, value in raw.items()
    }


def delta_text(item, digits=3):
    low, high = item["paired_bootstrap_95ci"]
    value = item["concurrent_minus_serial"]
    return f"{value:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-weights", type=Path, required=True)
    parser.add_argument("--concurrent-weights", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20266501)
    parser.add_argument(
        "--scenarios", nargs="+", choices=tuple(SCENARIOS),
        default=("MEDIUM", "DENSE", "BURST"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")

    torch.set_num_threads(4)
    serial_model, serial_checkpoint = load_checkpoint(
        args.serial_weights, torch.device("cpu"))
    concurrent_model, concurrent_checkpoint = load_checkpoint(
        args.concurrent_weights, torch.device("cpu"))
    seeds = list(range(args.seed, args.seed + args.episodes))

    results = {}
    assertions = {}
    for offset, name in enumerate(args.scenarios):
        scenario = SCENARIOS[name]
        serial = run_policy(
            serial_model, seeds, scenario, "PPO", "SERIAL")
        concurrent = run_policy(
            concurrent_model, seeds, scenario, "PPO", "CONCURRENT")
        comparison = comparison_with_clear_names(
            concurrent, serial, args.seed + 4000 + offset)
        scenario_assertions = {
            "task_streams_identical": (
                serial["task_fingerprint"] == concurrent["task_fingerprint"]),
            "serial_resolved_all_tasks": (
                serial["task_count"] == args.episodes * scenario.task_count),
            "concurrent_resolved_all_tasks": (
                concurrent["task_count"] == args.episodes * scenario.task_count),
            "serial_actions_all_legal": serial["illegal_action_count"] == 0,
            "concurrent_actions_all_legal": (
                concurrent["illegal_action_count"] == 0),
            "serial_has_no_resource_leak": serial["resource_leak_count"] == 0,
            "concurrent_has_no_resource_leak": (
                concurrent["resource_leak_count"] == 0),
            "serial_maximum_active_tasks_is_one": (
                serial["maximum_active_tasks"] == 1),
            "both_policies_expose_all_transport_modes": (
                set(serial["transport_modes"]) == {
                    "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"} and
                set(concurrent["transport_modes"]) == {
                    "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}),
        }
        assertions.update({
            f"{name.lower()}_{key}": value
            for key, value in scenario_assertions.items()})
        results[name] = {
            "description": scenario.description,
            "task_count_per_policy": args.episodes * scenario.task_count,
            "serial": compact(serial),
            "concurrent": compact(concurrent),
            "paired_comparison": comparison,
            "assertions": scenario_assertions,
        }
        print(
            f"{name}: SERIAL reward={serial['mean_episode_reward']:.3f} "
            f"success={serial['success_rate']:.2%}; CONCURRENT reward="
            f"{concurrent['mean_episode_reward']:.3f} "
            f"success={concurrent['success_rate']:.2%}")

    summary = {
        "stage": 15,
        "gate": "PAIRED_SERIAL_VS_CONCURRENT_PPO_ABLATION",
        "claim_boundary": (
            "Both policies use independently trained weights and identical "
            "task streams/common random numbers. Differences quantify the "
            "complete execution-mode systems, not only a scheduler flag."),
        "serial_weights": str(args.serial_weights.resolve()),
        "serial_checkpoint_update": serial_checkpoint.get("update"),
        "concurrent_weights": str(args.concurrent_weights.resolve()),
        "concurrent_checkpoint_update": concurrent_checkpoint.get("update"),
        "episode_count_per_scenario": args.episodes,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
        "difference_direction": "CONCURRENT_MINUS_SERIAL",
        "results": results,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        "# Stage 15 SERIAL与CONCURRENT PPO配对消融", "",
        f"- SERIAL权重：`{summary['serial_weights']}`（update "
        f"{summary['serial_checkpoint_update']}）",
        f"- CONCURRENT权重：`{summary['concurrent_weights']}`（update "
        f"{summary['concurrent_checkpoint_update']}）",
        f"- 每场景episode：{args.episodes}",
        f"- 固定种子：{seeds[0]}～{seeds[-1]}",
        "- 所有差值均为CONCURRENT减SERIAL；任务与交接随机样本严格配对。",
        "", "## 汇总", "",
        "| 场景 | SERIAL成功率 | CONCURRENT成功率 | SERIAL成功吞吐 | CONCURRENT成功吞吐 | SERIAL最大在途 | CONCURRENT最大在途 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        serial = item["serial"]
        concurrent = item["concurrent"]
        lines.append(
            f"| {name} | {serial['success_rate']:.2%} | "
            f"{concurrent['success_rate']:.2%} | "
            f"{serial['successful_throughput_tasks_per_hour']:.3f} | "
            f"{concurrent['successful_throughput_tasks_per_hour']:.3f} | "
            f"{serial['maximum_active_tasks']} | "
            f"{concurrent['maximum_active_tasks']} |")
    for name, item in results.items():
        lines.extend([
            "", f"## {name}配对差值", "",
            "| 指标 | CONCURRENT−SERIAL [95% CI] |",
            "|---|---:|",
        ])
        for key, label in METRIC_LABELS.items():
            digits = 4 if key == "success_rate" else 3
            lines.append(
                f"| {label} | "
                f"{delta_text(item['paired_comparison'][key], digits)} |")
        lines.extend([
            "", "运输模式：", "",
            "| 策略 | 单车 | 单狗 | 车—狗—车 |",
            "|---|---:|---:|---:|",
            f"| SERIAL | {item['serial']['transport_modes'].get('SINGLE_CAR', 0)} | "
            f"{item['serial']['transport_modes'].get('SINGLE_DOG', 0)} | "
            f"{item['serial']['transport_modes'].get('CAR_DOG_CAR', 0)} |",
            f"| CONCURRENT | {item['concurrent']['transport_modes'].get('SINGLE_CAR', 0)} | "
            f"{item['concurrent']['transport_modes'].get('SINGLE_DOG', 0)} | "
            f"{item['concurrent']['transport_modes'].get('CAR_DOG_CAR', 0)} |",
        ])
    lines.extend([
        "", "## 结论边界", "",
        "- PASS只表示任务配对、安全和结果完整，不等于CONCURRENT所有指标均更好。",
        "- 置信区间跨0的指标不宣称存在显著差异。",
        "- 两套权重独立训练，因此结果衡量完整SERIAL/CONCURRENT系统差异。",
    ])
    output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "report": str(output.with_suffix('.md')),
        "passed": summary["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
