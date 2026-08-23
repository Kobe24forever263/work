#!/usr/bin/env python3
"""Verify Reward V2 defeats the risk-free all-dog shortcut."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment, RewardConfig
from warehouse_core.stage14_training import (
    STAGE14_REWARD_CONFIG, WarehouseDispatchGymEnv)


def run_episode(seed, reward_config, avoid_handover):
    env = WarehouseDispatchGymEnv(
        seed=seed, execution_mode="CONCURRENT")
    _, info = env.reset(seed=seed)
    env.dispatch.reward_config = reward_config
    modes = Counter()
    reward_total = 0.0
    done = False
    while not done:
        rule_action = env.rule_action()
        rule_assignment = env.decision.action_ids[rule_action]
        action = rule_action
        if avoid_handover and isinstance(rule_assignment, Assignment):
            task = env.dispatch._task(rule_assignment.task_id).task
            preferred = ("SINGLE_DOG" if
                         task.source.floor != task.target.floor else
                         "SINGLE_CAR")
            alternatives = [
                (index, candidate)
                for index, (candidate, legal) in enumerate(zip(
                    env.decision.action_ids, info["action_mask"]))
                if legal and isinstance(candidate, Assignment)
                and candidate.task_id == rule_assignment.task_id
                and candidate.transport_mode == preferred]
            if alternatives:
                action = min(
                    alternatives,
                    key=lambda item: item[1].estimated_cost)[0]
        selected = env.decision.action_ids[action]
        if isinstance(selected, Assignment):
            modes[selected.transport_mode] += 1
        _, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        done = terminated or truncated
    dispatch = env.dispatch
    leak = bool(
        dispatch.resource_claims or
        getattr(dispatch, "active_standby_claims", {}) or
        getattr(dispatch, "active_tasks", {}) or
        any(runtime.robot.task_id or runtime.robot.cargo_id
            for runtime in dispatch.robots.values()))
    return {
        "reward": reward_total,
        "completed": dispatch.completed,
        "failed": dispatch.failed,
        "simulated_time": dispatch.now,
        "modes": dict(modes),
        "resource_leak": leak,
    }


def aggregate(rows):
    modes = Counter()
    for row in rows:
        modes.update(row["modes"])
    tasks = sum(row["completed"] + row["failed"] for row in rows)
    total_time = sum(row["simulated_time"] for row in rows)
    return {
        "episode_count": len(rows),
        "task_count": tasks,
        "success_rate": sum(row["completed"] for row in rows) / tasks,
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "throughput_tasks_per_hour": tasks / total_time * 3600,
        "transport_modes": dict(modes),
        "resource_leak_count": sum(row["resource_leak"] for row in rows),
    }


def main():
    seeds = range(20265001, 20265051)
    configs = {
        "LEGACY": RewardConfig(),
        "REWARD_V2": STAGE14_REWARD_CONFIG,
    }
    results = {}
    for name, config in configs.items():
        results[name] = {
            "efficient_rule": aggregate([
                run_episode(seed, config, False) for seed in seeds]),
            "risk_free_all_dog": aggregate([
                run_episode(seed, config, True) for seed in seeds]),
        }
    legacy = results["LEGACY"]
    v2 = results["REWARD_V2"]
    reward_fields = STAGE14_REWARD_CONFIG.__dataclass_fields__
    assertions = {
        "four_thousand_tasks_resolved": sum(
            item[policy]["task_count"] for item in results.values()
            for policy in item) == 4000,
        "legacy_reward_prefers_all_dog_shortcut": (
            legacy["risk_free_all_dog"]["mean_reward"] >
            legacy["efficient_rule"]["mean_reward"]),
        "reward_v2_prefers_efficient_mixed_transport": (
            v2["efficient_rule"]["mean_reward"] >
            v2["risk_free_all_dog"]["mean_reward"]),
        "efficient_policy_has_higher_throughput": (
            v2["efficient_rule"]["throughput_tasks_per_hour"] >
            v2["risk_free_all_dog"]["throughput_tasks_per_hour"]),
        "reward_has_no_transport_mode_specific_bonus": not any(
            token in field for field in reward_fields
            for token in ("mode", "dog", "car")),
        "all_runs_have_no_resource_leak": all(
            item[policy]["resource_leak_count"] == 0
            for item in results.values() for policy in item),
    }
    summary = {
        "stage": 14,
        "gate": "REWARD_V2_ANTI_MODE_COLLAPSE",
        "seed_start": 20265001,
        "seed_end": 20265050,
        "reward_v2": {
            "outstanding_time_scale":
                STAGE14_REWARD_CONFIG.outstanding_time_scale,
            "timeout_penalty": STAGE14_REWARD_CONFIG.timeout_penalty,
            "active_robot_time_penalty":
                STAGE14_REWARD_CONFIG.active_robot_time_penalty,
            "transport_mode_bonus": None,
        },
        "results": results,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = WORK_ROOT / "results" / "stage14" / "reward_v2"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "stage14_reward_v2_gate.summary.json"
    report_path = output / "stage14_reward_v2_gate_report.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report_path.write_text(
        "# Stage 14 Reward V2防模式塌缩门禁\n\n"
        "Reward V2不增加任何单车、单狗或车—狗—车专属奖金，只重新校准任务时间、"
        "失败和机器人占用的共同尺度。\n\n"
        "| 奖励 | 策略 | 平均回报 | 成功率 | 吞吐(任务/小时) |\n"
        "|---|---|---:|---:|---:|\n" +
        "\n".join(
            f"| {name} | {policy} | {values['mean_reward']:.4f} | "
            f"{values['success_rate']:.2%} | "
            f"{values['throughput_tasks_per_hour']:.3f} |"
            for name, rows in results.items()
            for policy, values in rows.items()) +
        "\n\n正式结论以重新训练后的独立留出评估为准，本门禁只验证奖励方向。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

