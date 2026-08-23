#!/usr/bin/env python3
"""Calibrate and validate the Stage 14 P95 handover timeout gate."""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.handover_timing import (
    HANDOVER_KINDS, HandoverTimeoutPolicy, build_provisional_calibration,
    nearest_rank_percentile)
from warehouse_core.stage14_training import WarehouseDispatchGymEnv


def describe(values):
    return {
        "count": len(values),
        "minimum_s": min(values),
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "p95_s": nearest_rank_percentile(values, .95),
        "p99_s": nearest_rank_percentile(values, .99),
        "maximum_s": max(values),
    }


def run_episode(seed):
    env = WarehouseDispatchGymEnv(seed=seed)
    _, info = env.reset(seed=seed)
    task_rows = []
    done = False
    while not done:
        action = env.rule_action()
        _, reward, terminated, truncated, info = env.step(action)
        selected = info["selected_action"]
        if selected.get("task_id"):
            task_rows.append({
                "seed": seed,
                "task_id": selected["task_id"],
                "task_type": info["task_type"],
                "transport_mode": selected.get("transport_mode", ""),
                "task_result": selected.get("task_result", ""),
                "failure_reason": selected.get("failure_reason", ""),
                "handover_durations_s": selected.get(
                    "handover_durations_s", []),
                "handover_timeout_s": selected.get("handover_timeout_s"),
                "reward": reward,
            })
        done = terminated or truncated
    dispatch = env.dispatch
    return {
        "task_rows": task_rows,
        "handover_events": [dict(event, seed=seed)
                            for event in dispatch.handover_events],
        "completed": dispatch.completed,
        "failed": dispatch.failed,
        "resolved": dispatch.completed + dispatch.failed,
        "resource_leak": bool(dispatch.resource_claims),
        "robot_state_leak": any(
            runtime.robot.task_id or runtime.robot.cargo_id
            for runtime in dispatch.robots.values()),
        "stair_leak": any(stair["state"] != "FREE"
                          for stair in dispatch.stairs.values()),
        "termination_reason": info["termination_reason"],
    }


def replay_signature(result):
    return [(
        row["task_id"], row["transport_mode"], row["task_result"],
        row["failure_reason"],
        tuple(round(value, 12) for value in row["handover_durations_s"]))
        for row in result["task_rows"]]


def main():
    samples = build_provisional_calibration()
    policy = HandoverTimeoutPolicy.from_samples(samples)
    durations = [sample.duration_s for sample in samples]
    calibration_timed_out = [sample for sample in samples
                             if policy.timed_out(sample.duration_s)]
    per_kind = {}
    for kind in HANDOVER_KINDS:
        values = [sample.duration_s for sample in samples
                  if sample.handover_kind == kind]
        per_kind[kind] = {
            **describe(values),
            "above_unified_p95_count": sum(
                value > policy.timeout_s for value in values),
            "above_unified_p95_rate": sum(
                value > policy.timeout_s for value in values) / len(values),
        }

    seeds = list(range(20261001, 20261501))
    episodes = []
    task_rows = []
    handover_events = []
    for index, seed in enumerate(seeds, start=1):
        episode = run_episode(seed)
        episodes.append(episode)
        task_rows.extend(episode["task_rows"])
        handover_events.extend(episode["handover_events"])
        if index % 50 == 0:
            print(f"[{index:03d}/{len(seeds)}] resolved="
                  f"{sum(item['resolved'] for item in episodes)}")

    replay = run_episode(seeds[0])
    replay_equal = replay_signature(replay) == replay_signature(episodes[0])
    timeout_events = [event for event in handover_events
                      if event["timed_out"]]
    failed_rows = [row for row in task_rows
                   if row["task_result"] == "FAILED"]
    cooperative_rows = [row for row in task_rows
                        if row["transport_mode"] == "CAR_DOG_CAR"]
    mode_counts = Counter(row["transport_mode"] for row in task_rows)
    p = len(calibration_timed_out) / len(samples)
    expected_two_handover_task_failure_rate = 1 - (1 - p) ** 2
    empirical_attempt_timeout_rate = len(timeout_events) / len(handover_events)
    empirical_cooperative_task_failure_rate = (
        len(failed_rows) / len(cooperative_rows))

    assertions = {
        "calibration_has_1000_current_mode_samples": (
            len(samples) == 1000 and
            {sample.handover_kind for sample in samples} ==
            set(HANDOVER_KINDS)),
        "nearest_rank_p95_has_exactly_five_percent_above": (
            len(calibration_timed_out) == 50),
        "strict_boundary_accepts_equal_and_rejects_above": (
            not policy.timed_out(policy.timeout_s) and
            policy.timed_out(math.nextafter(policy.timeout_s, math.inf))),
        "five_hundred_episodes_resolve_ten_thousand_tasks": (
            len(episodes) == 500 and len(task_rows) == 10000 and
            all(item["resolved"] == 20 for item in episodes)),
        "logical_attempt_timeout_rate_is_close_to_five_percent": (
            .04 <= empirical_attempt_timeout_rate <= .06),
        "cooperative_task_failure_rate_matches_two_attempt_exposure": (
            .065 <= empirical_cooperative_task_failure_rate <= .135),
        "every_failed_task_is_handover_timeout": (
            bool(failed_rows) and len(failed_rows) == len(timeout_events) and
            all(row["failure_reason"] == "HANDOVER_TIMEOUT"
                for row in failed_rows)),
        "no_resource_robot_or_stair_leaks": all(
            not item["resource_leak"] and
            not item["robot_state_leak"] and
            not item["stair_leak"] for item in episodes),
        "fixed_seed_replay_equal": replay_equal,
    }
    summary = {
        "stage": 14,
        "gate": "P95_HANDOVER_TIMEOUT_LOGIC_GATE",
        "data_quality_status": "PROVISIONAL_LOGIC_SIMULATION_ONLY",
        "eligible_for_pre_rl_logical_training": True,
        "eligible_as_rviz_or_hardware_threshold": False,
        "calibration": {
            "source": policy.source,
            "seed": 20260809,
            "timing_start": "BOTH_AGENTS_STABLE_ALIGNMENT_START",
            "timing_end": "UNIQUE_OWNER_TRANSFER_VERIFIED",
            "percentile_method": "NEAREST_RANK_CEIL",
            "percentile": policy.percentile,
            "comparison": policy.comparison,
            "timeout_s": policy.timeout_s,
            "all": describe(durations),
            "per_kind": per_kind,
            "above_p95_count": len(calibration_timed_out),
            "above_p95_rate": p,
        },
        "logical_gate": {
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "episode_count": len(episodes),
            "task_count": len(task_rows),
            "completed_task_count": sum(
                row["task_result"] == "COMPLETED" for row in task_rows),
            "failed_task_count": len(failed_rows),
            "transport_modes": dict(mode_counts),
            "cooperative_task_count": len(cooperative_rows),
            "handover_attempt_count": len(handover_events),
            "handover_timeout_count": len(timeout_events),
            "handover_timeout_rate": empirical_attempt_timeout_rate,
            "cooperative_task_failure_rate":
                empirical_cooperative_task_failure_rate,
            "expected_two_handover_task_failure_rate_if_independent":
                expected_two_handover_task_failure_rate,
        },
        "known_limitations": [
            "Stage 8 records have no start/end timestamps and cannot estimate P95.",
            "Calibration samples are reproducible logical-simulation values, not RViz or hardware measurements.",
            "A cooperative task has two handovers, so its failure probability is about 1-(0.95^2)=9.75%, not 5%.",
            "Distribution drift can move the observed timeout rate away from 5%.",
        ],
        "replacement_gate": {
            "minimum_measured_samples_per_kind": 500,
            "required_kinds": list(HANDOVER_KINDS),
            "required_fields": [
                "task_id", "handover_kind", "start_timestamp",
                "end_timestamp", "duration_s", "result"],
            "action": "Recompute the same nearest-rank P95 and rerun this gate.",
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }

    output = WORK_ROOT / "results" / "stage14"
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage14_handover_timing_calibration.jsonl").write_text(
        "".join(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
                for sample in samples), encoding="utf-8")
    (output / "stage14_handover_timeout_gate.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n"
                for event in handover_events), encoding="utf-8")
    (output / "stage14_handover_timeout_gate.tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in task_rows), encoding="utf-8")
    (output / "stage14_handover_timeout_gate.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    report = f"""# Stage 14 P95 交接超时逻辑门

## 结论

当前逻辑门{'通过' if summary['passed'] else '未通过'}。统一暂定 P95 为 `{policy.timeout_s:.6f} s`；严格大于该值的单次交接记为 `HANDOVER_TIMEOUT`，当前任务失败且不重试，资源、货物占用和机器人任务状态必须全部释放。

## 数据可信度

现有 Stage 8 的 300 条记录没有开始与结束时间，不能用于 P95。这里的 1000 条样本（车→狗 500、狗→车 500）属于带固定随机种子的逻辑仿真标定，只允许用于正式强化学习前的逻辑训练，不得宣称为 RViz 或实机交接性能。后续每类至少收集 500 条带时间戳的测量记录后，按同一 nearest-rank 方法替换阈值。

## 统计结果

- 标定样本超过 P95：`{len(calibration_timed_out)}/{len(samples)}`（`{p:.2%}`）。
- 500 个持久状态回合：`{len(task_rows)}` 个任务，其中成功 `{sum(row['task_result'] == 'COMPLETED' for row in task_rows)}`，交接超时失败 `{len(failed_rows)}`。
- 交接尝试：`{len(handover_events)}` 次，超时 `{len(timeout_events)}` 次（`{empirical_attempt_timeout_rate:.2%}`）。
- 协作任务：`{len(cooperative_rows)}` 个，任务失败率 `{empirical_cooperative_task_failure_rate:.2%}`；因为每个车—狗—车任务有两次交接，理论暴露失败率约为 `{expected_two_handover_task_failure_rate:.2%}`。
- 三种运输模式：`{dict(mode_counts)}`。

## 失败语义

`duration_s > P95` 才超时；`duration_s == P95` 仍成功。超时事务先回滚到唯一安全所有者，然后任务层将货物标为失败、清空参与机器人携货与任务字段、释放楼梯/交接点/区域资源，并让机器人从当前位置进入待命位。任务失败不立刻终止整个回合，只有回合已处理任务数达到上限时才正常结束。

## 验收断言

```json
{json.dumps(assertions, ensure_ascii=False, indent=2)}
```
"""
    (output / "stage14_handover_timeout_gate_report.md").write_text(
        report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
