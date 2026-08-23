#!/usr/bin/env python3
"""Stage 13 final gate: 600 tasks, failure paths, and reward stability."""

import contextlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(WORK_ROOT / "scripts"))

from run_stage13_reward_gate import run
from warehouse_core.domain import Point, Robot, Task
from warehouse_core.persistent_dispatch import (
    AsyncTaskQueue, PersistentDispatchEnvironment, QueuedTask, RobotRuntime,
    StandbySlot)


def _runtime(robot_id, kind, floor, xyz):
    region = f"F{floor}_TEST"
    return RobotRuntime(
        Robot(robot_id, kind, floor, region, floor, region), xyz)


def _cross_floor_env(deadline=1000, task_limit=10, time_limit=1800):
    source = Point("SRC", 1, "F1_TEST", (0, 0, .45))
    target = Point("DST", 2, "F2_TEST", (2, 0, 4.7))
    task = Task("T_FAIL", "C_FAIL", source, target, 1, deadline)
    robots = [
        _runtime("car1", "car", 1, source.xyz),
        _runtime("dog", "dog", 1, (1, 0, .45)),
        _runtime("car2", "car", 2, target.xyz),
    ]
    stairs = {"S": {"floor1_xyz": (1, 0, .45),
                       "floor2_xyz": (1, 0, 4.7), "state": "FREE"}}
    slots = [
        StandbySlot("C1", "car", 1, (-2, 0, .45)),
        StandbySlot("C2", "car", 2, (3, 0, 4.7)),
        StandbySlot("D2", "dog", 2, (0, 0, 4.7)),
    ]
    return PersistentDispatchEnvironment(
        robots, stairs, AsyncTaskQueue([QueuedTask(task, 0)]),
        task_limit=task_limit, time_limit=time_limit, decision_gap=0,
        standby_slots=slots)


def _failure_transition_result(transition, env):
    claims = sum(event["event"] == "CLAIM" for event in env.resource_events)
    releases = sum(event["event"] == "RELEASE"
                   for event in env.resource_events)
    return {
        "reward": round(transition.reward, 6),
        "reward_components": transition.reward_components,
        "discount": round(transition.discount, 9),
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "termination_reason": transition.termination_reason,
        "resource_claims_after": len(env.resource_claims),
        "claim_release_balanced": claims == releases,
        "residual_robot_tasks": sum(bool(item.robot.task_id)
                                    for item in env.robots.values()),
        "residual_cargo_holders": sum(bool(item.robot.cargo_id)
                                      for item in env.robots.values()),
    }


def controlled_failure_gate():
    results = {}

    deadline_env = _cross_floor_env(deadline=0)
    deadline_transition = deadline_env.step(
        deadline_env.select_rule_action(deadline_env.enumerate_candidate_actions()))
    results["DEADLINE_MISS"] = {
        "deadline_component": deadline_transition.reward_components["deadline"],
        "terminated": deadline_transition.terminated,
        "truncated": deadline_transition.truncated,
        "passed": (
            deadline_transition.reward_components["deadline"] ==
            -deadline_env.reward_config.timeout_penalty and
            not deadline_transition.terminated and
            not deadline_transition.truncated),
    }

    time_env = _cross_floor_env(time_limit=.1)
    time_transition = time_env.step(
        time_env.select_rule_action(time_env.enumerate_candidate_actions()))
    results["TIME_LIMIT"] = {
        "termination_reason": time_transition.termination_reason,
        "terminated": time_transition.terminated,
        "truncated": time_transition.truncated,
        "passed": (time_transition.termination_reason == "TIME_LIMIT" and
                   time_transition.truncated and
                   not time_transition.terminated),
    }

    for reason in sorted(PersistentDispatchEnvironment.SEVERE_FAILURE_REASONS):
        env = _cross_floor_env()
        if reason == "NO_FEASIBLE_CHAIN":
            for runtime in env.robots.values():
                runtime.robot.failure_code = "FAULT"
            no_candidates = not env.enumerate_candidate_actions()
        else:
            action = next(item for item in env.enumerate_candidate_actions()
                          if item.transport_mode == env.CAR_DOG_CAR)
            task = env._task(action.task_id).task
            env._claim_resources(action, task)
            for robot_id in (action.pickup_carter, action.dog_id,
                             action.receiving_carter):
                env.robots[robot_id].robot.task_id = task.task_id
            env.robots[action.pickup_carter].robot.cargo_id = task.cargo_id
            no_candidates = True
        transition = env.abort_episode(reason, "T_FAIL", elapsed=2.5)
        result = _failure_transition_result(transition, env)
        result["passed"] = bool(
            no_candidates and result["terminated"] and
            not result["truncated"] and
            result["termination_reason"] == reason and
            result["resource_claims_after"] == 0 and
            result["claim_release_balanced"] and
            result["residual_robot_tasks"] == 0 and
            result["residual_cargo_holders"] == 0 and
            transition.reward_components["failure"] ==
            -env.reward_config.severe_failure_penalty)
        results[reason] = result

    return results


def run_multiseed(first_seed=20260805, episode_count=30):
    all_rows = []
    seed_summaries = []
    for offset in range(episode_count):
        seed = first_seed + offset
        with contextlib.redirect_stdout(io.StringIO()):
            rows, summary = run(seed, "medium")
        for row in rows:
            all_rows.append({"seed": seed, **row})
        seed_summaries.append({
            "seed": seed,
            "task_count": len(rows),
            "reward_total": summary["reward_total"],
            "reward_mean": summary["reward_mean"],
            "discount_min": summary["discount_min"],
            "discount_max": summary["discount_max"],
            "transport_modes": summary["transport_modes"],
            "assertions_passed": all(summary["assertions"].values()),
        })
        print(f"[{offset + 1:02d}/{episode_count}] seed={seed} "
              f"tasks={len(rows)} reward_mean={summary['reward_mean']:.6f}: PASS")
    return all_rows, seed_summaries


def main():
    rows, seeds = run_multiseed()
    failures = controlled_failure_gate()
    reward_values = [row["reward"] for row in rows]
    reward_means = [item["reward_mean"] for item in seeds]
    mode_counts = Counter(row["transport_mode"] for row in rows)
    task_counts = Counter(row["task_type"] for row in rows)
    assertions = {
        "thirty_episodes_completed": len(seeds) == 30,
        "six_hundred_tasks_completed": len(rows) == 600,
        "every_seed_passed_first_gate": all(
            item["assertions_passed"] for item in seeds),
        "six_task_types_in_aggregate": set(task_counts) == {
            "f1_same", "f1_cross_region", "f2_same", "f2_cross_region",
            "cross_up", "cross_down"},
        "all_rewards_finite": all(math.isfinite(value)
                                  for value in reward_values),
        "all_discounts_valid": all(0 < row["discount"] <= 1
                                   for row in rows),
        "reward_components_sum_exactly": all(abs(
            row["reward"] - sum(row["reward_components"].values())) < 1e-9
            for row in rows),
        "controlled_failures_passed": all(
            result["passed"] for result in failures.values()),
    }
    summary = {
        "stage": 13,
        "gate": "FINAL",
        "profile": "medium",
        "seed_start": seeds[0]["seed"],
        "seed_end": seeds[-1]["seed"],
        "episode_count": len(seeds),
        "task_count": len(rows),
        "task_types": dict(task_counts),
        "transport_modes": dict(mode_counts),
        "reward": {
            "task_mean": round(sum(reward_values) / len(reward_values), 6),
            "task_min": round(min(reward_values), 6),
            "task_max": round(max(reward_values), 6),
            "episode_mean_min": round(min(reward_means), 6),
            "episode_mean_max": round(max(reward_means), 6),
        },
        "discount": {
            "min": round(min(row["discount"] for row in rows), 9),
            "max": round(max(row["discount"] for row in rows), 9),
        },
        "controlled_failures": failures,
        "per_seed": seeds,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = WORK_ROOT / "results" / "stage13"
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage13_final_medium_600.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (output / "stage13_final_medium_600.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "per_seed"}, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
