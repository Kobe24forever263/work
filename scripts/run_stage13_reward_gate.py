#!/usr/bin/env python3
"""Stage 13 decomposed-reward, discount, and mode-fairness gate."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(WORK_ROOT / "scripts"))

from run_stage11_persistent_20 import build_environment
from warehouse_core.domain import Point, Robot, Task
from warehouse_core.persistent_dispatch import (
    AsyncTaskQueue, PersistentDispatchEnvironment, QueuedTask, RobotRuntime,
    StandbySlot)


def _runtime(robot_id, kind, floor, xyz):
    region = f"F{floor}_TEST"
    return RobotRuntime(
        Robot(robot_id, kind, floor, region, floor, region), xyz)


def _fairness_env(dog_favored: bool):
    source = Point("SRC", 1, "F1_TEST", (0, 0, .45) if dog_favored
                   else (-10, 0, .45))
    target = Point("DST", 2, "F2_TEST", (0, 0, 4.7) if dog_favored
                   else (10, 0, 4.7))
    robots = [
        _runtime("car1", "car", 1,
                 (-20, 0, .45) if dog_favored else source.xyz),
        _runtime("dog", "dog", 1, source.xyz if dog_favored
                 else (0, 0, .45)),
        _runtime("car2", "car", 2,
                 (20, 0, 4.7) if dog_favored else target.xyz),
    ]
    stairs = {"S": {"floor1_xyz": (0, 0, .45),
                      "floor2_xyz": (0, 0, 4.7), "state": "FREE"}}
    task = Task("T", "C", source, target, 1, 1000)
    slots = [
        StandbySlot("C1", "car", 1, (0, 0, .45)),
        StandbySlot("C2", "car", 2, target.xyz),
        StandbySlot("D2", "dog", 2, target.xyz),
    ]
    if target.xyz != (0, 0, 4.7):
        slots.append(StandbySlot("D2_EXIT", "dog", 2, (0, 0, 4.7)))
    return PersistentDispatchEnvironment(
        robots, stairs, AsyncTaskQueue([QueuedTask(task, 0)]), task_limit=1,
        decision_gap=0, standby_slots=slots)


def _evaluate_mode(dog_favored: bool, mode: str):
    env = _fairness_env(dog_favored)
    candidates = env.enumerate_candidate_actions()
    index = next(index for index, action in enumerate(candidates)
                 if action.transport_mode == mode)
    action = candidates[index]
    transition = env.step(index)
    return {
        "mode": mode,
        "estimated_cost": round(action.estimated_cost, 6),
        "delta_time": round(transition.delta_time, 6),
        "reward": round(transition.reward, 6),
        "components": {key: round(value, 6) for key, value in
                       transition.reward_components.items()},
    }


def mode_fairness_gate():
    env = _fairness_env(False)
    single_dog = env.SINGLE_DOG
    cooperative = env.CAR_DOG_CAR
    coop_case = {
        single_dog: _evaluate_mode(False, single_dog),
        cooperative: _evaluate_mode(False, cooperative),
    }
    dog_case = {
        single_dog: _evaluate_mode(True, single_dog),
        cooperative: _evaluate_mode(True, cooperative),
    }
    return {
        "cooperation_favored_geometry": coop_case,
        "single_dog_favored_geometry": dog_case,
        "cooperation_wins_when_faster": (
            coop_case[cooperative]["reward"] > coop_case[single_dog]["reward"]),
        "single_dog_wins_when_faster": (
            dog_case[single_dog]["reward"] > dog_case[cooperative]["reward"]),
    }


def run(seed: int, profile: str):
    env, task_types = build_environment(seed, profile)
    rows = []
    while env.completed < env.task_limit:
        env.queue.advance(env.now)
        candidates = env.enumerate_candidate_actions()
        if not candidates:
            env.now += 1.0
            if env.now >= env.time_limit:
                raise RuntimeError("no legal action before time limit")
            continue
        selected = env.select_rule_action(candidates)
        action = candidates[selected]
        transition = env.step(selected)
        rows.append({
            "decision": len(rows) + 1,
            "task_id": action.task_id,
            "task_type": task_types[action.task_id],
            "transport_mode": action.transport_mode,
            "reward": transition.reward,
            "reward_components": transition.reward_components,
            "delta_time": transition.delta_time,
            "discount": transition.discount,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "termination_reason": transition.termination_reason,
        })
        print(f"[{env.completed}/20] {action.task_id} "
              f"mode={action.transport_mode} reward={transition.reward:.4f}: PASS")
    task_counts = Counter(row["task_type"] for row in rows)
    mode_counts = Counter(row["transport_mode"] for row in rows)
    fairness = mode_fairness_gate()
    assertions = {
        "twenty_tasks_completed": env.completed == 20,
        "six_task_types_covered": len(task_counts) == 6,
        "reward_components_sum_exactly": all(abs(
            row["reward"] - sum(row["reward_components"].values())) < 1e-9
            for row in rows),
        "rewards_and_discounts_finite": all(
            math.isfinite(row["reward"]) and 0 < row["discount"] <= 1
            for row in rows),
        "time_aware_discount_varies": len({round(row["discount"], 9)
                                            for row in rows}) > 1,
        "only_final_transition_terminates": (
            sum(row["terminated"] for row in rows) == 1 and
            rows[-1]["termination_reason"] == "TASK_LIMIT"),
        "no_unexpected_truncation": not any(row["truncated"] for row in rows),
        "reward_not_fixed_to_one_mode": (
            fairness["cooperation_wins_when_faster"] and
            fairness["single_dog_wins_when_faster"]),
    }
    summary = {
        "stage": 13,
        "seed": seed,
        "profile": profile,
        "trial_count": len(rows),
        "success_count": len(rows) if all(assertions.values()) else 0,
        "failure_count": 0 if all(assertions.values()) else 1,
        "task_types": dict(task_counts),
        "transport_modes": dict(mode_counts),
        "reward_total": round(sum(row["reward"] for row in rows), 6),
        "reward_mean": round(sum(row["reward"] for row in rows) / len(rows), 6),
        "discount_min": round(min(row["discount"] for row in rows), 9),
        "discount_max": round(max(row["discount"] for row in rows), 9),
        "mode_fairness": fairness,
        "assertions": assertions,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--profile", choices=("normal", "medium", "overload"),
                        default="medium")
    args = parser.parse_args()
    rows, summary = run(args.seed, args.profile)
    output = WORK_ROOT / "results" / "stage13"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"stage13_reward_{args.profile}_20"
    (output / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (output / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(summary["assertions"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
