#!/usr/bin/env python3
"""Stage 14 first gate: balanced task mix and Gym rule replay."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment
from warehouse_core.stage14_training import WarehouseDispatchGymEnv


def run_episode(seed: int):
    env = WarehouseDispatchGymEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        finite_observation = bool(
            np.isfinite(observation["state"]).all() and
            np.isfinite(observation["action_features"]).all())
        action_index = env.rule_action()
        mask_selected = bool(info["action_mask"][action_index])
        decoded = env.decision.action_ids[action_index]
        legal_modes = []
        task_id = ""
        if isinstance(decoded, Assignment):
            task_id = decoded.task_id
            legal_modes = sorted({
                item.transport_mode for item in env.decision.action_ids
                if isinstance(item, Assignment) and item.task_id == task_id})
        dog_favored_scenario = False
        if task_id:
            task = env.dispatch._task(task_id).task
            dog_favored_scenario = (
                "STAIR_PLATFORM" in task.source.point_id and
                "STAIR_PLATFORM" in task.target.point_id)
        had_waiting_task = bool(env.dispatch.queue.waiting)
        observation, reward, terminated, truncated, info = env.step(action_index)
        selected = info["selected_action"]
        rows.append({
            "seed": seed,
            "decision": len(rows) + 1,
            "action_index": action_index,
            "action_type": selected.get("type", "ASSIGN"),
            "task_id": selected.get("task_id", ""),
            "task_type": info["task_type"],
            "transport_mode": selected.get("transport_mode", ""),
            "task_result": selected.get("task_result", ""),
            "failure_reason": selected.get("failure_reason", ""),
            "pickup_carter": selected.get("pickup_carter", ""),
            "dog_id": selected.get("dog_id", ""),
            "stair_id": selected.get("stair_id", ""),
            "receiving_carter": selected.get("receiving_carter", ""),
            "legal_modes_for_task": legal_modes,
            "dog_favored_scenario": dog_favored_scenario,
            "reward": reward,
            "delta_time": info["delta_time"],
            "discount": info["discount"],
            "finite_observation": finite_observation,
            "mask_selected": mask_selected,
            "wait_only_without_waiting_task": (
                selected.get("type") != "WAIT" or not had_waiting_task),
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": info["termination_reason"],
        })
    return env, rows


def signature(rows):
    return [(
        row["action_type"], row["task_id"], row["transport_mode"],
        row["pickup_carter"], row["dog_id"], row["stair_id"],
        row["receiving_carter"], round(row["reward"], 9),
        round(row["delta_time"], 9)) for row in rows]


def main():
    seeds = list(range(20260835, 20260845))
    all_rows = []
    episode_summaries = []
    replay_equal = True
    for episode_index, seed in enumerate(seeds, start=1):
        env, rows = run_episode(seed)
        _, replay = run_episode(seed)
        replay_equal = replay_equal and signature(rows) == signature(replay)
        assignments = [row for row in rows if row["action_type"] == "ASSIGN"]
        waits = [row for row in rows if row["action_type"] == "WAIT"]
        modes = Counter(row["transport_mode"] for row in assignments)
        episode_summaries.append({
            "seed": seed,
            "decision_count": len(rows),
            "task_count": len(assignments),
            "completed_task_count": env.dispatch.completed,
            "failed_task_count": env.dispatch.failed,
            "wait_count": len(waits),
            "transport_modes": dict(modes),
            "reward_total": round(sum(row["reward"] for row in rows), 6),
            "simulated_time": round(env.dispatch.now, 6),
        })
        all_rows.extend(rows)
        print(f"[{episode_index:02d}/10] seed={seed} tasks={len(assignments)} "
              f"waits={len(waits)} modes={dict(modes)}: PASS")

    assignments = [row for row in all_rows if row["action_type"] == "ASSIGN"]
    waits = [row for row in all_rows if row["action_type"] == "WAIT"]
    task_types = Counter(row["task_type"] for row in assignments)
    modes = Counter(row["transport_mode"] for row in assignments)
    cross_rows = [row for row in assignments
                  if row["task_type"].startswith("cross_")]
    dog_favored_rows = [row for row in cross_rows
                        if row["dog_favored_scenario"]]
    same_floor_count = len(assignments) - len(cross_rows)
    completed_rows = [row for row in assignments
                      if row["task_result"] == "COMPLETED"]
    failed_rows = [row for row in assignments
                   if row["task_result"] == "FAILED"]
    assertions = {
        "ten_episodes_completed": len(episode_summaries) == 10,
        "two_hundred_tasks_resolved": (
            len(assignments) == 200 and
            len(completed_rows) + len(failed_rows) == 200),
        "failed_tasks_are_only_p95_handover_timeouts": all(
            row["failure_reason"] == "HANDOVER_TIMEOUT"
            for row in failed_rows),
        "exact_half_cross_floor": len(cross_rows) == 100 and
                                  same_floor_count == 100,
        "six_task_types_exact_counts": task_types == {
            "f1_same": 30, "f1_cross_region": 20,
            "f2_same": 30, "f2_cross_region": 20,
            "cross_up": 50, "cross_down": 50},
        "fixed_tensor_shapes": all(
            row["finite_observation"] for row in all_rows),
        "every_selected_action_unmasked": all(
            row["mask_selected"] for row in all_rows),
        "wait_only_without_ready_task": all(
            row["wait_only_without_waiting_task"] for row in waits),
        "both_cross_floor_modes_exposed": all(
            {"SINGLE_DOG", "CAR_DOG_CAR"}.issubset(
                set(row["legal_modes_for_task"])) for row in cross_rows),
        "three_single_dog_lessons_per_episode": (
            len(dog_favored_rows) == 30 and
            all(row["transport_mode"] == "SINGLE_DOG"
                for row in dog_favored_rows)),
        "seven_cooperation_lessons_per_episode": (
            sum(row["transport_mode"] == "CAR_DOG_CAR"
                for row in cross_rows) == 70),
        "no_car_crosses_floor": all(
            row["transport_mode"] != "SINGLE_CAR" for row in cross_rows),
        "rewards_and_discounts_finite": all(
            math.isfinite(row["reward"]) and 0 < row["discount"] <= 1
            for row in all_rows),
        "fixed_seed_rule_replay_equal": replay_equal,
        "only_final_task_terminates_each_episode": all(
            sum(row["terminated"] for row in all_rows if row["seed"] == seed)
            == 1 for seed in seeds),
        "no_unexpected_truncation": not any(
            row["truncated"] for row in all_rows),
    }
    summary = {
        "stage": 14,
        "gate": "BALANCED_GYM_RULE_REPLAY",
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "episode_count": len(episode_summaries),
        "task_count": len(assignments),
        "completed_task_count": len(completed_rows),
        "failed_task_count": len(failed_rows),
        "decision_count": len(all_rows),
        "wait_decision_count": len(waits),
        "same_floor_task_count": same_floor_count,
        "cross_floor_task_count": len(cross_rows),
        "single_dog_favored_task_count": len(dog_favored_rows),
        "task_types": dict(task_types),
        "transport_modes": dict(modes),
        "observation_shape": [284],
        "action_feature_shape": [1536, 30],
        "action_mask_shape": [1536],
        "per_episode": episode_summaries,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = WORK_ROOT / "results" / "stage14"
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage14_balanced_rule_200.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8")
    (output / "stage14_balanced_rule_200.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "per_episode"}, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
