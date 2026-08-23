#!/usr/bin/env python3
"""Stage 11 twenty-task persistent asynchronous dispatch gate."""

import argparse
import json
import random
import sys
from collections import Counter
from math import ceil
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.domain import Point, Robot, Task
from warehouse_core.persistent_dispatch import (
    Assignment, AsyncTaskQueue, PersistentDispatchEnvironment,
    QueuedTask, RobotRuntime, StandbySlot)


F1 = {
    "NE": Point("F1_NE", 1, "F1_NE", (12, 8, .45)),
    "NW": Point("F1_NW", 1, "F1_NW", (-12, 8, .45)),
    "SW": Point("F1_SW", 1, "F1_SW", (-12, -8, .45)),
    "SE": Point("F1_SE", 1, "F1_SE", (12, -8, .45)),
}
F2 = {
    "NE": Point("F2_NE", 2, "F2_EAST", (7, 6.5, 4.7)),
    "NW": Point("F2_NW", 2, "F2_WEST", (-7, 6.5, 4.7)),
    "SW": Point("F2_SW", 2, "F2_WEST", (-7, -6.5, 4.7)),
    "SE": Point("F2_SE", 2, "F2_EAST", (7, -6.5, 4.7)),
}
LOAD_PROFILES = {
    # Calibrated against the critical-path service model (about 50 s/task).
    # Normal leaves ample recovery margin; medium targets roughly 75% load.
    "normal": (80.0, 100.0),
    "medium": (55.0, 75.0),
    "overload": (2.0, 6.0),
}


def runtime(robot_id, kind, floor, region, xyz):
    return RobotRuntime(Robot(robot_id, kind, floor, region, floor, region), xyz)


def build_arrivals(seed: int, profile: str):
    task_rng = random.Random(seed)
    arrival_rng = random.Random(seed + 1100)
    interval_min, interval_max = LOAD_PROFILES[profile]
    keys = list(F1)
    arrivals = []
    now = 0.0
    task_no = 0
    # Three complete cycles cover all six required task classes.  Two final
    # same-region tasks bring the deterministic gate to twenty decisions.
    for _ in range(3):
        a, b = task_rng.sample(keys, 2)
        templates = [
            ("f1_same", F1[a], Point(f"F1_{a}_LOCAL", 1, F1[a].region,
                                     (F1[a].xyz[0] * .7, F1[a].xyz[1] * .7, .45))),
            ("f1_cross_region", F1[a], F1[b]),
            ("cross_up", F1[b], F2[b]),
            ("f2_same", F2[a], Point(f"F2_{a}_LOCAL", 2, F2[a].region,
                                     (F2[a].xyz[0] * .7, F2[a].xyz[1] * .7, 4.7))),
            ("f2_cross_region", F2[a], F2[b]),
            ("cross_down", F2[b], F1[task_rng.choice(keys)]),
        ]
        for task_type, source, target in templates:
            task_no += 1
            priority = 3 if task_type == "cross_up" else task_rng.randint(1, 2)
            deadline = now + 180.0 + task_rng.randint(0, 60)
            task = Task(f"T{task_no:02d}", f"C{task_no:02d}", source, target,
                        priority, deadline)
            arrivals.append((QueuedTask(task, now), task_type))
            now += arrival_rng.uniform(interval_min, interval_max)
    for floor_name, points in (("f1_same", F1), ("f2_same", F2)):
        key = task_rng.choice(keys)
        source = points[key]
        height = source.xyz[2]
        target = Point(f"{source.point_id}_FINAL_LOCAL", source.floor,
                       source.region, (source.xyz[0] * .7,
                                       source.xyz[1] * .7, height))
        task_no += 1
        task = Task(f"T{task_no:02d}", f"C{task_no:02d}", source, target,
                    task_rng.randint(1, 2), now + 210.0)
        arrivals.append((QueuedTask(task, now), floor_name))
        now += arrival_rng.uniform(interval_min, interval_max)
    return arrivals


def build_environment(seed: int, profile: str):
    robots = [
        runtime("dog_1", "dog", 1, "F1_NE", (20.8, 17.8, .45)),
        runtime("dog_2", "dog", 1, "F1_NW", (-20.8, 17.8, .45)),
        runtime("dog_3", "dog", 1, "F1_SW", (-20.8, -17.8, .45)),
        runtime("dog_4", "dog", 1, "F1_SE", (20.8, -17.8, .45)),
        runtime("car_f1_1", "car", 1, "F1_NE", (2.5, 2.5, .45)),
        runtime("car_f1_2", "car", 1, "F1_NW", (-2.5, 2.5, .45)),
        runtime("car_f1_3", "car", 1, "F1_SW", (-2.5, -2.5, .45)),
        runtime("car_f1_4", "car", 1, "F1_SE", (2.5, -2.5, .45)),
        runtime("car_f2_1", "car", 2, "F2_WEST", (-5.5, 0, 4.7)),
        runtime("car_f2_2", "car", 2, "F2_EAST", (5.5, 0, 4.7)),
    ]
    stairs = {
        "STAIR_NE": {"floor1_xyz": (20.3, 18.3, .45),
                     "floor2_xyz": (9.5, 6.5, 4.7), "state": "FREE"},
        "STAIR_NW": {"floor1_xyz": (-20.3, 18.3, .45),
                     "floor2_xyz": (-9.5, 6.5, 4.7), "state": "FREE"},
        "STAIR_SW": {"floor1_xyz": (-20.3, -18.3, .45),
                     "floor2_xyz": (-9.5, -6.5, 4.7), "state": "FREE"},
        "STAIR_SE": {"floor1_xyz": (20.3, -18.3, .45),
                     "floor2_xyz": (9.5, -6.5, 4.7), "state": "FREE"},
    }
    arrivals = build_arrivals(seed, profile)
    standby_slots = [
        StandbySlot(f"F1_CAR_{index + 1}", "car", 1, xyz)
        for index, xyz in enumerate(((-4.5, 3.0, .45), (4.5, 3.0, .45),
                                     (-4.5, -3.0, .45), (4.5, -3.0, .45)))
    ] + [
        StandbySlot(f"F2_CAR_{index + 1}", "car", 2, xyz)
        for index, xyz in enumerate(((-5.5, 0.0, 4.7), (5.5, 0.0, 4.7)))
    ] + [
        StandbySlot(f"F{floor}_DOG_{index + 1}", "dog", floor, xyz)
        for floor, height in ((1, .45), (2, 4.7))
        for index, xyz in enumerate(((-2.0, 1.5, height), (2.0, 1.5, height),
                                     (-2.0, -1.5, height), (2.0, -1.5, height)))
    ]
    env = PersistentDispatchEnvironment(
        robots, stairs, AsyncTaskQueue(item for item, _ in arrivals),
        task_limit=20, time_limit=3600, decision_gap=3.0,
        standby_slots=standby_slots)
    return env, {item.task.task_id: task_type for item, task_type in arrivals}


def invalid_variants(valid: Assignment, env):
    invalid_mode = Assignment(**{**valid.__dict__,
                                  "transport_mode": "INVALID"})
    if valid.transport_mode == env.SINGLE_DOG:
        missing_actor = Assignment(**{**valid.__dict__, "dog_id": ""})
    else:
        missing_actor = Assignment(**{**valid.__dict__,
                                      "pickup_carter": ""})
    return [invalid_mode, missing_actor]


def run_episode(seed: int, profile: str):
    env, types = build_environment(seed, profile)
    initial = {key: value.xyz for key, value in env.robots.items()}
    rows = []
    mask_rejections = 0
    queue_peak = len(env.queue.waiting)
    while env.completed < 20:
        env.queue.advance(env.now)
        candidates = env.enumerate_candidate_actions()
        if not candidates:
            env.now += 1.0
            if env.now >= env.time_limit:
                raise RuntimeError("no executable task before time limit")
            continue
        selected_index = env.select_rule_action(candidates)
        selected = candidates[selected_index]
        mixed = [selected] + invalid_variants(selected, env)
        mixed_mask = env.build_action_mask(mixed)
        mask_rejections += sum(not value for value in mixed_mask)
        queued = env._task(selected.task_id)
        wait_time = env.now - queued.arrival_time
        transition = env.step(selected_index)
        queue_peak = max(queue_peak, transition.state["waiting_count"],
                         transition.next_state["waiting_count"])
        row = {
            "decision": len(rows) + 1,
            "task_id": selected.task_id,
            "task_type": types[selected.task_id],
            "arrival_time": queued.arrival_time,
            "decision_time": transition.state["time"],
            "wait_time": round(wait_time, 4),
            "state": transition.state,
            "action": transition.action,
            "action_mask": transition.action_mask,
            "invalid_probe_mask": mixed_mask,
            "reward": transition.reward,
            "delta_time": transition.delta_time,
            "next_state": transition.next_state,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
        }
        rows.append(row)
        print(f"[{env.completed}/20] {selected.task_id} {types[selected.task_id]} "
              f"mode={selected.transport_mode} "
              f"car={selected.pickup_carter} dog={selected.dog_id or '-'} "
              f"stair={selected.stair_id or '-'} recv={selected.receiving_carter}: PASS")
    moved = [key for key, value in env.robots.items() if value.xyz != initial[key]]
    return env, rows, mask_rejections, moved, queue_peak


def signature(rows):
    return [(row["task_id"], row["action"]["pickup_carter"],
             row["action"]["dog_id"], row["action"]["stair_id"],
             row["action"]["receiving_carter"],
             row["action"]["transport_mode"]) for row in rows]


def allocation_signature(rows):
    return [(row["action"]["pickup_carter"], row["action"]["dog_id"],
             row["action"]["stair_id"], row["action"]["receiving_carter"],
             row["action"]["transport_mode"])
            for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--profile", choices=tuple(LOAD_PROFILES),
                        default="normal")
    parser.add_argument("--output-tag", default="",
                        help="suffix used to preserve an earlier baseline")
    args = parser.parse_args()
    env, rows, rejected, moved, queue_peak = run_episode(args.seed, args.profile)
    _, replay_rows, _, _, _ = run_episode(args.seed, args.profile)
    task_types = Counter(row["task_type"] for row in rows)
    transport_modes = Counter(row["action"]["transport_mode"] for row in rows)
    assignments = set(allocation_signature(rows))
    waits = sorted(row["wait_time"] for row in rows)
    durations = [row["delta_time"] for row in rows]
    mean_wait = sum(waits) / len(waits)
    mean_duration = sum(durations) / len(durations)
    p95_wait = waits[max(0, ceil(.95 * len(waits)) - 1)]
    assertions = {
        "twenty_tasks_completed": env.completed == 20,
        "six_task_types_covered": set(task_types) == {
            "f1_same", "f1_cross_region", "f2_same", "f2_cross_region",
            "cross_up", "cross_down"},
        "positions_persisted": bool(moved),
        "dynamic_assignments": len(assignments) >= 6,
        "invalid_actions_masked": rejected >= 40,
        "all_cargo_cleared": all(not item.robot.cargo_id for item in env.robots.values()),
        "all_task_ids_cleared": all(not item.robot.task_id for item in env.robots.values()),
        "queue_drained": not env.queue.waiting and not env.queue.pending_arrivals,
        "all_smdp_records_complete": all(
            row["delta_time"] > 0 and row["state"] and row["next_state"]
            and row["action_mask"] for row in rows),
        "fixed_seed_replay_equal": signature(rows) == signature(replay_rows),
        "waiting_time_recorded": any(row["wait_time"] > 0 for row in rows),
        "unique_final_robot_positions": len({value.xyz for value in env.robots.values()})
                                        == len(env.robots),
        "completed_robots_in_standby_slots": all(
            not value.tasks_completed or value.standby_slot_id
            for value in env.robots.values()),
        "all_resources_released": not env.resource_claims and all(
            stair["state"] == "FREE" for stair in env.stairs.values()),
        "resource_claim_release_balanced": (
            sum(event["event"] == "CLAIM" for event in env.resource_events) > 0
            and sum(event["event"] == "CLAIM" for event in env.resource_events)
            == sum(event["event"] == "RELEASE" for event in env.resource_events)),
    }
    summary = {
        "seed": args.seed, "profile": args.profile,
        "arrival_interval_seconds": LOAD_PROFILES[args.profile],
        "trial_count": 20,
        "success_count": 20 if all(assertions.values()) else 0,
        "failure_count": 0 if all(assertions.values()) else 1,
        "simulated_time": round(env.now, 4),
        "task_types": dict(task_types),
        "transport_modes": dict(transport_modes),
        "unique_assignment_count": len(assignments),
        "invalid_actions_masked": rejected,
        "mean_wait_time": round(mean_wait, 4),
        "p95_wait_time": round(p95_wait, 4),
        "max_wait_time": round(max(waits), 4),
        "mean_task_duration": round(mean_duration, 4),
        "queue_peak": queue_peak,
        "moved_robots": moved,
        "resource_event_count": len(env.resource_events),
        "assertions": assertions,
        "final_robots": {key: {
            "xyz": value.xyz, "floor": value.robot.current_floor,
            "distance_total": round(value.distance_total, 4),
            "tasks_completed": value.tasks_completed,
            "stairs_traversed": value.stairs_traversed,
            "standby_slot_id": value.standby_slot_id,
        } for key, value in sorted(env.robots.items())},
    }
    output = WORK_ROOT / "results/stage11"
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    stem = f"stage11_persistent_{args.profile}_20{suffix}"
    (output / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (output / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(assertions.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
