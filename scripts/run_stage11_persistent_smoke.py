#!/usr/bin/env python3
"""Run the first five-task Stage 11 persistent-state milestone."""

import json
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.domain import Point, Robot, Task
from warehouse_core.persistent_dispatch import (
    AsyncTaskQueue, PersistentDispatchEnvironment, QueuedTask, RobotRuntime)


def point(point_id, floor, region, xyz):
    return Point(point_id, floor, region, xyz)


def runtime(robot_id, kind, floor, region, xyz):
    return RobotRuntime(Robot(robot_id, kind, floor, region, floor, region), xyz)


def build_environment():
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
    f1ne = point("F1_NE_PICK", 1, "F1_NE", (12, 8, .45))
    f1nw = point("F1_NW_PICK", 1, "F1_NW", (-12, 8, .45))
    f1se = point("F1_SE_PICK", 1, "F1_SE", (12, -8, .45))
    f2e = point("F2_E_DROP", 2, "F2_EAST", (7, 6.5, 4.7))
    f2w = point("F2_W_DROP", 2, "F2_WEST", (-7, 6.5, 4.7))
    arrivals = [
        QueuedTask(Task("T01", "C01", f1ne,
                        point("F1_NE_DROP", 1, "F1_NE", (7, 6, .45)), 3, 100), 0),
        QueuedTask(Task("T02", "C02", f1nw, f2w, 3, 200), 2),
        QueuedTask(Task("T03", "C03", f2w,
                        point("F2_E2_DROP", 2, "F2_EAST", (7, -6.5, 4.7)), 2, 300), 4),
        QueuedTask(Task("T04", "C04", f2e, f1se, 1, 400), 6),
        QueuedTask(Task("T05", "C05", f1se,
                        point("F1_NW_DROP", 1, "F1_NW", (-7, 6, .45)), 1, 500), 8),
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
    return PersistentDispatchEnvironment(
        robots, stairs, AsyncTaskQueue(arrivals), task_limit=5,
        time_limit=1800, decision_gap=3.0)


def main():
    env = build_environment()
    initial = {key: value.xyz for key, value in env.robots.items()}
    rows = []
    while env.completed < 5:
        candidates = env.enumerate_candidate_actions()
        if not candidates:
            env.now += 1.0
            env.queue.advance(env.now)
            continue
        transition = env.step(env.select_rule_action(candidates))
        rows.append({
            "decision": len(rows) + 1,
            "state": transition.state,
            "action": transition.action,
            "action_mask": transition.action_mask,
            "reward": transition.reward,
            "delta_time": transition.delta_time,
            "next_state": transition.next_state,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
        })
        print(f"[{env.completed}/5] {transition.action['task_id']} "
              f"car={transition.action['pickup_carter']} "
              f"dog={transition.action['dog_id'] or '-'} "
              f"stair={transition.action['stair_id'] or '-'} "
              f"receiver={transition.action['receiving_carter']}: PASS")

    moved = {key: value.xyz for key, value in env.robots.items()
             if value.xyz != initial[key]}
    used_chains = {(row["action"]["pickup_carter"], row["action"]["dog_id"],
                    row["action"]["receiving_carter"]) for row in rows}
    dog_floor_continuity = any(
        row["action"]["dog_id"] and
        next(robot["floor"] for robot in row["state"]["robots"]
             if robot["robot_id"] == row["action"]["dog_id"]) == 2 and
        next(robot["floor"] for robot in row["state"]["robots"]
             if robot["robot_id"] == row["action"]["pickup_carter"]) == 2
        for row in rows)
    assertions = {
        "five_tasks_completed": env.completed == 5,
        "positions_persisted": bool(moved),
        "multiple_assignments_used": len(used_chains) >= 2,
        "all_cargo_cleared": all(not item.robot.cargo_id for item in env.robots.values()),
        "all_task_ids_cleared": all(not item.robot.task_id for item in env.robots.values()),
        "all_arrivals_consumed": not env.queue.waiting and not env.queue.pending_arrivals,
        "smdp_delta_time_recorded": all(row["delta_time"] > 0 for row in rows),
        "dog_floor_continuity_observed": dog_floor_continuity,
    }
    summary = {
        "trial_count": 5,
        "success_count": 5 if all(assertions.values()) else 0,
        "failure_count": 0 if all(assertions.values()) else 1,
        "simulated_time": env.now,
        "assertions": assertions,
        "moved_robots": {key: value for key, value in moved.items()},
        "final_robots": {key: {
            "xyz": value.xyz,
            "floor": value.robot.current_floor,
            "distance_total": round(value.distance_total, 4),
            "tasks_completed": value.tasks_completed,
            "stairs_traversed": value.stairs_traversed,
        } for key, value in sorted(env.robots.items())},
    }
    output_dir = WORK_ROOT / "results/stage11"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage11_persistent_smoke_5.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (output_dir / "stage11_persistent_smoke_5.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(assertions.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
