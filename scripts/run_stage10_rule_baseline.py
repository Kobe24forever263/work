#!/usr/bin/env python3
"""Stage 10 reproducible 200-task rule-scheduler baseline."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.cargo_task import CargoTaskLifecycle
from warehouse_core.domain import Cargo, Point, Robot, Task
from warehouse_core.handover import Alignment
from warehouse_core.scheduler import RuleScheduler


TASK_TYPES = ("same_region", "same_floor_handover", "cross_up", "cross_down")
F1 = {
    "NE": Point("F1_PICKUP_NE", 1, "F1_NE", (12.0, 8.0, .45)),
    "NW": Point("F1_PICKUP_NW", 1, "F1_NW", (-12.0, 8.0, .45)),
    "SW": Point("F1_PICKUP_SW", 1, "F1_SW", (-12.0, -8.0, .45)),
    "SE": Point("F1_PICKUP_SE", 1, "F1_SE", (12.0, -8.0, .45)),
}
F2 = {
    "NE": Point("F2_DROPOFF_NE", 2, "F2_EAST", (7.0, 6.5, 4.70)),
    "NW": Point("F2_DROPOFF_NW", 2, "F2_WEST", (-7.0, 6.5, 4.70)),
    "SW": Point("F2_DROPOFF_SW", 2, "F2_WEST", (-7.0, -6.5, 4.70)),
    "SE": Point("F2_DROPOFF_SE", 2, "F2_EAST", (7.0, -6.5, 4.70)),
}
ALIGNMENT = Alignment(True, True, .5, .35, .75, 0, .2, 0, 0, .03, .05, 2, 1.5)


def make_robots(source_floor: int) -> list[Robot]:
    specs = [
        ("dog_1", "dog", source_floor, "F1_NE"),
        ("dog_2", "dog", source_floor, "F1_NW"),
        ("dog_3", "dog", source_floor, "F1_SW"),
        ("dog_4", "dog", source_floor, "F1_SE"),
        ("car_f1_1", "car", 1, "F1_NE"),
        ("car_f1_2", "car", 1, "F1_NW"),
        ("car_f1_3", "car", 1, "F1_SW"),
        ("car_f1_4", "car", 1, "F1_SE"),
        ("car_f2_1", "car", 2, "F2_WEST"),
        ("car_f2_2", "car", 2, "F2_EAST"),
    ]
    return [Robot(rid, kind, home, region, home, region)
            for rid, kind, home, region in specs]


def stairs(rng: random.Random) -> dict:
    return {
        "STAIR_NE": {"dog_id": "dog_1", "floor1_xy": [20.3, 18.3],
                     "floor2_xy": [9.5, 6.5], "state": "FREE",
                     "queue_length": rng.randrange(0, 3), "risk": .05},
        "STAIR_NW": {"dog_id": "dog_2", "floor1_xy": [-20.3, 18.3],
                     "floor2_xy": [-9.5, 6.5], "state": "FREE",
                     "queue_length": rng.randrange(0, 3), "risk": .05},
        "STAIR_SW": {"dog_id": "dog_3", "floor1_xy": [-20.3, -18.3],
                     "floor2_xy": [-9.5, -6.5], "state": "FREE",
                     "queue_length": rng.randrange(0, 3), "risk": .05},
        "STAIR_SE": {"dog_id": "dog_4", "floor1_xy": [20.3, -18.3],
                     "floor2_xy": [9.5, -6.5], "state": "FREE",
                     "queue_length": rng.randrange(0, 3), "risk": .05},
    }


def points_for(kind: str, rng: random.Random) -> tuple[Point, Point]:
    keys = list(F1)
    a = rng.choice(keys)
    if kind == "same_region":
        source = F1[a]
        target = Point(f"F1_DELIVERY_{a}", 1, source.region,
                       (source.xyz[0] * .65, source.xyz[1] * .65, .45))
        return source, target
    if kind == "same_floor_handover":
        b = rng.choice([key for key in keys if F1[key].region != F1[a].region])
        return F1[a], Point(f"F1_DELIVERY_{b}", 1, F1[b].region, F1[b].xyz)
    if kind == "cross_up":
        return F1[a], F2[rng.choice(keys)]
    return F2[a], F1[rng.choice(keys)]


def validate_chain(kind: str, chain, robots_by_id: dict[str, Robot]) -> None:
    expected_legs = 1 if kind == "same_region" else (2 if kind == "same_floor_handover" else 3)
    if len(chain.legs) != expected_legs:
        raise AssertionError(f"wrong leg count: {len(chain.legs)}")
    if kind.startswith("cross"):
        kinds = [robots_by_id[leg.robot_id].robot_type for leg in chain.legs]
        if kinds != ["car", "dog", "car"]:
            raise AssertionError(f"illegal cross-floor robot chain: {kinds}")
        middle = chain.legs[1]
        if not any(resource.startswith("stair:") for resource in middle.resource_ids):
            raise AssertionError("cross-floor dog leg has no stair reservation")
        if len(middle.resource_ids) != 3:
            raise AssertionError("cross-floor resource order is incomplete")
    elif any(robots_by_id[leg.robot_id].robot_type != "car" for leg in chain.legs):
        raise AssertionError("same-floor chain contains non-car robot")


def execute_task(task: Task, chain, robots_by_id: dict[str, Robot]) -> int:
    carriers = tuple(robots_by_id[leg.robot_id] for leg in chain.legs)
    cargo = Cargo(task.cargo_id, task.task_id)
    lifecycle = CargoTaskLifecycle(cargo, carriers)
    lifecycle.pickup(carriers[0], f"constraint:{carriers[0].robot_id}")
    for sender, receiver in zip(carriers, carriers[1:]):
        lifecycle.handover(sender, receiver, ALIGNMENT,
                           f"constraint:{receiver.robot_id}")
    lifecycle.deliver(carriers[-1])
    return len(lifecycle.events)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((WORK_ROOT / "src/warehouse_bringup/config/scheduler.yaml").read_text())
    weights = config["rule_cost"]
    rng = random.Random(args.seed)
    schedule = [(kind, repeat) for kind in TASK_TYPES
                for repeat in range(1, args.per_type + 1)]
    rng.shuffle(schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial, (kind, repeat) in enumerate(schedule, 1):
            try:
                source, target = points_for(kind, rng)
                task = Task(f"task_{trial:04d}", f"cargo_{trial:04d}",
                            source, target, 1 + trial % 3, 300.0)
                robots = make_robots(source.floor)
                by_id = {robot.robot_id: robot for robot in robots}
                chain = RuleScheduler(robots, stairs(rng), weights).plan(task)
                validate_chain(kind, chain, by_id)
                events = execute_task(task, chain, by_id)
                row = {
                    "trial": trial, "task_type": kind, "repeat": repeat,
                    "success": True, "task_id": task.task_id,
                    "source": source.point_id, "target": target.point_id,
                    "stair_id": chain.stair_id,
                    "robot_chain": [leg.robot_id for leg in chain.legs],
                    "skills": [leg.skill for leg in chain.legs],
                    "resource_order": [list(leg.resource_ids) for leg in chain.legs],
                    "estimated_cost": round(chain.estimated_cost, 4),
                    "handover_count": len(chain.legs) - 1,
                    "ownership_event_count": events,
                    "final_state": "COMPLETED",
                }
            except Exception as exc:
                row = {"trial": trial, "task_type": kind, "repeat": repeat,
                       "success": False,
                       "failure_code": f"{type(exc).__name__}: {exc}"}
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{trial}/{len(schedule)}] {kind}: "
                  f"{'PASS' if row['success'] else 'FAIL'}")
            if not row["success"]:
                break

    passes = Counter(r["task_type"] for r in results if r["success"])
    costs = defaultdict(list)
    for row in results:
        if row["success"]:
            costs[row["task_type"]].append(row["estimated_cost"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "trial_count": len(results),
        "planned_trial_count": len(schedule),
        "success_count": sum(r["success"] for r in results),
        "failure_count": sum(not r["success"] for r in results),
        "by_task_type": {
            kind: {"trials": args.per_type, "passes": passes[kind],
                   "mean_estimated_cost": round(sum(costs[kind]) / len(costs[kind]), 4)}
            for kind in TASK_TYPES},
        "all_tasks_completed": all(r.get("final_state") == "COMPLETED" for r in results),
        "results_jsonl": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    accepted = (len(results) == len(schedule) and not summary["failure_count"] and
                summary["all_tasks_completed"])
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
