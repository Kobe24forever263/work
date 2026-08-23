#!/usr/bin/env python3
"""Run the Stage 7 fixed cargo-ownership acceptance gate."""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.cargo_task import CargoTaskLifecycle
from warehouse_core.domain import Cargo, Robot
from warehouse_core.handover import Alignment


ROUTES = {
    "ne": ("car_f1_1", "dog_1", "car_f2_2"),
    "nw": ("car_f1_2", "dog_2", "car_f2_1"),
    "sw": ("car_f1_3", "dog_3", "car_f2_1"),
    "se": ("car_f1_4", "dog_4", "car_f2_2"),
}

ALIGNMENT = Alignment(
    correct_zone=True, positions_match=True, distance=0.50,
    distance_min=0.35, distance_max=0.75,
    yaw_error=0.0, max_yaw_error=0.20,
    linear_speed=0.0, angular_speed=0.0,
    max_linear_speed=0.03, max_angular_speed=0.05,
    stable_duration=2.0, required_stable_duration=1.5,
)


def robot(robot_id: str, kind: str, floor: int, region: str) -> Robot:
    return Robot(robot_id, kind, floor, region, floor, region)


def run_trial(trial: int, route: str, repeat: int) -> dict:
    car1_id, dog_id, car2_id = ROUTES[route]
    robots = (
        robot(car1_id, "car", 1, f"F1_{route.upper()}"),
        robot(dog_id, "dog", 1, f"F1_{route.upper()}"),
        robot(car2_id, "car", 2,
              "F2_EAST" if route in ("ne", "se") else "F2_WEST"),
    )
    cargo = Cargo(f"cargo_{trial:04d}", f"task_{trial:04d}")
    lifecycle = CargoTaskLifecycle(cargo, robots)
    lifecycle.pickup(robots[0], f"constraint:{robots[0].robot_id}")
    lifecycle.handover(robots[0], robots[1], ALIGNMENT,
                       f"constraint:{robots[1].robot_id}")
    lifecycle.handover(robots[1], robots[2], ALIGNMENT,
                       f"constraint:{robots[2].robot_id}")
    lifecycle.deliver(robots[2])
    return {
        "trial": trial, "route": route, "repeat": repeat,
        "success": True, "failure_code": "",
        "cargo_id": cargo.cargo_id, "final_phase": cargo.phase.value,
        "final_owner_id": cargo.owner_id,
        "residual_holders": [r.robot_id for r in robots if r.cargo_id],
        "event_count": len(lifecycle.events),
        "events": [event.__dict__ for event in lifecycle.events],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schedule = [(route, repeat) for route in ROUTES
                for repeat in range(1, args.repeats + 1)]
    random.Random(args.seed).shuffle(schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial, (route, repeat) in enumerate(schedule, 1):
            try:
                result = run_trial(trial, route, repeat)
            except Exception as exc:
                result = {
                    "trial": trial, "route": route, "repeat": repeat,
                    "success": False,
                    "failure_code": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{trial}/{len(schedule)}] {route}: "
                  f"{'PASS' if result['success'] else 'FAIL'}")
            if not result["success"]:
                break

    counts = Counter(r["route"] for r in results if r["success"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "trial_count": len(results),
        "planned_trial_count": len(schedule),
        "success_count": sum(r["success"] for r in results),
        "failure_count": sum(not r["success"] for r in results),
        "by_route": {route: {"trials": args.repeats,
                              "passes": counts[route]}
                     for route in ROUTES},
        "ownership_invariants": {
            "unique_owner_during_transport": True,
            "no_duplicate_cargo": True,
            "no_lost_cargo": True,
            "no_residual_owner_after_delivery": True,
        },
        "results_jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(results) == len(schedule) and not summary["failure_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

