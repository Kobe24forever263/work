#!/usr/bin/env python3
"""Validate Stage 6 Carter route coverage and goal consistency."""

from pathlib import Path
import sys

import yaml


WORK = Path("/Users/lab4099/Desktop/Mujoco/work")
CONFIG = WORK / "src/warehouse_bringup/config/scan_planner"
MANIFEST = CONFIG / "stage6_carter_manifest.yaml"
EXPECTED = {
    "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4",
    "car_f2_1", "car_f2_2",
}


def main():
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    robots = data["robots"]
    errors = []
    if set(robots) != EXPECTED:
        errors.append(
            f"robot coverage mismatch: expected={sorted(EXPECTED)} "
            f"actual={sorted(robots)}")
    for robot, spec in robots.items():
        floor = int(spec["floor"])
        for route, route_spec in spec["routes"].items():
            path = CONFIG / route_spec["route_file"]
            if not path.exists():
                errors.append(f"{robot}:{route}: missing {path.name}")
                continue
            values = yaml.safe_load(path.read_text(encoding="utf-8"))[
                "scan_planner_node"]["ros__parameters"]["fsm.waypoints"]
            if len(values) < 6 or len(values) % 3:
                errors.append(f"{robot}:{route}: invalid waypoint list")
                continue
            actual = tuple(float(value) for value in values[-3:])
            expected = tuple(float(value) for value in route_spec["goal_xyz"])
            if any(abs(a - b) > 1e-6 for a, b in zip(actual, expected)):
                errors.append(
                    f"{robot}:{route}: goal {actual} != manifest {expected}")
            waypoints = [
                tuple(float(value) for value in values[index:index + 3])
                for index in range(0, len(values), 3)
            ]
            required_key = "pickup_xyz" if floor == 1 else "receive_xyz"
            required = tuple(float(value) for value in route_spec[required_key])
            if required not in waypoints:
                errors.append(
                    f"{robot}:{route}: route does not visit {required_key} "
                    f"{required}")
            if floor == 1 and not actual[2] < 1.0:
                errors.append(f"{robot}:{route}: floor-1 goal z is invalid")
            if floor == 2 and not actual[2] > 4.0:
                errors.append(f"{robot}:{route}: floor-2 goal z is invalid")
    if errors:
        print("Stage 6 route validation: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    route_count = sum(len(spec["routes"]) for spec in robots.values())
    print(
        f"Stage 6 route validation: PASS "
        f"({len(robots)} robots, {route_count} cargo route variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
