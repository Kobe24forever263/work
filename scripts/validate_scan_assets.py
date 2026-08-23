#!/usr/bin/env python3
"""Validate PCD structure and four SCAN-Planner upstairs YAML routes."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRINGUP = ROOT / "src" / "warehouse_bringup"
PCD = BRINGUP / "maps" / "warehouse_full.pcd"
ROUTES = BRINGUP / "config" / "scan_planner"


def main():
    errors = []
    with PCD.open(encoding="ascii") as stream:
        header = {}
        points = []
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "DATA" not in header:
                key, *values = stripped.split()
                header[key] = values
                if key == "DATA":
                    if values != ["ascii"]:
                        errors.append("PCD must use portable ASCII data")
                continue
            points.append(tuple(float(value) for value in stripped.split()))
    declared = int(header.get("POINTS", ["-1"])[0])
    if declared != len(points):
        errors.append(f"PCD count mismatch: header {declared}, data {len(points)}")
    if header.get("FIELDS") != ["x", "y", "z"]:
        errors.append("PCD fields must be x y z")
    if len(points) < 100000:
        errors.append("PCD is unexpectedly sparse")
    mins = tuple(min(point[i] for point in points) for i in range(3))
    maxs = tuple(max(point[i] for point in points) for i in range(3))
    if mins[0] > -23.9 or maxs[0] < 23.9:
        errors.append("PCD does not cover the full floor-1 width")
    if mins[1] > -19.9 or maxs[1] < 19.9:
        errors.append("PCD does not cover the full floor-1 height")
    if maxs[2] < 6.0:
        errors.append("PCD does not include upper walls and rails")

    robots = yaml.safe_load(
        (BRINGUP / "config" / "robots.yaml").read_text(encoding="utf-8")
    )["robots"]
    stairs = yaml.safe_load(
        (BRINGUP / "config" / "stairs.yaml").read_text(encoding="utf-8")
    )["stairs"]
    for stair_id, stair in stairs.items():
        dog = stair["dog_id"]
        for direction in ("up", "down"):
            route_file = (
                ROUTES / f"{dog}_{stair_id.lower()}_{direction}.yaml")
            payload = yaml.safe_load(route_file.read_text(encoding="utf-8"))
            params = payload["scan_planner_node"]["ros__parameters"]
            values = params["fsm.waypoints"]
            if params["fsm.navi_mode"] != 2 or len(values) % 3:
                errors.append(
                    f"{dog} {direction}: invalid SCAN waypoint schema")
                continue
            waypoints = list(zip(values[::3], values[1::3], values[2::3]))
            if len(waypoints) != 13:
                errors.append(
                    f"{dog} {direction}: expected 13 stair waypoints")
            if direction == "up":
                initial = robots[dog]["initial_xyz_yaw"]
                first_distance = (
                    (waypoints[0][0] - initial[0]) ** 2
                    + (waypoints[0][1] - initial[1]) ** 2
                ) ** 0.5
                if not 1.4 <= first_distance <= 1.8:
                    errors.append(
                        f"{dog}: first target must lead from spawn onto stair")
                if any(
                        b[2] < a[2]
                        for a, b in zip(waypoints, waypoints[1:])):
                    errors.append(f"{dog}: upstairs z values are not monotonic")
                if not 4.5 <= waypoints[-1][2] <= 4.65:
                    errors.append(
                        f"{dog}: route does not finish at floor-2 body height")
            else:
                if any(
                        b[2] > a[2]
                        for a, b in zip(waypoints, waypoints[1:])):
                    errors.append(
                        f"{dog}: downstairs z values are not monotonic")
                if not 0.4 <= waypoints[-1][2] <= 0.5:
                    errors.append(
                        f"{dog}: route does not finish at floor-1 body height")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print(
        f"SCAN assets valid: {len(points)} PCD points, bounds "
        f"{mins}..{maxs}, eight 13-waypoint bidirectional routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
