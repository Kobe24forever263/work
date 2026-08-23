#!/usr/bin/env python3
"""Generate SCAN-Planner PCD and four deterministic Go2 stair routes."""

from itertools import product
from math import cos, pi, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRINGUP = ROOT / "src" / "warehouse_bringup"
WORLD = BRINGUP / "worlds" / "two_floor_warehouse.sdf"
MAP_DIR = BRINGUP / "maps"
ROUTE_DIR = BRINGUP / "config" / "scan_planner"
RESOLUTION = 0.10
STAIR_STEP_COUNT = 24
STAIR_RISE = 4.0
BODY_CLEARANCE = 0.56


def pose_values(element):
    if element is None or not element.findtext("pose"):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(float(value) for value in element.findtext("pose").split())


def rotate(point, rpy):
    x, y, z = point
    roll, pitch, yaw = rpy
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return (
        (cy * cp) * x + (cy * sp * sr - sy * cr) * y
        + (cy * sp * cr + sy * sr) * z,
        (sy * cp) * x + (sy * sp * sr + cy * cr) * y
        + (sy * sp * cr - cy * sr) * z,
        (-sp) * x + (cp * sr) * y + (cp * cr) * z,
    )


def compose(parent, child):
    rotated = rotate(child[:3], parent[3:])
    return (
        parent[0] + rotated[0],
        parent[1] + rotated[1],
        parent[2] + rotated[2],
        parent[3] + child[3],
        parent[4] + child[4],
        parent[5] + child[5],
    )


def axis_samples(size):
    count = max(1, round(size / RESOLUTION))
    return [-size / 2 + index * size / count for index in range(count + 1)]


def box_surface(size):
    xs, ys, zs = (axis_samples(value) for value in size)
    for x, y in product(xs, ys):
        yield x, y, -size[2] / 2
        yield x, y, size[2] / 2
    for x, z in product(xs, zs):
        yield x, -size[1] / 2, z
        yield x, size[1] / 2, z
    for y, z in product(ys, zs):
        yield -size[0] / 2, y, z
        yield size[0] / 2, y, z


def cylinder_surface(radius, length):
    angles = max(16, round(2 * pi * radius / RESOLUTION))
    zs = axis_samples(length)
    radial = axis_samples(radius * 2)
    for index in range(angles):
        angle = 2 * pi * index / angles
        for z in zs:
            yield radius * cos(angle), radius * sin(angle), z
    for x, y in product(radial, radial):
        if x * x + y * y <= radius * radius:
            yield x, y, -length / 2
            yield x, y, length / 2


def environment_points():
    world = ET.parse(WORLD).getroot().find("world")
    voxels = set()
    for model in world.findall("model"):
        model_pose = pose_values(model)
        for link in model.findall("link"):
            link_pose = compose(model_pose, pose_values(link))
            for collision in link.findall("collision"):
                collision_pose = compose(link_pose, pose_values(collision))
                geometry = collision.find("geometry")
                box = geometry.find("box")
                cylinder = geometry.find("cylinder")
                if box is not None:
                    size = tuple(float(v) for v in box.findtext("size").split())
                    local_points = box_surface(size)
                elif cylinder is not None:
                    local_points = cylinder_surface(
                        float(cylinder.findtext("radius")),
                        float(cylinder.findtext("length")),
                    )
                else:
                    continue
                for local in local_points:
                    rotated = rotate(local, collision_pose[3:])
                    world_point = (
                        collision_pose[0] + rotated[0],
                        collision_pose[1] + rotated[1],
                        collision_pose[2] + rotated[2],
                    )
                    voxels.add(tuple(round(v / RESOLUTION) for v in world_point))
    return sorted(
        (x * RESOLUTION, y * RESOLUTION, z * RESOLUTION)
        for x, y, z in voxels
    )


def write_pcd(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\nHEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\nDATA ascii\n"
        )
        for x, y, z in points:
            stream.write(f"{x:.3f} {y:.3f} {z:.3f}\n")


def stair_waypoints(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    run = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / run, dy / run
    result = []
    tread = run / STAIR_STEP_COUNT
    riser = STAIR_RISE / STAIR_STEP_COUNT
    # Use the centre of every second tread, not the riser boundary.  The body
    # reference also needs enough clearance for SCAN's inflated double-cylinder.
    for step in range(2, STAIR_STEP_COUNT + 1, 2):
        distance = (step - 0.5) * tread
        result.append((
            start[0] + ux * distance,
            start[1] + uy * distance,
            BODY_CLEARANCE + riser * step,
        ))
    result.append((
        end[0] + ux * 1.0,
        end[1] + uy * 1.0,
        STAIR_RISE + BODY_CLEARANCE,
    ))
    return result


def write_routes():
    stairs = yaml.safe_load(
        (BRINGUP / "config" / "stairs.yaml").read_text(encoding="utf-8")
    )["stairs"]
    robots = yaml.safe_load(
        (BRINGUP / "config" / "robots.yaml").read_text(encoding="utf-8")
    )["robots"]
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)
    generated = {}
    for stair_id, stair in stairs.items():
        dog_id = stair["dog_id"]
        points = stair_waypoints(stair["floor1_xy"], stair["floor2_xy"])
        flat = [round(value, 4) for point in points for value in point]
        payload = {
            "scan_planner_node": {
                "ros__parameters": {
                    "fsm.navi_mode": 2,
                    "fsm.waypoints": flat,
                }
            }
        }
        up_path = ROUTE_DIR / f"{dog_id}_{stair_id.lower()}_up.yaml"
        up_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        home = robots[dog_id]["initial_xyz_yaw"]
        down_points = list(reversed(points[:-1]))
        down_points.append((home[0], home[1], home[2]))
        down_payload = {
            "scan_planner_node": {
                "ros__parameters": {
                    "fsm.navi_mode": 2,
                    "fsm.waypoints": [
                        round(value, 4)
                        for point in down_points for value in point
                    ],
                }
            }
        }
        down_path = ROUTE_DIR / f"{dog_id}_{stair_id.lower()}_down.yaml"
        down_path.write_text(
            yaml.safe_dump(down_payload, sort_keys=False), encoding="utf-8")
        generated[dog_id] = {
            "stair_id": stair_id,
            "namespace": robots[dog_id]["namespace"],
            "initial_xyz_yaw": robots[dog_id]["initial_xyz_yaw"],
            "up_route_file": str(up_path.relative_to(ROOT)),
            "down_route_file": str(down_path.relative_to(ROOT)),
            "up_initial_xyz": home[:3],
            "down_initial_xyz": list(points[-1]),
            "waypoint_count": len(points),
        }
    (ROUTE_DIR / "four_go2_manifest.yaml").write_text(
        yaml.safe_dump({"robots": generated}, sort_keys=False),
        encoding="utf-8",
    )
    return generated


def main():
    points = environment_points()
    full = MAP_DIR / "warehouse_full.pcd"
    write_pcd(full, points)
    write_pcd(MAP_DIR / "warehouse_floor1.pcd",
              [point for point in points if point[2] <= 2.2])
    write_pcd(MAP_DIR / "warehouse_floor2.pcd",
              [point for point in points if point[2] >= 3.7])
    routes = write_routes()
    print(f"Generated {full} with {len(points)} points")
    print("Generated routes: " + ", ".join(
        f"{dog}={record['waypoint_count']}" for dog, record in routes.items()))


if __name__ == "__main__":
    main()
