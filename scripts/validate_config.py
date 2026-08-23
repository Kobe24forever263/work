#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "warehouse_bringup" / "config"


def main():
    scene = (CONFIG / "scene.yaml").read_text(encoding="utf-8")
    task_section = scene.split("task_points:", 1)[1].split("regions:", 1)[0]
    robot_text = (CONFIG / "robots.yaml").read_text(encoding="utf-8")
    stair_text = (CONFIG / "stairs.yaml").read_text(encoding="utf-8")
    learning = (CONFIG / "learning.yaml").read_text(encoding="utf-8")
    model_text = (CONFIG / "robot_models.yaml").read_text(encoding="utf-8")
    robot_ids = re.findall(r"^  ((?:dog|car)_\w+):", robot_text, re.M)
    namespaces = re.findall(r"namespace:\s*([^,}]+)", robot_text)
    dogs = [name for name in robot_ids if name.startswith("dog_")]
    floor1_cars = [name for name in robot_ids if name.startswith("car_f1_")]
    floor2_cars = [name for name in robot_ids if name.startswith("car_f2_")]
    stair_ids = re.findall(r"^  (STAIR_\w+):", stair_text, re.M)
    stair_dogs = re.findall(r"dog_id:\s*(dog_\d+)", stair_text)
    f1_points = re.findall(r"^  F1_(?!CENTER:)[A-Z_]+:.*floor:\s*1", task_section, re.M)
    f1_center = re.findall(r"^  F1_CENTER:.*floor:\s*1", task_section, re.M)
    f2_points = re.findall(r"^  F2_[A-Z]+:.*floor:\s*2", task_section, re.M)
    errors = []
    if len(robot_ids) != 10:
        errors.append(f"expected 10 robots, got {len(robot_ids)}")
    if len(dogs) != 4:
        errors.append("expected 4 dogs")
    if len(floor1_cars) != 4:
        errors.append("expected 4 floor-1 cars")
    if len(floor2_cars) != 2:
        errors.append("expected 2 floor-2 cars")
    if len(f1_points) + len(f1_center) != 9:
        errors.append("floor 1 must have 9 task points")
    if len(f2_points) != 5:
        errors.append("floor 2 must have 5 task points")
    if len(stair_ids) != 4 or len(set(stair_dogs)) != 4:
        errors.append("four stairs must map one-to-one to four dogs")
    agent_line = re.search(r"agents:\s*\[([^\]]+)\]", learning)
    agents = set(re.findall(r"(?:dog|car)_\w+", agent_line.group(1))) if agent_line else set()
    if agents != set(robot_ids):
        errors.append("RL agents must exactly match physical robots")
    if len(namespaces) != len(set(namespaces)):
        errors.append("robot namespaces must be unique")
    lane_width = float(re.search(r"vehicle_lane_width:\s*([\d.]+)", scene).group(1))
    car_width = float(re.search(r"footprint_width_m:\s*([\d.]+)", model_text).group(1))
    margin = float(re.search(r"safety_margin_m:\s*([\d.]+)", model_text).group(1))
    if lane_width < 2.5 * (car_width + 2 * margin):
        errors.append("main lane must be >= 2.5x safety-expanded vehicle width")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors))
        return 1
    print("Configuration valid: 10 robots, 14 task points, 4 stairs, unique IDs/namespaces, lane clearance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
