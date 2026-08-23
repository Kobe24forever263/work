#!/usr/bin/env python3
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "src" / "warehouse_bringup" / "worlds" / "two_floor_warehouse.sdf"
MAPS = ROOT / "src" / "warehouse_bringup" / "maps"


def pose(link):
    return [float(v) for v in link.findtext("pose").split()]


def main():
    root = ET.parse(WORLD).getroot()
    world = root.find("world")
    errors = []
    stair_models = [m for m in world.findall("model")
                    if m.get("name", "").startswith("stair_")
                    and not m.get("name", "").endswith("_top_landing")
                    and not m.get("name", "").endswith("_bottom_landing")]
    if len(stair_models) != 4:
        errors.append(f"expected 4 stairs, got {len(stair_models)}")
    for model in stair_models:
        steps = [link for link in model.findall("link") if link.get("name", "").startswith("step_")]
        if len(steps) != 24:
            errors.append(f"{model.get('name')}: expected 24 steps")
            continue
        heights, centers = [], []
        for step in steps:
            p = pose(step)
            size = [float(v) for v in step.find("collision/geometry/box/size").text.split()]
            heights.append(size[2])
            centers.append(p)
            if size[1] < 1.4:
                errors.append(f"{model.get('name')}: clear width below 1.4 m")
        if any(b <= a for a, b in zip(heights, heights[1:])):
            errors.append(f"{model.get('name')}: non-increasing risers")
        if not math.isclose(heights[-1], 4.0, abs_tol=0.01):
            errors.append(f"{model.get('name')}: top is not at floor 2")
        for a, b in zip(centers[:-2], centers[1:-1]):
            if math.dist(a[:2], b[:2]) > 0.31:
                errors.append(f"{model.get('name')}: discontinuous tread run")
                break
        final_size = [float(v) for v in steps[-1].find(
            "collision/geometry/box/size").text.split()]
        previous_size = [float(v) for v in steps[-2].find(
            "collision/geometry/box/size").text.split()]
        if final_size[0] < previous_size[0] + 1.19:
            errors.append(f"{model.get('name')}: final tread does not extend into floor 2")
        if final_size[1] < 2.4:
            errors.append(f"{model.get('name')}: final tread is not wide enough at corner")
        if math.dist(centers[-2][:2], centers[-1][:2]) > (previous_size[0] + final_size[0]) / 2:
            errors.append(f"{model.get('name')}: final tread does not overlap penultimate tread")
        rails = [link for link in model.findall("link") if link.get("name", "").startswith("rail_")]
        if len(rails) != 2:
            errors.append(f"{model.get('name')}: missing side rails")
        for step in steps:
            ambient = step.findtext("visual/material/ambient", "")
            if not ambient.startswith("1.0 0.72 0.05"):
                errors.append(f"{model.get('name')}: stair tread is not yellow")
                break
    task_models = [m for m in world.findall("model") if m.get("name", "").startswith("task_")]
    handover_models = [m for m in world.findall("model") if m.get("name", "").startswith("handover_")]
    if len(task_models) != 14:
        errors.append(f"expected 14 task markers, got {len(task_models)}")
    if len(handover_models) != 8:
        errors.append(f"expected 8 handover markers, got {len(handover_models)}")
    floor2_slabs = [m for m in world.findall("model")
                    if m.get("name", "").startswith("floor2_")
                    and "wall" not in m.get("name", "")]
    atrium_rails = [m for m in world.findall("model")
                    if m.get("name", "").startswith("atrium_rail_")]
    if len(floor2_slabs) != 4:
        errors.append(f"expected 4 floor-2 ring slabs around atrium, got {len(floor2_slabs)}")
    if len(atrium_rails) != 4:
        errors.append(f"expected 4 atrium guardrails, got {len(atrium_rails)}")
    platform_models = [m for m in world.findall("model") if "_platform_" in m.get("name", "")]
    column_models = [m for m in world.findall("model") if m.get("name", "").startswith("column_")]
    lane_models = [m for m in world.findall("model") if m.get("name", "").startswith("lane_")]
    if len(platform_models) != 12:
        errors.append(f"expected 12 configured loading platforms, got {len(platform_models)}")
    if len(column_models) != 5:
        errors.append(f"expected 5 structural columns, got {len(column_models)}")
    for platform in platform_models:
        size = [float(v) for v in platform.findtext(
            "link/collision/geometry/box/size").split()]
        if size[2] > 0.31:
            errors.append(f"{platform.get('name')}: loading platform is too high")
    for column in column_models:
        height = float(column.findtext("link/collision/geometry/cylinder/length"))
        if not math.isclose(height, 4.0, abs_tol=1e-6):
            errors.append(f"{column.get('name')}: column does not support floor 2")
    if len(lane_models) != 11:
        errors.append(f"expected 11 lane segments, got {len(lane_models)}")
    floor1_lights = [light for light in world.findall("light")
                     if light.get("name", "").startswith("f1_light_")]
    if len(floor1_lights) != 9:
        errors.append(f"expected 9 floor-1 lights, got {len(floor1_lights)}")
    for light in floor1_lights:
        if float(light.findtext("intensity", "0")) < 2.0:
            errors.append(f"{light.get('name')}: insufficient floor-1 light intensity")
        if light.findtext("cast_shadows") != "false":
            errors.append(f"{light.get('name')}: per-light shadows must stay disabled")
    robot_includes = [include for include in world.findall("include")
                      if include.findtext("uri") in {
                          "model://go2_description", "model://carter_description"}]
    robot_names = [include.findtext("name") for include in robot_includes]
    if len(robot_includes) != 10:
        errors.append(f"expected 10 physical robot includes, got {len(robot_includes)}")
    if len(robot_names) != len(set(robot_names)):
        errors.append("physical robot model names must be unique")
    for include in robot_includes:
        expected_static = "false"
        if include.findtext("static") != expected_static:
            errors.append(
                f"{include.findtext('name')}: incorrect controller commissioning static state")
    if sum(include.findtext("uri") == "model://go2_description"
           for include in robot_includes) != 4:
        errors.append("expected 4 Go2 model instances")
    if sum(include.findtext("uri") == "model://carter_description"
           for include in robot_includes) != 6:
        errors.append("expected 6 Carter model instances")
    world_plugins = {
        plugin.get("name") for plugin in world.findall("plugin")
    }
    for required_plugin in {
        "gz::sim::systems::Physics",
        "gz::sim::systems::UserCommands",
        "gz::sim::systems::SceneBroadcaster",
        "gz::sim::systems::Sensors",
        "gz::sim::systems::Imu",
    }:
        if required_plugin not in world_plugins:
            errors.append(f"missing world sensor system {required_plugin}")
    for suffix, dog_name in {
        "ne": "dog_1", "nw": "dog_2", "sw": "dog_3", "se": "dog_4"
    }.items():
        dock = world.find(f"model[@name='handover_stair_{suffix}_f1_send']")
        dog = world.find(f"include[name='{dog_name}']")
        if dock is None or dog is None:
            errors.append(f"{suffix.upper()}: missing stair-side dog handover pairing")
            continue
        dock_xy = [float(v) for v in dock.findtext("pose").split()[:2]]
        dog_xy = [float(v) for v in dog.findtext("pose").split()[:2]]
        separation = math.dist(dock_xy, dog_xy)
        if not 0.50 <= separation <= 0.75:
            errors.append(
                f"{suffix.upper()}: Carter dock must be 0.50-0.75 m from Go2, got {separation:.2f}")
    expected_maps = {"floor1.pgm": (480, 400), "floor2.pgm": (300, 240)}
    for filename, expected in expected_maps.items():
        tokens = (MAPS / filename).read_text(encoding="ascii").split()
        dims = tuple(map(int, tokens[1:3]))
        if dims != expected:
            errors.append(f"{filename}: expected {expected}, got {dims}")
        border_only = 2 * expected[0] + 2 * expected[1] - 4
        if tokens[4:].count("0") <= border_only:
            errors.append(f"{filename}: configured obstacles are missing from occupancy map")
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("World valid: 2 floors with 8x6 m atrium, 4 atrium rails, "
          "4x24 yellow steps with integrated floor-height bridge treads, stair rails, "
          "14 tasks, 8 handover zones, "
          "12 mapped loading platforms, 5 structural columns, 9 floor-1 lights, "
          "10 physical robots, 11 lane segments, 2 maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
