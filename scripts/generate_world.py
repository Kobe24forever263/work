#!/usr/bin/env python3
"""Generate deterministic SDF and occupancy maps from the YAML configuration.

The parser intentionally reads only the stable inline records used by scene.yaml
and stairs.yaml, keeping PyYAML optional for generation on minimal systems.
"""
from math import atan2, cos, hypot, sin
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "src" / "warehouse_bringup" / "config"
WORLD = ROOT / "src" / "warehouse_bringup" / "worlds" / "two_floor_warehouse.sdf"
MAPS = ROOT / "src" / "warehouse_bringup" / "maps"


def numbers(value):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value)]


def inline_records(text, section):
    body = text.split(section + ":", 1)[1]
    body = re.split(r"^\S", body, maxsplit=1, flags=re.M)[0]
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^  ([A-Za-z0-9_]+):\s*\{([^}]*)\}", body, re.M)}


def list_records(text, section):
    body = text.split(section + ":", 1)[1]
    body = re.split(r"^\S", body, maxsplit=1, flags=re.M)[0]
    result = {}
    for match in re.finditer(r"^  ([A-Za-z0-9_]+):\s*(\[\[.*\]\])", body, re.M):
        values = numbers(match.group(2))
        result[match.group(1)] = list(zip(values[::2], values[1::2]))
    return result


def field(record, name):
    match = re.search(rf"{name}:\s*(\[[^\]]+\]|[^,]+)", record)
    if not match:
        raise ValueError(f"missing {name} in {record}")
    return match.group(1).strip()


def add_box(parent, name, xyz, size, color, collision=True):
    model = ET.SubElement(parent, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = " ".join(f"{v:.4f}" for v in (*xyz, 0, 0, 0))
    link = ET.SubElement(model, "link", name="link")
    if collision:
        col = ET.SubElement(link, "collision", name="collision")
        geo = ET.SubElement(col, "geometry")
        ET.SubElement(ET.SubElement(geo, "box"), "size").text = " ".join(map(str, size))
    visual = ET.SubElement(link, "visual", name="visual")
    geo = ET.SubElement(visual, "geometry")
    ET.SubElement(ET.SubElement(geo, "box"), "size").text = " ".join(map(str, size))
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color
    ET.SubElement(material, "diffuse").text = color


def add_cylinder(parent, name, xyz, radius, length, color):
    model = ET.SubElement(parent, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = " ".join(f"{v:.4f}" for v in (*xyz, 0, 0, 0))
    link = ET.SubElement(model, "link", name="link")
    visual = ET.SubElement(link, "visual", name="visual")
    geo = ET.SubElement(visual, "geometry")
    cyl = ET.SubElement(geo, "cylinder")
    ET.SubElement(cyl, "radius").text = str(radius)
    ET.SubElement(cyl, "length").text = str(length)
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color


def add_column(parent, name, xyz, radius, height):
    model = ET.SubElement(parent, "model", name=name.lower())
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = " ".join(f"{v:.4f}" for v in (*xyz, 0, 0, 0))
    link = ET.SubElement(model, "link", name="link")
    for tag in ("collision", "visual"):
        item = ET.SubElement(link, tag, name=tag)
        geo = ET.SubElement(item, "geometry")
        cylinder = ET.SubElement(geo, "cylinder")
        ET.SubElement(cylinder, "radius").text = f"{radius:.4f}"
        ET.SubElement(cylinder, "length").text = f"{height:.4f}"
        if tag == "visual":
            material = ET.SubElement(item, "material")
            ET.SubElement(material, "ambient").text = "0.72 0.74 0.76 1"
            ET.SubElement(material, "diffuse").text = "0.72 0.74 0.76 1"


def add_point_light(parent, name, xyz, intensity, light_range):
    light = ET.SubElement(parent, "light", name=name.lower(), type="point")
    ET.SubElement(light, "pose").text = " ".join(f"{v:.4f}" for v in (*xyz, 0, 0, 0))
    ET.SubElement(light, "diffuse").text = "1.0 0.94 0.82 1"
    ET.SubElement(light, "specular").text = "0.25 0.23 0.20 1"
    ET.SubElement(light, "intensity").text = f"{intensity:.3f}"
    ET.SubElement(light, "cast_shadows").text = "false"
    attenuation = ET.SubElement(light, "attenuation")
    ET.SubElement(attenuation, "range").text = f"{light_range:.3f}"
    ET.SubElement(attenuation, "constant").text = "0.35"
    ET.SubElement(attenuation, "linear").text = "0.025"
    ET.SubElement(attenuation, "quadratic").text = "0.006"


def add_stair(world, stair_id, start, end, steps=24, width=1.4, riser=4/24):
    dx, dy = end[0] - start[0], end[1] - start[1]
    run = hypot(dx, dy)
    ux, uy = dx / run, dy / run
    yaw = atan2(dy, dx)
    tread = run / steps
    model = ET.SubElement(world, "model", name=stair_id.lower())
    ET.SubElement(model, "static").text = "true"
    for i in range(steps):
        height = (i + 1) * riser
        is_final = i == steps - 1
        bridge_extension = 1.20 if is_final else 0.0
        along = (i + 0.5) * tread + bridge_extension / 2
        x, y = start[0] + ux * along, start[1] + uy * along
        link = ET.SubElement(model, "link", name=f"step_{i+1:02d}")
        ET.SubElement(link, "pose").text = f"{x:.4f} {y:.4f} {height/2:.4f} 0 0 {yaw:.6f}"
        for tag in ("collision", "visual"):
            item = ET.SubElement(link, tag, name=tag)
            geo = ET.SubElement(item, "geometry")
            step_length = tread + 0.008 + bridge_extension
            step_width = 2.40 if is_final else width
            ET.SubElement(ET.SubElement(geo, "box"), "size").text = f"{step_length:.4f} {step_width:.4f} {height:.4f}"
            if tag == "visual":
                mat = ET.SubElement(item, "material")
                ET.SubElement(mat, "ambient").text = "1.0 0.72 0.05 1"
                ET.SubElement(mat, "diffuse").text = "1.0 0.72 0.05 1"
    # Side rails are visual and collision-bearing, located outside the clear width.
    slope = atan2(4.0, run)
    for side in (-1, 1):
        nx, ny = -uy * side, ux * side
        x = (start[0] + end[0]) / 2 + nx * (width / 2 + 0.06)
        y = (start[1] + end[1]) / 2 + ny * (width / 2 + 0.06)
        link = ET.SubElement(model, "link", name=f"rail_{side:+d}")
        ET.SubElement(link, "pose").text = f"{x:.4f} {y:.4f} 2.55 0 {-slope:.6f} {yaw:.6f}"
        for tag in ("collision", "visual"):
            item = ET.SubElement(link, tag, name=tag)
            geo = ET.SubElement(item, "geometry")
            ET.SubElement(ET.SubElement(geo, "box"), "size").text = f"{run/cos(slope):.4f} 0.08 0.10"
            if tag == "visual":
                mat = ET.SubElement(item, "material")
                ET.SubElement(mat, "ambient").text = "0.12 0.15 0.18 1"


def write_map(name, width_m, height_m, elevation, obstacles):
    resolution = 0.10
    width, height = round(width_m / resolution), round(height_m / resolution)
    MAPS.mkdir(parents=True, exist_ok=True)
    pgm = MAPS / f"{name}.pgm"
    with pgm.open("w", encoding="ascii") as stream:
        stream.write(f"P2\n{width} {height}\n255\n")
        for y in range(height):
            wy = height_m / 2 - (y + .5) * resolution
            row = []
            for x in range(width):
                wx = -width_m / 2 + (x + .5) * resolution
                occupied = y in (0, height-1) or x in (0, width-1)
                occupied |= any(abs(wx-ox) <= sx/2 and abs(wy-oy) <= sy/2
                                for ox, oy, sx, sy in obstacles)
                row.append("0" if occupied else "254")
            stream.write(" ".join(row) + "\n")
    (MAPS / f"{name}.yaml").write_text(
        f"image: {name}.pgm\nmode: trinary\nresolution: {resolution}\n"
        f"origin: [{-width_m/2}, {-height_m/2}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.196\n"
        f"# world_elevation: {elevation}\n", encoding="utf-8")


def main():
    scene = (CFG / "scene.yaml").read_text(encoding="utf-8")
    stair_text = (CFG / "stairs.yaml").read_text(encoding="utf-8")
    robot_text = (CFG / "robots.yaml").read_text(encoding="utf-8")
    task_records = inline_records(scene, "task_points")
    building_records = inline_records(scene, "building")
    handover_records = inline_records(scene, "handover_points")
    obstacle_records = inline_records(scene, "obstacles")
    column_records = inline_records(scene, "structural_columns")
    light_records = inline_records(scene, "lights")
    lanes = list_records(scene, "lane_centerlines")
    stair_records = inline_records(stair_text, "stairs")
    robot_records = inline_records(robot_text, "robots")
    sdf = ET.Element("sdf", version="1.7")
    world = ET.SubElement(sdf, "world", name="two_floor_warehouse")
    physics = ET.SubElement(world, "physics", name="default", type="ode")
    ET.SubElement(physics, "max_step_size").text = "0.01"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"
    ET.SubElement(
        world, "plugin", filename="gz-sim-physics-system",
        name="gz::sim::systems::Physics")
    ET.SubElement(
        world, "plugin", filename="gz-sim-user-commands-system",
        name="gz::sim::systems::UserCommands")
    ET.SubElement(
        world, "plugin", filename="gz-sim-scene-broadcaster-system",
        name="gz::sim::systems::SceneBroadcaster")
    sensors = ET.SubElement(
        world, "plugin", filename="gz-sim-sensors-system",
        name="gz::sim::systems::Sensors")
    ET.SubElement(sensors, "render_engine").text = "ogre2"
    ET.SubElement(
        world, "plugin", filename="gz-sim-imu-system",
        name="gz::sim::systems::Imu")
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "pose").text = "0 0 30 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.9 0.9 0.9 1"
    ET.SubElement(light, "specular").text = "0.25 0.25 0.25 1"
    ET.SubElement(light, "direction").text = "-0.35 0.25 -0.9"
    ET.SubElement(light, "cast_shadows").text = "true"
    add_box(world, "floor1", (0, 0, -0.1), (48, 40, .2), ".42 .44 .47 1")
    floor2_size = numbers(field(building_records["floor2"], "size_xy"))
    void_size = numbers(field(building_records["floor2"], "central_void_xy"))
    f2w, f2h = floor2_size
    vw, vh = void_size
    north_depth = (f2h - vh) / 2
    side_width = (f2w - vw) / 2
    add_box(world, "floor2_north", (0, (f2h+vh)/4, 3.9),
            (f2w, north_depth, .2), ".38 .40 .44 1")
    add_box(world, "floor2_south", (0, -(f2h+vh)/4, 3.9),
            (f2w, north_depth, .2), ".38 .40 .44 1")
    add_box(world, "floor2_west", (-(f2w+vw)/4, 0, 3.9),
            (side_width, vh, .2), ".38 .40 .44 1")
    add_box(world, "floor2_east", ((f2w+vw)/4, 0, 3.9),
            (side_width, vh, .2), ".38 .40 .44 1")
    # Four collision-bearing guardrails enclose the central atrium.
    rail_h, rail_t = 1.1, .08
    add_box(world, "atrium_rail_north", (0, vh/2, 4+rail_h/2),
            (vw, rail_t, rail_h), ".12 .15 .18 1")
    add_box(world, "atrium_rail_south", (0, -vh/2, 4+rail_h/2),
            (vw, rail_t, rail_h), ".12 .15 .18 1")
    add_box(world, "atrium_rail_west", (-vw/2, 0, 4+rail_h/2),
            (rail_t, vh, rail_h), ".12 .15 .18 1")
    add_box(world, "atrium_rail_east", (vw/2, 0, 4+rail_h/2),
            (rail_t, vh, rail_h), ".12 .15 .18 1")
    # Perimeter walls stop vehicles and eliminate cross-floor rays.
    for floor, z, sx, sy in ((1, 1.0, 48, 40), (2, 5.0, 30, 24)):
        # Floor 2 leaves 2 m openings at all four corners for external stairs.
        gap = 2.0 if floor == 2 else 0.0
        add_box(world, f"f{floor}_wall_n", (0, sy/2, z), (sx-2*gap, .2, 2), ".7 .72 .75 1")
        add_box(world, f"f{floor}_wall_s", (0, -sy/2, z), (sx-2*gap, .2, 2), ".7 .72 .75 1")
        add_box(world, f"f{floor}_wall_e", (sx/2, 0, z), (.2, sy-2*gap, 2), ".7 .72 .75 1")
        add_box(world, f"f{floor}_wall_w", (-sx/2, 0, z), (.2, sy-2*gap, 2), ".7 .72 .75 1")
    for stair_id, record in stair_records.items():
        add_stair(world, stair_id, numbers(field(record, "floor1_xy")),
                  numbers(field(record, "floor2_xy")))
    for point_id, record in task_records.items():
        xyz = numbers(field(record, "xyz"))
        add_cylinder(world, "task_" + point_id.lower(), (xyz[0], xyz[1], xyz[2] + .015),
                     .45, .03, "1 .48 .05 1")
    for point_id, record in handover_records.items():
        xyz = numbers(field(record, "xyz"))
        add_cylinder(world, "handover_" + point_id.lower(), (xyz[0], xyz[1], xyz[2] + .02),
                     .7, .04, "0 .8 .78 .7")
    for light_id, record in light_records.items():
        add_point_light(world, light_id, numbers(field(record, "xyz")),
                        float(field(record, "intensity")),
                        float(field(record, "range")))
    obstacle_maps = {1: [], 2: []}
    obstacle_maps[2].append((0.0, 0.0, vw, vh))
    for obstacle_id, record in obstacle_records.items():
        floor = int(float(field(record, "floor")))
        xyz, size = numbers(field(record, "xyz")), numbers(field(record, "size_xyz"))
        add_box(world, obstacle_id.lower(), tuple(xyz), tuple(size), ".12 .46 .72 1")
        obstacle_maps[floor].append((xyz[0], xyz[1], size[0], size[1]))
    for column_id, record in column_records.items():
        floor = int(float(field(record, "floor")))
        xyz = numbers(field(record, "xyz"))
        radius, height = float(field(record, "radius")), float(field(record, "height"))
        add_column(world, column_id, xyz, radius, height)
        obstacle_maps[floor].append((xyz[0], xyz[1], radius * 2, radius * 2))
    # Painted centerlines are visual-only; planning uses the generated occupancy maps.
    for lane_id, points in lanes.items():
        z = 4.015 if lane_id.startswith("floor2") else .015
        for index, (a, b) in enumerate(zip(points, points[1:])):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length, yaw = hypot(dx, dy), atan2(dy, dx)
            model = ET.SubElement(world, "model", name=f"lane_{lane_id}_{index}")
            ET.SubElement(model, "static").text = "true"
            ET.SubElement(model, "pose").text = f"{(a[0]+b[0])/2:.4f} {(a[1]+b[1])/2:.4f} {z:.4f} 0 0 {yaw:.6f}"
            link = ET.SubElement(model, "link", name="link")
            visual = ET.SubElement(link, "visual", name="visual")
            geo = ET.SubElement(visual, "geometry")
            ET.SubElement(ET.SubElement(geo, "box"), "size").text = f"{length:.4f} .08 .01"
            mat = ET.SubElement(visual, "material")
            ET.SubElement(mat, "ambient").text = "0.95 0.85 0.12 1"
    for robot_id, record in robot_records.items():
        robot_type = field(record, "type")
        pose = numbers(field(record, "initial_xyz_yaw"))
        include = ET.SubElement(world, "include")
        model_uri = "model://go2_description" if robot_type == "dog" else "model://carter_description"
        ET.SubElement(include, "uri").text = model_uri
        ET.SubElement(include, "name").text = robot_id
        # Go2 bases are driven by the SCAN pose bridge while their twelve
        # articulated joints are animated by kinematic gait controllers.
        ET.SubElement(include, "static").text = "false"
        ET.SubElement(include, "pose").text = (
            f"{pose[0]:.4f} {pose[1]:.4f} {pose[2]:.4f} 0 0 {pose[3]:.4f}")
    ET.indent(sdf, space="  ")
    ET.ElementTree(sdf).write(WORLD, encoding="utf-8", xml_declaration=True)
    write_map("floor1", 48, 40, 0, obstacle_maps[1])
    write_map("floor2", 30, 24, 4, obstacle_maps[2])
    print(f"Generated {WORLD} with {len(stair_records)} stairs, "
          f"{len(task_records)} task points, {len(handover_records)} handover points, "
          f"{len(obstacle_records)} platforms, {len(column_records)} columns, "
          f"{len(light_records)} lights, {len(robot_records)} robots "
          f"and {len(lanes)} lane networks")


if __name__ == "__main__":
    main()
