#!/usr/bin/env python3
"""Generate a namespaced Carter Nav2+MPPI config from the installed baseline."""

from pathlib import Path
from copy import deepcopy

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/"
    "share/nav2_bringup/params/nav2_params.yaml"
)
OUTPUT = (
    ROOT / "src" / "warehouse_bringup" / "config" / "nav2_carter.yaml"
)
ROBOTS = (
    "car_f1_1", "car_f1_2", "car_f1_3",
    "car_f1_4", "car_f2_1", "car_f2_2",
)
NS = "<robot_namespace>"
BASE = f"{NS}/base_link"
ODOM = f"{NS}/odom"


def params(config, node):
    return config[node]["ros__parameters"]


def main():
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))

    amcl = params(config, "amcl")
    amcl.update({
        "base_frame_id": BASE,
        "odom_frame_id": ODOM,
        "global_frame_id": "map",
        "scan_topic": "scan",
        # Interactive/demo mode uses Gazebo ground-truth map->odom from launch.
        # AMCL still visualizes localization but must not publish a competing TF.
        "tf_broadcast": False,
        "max_particles": 1200,
        "min_particles": 300,
        "update_min_d": 0.08,
        "update_min_a": 0.08,
    })
    bt = params(config, "bt_navigator")
    bt.update({
        "global_frame": "map",
        "robot_base_frame": BASE,
        "odom_topic": f"{NS}/odom",
    })
    controller = params(config, "controller_server")
    controller.update({
        "odom_topic": f"{NS}/odom",
        "controller_frequency": 20.0,
    })
    controller["progress_checker"].update({
        "required_movement_radius": 0.15,
        "movement_time_allowance": 20.0,
    })
    controller["general_goal_checker"].update({
        "xy_goal_tolerance": 0.30,
        "yaw_goal_tolerance": 0.30,
    })
    mppi = controller["FollowPath"]
    mppi.update({
        "motion_model": "DiffDrive",
        "time_steps": 32,
        "batch_size": 400,
        "vx_max": 0.35,
        "vx_min": -0.20,
        "wz_max": 1.2,
        "ax_max": 0.55,
        "ax_min": -0.55,
        "visualize": False,
    })

    local = config["local_costmap"]["local_costmap"]["ros__parameters"]
    local.update({
        "global_frame": ODOM,
        "robot_base_frame": BASE,
        "rolling_window": True,
        "width": 5,
        "height": 5,
        "resolution": 0.10,
        "footprint": "[[0.48, 0.34], [0.48, -0.34], "
                     "[-0.48, -0.34], [-0.48, 0.34]]",
        "plugins": ["obstacle_layer", "inflation_layer"],
        "obstacle_layer": {
            "plugin": "nav2_costmap_2d::ObstacleLayer",
            "enabled": True,
            "observation_sources": "scan",
            "scan": {
                "topic": "scan",
                "data_type": "LaserScan",
                "clearing": True,
                "marking": True,
                "raytrace_max_range": 12.0,
                "obstacle_max_range": 10.0,
                "max_obstacle_height": 2.0,
            },
        },
    })
    local["inflation_layer"].update({
        "inflation_radius": 0.70,
        "cost_scaling_factor": 4.0,
    })
    local.pop("voxel_layer", None)

    global_costmap = (
        config["global_costmap"]["global_costmap"]["ros__parameters"])
    global_costmap.update({
        "global_frame": "map",
        "robot_base_frame": BASE,
        "resolution": 0.10,
        "footprint": local["footprint"],
    })
    global_costmap["obstacle_layer"]["scan"].update({
        "topic": "scan",
        "raytrace_max_range": 12.0,
        "obstacle_max_range": 10.0,
    })
    global_costmap["inflation_layer"].update({
        "inflation_radius": 0.70,
        "cost_scaling_factor": 4.0,
    })

    planner = params(config, "planner_server")
    planner["GridBased"] = {
        "plugin": "nav2_smac_planner::SmacPlanner2D",
        "tolerance": 0.25,
        "downsample_costmap": False,
        "allow_unknown": False,
        "max_iterations": 100000,
        "max_on_approach_iterations": 1000,
        "max_planning_time": 2.0,
        "cost_travel_multiplier": 2.0,
        "use_final_approach_orientation": False,
    }
    behavior = params(config, "behavior_server")
    behavior.update({
        "global_frame": "map",
        "robot_base_frame": BASE,
        "local_frame": ODOM,
    })
    params(config, "velocity_smoother").update({
        "odom_topic": f"{NS}/odom",
        "max_velocity": [0.35, 0.0, 1.2],
        "min_velocity": [-0.20, 0.0, -1.2],
        "max_accel": [0.55, 0.0, 1.5],
        "max_decel": [-0.55, 0.0, -1.5],
    })
    if "collision_monitor" in config:
        params(config, "collision_monitor").update({
            "base_frame_id": BASE,
            "odom_frame_id": ODOM,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    for robot in ROBOTS:
        specific = yaml.safe_load(
            yaml.safe_dump(config, sort_keys=False).replace(NS, robot))
        params(specific, "bt_navigator")["odom_topic"] = f"/{robot}/odom"
        params(specific, "controller_server")["odom_topic"] = \
            f"/{robot}/odom"
        params(specific, "velocity_smoother")["odom_topic"] = \
            f"/{robot}/odom"
        if "docking_server" in specific:
            params(specific, "docking_server")["base_frame"] = \
                f"{robot}/base_link"
            params(specific, "docking_server")["fixed_frame"] = "map"
        output = OUTPUT.with_name(f"nav2_{robot}.yaml")
        output.write_text(
            yaml.safe_dump(specific, sort_keys=False), encoding="utf-8")
        generated.append(output.name)
    print("Generated " + ", ".join(generated))


if __name__ == "__main__":
    main()
