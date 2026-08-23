#!/usr/bin/env python3
"""Send one configured Carter to its requested stair handover pose."""

import argparse
from math import cos, sin
from pathlib import Path
import sys
import time

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
import yaml


WORK = Path("/Users/lab4099/Desktop/Mujoco/work")
CONFIG = WORK / "src" / "warehouse_bringup" / "config"
ALLOWED = {
    "car_f1_1": ("STAIR_NE",),
    "car_f1_2": ("STAIR_NW",),
    "car_f1_3": ("STAIR_SW",),
    "car_f1_4": ("STAIR_SE",),
    "car_f2_1": ("STAIR_NW", "STAIR_SW"),
    "car_f2_2": ("STAIR_NE", "STAIR_SE"),
}


def pose(navigator, x, y, yaw):
    message = PoseStamped()
    message.header.frame_id = "map"
    # Zero requests the latest transform and avoids mixing the command
    # process wall clock with Gazebo simulation time.
    message.pose.position.x = float(x)
    message.pose.position.y = float(y)
    message.pose.orientation.z = sin(yaw / 2.0)
    message.pose.orientation.w = cos(yaw / 2.0)
    return message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", choices=tuple(ALLOWED))
    parser.add_argument("stair")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.stair not in ALLOWED[args.robot]:
        parser.error(
            f"{args.robot} may only serve {', '.join(ALLOWED[args.robot])}")

    robots = yaml.safe_load(
        (CONFIG / "robots.yaml").read_text(encoding="utf-8"))["robots"]
    scene = yaml.safe_load(
        (CONFIG / "scene.yaml").read_text(encoding="utf-8"))
    floor = int(robots[args.robot]["home_floor"])
    suffix = "F1_SEND" if floor == 1 else "F2_RECEIVE"
    target = scene["handover_points"][f"{args.stair}_{suffix}"]
    initial = robots[args.robot]["initial_xyz_yaw"]

    rclpy.init()
    navigator = BasicNavigator(
        node_name=f"{args.robot}_stair_goal", namespace=args.robot)
    navigator.setInitialPose(
        pose(navigator, initial[0], initial[1], initial[3]))
    navigator.waitUntilNav2Active()
    navigator.goToPose(
        pose(navigator, target["xyz"][0], target["xyz"][1], target["yaw"]))
    started = time.monotonic()
    while not navigator.isTaskComplete():
        if time.monotonic() - started > args.timeout:
            navigator.cancelTask()
            print("Carter navigation: TIMEOUT")
            navigator.destroy_node()
            rclpy.shutdown()
            return 1
        time.sleep(0.2)
    result = navigator.getResult()
    navigator.destroy_node()
    rclpy.shutdown()
    if result == TaskResult.SUCCEEDED:
        print(
            f"Carter navigation: PASS robot={args.robot} "
            f"target={args.stair}_{suffix}")
        return 0
    print(f"Carter navigation: FAIL result={result}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
