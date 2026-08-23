#!/usr/bin/env python3
"""Wait until one SCAN open-loop agent is stably inside a goal tolerance."""

import argparse
from math import sqrt
import sys
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


class PoseWaiter(Node):
    def __init__(self, robot, target, tolerance, stable_seconds):
        super().__init__(f"wait_for_{robot}_scan_pose")
        self.target = target
        self.tolerance = tolerance
        self.stable_seconds = stable_seconds
        self.inside_since = None
        self.done = False
        self.armed = False
        self.last_distance = float("inf")
        self.create_subscription(Odometry, f"/{robot}/odom", self._odom, 20)

    def _odom(self, message):
        point = message.pose.pose.position
        self.last_distance = sqrt(
            (point.x - self.target[0]) ** 2
            + (point.y - self.target[1]) ** 2
            + (point.z - self.target[2]) ** 2)
        now = time.monotonic()
        # Ignore a retained/stale final odometry sample from a previous run.
        # The new controller always publishes its configured start pose first.
        if self.last_distance > max(self.tolerance * 2.0, 0.75):
            self.armed = True
        if not self.armed:
            return
        if self.last_distance <= self.tolerance:
            self.inside_since = self.inside_since or now
            self.done = now - self.inside_since >= self.stable_seconds
        else:
            self.inside_since = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--stable", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    rclpy.init()
    node = PoseWaiter(
        args.robot, (args.x, args.y, args.z), args.tolerance, args.stable)
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and not node.done and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.done:
        print(f"SCAN STAGE PASS {args.robot} distance={node.last_distance:.3f}")
        result = 0
    else:
        print(f"SCAN STAGE FAIL {args.robot} distance={node.last_distance:.3f}")
        result = 1
    node.destroy_node()
    rclpy.shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())
