#!/usr/bin/env python3
"""Smoothly publish a sub-two-metre alignment leg without claiming SCAN."""

from __future__ import annotations

import argparse
import json
from math import atan2, cos, dist, sin
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


class PoseSync(Node):
    def __init__(self, robot, path, speed):
        super().__init__(f"stage26_pose_sync_{robot}")
        self.publisher = self.create_publisher(Odometry, f"/{robot}/odom", 10)
        self.robot = robot
        self.path = [tuple(float(value) for value in point) for point in path]
        self.speed = max(0.05, float(speed))

    def run(self):
        for start, target in zip(self.path, self.path[1:]):
            length = dist(start, target)
            duration = max(0.2, length / self.speed)
            begin = time.monotonic()
            yaw = atan2(target[1] - start[1], target[0] - start[0])
            while True:
                elapsed = time.monotonic() - begin
                ratio = min(1.0, elapsed / duration)
                xyz = tuple(
                    start[index] + (target[index] - start[index]) * ratio
                    for index in range(3))
                message = Odometry()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = "map"
                message.child_frame_id = self.robot
                message.pose.pose.position.x = xyz[0]
                message.pose.pose.position.y = xyz[1]
                message.pose.pose.position.z = xyz[2]
                message.pose.pose.orientation.z = sin(yaw / 2.0)
                message.pose.pose.orientation.w = cos(yaw / 2.0)
                message.twist.twist.linear.x = self.speed if ratio < 1 else 0.0
                self.publisher.publish(message)
                rclpy.spin_once(self, timeout_sec=0.0)
                if ratio >= 1.0:
                    break
                time.sleep(1.0 / 30.0)
        for _ in range(8):
            self.publisher.publish(message)
            time.sleep(1.0 / 30.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True)
    parser.add_argument("--path-json", required=True)
    parser.add_argument("--speed", required=True, type=float)
    args = parser.parse_args()
    path = json.loads(args.path_json)
    rclpy.init()
    node = PoseSync(args.robot, path, args.speed)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
