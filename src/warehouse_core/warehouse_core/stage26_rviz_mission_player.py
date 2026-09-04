"""Replay one Stage 26 policy decision with full robot bodies in RViz."""

from __future__ import annotations

from copy import deepcopy
import json
from math import atan2, cos, dist, sin
from pathlib import Path
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray


class Stage26RvizMissionPlayer(Node):
    def __init__(self):
        super().__init__("stage26_rviz_mission_player")
        plan_file = Path(self.declare_parameter("plan_file", "").value)
        if not plan_file.exists():
            raise FileNotFoundError(f"Stage 26 RViz plan not found: {plan_file}")
        self._plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if self._plan.get("scope") != "ONE_RANDOM_TASK_ONLY":
            raise ValueError("RViz player accepts one-task plans only")
        self._time_scale = float(
            self.declare_parameter("time_scale", 1.0).value)
        if self._time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self._display = self._plan["display_robots"]
        self._robots = tuple(self._plan.get(
            "robot_ids", self._plan["robot_initial_poses"].keys()))
        if len(self._robots) != 10:
            raise ValueError(
                "Stage 26 single-task acceptance must display all 10 robots")
        self._positions = {
            name: tuple(self._plan["robot_initial_poses"][name])
            for name in self._robots}
        self._yaw = {name: 0.0 for name in self._robots}
        self._speed = {name: 0.0 for name in self._robots}
        self._odom_pubs = {
            name: self.create_publisher(Odometry, f"/{name}/odom", 20)
            for name in self._robots}
        self._cargo_pub = self.create_publisher(
            String, "/task2/cargo_owner", 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/task2/stage26_acceptance_markers", 10)
        self._path_pubs = {
            role: self.create_publisher(
                PathMessage, f"/task2/stage26/path_{role}", 10)
            for role in self._display}
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/map_generator/global_cloud", QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._segments = self._plan["segments"]
        self._cloud = self._load_cloud()
        self._planned_paths = self._build_planned_paths()
        self._segment_index = 0
        self._waypoint_index = 1
        self._pause_remaining = 0.0
        self._complete = False
        self._cargo_state = "PICKUP"
        self._cargo_repeats = 30
        self._status = "任务已下发，准备执行"
        self._last_tick = time.monotonic()
        if self._segments:
            self._start_segment(0)
        self.create_timer(1.0 / 30.0, self._tick)
        self.create_timer(2.0, self._publish_static)
        decision = self._plan["policy_decision"]["assignment"]
        self.get_logger().info(
            "Stage 26 one-task RViz acceptance: "
            f"{self._plan['task']['task_id']} -> "
            f"{decision['transport_mode']}; no further tasks will be released")

    def _load_cloud(self):
        pcd = (Path(__file__).resolve().parents[3] / "warehouse_bringup" /
               "maps" / "warehouse_full.pcd")
        if not pcd.exists():
            # Installed entry points resolve under install/, so retain the
            # canonical local project fallback used by existing launch files.
            pcd = Path(
                "/Users/lab4099/Desktop/Mujoco/work/src/warehouse_bringup/"
                "maps/warehouse_full.pcd")
        data = np.loadtxt(pcd, skiprows=11, dtype=np.float32)[::4]
        header = Header()
        header.frame_id = "map"
        return point_cloud2.create_cloud_xyz32(header, data)

    def _build_planned_paths(self):
        paths = {name: [self._positions[name]] for name in self._robots}
        for segment in self._segments:
            paths[segment["robot"]].extend(
                tuple(point) for point in segment["path"])
        return {name: self._deduplicate(points)
                for name, points in paths.items()}

    @staticmethod
    def _deduplicate(points):
        output = []
        for point in points:
            point = tuple(float(item) for item in point)
            if not output or dist(output[-1], point) > 1e-6:
                output.append(point)
        return output

    def _cargo_value(self, state):
        if state == "PICKUP":
            x, y, z = self._plan["cargo_pickup_xyz"]
            return f"PICKUP:{x}:{y}:{z + 0.06}"
        if state == "DELIVERED":
            x, y, z = self._plan["cargo_delivery_xyz"]
            return f"DELIVERED:{x}:{y}:{z + 0.06}"
        return state

    def _start_segment(self, index):
        segment = self._segments[index]
        self._status = segment["label"]
        self._cargo_state = segment["cargo_owner"]
        self._cargo_repeats = 15
        robot = segment["robot"]
        path = [tuple(point) for point in segment["path"]]
        if path:
            self._positions[robot] = path[0]
        self._waypoint_index = 1

    def _advance(self, dt):
        if self._complete or not self._segments:
            return
        if self._pause_remaining > 0:
            self._pause_remaining = max(0.0, self._pause_remaining - dt)
            return
        segment = self._segments[self._segment_index]
        robot = segment["robot"]
        path = [tuple(point) for point in segment["path"]]
        self._speed = {name: 0.0 for name in self._robots}
        budget = float(segment["speed_mps"]) * dt * self._time_scale
        while budget > 0 and self._waypoint_index < len(path):
            current = self._positions[robot]
            target = path[self._waypoint_index]
            remaining = dist(current, target)
            if remaining <= 1e-8:
                self._waypoint_index += 1
                continue
            ratio = min(1.0, budget / remaining)
            next_position = tuple(
                current[i] + (target[i] - current[i]) * ratio
                for i in range(3))
            self._yaw[robot] = atan2(
                target[1] - current[1], target[0] - current[0])
            self._positions[robot] = next_position
            self._speed[robot] = float(segment["speed_mps"])
            budget -= min(budget, remaining)
            if ratio >= 1.0:
                self._waypoint_index += 1
        if self._waypoint_index >= len(path):
            self._speed[robot] = 0.0
            self._pause_remaining = float(segment.get("pause_after_s", 0.0))
            self._segment_index += 1
            if self._segment_index >= len(self._segments):
                self._complete = True
                self._status = "单任务验收完成；不会继续发布任务"
                self._cargo_state = "DELIVERED"
                self._cargo_repeats = 30
                self.get_logger().info(
                    "Stage 26 one-task RViz acceptance completed; "
                    "holding final state with no further task release")
            else:
                self._start_segment(self._segment_index)

    def _publish_odometry(self, stamp):
        moving = (None if self._complete else
                  self._segments[min(
                      self._segment_index, len(self._segments) - 1)]["robot"])
        for name in self._robots:
            message = Odometry()
            message.header.stamp = stamp
            message.header.frame_id = "map"
            message.child_frame_id = name
            x, y, z = self._positions[name]
            message.pose.pose.position.x = x
            message.pose.pose.position.y = y
            message.pose.pose.position.z = z
            yaw = self._yaw[name]
            message.pose.pose.orientation.z = sin(yaw / 2.0)
            message.pose.pose.orientation.w = cos(yaw / 2.0)
            message.twist.twist.linear.x = (
                self._speed[name] if moving == name else 0.0)
            self._odom_pubs[name].publish(message)

    def _publish_cargo(self):
        if self._cargo_repeats <= 0:
            return
        message = String()
        message.data = self._cargo_value(self._cargo_state)
        self._cargo_pub.publish(message)
        self._cargo_repeats -= 1

    def _markers(self, stamp):
        output = MarkerArray()
        source = self._plan["cargo_pickup_xyz"]
        target = self._plan["cargo_delivery_xyz"]
        for marker_id, (xyz, label, color) in enumerate((
                (source, "随机任务起点", (0.1, 1.0, 0.2)),
                (target, "随机任务终点", (1.0, 0.15, 0.15))), start=1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "stage26_task_points"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xyz[0])
            marker.pose.position.y = float(xyz[1])
            marker.pose.position.z = float(xyz[2]) + 0.22
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 1.15
            marker.scale.z = 0.18
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.85
            output.markers.append(marker)
            text = deepcopy(marker)
            text.ns = "stage26_task_labels"
            text.id = marker_id + 10
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.z += 0.8
            text.scale.z = 0.48
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = label
            output.markers.append(text)
        status = Marker()
        status.header.frame_id = "map"
        status.header.stamp = stamp
        status.ns = "stage26_status"
        status.id = 100
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.x = 0.0
        status.pose.position.y = 0.0
        status.pose.position.z = 6.6
        status.pose.orientation.w = 1.0
        status.scale.z = 0.65
        status.color.r, status.color.g, status.color.b = (1.0, 0.82, 0.08)
        status.color.a = 1.0
        mode = self._plan["policy_decision"]["assignment"]["transport_mode"]
        seed = self._plan["checkpoint_selection"]["selected_seed_index"]
        status.text = f"Stage 26 seed {seed:02d} | {mode} | {self._status}"
        output.markers.append(status)
        return output

    def _publish_paths(self, stamp):
        for role, name in self._display.items():
            message = PathMessage()
            message.header.frame_id = "map"
            message.header.stamp = stamp
            for xyz in self._planned_paths[name]:
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = xyz[0]
                pose.pose.position.y = xyz[1]
                pose.pose.position.z = xyz[2] + 0.08
                pose.pose.orientation.w = 1.0
                message.poses.append(pose)
            self._path_pubs[role].publish(message)

    def _publish_static(self):
        stamp = self.get_clock().now().to_msg()
        self._cloud.header.stamp = stamp
        self._cloud_pub.publish(self._cloud)
        self._publish_paths(stamp)

    def _tick(self):
        tick = time.monotonic()
        dt = min(0.1, max(0.0, tick - self._last_tick))
        self._last_tick = tick
        self._advance(dt)
        stamp = self.get_clock().now().to_msg()
        self._publish_odometry(stamp)
        self._publish_cargo()
        self._marker_pub.publish(self._markers(stamp))


def main(args=None):
    rclpy.init(args=args)
    node = Stage26RvizMissionPlayer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
