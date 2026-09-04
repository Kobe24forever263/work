"""RViz status and active-SCAN topic relay for the ten-task campaign."""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from nav_msgs.msg import Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class Stage26ScanCampaignMonitor(Node):
    def __init__(self):
        super().__init__("stage26_scan_campaign_monitor")
        campaign_file = Path(
            self.declare_parameter("campaign_file", "").value)
        if not campaign_file.is_file():
            raise FileNotFoundError(
                f"Stage 26 campaign not found: {campaign_file}")
        self._campaign = json.loads(
            campaign_file.read_text(encoding="utf-8"))
        if self._campaign.get("scope") != "TEN_SEQUENTIAL_TASKS":
            raise ValueError("monitor accepts TEN_SEQUENTIAL_TASKS only")
        self._task_index = 0
        self._segment_index = 0
        self._active_robot = ""
        self._status = "等待第 1/10 个随机任务"
        self._marker_pub = self.create_publisher(
            MarkerArray, "/task2/stage26_scan/campaign_markers", 10)
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/task2/stage26_scan/local_cloud",
            qos_profile_sensor_data)
        self._occupancy_pub = self.create_publisher(
            PointCloud2, "/task2/stage26_scan/occupancy",
            qos_profile_sensor_data)
        self._inflated_pub = self.create_publisher(
            PointCloud2, "/task2/stage26_scan/occupancy_inflate",
            qos_profile_sensor_data)
        self._bbox_pub = self.create_publisher(
            Marker, "/task2/stage26_scan/sliding_map_bbox", 10)
        self._path_pub = self.create_publisher(
            PathMessage, "/task2/stage26_scan/active_path", 10)
        self.create_subscription(
            String, "/task2/stage26_scan/control", self._on_control, 10)
        for robot in self._campaign["robot_ids"]:
            self.create_subscription(
                PointCloud2, f"/{robot}/cloud",
                lambda msg, name=robot: self._relay_cloud(
                    name, msg, self._cloud_pub), qos_profile_sensor_data)
            self.create_subscription(
                PointCloud2, f"/{robot}/grid_map/occupancy",
                lambda msg, name=robot: self._relay_cloud(
                    name, msg, self._occupancy_pub), qos_profile_sensor_data)
            self.create_subscription(
                PointCloud2, f"/{robot}/grid_map/occupancy_inflate",
                lambda msg, name=robot: self._relay_cloud(
                    name, msg, self._inflated_pub), qos_profile_sensor_data)
            self.create_subscription(
                Marker, f"/{robot}/grid_map/sliding_map_bbox",
                lambda msg, name=robot: self._relay_marker(name, msg), 10)
            self.create_subscription(
                PathMessage, f"/{robot}/path",
                lambda msg, name=robot: self._relay_path(name, msg), 10)
        self.create_timer(0.1, self._publish_markers)
        self.get_logger().info(
            "Stage 26 SCAN monitor ready for 10 sequential tasks")

    def _relay_cloud(self, robot, message, publisher):
        if robot == self._active_robot:
            publisher.publish(message)

    def _relay_marker(self, robot, message):
        if robot == self._active_robot:
            self._bbox_pub.publish(message)

    def _relay_path(self, robot, message):
        if robot == self._active_robot:
            self._path_pub.publish(message)

    def _clear_active_scan(self):
        stamp = self.get_clock().now().to_msg()
        for publisher in (
                self._cloud_pub, self._occupancy_pub,
                self._inflated_pub):
            empty = PointCloud2()
            empty.header.stamp = stamp
            empty.header.frame_id = "map"
            publisher.publish(empty)
        empty_path = PathMessage()
        empty_path.header.stamp = stamp
        empty_path.header.frame_id = "map"
        self._path_pub.publish(empty_path)

    def _on_control(self, message):
        value = message.data.strip()
        if value.startswith("TASK:"):
            index = int(value.split(":", 1)[1]) - 1
            if not 0 <= index < len(self._campaign["tasks"]):
                self.get_logger().error(f"invalid task control: {value}")
                return
            self._task_index = index
            self._segment_index = 0
            task = self._campaign["tasks"][index]
            mode = task["policy_decision"]["assignment"]["transport_mode"]
            self._status = (
                f"任务 {index + 1}/10 已发布 | {task['task_type']} | {mode}")
        elif value.startswith("ACTIVE:"):
            parts = value.split(":", 3)
            if len(parts) == 4:
                _, backend, robot, segment = parts
            else:
                _, robot, segment = parts
                backend = "SCAN"
            self._clear_active_scan()
            self._active_robot = robot
            self._segment_index = max(0, int(segment) - 1)
            task = self._campaign["tasks"][self._task_index]
            label = task["segments"][self._segment_index]["label"]
            self._status = (
                f"任务 {self._task_index + 1}/10 | {backend} {robot} | {label}")
        elif value.startswith("TASK_COMPLETE:"):
            index = int(value.split(":", 1)[1])
            self._status = f"任务 {index}/10 完成，准备发布下一任务"
        elif value == "CAMPAIGN_COMPLETE":
            self._active_robot = ""
            self._clear_active_scan()
            self._status = "10/10 任务全部完成；机器人位置未重置"
        elif value.startswith("FAILED:"):
            self._status = value.replace("FAILED:", "验收失败：", 1)
        self.get_logger().info(value)

    def _publish_markers(self):
        task = self._campaign["tasks"][self._task_index]
        source = task["task"]["source"]["xyz"]
        target = task["task"]["target"]["xyz"]
        stamp = self.get_clock().now().to_msg()
        output = MarkerArray()
        for marker_id, (xyz, label, color) in enumerate((
                (source, "当前任务起点", (0.1, 1.0, 0.2)),
                (target, "当前任务终点", (1.0, 0.15, 0.15))), start=1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "stage26_scan_task_points"
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
            marker.color.a = 0.9
            output.markers.append(marker)
            text = Marker()
            text.header = marker.header
            text.ns = "stage26_scan_task_labels"
            text.id = marker_id + 10
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(xyz[0])
            text.pose.position.y = float(xyz[1])
            text.pose.position.z = float(xyz[2]) + 1.0
            text.pose.orientation.w = 1.0
            text.scale.z = 0.48
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = label
            output.markers.append(text)
        status = Marker()
        status.header.frame_id = "map"
        status.header.stamp = stamp
        status.ns = "stage26_scan_status"
        status.id = 100
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.z = 7.0
        status.pose.orientation.w = 1.0
        status.scale.z = 0.65
        status.color.r, status.color.g, status.color.b = (1.0, 0.82, 0.08)
        status.color.a = 1.0
        seed = self._campaign["checkpoint_selection"]["selected_seed_index"]
        status.text = f"Stage 26 seed {seed:02d} | {self._status}"
        output.markers.append(status)
        self._marker_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = Stage26ScanCampaignMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
