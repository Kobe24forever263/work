"""Publish labels plus a kinematic cargo cube for the RViz handover demo."""

from copy import deepcopy
from math import cos, sin

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from warehouse_interfaces.msg import RobotState


class RobotMarkers(Node):
    def __init__(self):
        super().__init__("rviz_robot_markers")
        self._floor1_car = self.declare_parameter(
            "floor1_car", "car_f1_1").value
        self._dog = self.declare_parameter("dog", "dog_1").value
        self._floor2_car = self.declare_parameter(
            "floor2_car", "car_f2_2").value
        robot_ids_csv = self.declare_parameter(
            "robot_ids_csv", "").value.strip()
        self._robots = tuple(dict.fromkeys(
            item.strip() for item in robot_ids_csv.split(",")
            if item.strip())) if robot_ids_csv else (
                self._floor1_car, self._dog, self._floor2_car)
        self._states = {}
        self._cargo_owner = ""
        self._delivered_xyz = None
        self._cargo_failed = False
        self._publisher = self.create_publisher(
            MarkerArray, "/task2/rviz_robot_markers", 10)
        for robot in self._robots:
            self.create_subscription(
                RobotState,
                f"/{robot}/robot_state",
                lambda message, name=robot: self._states.__setitem__(
                    name, message),
                20,
            )
        self.create_subscription(
            String, "/task2/cargo_owner", self._on_cargo_owner, 10)
        self.create_timer(0.1, self._publish)

    def _on_cargo_owner(self, message):
        value = message.data.strip()
        if value.startswith(("DELIVERED:", "PICKUP:", "FAILED:")):
            try:
                self._delivered_xyz = tuple(
                    float(item) for item in value.split(":")[1:4])
            except (TypeError, ValueError):
                self.get_logger().error(f"Invalid delivered cargo pose: {value}")
                return
            self._cargo_owner = ""
            self._cargo_failed = value.startswith("FAILED:")
        else:
            self._cargo_owner = value
            self._delivered_xyz = None
            self._cargo_failed = False
        self.get_logger().info(
            f"Cargo C_DEMO owner -> {value or 'NONE'}")

    def _cargo_markers(self):
        if self._cargo_owner:
            state = self._states.get(self._cargo_owner)
            if state is None:
                return []
            pose = deepcopy(state.pose)
            # Keep the cube clearly above the full URDF body.  Dense SCAN
            # point clouds otherwise visually swallow a marker embedded in it.
            pose.position.z += (
                0.58 if self._cargo_owner.startswith("dog_") else 0.54)
            owner_text = self._cargo_owner
            stamp = state.stamp
        elif self._delivered_xyz is not None:
            pose = deepcopy(next(iter(self._states.values())).pose)
            pose.position.x, pose.position.y, pose.position.z = self._delivered_xyz
            pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            owner_text = "DELIVERED"
            stamp = self.get_clock().now().to_msg()
        else:
            return []

        cube = Marker()
        cube.header.frame_id = "map"
        cube.header.stamp = stamp
        cube.ns = "task2_cargo"
        cube.id = 100
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose = pose
        cube.scale.x = cube.scale.y = cube.scale.z = 0.42
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = (
            (0.95, 0.08, 0.08, 1.0) if self._cargo_failed else
            (1.0, 0.48, 0.03, 1.0))

        return [cube]

    @staticmethod
    def _orientation(pose):
        return pose.orientation

    def _base(self, state, marker_id, kind):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = state.stamp
        marker.ns = "task2_robots"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose = state.pose
        marker.color.a = 0.95
        if kind == "car":
            marker.type = Marker.CUBE
            marker.scale.x = 1.0
            marker.scale.y = 0.70
            marker.scale.z = 0.28
            marker.pose.position.z += 0.22
            marker.color.r, marker.color.g, marker.color.b = 0.05, 0.35, 0.95
        else:
            marker.type = Marker.CUBE
            marker.scale.x = 0.82
            marker.scale.y = 0.45
            marker.scale.z = 0.30
            marker.pose.position.z += 0.52
            marker.color.r, marker.color.g, marker.color.b = 0.92, 0.92, 0.96
        return marker

    def _label(self, state, marker_id, text, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = state.stamp
        marker.ns = "task2_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = state.pose.position.x
        marker.pose.position.y = state.pose.position.y
        marker.pose.position.z = state.pose.position.z + 1.35
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.55
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (*color, 1.0)
        marker.text = text
        return marker

    def _publish(self):
        output = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)
        for index, robot in enumerate(self._robots, start=1):
            state = self._states.get(robot)
            if state is None:
                continue
            is_dog = robot.startswith("dog_")
            output.markers.append(self._label(
                state, 10 + index,
                f"{'Go2' if is_dog else 'Carter'} {robot}",
                (1.0, 0.85, 0.1) if is_dog else (0.15, 0.72, 1.0)))
        output.markers.extend(self._cargo_markers())
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = RobotMarkers()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
