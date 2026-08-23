"""Bridge Gazebo Carter odometry to collision-free ROS odom and TF frames."""

import os
from copy import deepcopy
from math import cos, hypot, sin
import threading
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry as RosOdometry
import rclpy
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from tf2_msgs.msg import TFMessage
from warehouse_interfaces.msg import RobotState
import yaml

from gz.msgs10.odometry_pb2 import Odometry as GzOdometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode


CARTERS = (
    "car_f1_1", "car_f1_2", "car_f1_3",
    "car_f1_4", "car_f2_1", "car_f2_2",
)


class CarterStateBridge(RosNode):
    def __init__(self):
        super().__init__("carter_state_bridge")
        self.declare_parameter(
            "robots_config",
            "/Users/lab4099/Desktop/Mujoco/work/src/"
            "warehouse_bringup/config/robots.yaml")
        config_path = self.get_parameter("robots_config").value
        with open(config_path, encoding="utf-8") as stream:
            all_robots = yaml.safe_load(stream)["robots"]
        self._configs = {robot: all_robots[robot] for robot in CARTERS}
        self._tf = TransformBroadcaster(self)
        self._gz = GzNode()
        self._callback_lock = threading.Lock()
        self._active = True
        self._gz_topics = []
        self._odom_publishers = {
            robot: self.create_publisher(RosOdometry, f"/{robot}/odom", 20)
            for robot in CARTERS
        }
        self._state_publishers = {
            robot: self.create_publisher(
                RobotState, f"/{robot}/robot_state", 20)
            for robot in CARTERS
        }
        self._diagnostic_publishers = {
            robot: self.create_publisher(
                DiagnosticArray, f"/{robot}/diagnostics", 10)
            for robot in CARTERS
        }
        self._namespaced_tf_publishers = {
            robot: self.create_publisher(TFMessage, f"/{robot}/tf", 100)
            for robot in CARTERS
        }
        self._scan_publishers = {
            robot: self.create_publisher(
                LaserScan, f"/{robot}/scan", qos_profile_sensor_data)
            for robot in CARTERS
        }
        self._latest_transforms = {robot: None for robot in CARTERS}
        self._scan_subscriptions = [
            self.create_subscription(
                LaserScan,
                f"/{robot}/scan_raw",
                lambda message, name=robot: self._restamp_scan(
                    name, message),
                qos_profile_sensor_data,
            )
            for robot in CARTERS
        ]
        self._last_rx = {robot: 0.0 for robot in CARTERS}
        self._last_progress = {
            robot: self.get_clock().now().to_msg() for robot in CARTERS
        }
        self._motion_cost = {robot: 0.0 for robot in CARTERS}
        self._previous_xy = {robot: (0.0, 0.0) for robot in CARTERS}
        self._callbacks = {}
        self._world_poses = {}
        pose_topic = "/world/two_floor_warehouse/dynamic_pose/info"
        self._gz_topics.append(pose_topic)
        if not self._gz.subscribe(Pose_V, pose_topic, self._pose_callback):
            raise RuntimeError(f"failed to subscribe to {pose_topic}")
        for robot in CARTERS:
            callback = self._make_callback(robot)
            self._callbacks[robot] = callback
            topic = f"/model/{robot}/odometry"
            self._gz_topics.append(topic)
            if not self._gz.subscribe(GzOdometry, topic, callback):
                raise RuntimeError(f"failed to subscribe to {topic}")
        self.get_logger().info(
            "Carter odom/TF bridge ready for: " + ", ".join(CARTERS))
        self.create_timer(1.0, self._publish_diagnostics)

    def _pose_callback(self, message):
        with self._callback_lock:
            for pose in message.pose:
                if pose.name in CARTERS:
                    self._world_poses[pose.name] = deepcopy(pose)

    def _restamp_scan(self, robot, message):
        # Gazebo's rendered sensor header lags the simulation clock on macOS.
        # Re-stamp at bridge reception so TF consumers get a coherent sample.
        message.header.stamp = self.get_clock().now().to_msg()
        with self._callback_lock:
            cached = self._latest_transforms[robot]
            if cached is not None:
                transform = deepcopy(cached)
                transform.header.stamp = message.header.stamp
                self._tf.sendTransform(transform)
                self._namespaced_tf_publishers[robot].publish(
                    TFMessage(transforms=[transform]))
        self._scan_publishers[robot].publish(message)

    def _make_callback(self, robot):
        def callback(message):
            with self._callback_lock:
                if not self._active or not rclpy.ok():
                    return
                self._publish_state(robot, message)
        return callback

    def _publish_state(self, robot, message):
        stamp = self.get_clock().now().to_msg()
        frame = f"{robot}/odom"
        child = f"{robot}/base_link"
        config = self._configs[robot]
        initial = config["initial_xyz_yaw"]
        yaw = float(initial[3])
        world_pose = self._world_poses.get(robot)
        if world_pose is not None:
            dx = world_pose.position.x - float(initial[0])
            dy = world_pose.position.y - float(initial[1])
            rx = cos(yaw) * dx + sin(yaw) * dy
            ry = -sin(yaw) * dx + cos(yaw) * dy
            rz = world_pose.position.z - float(initial[2])
            qz0, qw0 = sin(yaw / 2), cos(yaw / 2)
            qz = qw0 * world_pose.orientation.z - qz0 * world_pose.orientation.w
            qw = qw0 * world_pose.orientation.w + qz0 * world_pose.orientation.z
            world_x, world_y = world_pose.position.x, world_pose.position.y
        else:
            rx, ry, rz = (
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            )
            qz, qw = message.pose.orientation.z, message.pose.orientation.w
            world_x = initial[0] + cos(yaw) * rx - sin(yaw) * ry
            world_y = initial[1] + sin(yaw) * rx + cos(yaw) * ry

        odom = RosOdometry()
        odom.header.stamp = stamp
        odom.header.frame_id = frame
        odom.child_frame_id = child
        odom.pose.pose.position.x = rx
        odom.pose.pose.position.y = ry
        odom.pose.pose.position.z = rz
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = message.twist.linear.x
        odom.twist.twist.linear.y = message.twist.linear.y
        odom.twist.twist.linear.z = message.twist.linear.z
        odom.twist.twist.angular.x = message.twist.angular.x
        odom.twist.twist.angular.y = message.twist.angular.y
        odom.twist.twist.angular.z = message.twist.angular.z
        self._odom_publishers[robot].publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame
        transform.child_frame_id = child
        transform.transform.translation.x = rx
        transform.transform.translation.y = ry
        transform.transform.translation.z = rz
        transform.transform.rotation = odom.pose.pose.orientation
        self._latest_transforms[robot] = deepcopy(transform)
        self._tf.sendTransform(transform)
        self._namespaced_tf_publishers[robot].publish(
            TFMessage(transforms=[transform]))

        qz0, qw0 = sin(yaw / 2), cos(yaw / 2)

        previous = self._previous_xy[robot]
        distance = hypot(rx - previous[0], ry - previous[1])
        self._motion_cost[robot] += distance
        self._previous_xy[robot] = (rx, ry)
        speed = hypot(message.twist.linear.x, message.twist.linear.y)
        if distance > 0.001:
            self._last_progress[robot] = stamp
        self._last_rx[robot] = time.monotonic()

        state = RobotState()
        state.stamp = stamp
        state.robot_id = robot
        state.robot_type = "differential_car"
        state.home_floor = int(config["home_floor"])
        state.current_floor = int(config["home_floor"])
        state.home_region = config["home_region"]
        state.current_region = config["home_region"]
        state.pose.position.x = world_x
        state.pose.position.y = world_y
        state.pose.position.z = float(initial[2])
        state.pose.orientation.z = qz0 * qw + qw0 * qz
        state.pose.orientation.w = qw0 * qw - qz0 * qz
        state.velocity = odom.twist.twist
        state.navigation_state = "MOVING" if speed > 0.02 else "IDLE"
        state.task_id = ""
        state.cargo_id = ""
        state.battery_or_motion_cost = float(self._motion_cost[robot])
        state.available = True
        state.failure_code = ""
        state.last_progress_time = self._last_progress[robot]
        self._state_publishers[robot].publish(state)

    def _publish_diagnostics(self):
        now = self.get_clock().now().to_msg()
        monotonic_now = time.monotonic()
        for robot in CARTERS:
            age = monotonic_now - self._last_rx[robot]
            healthy = age < 1.0
            status = DiagnosticStatus()
            status.level = (
                DiagnosticStatus.OK if healthy else DiagnosticStatus.STALE)
            status.name = f"{robot}/interface"
            status.hardware_id = robot
            status.message = "OK" if healthy else "ODOMETRY_TIMEOUT"
            status.values = [
                KeyValue(key="odom_age_s", value=f"{age:.3f}"),
                KeyValue(
                    key="motion_cost_m",
                    value=f"{self._motion_cost[robot]:.3f}"),
                KeyValue(
                    key="home_floor",
                    value=str(self._configs[robot]["home_floor"])),
                KeyValue(
                    key="home_region",
                    value=self._configs[robot]["home_region"]),
            ]
            array = DiagnosticArray()
            array.header.stamp = now
            array.status = [status]
            self._diagnostic_publishers[robot].publish(array)

    def stop(self):
        with self._callback_lock:
            self._active = False
        for topic in self._gz_topics:
            self._gz.unsubscribe(topic)


def main():
    os.environ.setdefault("GZ_IP", "127.0.0.1")
    rclpy.init()
    node = CarterStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
