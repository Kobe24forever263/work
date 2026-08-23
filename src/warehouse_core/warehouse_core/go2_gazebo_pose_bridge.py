"""Mirror SCAN odometry into the four Go2 model poses in Gazebo."""

from math import atan2, cos, hypot, sin
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.double_pb2 import Double
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
from warehouse_interfaces.msg import RobotState
import yaml


class Go2GazeboPoseBridge(Node):
    ROBOTS = ("dog_1", "dog_2", "dog_3", "dog_4")
    FLOOR2_GOALS = {
        "dog_1": (14.2929, 11.2929),
        "dog_2": (-14.2929, 11.2929),
        "dog_3": (-14.2929, -11.2929),
        "dog_4": (14.2929, -11.2929),
    }
    STANCE = (
        0.05, 0.82, -1.58, -0.05, 0.82, -1.58,
        0.05, 0.95, -1.62, -0.05, 0.95, -1.62,
    )
    JOINTS = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )

    def __init__(self):
        super().__init__("go2_gazebo_pose_bridge")
        world = self.declare_parameter(
            "world_name", "two_floor_warehouse").value
        robots_config = self.declare_parameter(
            "robots_config",
            "/Users/lab4099/Desktop/Mujoco/work/src/"
            "warehouse_bringup/config/robots.yaml",
        ).value
        with open(robots_config, encoding="utf-8") as stream:
            robot_data = yaml.safe_load(stream)["robots"]
        self._configs = {name: robot_data[name] for name in self.ROBOTS}
        rate = float(self.declare_parameter("update_rate", 30.0).value)
        self._lock_floor2_arrival = bool(self.declare_parameter(
            "lock_floor2_arrival", True).value)
        self._latest_odom = {}
        self._display_state = {}
        self._arrived = set()
        self._last_rx = {robot: 0.0 for robot in self.ROBOTS}
        self._last_progress = {
            robot: self.get_clock().now().to_msg() for robot in self.ROBOTS
        }
        self._motion_cost = {robot: 0.0 for robot in self.ROBOTS}
        self._previous_xyz = {robot: None for robot in self.ROBOTS}
        self._gz = GazeboNode()
        self._service = f"/world/{world}/set_pose_vector/blocking"
        self._label_publisher = self.create_publisher(
            Marker, "/task2/go2_labels", 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._joint_publishers = {}
        self._state_publishers = {
            robot: self.create_publisher(
                RobotState, f"/{robot}/robot_state", 20)
            for robot in self.ROBOTS
        }
        self._diagnostic_publishers = {
            robot: self.create_publisher(
                DiagnosticArray, f"/{robot}/diagnostics", 10)
            for robot in self.ROBOTS
        }
        self._subscriptions = []
        for robot in self.ROBOTS:
            self._subscriptions.append(self.create_subscription(
                Odometry,
                f"/{robot}/odom",
                lambda message, name=robot: self._on_odom(name, message),
                10,
            ))
            self._subscriptions.append(self.create_subscription(
                JointState,
                f"/{robot}/gait_joint_states",
                lambda message, name=robot: self._on_joints(name, message),
                10,
            ))
            self._joint_publishers[robot] = {
                joint: self._gz.advertise(
                    f"/model/{robot}/joint/{joint}/0/cmd_pos", Double)
                for joint in self.JOINTS
            }
        self._pose_timer = self.create_timer(
            1.0 / max(rate, 1.0), self._publish_pose_batch)
        self._diagnostic_timer = self.create_timer(
            1.0, self._publish_diagnostics)
        self.get_logger().info(
            f"Mirroring four Go2 odometry streams to {self._service}")

    def _on_odom(self, robot, message):
        self._latest_odom[robot] = message
        self._last_rx[robot] = time.monotonic()
        point = message.pose.pose.position
        previous = self._previous_xyz[robot]
        if previous is not None:
            distance = (
                (point.x - previous[0]) ** 2
                + (point.y - previous[1]) ** 2
                + (point.z - previous[2]) ** 2
            ) ** 0.5
            self._motion_cost[robot] += distance
            if distance > 0.001:
                self._last_progress[robot] = message.header.stamp
        self._previous_xyz[robot] = (point.x, point.y, point.z)
        self._publish_robot_state(robot, message)

    def _publish_robot_state(self, robot, message):
        config = self._configs[robot]
        point = message.pose.pose.position
        speed = hypot(
            message.twist.twist.linear.x, message.twist.twist.linear.y)
        upstairs = point.z >= 4.35
        state = RobotState()
        state.stamp = message.header.stamp
        state.robot_id = robot
        state.robot_type = "quadruped"
        state.home_floor = int(config["home_floor"])
        state.current_floor = 2 if upstairs else 1
        state.home_region = config["home_region"]
        state.current_region = (
            ("F2_EAST" if point.x >= 0.0 else "F2_WEST")
            if upstairs else config["home_region"]
        )
        state.pose = message.pose.pose
        state.velocity = message.twist.twist
        state.navigation_state = (
            "ARRIVED" if robot in self._arrived
            else "MOVING" if speed > 0.02
            else "IDLE"
        )
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
        for robot in self.ROBOTS:
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
                    key="floor",
                    value="2" if robot in self._arrived else "1"),
                KeyValue(
                    key="pose_locked",
                    value=str(robot in self._arrived).lower()),
            ]
            array = DiagnosticArray()
            array.header.stamp = now
            array.status = [status]
            self._diagnostic_publishers[robot].publish(array)

    def _publish_pose_batch(self):
        if not self._latest_odom:
            return
        request = Pose_V()
        for robot, message in self._latest_odom.items():
            x, y, display_z, yaw, pitch = self._smoothed_pose(robot, message)
            display_q = self._quaternion(yaw, pitch)
            pose = request.pose.add()
            pose.name = robot
            pose.position.x = x
            pose.position.y = y
            pose.position.z = display_z
            pose.orientation.x = display_q[0]
            pose.orientation.y = display_q[1]
            pose.orientation.z = display_q[2]
            pose.orientation.w = display_q[3]
            self._publish_tf_and_label(
                robot, message, x, y, display_z, display_q)
        executed, response = self._gz.request(
            self._service, request, Pose_V, Boolean, 200)
        if not executed or not response.data:
            self.get_logger().warning(
                "Gazebo rejected batched Go2 pose update",
                throttle_duration_sec=2.0,
            )

    def _smoothed_pose(self, robot, message):
        """Low-pass planner samples so stair motion remains visually continuous."""
        planner_z = message.pose.pose.position.z
        blend = min(1.0, max(0.0, (planner_z - 0.45) / 0.35))
        target_z = planner_z - 0.13 * blend
        q = message.pose.pose.orientation
        target_yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        velocity = message.twist.twist.linear
        horizontal = hypot(velocity.x, velocity.y)
        slope = atan2(velocity.z, max(horizontal, 0.05))
        target_pitch = -max(-0.55, min(0.55, slope))
        target = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            target_z,
            target_yaw,
            target_pitch,
        )
        previous = self._display_state.get(robot)
        if previous is None:
            self._display_state[robot] = target
            return target
        if robot in self._arrived:
            return previous
        if (self._lock_floor2_arrival
                and self._is_at_floor2_goal(robot, message)):
            # The planner keeps making tiny terminal corrections.  Freeze the
            # complete visual body pose on first arrival so Gazebo does not
            # turn that harmless odometry noise into whole-body vibration.
            settled = (previous[0], previous[1], previous[2], previous[3], 0.0)
            self._display_state[robot] = settled
            self._arrived.add(robot)
            self.get_logger().info(f"{robot} reached floor 2; visual pose locked")
            return settled

        def angle_step(current, desired, alpha):
            delta = atan2(sin(desired - current), cos(desired - current))
            return current + alpha * delta

        smoothed = (
            previous[0] + 0.35 * (target[0] - previous[0]),
            previous[1] + 0.35 * (target[1] - previous[1]),
            previous[2] + 0.10 * (target[2] - previous[2]),
            angle_step(previous[3], target[3], 0.25),
            angle_step(previous[4], target[4], 0.08),
        )
        self._display_state[robot] = smoothed
        return smoothed

    @staticmethod
    def _quaternion(yaw, pitch):
        half_yaw, half_pitch = yaw * 0.5, pitch * 0.5
        return (
            -sin(half_pitch) * sin(half_yaw),
            sin(half_pitch) * cos(half_yaw),
            cos(half_pitch) * sin(half_yaw),
            cos(half_pitch) * cos(half_yaw),
        )

    def _publish_tf_and_label(
            self, robot, message, x, y, display_z, display_q):
        transform = TransformStamped()
        transform.header = message.header
        transform.header.frame_id = "world"
        transform.child_frame_id = f"{robot}/base"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = display_z
        transform.transform.rotation.x = display_q[0]
        transform.transform.rotation.y = display_q[1]
        transform.transform.rotation.z = display_q[2]
        transform.transform.rotation.w = display_q[3]
        self._tf_broadcaster.sendTransform(transform)
        label = Marker()
        label.header = message.header
        label.header.frame_id = "world"
        label.ns = "go2_numbers"
        label.id = int(robot.rsplit("_", 1)[1])
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = display_z + 0.75
        label.pose.orientation.w = 1.0
        label.scale.z = 0.55
        label.color.r = 1.0
        label.color.g = 0.85
        label.color.b = 0.05
        label.color.a = 1.0
        label.text = f"Go2-{label.id}"
        self._label_publisher.publish(label)

    def _on_joints(self, robot, message):
        positions = dict(zip(message.name, message.position))
        at_goal = robot in self._arrived
        for index, (joint, publisher) in enumerate(
                self._joint_publishers[robot].items()):
            if joint not in positions:
                continue
            command = Double()
            command.data = self.STANCE[index] if at_goal else positions[joint]
            publisher.publish(command)

    def _is_at_floor2_goal(self, robot, message):
        point = message.pose.pose.position
        goal_x, goal_y = self.FLOOR2_GOALS[robot]
        return point.z >= 4.35 and hypot(
            point.x - goal_x, point.y - goal_y) <= 0.45


def main(args=None):
    rclpy.init(args=args)
    node = Go2GazeboPoseBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
