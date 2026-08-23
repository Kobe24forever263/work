"""Keep heterogeneous SCAN-Planner agents visible in one persistent RViz.

Each SCAN instance is intentionally short-lived.  This bridge caches its last
odometry sample and keeps publishing RobotState, TF and joint states after the
planner exits, so the completed handover remains visible while the next stage
starts.
"""

from copy import deepcopy
from math import cos, hypot, sin
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from warehouse_interfaces.msg import RobotState


ROBOT_CONFIGS = {
    "car_f1_1": {"type": "differential_car", "floor": 1, "z_offset": -0.15},
    "car_f1_2": {"type": "differential_car", "floor": 1, "z_offset": -0.15},
    "car_f1_3": {"type": "differential_car", "floor": 1, "z_offset": -0.15},
    "car_f1_4": {"type": "differential_car", "floor": 1, "z_offset": -0.15},
    "dog_1": {"type": "quadruped", "floor": 1, "z_offset": 0.0},
    "dog_2": {"type": "quadruped", "floor": 1, "z_offset": 0.0},
    "dog_3": {"type": "quadruped", "floor": 1, "z_offset": 0.0},
    "dog_4": {"type": "quadruped", "floor": 1, "z_offset": 0.0},
    "car_f2_1": {"type": "differential_car", "floor": 2, "z_offset": -0.40},
    "car_f2_2": {"type": "differential_car", "floor": 2, "z_offset": -0.40},
}
INITIAL_POSES = {
    "car_f1_1": (2.5, 2.5, 0.45, 0.0),
    "car_f1_2": (-2.5, 2.5, 0.45, 3.14),
    "car_f1_3": (-2.5, -2.5, 0.45, 3.14),
    "car_f1_4": (2.5, -2.5, 0.45, 0.0),
    "dog_1": (20.8, 17.8, 0.45, -2.356),
    "dog_2": (-20.8, 17.8, 0.45, -0.785),
    "dog_3": (-20.8, -17.8, 0.45, 0.785),
    "dog_4": (20.8, -17.8, 0.45, 2.356),
    "car_f2_1": (-5.5, 0.0, 4.70, 3.14),
    "car_f2_2": (5.5, 0.0, 4.70, 0.0),
}

GO2_JOINTS = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)
GO2_STANCE = (
    0.05, 0.82, -1.58, -0.05, 0.82, -1.58,
    0.05, 0.95, -1.62, -0.05, 0.95, -1.62,
)
CARTER_JOINTS = ("left_wheel", "right_wheel", "rear_pivot", "rear_axle")


class MultifloorRvizStateBridge(Node):
    def __init__(self):
        super().__init__("multifloor_rviz_state_bridge")
        selected = (
            self.declare_parameter("floor1_car", "car_f1_1").value,
            self.declare_parameter("dog", "dog_1").value,
            self.declare_parameter("floor2_car", "car_f2_2").value,
        )
        if any(name not in ROBOT_CONFIGS for name in selected):
            raise ValueError(f"Unknown RViz mission robots: {selected}")
        self._robots = {name: ROBOT_CONFIGS[name] for name in selected}
        self._dog = selected[1]
        self._tf = TransformBroadcaster(self)
        self._odom = {}
        self._received_at = {}
        self._wheel_angle = {name: 0.0 for name in self._robots}
        self._last_tick = time.monotonic()
        self._state_pubs = {
            name: self.create_publisher(RobotState, f"/{name}/robot_state", 10)
            for name in self._robots
        }
        self._joint_pubs = {
            name: self.create_publisher(
                JointState, f"/task2/{name}/joint_states", 10)
            for name in self._robots
        }
        for name in self._robots:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot=name: self._on_odom(robot, msg),
                20,
            )
            initial = Odometry()
            initial.header.frame_id = "map"
            initial.child_frame_id = name
            initial.pose.pose.position.x = INITIAL_POSES[name][0]
            initial.pose.pose.position.y = INITIAL_POSES[name][1]
            initial.pose.pose.position.z = INITIAL_POSES[name][2]
            yaw = INITIAL_POSES[name][3]
            initial.pose.pose.orientation.z = sin(yaw / 2.0)
            initial.pose.pose.orientation.w = cos(yaw / 2.0)
            self._odom[name] = initial
            self._received_at[name] = 0.0
        self.create_timer(1.0 / 30.0, self._publish)
        self.get_logger().info(
            f"Persistent RViz bridge ready for {', '.join(selected)}")

    def _on_odom(self, robot, message):
        self._odom[robot] = message
        self._received_at[robot] = time.monotonic()

    def _publish(self):
        now = self.get_clock().now().to_msg()
        tick = time.monotonic()
        dt = min(0.1, max(0.0, tick - self._last_tick))
        self._last_tick = tick
        for robot, odom in self._odom.items():
            config = self._robots[robot]
            pose = odom.pose.pose
            display_z = pose.position.z + config["z_offset"]
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = "map"
            transform.child_frame_id = (
                f"{robot}/base" if robot == self._dog
                else f"{robot}/chassis_link"
            )
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = display_z
            transform.transform.rotation = pose.orientation
            self._tf.sendTransform(transform)

            state = RobotState()
            state.stamp = now
            state.robot_id = robot
            state.robot_type = config["type"]
            state.home_floor = config["floor"]
            state.current_floor = 2 if pose.position.z > 3.0 else config["floor"]
            state.home_region = "F2_EAST" if config["floor"] == 2 else "F1_NE"
            state.current_region = state.home_region
            state.pose = deepcopy(pose)
            state.pose.position.z = display_z
            state.velocity = odom.twist.twist
            speed = hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
            fresh = tick - self._received_at.get(robot, 0.0) < 0.35
            state.navigation_state = "MOVING" if fresh and speed > 0.02 else "ARRIVED"
            state.available = True
            state.last_progress_time = now
            self._state_pubs[robot].publish(state)

            if robot == self._dog:
                self._publish_go2_joints(now, speed if fresh else 0.0, tick)
            else:
                self._publish_carter_joints(robot, now, speed if fresh else 0.0, dt)

    def _publish_go2_joints(self, stamp, speed, tick):
        message = JointState()
        message.header.stamp = stamp
        message.name = list(GO2_JOINTS)
        if speed < 0.03:
            message.position = list(GO2_STANCE)
        else:
            phase = tick * 5.0
            values = []
            for leg_index in range(4):
                swing = sin(phase + (3.14159 if leg_index in (1, 2) else 0.0))
                base = GO2_STANCE[leg_index * 3:leg_index * 3 + 3]
                values.extend((base[0], base[1] + 0.10 * swing,
                               base[2] - 0.13 * swing))
            message.position = values
        self._joint_pubs[self._dog].publish(message)

    def _publish_carter_joints(self, robot, stamp, speed, dt):
        self._wheel_angle[robot] += speed * dt / 0.24
        message = JointState()
        message.header.stamp = stamp
        message.name = list(CARTER_JOINTS)
        angle = self._wheel_angle[robot]
        message.position = [angle, angle, 0.0, angle]
        self._joint_pubs[robot].publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = MultifloorRvizStateBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
