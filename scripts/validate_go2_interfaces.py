#!/usr/bin/env python3
"""Runtime namespace and identity check for the four Go2 interfaces."""

import sys
import time

from diagnostic_msgs.msg import DiagnosticArray
import rclpy
from rclpy.node import Node
from warehouse_interfaces.msg import RobotState


ROBOTS = ("dog_1", "dog_2", "dog_3", "dog_4")


class InterfaceValidator(Node):
    def __init__(self):
        super().__init__("go2_interface_validator")
        self.states = {robot: [] for robot in ROBOTS}
        self.diagnostics = {robot: [] for robot in ROBOTS}
        self._kept_subscriptions = []
        for robot in ROBOTS:
            self._kept_subscriptions.append(self.create_subscription(
                RobotState,
                f"/{robot}/robot_state",
                lambda msg, name=robot: self.states[name].append(msg),
                10,
            ))
            self._kept_subscriptions.append(self.create_subscription(
                DiagnosticArray,
                f"/{robot}/diagnostics",
                lambda msg, name=robot: self.diagnostics[name].append(msg),
                10,
            ))


def main():
    rclpy.init()
    node = InterfaceValidator()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    failures = []
    for robot in ROBOTS:
        states = node.states[robot]
        diagnostics = node.diagnostics[robot]
        if not states:
            failures.append(f"{robot}: no robot_state")
        elif any(state.robot_id != robot for state in states):
            failures.append(f"{robot}: robot_id crossed namespace")
        elif any(state.robot_type != "quadruped" for state in states):
            failures.append(f"{robot}: unexpected robot_type")
        if not diagnostics:
            failures.append(f"{robot}: no diagnostics")
        elif any(
            not array.status
            or array.status[0].hardware_id != robot
            or array.status[0].name != f"{robot}/interface"
            for array in diagnostics
        ):
            failures.append(f"{robot}: diagnostic identity mismatch")

    node.destroy_node()
    rclpy.shutdown()
    if failures:
        print("Go2 interface isolation: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Go2 interface isolation: PASS")
    for robot in ROBOTS:
        latest = node.states[robot][-1]
        print(
            f"  {robot}: floor={latest.current_floor}, "
            f"state={latest.navigation_state}, "
            f"samples={len(node.states[robot])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
