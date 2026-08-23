#!/usr/bin/env python3
"""Observe a four-Go2 run and enforce stage-4 spatial invariants."""

import sys
import time

import rclpy
from rclpy.node import Node
from warehouse_interfaces.msg import RegionState, RobotState, StairState
from warehouse_interfaces.srv import ReserveResource


DOGS = ("dog_1", "dog_2", "dog_3", "dog_4")
STAIRS = ("STAIR_NE", "STAIR_NW", "STAIR_SW", "STAIR_SE")


class Validator(Node):
    def __init__(self):
        super().__init__("stage4_runtime_validator")
        self.latest_dogs = {}
        self.latest_stairs = {}
        self.regions = set()
        self.early_switches = []
        self._kept = [
            self.create_subscription(
                RobotState, "/warehouse/state/robots",
                self._on_robot, 50),
            self.create_subscription(
                StairState, "/warehouse/state/stairs",
                lambda msg: self.latest_stairs.__setitem__(
                    msg.stair_id, msg), 20),
            self.create_subscription(
                RegionState, "/warehouse/state/regions",
                lambda msg: self.regions.add(msg.region_id), 20),
        ]
        self.reserve_client = self.create_client(
            ReserveResource, "/warehouse/reservations/reserve")

    def _on_robot(self, message):
        if message.robot_id not in DOGS:
            return
        if message.current_floor == 2 and message.pose.position.z < 4.35:
            self.early_switches.append(message.robot_id)
        self.latest_dogs[message.robot_id] = message


def main():
    rclpy.init()
    node = Validator()
    deadline = time.monotonic() + 150.0
    car_rejected = False
    request_sent = False
    future = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if not request_sent and node.reserve_client.wait_for_service(0.0):
            request = ReserveResource.Request()
            request.resource_type = "stair"
            request.resource_id = "STAIR_NE"
            request.requester_id = "car_f1_1"
            request.task_id = "illegal_runtime_probe"
            request.lease_seconds = 10.0
            future = node.reserve_client.call_async(request)
            request_sent = True
        if future is not None and future.done():
            response = future.result()
            car_rejected = (
                not response.accepted
                and response.reason == "ROBOT_NOT_STAIR_CAPABLE")
            future = None
        all_upstairs = all(
            node.latest_dogs.get(dog)
            and node.latest_dogs[dog].current_floor == 2
            for dog in DOGS)
        all_free = all(
            node.latest_stairs.get(stair)
            and node.latest_stairs[stair].state == "FREE"
            for stair in STAIRS)
        if all_upstairs and all_free and car_rejected:
            break

    failures = []
    if node.early_switches:
        failures.append("floor switched below exit elevation")
    if not car_rejected:
        failures.append("Carter stair reservation was not rejected")
    for dog in DOGS:
        state = node.latest_dogs.get(dog)
        if state is None or state.current_floor != 2:
            failures.append(f"{dog} did not complete guarded floor switch")
    for stair in STAIRS:
        state = node.latest_stairs.get(stair)
        if state is None or state.state != "FREE":
            failures.append(f"{stair} did not return to FREE")
    if len(node.regions) < 7:
        failures.append("region state coverage incomplete")

    snapshot = dict(node.latest_dogs)
    node.destroy_node()
    rclpy.shutdown()
    if failures:
        print("Stage 4 runtime validation: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Stage 4 runtime validation: PASS")
    for dog in DOGS:
        state = snapshot[dog]
        print(
            f"  {dog}: floor={state.current_floor}, "
            f"region={state.current_region}")
    print("  four stairs: FREE; Carter permission probe: REJECTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
