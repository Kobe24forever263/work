#!/usr/bin/env python3
"""Drive one floor-2 Carter around the outer ring to a stair receive line."""

import math
import os
import sys
import time

from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy


ROBOT = os.environ.get("CARTER_F2_ROBOT", "car_f2_2")
if ROBOT == "car_f2_1":
    ROUTE = (
        ("F2_WEST_ENTRY", -9.0, 0.0, math.pi),
        ("F2_WEST_APPROACH", -9.0, 4.5, math.pi / 2.0),
        ("STAIR_NW_F2_RECEIVE", -9.0, 5.2, math.pi / 2.0),
    )
    INITIAL = (-5.5, 0.0, math.pi)
    HANDOVER = "STAIR_NW_F2_RECEIVE"
else:
    ROUTE = (
        ("F2_EAST_ENTRY", 9.0, 0.0, 0.0),
        ("F2_EAST_APPROACH", 9.0, 4.5, math.pi / 2.0),
        ("STAIR_NE_F2_RECEIVE", 9.0, 5.2, math.pi / 2.0),
    )
    INITIAL = (5.5, 0.0, 0.0)
    HANDOVER = "STAIR_NE_F2_RECEIVE"


def pose(x, y, yaw):
    message = PoseStamped()
    message.header.frame_id = "map"
    message.pose.position.x = float(x)
    message.pose.position.y = float(y)
    message.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def main():
    rclpy.init()
    navigator = BasicNavigator(
        node_name=f"{ROBOT}_floor2_receive_route", namespace=ROBOT)

    def call(client, request, timeout=30.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(navigator, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def wait_state(client, accepted, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = call(client, GetState.Request(), timeout=2.0)
            state_id = result.current_state.id if result else None
            if state_id in accepted:
                return state_id
            time.sleep(0.2)
        return None

    for node_name in ("map_server", "controller_server", "planner_server",
                      "velocity_smoother", "collision_monitor"):
        get_client = navigator.create_client(GetState, f"{node_name}/get_state")
        change_client = navigator.create_client(ChangeState, f"{node_name}/change_state")
        if not get_client.wait_for_service(timeout_sec=30.0):
            print(f"F2 NAV FAIL missing lifecycle service: {node_name}")
            return 1
        change_client.wait_for_service(timeout_sec=30.0)
        state = None
        for _ in range(80):
            result = call(get_client, GetState.Request(), timeout=2.0)
            state = result.current_state.id if result else None
            if state in (State.PRIMARY_STATE_UNCONFIGURED,
                         State.PRIMARY_STATE_INACTIVE,
                         State.PRIMARY_STATE_ACTIVE):
                break
            time.sleep(0.2)
        if state == State.PRIMARY_STATE_UNCONFIGURED:
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_CONFIGURE
            call(change_client, request)
            state = wait_state(get_client, (State.PRIMARY_STATE_INACTIVE,
                                             State.PRIMARY_STATE_ACTIVE))
        if state == State.PRIMARY_STATE_INACTIVE:
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_ACTIVATE
            call(change_client, request)
            state = wait_state(get_client, (State.PRIMARY_STATE_ACTIVE,))
        if state != State.PRIMARY_STATE_ACTIVE:
            print(f"F2 NAV FAIL lifecycle {node_name} state={state}")
            return 1
        print(f"F2 NAV NODE ACTIVE {node_name}")

    time.sleep(3.0)
    current = pose(*INITIAL)
    for index, (name, x, y, yaw) in enumerate(ROUTE, 1):
        goal = pose(x, y, yaw)
        print(f"F2 ROUTE {index}/{len(ROUTE)} START {name} ({x:.2f}, {y:.2f})")
        path = None
        for attempt in range(1, 6):
            path = navigator.getPath(current, goal, use_start=False)
            if path is not None and path.poses:
                break
            print(f"F2 ROUTE PLAN RETRY {attempt}/5 {name}")
            time.sleep(1.5)
        if path is None or not path.poses:
            print(f"F2 ROUTE FAIL no path {name}")
            return 1
        print(f"F2 ROUTE PLAN {name} poses={len(path.poses)}")
        navigator.followPath(path)
        started = time.monotonic()
        while not navigator.isTaskComplete():
            if time.monotonic() - started > 180.0:
                navigator.cancelTask()
                print(f"F2 ROUTE FAIL timeout {name}")
                return 1
            time.sleep(0.2)
        if navigator.getResult() != TaskResult.SUCCEEDED:
            print(f"F2 ROUTE FAIL controller aborted {name}")
            return 1
        print(f"F2 ROUTE {index}/{len(ROUTE)} PASS {name}")
        current = goal

    print(f"F2 HANDOVER READY {ROBOT} -> {HANDOVER}")
    navigator.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
