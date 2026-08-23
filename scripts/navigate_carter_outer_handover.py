#!/usr/bin/env python3
"""Drive car_f1_1 to the east outer ring, then along it to STAIR_NE."""

from math import cos, pi, sin
import sys
import time

from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy


ROBOT = "car_f1_1"
ROUTE = (
    ("EAST_OUTER_ENTRY", 20.30, 0.00, pi / 2),
    ("EAST_OUTER_MID", 20.30, 8.00, pi / 2),
    # The wheeled robot's safe stop line.  Beyond y=14 the stair structure and
    # dog_1 enter Carter's protected stopping area.
    ("STAIR_NE_F1_SEND", 20.30, 14.00, pi / 2),
)


def pose(x, y, yaw):
    message = PoseStamped()
    message.header.frame_id = "map"
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.orientation.z = sin(yaw / 2)
    message.pose.orientation.w = cos(yaw / 2)
    return message


def main():
    rclpy.init()
    navigator = BasicNavigator(
        node_name="car_f1_1_outer_handover_route", namespace=ROBOT)
    navigator.setInitialPose(pose(2.5, 2.5, 0.0))
    lifecycle_nodes = (
        "map_server",
        "controller_server",
        "planner_server",
        "velocity_smoother",
        "collision_monitor",
    )

    def call(client, request, timeout=30.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(navigator, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def wait_state(get_client, accepted, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = call(get_client, GetState.Request(), timeout=2.0)
            state_id = result.current_state.id if result else None
            if state_id in accepted:
                return state_id
            time.sleep(0.2)
        return None

    for node_name in lifecycle_nodes:
        get_client = navigator.create_client(
            GetState, f"{node_name}/get_state")
        change_client = navigator.create_client(
            ChangeState, f"{node_name}/change_state")
        if not get_client.wait_for_service(timeout_sec=20.0):
            print(f"NAV2 FAIL missing lifecycle service: {node_name}")
            return 1
        change_client.wait_for_service(timeout_sec=20.0)

        # Wait out a transition that an earlier autostart attempt may still
        # be completing, then apply only the transition this node needs.
        state = None
        for _ in range(60):
            result = call(get_client, GetState.Request(), timeout=2.0)
            state = result.current_state.id if result else None
            if state in (
                    State.PRIMARY_STATE_UNCONFIGURED,
                    State.PRIMARY_STATE_INACTIVE,
                    State.PRIMARY_STATE_ACTIVE):
                break
            time.sleep(0.2)
        if state == State.PRIMARY_STATE_UNCONFIGURED:
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_CONFIGURE
            result = call(change_client, request)
            state = wait_state(
                get_client,
                (State.PRIMARY_STATE_INACTIVE, State.PRIMARY_STATE_ACTIVE),
            )
            if state is None:
                print(f"NAV2 FAIL configure: {node_name}")
                return 1
        if state == State.PRIMARY_STATE_INACTIVE:
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_ACTIVATE
            result = call(change_client, request)
            state = wait_state(
                get_client, (State.PRIMARY_STATE_ACTIVE,))
            if state is None:
                print(f"NAV2 FAIL activate: {node_name}")
                return 1
        elif state != State.PRIMARY_STATE_ACTIVE:
            print(f"NAV2 FAIL unexpected state {state}: {node_name}")
            return 1
        print(f"NAV2 NODE ACTIVE {node_name}")

    print("NAV2 DIRECT PLANNER/CONTROLLER ACTIVE ground-truth localization")
    # Lifecycle state becomes ACTIVE slightly before the action callbacks are
    # ready on this machine.  Give the servers time to finish activation.
    time.sleep(3.0)

    current = pose(2.5, 2.5, 0.0)
    for index, (name, x, y, yaw) in enumerate(ROUTE, start=1):
        print(f"ROUTE {index}/{len(ROUTE)} START {name} ({x:.2f}, {y:.2f})")
        goal = pose(x, y, yaw)
        path = None
        for attempt in range(1, 6):
            # Let planner_server read the live map->base transform.  Reusing
            # the nominal previous goal can leave a discontinuity after the
            # controller accepts a goal within tolerance.
            path = navigator.getPath(current, goal, use_start=False)
            if path is not None and path.poses:
                break
            print(f"ROUTE PLAN RETRY {attempt}/5 {name}")
            time.sleep(1.5)
        if path is None or not path.poses:
            print(f"ROUTE FAIL no global path to {name}")
            navigator.destroy_node()
            rclpy.shutdown()
            return 1
        print(f"ROUTE PLAN {name} poses={len(path.poses)}")
        navigator.followPath(path)
        started = time.monotonic()
        while not navigator.isTaskComplete():
            if time.monotonic() - started > 180:
                navigator.cancelTask()
                print(f"ROUTE FAIL timeout at {name}")
                navigator.destroy_node()
                rclpy.shutdown()
                return 1
            time.sleep(0.2)
        if navigator.getResult() != TaskResult.SUCCEEDED:
            print(f"ROUTE FAIL navigation aborted at {name}")
            navigator.destroy_node()
            rclpy.shutdown()
            return 1
        print(f"ROUTE {index}/{len(ROUTE)} PASS {name}")
        current = goal

    print("HANDOVER READY car_f1_1 <-> dog_1 at STAIR_NE_F1_SEND")
    navigator.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
