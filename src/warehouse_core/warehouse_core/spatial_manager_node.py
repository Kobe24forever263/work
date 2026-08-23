"""ROS 2 runtime wrapper for region, stair, and floor-transition management."""

from copy import deepcopy
from math import hypot
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from warehouse_interfaces.msg import RegionState, RobotState, StairState
from warehouse_interfaces.srv import ReleaseResource, ReserveResource
import yaml

from .spatial import (
    FloorTransitionTracker, RegionManager, StairManager, distance_xy)


class SpatialManagerNode(Node):
    DOGS = ("dog_1", "dog_2", "dog_3", "dog_4")

    def __init__(self):
        super().__init__("spatial_manager")
        robots_path = self.declare_parameter("robots_config", "").value
        stairs_path = self.declare_parameter("stairs_config", "").value
        scene_path = self.declare_parameter("scene_config", "").value
        with open(robots_path, encoding="utf-8") as stream:
            robots = yaml.safe_load(stream)["robots"]
        with open(stairs_path, encoding="utf-8") as stream:
            stairs = yaml.safe_load(stream)["stairs"]
        with open(scene_path, encoding="utf-8") as stream:
            regions = yaml.safe_load(stream)["regions"]

        self._robots_config = robots
        self._stairs_config = stairs
        self._dog_stair = {
            config["dog_id"]: stair_id
            for stair_id, config in stairs.items()
        }
        self._stairs = StairManager(
            stairs, {name: config["type"] for name, config in robots.items()})
        self._regions = RegionManager(regions)
        self._trackers = {
            dog: FloorTransitionTracker(stable_seconds=1.5)
            for dog in self.DOGS
        }
        self._latest = {}
        self._last_rx = {}
        self._canonical_pub = self.create_publisher(
            RobotState, "/warehouse/state/robots", 50)
        self._stair_pub = self.create_publisher(
            StairState, "/warehouse/state/stairs", 20)
        self._region_pub = self.create_publisher(
            RegionState, "/warehouse/state/regions", 20)
        self._event_pub = self.create_publisher(
            String, "/warehouse/events", 50)
        self._kept_subscriptions = []
        for robot in robots:
            self._kept_subscriptions.append(self.create_subscription(
                RobotState,
                f"/{robot}/robot_state",
                lambda message, name=robot: self._on_robot(name, message),
                20,
            ))
        self.create_service(
            ReserveResource, "/warehouse/reservations/reserve",
            self._reserve)
        self.create_service(
            ReleaseResource, "/warehouse/reservations/release",
            self._release)
        self.create_timer(0.2, self._tick)

        # The four validation routes start immediately. Acquire their four
        # distinct configured stairs before accepting traversal observations.
        for dog, stair_id in self._dog_stair.items():
            self._stairs.reserve(
                stair_id, dog, "stage4_validation", lease_seconds=300.0)
        self.get_logger().info(
            "Spatial manager ready: guarded floors, regions, and four stairs")

    def _on_robot(self, robot, message):
        self._latest[robot] = message
        self._last_rx[robot] = time.monotonic()

    def _reserve(self, request, response):
        if request.resource_type != "stair":
            response.accepted = False
            response.reason = "UNSUPPORTED_RESOURCE_TYPE"
            return response
        lease, reason = self._stairs.reserve(
            request.resource_id, request.requester_id, request.task_id,
            float(request.lease_seconds))
        response.accepted = lease is not None
        response.reservation_id = lease.reservation_id if lease else ""
        response.reason = reason
        return response

    def _release(self, request, response):
        response.released = self._stairs.release(
            request.reservation_id, request.requester_id)
        response.message = "RELEASED" if response.released else \
            "NOT_OWNER_OR_UNKNOWN_RESERVATION"
        return response

    def _tick(self):
        self._stairs.purge()
        canonical = {}
        robot_regions = {}
        now_mono = time.monotonic()
        for robot, source in self._latest.items():
            state = deepcopy(source)
            localization_ok = now_mono - self._last_rx.get(robot, 0.0) < 1.0
            if robot in self._trackers:
                stair_id = self._dog_stair[robot]
                config = self._stairs_config[stair_id]
                resource = self._stairs.resources[stair_id]
                lease = resource.lease
                reserved = lease is not None and lease.robot_id == robot
                point = source.pose.position
                entry_distance = distance_xy(point, config["floor1_xy"])
                exit_distance = distance_xy(point, config["floor2_xy"])
                if reserved and entry_distance <= 1.1 and point.z >= 0.75:
                    self._stairs.enter(stair_id, robot)
                previous_floor = self._trackers[robot].floor
                floor = self._trackers[robot].update(
                    z=point.z,
                    entry_distance=entry_distance,
                    exit_distance=exit_distance,
                    speed=hypot(
                        source.velocity.linear.x, source.velocity.linear.y),
                    reserved=reserved,
                    occupied=bool(lease and lease.occupied),
                    localization_ok=localization_ok,
                )
                if floor != previous_floor:
                    event = String()
                    event.data = (
                        f"FLOOR_CHANGED robot={robot} from={previous_floor} "
                        f"to={floor} stair={stair_id}")
                    self._event_pub.publish(event)
                    if lease:
                        self._stairs.release(lease.reservation_id, robot)
                state.current_floor = floor
            region = self._regions.locate(
                state.pose.position.x, state.pose.position.y,
                state.current_floor)
            state.current_region = region
            if not localization_ok:
                state.available = False
                state.failure_code = "LOCALIZATION_STALE"
            canonical[robot] = state
            robot_regions[robot] = region
            self._canonical_pub.publish(state)
        self._publish_stairs()
        self._publish_regions(robot_regions)

    def _publish_stairs(self):
        stamp = self.get_clock().now().to_msg()
        for stair_id, resource in self._stairs.resources.items():
            message = StairState()
            message.stamp = stamp
            message.stair_id = stair_id
            if resource.failure_code:
                message.state = "FAILED"
            elif resource.lease and resource.lease.occupied:
                message.state = "OCCUPIED"
            elif resource.lease:
                message.state = "RESERVED"
            else:
                message.state = "FREE"
            message.reserved_by = (
                resource.lease.robot_id if resource.lease else "")
            message.occupied_by = (
                resource.lease.robot_id
                if resource.lease and resource.lease.occupied else "")
            message.queue_length = len(resource.queue)
            message.failure_code = resource.failure_code
            self._stair_pub.publish(message)

    def _publish_regions(self, robot_regions):
        stamp = self.get_clock().now().to_msg()
        occupancy = self._regions.occupancy(robot_regions)
        for region_id, config in self._regions.regions.items():
            message = RegionState()
            message.stamp = stamp
            message.region_id = region_id
            message.floor = int(config["floor"])
            message.occupants = occupancy[region_id]
            message.reservations = []
            message.capacity = int(config["capacity"])
            message.congested = len(message.occupants) >= message.capacity
            self._region_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SpatialManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
