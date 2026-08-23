"""Pure logic for regions, stair leases, and guarded floor transitions."""

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from time import monotonic
from uuid import uuid4


@dataclass
class StairLease:
    reservation_id: str
    stair_id: str
    robot_id: str
    task_id: str
    expires_at: float
    occupied: bool = False


@dataclass
class StairResource:
    stair_id: str
    lease: StairLease | None = None
    queue: deque = field(default_factory=deque)
    failure_code: str = ""


class StairManager:
    """Exclusive stair leases with FIFO queues and timeout recovery."""

    def __init__(self, stair_ids, robot_types, clock=monotonic):
        self.clock = clock
        self.robot_types = dict(robot_types)
        self.resources = {
            stair_id: StairResource(stair_id) for stair_id in stair_ids}

    def purge(self):
        now = self.clock()
        for resource in self.resources.values():
            if resource.lease and resource.lease.expires_at <= now:
                resource.lease = None
                self._promote(resource)

    def reserve(self, stair_id, robot_id, task_id, lease_seconds=30.0):
        if stair_id not in self.resources:
            return None, "UNKNOWN_STAIR"
        if self.robot_types.get(robot_id) != "dog":
            return None, "ROBOT_NOT_STAIR_CAPABLE"
        if lease_seconds <= 0:
            return None, "INVALID_LEASE"
        self.purge()
        resource = self.resources[stair_id]
        if resource.failure_code:
            return None, "STAIR_FAILED"
        if resource.lease is None:
            lease = StairLease(
                str(uuid4()), stair_id, robot_id, task_id,
                self.clock() + lease_seconds)
            resource.lease = lease
            return lease, "ACCEPTED"
        if resource.lease.robot_id == robot_id:
            resource.lease.expires_at = self.clock() + lease_seconds
            return resource.lease, "RENEWED"
        item = (robot_id, task_id, lease_seconds)
        if item not in resource.queue:
            resource.queue.append(item)
        return None, "QUEUED"

    def enter(self, stair_id, robot_id):
        self.purge()
        resource = self.resources[stair_id]
        if resource.lease is None or resource.lease.robot_id != robot_id:
            return False
        resource.lease.occupied = True
        return True

    def release(self, reservation_id, robot_id):
        self.purge()
        for resource in self.resources.values():
            lease = resource.lease
            if (lease and lease.reservation_id == reservation_id
                    and lease.robot_id == robot_id):
                resource.lease = None
                self._promote(resource)
                return True
        return False

    def release_task(self, task_id):
        """Force-release active and queued stair claims owned by a task."""
        released = 0
        for resource in self.resources.values():
            if resource.lease and resource.lease.task_id == task_id:
                resource.lease = None
                released += 1
            kept = deque(item for item in resource.queue if item[1] != task_id)
            released += len(resource.queue) - len(kept)
            resource.queue = kept
            if resource.lease is None:
                self._promote(resource)
        return released

    def _promote(self, resource):
        if not resource.queue:
            return
        robot_id, task_id, seconds = resource.queue.popleft()
        resource.lease = StairLease(
            str(uuid4()), resource.stair_id, robot_id, task_id,
            self.clock() + seconds)


class RegionManager:
    def __init__(self, regions):
        self.regions = regions

    def locate(self, x, y, floor):
        shared = None
        for region_id, config in self.regions.items():
            if int(config["floor"]) != int(floor):
                continue
            x0, x1, y0, y1 = config["bounds_xy"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                if region_id.endswith("SHARED"):
                    shared = region_id
                else:
                    return region_id
        return shared or ""

    def occupancy(self, robot_regions):
        result = {region_id: [] for region_id in self.regions}
        for robot_id, region_id in robot_regions.items():
            if region_id in result:
                result[region_id].append(robot_id)
        return result


class FloorTransitionTracker:
    """Require lease, traversal order, stable exit, and healthy localization."""

    def __init__(self, stable_seconds=1.5, clock=monotonic):
        self.clock = clock
        self.stable_seconds = stable_seconds
        self.floor = 1
        self.phase = "FLOOR1"
        self.stable_since = None

    def update(self, *, z, entry_distance, exit_distance, speed,
               reserved, occupied, localization_ok):
        if self.floor == 2:
            return 2
        if not localization_ok:
            self.stable_since = None
            return 1
        if self.phase == "FLOOR1" and reserved and entry_distance <= 0.9:
            self.phase = "ENTRY"
        if self.phase == "ENTRY" and occupied and z >= 0.8:
            self.phase = "TRAVERSING"
        if self.phase == "TRAVERSING" and z >= 4.35 and exit_distance <= 1.2:
            self.phase = "EXIT"
        if self.phase == "EXIT":
            if speed <= 0.05 and z >= 4.35:
                if self.stable_since is None:
                    self.stable_since = self.clock()
                elif self.clock() - self.stable_since >= self.stable_seconds:
                    self.floor = 2
                    self.phase = "FLOOR2"
            else:
                self.stable_since = None
        return self.floor


def distance_xy(point, target):
    return hypot(point.x - target[0], point.y - target[1])
