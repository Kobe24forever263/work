"""Event-driven persistent-state dispatch environment for Stage 11."""

from dataclasses import dataclass, field
from math import dist
from typing import Callable, Iterable

from .cargo_task import CargoTaskLifecycle
from .domain import Cargo, CargoPhase, Point, Robot, Task
from .handover import Alignment, HandoverTimeoutError
from .handover_timing import CAR_TO_DOG, DOG_TO_CAR


@dataclass
class RobotRuntime:
    robot: Robot
    xyz: tuple[float, float, float]
    yaw: float = 0.0
    distance_total: float = 0.0
    tasks_completed: int = 0
    stairs_traversed: int = 0
    last_finish_time: float = 0.0
    idle_since: float = 0.0
    previous_result: str = ""
    previous_finish_xyz: tuple[float, float, float] | None = None
    standby_slot_id: str = ""

    def move_to(self, target: tuple[float, float, float]) -> float:
        travelled = dist(self.xyz, target)
        self.xyz = target
        self.distance_total += travelled
        return travelled


@dataclass(frozen=True)
class QueuedTask:
    task: Task
    arrival_time: float


@dataclass(frozen=True)
class StandbySlot:
    """A discrete, non-overlapping idle pose reserved by at most one robot."""

    slot_id: str
    robot_type: str
    floor: int
    xyz: tuple[float, float, float]


class AsyncTaskQueue:
    """Full lossless queue with a deterministic top-K policy view."""

    def __init__(self, arrivals: Iterable[QueuedTask] = ()):
        self.pending_arrivals = sorted(arrivals, key=lambda item: (
            item.arrival_time, item.task.task_id))
        self.waiting: list[QueuedTask] = []

    def advance(self, now: float) -> int:
        ready = [item for item in self.pending_arrivals
                 if item.arrival_time <= now]
        self.pending_arrivals = [item for item in self.pending_arrivals
                                 if item.arrival_time > now]
        self.waiting.extend(ready)
        return len(ready)

    def policy_view(self, now: float, limit: int = 8) -> list[QueuedTask]:
        return sorted(self.waiting, key=lambda item: (
            -item.task.priority, item.task.deadline,
            -(now - item.arrival_time), item.task.task_id))[:limit]

    def remove(self, task_id: str) -> QueuedTask:
        for index, item in enumerate(self.waiting):
            if item.task.task_id == task_id:
                return self.waiting.pop(index)
        raise KeyError(task_id)


@dataclass(frozen=True)
class Assignment:
    task_id: str
    pickup_carter: str
    dog_id: str
    stair_id: str
    receiving_carter: str
    pickup_handover_point: str
    receive_handover_point: str
    post_task_policy: str
    estimated_cost: float
    transport_mode: str = ""


@dataclass(frozen=True)
class RewardConfig:
    """Conservative Stage 13 reward and time-aware discount settings."""

    outstanding_time_scale: float = 100.0
    completion_bonus: float = 2.0
    on_time_bonus: float = 0.5
    timeout_penalty: float = 5.0
    handover_penalty: float = 0.05
    active_robot_time_penalty: float = 0.001
    distance_penalty: float = 0.0
    severe_failure_penalty: float = 10.0
    gamma0: float = 0.99
    discount_tau_s: float = 10.0


@dataclass
class DispatchTransition:
    state: dict
    action: dict
    action_mask: list[bool]
    reward: float
    delta_time: float
    next_state: dict
    terminated: bool
    truncated: bool
    reward_components: dict[str, float] = field(default_factory=dict)
    discount: float = 1.0
    termination_reason: str = ""


@dataclass(frozen=True)
class AssignmentExecution:
    travelled: float
    duration: float
    success: bool = True
    failure_reason: str = ""
    handover_durations: tuple[float, ...] = ()


class PersistentDispatchEnvironment:
    SINGLE_CAR = "SINGLE_CAR"
    SINGLE_DOG = "SINGLE_DOG"
    CAR_DOG_CAR = "CAR_DOG_CAR"
    TRANSPORT_MODES = (SINGLE_CAR, SINGLE_DOG, CAR_DOG_CAR)
    SEVERE_FAILURE_REASONS = frozenset({
        "CARGO_UNRECOVERABLE", "NO_FEASIBLE_CHAIN", "GLOBAL_DEADLOCK"})
    DEFAULT_SPEEDS = {"car": 2.0, "dog": 0.8}
    ALIGNMENT = Alignment(
        True, True, .5, .35, .75, 0, .2, 0, 0, .03, .05, 2, 1.5)

    def __init__(self, robots: Iterable[RobotRuntime], stairs: dict,
                 queue: AsyncTaskQueue, task_limit: int = 50,
                 time_limit: float = 1800.0, decision_gap: float = 3.0,
                 standby_slots: Iterable[StandbySlot] = (),
                 robot_speeds: dict[str, float] | None = None,
                 reward_config: RewardConfig | None = None,
                 handover_timeout_s: float | None = None,
                 handover_duration_provider: Callable[[str], float] | None = None,
                 handover_nominal_s: float = 2.0,
                 handover_timing_source: str = "fixed_nominal_legacy"):
        self.robots = {item.robot.robot_id: item for item in robots}
        self.stairs = stairs
        self.queue = queue
        self.task_limit = task_limit
        self.time_limit = time_limit
        self.decision_gap = decision_gap
        self.standby_slots = {slot.slot_id: slot for slot in standby_slots}
        self.robot_speeds = dict(self.DEFAULT_SPEEDS)
        if robot_speeds:
            self.robot_speeds.update(robot_speeds)
        if any(speed <= 0 for speed in self.robot_speeds.values()):
            raise ValueError("robot speeds must be positive")
        self.reward_config = reward_config or RewardConfig()
        if (self.reward_config.outstanding_time_scale <= 0 or
                self.reward_config.discount_tau_s <= 0 or
                not 0 < self.reward_config.gamma0 <= 1):
            raise ValueError("invalid reward or discount configuration")
        if handover_timeout_s is not None and handover_timeout_s <= 0:
            raise ValueError("handover timeout must be positive")
        if handover_nominal_s <= 0:
            raise ValueError("nominal handover duration must be positive")
        self.handover_timeout_s = handover_timeout_s
        self.handover_duration_provider = handover_duration_provider
        self.handover_nominal_s = handover_nominal_s
        self.handover_timing_source = handover_timing_source
        self.execution_mode = "SERIAL"
        self.now = 0.0
        self.completed = 0
        self.failed = 0
        self.transitions: list[DispatchTransition] = []
        self.resource_events: list[dict] = []
        self.handover_events: list[dict] = []
        self.resource_claims: dict[str, str] = {}
        self.queue.advance(self.now)

    def get_mdp_state(self, limit: int = 8) -> dict:
        visible = self.queue.policy_view(self.now, limit)
        robot_ids = sorted(self.robots)
        active_tasks = getattr(self, "active_tasks", {})
        return {
            "execution_mode": self.execution_mode,
            "time": round(self.now, 3),
            "waiting_count": len(self.queue.waiting),
            "active_task_count": len(active_tasks),
            "outstanding_count": len(self.queue.waiting) + len(active_tasks),
            "completed_count": self.completed,
            "failed_count": self.failed,
            "task_ids": [item.task.task_id for item in visible] +
                        [""] * (limit - len(visible)),
            "task_mask": [True] * len(visible) +
                         [False] * (limit - len(visible)),
            "active_tasks": [{
                "task_id": item.task.task_id,
                "transport_mode": item.assignment.transport_mode,
                "started_at": item.started_at,
                "finish_at": item.finish_at,
                "remaining_time": max(0.0, item.finish_at - self.now),
                "participant_ids": item.participant_ids,
            } for _, item in sorted(active_tasks.items())],
            "robots": [{
                "robot_id": robot_id,
                "type": self.robots[robot_id].robot.robot_type,
                "nominal_speed_mps": self.robot_speeds[
                    self.robots[robot_id].robot.robot_type],
                "floor": self.robots[robot_id].robot.current_floor,
                "xyz": self.robots[robot_id].xyz,
                "available": self.robots[robot_id].robot.available,
                "failure_code": self.robots[robot_id].robot.failure_code,
                "cargo_id": self.robots[robot_id].robot.cargo_id,
                "task_id": self.robots[robot_id].robot.task_id,
                "distance_total": round(self.robots[robot_id].distance_total, 4),
                "tasks_completed": self.robots[robot_id].tasks_completed,
                "idle_duration": round(max(0.0, self.now -
                                            self.robots[robot_id].idle_since), 3),
                "previous_result": self.robots[robot_id].previous_result,
                "previous_finish_xyz": self.robots[robot_id].previous_finish_xyz,
                "standby_slot_id": self.robots[robot_id].standby_slot_id,
            } for robot_id in robot_ids],
            "stairs": [{
                "stair_id": stair_id,
                "state": config.get("state", "FREE"),
                "floor1_xyz": tuple(config["floor1_xyz"]),
                "floor2_xyz": tuple(config["floor2_xyz"]),
            } for stair_id, config in sorted(self.stairs.items())],
        }

    def _available(self, kind: str, floor: int) -> list[RobotRuntime]:
        return [runtime for runtime in self.robots.values()
                if runtime.robot.robot_type == kind
                and runtime.robot.current_floor == floor
                and runtime.robot.available and not runtime.robot.failure_code
                and not runtime.robot.task_id and not runtime.robot.cargo_id]

    def enumerate_candidate_actions(self) -> list[Assignment]:
        candidates: list[Assignment] = []
        car_speed = self.robot_speeds["car"]
        dog_speed = self.robot_speeds["dog"]
        for queued in self.queue.policy_view(self.now):
            task = queued.task
            source_cars = self._available("car", task.source.floor)
            target_cars = self._available("car", task.target.floor)
            dogs = self._available("dog", task.source.floor)
            if task.source.floor == task.target.floor:
                for car in source_cars:
                    cost = (dist(car.xyz, task.source.xyz) +
                            dist(task.source.xyz, task.target.xyz)) / car_speed
                    cost += .5 * car.tasks_completed
                    candidates.append(Assignment(
                        task.task_id, car.robot.robot_id, "", "",
                        car.robot.robot_id, "", "", "NEAREST_STANDBY",
                        cost, self.SINGLE_CAR))
                for dog in dogs:
                    cost = (dist(dog.xyz, task.source.xyz) +
                            dist(task.source.xyz, task.target.xyz)) / dog_speed
                    cost += .5 * dog.tasks_completed
                    candidates.append(Assignment(
                        task.task_id, "", dog.robot.robot_id, "", "", "", "",
                        "NEAREST_STANDBY", cost, self.SINGLE_DOG))
                continue
            for dog in dogs:
                for stair_id, stair in sorted(self.stairs.items()):
                    if stair.get("state", "FREE") != "FREE":
                        continue
                    entry = tuple(stair[f"floor{task.source.floor}_xyz"])
                    exit_point = tuple(stair[f"floor{task.target.floor}_xyz"])
                    cost = (dist(dog.xyz, task.source.xyz) +
                            dist(task.source.xyz, entry) +
                            dist(entry, exit_point) +
                            dist(exit_point, task.target.xyz)) / dog_speed
                    cost += .5 * dog.tasks_completed
                    candidates.append(Assignment(
                        task.task_id, "", dog.robot.robot_id, stair_id, "",
                        "", "", "NEAREST_STANDBY", cost, self.SINGLE_DOG))
            for source_car in source_cars:
                for dog in dogs:
                    for stair_id, stair in sorted(self.stairs.items()):
                        if stair.get("state", "FREE") != "FREE":
                            continue
                        entry = tuple(stair[f"floor{task.source.floor}_xyz"])
                        exit_point = tuple(stair[f"floor{task.target.floor}_xyz"])
                        for target_car in target_cars:
                            source_ready = max(
                                (dist(source_car.xyz, task.source.xyz) +
                                 dist(task.source.xyz, entry)) / car_speed,
                                dist(dog.xyz, entry) / dog_speed)
                            dog_at_receive = (
                                source_ready + self.handover_nominal_s +
                                dist(entry, exit_point) / dog_speed)
                            cost = (max(
                                dog_at_receive,
                                dist(target_car.xyz, exit_point) / car_speed) +
                                self.handover_nominal_s +
                                dist(exit_point, task.target.xyz) / car_speed +
                                .5 * (source_car.tasks_completed +
                                      dog.tasks_completed +
                                      target_car.tasks_completed))
                            candidates.append(Assignment(
                                task.task_id, source_car.robot.robot_id,
                                dog.robot.robot_id, stair_id,
                                target_car.robot.robot_id,
                                f"{stair_id}_F{task.source.floor}_SEND",
                                f"{stair_id}_F{task.target.floor}_RECEIVE",
                                "NEAREST_STANDBY", cost,
                                self.CAR_DOG_CAR))
        return sorted(candidates, key=lambda item: (
            item.estimated_cost, item.task_id, item.pickup_carter,
            item.dog_id, item.stair_id, item.receiving_carter,
            item.transport_mode))

    def is_assignment_legal(self, assignment: Assignment) -> bool:
        try:
            task = self._task(assignment.task_id).task
        except (KeyError, StopIteration):
            return False
        def ready(runtime: RobotRuntime) -> bool:
            robot = runtime.robot
            return bool(robot.available and not robot.failure_code and
                        not robot.task_id and not robot.cargo_id)

        mode = assignment.transport_mode or (
            self.CAR_DOG_CAR if assignment.dog_id else
            self.SINGLE_CAR)
        same_floor = task.source.floor == task.target.floor
        pickup = self.robots.get(assignment.pickup_carter)
        receiver = self.robots.get(assignment.receiving_carter)
        dog = self.robots.get(assignment.dog_id)
        stair_free = bool(
            assignment.stair_id in self.stairs and
            self.stairs[assignment.stair_id].get("state", "FREE") == "FREE")

        resources_free = self._assignment_resources_available(assignment, task)
        if mode == self.SINGLE_CAR:
            return bool(resources_free and same_floor and pickup and pickup is receiver and
                        pickup.robot.robot_type == "car" and
                        pickup.robot.current_floor == task.source.floor and
                        ready(pickup) and not assignment.dog_id and
                        not assignment.stair_id)
        if mode == self.SINGLE_DOG:
            return bool(resources_free and dog and dog.robot.robot_type == "dog" and
                        dog.robot.current_floor == task.source.floor and
                        ready(dog) and not assignment.pickup_carter and
                        not assignment.receiving_carter and
                        ((same_floor and not assignment.stair_id) or
                         (not same_floor and stair_free)))
        if mode == self.CAR_DOG_CAR:
            return bool(resources_free and not same_floor and pickup and receiver and dog and
                        pickup.robot.robot_type == "car" and
                        receiver.robot.robot_type == "car" and
                        dog.robot.robot_type == "dog" and
                        pickup.robot.current_floor == task.source.floor and
                        receiver.robot.current_floor == task.target.floor and
                        dog.robot.current_floor == task.source.floor and
                        ready(pickup) and ready(receiver) and ready(dog) and
                        stair_free)
        return False

    def build_action_mask(self, candidates: list[Assignment]) -> list[bool]:
        return [self.is_assignment_legal(candidate) for candidate in candidates]

    def select_rule_action(self, candidates: list[Assignment]) -> int:
        """Pick the cheapest action for the highest-ranked executable task."""
        for queued in self.queue.policy_view(self.now):
            choices = [(index, item) for index, item in enumerate(candidates)
                       if item.task_id == queued.task.task_id and
                       self.is_assignment_legal(item)]
            if choices:
                return min(choices, key=lambda pair: (
                    pair[1].estimated_cost, pair[1].pickup_carter,
                    pair[1].dog_id, pair[1].stair_id,
                    pair[1].receiving_carter))[0]
        raise RuntimeError("no executable task in the policy view")

    @staticmethod
    def decode_action(index: int, candidates: list[Assignment]) -> Assignment:
        if index < 0 or index >= len(candidates):
            raise ValueError("action index is outside the candidate set")
        return candidates[index]

    def _task(self, task_id: str) -> QueuedTask:
        return next(item for item in self.queue.waiting
                    if item.task.task_id == task_id)

    def _assignment_resources(self, assignment: Assignment,
                              task: Task) -> list[str]:
        resources = [f"region:{task.source.region}",
                     f"region:{task.target.region}"]
        if assignment.pickup_handover_point:
            resources.append(f"handover:{assignment.pickup_handover_point}")
        if (assignment.receive_handover_point and
                assignment.receive_handover_point != assignment.pickup_handover_point):
            resources.append(f"handover:{assignment.receive_handover_point}")
        if assignment.stair_id:
            resources.append(f"stair:{assignment.stair_id}")
        return list(dict.fromkeys(resources))

    def _assignment_resources_available(self, assignment: Assignment,
                                        task: Task) -> bool:
        return not any(resource in self.resource_claims
                       for resource in self._assignment_resources(
                           assignment, task))

    def _claim_resources(self, assignment: Assignment, task: Task) -> list[str]:
        resources = self._assignment_resources(assignment, task)
        blocked = [resource for resource in resources
                   if resource in self.resource_claims]
        if blocked:
            raise RuntimeError(f"resources already occupied: {blocked}")
        for resource in resources:
            self.resource_claims[resource] = task.task_id
            if resource.startswith("stair:"):
                self.stairs[resource.split(":", 1)[1]]["state"] = "OCCUPIED"
            self.resource_events.append({"event": "CLAIM", "resource": resource,
                                         "task_id": task.task_id})
        return resources

    def _release_resources(self, resources: Iterable[str], task_id: str) -> None:
        for resource in reversed(list(resources)):
            if self.resource_claims.get(resource) != task_id:
                raise RuntimeError(f"resource owner mismatch: {resource}")
            del self.resource_claims[resource]
            if resource.startswith("stair:"):
                self.stairs[resource.split(":", 1)[1]]["state"] = "FREE"
            self.resource_events.append({"event": "RELEASE", "resource": resource,
                                         "task_id": task_id})

    def _force_release_all_resources(self, reason: str) -> None:
        """Release every transient claim when an episode cannot continue."""
        for resource, task_id in reversed(list(self.resource_claims.items())):
            del self.resource_claims[resource]
            if resource.startswith("stair:"):
                stair_id = resource.split(":", 1)[1]
                if stair_id in self.stairs:
                    self.stairs[stair_id]["state"] = "FREE"
            self.resource_events.append({
                "event": "RELEASE", "resource": resource,
                "task_id": task_id, "reason": reason})

    def abort_episode(self, reason: str, task_id: str = "",
                      elapsed: float = 1.0) -> DispatchTransition:
        """Create an auditable terminal transition for an unrecoverable fault.

        Severe failures end the current training episode, but still perform the
        task-reset cleanup invariant: no cargo holder, robot task, stair claim,
        handover claim, or shared-region claim may leak into the next reset.
        Robot positions, floors, counters, and diagnostic failure codes remain
        visible in the terminal observation.
        """
        if reason not in self.SEVERE_FAILURE_REASONS:
            raise ValueError(f"unsupported severe failure reason: {reason}")
        if elapsed < 0:
            raise ValueError("failure elapsed time must be non-negative")

        state = self.get_mdp_state()
        if not task_id and self.queue.waiting:
            task_id = self.queue.policy_view(self.now, 1)[0].task.task_id
        waiting_ids = {item.task.task_id for item in self.queue.waiting}
        if task_id and task_id not in waiting_ids:
            raise ValueError(f"failed task is not waiting: {task_id}")

        start = self.now
        end = start + elapsed
        waiting_cost = self._outstanding_time_cost(
            start, end, len(self.queue.waiting))
        failed_task = None
        if task_id:
            failed_task = self.queue.remove(task_id)
        self.failed += 1
        self.now = end
        self.queue.advance(self.now)

        self._force_release_all_resources(reason)
        failed_cargo_id = failed_task.task.cargo_id if failed_task else ""
        for runtime in self.robots.values():
            involved = bool(
                runtime.robot.task_id or runtime.robot.cargo_id or
                (task_id and runtime.robot.task_id == task_id) or
                (failed_cargo_id and runtime.robot.cargo_id == failed_cargo_id))
            runtime.robot.task_id = ""
            runtime.robot.cargo_id = ""
            if involved:
                runtime.last_finish_time = self.now
                runtime.idle_since = self.now
                runtime.previous_result = reason
                runtime.previous_finish_xyz = runtime.xyz

        config = self.reward_config
        components = {
            "outstanding_time": -waiting_cost /
                                config.outstanding_time_scale,
            "completion": 0.0,
            "deadline": 0.0,
            "handover": 0.0,
            "active_robot_time": 0.0,
            "distance": 0.0,
            "failure": -config.severe_failure_penalty,
        }
        transition = DispatchTransition(
            state,
            {"type": "ABORT", "task_id": task_id, "reason": reason},
            [], sum(components.values()), elapsed, self.get_mdp_state(),
            True, False, components,
            config.gamma0 ** (elapsed / config.discount_tau_s), reason)
        self.transitions.append(transition)
        return transition

    def wait_for_next_arrival(self) -> DispatchTransition:
        """Advance an empty decision epoch to the next asynchronous arrival."""
        if self.queue.waiting:
            raise RuntimeError("WAIT is illegal while a task is waiting")
        if not self.queue.pending_arrivals:
            raise RuntimeError("WAIT has no future arrival")

        state = self.get_mdp_state()
        start = self.now
        arrival_time = self.queue.pending_arrivals[0].arrival_time
        end = min(arrival_time, self.time_limit)
        if end < start:
            raise RuntimeError("next arrival precedes current environment time")
        self.now = end
        self.queue.advance(self.now)
        elapsed = self.now - start
        truncated = self.now >= self.time_limit and not self.queue.waiting
        config = self.reward_config
        components = {
            "outstanding_time": 0.0,
            "completion": 0.0,
            "deadline": 0.0,
            "handover": 0.0,
            "active_robot_time": 0.0,
            "distance": 0.0,
            "failure": 0.0,
        }
        transition = DispatchTransition(
            state, {"type": "WAIT"}, [True], 0.0, elapsed,
            self.get_mdp_state(), False, truncated, components,
            config.gamma0 ** (elapsed / config.discount_tau_s),
            "TIME_LIMIT" if truncated else "")
        self.transitions.append(transition)
        return transition

    def execute_joint_assignment(self, assignment: Assignment) -> AssignmentExecution:
        queued = self._task(assignment.task_id)
        task = queued.task
        claimed = self._claim_resources(assignment, task)
        try:
            return self._execute_claimed_assignment(assignment, task)
        finally:
            self._release_resources(claimed, task.task_id)

    def _execute_claimed_assignment(self, assignment: Assignment,
                                    task: Task) -> AssignmentExecution:
        mode = assignment.transport_mode or (
            self.CAR_DOG_CAR if assignment.pickup_carter and
            assignment.dog_id else
            self.SINGLE_DOG if assignment.dog_id else self.SINGLE_CAR)
        cargo = Cargo(task.cargo_id, task.task_id)
        travelled = 0.0

        if mode == self.SINGLE_CAR:
            carrier = self.robots[assignment.pickup_carter]
            carrier.standby_slot_id = ""
            carrier.robot.task_id = task.task_id
            to_source = carrier.move_to(task.source.xyz)
            to_target = carrier.move_to(task.target.xyz)
            travelled = to_source + to_target
            delivery_duration = travelled / self.robot_speeds["car"]
            carriers = [carrier]
            lifecycle = CargoTaskLifecycle(
                cargo, tuple(runtime.robot for runtime in carriers))
            lifecycle.pickup(carrier.robot,
                             f"constraint:{carrier.robot.robot_id}")
            lifecycle.deliver(carrier.robot)

        elif mode == self.SINGLE_DOG:
            dog = self.robots[assignment.dog_id]
            dog.standby_slot_id = ""
            dog.robot.task_id = task.task_id
            to_source = dog.move_to(task.source.xyz)
            travelled = to_source
            if task.source.floor != task.target.floor:
                stair = self.stairs[assignment.stair_id]
                entry = tuple(stair[f"floor{task.source.floor}_xyz"])
                exit_point = tuple(stair[f"floor{task.target.floor}_xyz"])
                to_entry = dog.move_to(entry)
                stair_travel = dog.move_to(exit_point)
                travelled += to_entry + stair_travel
                dog.robot.current_floor = task.target.floor
                dog.robot.current_region = task.target.region
                dog.stairs_traversed += 1
            to_target = dog.move_to(task.target.xyz)
            travelled += to_target
            delivery_duration = travelled / self.robot_speeds["dog"]
            carriers = [dog]
            lifecycle = CargoTaskLifecycle(
                cargo, tuple(runtime.robot for runtime in carriers))
            lifecycle.pickup(dog.robot, f"constraint:{dog.robot.robot_id}")
            lifecycle.deliver(dog.robot)

        elif mode == self.CAR_DOG_CAR:
            pickup = self.robots[assignment.pickup_carter]
            dog = self.robots[assignment.dog_id]
            receiver = self.robots[assignment.receiving_carter]
            carriers = [pickup, dog, receiver]
            for runtime in carriers:
                runtime.standby_slot_id = ""
                runtime.robot.task_id = task.task_id
            stair = self.stairs[assignment.stair_id]
            entry = tuple(stair[f"floor{task.source.floor}_xyz"])
            exit_point = tuple(stair[f"floor{task.target.floor}_xyz"])
            pickup_to_source = pickup.move_to(task.source.xyz)
            pickup_to_entry = pickup.move_to(entry)
            dog_to_entry = dog.move_to(entry)
            receiver_to_exit = receiver.move_to(exit_point)
            travelled = (pickup_to_source + pickup_to_entry + dog_to_entry +
                         receiver_to_exit)
            source_ready = max(
                (pickup_to_source + pickup_to_entry) /
                self.robot_speeds["car"],
                dog_to_entry / self.robot_speeds["dog"])
            receiver_ready = receiver_to_exit / self.robot_speeds["car"]
            lifecycle = CargoTaskLifecycle(
                cargo, tuple(runtime.robot for runtime in carriers))
            lifecycle.pickup(pickup.robot,
                             f"constraint:{pickup.robot.robot_id}")
            handover_durations = []
            first_duration = self._perform_handover(
                task, lifecycle, pickup, dog, CAR_TO_DOG,
                started_at_s=self.now + source_ready)
            handover_durations.append(first_duration)
            if first_duration > (self.handover_timeout_s
                                 if self.handover_timeout_s is not None
                                 else float("inf")):
                # The cargo task fails, but persistent fleet balance must not be
                # reset.  The dog clears the reserved stair without cargo and
                # reaches the opposite-floor standby pool before resources are
                # released; otherwise one early timeout can strand every dog on
                # one floor and create a false global deadlock.
                recovery_stair_travel = dog.move_to(exit_point)
                travelled += recovery_stair_travel
                dog.robot.current_floor = task.target.floor
                dog.robot.current_region = task.target.region
                dog.stairs_traversed += 1
                failure_duration = max(
                    source_ready + first_duration +
                    recovery_stair_travel / self.robot_speeds["dog"],
                    receiver_ready)
                return self._finish_failed_assignment(
                    carriers, cargo, travelled, failure_duration,
                    HandoverTimeoutError.failure_code,
                    tuple(handover_durations))

            stair_travel = dog.move_to(exit_point)
            travelled += stair_travel
            dog.robot.current_floor = task.target.floor
            dog.robot.current_region = task.target.region
            dog.stairs_traversed += 1
            arrival_at_second = max(
                source_ready + first_duration +
                stair_travel / self.robot_speeds["dog"],
                receiver_ready)
            second_duration = self._perform_handover(
                task, lifecycle, dog, receiver, DOG_TO_CAR,
                started_at_s=self.now + arrival_at_second)
            handover_durations.append(second_duration)
            if second_duration > (self.handover_timeout_s
                                  if self.handover_timeout_s is not None
                                  else float("inf")):
                failure_duration = arrival_at_second + second_duration
                return self._finish_failed_assignment(
                    carriers, cargo, travelled, failure_duration,
                    HandoverTimeoutError.failure_code,
                    tuple(handover_durations))

            exit_to_target = receiver.move_to(task.target.xyz)
            travelled += exit_to_target
            lifecycle.deliver(receiver.robot)
            delivery_duration = (
                arrival_at_second + second_duration +
                exit_to_target / self.robot_speeds["car"])
        else:
            raise ValueError(f"unsupported transport mode: {mode}")

        delivery_finish = {runtime.robot.robot_id: runtime.xyz
                           for runtime in carriers}
        reposition_distance, reposition_duration = self._move_to_standby(carriers)
        travelled += reposition_distance

        for runtime in carriers:
            runtime.robot.task_id = ""
            runtime.tasks_completed += 1
            runtime.previous_result = "COMPLETED"
            runtime.previous_finish_xyz = delivery_finish[runtime.robot.robot_id]
        # Robot distance remains additive for energy/accounting, but elapsed
        # time follows the parallel collaboration critical path.
        duration = max(1.0, delivery_duration + reposition_duration)
        for runtime in carriers:
            runtime.last_finish_time = self.now + duration
            runtime.idle_since = self.now + duration
        durations = (tuple(handover_durations)
                     if mode == self.CAR_DOG_CAR else ())
        return AssignmentExecution(travelled, duration, True, "", durations)

    def _sample_handover_duration(self, handover_kind: str,
                                  task_id: str = "") -> float:
        if self.handover_duration_provider is None:
            duration = self.handover_nominal_s
        else:
            try:
                duration = self.handover_duration_provider(
                    handover_kind, task_id)
            except TypeError:
                # Backward compatibility for one-argument test providers.
                duration = self.handover_duration_provider(handover_kind)
        if duration < 0:
            raise ValueError("handover duration provider returned a negative value")
        return float(duration)

    def _perform_handover(self, task: Task, lifecycle: CargoTaskLifecycle,
                          sender: RobotRuntime, receiver: RobotRuntime,
                          handover_kind: str, *, started_at_s: float) -> float:
        duration = self._sample_handover_duration(
            handover_kind, task.task_id)
        timed_out = bool(
            self.handover_timeout_s is not None and
            duration > self.handover_timeout_s)
        event = {
            "task_id": task.task_id,
            "cargo_id": task.cargo_id,
            "handover_kind": handover_kind,
            "sender_id": sender.robot.robot_id,
            "receiver_id": receiver.robot.robot_id,
            "started_at_s": started_at_s,
            "finished_at_s": started_at_s + duration,
            "duration_s": duration,
            "timeout_s": self.handover_timeout_s,
            "comparison": "STRICT_GREATER_THAN",
            "timed_out": timed_out,
            "result": (HandoverTimeoutError.failure_code
                       if timed_out else "COMPLETED"),
            "timing_source": self.handover_timing_source,
        }
        self.handover_events.append(event)
        try:
            lifecycle.handover(
                sender.robot, receiver.robot, self.ALIGNMENT,
                f"constraint:{receiver.robot.robot_id}",
                elapsed_s=duration, timeout_s=self.handover_timeout_s)
        except HandoverTimeoutError:
            # The transaction already restored unique ownership to sender.
            return duration
        return duration

    def _finish_failed_assignment(
            self, carriers: list[RobotRuntime], cargo: Cargo,
            travelled: float, failure_duration: float, reason: str,
            handover_durations: tuple[float, ...]) -> AssignmentExecution:
        """Fail one task, clear its cargo, and preserve persistent positions."""
        cargo.phase = CargoPhase.FAILED
        cargo.owner_id = None
        cargo.constraint_id = None
        failure_positions = {runtime.robot.robot_id: runtime.xyz
                             for runtime in carriers}
        for runtime in carriers:
            runtime.robot.cargo_id = ""
            runtime.robot.task_id = ""
        reposition_distance, reposition_duration = self._move_to_standby(carriers)
        travelled += reposition_distance
        duration = max(1.0, failure_duration + reposition_duration)
        for runtime in carriers:
            runtime.last_finish_time = self.now + duration
            runtime.idle_since = self.now + duration
            runtime.previous_result = reason
            runtime.previous_finish_xyz = failure_positions[
                runtime.robot.robot_id]
        return AssignmentExecution(
            travelled, duration, False, reason, handover_durations)

    def _outstanding_time_cost(self, start: float, end: float,
                               base_count: int) -> float:
        """Integral of outstanding task count over an interval."""
        cost = max(0, base_count) * max(0.0, end - start)
        for item in self.queue.pending_arrivals:
            if start < item.arrival_time <= end:
                cost += end - item.arrival_time
        return cost

    def _move_to_standby(self, runtimes: Iterable[RobotRuntime]) -> tuple[float, float]:
        """Move completed participants to unique nearest compatible idle slots.

        Task delivery is already complete when this runs.  Slot occupancy is
        persistent across tasks and a slot is released as soon as its robot is
        selected for another assignment.
        """
        runtimes = list(dict.fromkeys(item.robot.robot_id for item in runtimes))
        participants = [self.robots[robot_id] for robot_id in runtimes]
        occupied = {item.standby_slot_id for item in self.robots.values()
                    if item.standby_slot_id and item not in participants}
        occupied.update(getattr(self, "active_standby_claims", {}))
        travelled = 0.0
        longest = 0.0
        for runtime in participants:
            choices = [slot for slot in self.standby_slots.values()
                       if slot.robot_type == runtime.robot.robot_type
                       and slot.floor == runtime.robot.current_floor
                       and slot.slot_id not in occupied]
            if not choices:
                raise RuntimeError(
                    f"no standby slot for {runtime.robot.robot_id} on "
                    f"floor {runtime.robot.current_floor}")
            slot = min(choices, key=lambda item: (dist(runtime.xyz, item.xyz),
                                                  item.slot_id))
            distance = runtime.move_to(slot.xyz)
            travelled += distance
            speed = self.robot_speeds[runtime.robot.robot_type]
            longest = max(longest, distance / speed)
            runtime.standby_slot_id = slot.slot_id
            occupied.add(slot.slot_id)
        return travelled, longest

    def step(self, action_index: int) -> DispatchTransition:
        state = self.get_mdp_state()
        candidates = self.enumerate_candidate_actions()
        mask = self.build_action_mask(candidates)
        assignment = self.decode_action(action_index, candidates)
        if not mask[action_index]:
            raise ValueError("selected assignment is masked")
        selected_task = self._task(assignment.task_id).task
        start = self.now
        outstanding = len(self.queue.waiting)
        execution = self.execute_joint_assignment(assignment)
        travelled, duration = execution.travelled, execution.duration
        execution_end = start + duration
        waiting_cost = self._outstanding_time_cost(start, execution_end,
                                                   outstanding)
        self.queue.remove(assignment.task_id)
        if execution.success:
            self.completed += 1
        else:
            self.failed += 1
        self.now = execution_end
        self.queue.advance(self.now)
        gap_start = self.now
        gap_end = gap_start + self.decision_gap
        waiting_cost += self._outstanding_time_cost(
            gap_start, gap_end, len(self.queue.waiting))
        self.now = gap_end
        self.queue.advance(self.now)
        elapsed = self.now - start
        terminated = self.completed + self.failed >= self.task_limit
        truncated = self.now >= self.time_limit and not terminated
        mode = assignment.transport_mode
        participant_count = 3 if mode == self.CAR_DOG_CAR else 1
        handover_count = 2 if mode == self.CAR_DOG_CAR else 0
        on_time = execution_end <= selected_task.deadline
        config = self.reward_config
        components = {
            "outstanding_time": -waiting_cost /
                                config.outstanding_time_scale,
            "completion": (config.completion_bonus
                           if execution.success else 0.0),
            "deadline": ((config.on_time_bonus if on_time else
                          -config.timeout_penalty)
                         if execution.success else 0.0),
            "handover": (-len(execution.handover_durations) *
                         config.handover_penalty),
            "active_robot_time": -(participant_count * duration *
                                   config.active_robot_time_penalty),
            "distance": -travelled * config.distance_penalty,
            "failure": (0.0 if execution.success else
                        -config.timeout_penalty),
        }
        reward = sum(components.values())
        discount = config.gamma0 ** (elapsed / config.discount_tau_s)
        termination_reason = (
            "TASK_LIMIT" if terminated else
            "TIME_LIMIT" if truncated else "")
        action = dict(assignment.__dict__)
        action.update({
            "task_result": "COMPLETED" if execution.success else "FAILED",
            "failure_reason": execution.failure_reason,
            "handover_durations_s": list(execution.handover_durations),
            "handover_timeout_s": self.handover_timeout_s,
        })
        transition = DispatchTransition(
            state, action, mask, reward, elapsed,
            self.get_mdp_state(), terminated, truncated,
            components, discount, termination_reason)
        self.transitions.append(transition)
        return transition
