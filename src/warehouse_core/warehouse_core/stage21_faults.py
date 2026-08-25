"""Reproducible logical fault injection for Stage 21 robustness tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random

from .concurrent_dispatch import ConcurrentPersistentDispatchEnvironment
from .persistent_dispatch import Assignment, AssignmentExecution
from .stage14_training import WarehouseDispatchGymEnv


CONTROL = "CONTROL"
NAVIGATION_FAILURE = "NAVIGATION_FAILURE"
STAIR_OUTAGE = "STAIR_OUTAGE"
ROBOT_OUTAGE = "ROBOT_OUTAGE"
FAULT_SCENARIOS = (CONTROL, NAVIGATION_FAILURE, STAIR_OUTAGE, ROBOT_OUTAGE)
EXTERNAL_ROBOT_CODE = "EXTERNAL_ROBOT_UNAVAILABLE"
EXTERNAL_STAIR_STATE = "EXTERNAL_BLOCKED"


@dataclass(frozen=True)
class FaultWindow:
    window_id: str
    kind: str
    target_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class FaultPlan:
    schema_version: str
    scenario: str
    seed: int
    windows: tuple[FaultWindow, ...]
    navigation_per_leg_probability: float = 0.0
    navigation_recovery_delay_s: float = 0.0
    active_task_preemption: bool = False
    randomness: str = "TASK_KEYED_PREFIX_COUPLING"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _anchor(arrivals: list[float], fraction: float) -> float:
    index = min(len(arrivals) - 1, max(0, int(fraction * len(arrivals))))
    return float(arrivals[index])


def build_fault_plan(dispatch, scenario: str, seed: int) -> FaultPlan:
    """Build an absolute-time plan from the shared task stream."""
    scenario = scenario.upper()
    if scenario not in FAULT_SCENARIOS:
        raise ValueError(f"unsupported Stage 21 fault scenario: {scenario}")
    arrivals = sorted(
        item.arrival_time for item in (
            list(dispatch.queue.waiting) +
            list(dispatch.queue.pending_arrivals)))
    if len(arrivals) < 20:
        raise ValueError("Stage 21 fault plans require at least 20 tasks")
    rng = random.Random(seed + 210021)
    windows: list[FaultWindow] = []
    nav_probability = 0.0
    recovery_delay = 0.0
    if scenario == NAVIGATION_FAILURE:
        windows = [
            FaultWindow(
                "NAV_1", scenario, "ALL_ROUTES",
                _anchor(arrivals, .18), _anchor(arrivals, .43)),
            FaultWindow(
                "NAV_2", scenario, "ALL_ROUTES",
                _anchor(arrivals, .56), _anchor(arrivals, .81)),
        ]
        nav_probability = .05
        recovery_delay = 15.0
    elif scenario == STAIR_OUTAGE:
        stairs = sorted(dispatch.stairs)
        first, second = rng.sample(stairs, 2)
        windows = [
            FaultWindow(
                "STAIR_1", scenario, first,
                _anchor(arrivals, .25), _anchor(arrivals, .40)),
            FaultWindow(
                "STAIR_2", scenario, second,
                _anchor(arrivals, .60), _anchor(arrivals, .75)),
        ]
    elif scenario == ROBOT_OUTAGE:
        cars = sorted(
            robot_id for robot_id, runtime in dispatch.robots.items()
            if runtime.robot.robot_type == "car")
        dogs = sorted(
            robot_id for robot_id, runtime in dispatch.robots.items()
            if runtime.robot.robot_type == "dog")
        windows = [
            FaultWindow(
                "ROBOT_1", scenario, rng.choice(cars),
                _anchor(arrivals, .20), _anchor(arrivals, .40)),
            FaultWindow(
                "ROBOT_2", scenario, rng.choice(dogs),
                _anchor(arrivals, .55), _anchor(arrivals, .75)),
        ]
    return FaultPlan(
        schema_version="warehouse_stage21_fault_plan_v1",
        scenario=scenario, seed=seed, windows=tuple(windows),
        navigation_per_leg_probability=nav_probability,
        navigation_recovery_delay_s=recovery_delay)


def _unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class FaultAwareConcurrentDispatchEnvironment(
        ConcurrentPersistentDispatchEnvironment):
    """Concurrent logical simulator with non-preemptive external faults."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_plan: FaultPlan | None = None
        self.fault_events: list[dict] = []
        self._active_fault_windows: set[str] = set()
        self._external_robot_ids: set[str] = set()
        self._external_stair_ids: set[str] = set()

    @property
    def fault_schedule_fingerprint(self) -> str:
        return self.fault_plan.fingerprint if self.fault_plan else ""

    def configure_fault_plan(self, plan: FaultPlan) -> None:
        self.fault_plan = plan
        self._sync_fault_state()

    def _windows_at(self, now: float) -> tuple[FaultWindow, ...]:
        if self.fault_plan is None:
            return ()
        return tuple(
            window for window in self.fault_plan.windows
            if window.start_s <= now < window.end_s)

    def _sync_fault_state(self) -> None:
        active = self._windows_at(self.now)
        active_ids = {window.window_id for window in active}
        for window in (self.fault_plan.windows if self.fault_plan else ()):
            if (window.window_id in active_ids and
                    window.window_id not in self._active_fault_windows):
                self.fault_events.append({
                    "event": "FAULT_WINDOW_START",
                    "observed_at_s": self.now,
                    **asdict(window),
                })
            elif (window.window_id not in active_ids and
                  window.window_id in self._active_fault_windows):
                self.fault_events.append({
                    "event": "FAULT_WINDOW_END",
                    "observed_at_s": self.now,
                    **asdict(window),
                })
        self._active_fault_windows = active_ids

        desired_robots = {
            window.target_id for window in active
            if window.kind == ROBOT_OUTAGE}
        for robot_id in self._external_robot_ids - desired_robots:
            runtime = self.robots[robot_id]
            if runtime.robot.failure_code == EXTERNAL_ROBOT_CODE:
                runtime.robot.failure_code = ""
            if not runtime.robot.task_id and not runtime.robot.cargo_id:
                runtime.robot.available = True
        for robot_id in desired_robots:
            runtime = self.robots[robot_id]
            runtime.robot.failure_code = EXTERNAL_ROBOT_CODE
            runtime.robot.available = False
        self._external_robot_ids = desired_robots

        desired_stairs = {
            window.target_id for window in active
            if window.kind == STAIR_OUTAGE}
        for stair_id in self._external_stair_ids - desired_stairs:
            resource = f"stair:{stair_id}"
            if (resource not in self.resource_claims and
                    self.stairs[stair_id].get("state") ==
                    EXTERNAL_STAIR_STATE):
                self.stairs[stair_id]["state"] = "FREE"
        for stair_id in desired_stairs:
            resource = f"stair:{stair_id}"
            if resource not in self.resource_claims:
                self.stairs[stair_id]["state"] = EXTERNAL_STAIR_STATE
        self._external_stair_ids = desired_stairs

    def enumerate_candidate_actions(self) -> list[Assignment]:
        self._sync_fault_state()
        return super().enumerate_candidate_actions()

    def _release_resources(self, resources, task_id: str) -> None:
        super()._release_resources(resources, task_id)
        self._sync_fault_state()

    def _finalize_active(self, active) -> None:
        super()._finalize_active(active)
        self._sync_fault_state()

    def _advance_to(self, target_time: float):
        metrics = super()._advance_to(target_time)
        self._sync_fault_state()
        return metrics

    def _navigation_failure(self, assignment: Assignment, task) -> bool:
        plan = self.fault_plan
        if (plan is None or plan.scenario != NAVIGATION_FAILURE or
                not self._windows_at(self.now)):
            return False
        if assignment.transport_mode == self.CAR_DOG_CAR:
            segment_count = 6
        elif (assignment.transport_mode == self.SINGLE_DOG and
              task.source.floor != task.target.floor):
            segment_count = 4
        else:
            segment_count = 2
        return any(
            _unit_interval(
                f"{plan.seed}|NAVIGATION|{task.task_id}|LEG|{index}") <
            plan.navigation_per_leg_probability
            for index in range(segment_count))

    def _execute_claimed_assignment(self, assignment: Assignment, task):
        execution = super()._execute_claimed_assignment(assignment, task)
        if not execution.success or not self._navigation_failure(
                assignment, task):
            return execution
        participant_ids = tuple(dict.fromkeys(
            robot_id for robot_id in (
                assignment.pickup_carter, assignment.dog_id,
                assignment.receiving_carter) if robot_id))
        delay = self.fault_plan.navigation_recovery_delay_s
        for robot_id in participant_ids:
            runtime = self.robots[robot_id]
            runtime.tasks_completed = max(0, runtime.tasks_completed - 1)
            runtime.previous_result = NAVIGATION_FAILURE
            runtime.last_finish_time += delay
            runtime.idle_since += delay
        self.fault_events.append({
            "event": "TASK_NAVIGATION_FAILURE",
            "observed_at_s": self.now,
            "task_id": task.task_id,
            "transport_mode": assignment.transport_mode,
            "participant_ids": participant_ids,
            "recovery_delay_s": delay,
        })
        return AssignmentExecution(
            travelled=execution.travelled,
            duration=execution.duration + delay,
            success=False,
            failure_reason=NAVIGATION_FAILURE,
            handover_durations=execution.handover_durations)


class FaultAwareWarehouseDispatchGymEnv(WarehouseDispatchGymEnv):
    """Gym wrapper that installs a shared Stage 21 fault plan before encoding."""

    def __init__(self, *, fault_scenario: str, fault_seed: int | None = None,
                 **kwargs):
        self.fault_scenario = fault_scenario.upper()
        self.fault_seed = fault_seed
        super().__init__(
            dispatch_environment_class=
            FaultAwareConcurrentDispatchEnvironment,
            dispatch_setup_callback=self._configure_faults,
            **kwargs)

    def _configure_faults(self, dispatch, episode_seed: int) -> None:
        seed = self.fault_seed if self.fault_seed is not None else episode_seed
        dispatch.configure_fault_plan(build_fault_plan(
            dispatch, self.fault_scenario, seed))
