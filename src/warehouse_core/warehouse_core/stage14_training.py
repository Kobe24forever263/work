"""Stage 14 balanced curriculum and masked Gymnasium training interface."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .concurrent_dispatch import ConcurrentPersistentDispatchEnvironment
from .domain import Point, Robot, Task
from .handover_timing import (
    build_provisional_policy_and_keyed_provider,
    build_provisional_policy_and_provider)
from .persistent_dispatch import (
    Assignment, AsyncTaskQueue, PersistentDispatchEnvironment, QueuedTask,
    RobotRuntime, RewardConfig, StandbySlot)
from .stage12_encoding import EncodedDecision, Stage12Encoder


F1 = {
    "NE": Point("F1_NE", 1, "F1_NE", (12, 8, .45)),
    "NW": Point("F1_NW", 1, "F1_NW", (-12, 8, .45)),
    "SW": Point("F1_SW", 1, "F1_SW", (-12, -8, .45)),
    "SE": Point("F1_SE", 1, "F1_SE", (12, -8, .45)),
}
F2 = {
    "NE": Point("F2_NE", 2, "F2_EAST", (7, 6.5, 4.7)),
    "NW": Point("F2_NW", 2, "F2_WEST", (-7, 6.5, 4.7)),
    "SW": Point("F2_SW", 2, "F2_WEST", (-7, -6.5, 4.7)),
    "SE": Point("F2_SE", 2, "F2_EAST", (7, -6.5, 4.7)),
}
STAIR_TASK_F1 = {
    "NE": Point("F1_NE_STAIR_PLATFORM", 1, "F1_NE", (20.0, 18.0, .45)),
    "NW": Point("F1_NW_STAIR_PLATFORM", 1, "F1_NW", (-20.0, 18.0, .45)),
    "SW": Point("F1_SW_STAIR_PLATFORM", 1, "F1_SW", (-20.0, -18.0, .45)),
    "SE": Point("F1_SE_STAIR_PLATFORM", 1, "F1_SE", (20.0, -18.0, .45)),
}
STAIR_TASK_F2 = {
    "NE": Point("F2_NE_STAIR_PLATFORM", 2, "F2_EAST", (9.2, 6.3, 4.7)),
    "NW": Point("F2_NW_STAIR_PLATFORM", 2, "F2_WEST", (-9.2, 6.3, 4.7)),
    "SW": Point("F2_SW_STAIR_PLATFORM", 2, "F2_WEST", (-9.2, -6.3, 4.7)),
    "SE": Point("F2_SE_STAIR_PLATFORM", 2, "F2_EAST", (9.2, -6.3, 4.7)),
}
MEDIUM_INTERVAL = (55.0, 75.0)
DENSE_INTERVAL = (15.0, 25.0)
NORMAL_INTERVAL = (80.0, 100.0)
BURST_INTERVAL = (1.0, 3.0)
STAGE14_REWARD_CONFIG = RewardConfig(
    # Reward V2: time/throughput must outweigh the risk-free all-dog shortcut
    # without adding any transport-mode-specific bonus.
    outstanding_time_scale=30.0,
    timeout_penalty=2.0,
    active_robot_time_penalty=0.0002,
)
STAGE23_REWARD_CONFIG = RewardConfig(
    outstanding_time_scale=30.0,
    timeout_penalty=2.0,
    active_robot_time_penalty=0.0002,
    continuous_time_discounting=True,
)


@dataclass(frozen=True)
class Stage14EpisodeSpec:
    """Exact task mix for the first persistent-state curriculum lesson."""

    f1_same: int = 3
    f1_cross_region: int = 2
    f2_same: int = 3
    f2_cross_region: int = 2
    cross_up: int = 5
    cross_down: int = 5

    @property
    def task_count(self) -> int:
        return sum((self.f1_same, self.f1_cross_region, self.f2_same,
                    self.f2_cross_region, self.cross_up, self.cross_down))

    @property
    def cross_floor_count(self) -> int:
        return self.cross_up + self.cross_down


def _runtime(robot_id: str, kind: str, floor: int, region: str,
             xyz: tuple[float, float, float]) -> RobotRuntime:
    return RobotRuntime(
        Robot(robot_id, kind, floor, region, floor, region), xyz)


def _task_points(task_type: str, rng: random.Random, sequence: int,
                 dog_favored: bool = False):
    keys = list(F1)
    if task_type == "f1_same":
        key = rng.choice(keys)
        source = F1[key]
        return source, Point(
            f"F1_{key}_LOCAL_{sequence}", 1, source.region,
            (source.xyz[0] * .7, source.xyz[1] * .7, .45))
    if task_type == "f2_same":
        key = rng.choice(keys)
        source = F2[key]
        return source, Point(
            f"F2_{key}_LOCAL_{sequence}", 2, source.region,
            (source.xyz[0] * .7, source.xyz[1] * .7, 4.7))
    if task_type == "f1_cross_region":
        source_key, target_key = rng.sample(keys, 2)
        return F1[source_key], F1[target_key]
    if task_type == "f2_cross_region":
        source_key, target_key = rng.sample(keys, 2)
        return F2[source_key], F2[target_key]
    if dog_favored and task_type in {"cross_up", "cross_down"}:
        stair_key = rng.choice(keys)
        if task_type == "cross_up":
            return STAIR_TASK_F1[stair_key], STAIR_TASK_F2[stair_key]
        return STAIR_TASK_F2[stair_key], STAIR_TASK_F1[stair_key]
    source_key, target_key = rng.sample(keys, 2)
    if task_type == "cross_up":
        return F1[source_key], F2[target_key]
    if task_type == "cross_down":
        return F2[source_key], F1[target_key]
    raise ValueError(f"unknown Stage 14 task type: {task_type}")


def build_balanced_arrivals(seed: int,
                            spec: Stage14EpisodeSpec | None = None,
                            arrival_interval: tuple[float, float] =
                            MEDIUM_INTERVAL):
    """Build a deterministic 50/50 same-floor/cross-floor task sequence.

    Cross-floor directions preserve an up/down order so the persistent Go2
    population never becomes stranded on one floor solely because of random
    task ordering. Same-floor task classes and their interleaving remain
    seed-dependent.
    """
    spec = spec or Stage14EpisodeSpec()
    if (len(arrival_interval) != 2 or arrival_interval[0] <= 0 or
            arrival_interval[1] < arrival_interval[0]):
        raise ValueError("arrival interval must be a positive ordered pair")
    if spec.task_count <= 0 or spec.cross_up != spec.cross_down:
        raise ValueError(
            "curriculum episodes require a positive task count and balanced "
            "cross-floor directions")
    task_rng = random.Random(seed)
    arrival_rng = random.Random(seed + 1400)
    same_types = (
        ["f1_same"] * spec.f1_same +
        ["f1_cross_region"] * spec.f1_cross_region +
        ["f2_same"] * spec.f2_same +
        ["f2_cross_region"] * spec.f2_cross_region)
    task_rng.shuffle(same_types)
    cross_types = [item for _ in range(spec.cross_up)
                   for item in ("cross_up", "cross_down")]
    cross_positions = set(task_rng.sample(range(spec.task_count),
                                          spec.cross_floor_count))
    dog_favored_count = int(spec.cross_floor_count * .3 + .5)
    dog_favored_cross_indices = set(task_rng.sample(
        range(spec.cross_floor_count), dog_favored_count))
    ordered_types = []
    same_index = 0
    cross_index = 0
    for index in range(spec.task_count):
        if index in cross_positions:
            ordered_types.append(cross_types[cross_index])
            cross_index += 1
        else:
            ordered_types.append(same_types[same_index])
            same_index += 1

    arrivals = []
    now = 0.0
    cross_scenario_index = 0
    for task_number, task_type in enumerate(ordered_types, start=1):
        is_cross = task_type.startswith("cross_")
        dog_favored = (is_cross and
                       cross_scenario_index in dog_favored_cross_indices)
        source, target = _task_points(
            task_type, task_rng, task_number, dog_favored=dog_favored)
        if is_cross:
            cross_scenario_index += 1
        priority = 3 if task_type.startswith("cross_") else task_rng.randint(1, 2)
        deadline = now + 180.0 + task_rng.randint(0, 60)
        task = Task(f"S14_T{task_number:02d}", f"S14_C{task_number:02d}",
                    source, target, priority, deadline)
        arrivals.append((QueuedTask(task, now), task_type))
        now += arrival_rng.uniform(*arrival_interval)
    return arrivals


def build_mixed_load_arrivals(seed: int):
    """Build one persistent 60-task NORMAL -> DENSE -> BURST episode.

    Each phase contains the same balanced 20-task class mix.  Task and cargo
    IDs remain unique across phase boundaries, while deadlines and arrivals
    are shifted onto one continuous clock.  No robot or resource state is
    reset between phases because all rows belong to one environment episode.
    """
    phases = (
        ("NORMAL", NORMAL_INTERVAL),
        ("DENSE", DENSE_INTERVAL),
        ("BURST", BURST_INTERVAL),
    )
    arrivals = []
    offset = 0.0
    global_index = 0
    for phase_index, (phase, interval) in enumerate(phases):
        segment = build_balanced_arrivals(
            seed + phase_index * 10000, arrival_interval=interval)
        if phase_index:
            # The next load begins shortly after the previous phase's final
            # arrival, so in-flight work and queue state can cross the boundary.
            offset = arrivals[-1][0].arrival_time + sum(interval) / 2.0
        for queued, task_type in segment:
            global_index += 1
            task = queued.task
            shifted = Task(
                f"S15_MIX_T{global_index:03d}",
                f"S15_MIX_C{global_index:03d}",
                task.source, task.target, task.priority,
                task.deadline + offset)
            arrivals.append((
                QueuedTask(shifted, queued.arrival_time + offset),
                f"{phase}:{task_type}"))
    return arrivals


def build_mixed_recovery_arrivals(seed: int):
    """Build one persistent 80-task load-rise-and-recovery episode.

    The four balanced 20-task phases share one continuous simulation clock:
    NORMAL -> DENSE -> BURST -> RECOVERY.  RECOVERY returns to the normal
    arrival interval without resetting robots, queues, cargo, or resources,
    so the final phase measures whether burst backlog is actually cleared.
    """
    phases = (
        ("NORMAL", NORMAL_INTERVAL),
        ("DENSE", DENSE_INTERVAL),
        ("BURST", BURST_INTERVAL),
        ("RECOVERY", NORMAL_INTERVAL),
    )
    arrivals = []
    offset = 0.0
    global_index = 0
    for phase_index, (phase, interval) in enumerate(phases):
        segment = build_balanced_arrivals(
            seed + phase_index * 10000, arrival_interval=interval)
        if phase_index:
            offset = arrivals[-1][0].arrival_time + sum(interval) / 2.0
        for queued, task_type in segment:
            global_index += 1
            task = queued.task
            shifted = Task(
                f"S19_MIX_T{global_index:03d}",
                f"S19_MIX_C{global_index:03d}",
                task.source, task.target, task.priority,
                task.deadline + offset)
            arrivals.append((
                QueuedTask(shifted, queued.arrival_time + offset),
                f"{phase}:{task_type}"))
    return arrivals


def _mixed_curriculum_lengths(rng: random.Random) -> list[int]:
    """Draw four 4-task-granular phase lengths that sum to 80.

    The 12--28 range prevents a phase from becoming a token transition while
    retaining enough variation that policy time cannot identify the load.
    """
    units = [3, 3, 3, 3]
    for _ in range(8):
        candidates = [index for index, value in enumerate(units) if value < 7]
        units[rng.choice(candidates)] += 1
    lengths = [value * 4 for value in units]
    assert sum(lengths) == 80 and all(12 <= value <= 28 for value in lengths)
    return lengths


def _mixed_curriculum_spec(task_count: int, rng: random.Random) -> Stage14EpisodeSpec:
    if task_count % 4:
        raise ValueError("mixed-curriculum phase length must be divisible by four")
    cross_each_direction = task_count // 4
    same_total = task_count - 2 * cross_each_direction
    same_counts = [same_total // 4] * 4
    for index in rng.sample(range(4), same_total % 4):
        same_counts[index] += 1
    return Stage14EpisodeSpec(
        f1_same=same_counts[0], f1_cross_region=same_counts[1],
        f2_same=same_counts[2], f2_cross_region=same_counts[3],
        cross_up=cross_each_direction, cross_down=cross_each_direction)


def build_mixed_curriculum_arrivals(seed: int, *, return_metadata: bool = False):
    """Build a randomized persistent four-load 80-task training episode.

    Phase order, phase lengths, and inter-phase gaps vary by seed. Recovery is
    constrained to occur after Burst so it remains semantically meaningful;
    all other ordering is free. The task mix within every phase remains half
    same-floor and half cross-floor with balanced up/down directions.
    """
    schedule_rng = random.Random(seed + 202000)
    orders = []
    base = ["NORMAL", "DENSE", "BURST", "RECOVERY"]
    # Enumerate with a local RNG instead of relying on set iteration order.
    for order in itertools.permutations(base):
        if order.index("BURST") < order.index("RECOVERY"):
            orders.append(order)
    phase_order = list(schedule_rng.choice(orders))
    lengths = _mixed_curriculum_lengths(schedule_rng)
    intervals = {
        "NORMAL": NORMAL_INTERVAL,
        "DENSE": DENSE_INTERVAL,
        "BURST": BURST_INTERVAL,
        "RECOVERY": NORMAL_INTERVAL,
    }
    arrivals = []
    offset = 0.0
    global_index = 0
    phase_rows = []
    for phase_index, (phase, task_count) in enumerate(zip(phase_order, lengths)):
        interval = intervals[phase]
        phase_rng = random.Random(seed + 310000 + phase_index * 1000)
        spec = _mixed_curriculum_spec(task_count, phase_rng)
        segment = build_balanced_arrivals(
            seed + phase_index * 10000, spec=spec, arrival_interval=interval)
        gap = 0.0
        if phase_index:
            mean_interval = sum(interval) / 2.0
            gap = schedule_rng.uniform(.35, 1.35) * mean_interval
            offset = arrivals[-1][0].arrival_time + gap
        phase_start = offset
        for queued, task_type in segment:
            global_index += 1
            task = queued.task
            shifted = Task(
                f"S20_CURR_T{global_index:03d}",
                f"S20_CURR_C{global_index:03d}",
                task.source, task.target, task.priority,
                task.deadline + offset)
            arrivals.append((
                QueuedTask(shifted, queued.arrival_time + offset),
                f"{phase}:{task_type}"))
        phase_rows.append({
            "phase_index": phase_index,
            "phase": phase,
            "task_count": task_count,
            "arrival_interval_s": list(interval),
            "switch_gap_s": gap,
            "first_arrival_s": phase_start,
            "last_arrival_s": arrivals[-1][0].arrival_time,
        })
    metadata = {
        "schema_version": "warehouse_stage20_mixed_curriculum_v1",
        "seed": seed,
        "phase_order": phase_order,
        "phase_lengths": lengths,
        "total_tasks": len(arrivals),
        "state_reset_between_phases": False,
        "phase_rows": phase_rows,
    }
    if return_metadata:
        return arrivals, metadata
    return arrivals


def build_stage14_environment(
        seed: int, execution_mode: str = "SERIAL",
        arrival_interval: tuple[float, float] = MEDIUM_INTERVAL,
        episode_spec: Stage14EpisodeSpec | None = None,
        handover_sampling: str = "SEQUENTIAL",
        arrival_schedule: str = "BALANCED",
        environment_class_override=None,
        reward_contract: str = "REWARD_V2_LEGACY"):
    execution_mode = execution_mode.upper()
    if execution_mode not in {"SERIAL", "CONCURRENT"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    robots = [
        _runtime("dog_1", "dog", 1, "F1_NE", (20.8, 17.8, .45)),
        _runtime("dog_2", "dog", 1, "F1_NW", (-20.8, 17.8, .45)),
        _runtime("dog_3", "dog", 1, "F1_SW", (-20.8, -17.8, .45)),
        _runtime("dog_4", "dog", 1, "F1_SE", (20.8, -17.8, .45)),
        _runtime("car_f1_1", "car", 1, "F1_NE", (2.5, 2.5, .45)),
        _runtime("car_f1_2", "car", 1, "F1_NW", (-2.5, 2.5, .45)),
        _runtime("car_f1_3", "car", 1, "F1_SW", (-2.5, -2.5, .45)),
        _runtime("car_f1_4", "car", 1, "F1_SE", (2.5, -2.5, .45)),
        _runtime("car_f2_1", "car", 2, "F2_WEST", (-5.5, 0, 4.7)),
        _runtime("car_f2_2", "car", 2, "F2_EAST", (5.5, 0, 4.7)),
    ]
    stairs = {
        "STAIR_NE": {"floor1_xyz": (20.3, 18.3, .45),
                     "floor2_xyz": (9.5, 6.5, 4.7), "state": "FREE"},
        "STAIR_NW": {"floor1_xyz": (-20.3, 18.3, .45),
                     "floor2_xyz": (-9.5, 6.5, 4.7), "state": "FREE"},
        "STAIR_SW": {"floor1_xyz": (-20.3, -18.3, .45),
                     "floor2_xyz": (-9.5, -6.5, 4.7), "state": "FREE"},
        "STAIR_SE": {"floor1_xyz": (20.3, -18.3, .45),
                     "floor2_xyz": (9.5, -6.5, 4.7), "state": "FREE"},
    }
    slots = [
        StandbySlot(f"F1_CAR_{index + 1}", "car", 1, xyz)
        for index, xyz in enumerate(((-4.5, 3.0, .45), (4.5, 3.0, .45),
                                     (-4.5, -3.0, .45), (4.5, -3.0, .45)))
    ] + [
        StandbySlot(f"F2_CAR_{index + 1}", "car", 2, xyz)
        for index, xyz in enumerate(((-5.5, 0.0, 4.7), (5.5, 0.0, 4.7)))
    ] + [
        StandbySlot(f"F{floor}_DOG_{index + 1}", "dog", floor, xyz)
        for floor, height in ((1, .45), (2, 4.7))
        for index, xyz in enumerate(((-2.0, 1.5, height), (2.0, 1.5, height),
                                     (-2.0, -1.5, height), (2.0, -1.5, height)))
    ]
    arrival_schedule = arrival_schedule.upper()
    schedule_metadata = {
        "schema_version": "warehouse_fixed_arrival_schedule_v1",
        "schedule": arrival_schedule,
        "seed": seed,
    }
    if arrival_schedule == "BALANCED":
        arrivals = build_balanced_arrivals(
            seed, spec=episode_spec, arrival_interval=arrival_interval)
    elif arrival_schedule == "MIXED_LOAD":
        if episode_spec is not None:
            raise ValueError(
                "MIXED_LOAD owns its three fixed 20-task phase specs")
        arrivals = build_mixed_load_arrivals(seed)
    elif arrival_schedule == "MIXED_LOAD_RECOVERY":
        if episode_spec is not None:
            raise ValueError(
                "MIXED_LOAD_RECOVERY owns its four fixed 20-task phase specs")
        arrivals = build_mixed_recovery_arrivals(seed)
    elif arrival_schedule == "MIXED_CURRICULUM":
        if episode_spec is not None:
            raise ValueError(
                "MIXED_CURRICULUM owns its randomized phase specs")
        arrivals, schedule_metadata = build_mixed_curriculum_arrivals(
            seed, return_metadata=True)
    else:
        raise ValueError(f"unsupported arrival schedule: {arrival_schedule}")
    sampling = handover_sampling.upper()
    if sampling == "SEQUENTIAL":
        timeout_policy, duration_provider = \
            build_provisional_policy_and_provider(seed + 141400)
    elif sampling == "TASK_KEYED":
        timeout_policy, duration_provider = \
            build_provisional_policy_and_keyed_provider(seed + 141400)
    else:
        raise ValueError(f"unsupported handover sampling: {handover_sampling}")
    environment_class = environment_class_override or (
        PersistentDispatchEnvironment if execution_mode == "SERIAL"
        else ConcurrentPersistentDispatchEnvironment)
    expected_base = (PersistentDispatchEnvironment
                     if execution_mode == "SERIAL"
                     else ConcurrentPersistentDispatchEnvironment)
    if not issubclass(environment_class, expected_base):
        raise TypeError(
            "environment_class_override must preserve the requested "
            "dispatch execution semantics")
    time_limit = (6000 if arrival_schedule == "MIXED_CURRICULUM" else
                  5400 if arrival_schedule == "MIXED_LOAD_RECOVERY" else 3600)
    reward_contract = reward_contract.upper()
    reward_configs = {
        "REWARD_V2_LEGACY": STAGE14_REWARD_CONFIG,
        "CONTINUOUS_TIME_V3": STAGE23_REWARD_CONFIG,
    }
    if reward_contract not in reward_configs:
        raise ValueError(f"unsupported reward contract: {reward_contract}")
    env = environment_class(
        robots, stairs, AsyncTaskQueue(item for item, _ in arrivals),
        task_limit=len(arrivals), time_limit=time_limit, decision_gap=3.0,
        standby_slots=slots,
        reward_config=reward_configs[reward_contract],
        handover_timeout_s=timeout_policy.timeout_s,
        handover_duration_provider=duration_provider,
        handover_nominal_s=duration_provider.mean_duration_s,
        handover_timing_source=timeout_policy.source)
    env.arrival_schedule_metadata = schedule_metadata
    env.reward_contract = reward_contract
    return env, {item.task.task_id: task_type for item, task_type in arrivals}


class WarehouseDispatchGymEnv(gym.Env):
    """Centralized masked policy interface for the Stage 14 first lesson.

    The policy chooses one legal joint assignment. Low-level navigation and
    gait control remain outside the RL action space. The separate candidate
    feature tensor is part of the observation so a policy can compare dynamic
    assignments instead of memorizing unstable action indices.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 20260835,
                 execution_mode: str = "SERIAL",
                 arrival_interval: tuple[float, float] = MEDIUM_INTERVAL,
                 episode_spec: Stage14EpisodeSpec | None = None,
                 handover_sampling: str = "SEQUENTIAL",
                 arrival_schedule: str = "BALANCED",
                 observation_variant: str = "FULL_CONTEXT_V2",
                 allowed_transport_modes: tuple[str, ...] | None = None,
                 dispatch_environment_class=None,
                 dispatch_setup_callback=None,
                 reward_contract: str = "REWARD_V2_LEGACY"):
        super().__init__()
        execution_mode = execution_mode.upper()
        if execution_mode not in {"SERIAL", "CONCURRENT", "MIXED"}:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        self.base_seed = seed
        self.requested_execution_mode = execution_mode
        self.arrival_interval = arrival_interval
        self.episode_spec = episode_spec
        self.handover_sampling = handover_sampling
        self.allowed_transport_modes = (
            frozenset(mode.upper() for mode in allowed_transport_modes)
            if allowed_transport_modes is not None else None)
        valid_modes = set(PersistentDispatchEnvironment.TRANSPORT_MODES)
        if (self.allowed_transport_modes is not None and
                (not self.allowed_transport_modes or
                 not self.allowed_transport_modes <= valid_modes)):
            raise ValueError(
                f"unsupported allowed transport modes: "
                f"{sorted(self.allowed_transport_modes or ())}")
        self.arrival_schedule = arrival_schedule.upper()
        if self.arrival_schedule not in {
                "BALANCED", "MIXED_LOAD", "MIXED_LOAD_RECOVERY",
                "MIXED_CURRICULUM"}:
            raise ValueError(
                f"unsupported arrival schedule: {arrival_schedule}")
        self.current_execution_mode = "SERIAL"
        self.dispatch_environment_class = dispatch_environment_class
        self.dispatch_setup_callback = dispatch_setup_callback
        self.reward_contract = reward_contract.upper()
        if self.reward_contract not in {
                "REWARD_V2_LEGACY", "CONTINUOUS_TIME_V3"}:
            raise ValueError(
                f"unsupported reward contract: {reward_contract}")
        self.reset_count = 0
        self.encoder = Stage12Encoder(observation_variant)
        self.action_space = spaces.Discrete(self.encoder.MAX_ACTIONS)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.encoder.OBSERVATION_SIZE,), dtype=np.float32),
            "action_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.encoder.MAX_ACTIONS, self.encoder.ACTION_WIDTH),
                dtype=np.float32),
        })
        self.dispatch: PersistentDispatchEnvironment | None = None
        self.task_types: dict[str, str] = {}
        self.decision: EncodedDecision | None = None

    def _encode_decision(self) -> EncodedDecision:
        if self.dispatch is None:
            raise RuntimeError("environment must be reset before encoding")
        decision = self.encoder.encode(self.dispatch)
        if self.allowed_transport_modes is None:
            return decision
        mask = decision.action_mask.copy()
        action_ids = list(decision.action_ids)
        features = decision.action_features.copy()
        for index, action in enumerate(action_ids):
            if (isinstance(action, Assignment) and
                    action.transport_mode not in self.allowed_transport_modes):
                mask[index] = False
        if not mask.any():
            # The restricted controller has no legal assignment even if an
            # excluded mode would be feasible.  Expose the normal event-driven
            # WAIT action so busy permitted robots can become available.
            features[0] = 0.0
            features[0, 0] = 1.0
            mask[0] = True
            action_ids[0] = self.encoder.WAIT_ACTION
        return EncodedDecision(
            decision.observation, features, mask, tuple(action_ids))

    def _observation(self):
        if self.decision is None:
            raise RuntimeError("environment must be reset before observation")
        return {
            "state": self.decision.observation,
            "action_features": self.decision.action_features,
        }

    def _info(self, transition=None):
        if self.decision is None or self.dispatch is None:
            raise RuntimeError("environment must be reset before info")
        info = {
            "action_mask": self.decision.action_mask.copy(),
            "time": self.dispatch.now,
            "completed": self.dispatch.completed,
            "failed": self.dispatch.failed,
            "waiting_count": len(self.dispatch.queue.waiting),
            "pending_arrivals": len(self.dispatch.queue.pending_arrivals),
            "execution_mode": self.current_execution_mode,
            "active_task_count": len(getattr(
                self.dispatch, "active_tasks", {})),
            "arrival_schedule": self.arrival_schedule,
            "arrival_schedule_metadata": getattr(
                self.dispatch, "arrival_schedule_metadata", {}),
            "reward_contract": self.reward_contract,
        }
        if transition is not None:
            task_id = transition.action.get("task_id", "")
            info.update({
                "selected_action": transition.action,
                "task_type": self.task_types.get(task_id, ""),
                "reward_components": transition.reward_components,
                "delta_time": transition.delta_time,
                "discount": transition.discount,
                "termination_reason": transition.termination_reason,
            })
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = (seed if seed is not None else
                        self.base_seed + self.reset_count)
        self.reset_count += 1
        self.current_execution_mode = (
            "SERIAL" if self.requested_execution_mode == "MIXED" and
            episode_seed % 2 == 0 else
            "CONCURRENT" if self.requested_execution_mode == "MIXED" else
            self.requested_execution_mode)
        self.dispatch, self.task_types = build_stage14_environment(
            episode_seed, self.current_execution_mode,
            self.arrival_interval, self.episode_spec,
            self.handover_sampling, self.arrival_schedule,
            self.dispatch_environment_class,
            self.reward_contract)
        if self.dispatch_setup_callback is not None:
            self.dispatch_setup_callback(self.dispatch, episode_seed)
        self.decision = self._encode_decision()
        return self._observation(), self._info()

    def action_masks(self):
        if self.decision is None:
            raise RuntimeError("environment must be reset before action mask")
        return self.decision.action_mask.copy()

    def rule_action(self) -> int:
        if self.decision is None or self.dispatch is None:
            raise RuntimeError("environment must be reset before action selection")
        legal_decoded = [
            item for item, legal in zip(
                self.decision.action_ids, self.decision.action_mask)
            if legal]
        if legal_decoded == [self.encoder.WAIT_ACTION]:
            return 0
        for queued in self.dispatch.queue.policy_view(self.dispatch.now):
            choices = [
                (index, item) for index, (item, legal) in enumerate(zip(
                    self.decision.action_ids, self.decision.action_mask))
                if legal and isinstance(item, Assignment) and
                item.task_id == queued.task.task_id]
            if choices:
                return min(choices, key=lambda pair: (
                    pair[1].estimated_cost, pair[1].pickup_carter,
                    pair[1].dog_id, pair[1].stair_id,
                    pair[1].receiving_carter))[0]
        raise RuntimeError("no executable task in the encoded policy view")

    def step(self, action: int):
        if self.decision is None or self.dispatch is None:
            raise RuntimeError("environment must be reset before step")
        decoded = self.encoder.decode(self.decision, int(action))
        if decoded == self.encoder.WAIT_ACTION:
            if (self.current_execution_mode == "CONCURRENT" and
                    (getattr(self.dispatch, "active_tasks", {}) or
                     self.dispatch.queue.pending_arrivals)):
                transition = self.dispatch.wait_for_next_event()
            elif self.dispatch.queue.waiting:
                reason = ("GLOBAL_DEADLOCK" if self.dispatch.resource_claims
                          else "NO_FEASIBLE_CHAIN")
                transition = self.dispatch.abort_episode(reason, elapsed=1.0)
            else:
                transition = self.dispatch.wait_for_next_arrival()
        else:
            transition = self.dispatch.step(int(action))
        self.decision = self._encode_decision()
        return (self._observation(), float(transition.reward),
                transition.terminated, transition.truncated,
                self._info(transition))
