"""Fixed-shape Stage 12 observation and masked candidate-action encoding."""

from dataclasses import dataclass
from math import dist

import numpy as np

from .persistent_dispatch import Assignment, PersistentDispatchEnvironment


@dataclass(frozen=True)
class EncodedDecision:
    observation: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action_ids: tuple[Assignment | str | None, ...]


class Stage12Encoder:
    """Turn the event-driven environment into deterministic policy tensors.

    Candidate actions retain separate source-side and target-side costs.  Stair
    selection is therefore learnable and is never hard-wired to a task region.
    """

    MAX_TASKS = 8
    MAX_ROBOTS = 10
    MAX_STAIRS = 4
    MAX_ACTIONS = 1536
    WAIT_ACTION = "WAIT"
    TASK_WIDTH = 13
    ROBOT_WIDTH = 14
    STAIR_WIDTH = 8
    # The first 16 values describe the assignment itself.  The remaining 12
    # bind the selected task's context and the assignment's relative cost to
    # that candidate, so a policy can switch modes per task instead of learning
    # one global transport-mode preference from the shared observation.
    ACTION_WIDTH = 30
    GLOBAL_WIDTH = 8
    OBSERVATION_VARIANTS = (
        "FULL_CONTEXT_V2", "NO_QUEUE_RESOURCE",
        "NO_HANDOVER_CUES", "NO_PERSISTENT_POSITION",
        "NO_POSITION_ONLY", "NO_ETA_COST_ONLY", "NO_HISTORY_ONLY",
        "NO_EXPLICIT_QUEUE_RESOURCE", "NO_EXPLICIT_HANDOVER_RISK")
    # Explicitly appended risk fields. Keeping these separate makes the
    # handover ablation remove only risk information, not transport semantics.
    HANDOVER_COUNT_INDEX = 28
    HANDOVER_TIMEOUT_PROBABILITY_INDEX = 29
    SINGLE_HANDOVER_TIMEOUT_PROBABILITY = 0.05
    # The trained Stage 14 contract contains 20 scheduled tasks per episode.
    # Longer Stage 15 episodes must not expose an out-of-range count of all
    # future releases; waiting/active counts remain the causal load signals.
    PENDING_ARRIVAL_COUNT_CAP = 20
    OBSERVATION_SIZE = (
        GLOBAL_WIDTH + MAX_TASKS * TASK_WIDTH +
        MAX_ROBOTS * ROBOT_WIDTH + MAX_STAIRS * STAIR_WIDTH)

    def __init__(self, observation_variant: str = "FULL_CONTEXT_V2"):
        observation_variant = observation_variant.upper()
        if observation_variant not in self.OBSERVATION_VARIANTS:
            raise ValueError(
                f"unsupported observation variant: {observation_variant}")
        self.observation_variant = observation_variant

    @classmethod
    def queue_resource_state_indices(cls) -> tuple[int, ...]:
        stair_start = (cls.GLOBAL_WIDTH + cls.MAX_TASKS * cls.TASK_WIDTH +
                       cls.MAX_ROBOTS * cls.ROBOT_WIDTH)
        return (1, 2, 3, *(
            stair_start + index * cls.STAIR_WIDTH + 1
            for index in range(cls.MAX_STAIRS)))

    @classmethod
    def persistent_position_state_indices(cls) -> tuple[int, ...]:
        robot_start = cls.GLOBAL_WIDTH + cls.MAX_TASKS * cls.TASK_WIDTH
        offsets = (3, 8, 9, 10, 11, 12, 13)
        return tuple(
            robot_start + robot * cls.ROBOT_WIDTH + offset
            for robot in range(cls.MAX_ROBOTS) for offset in offsets)

    @classmethod
    def position_only_state_indices(cls) -> tuple[int, ...]:
        """Absolute geometry only; cross-floor flags and derived costs stay."""
        task_start = cls.GLOBAL_WIDTH
        robot_start = task_start + cls.MAX_TASKS * cls.TASK_WIDTH
        stair_start = robot_start + cls.MAX_ROBOTS * cls.ROBOT_WIDTH
        task_indices = tuple(
            task_start + task * cls.TASK_WIDTH + offset
            for task in range(cls.MAX_TASKS)
            for offset in (1, 2, 7, 8, 9, 10, 11, 12))
        robot_indices = tuple(
            robot_start + robot * cls.ROBOT_WIDTH + offset
            for robot in range(cls.MAX_ROBOTS) for offset in (3, 8, 9, 10))
        stair_indices = tuple(
            stair_start + stair * cls.STAIR_WIDTH + offset
            for stair in range(cls.MAX_STAIRS) for offset in (2, 3, 4, 5, 6, 7))
        return task_indices + robot_indices + stair_indices

    @classmethod
    def history_only_state_indices(cls) -> tuple[int, ...]:
        robot_start = cls.GLOBAL_WIDTH + cls.MAX_TASKS * cls.TASK_WIDTH
        return (5, 6, *(
            robot_start + robot * cls.ROBOT_WIDTH + offset
            for robot in range(cls.MAX_ROBOTS) for offset in (11, 12, 13)))

    @staticmethod
    def position_only_action_indices() -> tuple[int, ...]:
        return (19, 20, 21, 22, 23, 24)

    @staticmethod
    def eta_cost_only_action_indices() -> tuple[int, ...]:
        return (10, 11, 12, 13, 14, 15, 25, 26)

    @staticmethod
    def persistent_position_action_indices() -> tuple[int, ...]:
        return (10, 11, 12, 14, 15, 25, 26)

    def _apply_observation_ablation(self, values: np.ndarray) -> np.ndarray:
        output = values.copy()
        if self.observation_variant == "NO_QUEUE_RESOURCE":
            output[list(self.queue_resource_state_indices())] = 0.0
        elif self.observation_variant == "NO_PERSISTENT_POSITION":
            output[list(self.persistent_position_state_indices())] = 0.0
        elif self.observation_variant == "NO_POSITION_ONLY":
            output[list(self.position_only_state_indices())] = 0.0
        elif self.observation_variant == "NO_HISTORY_ONLY":
            output[list(self.history_only_state_indices())] = 0.0
        elif self.observation_variant == "NO_EXPLICIT_QUEUE_RESOURCE":
            output[list(self.queue_resource_state_indices())] = 0.0
        return output

    def _apply_action_ablation(self, features: np.ndarray) -> np.ndarray:
        output = features.copy()
        if self.observation_variant == "NO_HANDOVER_CUES":
            output[:, [self.HANDOVER_COUNT_INDEX,
                       self.HANDOVER_TIMEOUT_PROBABILITY_INDEX]] = 0.0
        elif self.observation_variant == "NO_PERSISTENT_POSITION":
            output[:, list(self.persistent_position_action_indices())] = 0.0
        elif self.observation_variant == "NO_POSITION_ONLY":
            output[:, list(self.position_only_action_indices())] = 0.0
        elif self.observation_variant == "NO_ETA_COST_ONLY":
            output[:, list(self.eta_cost_only_action_indices())] = 0.0
        elif self.observation_variant == "NO_EXPLICIT_HANDOVER_RISK":
            output[:, [self.HANDOVER_COUNT_INDEX,
                       self.HANDOVER_TIMEOUT_PROBABILITY_INDEX]] = 0.0
        return output

    @staticmethod
    def _xyz(values):
        return (values[0] / 25.0, values[1] / 25.0, values[2] / 5.0)

    def encode_observation(self, env: PersistentDispatchEnvironment):
        visible = env.queue.policy_view(env.now, self.MAX_TASKS)
        values = [
            env.now / max(1.0, env.time_limit),
            len(env.queue.waiting) / 50.0,
            min(len(env.queue.pending_arrivals),
                self.PENDING_ARRIVAL_COUNT_CAP) / 50.0,
            len(getattr(env, "active_tasks", {})) / max(
                1, getattr(env, "max_concurrent_tasks", 1)),
            float(getattr(env, "execution_mode", "SERIAL") == "CONCURRENT"),
            env.completed / max(1, env.task_limit),
            env.failed / max(1, env.task_limit),
            env.decision_gap / 10.0,
        ]
        for index in range(self.MAX_TASKS):
            if index >= len(visible):
                values.extend([0.0] * self.TASK_WIDTH)
                continue
            item = visible[index]
            task = item.task
            values.extend([
                1.0,
                task.source.floor / 2.0,
                task.target.floor / 2.0,
                float(task.source.floor != task.target.floor),
                task.priority / 3.0,
                max(0.0, env.now - item.arrival_time) / 600.0,
                max(0.0, task.deadline - env.now) / 600.0,
                *self._xyz(task.source.xyz),
                *self._xyz(task.target.xyz),
            ])
        robot_ids = sorted(env.robots)
        for index in range(self.MAX_ROBOTS):
            if index >= len(robot_ids):
                values.extend([0.0] * self.ROBOT_WIDTH)
                continue
            runtime = env.robots[robot_ids[index]]
            values.extend([
                1.0,
                float(runtime.robot.robot_type == "dog"),
                env.robot_speeds[runtime.robot.robot_type] / 2.0,
                runtime.robot.current_floor / 2.0,
                float(runtime.robot.available),
                float(bool(runtime.robot.failure_code)),
                float(bool(runtime.robot.cargo_id)),
                float(bool(runtime.robot.task_id)),
                *self._xyz(runtime.xyz),
                runtime.distance_total / 1000.0,
                runtime.tasks_completed / 50.0,
                max(0.0, env.now - runtime.idle_since) / 600.0,
            ])
        for index, (_, stair) in enumerate(sorted(env.stairs.items())):
            if index >= self.MAX_STAIRS:
                break
            values.extend([
                1.0,
                float(stair.get("state", "FREE") == "FREE"),
                *self._xyz(stair["floor1_xyz"]),
                *self._xyz(stair["floor2_xyz"]),
            ])
        stair_padding = self.MAX_STAIRS - min(len(env.stairs), self.MAX_STAIRS)
        values.extend([0.0] * (stair_padding * self.STAIR_WIDTH))
        output = np.asarray(values, dtype=np.float32)
        if output.shape != (self.OBSERVATION_SIZE,):
            raise RuntimeError(f"invalid observation shape {output.shape}")
        return self._apply_observation_ablation(output)

    def _action_feature(self, env, assignment, task_slots, robot_slots,
                        stair_slots, minimum_task_cost):
        queued_task = env._task(assignment.task_id)
        task = queued_task.task
        pickup = env.robots.get(assignment.pickup_carter)
        receiver = env.robots.get(assignment.receiving_carter)
        dog = env.robots.get(assignment.dog_id)
        car_speed = env.robot_speeds["car"]
        dog_speed = env.robot_speeds["dog"]
        pickup_eta = (dist(pickup.xyz, task.source.xyz) / car_speed
                      if pickup else 0.0)
        dog_eta = (dist(dog.xyz, task.source.xyz) / dog_speed
                   if assignment.transport_mode == env.SINGLE_DOG and dog
                   else 0.0)
        source_side_eta = pickup_eta or dog_eta
        stair_eta = 0.0
        target_side_eta = (dist(task.source.xyz, task.target.xyz) /
                           (dog_speed if assignment.transport_mode ==
                            env.SINGLE_DOG else car_speed))
        if assignment.stair_id:
            stair = env.stairs[assignment.stair_id]
            entry = tuple(stair[f"floor{task.source.floor}_xyz"])
            exit_point = tuple(stair[f"floor{task.target.floor}_xyz"])
            stair_eta = dist(entry, exit_point) / dog_speed
            if assignment.transport_mode == env.SINGLE_DOG:
                source_side_eta = (dog_eta +
                                   dist(task.source.xyz, entry) / dog_speed)
                target_side_eta = dist(exit_point, task.target.xyz) / dog_speed
            else:
                dog_eta = dist(dog.xyz, entry) / dog_speed
                source_side_eta = pickup_eta + max(
                    dist(task.source.xyz, entry) / car_speed, dog_eta)
                target_side_eta = (
                    dist(receiver.xyz, exit_point) / car_speed +
                    dist(exit_point, task.target.xyz) / car_speed)
        modes = [float(assignment.transport_mode == mode)
                 for mode in env.TRANSPORT_MODES]
        participant_ids = {
            robot_id for robot_id in (
                assignment.pickup_carter, assignment.dog_id,
                assignment.receiving_carter)
            if robot_id is not None}
        cost_ratio = assignment.estimated_cost / max(
            minimum_task_cost, 1e-6)
        handover_count = 2 if assignment.transport_mode == \
            env.CAR_DOG_CAR else 0
        timeout_probability = 1.0 - (
            1.0 - self.SINGLE_HANDOVER_TIMEOUT_PROBABILITY
        ) ** handover_count
        return [
            1.0,
            *modes,
            task_slots[assignment.task_id] / self.MAX_TASKS,
            (robot_slots.get(assignment.pickup_carter, -1) + 1) /
            (self.MAX_ROBOTS + 1),
            (robot_slots.get(assignment.dog_id, -1) + 1) /
            (self.MAX_ROBOTS + 1),
            (stair_slots.get(assignment.stair_id, -1) + 1) /
            (self.MAX_STAIRS + 1),
            (robot_slots.get(assignment.receiving_carter, -1) + 1) /
            (self.MAX_ROBOTS + 1),
            float(task.source.floor != task.target.floor),
            pickup_eta / 100.0,
            dog_eta / 100.0,
            source_side_eta / 200.0,
            stair_eta / 100.0,
            target_side_eta / 200.0,
            assignment.estimated_cost / 300.0,
            task.priority / 3.0,
            max(0.0, env.now - queued_task.arrival_time) / 600.0,
            max(0.0, task.deadline - env.now) / 600.0,
            *self._xyz(task.source.xyz),
            *self._xyz(task.target.xyz),
            max(0.0, assignment.estimated_cost - minimum_task_cost) / 300.0,
            min(3.0, cost_ratio) / 3.0,
            len(participant_ids) / 3.0,
            handover_count / 2.0,
            timeout_probability,
        ]

    def encode(self, env: PersistentDispatchEnvironment) -> EncodedDecision:
        candidates = env.enumerate_candidate_actions()
        if len(candidates) > self.MAX_ACTIONS:
            raise RuntimeError(
                f"candidate count {len(candidates)} exceeds {self.MAX_ACTIONS}")
        visible = env.queue.policy_view(env.now, self.MAX_TASKS)
        task_slots = {item.task.task_id: index
                      for index, item in enumerate(visible)}
        robot_slots = {robot_id: index
                       for index, robot_id in enumerate(sorted(env.robots))}
        stair_slots = {stair_id: index
                       for index, stair_id in enumerate(sorted(env.stairs))}
        features = np.zeros(
            (self.MAX_ACTIONS, self.ACTION_WIDTH), dtype=np.float32)
        mask = np.zeros(self.MAX_ACTIONS, dtype=np.bool_)
        legal = env.build_action_mask(candidates)
        action_ids = [None] * self.MAX_ACTIONS
        minimum_task_costs = {}
        for candidate, is_legal in zip(candidates, legal):
            if not is_legal:
                continue
            previous = minimum_task_costs.get(candidate.task_id)
            if previous is None or candidate.estimated_cost < previous:
                minimum_task_costs[candidate.task_id] = candidate.estimated_cost
        for index, (candidate, is_legal) in enumerate(zip(candidates, legal)):
            features[index] = self._action_feature(
                env, candidate, task_slots, robot_slots, stair_slots,
                minimum_task_costs.get(
                    candidate.task_id, candidate.estimated_cost))
            mask[index] = is_legal
            action_ids[index] = candidate
        if not mask.any():
            # Index 0 is a controlled event-driven wait, not padding.
            features[0, 0] = 1.0
            mask[0] = True
            action_ids[0] = self.WAIT_ACTION
        return EncodedDecision(
            self.encode_observation(env),
            self._apply_action_ablation(features), mask, tuple(action_ids))

    def metadata(self) -> dict:
        return {
            "observation_variant": self.observation_variant,
            "state_width": self.OBSERVATION_SIZE,
            "action_width": self.ACTION_WIDTH,
            "queue_resource_state_indices": list(
                self.queue_resource_state_indices()),
            "persistent_position_state_indices": list(
                self.persistent_position_state_indices()),
            "persistent_position_action_indices": list(
                self.persistent_position_action_indices()),
            "handover_cue_action_indices": [
                self.HANDOVER_COUNT_INDEX,
                self.HANDOVER_TIMEOUT_PROBABILITY_INDEX],
            "position_only_state_indices": list(
                self.position_only_state_indices()),
            "position_only_action_indices": list(
                self.position_only_action_indices()),
            "eta_cost_only_action_indices": list(
                self.eta_cost_only_action_indices()),
            "history_only_state_indices": list(
                self.history_only_state_indices()),
            "construct_boundary": {
                "NO_EXPLICIT_QUEUE_RESOURCE": (
                    "removes explicit telemetry; safety action mask remains"),
                "NO_EXPLICIT_HANDOVER_RISK": (
                    "removes risk fields; transport-mode semantics remain"),
            },
        }

    def decode(self, decision: EncodedDecision,
               action_index: int) -> Assignment | str:
        if action_index < 0 or action_index >= self.MAX_ACTIONS:
            raise ValueError("encoded action index is outside the action tensor")
        if not decision.action_mask[action_index]:
            raise ValueError("encoded action is masked")
        action = decision.action_ids[action_index]
        if action is None:
            raise ValueError("encoded action points to padding")
        return action
