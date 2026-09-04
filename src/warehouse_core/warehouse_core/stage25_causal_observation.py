"""Causal released-only observation contract for Stage 25.

The frozen Stage 23 ``MARKOV_CONTEXT_V3`` encoder includes every unreleased
task and the realised completion record of active assignments.  That is valid
only for a fully announced order book.  This version keeps the same fixed
tensor width and action catalogue, but exposes only information available to
an online dispatcher at the current decision time.

The resulting observation is deliberately described as a causal POMDP/SMDP
observation, not as a fully observed Markov state: future arrivals and
unrealised handover outcomes remain latent.
"""

from __future__ import annotations

import numpy as np

from .stage12_encoding import Stage12Encoder
from .stage14_training import WarehouseDispatchGymEnv, spaces


CAUSAL_RELEASED_V4 = "CAUSAL_RELEASED_V4"


class CausalReleasedEncoder(Stage12Encoder):
    """Versioned, shape-compatible released-only encoder.

    The 284-value legacy prefix is retained for architecture compatibility,
    except that its pending-arrival count is always zero.  The appended task
    table contains released waiting and active tasks only.  The active-event
    table contains current reservations and elapsed service context, while
    exact sampled finish times, outcomes, handover durations, and final robot
    poses are withheld.
    """

    OBSERVATION_VARIANT = CAUSAL_RELEASED_V4

    def __init__(self, observation_variant: str = CAUSAL_RELEASED_V4):
        if observation_variant.upper() != self.OBSERVATION_VARIANT:
            raise ValueError(
                f"unsupported causal observation variant: {observation_variant}")
        # Reuse only the frozen V3 tensor dimensions and action encoder.
        super().__init__("MARKOV_CONTEXT_V3")
        self.observation_variant = self.OBSERVATION_VARIANT

    def _markov_v3_context(self, env):
        """Encode released state without querying unreleased task records."""

        active_tasks = getattr(env, "active_tasks", {})
        records = {}
        for item in env.queue.waiting:
            records[item.task.task_id] = (
                item.task, "WAITING", item.arrival_time)
        for task_id, active in active_tasks.items():
            records[task_id] = (
                active.task, "ACTIVE", active.started_at)
        ordered = sorted(records.items())
        if len(ordered) > self.MARKOV_MAX_TASKS:
            raise RuntimeError(
                f"Causal V4 task table overflow: {len(ordered)} > "
                f"{self.MARKOV_MAX_TASKS}")
        task_slots = {
            task_id: index for index, (task_id, _) in enumerate(ordered)}
        values = []
        for index in range(self.MARKOV_MAX_TASKS):
            if index >= len(ordered):
                values.extend([0.0] * self.MARKOV_TASK_WIDTH)
                continue
            _, (task, status, release_time) = ordered[index]
            values.extend([
                1.0,
                float(status == "WAITING"),
                0.0,  # PENDING is structurally absent in released-only V4.
                float(status == "ACTIVE"),
                task.source.floor / 2.0,
                task.target.floor / 2.0,
                float(task.source.floor != task.target.floor),
                task.priority / 3.0,
                (release_time - env.now) / 600.0,
                (task.deadline - env.now) / 600.0,
                *self._xyz(task.source.xyz),
                *self._xyz(task.target.xyz),
            ])

        robot_ids = sorted(env.robots)
        robot_slots = {
            robot_id: index for index, robot_id in enumerate(robot_ids)}
        stair_slots = {
            stair_id: index for index, stair_id in enumerate(sorted(env.stairs))}

        ordered_active = sorted(active_tasks.items())
        if len(ordered_active) > self.MARKOV_MAX_ACTIVE:
            raise RuntimeError(
                f"Causal V4 active table overflow: {len(ordered_active)} > "
                f"{self.MARKOV_MAX_ACTIVE}")
        for index in range(self.MARKOV_MAX_ACTIVE):
            if index >= len(ordered_active):
                values.extend([0.0] * self.MARKOV_ACTIVE_WIDTH)
                continue
            task_id, active = ordered_active[index]
            assignment = active.assignment
            roles = (
                assignment.pickup_carter, assignment.dog_id,
                assignment.receiving_carter)
            role_slots = [
                (robot_slots.get(robot_id, -1) + 1) /
                (self.MAX_ROBOTS + 1) for robot_id in roles]
            mode_flags = [
                float(assignment.transport_mode == mode)
                for mode in env.TRANSPORT_MODES]
            handover_count = int(
                assignment.transport_mode == env.CAR_DOG_CAR) * 2
            timeout_probability = 1.0 - (
                1.0 - self.SINGLE_HANDOVER_TIMEOUT_PROBABILITY
            ) ** handover_count
            # Keep the 39-field V3 shape.  Fields that would reveal the exact
            # sampled future (finish time, realised route/outcome, handover
            # samples, final poses) are zero by contract.
            values.extend([
                1.0,
                (task_slots[task_id] + 1) / (self.MARKOV_MAX_TASKS + 1),
                0.0,  # exact remaining completion time is latent
                max(0.0, env.now - active.started_at) / 600.0,
                assignment.estimated_cost / 300.0,
                *mode_flags,
                *role_slots,
                (stair_slots.get(assignment.stair_id, -1) + 1) /
                (self.MAX_STAIRS + 1),
                len(active.resources) / 8.0,
                len(active.participant_ids) / 3.0,
                0.0,  # realised travelled distance is latent
                handover_count / 2.0,
                timeout_probability,
                (active.task.deadline - env.now) / 600.0,
                *([0.0] * 21),  # exact final runtimes are latent
            ])
        expected = (
            self.MARKOV_MAX_TASKS * self.MARKOV_TASK_WIDTH +
            self.MARKOV_MAX_ACTIVE * self.MARKOV_ACTIVE_WIDTH)
        if len(values) != expected:
            raise RuntimeError(
                f"invalid Causal V4 context width {len(values)} != {expected}")
        return values

    def encode_observation(self, env):
        # The frozen implementation appends its extended table only when the
        # internal selector equals MARKOV_CONTEXT_V3.  Switch that selector
        # only for the duration of the shape-building call; polymorphism still
        # routes the appended table to our released-only implementation.
        variant = self.observation_variant
        self.observation_variant = "MARKOV_CONTEXT_V3"
        try:
            output = super().encode_observation(env)
        finally:
            self.observation_variant = variant
        # Legacy-prefix index 2 was a count of all future releases.  It must
        # not vary with an unreleased schedule under the strict online contract.
        output[2] = np.float32(0.0)
        return output

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update({
            "observation_variant": self.OBSERVATION_VARIANT,
            "visibility_contract": "RELEASED_ONLY",
            "decision_process_claim": "CAUSAL_PARTIALLY_OBSERVED_SMDP",
            "pending_task_records_exposed": 0,
            "pending_task_count_exposed": False,
            "future_arrival_phase_exposed": False,
            "realised_active_completion_exposed": False,
            "state_width": self.OBSERVATION_SIZE,
            "shape_compatible_with": "MARKOV_CONTEXT_V3",
            "released_task_table": {
                "max_tasks": self.MARKOV_MAX_TASKS,
                "task_width": self.MARKOV_TASK_WIDTH,
                "statuses": ["WAITING", "ACTIVE"],
            },
            "causal_active_table": {
                "max_active": self.MARKOV_MAX_ACTIVE,
                "active_width": self.MARKOV_ACTIVE_WIDTH,
                "exact_finish_and_outcome_fields_zeroed": True,
            },
        })
        metadata["markov_v3_task_table"] = None
        metadata["markov_v3_active_event_table"] = None
        return metadata


class CausalReleasedWarehouseDispatchGymEnv(WarehouseDispatchGymEnv):
    """Warehouse Gym adapter using :class:`CausalReleasedEncoder`."""

    def __init__(self, *args,
                 observation_variant: str = CAUSAL_RELEASED_V4, **kwargs):
        if observation_variant.upper() != CAUSAL_RELEASED_V4:
            raise ValueError(
                "Stage 25 environment requires CAUSAL_RELEASED_V4")
        # The base constructor builds the unchanged dispatch/action machinery.
        # Task-keyed execution randomness is stable under action-order changes
        # and is the causal Stage 25 default for both training and validation.
        kwargs.setdefault("handover_sampling", "TASK_KEYED")
        super().__init__(
            *args, observation_variant="MARKOV_CONTEXT_V3", **kwargs)
        self.encoder = CausalReleasedEncoder()
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
