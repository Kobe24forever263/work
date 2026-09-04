#!/usr/bin/env python3
"""Audit the Stage 25 released-only causal observation boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment, QueuedTask
from warehouse_core.stage12_encoding import Stage12Encoder
from warehouse_core.stage25_causal_observation import (
    CAUSAL_RELEASED_V4, CausalReleasedWarehouseDispatchGymEnv)


OUTPUT = (ROOT / "results" / "stage25_causal_online" / "gates" /
          "stage25_causal_interface_gate.json")


def mutate_future_schedule(env) -> None:
    """Change future count, order, content, deadlines, and release times."""
    mutated = []
    for index, item in enumerate(env.dispatch.queue.pending_arrivals[::2]):
        task = replace(
            item.task,
            source=item.task.target,
            target=item.task.source,
            priority=1 + (item.task.priority + index) % 3,
            deadline=item.task.deadline + 900.0 + index)
        mutated.append(QueuedTask(
            task, item.arrival_time + 100.0 + index * 0.125))
    env.dispatch.queue.pending_arrivals = list(reversed(mutated))


def first_assignment(env) -> int:
    for index, (action, legal) in enumerate(zip(
            env.decision.action_ids, env.decision.action_mask)):
        if legal and isinstance(action, Assignment):
            return index
    raise RuntimeError("no legal assignment exposed")


def run_full_episode(seed: int) -> dict:
    env = CausalReleasedWarehouseDispatchGymEnv(
        seed=seed, execution_mode="CONCURRENT",
        arrival_schedule="MIXED_CURRICULUM",
        reward_contract="CONTINUOUS_TIME_V3")
    _, info = env.reset(seed=seed)
    illegal = 0
    steps = 0
    done = False
    while not done:
        action = env.rule_action()
        illegal += int(not info["action_mask"][action])
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    dispatch = env.dispatch
    leak = bool(
        dispatch.resource_claims or
        getattr(dispatch, "active_standby_claims", {}) or
        getattr(dispatch, "active_tasks", {}) or
        any(runtime.robot.task_id or runtime.robot.cargo_id
            for runtime in dispatch.robots.values()))
    return {
        "seed": seed,
        "steps": steps,
        "completed": dispatch.completed,
        "failed": dispatch.failed,
        "illegal_actions": illegal,
        "resource_leak": leak,
    }


def main() -> int:
    seed = 72510001
    env = CausalReleasedWarehouseDispatchGymEnv(
        seed=seed, execution_mode="CONCURRENT",
        arrival_schedule="MIXED_CURRICULUM",
        reward_contract="CONTINUOUS_TIME_V3")
    observation, info = env.reset(seed=seed)
    before_state = observation["state"].copy()
    before_features = observation["action_features"].copy()
    before_mask = info["action_mask"].copy()

    legacy = Stage12Encoder("MARKOV_CONTEXT_V3")
    legacy_before = legacy.encode_observation(env.dispatch)
    pending_before = len(env.dispatch.queue.pending_arrivals)
    mutate_future_schedule(env)
    pending_after = len(env.dispatch.queue.pending_arrivals)
    after = env.encoder.encode(env.dispatch)
    legacy_after = legacy.encode_observation(env.dispatch)

    future_state_delta = float(np.max(np.abs(
        before_state - after.observation)))
    future_feature_delta = float(np.max(np.abs(
        before_features - after.action_features)))
    future_mask_changes = int(np.count_nonzero(
        before_mask != after.action_mask))
    legacy_future_delta = float(np.max(np.abs(
        legacy_before - legacy_after)))

    # A released-task edit must still be visible, proving the invariance test
    # did not accidentally freeze the whole tensor.
    released_env = deepcopy(env)
    released_before = released_env.encoder.encode_observation(
        released_env.dispatch)
    queued = released_env.dispatch.queue.waiting[0]
    released_env.dispatch.queue.waiting[0] = QueuedTask(
        replace(queued.task, priority=(queued.task.priority % 3) + 1),
        queued.arrival_time)
    released_after = released_env.encoder.encode_observation(
        released_env.dispatch)
    released_delta = float(np.max(np.abs(
        released_before - released_after)))

    # Dispatch one concurrent job.  Exact sampled finish/final-state changes
    # must not alter the causal observation, while legacy V3 must react.
    active_env = CausalReleasedWarehouseDispatchGymEnv(
        seed=seed + 1, execution_mode="CONCURRENT",
        arrival_schedule="MIXED_CURRICULUM",
        reward_contract="CONTINUOUS_TIME_V3")
    active_env.reset(seed=seed + 1)
    active_env.step(first_assignment(active_env))
    causal_active_before = active_env.encoder.encode_observation(
        active_env.dispatch)
    legacy_active_before = legacy.encode_observation(active_env.dispatch)
    active = next(iter(active_env.dispatch.active_tasks.values()))
    active.finish_at += 777.0
    for runtime in active.final_runtimes.values():
        runtime.xyz = (runtime.xyz[0] + 9.0,
                       runtime.xyz[1] - 8.0,
                       runtime.xyz[2] + 7.0)
    causal_active_after = active_env.encoder.encode_observation(
        active_env.dispatch)
    legacy_active_after = legacy.encode_observation(active_env.dispatch)
    causal_active_delta = float(np.max(np.abs(
        causal_active_before - causal_active_after)))
    legacy_active_delta = float(np.max(np.abs(
        legacy_active_before - legacy_active_after)))

    episode = run_full_episode(seed + 2)
    metadata = env.encoder.metadata()
    assertions = {
        "state_width_is_1720": before_state.shape == (1720,),
        "action_tensor_shape_is_stable": (
            before_features.shape == (1536, 30)),
        "pending_count_is_not_encoded": before_state[2] == 0.0,
        "future_schedule_was_materially_changed": (
            pending_before != pending_after and legacy_future_delta > 0.0),
        "future_schedule_does_not_change_policy_state": (
            future_state_delta == 0.0),
        "future_schedule_does_not_change_action_features": (
            future_feature_delta == 0.0),
        "future_schedule_does_not_change_action_mask": (
            future_mask_changes == 0),
        "released_task_change_remains_observable": released_delta > 0.0,
        "unrealised_active_outcome_does_not_change_causal_state": (
            causal_active_delta == 0.0),
        "legacy_active_table_detects_same_future_mutation": (
            legacy_active_delta > 0.0),
        "metadata_declares_released_only_partial_observation": (
            metadata["visibility_contract"] == "RELEASED_ONLY" and
            metadata["decision_process_claim"] ==
            "CAUSAL_PARTIALLY_OBSERVED_SMDP"),
        "full_episode_resolves_all_tasks": (
            episode["completed"] + episode["failed"] == 80),
        "full_episode_actions_are_legal": episode["illegal_actions"] == 0,
        "full_episode_has_no_resource_leak": not episode["resource_leak"],
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    result = {
        "stage": 25,
        "gate": "CAUSAL_RELEASED_V4_INTERFACE_AND_INVARIANCE_GATE",
        "claim_boundary": (
            "Interface and causal-invariance evidence only; no learned-policy "
            "performance or convergence claim."),
        "observation_variant": CAUSAL_RELEASED_V4,
        "pending_records_before_mutation": pending_before,
        "pending_records_after_mutation": pending_after,
        "maximum_deltas": {
            "causal_state_after_future_schedule_mutation": future_state_delta,
            "causal_action_features_after_future_schedule_mutation":
            future_feature_delta,
            "legacy_v3_state_after_future_schedule_mutation":
            legacy_future_delta,
            "causal_state_after_unrealised_active_mutation":
            causal_active_delta,
            "legacy_v3_state_after_unrealised_active_mutation":
            legacy_active_delta,
            "causal_state_after_released_task_mutation": released_delta,
        },
        "action_mask_changes_after_future_mutation": future_mask_changes,
        "encoder_metadata": metadata,
        "full_episode": episode,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
