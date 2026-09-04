#!/usr/bin/env python3
"""Stage 24 rolling-optimizer causality and safety interface gate.

This gate is intentionally PILOT_ONLY.  It must not run a formal locked test
or make a performance claim.  The optimizer import is delayed until ``main``
so the protocol can be reviewed before the implementation module is present.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage24_rolling_optimizer.yaml")
OUTPUT = (ROOT / "results" / "stage24_rolling_optimizer" /
          "interface_gate" / "stage24_optimizer_interface_gate.json")


def _load_optimizer_types():
    """Import the Stage 24 module only when the executable gate is invoked."""
    try:
        module = importlib.import_module(
            "warehouse_core.stage24_rolling_optimizer")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stage 24 optimizer module is not available yet. Expected "
            "warehouse_core.stage24_rolling_optimizer before running this "
            "interface gate.") from exc
    required = ("RollingHorizonOptimizer", "Stage24OptimizerConfig")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "Stage 24 optimizer module is missing required API: " +
            ", ".join(missing))
    return module.RollingHorizonOptimizer, module.Stage24OptimizerConfig


def _load_environment_type():
    module = importlib.import_module("warehouse_core.stage14_training")
    return module.WarehouseDispatchGymEnv


def _jsonable(value: Any) -> Any:
    """Convert state used by the selector to a canonical JSON structure."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item)
                for key, item in sorted(value.items(), key=lambda pair: str(
                    pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(
            getattr(value, "value"), (str, int, float, bool)):
        return value.value
    return repr(value)


def _environment_snapshot(env) -> str:
    """Hash all mutable scheduling state that action selection may observe."""
    dispatch = env.dispatch
    decision = env.decision
    payload = {
        "gym": {
            "base_seed": env.base_seed,
            "reset_count": env.reset_count,
            "np_random_state": env.np_random.bit_generator.state,
            "task_types": env.task_types,
            "allowed_transport_modes": env.allowed_transport_modes,
        },
        "dispatch": {
            "now": dispatch.now,
            "completed": dispatch.completed,
            "failed": dispatch.failed,
            "queue_waiting": dispatch.queue.waiting,
            "queue_pending_arrivals": dispatch.queue.pending_arrivals,
            "robots": dispatch.robots,
            "stairs": dispatch.stairs,
            "standby_slots": dispatch.standby_slots,
            "active_tasks": getattr(dispatch, "active_tasks", {}),
            "resource_claims": dispatch.resource_claims,
            "active_standby_claims": getattr(
                dispatch, "active_standby_claims", {}),
            "resource_events": dispatch.resource_events,
            "handover_events": dispatch.handover_events,
            "completion_events": getattr(dispatch, "completion_events", []),
            "transition_count": len(dispatch.transitions),
            "handover_nominal_s": dispatch.handover_nominal_s,
            "handover_timeout_s": dispatch.handover_timeout_s,
            "handover_timing_source": dispatch.handover_timing_source,
        },
        "decision": None if decision is None else {
            "observation": decision.observation,
            "action_features": decision.action_features,
            "action_mask": decision.action_mask,
            "action_ids": decision.action_ids,
        },
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_set(config: dict) -> set[int]:
    return set(range(int(config["seed_start"]),
                     int(config["seed_start"]) + int(config["seed_count"])))


def _make_env(environment_type, seed: int):
    env = environment_type(
        seed=seed,
        execution_mode="CONCURRENT",
        arrival_schedule="MIXED_CURRICULUM",
        observation_variant="MARKOV_CONTEXT_V3",
        handover_sampling="TASK_KEYED",
        reward_contract="CONTINUOUS_TIME_V3")
    _, info = env.reset(seed=seed)
    return env, info


def _make_selector(optimizer_type, config_type, protocol: dict,
                   *, visibility_mode: str = "RELEASED_ONLY",
                   budget_ms: float | None = None):
    optimizer = protocol["optimizer"]
    compute = optimizer["compute_budget"]
    selected_budget = float(
        compute["primary_wall_clock_budget_ms"]
        if budget_ms is None else budget_ms)
    reserve_ms = min(
        float(compute["budget_reserve_ms"]),
        max(0.0, selected_budget * 0.1))
    configuration = config_type(
        visibility_mode=visibility_mode,
        horizon_decisions=int(optimizer["horizon_decisions"]),
        wall_clock_budget_ms=selected_budget,
        max_root_rollouts=int(optimizer["max_root_rollouts"]),
        budget_reserve_ms=reserve_ms,
        fallback_policy=str(optimizer["fallback"]["policy"]),
    )
    return optimizer_type(configuration)


def _legal(info: dict, action: int) -> bool:
    mask = np.asarray(info["action_mask"], dtype=bool)
    return isinstance(action, (int, np.integer)) and 0 <= int(action) < len(
        mask) and bool(mask[int(action)])


def _mutate_only_unreleased_future_tasks(env) -> int:
    """Change all pending task specifications without changing released work."""
    pending = env.dispatch.queue.pending_arrivals
    replacements = []
    for index, item in enumerate(pending):
        if not item.arrival_time > env.dispatch.now:
            raise RuntimeError("test fixture contains a released pending task")
        replacements.append(replace(
            item,
            task=replace(
                item.task,
                priority=int(item.task.priority) + 10 + (index % 3),
                deadline=float(item.task.deadline) + 10000.0 + index),
        ))
    env.dispatch.queue.pending_arrivals = replacements
    return len(replacements)


def _run_selection(selector, env) -> tuple[int | None, dict, str | None]:
    try:
        action = int(selector.select_action(env))
        diagnostics = dict(getattr(selector, "last_diagnostics", {}) or {})
        return action, diagnostics, None
    except Exception as exc:  # Evidence is recorded in the gate JSON.
        return None, dict(getattr(selector, "last_diagnostics", {}) or {}), (
            f"{type(exc).__name__}: {exc}")


def _provider_that_must_not_be_called(*_args, **_kwargs):
    raise AssertionError(
        "selector attempted to read a realized task-keyed handover sample")


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    optimizer_type, config_type = _load_optimizer_types()
    environment_type = _load_environment_type()

    seeds = protocol["pilot_design"]["seeds"]
    interface_seeds = _seed_set(seeds["interface_gate"])
    pilot_seeds = _seed_set(seeds["paired_pilot"])
    microcase_seeds = _seed_set(seeds["exact_gap_microcases"])
    stage23_locked = set(range(
        int(protocol["stage23_reference"][
            "forbidden_stage23_locked_seed_range"][0]),
        int(protocol["stage23_reference"][
            "forbidden_stage23_locked_seed_range"][1]) + 1))
    selected_seed = min(interface_seeds)

    # Determinism and action legality use independent but identical fixtures.
    env_a, info_a = _make_env(environment_type, selected_seed)
    env_b, info_b = _make_env(environment_type, selected_seed)
    selector_a = _make_selector(optimizer_type, config_type, protocol)
    selector_b = _make_selector(optimizer_type, config_type, protocol)
    action_a, diagnostics_a, error_a = _run_selection(selector_a, env_a)
    action_b, diagnostics_b, error_b = _run_selection(selector_b, env_b)

    # Selecting must be a pure read of the real environment.
    mutation_env, mutation_info = _make_env(environment_type, selected_seed + 1)
    mutation_selector = _make_selector(
        optimizer_type, config_type, protocol)
    snapshot_before = _environment_snapshot(mutation_env)
    mutation_action, mutation_diagnostics, mutation_error = _run_selection(
        mutation_selector, mutation_env)
    snapshot_after = _environment_snapshot(mutation_env)

    # RELEASED_ONLY must not react to any change in unreleased future tasks.
    released_env_a, released_info_a = _make_env(
        environment_type, selected_seed + 2)
    released_env_b, released_info_b = _make_env(
        environment_type, selected_seed + 2)
    mutated_future_task_count = _mutate_only_unreleased_future_tasks(
        released_env_b)
    released_selector_a = _make_selector(
        optimizer_type, config_type, protocol,
        visibility_mode="RELEASED_ONLY")
    released_selector_b = _make_selector(
        optimizer_type, config_type, protocol,
        visibility_mode="RELEASED_ONLY")
    released_action_a, released_diagnostics_a, released_error_a = \
        _run_selection(released_selector_a, released_env_a)
    released_action_b, released_diagnostics_b, released_error_b = \
        _run_selection(released_selector_b, released_env_b)

    # The optimizer may use a frozen risk model, but never the realized keyed
    # sample.  Replacing the real provider with a fail-fast sentinel verifies
    # that selection does not query it.
    handover_env, handover_info = _make_env(
        environment_type, selected_seed + 3)
    handover_env.dispatch.handover_duration_provider = \
        _provider_that_must_not_be_called
    handover_selector = _make_selector(
        optimizer_type, config_type, protocol)
    handover_action, handover_diagnostics, handover_error = _run_selection(
        handover_selector, handover_env)

    # A deliberately tiny budget must terminate in a declared legal fallback.
    fallback_env, fallback_info = _make_env(
        environment_type, selected_seed + 4)
    tiny_budget = float(protocol["optimizer"]["compute_budget"][
        "interface_tiny_budget_ms"])
    fallback_selector = _make_selector(
        optimizer_type, config_type, protocol, budget_ms=tiny_budget)
    fallback_action, fallback_diagnostics, fallback_error = _run_selection(
        fallback_selector, fallback_env)

    formal = protocol["formal_locked_test"]
    visibility = protocol["information_contract"]
    checks = {
        "protocol_is_pilot_only": (
            protocol["protocol"]["status"] == "PILOT_ONLY" and
            protocol["pilot_design"]["status"] == "PILOT_ONLY"),
        "released_and_announced_visibility_modes_are_explicit": (
            set(visibility["visibility_modes"]) == {
                "RELEASED_ONLY", "ANNOUNCED_PENDING"}),
        "formal_test_is_blocked_pending_visibility_decision": (
            visibility["formal_test_status"] ==
            "blocked_pending_visibility_decision" and
            formal["status"] == "blocked_pending_visibility_decision" and
            not formal["authorized"] and formal["seed_count"] == 0 and
            formal["seed_start"] is None),
        "pilot_seed_partitions_are_new_and_disjoint": not any((
            interface_seeds & pilot_seeds,
            interface_seeds & microcase_seeds,
            pilot_seeds & microcase_seeds,
            interface_seeds & stage23_locked,
            pilot_seeds & stage23_locked,
            microcase_seeds & stage23_locked,
        )),
        "identical_seed_produces_identical_decision": (
            error_a is None and error_b is None and action_a == action_b),
        "selected_actions_are_legal": (
            error_a is None and error_b is None and
            _legal(info_a, action_a) and _legal(info_b, action_b)),
        "selector_does_not_mutate_real_environment": (
            mutation_error is None and
            _legal(mutation_info, mutation_action) and
            snapshot_before == snapshot_after),
        "released_only_is_invariant_to_future_task_mutation": (
            mutated_future_task_count > 0 and
            released_error_a is None and released_error_b is None and
            released_action_a == released_action_b and
            _legal(released_info_a, released_action_a) and
            _legal(released_info_b, released_action_b)),
        "selector_does_not_read_realized_task_keyed_handover": (
            handover_error is None and
            _legal(handover_info, handover_action)),
        "tiny_budget_uses_legal_fallback": (
            fallback_error is None and
            _legal(fallback_info, fallback_action) and
            fallback_diagnostics.get("fallback_used") is True and
            fallback_diagnostics.get("timed_out") is True),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "warehouse_stage24_optimizer_interface_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 24,
        "gate": "ROLLING_OPTIMIZER_CAUSAL_INTERFACE_V1",
        "status": "PILOT_ONLY",
        "claim_boundary": (
            "Interface, causality, determinism, safety and fallback gate only; "
            "not a performance, optimality or formal locked-test claim."),
        "protocol": protocol["protocol"]["version"],
        "protocol_path": str(CONFIG),
        "visibility": {
            "tested_mode": "RELEASED_ONLY",
            "formal_status": formal["status"],
            "mutated_future_task_count": mutated_future_task_count,
        },
        "seed_partitions": {
            "interface_gate": [min(interface_seeds), max(interface_seeds)],
            "paired_pilot": [min(pilot_seeds), max(pilot_seeds)],
            "exact_gap_microcases": [
                min(microcase_seeds), max(microcase_seeds)],
            "stage23_locked_forbidden": [
                min(stage23_locked), max(stage23_locked)],
        },
        "selection_evidence": {
            "determinism": {
                "seed": selected_seed,
                "actions": [action_a, action_b],
                "diagnostics": [diagnostics_a, diagnostics_b],
                "errors": [error_a, error_b],
            },
            "environment_purity": {
                "action": mutation_action,
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "diagnostics": mutation_diagnostics,
                "error": mutation_error,
            },
            "released_only_future_invariance": {
                "actions": [released_action_a, released_action_b],
                "diagnostics": [
                    released_diagnostics_a, released_diagnostics_b],
                "errors": [released_error_a, released_error_b],
            },
            "handover_provider_sentinel": {
                "action": handover_action,
                "diagnostics": handover_diagnostics,
                "error": handover_error,
            },
            "tiny_budget_fallback": {
                "budget_ms": tiny_budget,
                "action": fallback_action,
                "diagnostics": fallback_diagnostics,
                "error": fallback_error,
            },
        },
        "assertions": checks,
        "passed": passed,
        "next_gate": (
            "PAIRED_NONFORMAL_VISIBILITY_AND_BUDGET_PILOT" if passed else
            "REPAIR_INTERFACE_GATE_FAILURES"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "assertion_count": len(checks),
        "passed": passed,
        "next_gate": report["next_gate"],
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
