#!/usr/bin/env python3
"""Paired, pre-registered Stage 15 stress evaluation for PPO and rule policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

# NumPy must load before torch in the Apple Silicon Pixi environment.
import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment
from warehouse_core.handover_timing import HANDOVER_KINDS
from warehouse_core.stage14_ppo import load_checkpoint
from warehouse_core.stage14_training import (
    DENSE_INTERVAL, MEDIUM_INTERVAL, Stage14EpisodeSpec,
    WarehouseDispatchGymEnv)


RESULT_SCHEMA_VERSION = "warehouse_stage16_evaluation_v1"


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str
    arrival_interval: tuple[float, float]
    episode_spec: Stage14EpisodeSpec | None
    arrival_schedule: str = "BALANCED"

    @property
    def task_count(self) -> int:
        return {
            "MIXED_LOAD": 60,
            "MIXED_LOAD_RECOVERY": 80,
            "MIXED_CURRICULUM": 80,
        }.get(self.arrival_schedule, 20)


DEFAULT_SPEC = Stage14EpisodeSpec()
SCENARIOS = {
    "MEDIUM": StressScenario(
        "MEDIUM", "Stage 14 in-distribution anchor", MEDIUM_INTERVAL,
        DEFAULT_SPEC),
    "DENSE": StressScenario(
        "DENSE", "15-25 s arrivals with the original 50/50 task mix",
        DENSE_INTERVAL, DEFAULT_SPEC),
    "CROSS_HEAVY": StressScenario(
        "CROSS_HEAVY",
        "80% cross-floor tasks; up/down directions remain balanced",
        MEDIUM_INTERVAL,
        Stage14EpisodeSpec(
            f1_same=1, f1_cross_region=1,
            f2_same=1, f2_cross_region=1,
            cross_up=8, cross_down=8)),
    "BURST": StressScenario(
        "BURST", "1-3 s burst arrivals with the original 50/50 task mix",
        (1.0, 3.0), DEFAULT_SPEC),
    "MIXED_CONTINUOUS": StressScenario(
        "MIXED_CONTINUOUS",
        "One persistent 60-task NORMAL -> DENSE -> BURST episode",
        MEDIUM_INTERVAL, None, "MIXED_LOAD"),
    "MIXED_CONTINUOUS_RECOVERY": StressScenario(
        "MIXED_CONTINUOUS_RECOVERY",
        "One persistent 80-task NORMAL -> DENSE -> BURST -> RECOVERY episode",
        MEDIUM_INTERVAL, None, "MIXED_LOAD_RECOVERY"),
}

# This baseline is intentionally fixed before the three-way comparison.  It
# does not inspect the realised task-keyed handover sample.  It only knows the
# declared P95 timeout semantics (5% tail probability per handover) and the
# public Reward V2 coefficients, so it is a stronger but still causal rule.
SINGLE_HANDOVER_TIMEOUT_PROBABILITY = 0.05
CDC_TIMEOUT_PROBABILITY = 1.0 - (
    1.0 - SINGLE_HANDOVER_TIMEOUT_PROBABILITY) ** 2
EXPECTED_CDC_HANDOVER_ATTEMPTS = (
    1.0 + (1.0 - SINGLE_HANDOVER_TIMEOUT_PROBABILITY))


def nested_matrix(counts: Counter) -> dict[str, dict[str, int]]:
    rows = {}
    for (context, selected), count in sorted(counts.items()):
        rows.setdefault(context, {})[selected] = count
    return rows


def percentile(values, quantile):
    return float(np.percentile(values, quantile)) if values else 0.0


def task_fingerprint(env: WarehouseDispatchGymEnv) -> tuple[
        str, dict[str, float], dict[str, dict]]:
    items = list(env.dispatch.queue.waiting) + list(
        env.dispatch.queue.pending_arrivals)
    items.sort(key=lambda item: item.task.task_id)
    rows = [{
        "task_id": item.task.task_id,
        "arrival": item.arrival_time,
        "source": [item.task.source.floor, *item.task.source.xyz],
        "target": [item.task.target.floor, *item.task.target.xyz],
        "priority": item.task.priority,
        "deadline": item.task.deadline,
    } for item in items]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    arrivals = {item.task.task_id: item.arrival_time for item in items}
    task_specs = {row["task_id"]: row for row in rows}
    return hashlib.sha256(payload.encode()).hexdigest(), arrivals, task_specs


def reward_aligned_risk_rule_action(env: WarehouseDispatchGymEnv) -> int:
    """Greedy rule with an expected handover-risk premium.

    The task priority order stays identical to the original rule.  Within the
    first executable task, candidates are ranked by nominal ETA plus the
    Reward V2-equivalent expected loss from a two-handover failure chain.
    This is not an oracle: it never sees the realised keyed duration sample.
    """
    if env.decision is None or env.dispatch is None:
        raise RuntimeError("environment must be reset before action selection")
    legal_decoded = [
        item for item, legal in zip(
            env.decision.action_ids, env.decision.action_mask)
        if legal]
    if legal_decoded == [env.encoder.WAIT_ACTION]:
        return 0

    dispatch = env.dispatch
    candidates = dispatch.enumerate_candidate_actions()
    config = dispatch.reward_config

    def risk_adjusted_cost(candidate: Assignment) -> tuple:
        mode = candidate.transport_mode
        participant_count = 3 if mode == dispatch.CAR_DOG_CAR else 1
        # Express reward terms in seconds via outstanding_time_scale so that
        # they can be added to the candidate's ETA estimate.
        active_robot_premium = (
            participant_count * candidate.estimated_cost *
            config.active_robot_time_penalty *
            config.outstanding_time_scale)
        risk_premium = 0.0
        handover_premium = 0.0
        if mode == dispatch.CAR_DOG_CAR:
            success_value = config.completion_bonus + config.on_time_bonus
            failure_value = config.timeout_penalty
            risk_premium = (
                CDC_TIMEOUT_PROBABILITY *
                (success_value + failure_value) *
                config.outstanding_time_scale)
            handover_premium = (
                EXPECTED_CDC_HANDOVER_ATTEMPTS * config.handover_penalty *
                config.outstanding_time_scale)
        return (
            candidate.estimated_cost + active_robot_premium +
            risk_premium + handover_premium,
            candidate.pickup_carter, candidate.dog_id,
            candidate.stair_id, candidate.receiving_carter)

    for queued in dispatch.queue.policy_view(dispatch.now):
        choices = [
            (index, item) for index, item in enumerate(candidates)
            if item.task_id == queued.task.task_id and
            dispatch.is_assignment_legal(item)]
        if choices:
            selected = min(choices, key=lambda pair: risk_adjusted_cost(pair[1]))
            if env.decision.action_ids[selected[0]] != selected[1]:
                raise RuntimeError(
                    "encoded and environment action catalogs diverged")
            return selected[0]
    raise RuntimeError("no executable task in the policy view")


def run_policy(model, seeds, scenario: StressScenario, policy_kind: str,
               execution_mode: str,
               observation_variant: str = "FULL_CONTEXT_V2",
               allowed_transport_modes: tuple[str, ...] | None = None,
               env_factory=None, progress_callback=None,
               reward_contract: str = "REWARD_V2_LEGACY"):
    seeds = tuple(int(seed) for seed in seeds)
    episodes = []
    total_modes = Counter()
    total_failures = Counter()
    total_context = Counter()
    total_floor_relation = Counter()
    total_phase_modes = Counter()
    total_phase_outcomes = Counter()
    phase_waiting = defaultdict(list)
    phase_flow = defaultdict(list)
    all_waiting = []
    all_flow = []
    fingerprints = []
    handover_fingerprints = []
    fault_fingerprints = []
    for episode_index, seed in enumerate(seeds, start=1):
        factory = env_factory or WarehouseDispatchGymEnv
        env = factory(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=scenario.arrival_interval,
            episode_spec=scenario.episode_spec,
            handover_sampling="TASK_KEYED",
            arrival_schedule=scenario.arrival_schedule,
            observation_variant=observation_variant,
            allowed_transport_modes=allowed_transport_modes,
            reward_contract=reward_contract)
        observation, info = env.reset(seed=seed)
        fingerprint, arrivals, task_specs = task_fingerprint(env)
        fingerprints.append(fingerprint)
        # The keyed provider is pure in (episode seed, task id, handover kind).
        # Materialising every potential sample gives a policy-independent CRN
        # fingerprint even when compared policies choose different modes.
        handover_provider = env.dispatch.handover_duration_provider
        handover_rows = [{
            "task_id": task_id,
            "handover_kind": handover_kind,
            "duration_s": handover_provider(handover_kind, task_id),
        } for task_id in sorted(task_specs)
            for handover_kind in HANDOVER_KINDS]
        handover_payload = json.dumps(
            handover_rows, sort_keys=True, separators=(",", ":"))
        handover_fingerprint = hashlib.sha256(
            handover_payload.encode()).hexdigest()
        handover_fingerprints.append(handover_fingerprint)
        fault_fingerprint = getattr(
            env.dispatch, "fault_schedule_fingerprint", "")
        fault_fingerprints.append(fault_fingerprint)
        reward_total = 0.0
        illegal = 0
        peak_active = 0
        modes = Counter()
        context = Counter()
        floor_relation = Counter()
        phase_modes = Counter()
        serial_events = []
        done = False
        while not done:
            if policy_kind == "PPO":
                action = model.select_action(
                    observation, info["action_mask"], deterministic=True)
            elif policy_kind == "TIME_GREEDY_RULE":
                action = env.rule_action()
            elif policy_kind == "RISK_AWARE_RULE":
                action = reward_aligned_risk_rule_action(env)
            else:
                raise ValueError(f"unsupported policy kind: {policy_kind}")
            illegal += int(not info["action_mask"][action])
            decoded = env.decision.action_ids[action]
            decision_started_at = env.dispatch.now
            if isinstance(decoded, Assignment):
                peak_active = max(peak_active, 1)
                modes[decoded.transport_mode] += 1
                task_label = env.task_types.get(decoded.task_id, "")
                phase = (task_label.split(":", 1)[0]
                         if ":" in task_label else scenario.name)
                phase_modes[(phase, decoded.transport_mode)] += 1
                legal_same_task = [
                    candidate for candidate, legal in zip(
                        env.decision.action_ids, info["action_mask"])
                    if legal and isinstance(candidate, Assignment) and
                    candidate.task_id == decoded.task_id]
                reference = min(
                    legal_same_task,
                    key=lambda candidate: candidate.estimated_cost)
                context[(reference.transport_mode,
                         decoded.transport_mode)] += 1
                task = env.dispatch._task(decoded.task_id).task
                relation = ("CROSS_FLOOR" if
                            task.source.floor != task.target.floor else
                            "SAME_FLOOR")
                floor_relation[(relation, decoded.transport_mode)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            if (execution_mode == "SERIAL" and
                    isinstance(decoded, Assignment)):
                selected = info["selected_action"]
                serial_events.append({
                    "task_id": decoded.task_id,
                    "started_at": decision_started_at,
                    "finished_at": (
                        decision_started_at + info["delta_time"] -
                        env.dispatch.decision_gap),
                    "task_result": selected["task_result"],
                    "failure_reason": selected["failure_reason"],
                    "transport_mode": decoded.transport_mode,
                })
            reward_total += reward
            peak_active = max(peak_active, info["active_task_count"])
            done = terminated or truncated

        dispatch = env.dispatch
        events = (dispatch.completion_events
                  if hasattr(dispatch, "completion_events")
                  else serial_events)
        task_rows = []
        event_task_ids = set()
        for event in events:
            task_id = event["task_id"]
            event_task_ids.add(task_id)
            task_label = env.task_types.get(task_id, "")
            phase = (task_label.split(":", 1)[0]
                     if ":" in task_label else scenario.name)
            spec = task_specs[task_id]
            task_rows.append({
                "schema_version": RESULT_SCHEMA_VERSION,
                "scene": scenario.name,
                "policy_kind": policy_kind,
                "eval_seed": seed,
                "task_stream_fingerprint": fingerprint,
                "task_id": task_id,
                "task_label": task_label,
                "phase": phase,
                "arrival_time": arrivals[task_id],
                "started_at": event["started_at"],
                "finished_at": event["finished_at"],
                "waiting_time": event["started_at"] - arrivals[task_id],
                "flow_time": event["finished_at"] - arrivals[task_id],
                "task_result": event["task_result"],
                "failure_reason": event["failure_reason"],
                "transport_mode": event["transport_mode"],
                "participant_ids": list(event.get("participant_ids", [])),
                "handover_durations_s": list(
                    event.get("handover_durations_s", [])),
                "source": spec["source"],
                "target": spec["target"],
                "priority": spec["priority"],
                "deadline": spec["deadline"],
                "floor_relation": (
                    "CROSS_FLOOR" if spec["source"][0] != spec["target"][0]
                    else "SAME_FLOOR"),
            })
        # A fatal terminal transition (for example NO_FEASIBLE_CHAIN under a
        # restricted single-dog controller) resolves the episode but the core
        # environment records only one global failure.  Explicitly account for
        # every unserved task here so blank-control success is not inflated.
        for task_id in sorted(set(task_specs) - event_task_ids):
            task_label = env.task_types.get(task_id, "")
            phase = (task_label.split(":", 1)[0]
                     if ":" in task_label else scenario.name)
            spec = task_specs[task_id]
            task_rows.append({
                "schema_version": RESULT_SCHEMA_VERSION,
                "scene": scenario.name,
                "policy_kind": policy_kind,
                "eval_seed": seed,
                "task_stream_fingerprint": fingerprint,
                "task_id": task_id,
                "task_label": task_label,
                "phase": phase,
                "arrival_time": arrivals[task_id],
                "started_at": dispatch.now,
                "finished_at": dispatch.now,
                "waiting_time": max(0.0, dispatch.now - arrivals[task_id]),
                "flow_time": max(0.0, dispatch.now - arrivals[task_id]),
                "task_result": "FAILED",
                "failure_reason": "TERMINAL_UNRESOLVED",
                "transport_mode": "UNASSIGNED",
                "participant_ids": [],
                "handover_durations_s": [],
                "source": spec["source"],
                "target": spec["target"],
                "priority": spec["priority"],
                "deadline": spec["deadline"],
                "floor_relation": (
                    "CROSS_FLOOR" if spec["source"][0] != spec["target"][0]
                    else "SAME_FLOOR"),
            })
        waiting = [row["waiting_time"] for row in task_rows]
        flow = [row["flow_time"] for row in task_rows]
        failures = Counter(
            row["failure_reason"] for row in task_rows
            if row["task_result"] == "FAILED")
        for row in task_rows:
            task_label = row["task_label"]
            phase = (task_label.split(":", 1)[0]
                     if ":" in task_label else scenario.name)
            total_phase_outcomes[(phase, row["task_result"])] += 1
            phase_waiting[phase].append(row["waiting_time"])
            phase_flow[phase].append(row["flow_time"])
        distance_total = sum(
            runtime.distance_total for runtime in dispatch.robots.values())
        task_count = len(task_specs)
        effective_failed = task_count - dispatch.completed
        leak = bool(
            dispatch.resource_claims or
            getattr(dispatch, "active_standby_claims", {}) or
            getattr(dispatch, "active_tasks", {}) or
            any(runtime.robot.task_id or runtime.robot.cargo_id
                for runtime in dispatch.robots.values()))
        episode = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "seed": seed,
            "task_stream_fingerprint": fingerprint,
            "handover_potential_fingerprint": handover_fingerprint,
            "fault_schedule_fingerprint": fault_fingerprint,
            "task_count": task_count,
            "completed": dispatch.completed,
            "failed": effective_failed,
            "success_rate": dispatch.completed / max(task_count, 1),
            "reward": reward_total,
            "simulated_time": dispatch.now,
            "throughput_tasks_per_hour": (
                task_count / max(dispatch.now, 1e-9) * 3600),
            "successful_throughput_tasks_per_hour": (
                dispatch.completed / max(dispatch.now, 1e-9) * 3600),
            "mean_waiting_time": float(np.mean(waiting)),
            "p95_waiting_time": percentile(waiting, 95),
            "mean_flow_time": float(np.mean(flow)),
            "p95_flow_time": percentile(flow, 95),
            "distance_per_task": distance_total / max(task_count, 1),
            "illegal_actions": illegal,
            "resource_leak": leak,
            "peak_active_tasks": peak_active,
            "termination_reason": info.get("termination_reason", ""),
            "unresolved_at_terminal": len(set(task_specs) - event_task_ids),
            "transport_modes": dict(modes),
            "transport_modes_by_phase": nested_matrix(phase_modes),
            "failure_reasons": dict(failures),
            "tasks": task_rows,
            "fault_schedule": (
                env.dispatch.fault_plan.to_dict()
                if getattr(env.dispatch, "fault_plan", None) else None),
            "fault_events": list(getattr(
                env.dispatch, "fault_events", [])),
        }
        episodes.append(episode)
        total_modes.update(modes)
        total_failures.update(failures)
        total_context.update(context)
        total_floor_relation.update(floor_relation)
        total_phase_modes.update(phase_modes)
        all_waiting.extend(waiting)
        all_flow.extend(flow)
        if progress_callback is not None:
            progress_callback(episode_index, len(seeds), episode)

    task_count = sum(item["task_count"] for item in episodes)
    total_time = sum(item["simulated_time"] for item in episodes)
    total_distance = sum(
        item["distance_per_task"] * item["task_count"] for item in episodes)
    phase_names = sorted(
        phase_waiting,
        key=lambda phase: (
            {"NORMAL": 0, "DENSE": 1, "BURST": 2,
             "RECOVERY": 3}.get(phase, 10), phase))
    phase_results = {}
    phase_mode_matrix = nested_matrix(total_phase_modes)
    for phase in phase_names:
        completed = total_phase_outcomes[(phase, "COMPLETED")]
        failed = total_phase_outcomes[(phase, "FAILED")]
        count = completed + failed
        phase_results[phase] = {
            "task_count": count,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / max(count, 1),
            "mean_waiting_time": float(np.mean(phase_waiting[phase])),
            "p95_waiting_time": percentile(phase_waiting[phase], 95),
            "mean_flow_time": float(np.mean(phase_flow[phase])),
            "p95_flow_time": percentile(phase_flow[phase], 95),
            "transport_modes": phase_mode_matrix.get(phase, {}),
        }
    return {
        "policy_kind": policy_kind,
        "allowed_transport_modes": (
            list(allowed_transport_modes)
            if allowed_transport_modes is not None else None),
        "episode_count": len(episodes),
        "task_count": task_count,
        "completed": sum(item["completed"] for item in episodes),
        "failed": sum(item["failed"] for item in episodes),
        "success_rate": sum(item["completed"] for item in episodes) /
                        max(task_count, 1),
        "mean_episode_reward": float(np.mean(
            [item["reward"] for item in episodes])),
        "throughput_tasks_per_hour": task_count / max(total_time, 1e-9) * 3600,
        "successful_throughput_tasks_per_hour": (
            sum(item["completed"] for item in episodes) /
            max(total_time, 1e-9) * 3600),
        "mean_waiting_time": float(np.mean(all_waiting)),
        "p95_waiting_time": percentile(all_waiting, 95),
        "mean_flow_time": float(np.mean(all_flow)),
        "p95_flow_time": percentile(all_flow, 95),
        "distance_per_task": total_distance / max(task_count, 1),
        "maximum_active_tasks": max(
            item["peak_active_tasks"] for item in episodes),
        "illegal_action_count": sum(
            item["illegal_actions"] for item in episodes),
        "resource_leak_count": sum(
            item["resource_leak"] for item in episodes),
        "transport_modes": dict(total_modes),
        "phase_results": phase_results,
        "failure_reasons": dict(total_failures),
        "selected_mode_by_cost_reference": nested_matrix(total_context),
        "selected_mode_by_floor_relation": nested_matrix(
            total_floor_relation),
        "task_fingerprint": hashlib.sha256(
            "".join(fingerprints).encode()).hexdigest(),
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
        "handover_potential_fingerprint": hashlib.sha256(
            "".join(handover_fingerprints).encode()).hexdigest(),
        "reward_contract": reward_contract.upper(),
        "fault_fingerprint": hashlib.sha256(
            "".join(fault_fingerprints).encode()).hexdigest(),
        "episodes": episodes,
    }


def paired_bootstrap(policy, rule, seed: int, samples: int = 4000):
    keys = (
        "success_rate", "reward", "throughput_tasks_per_hour",
        "successful_throughput_tasks_per_hour",
        "mean_waiting_time", "p95_waiting_time",
        "mean_flow_time", "p95_flow_time", "distance_per_task")
    rng = np.random.default_rng(seed)
    count = len(policy["episodes"])
    indices = rng.integers(0, count, size=(samples, count))
    output = {}
    for key in keys:
        differences = np.asarray([
            p[key] - r[key]
            for p, r in zip(policy["episodes"], rule["episodes"])
        ], dtype=np.float64)
        means = differences[indices].mean(axis=1)
        output[key] = {
            "policy_minus_rule": float(differences.mean()),
            "paired_bootstrap_95ci": [
                float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))],
        }
    return output


def compact(result):
    return {key: value for key, value in result.items()
            if key != "episodes"}


def raw_result(result: dict) -> dict:
    """Return independently reusable episode and task evidence."""
    return {
        "policy_kind": result["policy_kind"],
        "episodes": result["episodes"],
        "tasks": [task for episode in result["episodes"]
                  for task in episode["tasks"]],
    }


def delta_with_ci(comparison: dict, key: str, digits: int = 3) -> str:
    item = comparison[key]
    low, high = item["paired_bootstrap_95ci"]
    value = item["policy_minus_rule"]
    return (f"{value:+.{digits}f} "
            f"[{low:+.{digits}f}, {high:+.{digits}f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43000000)
    parser.add_argument("--execution-mode", choices=("SERIAL", "CONCURRENT"),
                        default="CONCURRENT")
    parser.add_argument("--scenarios", nargs="+",
                        choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")

    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.weights, torch.device("cpu"))
    observation_variant = checkpoint.get(
        "observation_variant", "FULL_CONTEXT_V2")
    seeds = list(range(args.seed, args.seed + args.episodes))
    results = {}
    all_assertions = {}
    for offset, scenario_name in enumerate(args.scenarios):
        scenario = SCENARIOS[scenario_name]
        policy = run_policy(
            model, seeds, scenario, policy_kind="PPO",
            execution_mode=args.execution_mode,
            observation_variant=observation_variant)
        rule = run_policy(
            model, seeds, scenario, policy_kind="TIME_GREEDY_RULE",
            execution_mode=args.execution_mode,
            observation_variant=observation_variant)
        risk_rule = run_policy(
            model, seeds, scenario, policy_kind="RISK_AWARE_RULE",
            execution_mode=args.execution_mode,
            observation_variant=observation_variant)
        comparison = paired_bootstrap(
            policy, rule, args.seed + 1500 + offset)
        risk_comparison = paired_bootstrap(
            policy, risk_rule, args.seed + 2500 + offset)
        adaptive_mode_summary = None
        if scenario.arrival_schedule == "MIXED_LOAD":
            def adaptation(result):
                phases = result["phase_results"]
                shares = {
                    phase: (phases[phase]["transport_modes"].get(
                        "CAR_DOG_CAR", 0) /
                        max(phases[phase]["task_count"] / 2.0, 1.0))
                    for phase in ("NORMAL", "DENSE", "BURST")}
                return {
                    "car_dog_car_share_of_cross_floor_tasks": shares,
                    "normal_to_burst_change": (
                        shares["BURST"] - shares["NORMAL"]),
                }
            adaptive_mode_summary = {
                "policy": adaptation(policy),
                "time_rule": adaptation(rule),
                "risk_rule": adaptation(risk_rule),
            }
        assertions = {
            "paired_task_streams_identical": (
                policy["task_fingerprint"] == rule["task_fingerprint"]),
            "policy_resolved_all_tasks": (
                policy["task_count"] == args.episodes * scenario.task_count),
            "rule_resolved_all_tasks": (
                rule["task_count"] == args.episodes * scenario.task_count),
            "policy_actions_all_legal": (
                policy["illegal_action_count"] == 0),
            "policy_has_no_resource_leak": (
                policy["resource_leak_count"] == 0),
            "rule_has_no_resource_leak": (
                rule["resource_leak_count"] == 0),
            "risk_rule_resolved_all_tasks": (
                risk_rule["task_count"] ==
                args.episodes * scenario.task_count),
            "risk_rule_actions_all_legal": (
                risk_rule["illegal_action_count"] == 0),
            "risk_rule_has_no_resource_leak": (
                risk_rule["resource_leak_count"] == 0),
            "risk_rule_task_streams_identical": (
                policy["task_fingerprint"] == risk_rule["task_fingerprint"]),
        }
        if adaptive_mode_summary is not None:
            policy_shares = adaptive_mode_summary["policy"][
                "car_dog_car_share_of_cross_floor_tasks"]
            assertions.update({
                "policy_transport_distribution_changes_by_phase": (
                    len(set(policy_shares.values())) > 1),
                "policy_reduces_resource_intensive_chain_under_burst": (
                    policy_shares["BURST"] < policy_shares["NORMAL"]),
            })
        all_assertions.update({
            f"{scenario_name.lower()}_{key}": value
            for key, value in assertions.items()})
        results[scenario_name] = {
            "description": scenario.description,
            "arrival_interval": list(scenario.arrival_interval),
            "arrival_schedule": scenario.arrival_schedule,
            "task_count_per_episode": scenario.task_count,
            "episode_spec": (scenario.episode_spec.__dict__
                             if scenario.episode_spec is not None else None),
            "policy": compact(policy),
            "rule_baseline": compact(rule),
            "risk_aware_rule_baseline": compact(risk_rule),
            "paired_comparison": comparison,
            "paired_comparison_vs_risk_aware_rule": risk_comparison,
            "adaptive_mode_summary": adaptive_mode_summary,
            "assertions": assertions,
            "_raw": {
                "policy": raw_result(policy),
                "rule_baseline": raw_result(rule),
                "risk_aware_rule_baseline": raw_result(risk_rule),
            },
        }
        print(
            f"{scenario_name}: policy reward="
            f"{policy['mean_episode_reward']:.3f} success="
            f"{policy['success_rate']:.2%}; rule reward="
            f"{rule['mean_episode_reward']:.3f} success="
            f"{rule['success_rate']:.2%}; risk-rule reward="
            f"{risk_rule['mean_episode_reward']:.3f} success="
            f"{risk_rule['success_rate']:.2%}")

    output = args.output or (
        WORK_ROOT / "results" / "stage15" /
        f"stage15_stress_{args.episodes}ep.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".md")
    raw_path = output.with_suffix(".raw.json")
    raw_scenarios = {
        name: item.pop("_raw") for name, item in results.items()
    }
    raw_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "stage": 16,
        "artifact_kind": "RAW_PAIRED_EVALUATION_ROWS",
        "weights": str(args.weights.resolve()),
        "checkpoint_update": checkpoint.get("update"),
        "execution_mode": args.execution_mode,
        "observation_variant": observation_variant,
        "train_seed": checkpoint.get("base_seed"),
        "eval_seed_start": seeds[0],
        "eval_seed_end": seeds[-1],
        "scenarios": raw_scenarios,
    }
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "stage": 15,
        "gate": "PRE_REGISTERED_PAIRED_STRESS_EVALUATION",
        "claim_boundary": (
            "Safety/reproducibility gate. Superiority is reported with paired "
            "confidence intervals and is not required for PASS."),
        "weights": str(args.weights.resolve()),
        "raw_results": str(raw_path),
        "checkpoint_update": checkpoint.get("update"),
        "execution_mode": args.execution_mode,
        "episode_count_per_scenario": args.episodes,
        "task_count_per_policy_per_scenario": {
            name: args.episodes * SCENARIOS[name].task_count
            for name in args.scenarios},
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "scenario_order_pre_registered": list(args.scenarios),
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
        "risk_aware_rule_definition": {
            "causal": True,
            "realised_handover_sample_visible": False,
            "single_handover_timeout_probability":
                SINGLE_HANDOVER_TIMEOUT_PROBABILITY,
            "two_handover_task_timeout_probability": CDC_TIMEOUT_PROBABILITY,
            "expected_handover_attempts": EXPECTED_CDC_HANDOVER_ATTEMPTS,
            "objective": (
                "estimated ETA + Reward V2-equivalent expected timeout, "
                "handover and active-robot premiums"),
        },
        "results": results,
        "assertions": all_assertions,
        "passed": all(all_assertions.values()),
    }
    raw_path.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        "# Stage 15预注册配对压力评估", "",
        f"- 权重：`{summary['weights']}`",
        f"- 执行模式：{args.execution_mode}",
        f"- 每场景episode：{args.episodes}",
        f"- 固定种子：{seeds[0]}～{seeds[-1]}",
        "- 所有差值均为PPO减规则基线；场景和种子在运行前固定。", "",
        "| 场景 | PPO成功率 | 时间规则 | 风险规则 | PPO回报 | 时间规则 | 风险规则 |", 
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        policy = item["policy"]
        rule = item["rule_baseline"]
        risk_rule = item["risk_aware_rule_baseline"]
        lines.append(
            f"| {name} | {policy['success_rate']:.2%} | "
            f"{rule['success_rate']:.2%} | "
            f"{risk_rule['success_rate']:.2%} | "
            f"{policy['mean_episode_reward']:.3f} | "
            f"{rule['mean_episode_reward']:.3f} | "
            f"{risk_rule['mean_episode_reward']:.3f} |")
    for name, item in results.items():
        phases = item["policy"].get("phase_results", {})
        if not {"NORMAL", "DENSE", "BURST"}.issubset(phases):
            continue
        lines.extend([
            "", f"## {name}连续阶段内行为", "",
            "| 策略 | 阶段 | 成功率 | 平均等待(s) | P95等待(s) | 单车 | 单狗 | 车—狗—车 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("PPO", "policy"),
                           ("时间规则", "rule_baseline"),
                           ("风险规则", "risk_aware_rule_baseline")):
            for phase in ("NORMAL", "DENSE", "BURST"):
                row = item[key]["phase_results"][phase]
                modes = row["transport_modes"]
                lines.append(
                    f"| {label} | {phase} | {row['success_rate']:.2%} | "
                    f"{row['mean_waiting_time']:.3f} | "
                    f"{row['p95_waiting_time']:.3f} | "
                    f"{modes.get('SINGLE_CAR', 0)} | "
                    f"{modes.get('SINGLE_DOG', 0)} | "
                    f"{modes.get('CAR_DOG_CAR', 0)} |")
        adaptation = item.get("adaptive_mode_summary")
        if adaptation:
            lines.extend([
                "", "跨层任务中车—狗—车占比：", "",
                "| 策略 | NORMAL | DENSE | BURST | NORMAL→BURST变化 |",
                "|---|---:|---:|---:|---:|",
            ])
            for label, key in (("PPO", "policy"),
                               ("时间规则", "time_rule"),
                               ("风险规则", "risk_rule")):
                row = adaptation[key]
                shares = row["car_dog_car_share_of_cross_floor_tasks"]
                lines.append(
                    f"| {label} | {shares['NORMAL']:.2%} | "
                    f"{shares['DENSE']:.2%} | {shares['BURST']:.2%} | "
                    f"{row['normal_to_burst_change']:+.2%} |")
    lines.extend([
        "", "差值和95%置信区间完整保存在JSON中，均为PPO减对应基线。",
        "PASS仅代表任务流配对、安全和结果完整，不自动宣称PPO优于规则。",
        "是否存在显著优势必须结合JSON中的配对bootstrap 95%置信区间判断。",
        "", "## 与更强风险规则的配对结果", "",
        "风险规则不知道实际抽中的交接耗时，只使用公开P95风险和Reward V2系数。", "",
        "| 场景 | 成功率差 | 回报差 [95% CI] | 平均等待差(s) [95% CI] | P95等待差(s) [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in results.items():
        delta = item["paired_comparison_vs_risk_aware_rule"]
        lines.append(
            f"| {name} | {delta_with_ci(delta, 'success_rate', 4)} | "
            f"{delta_with_ci(delta, 'reward')} | "
            f"{delta_with_ci(delta, 'mean_waiting_time')} | "
            f"{delta_with_ci(delta, 'p95_waiting_time')} |")
    lines.extend([
        "", "| 场景 | 已解决吞吐差(任务/h) [95% CI] | 成功吞吐差(任务/h) [95% CI] | 平均流转差(s) [95% CI] | 每任务总里程差(m) [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in results.items():
        delta = item["paired_comparison_vs_risk_aware_rule"]
        lines.append(
            f"| {name} | "
            f"{delta_with_ci(delta, 'throughput_tasks_per_hour')} | "
            f"{delta_with_ci(delta, 'successful_throughput_tasks_per_hour')} | "
            f"{delta_with_ci(delta, 'mean_flow_time')} | "
            f"{delta_with_ci(delta, 'distance_per_task')} |")
    lines.extend([
        "", "## 结论边界", "",
        "- 置信区间跨0的指标不宣称存在显著差异。",
        "- 成功率、回报、等待、流转、吞吐和里程必须同时报告，不允许只挑优势指标。",
        "- 本轮PASS不等于Stage 15全部结束；策略消融仍需独立完成。",
        f"- 本轮共解析{sum(SCENARIOS[name].task_count for name in args.scenarios) * args.episodes * 3}个任务结果。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "report": str(report_path),
        "passed": summary["passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
