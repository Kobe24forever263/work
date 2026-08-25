#!/usr/bin/env python3
"""Stage 21 paired fault-robustness evaluation for three schedulers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Stable Apple Silicon import order before torch.
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))

from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402
from warehouse_core.stage14_training import MEDIUM_INTERVAL  # noqa: E402
from warehouse_core.stage21_faults import (  # noqa: E402
    CONTROL, FAULT_SCENARIOS, NAVIGATION_FAILURE, ROBOT_OUTAGE,
    STAIR_OUTAGE, FaultAwareWarehouseDispatchGymEnv)
from run_stage15_stress_evaluation import (  # noqa: E402
    StressScenario, compact, paired_bootstrap, raw_result, run_policy)


MANIFEST = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json"
)
SCENARIO = StressScenario(
    "STAGE21_MIXED_CURRICULUM",
    "Randomized persistent mixed load with a shared external fault plan",
    MEDIUM_INTERVAL, None, "MIXED_CURRICULUM")
SMOKE_SEED_START = 66000000
FORMAL_SEED_START = 66100000
SMOKE_OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_smoke.json")
FORMAL_OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_formal_single_weight.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_weight(seed_index: int) -> tuple[Path, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in manifest["runs"]:
        if int(row["training_seed_index"]) != seed_index:
            continue
        weight = Path(row["best_weight_path"])
        actual = sha256(weight)
        if actual != row["best_weight_sha256"]:
            raise RuntimeError(f"frozen checkpoint hash mismatch: {weight}")
        return weight, actual
    raise ValueError(f"missing frozen Stage 20 seed {seed_index}")


def env_factory(fault_scenario: str):
    def build(**kwargs):
        return FaultAwareWarehouseDispatchGymEnv(
            fault_scenario=fault_scenario, **kwargs)
    return build


def fault_events(result: dict) -> list[dict]:
    return [
        event for episode in result["episodes"]
        for event in episode.get("fault_events", [])]


def recovery_after_windows(result: dict) -> dict:
    samples = []
    unresolved = 0
    for episode in result["episodes"]:
        schedule = episode.get("fault_schedule") or {}
        completed = sorted(
            task["finished_at"] for task in episode["tasks"]
            if task["task_result"] == "COMPLETED")
        for window in schedule.get("windows", []):
            end = float(window["end_s"])
            later = [timestamp for timestamp in completed if timestamp >= end]
            if later:
                samples.append(later[0] - end)
            else:
                unresolved += 1
    return {
        "resolved_windows": len(samples),
        "unresolved_windows": unresolved,
        "mean_time_to_next_completion_s": (
            float(np.mean(samples)) if samples else None),
        "p95_time_to_next_completion_s": (
            float(np.percentile(samples, 95)) if samples else None),
        "samples_s": samples,
    }


def degradation(result: dict, control: dict) -> dict[str, float]:
    return {
        "success_rate_delta": (
            result["success_rate"] - control["success_rate"]),
        "reward_delta": (
            result["mean_episode_reward"] -
            control["mean_episode_reward"]),
        "successful_throughput_delta": (
            result["successful_throughput_tasks_per_hour"] -
            control["successful_throughput_tasks_per_hour"]),
        "mean_waiting_time_delta_s": (
            result["mean_waiting_time"] - control["mean_waiting_time"]),
        "mean_flow_time_delta_s": (
            result["mean_flow_time"] - control["mean_flow_time"]),
    }


def method_valid(result: dict, episode_count: int) -> bool:
    return bool(
        result["episode_count"] == episode_count and
        result["task_count"] == episode_count * SCENARIO.task_count and
        result["completed"] + result["failed"] ==
        episode_count * SCENARIO.task_count and
        result["illegal_action_count"] == 0 and
        result["resource_leak_count"] == 0 and
        len(result["episodes"]) == episode_count and
        all(len(episode["tasks"]) == SCENARIO.task_count
            for episode in result["episodes"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage20-seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument(
        "--faults", nargs="+", choices=FAULT_SCENARIOS,
        default=list(FAULT_SCENARIOS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.episodes = min(args.episodes, 2)
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if CONTROL not in args.faults:
        parser.error("CONTROL is required to calculate paired degradation")

    seed_start = args.seed_start or (
        SMOKE_SEED_START if args.smoke else FORMAL_SEED_START)
    seeds = list(range(seed_start, seed_start + args.episodes))
    weight, weight_hash = frozen_weight(args.stage20_seed)
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(weight, torch.device("cpu"))
    model.eval()
    observation_variant = checkpoint.get(
        "observation_variant", "FULL_CONTEXT_V2")

    raw = {}
    results = {}
    assertions = {}
    for fault_index, fault in enumerate(args.faults):
        print(f"Stage21 fault scenario: {fault}", flush=True)
        factory = env_factory(fault)
        policy = run_policy(
            model, seeds, SCENARIO, "PPO", "CONCURRENT",
            observation_variant=observation_variant,
            env_factory=factory)
        rule = run_policy(
            model, seeds, SCENARIO, "TIME_GREEDY_RULE", "CONCURRENT",
            observation_variant=observation_variant,
            env_factory=factory)
        dog = run_policy(
            model, seeds, SCENARIO, "TIME_GREEDY_RULE", "CONCURRENT",
            observation_variant=observation_variant,
            allowed_transport_modes=("SINGLE_DOG",),
            env_factory=factory)
        dog["policy_kind"] = "SINGLE_DOG_ONLY"
        methods = {"ppo": policy, "time_greedy": rule, "single_dog": dog}

        shared_tasks = len({
            item["task_fingerprint"] for item in methods.values()}) == 1
        shared_faults = len({
            item["fault_fingerprint"] for item in methods.values()}) == 1
        scenario_events = {
            name: fault_events(result) for name, result in methods.items()}
        if fault == CONTROL:
            fault_activated = all(
                not events for events in scenario_events.values())
        elif fault == NAVIGATION_FAILURE:
            fault_activated = all(any(
                event["event"] == "TASK_NAVIGATION_FAILURE"
                for event in events) for events in scenario_events.values())
        else:
            fault_activated = all(any(
                event["event"] == "FAULT_WINDOW_START"
                for event in events) for events in scenario_events.values())

        local = {
            "shared_task_stream_fingerprint": shared_tasks,
            "shared_fault_schedule_fingerprint": shared_faults,
            "ppo_complete_legal_and_leak_free": method_valid(
                policy, args.episodes),
            "time_greedy_complete_legal_and_leak_free": method_valid(
                rule, args.episodes),
            "single_dog_complete_legal_and_leak_free": method_valid(
                dog, args.episodes),
            "fault_activation_observed": fault_activated,
        }
        assertions.update({
            f"{fault.lower()}_{key}": value
            for key, value in local.items()})
        results[fault] = {
            "policy": compact(policy),
            "time_greedy": compact(rule),
            "single_dog": compact(dog),
            "policy_minus_time_greedy": paired_bootstrap(
                policy, rule, seed_start + 2100 + fault_index,
                samples=1000 if args.smoke else 4000),
            "policy_minus_single_dog": paired_bootstrap(
                policy, dog, seed_start + 3100 + fault_index,
                samples=1000 if args.smoke else 4000),
            "recovery": {
                name: recovery_after_windows(result)
                for name, result in methods.items()},
            "event_counts": {
                name: len(events) for name, events in scenario_events.items()},
            "assertions": local,
        }
        raw[fault] = {
            name: raw_result(result) | {
                "fault_fingerprint": result["fault_fingerprint"],
                "fault_schedules": [
                    episode.get("fault_schedule")
                    for episode in result["episodes"]],
                "fault_events": [
                    episode.get("fault_events", [])
                    for episode in result["episodes"]],
            } for name, result in methods.items()
        }

    control = results[CONTROL]
    for fault in args.faults:
        results[fault]["degradation_from_control"] = {
            method: degradation(
                results[fault][method], control[method])
            for method in ("policy", "time_greedy", "single_dog")}

    output = args.output or (
        SMOKE_OUTPUT if args.smoke else FORMAL_OUTPUT)
    raw_output = output.with_suffix(".raw.json")
    payload = {
        "schema_version": "warehouse_stage21_fault_robustness_v1",
        "stage": 21,
        "gate": (
            "FAULT_ROBUSTNESS_SMOKE" if args.smoke
            else "FAULT_ROBUSTNESS_SINGLE_WEIGHT"),
        "claim_boundary": (
            "Logical SMDP fault-injection evidence only. Navigation failure "
            "is a task-keyed route-segment abstraction, not a ROS planner or "
            "physical robot failure."),
        "weight": {
            "training_stage": 20,
            "training_seed_index": args.stage20_seed,
            "path": str(weight),
            "sha256": weight_hash,
        },
        "protocol": {
            "execution_mode": "CONCURRENT",
            "arrival_schedule": SCENARIO.arrival_schedule,
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "episode_count_per_method_scenario": args.episodes,
            "task_count_per_episode": SCENARIO.task_count,
            "fault_scenarios": list(args.faults),
            "methods": ["PPO", "TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"],
            "task_randomness": "SHARED_TEST_SEEDS",
            "fault_randomness": "SHARED_PLAN_AND_TASK_KEYED_PREFIX_COUPLING",
            "active_task_preemption": False,
        },
        "raw_results": str(raw_output),
        "results": results,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    raw_payload = {
        "schema_version": "warehouse_stage21_fault_robustness_raw_v1",
        "stage": 21,
        "summary": str(output),
        "scenarios": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "raw_output": str(raw_output),
        "passed": payload["passed"],
        "assertions": assertions,
        "headline": {
            fault: {
                method: {
                    "success_rate": results[fault][method]["success_rate"],
                    "mean_waiting_time": results[fault][method][
                        "mean_waiting_time"],
                    "failure_reasons": results[fault][method][
                        "failure_reasons"],
                } for method in ("policy", "time_greedy", "single_dog")
            } for fault in args.faults},
    }, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
