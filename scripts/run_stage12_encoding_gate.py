#!/usr/bin/env python3
"""Stage 12 fixed-tensor and reversible-action 20-task gate."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(WORK_ROOT / "scripts"))

from run_stage11_persistent_20 import build_environment
from warehouse_core.stage12_encoding import Stage12Encoder


def abnormal_mask_matrix(seed: int, profile: str):
    """Probe every safety-critical invalid-state family directly."""
    env, _ = build_environment(seed, profile)
    env.now = max(item.arrival_time for item in env.queue.pending_arrivals)
    env.queue.advance(env.now)
    cross = next(item for item in env.enumerate_candidate_actions()
                 if item.transport_mode == env.CAR_DOG_CAR)
    task = env._task(cross.task_id).task
    probes = []

    def robot_probe(name, robot_id, field, value):
        robot = env.robots[robot_id].robot
        previous = getattr(robot, field)
        setattr(robot, field, value)
        rejected = not env.build_action_mask([cross])[0]
        setattr(robot, field, previous)
        probes.append({"probe": name, "rejected": rejected})

    for role, robot_id in (("pickup", cross.pickup_carter),
                           ("receiver", cross.receiving_carter),
                           ("dog", cross.dog_id)):
        robot_probe(f"{role}_unavailable", robot_id, "available", False)
        robot_probe(f"{role}_faulted", robot_id, "failure_code", "FAULT")
        robot_probe(f"{role}_busy", robot_id, "task_id", "OTHER_TASK")
        robot_probe(f"{role}_loaded", robot_id, "cargo_id", "OTHER_CARGO")

    stair = env.stairs[cross.stair_id]
    stair["state"] = "OCCUPIED"
    probes.append({"probe": "stair_occupied",
                   "rejected": not env.build_action_mask([cross])[0]})
    stair["state"] = "FREE"

    wrong_floor_car = next(
        robot_id for robot_id, runtime in env.robots.items()
        if runtime.robot.robot_type == "car"
        and runtime.robot.current_floor != task.source.floor)
    probes.append({
        "probe": "pickup_wrong_floor",
        "rejected": not env.build_action_mask([
            replace(cross, pickup_carter=wrong_floor_car)])[0],
    })

    same_floor = next(item for item in env.enumerate_candidate_actions()
                      if not item.dog_id)
    probes.append({
        "probe": "same_floor_calls_dog",
        "rejected": not env.build_action_mask([
            replace(same_floor, dog_id=cross.dog_id,
                    stair_id=cross.stair_id)])[0],
    })
    probes.append({
        "probe": "cross_floor_single_car",
        "rejected": not env.build_action_mask([
            replace(cross, transport_mode=env.SINGLE_CAR,
                    dog_id="", stair_id="",
                    receiving_carter=cross.pickup_carter)])[0],
    })
    probes.append({
        "probe": "same_floor_car_dog_car",
        "rejected": not env.build_action_mask([
            replace(same_floor, transport_mode=env.CAR_DOG_CAR,
                    dog_id=cross.dog_id, stair_id=cross.stair_id,
                    receiving_carter=cross.receiving_carter)])[0],
    })
    probes.append({
        "probe": "car_car_mode_removed",
        "rejected": not env.build_action_mask([
            replace(same_floor, transport_mode="CAR_CAR_HANDOVER")])[0],
    })
    probes.append({
        "probe": "single_dog_with_car_fields",
        "rejected": not env.build_action_mask([
            replace(cross, transport_mode=env.SINGLE_DOG)])[0],
    })
    return probes


def run(seed: int, profile: str):
    env, task_types = build_environment(seed, profile)
    encoder = Stage12Encoder()
    rows = []
    samples = {"observations": [], "action_features": [],
               "action_masks": [], "selected_indices": [],
               "rewards": [], "delta_times": []}
    while env.completed < env.task_limit:
        env.queue.advance(env.now)
        candidates = env.enumerate_candidate_actions()
        if not candidates:
            decision = encoder.encode(env)
            assert encoder.decode(decision, 0) == encoder.WAIT_ACTION
            env.now += 1.0
            if env.now >= env.time_limit:
                raise RuntimeError("WAIT reached the episode time limit")
            continue
        decision = encoder.encode(env)
        repeat = encoder.encode(env)
        selected = env.select_rule_action(candidates)
        decoded = encoder.decode(decision, selected)
        assert decoded == candidates[selected]
        task = env._task(decoded.task_id).task
        stair_choices = {
            action.stair_id for action, legal in zip(
                decision.action_ids, decision.action_mask)
            if legal and hasattr(action, "task_id")
            and action.task_id == decoded.task_id and action.stair_id
        }
        candidate_modes = sorted({
            action.transport_mode for action, legal in zip(
                decision.action_ids, decision.action_mask)
            if legal and hasattr(action, "task_id")
            and action.task_id == decoded.task_id
        })
        ready_source_cars = len(env._available("car", task.source.floor))
        ready_target_cars = len(env._available("car", task.target.floor))
        ready_source_dogs = len(env._available("dog", task.source.floor))
        transition = env.step(selected)
        samples["observations"].append(decision.observation)
        samples["action_features"].append(decision.action_features)
        samples["action_masks"].append(decision.action_mask)
        samples["selected_indices"].append(selected)
        samples["rewards"].append(transition.reward)
        samples["delta_times"].append(transition.delta_time)
        rows.append({
            "decision": len(rows) + 1,
            "task_id": decoded.task_id,
            "task_type": task_types[decoded.task_id],
            "transport_mode": decoded.transport_mode,
            "cross_floor": task.source.floor != task.target.floor,
            "legal_action_count": int(decision.action_mask.sum()),
            "stair_candidate_count": len(stair_choices),
            "candidate_modes": candidate_modes,
            "ready_source_cars": ready_source_cars,
            "ready_target_cars": ready_target_cars,
            "ready_source_dogs": ready_source_dogs,
            "selected_action": decoded.__dict__,
            "observation_shape": list(decision.observation.shape),
            "action_feature_shape": list(decision.action_features.shape),
            "deterministic_encoding": bool(
                np.array_equal(decision.observation, repeat.observation)
                and np.array_equal(decision.action_features,
                                   repeat.action_features)
                and np.array_equal(decision.action_mask, repeat.action_mask)),
            "delta_time": transition.delta_time,
        })
        print(f"[{env.completed}/20] {decoded.task_id} "
              f"legal={int(decision.action_mask.sum())} "
              f"stairs={len(stair_choices)}: PASS")
    counts = Counter(row["task_type"] for row in rows)
    mode_counts = Counter(row["transport_mode"] for row in rows)
    probes = abnormal_mask_matrix(seed, profile)
    assertions = {
        "twenty_tasks_completed": env.completed == 20,
        "fixed_observation_shape": all(
            row["observation_shape"] == [encoder.OBSERVATION_SIZE]
            for row in rows),
        "fixed_action_shape": all(
            row["action_feature_shape"] == [encoder.MAX_ACTIONS,
                                             encoder.ACTION_WIDTH]
            for row in rows),
        "deterministic_encoding": all(
            row["deterministic_encoding"] for row in rows),
        "six_task_types_covered": set(counts) == {
            "f1_same", "f1_cross_region", "f2_same", "f2_cross_region",
            "cross_up", "cross_down"},
        "cross_floor_has_multiple_stairs": all(
            not row["cross_floor"] or row["stair_candidate_count"] >= 2
            for row in rows),
        "mode_choice_matches_task_floor_relation": all(
            set(row["candidate_modes"]) == ({
                mode for mode, enabled in (
                    (env.SINGLE_DOG, row["ready_source_dogs"] > 0),
                    (env.CAR_DOG_CAR, row["cross_floor"] and
                     row["ready_source_cars"] > 0 and
                     row["ready_target_cars"] > 0 and
                     row["ready_source_dogs"] > 0),
                    (env.SINGLE_CAR, not row["cross_floor"] and
                     row["ready_source_cars"] > 0),
                ) if enabled})
            for row in rows),
        "actions_reversibly_decoded": len(rows) == 20,
        "stage11_cleanup_preserved": (
            not env.resource_claims and not env.queue.waiting
            and not env.queue.pending_arrivals
            and all(not runtime.robot.cargo_id and not runtime.robot.task_id
                    for runtime in env.robots.values())),
        "abnormal_action_mask_100_percent": all(
            probe["rejected"] for probe in probes),
    }
    summary = {
        "stage": 12,
        "seed": seed,
        "profile": profile,
        "trial_count": len(rows),
        "success_count": len(rows) if all(assertions.values()) else 0,
        "failure_count": 0 if all(assertions.values()) else 1,
        "observation_size": encoder.OBSERVATION_SIZE,
        "action_tensor_shape": [encoder.MAX_ACTIONS, encoder.ACTION_WIDTH],
        "task_types": dict(counts),
        "transport_modes": dict(mode_counts),
        "robot_speeds_mps": dict(env.robot_speeds),
        "min_legal_actions": min(row["legal_action_count"] for row in rows),
        "max_legal_actions": max(row["legal_action_count"] for row in rows),
        "min_cross_floor_stair_candidates": min(
            row["stair_candidate_count"] for row in rows
            if row["cross_floor"]),
        "abnormal_mask_probes": probes,
        "assertions": assertions,
    }
    packed = {key: np.asarray(value) for key, value in samples.items()}
    return rows, summary, packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--profile", choices=("normal", "medium", "overload"),
                        default="medium")
    args = parser.parse_args()
    rows, summary, samples = run(args.seed, args.profile)
    output = WORK_ROOT / "results" / "stage12"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"stage12_encoding_{args.profile}_20"
    (output / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (output / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    np.savez_compressed(output / f"{stem}.samples.npz", **samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(summary["assertions"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
