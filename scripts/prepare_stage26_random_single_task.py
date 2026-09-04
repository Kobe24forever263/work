#!/usr/bin/env python3
"""Select one random warehouse task with the best completed Stage 26 seed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import dist
from pathlib import Path
import random
import secrets
import sys

import numpy as np  # Bind the Apple runtime before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import (
    Assignment, AsyncTaskQueue, QueuedTask)
from warehouse_core.stage14_training import build_balanced_arrivals
from warehouse_core.stage25_causal_observation import (
    CausalReleasedWarehouseDispatchGymEnv)
from warehouse_core.stage26_recurrent_ppo import load_recurrent_checkpoint


MISSION_DEFAULTS = {
    "ne": ("car_f1_1", "dog_1", "car_f2_2"),
    "nw": ("car_f1_2", "dog_2", "car_f2_1"),
    "sw": ("car_f1_3", "dog_3", "car_f2_1"),
    "se": ("car_f1_4", "dog_4", "car_f2_2"),
}
STAIR_SUFFIX = {
    "STAIR_NE": "ne", "STAIR_NW": "nw",
    "STAIR_SW": "sw", "STAIR_SE": "se",
}
CANONICAL_DOG = {
    "ne": "dog_1", "nw": "dog_2", "sw": "dog_3", "se": "dog_4",
}
ALL_ROBOTS = (
    "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4",
    "dog_1", "dog_2", "dog_3", "dog_4",
    "car_f2_1", "car_f2_2",
)


def best_completed_seed() -> tuple[Path, dict, list[dict]]:
    candidates = []
    pattern = (ROOT / "results" / "stage26_recurrent_causal" / "long").glob(
        "seed_*/stage26_recurrent_ppo.summary.json")
    for summary_path in sorted(pattern):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        weight = Path(summary.get("final_weight", ""))
        if not (summary.get("passed") is True and
                int(summary.get("updates", 0)) >= 2000 and weight.exists()):
            continue
        validation_path = next((
            path for path in (
                summary_path.parent / "sentinel_fresh_validation.json",
                summary_path.parent / "fresh_validation.json",
            ) if path.is_file()), None)
        if validation_path is None:
            continue
        validation = json.loads(
            validation_path.read_text(encoding="utf-8"))
        if validation.get("passed") is not True:
            continue
        seed_index = int(summary_path.parent.name.split("_")[-1])
        candidates.append({
            "seed_index": seed_index,
            "summary": str(summary_path),
            "weight": str(weight),
            "fresh_validation": str(validation_path),
            "mean_reward": float(validation["policy"]["mean_reward"]),
            "success_rate": float(validation["policy"]["success_rate"]),
            "passed": True,
        })
    if not candidates:
        raise FileNotFoundError(
            "no completed Stage 26 seed with passed fresh validation was found")
    selected = max(candidates, key=lambda item: (
        item["mean_reward"], item["success_rate"], -item["seed_index"]))
    summary = json.loads(Path(selected["summary"]).read_text(encoding="utf-8"))
    return Path(selected["weight"]), summary, candidates


def unique_path(points):
    output = []
    for point in points:
        value = tuple(float(item) for item in point)
        if not output or dist(output[-1], value) > 1e-6:
            output.append(value)
    return output


def corridor_path(start, target):
    """Simple orthogonal replay path between already validated task points."""
    start = tuple(start)
    target = tuple(target)
    middle_x = (target[0], start[1], start[2])
    middle_z = (target[0], start[1], target[2])
    return unique_path((start, middle_x, middle_z, target))


def stair_path(stair_id: str, direction: str):
    suffix = STAIR_SUFFIX[stair_id]
    dog = CANONICAL_DOG[suffix]
    path = (ROOT / "src" / "warehouse_bringup" / "config" /
            "scan_planner" / f"{dog}_stair_{suffix}_{direction}.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = data["scan_planner_node"]["ros__parameters"]["fsm.waypoints"]
    return [tuple(float(value) for value in values[index:index + 3])
            for index in range(0, len(values), 3)]


def nearest_standby(dispatch, robot_id: str, xyz, floor: int | None = None):
    runtime = dispatch.robots[robot_id]
    target_floor = floor or runtime.robot.current_floor
    choices = [
        slot for slot in dispatch.standby_slots.values()
        if slot.robot_type == runtime.robot.robot_type and
        slot.floor == target_floor]
    if not choices:
        return tuple(xyz)
    return tuple(min(choices, key=lambda slot: dist(slot.xyz, xyz)).xyz)


def quadrant(xyz) -> str:
    x, y = xyz[:2]
    return ("n" if y >= 0 else "s") + ("e" if x >= 0 else "w")


def assignment_dict(item: Assignment) -> dict:
    return asdict(item)


def make_single_task_snapshot_feasible(dispatch, task) -> dict:
    """Restore a plausible persistent fleet state for an isolated down task."""
    adjustments = {}
    if task.source.floor == 2 and task.target.floor == 1:
        floor2_dogs = {
            "dog_3": (-8.8, -6.0, 4.7),
            "dog_4": (8.8, -6.0, 4.7),
        }
        for robot_id, xyz in floor2_dogs.items():
            runtime = dispatch.robots[robot_id]
            runtime.robot.current_floor = 2
            runtime.robot.current_region = (
                "F2_WEST" if xyz[0] < 0 else "F2_EAST")
            runtime.xyz = xyz
            adjustments[robot_id] = {
                "current_floor": 2,
                "current_region": runtime.robot.current_region,
                "xyz": list(xyz),
                "reason": "feasible persistent snapshot for isolated cross_down",
            }
    return adjustments


def build_segments(dispatch, task, assignment: Assignment):
    mode = assignment.transport_mode
    initial = {name: tuple(runtime.xyz)
               for name, runtime in dispatch.robots.items()}
    source = tuple(task.source.xyz)
    target = tuple(task.target.xyz)
    segments = []

    def add(label, robot, path, owner, speed, pause=0.8):
        segments.append({
            "label": label, "robot": robot,
            "path": [list(point) for point in unique_path(path)],
            "cargo_owner": owner, "speed_mps": speed,
            "pause_after_s": pause,
        })

    if mode == "SINGLE_CAR":
        car = assignment.pickup_carter
        add("车辆前往装货平台", car,
            corridor_path(initial[car], source), "PICKUP", 2.0)
        add("货物吸附，车辆送往目标平台", car,
            corridor_path(source, target), car, 2.0, 1.0)
        standby = nearest_standby(dispatch, car, target)
        add("货物卸下，车辆驶向待命泊位", car,
            corridor_path(target, standby), "DELIVERED", 2.0, 1.0)
    elif mode == "SINGLE_DOG":
        dog = assignment.dog_id
        add("Go2 前往装货平台", dog,
            corridor_path(initial[dog], source), "PICKUP", 0.8)
        if task.source.floor == task.target.floor:
            delivery_path = corridor_path(source, target)
        else:
            stair = dispatch.stairs[assignment.stair_id]
            entry = tuple(stair[f"floor{task.source.floor}_xyz"])
            direction = "up" if task.target.floor > task.source.floor else "down"
            stairs = stair_path(assignment.stair_id, direction)
            delivery_path = unique_path(
                corridor_path(source, entry) + stairs + [target])
        add("货物吸附，Go2 独立送往目标平台", dog,
            delivery_path, dog, 0.8, 1.0)
        standby = nearest_standby(
            dispatch, dog, target, floor=task.target.floor)
        add("货物卸下，Go2 前往待命泊位", dog,
            corridor_path(target, standby), "DELIVERED", 0.8, 1.0)
    elif mode == "CAR_DOG_CAR":
        source_car = assignment.pickup_carter
        dog = assignment.dog_id
        target_car = assignment.receiving_carter
        stair = dispatch.stairs[assignment.stair_id]
        entry = tuple(stair[f"floor{task.source.floor}_xyz"])
        direction = "up" if task.target.floor > task.source.floor else "down"
        stairs = stair_path(assignment.stair_id, direction)
        visual_exit = tuple(stairs[-1])
        add("Go2 预定位至楼梯交接点", dog,
            corridor_path(initial[dog], entry), "PICKUP", 0.8, 0.5)
        add("目标层车辆预定位至接收点", target_car,
            corridor_path(initial[target_car], visual_exit),
            "PICKUP", 2.0, 0.5)
        add("源层车辆前往装货平台", source_car,
            corridor_path(initial[source_car], source), "PICKUP", 2.0)
        add("货物吸附，源层车辆前往楼梯", source_car,
            corridor_path(source, entry), source_car, 2.0, 1.0)
        add("第一次交接完成，Go2 通过楼梯", dog,
            [entry] + stairs, dog, 0.8, 1.0)
        add("第二次交接完成，目标层车辆送货", target_car,
            corridor_path(visual_exit, target), target_car, 2.0, 1.0)
        standby = nearest_standby(dispatch, target_car, target)
        add("货物卸下，目标层车辆驶向待命泊位", target_car,
            corridor_path(target, standby), "DELIVERED", 2.0, 1.0)
    else:
        raise ValueError(f"unsupported transport mode: {mode}")
    return segments


def display_robots(dispatch, task, assignment: Assignment):
    suffix = STAIR_SUFFIX.get(assignment.stair_id, quadrant(task.source.xyz))
    floor1_car, dog, floor2_car = MISSION_DEFAULTS[suffix]
    for robot_id in (assignment.pickup_carter,
                     assignment.receiving_carter):
        if not robot_id:
            continue
        floor = dispatch.robots[robot_id].robot.current_floor
        if floor == 1:
            floor1_car = robot_id
        else:
            floor2_car = robot_id
    if assignment.dog_id:
        dog = assignment.dog_id
    return {
        "floor1_car": floor1_car, "dog": dog,
        "floor2_car": floor2_car,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-seed", type=int)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "results" / "stage26_recurrent_causal" /
        "rviz_single_task" / "latest_plan.json"))
    args = parser.parse_args()
    task_seed = (args.task_seed if args.task_seed is not None else
                 74000000 + secrets.randbelow(1_000_000))
    weight, training_summary, seed_candidates = best_completed_seed()
    device = torch.device("cpu")
    torch.set_num_threads(4)
    model, checkpoint = load_recurrent_checkpoint(weight, device)
    model.eval()

    pool = build_balanced_arrivals(task_seed)
    pool_index = random.Random(task_seed + 26).randrange(len(pool))
    selected_queued, task_type = pool[pool_index]
    task = selected_queued.task
    env = CausalReleasedWarehouseDispatchGymEnv(
        seed=task_seed, execution_mode="CONCURRENT",
        arrival_schedule="BALANCED", handover_sampling="TASK_KEYED",
        reward_contract="CONTINUOUS_TIME_V3")
    env.reset(seed=task_seed)
    dispatch = env.dispatch
    dispatch.queue = AsyncTaskQueue((QueuedTask(task, 0.0),))
    dispatch.queue.advance(dispatch.now)
    dispatch.task_limit = 1
    snapshot_adjustments = make_single_task_snapshot_feasible(dispatch, task)
    env.task_types = {task.task_id: task_type}
    env.decision = env._encode_decision()
    observation = env._observation()
    mask = env.action_masks()
    memory = model.initial_memory(device=device).squeeze(0)
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    features = torch.as_tensor(
        observation["action_features"][..., :model.action_width],
        dtype=torch.float32)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    with torch.no_grad():
        logits, value, next_memory = model(
            state, features, mask_tensor, memory)
        distribution = torch.distributions.Categorical(logits=logits)
        action = int(torch.argmax(logits).item())
        probabilities = distribution.probs
    selected = env.decision.action_ids[action]
    if not isinstance(selected, Assignment):
        raise RuntimeError("best Stage 26 policy selected WAIT for a free fleet")
    rule_index = env.rule_action()
    rule = env.decision.action_ids[rule_index]
    legal_indices = torch.nonzero(mask_tensor, as_tuple=False).flatten()
    top_indices = legal_indices[torch.argsort(
        probabilities[legal_indices], descending=True)[:5]].tolist()
    alternatives = []
    for index in top_indices:
        candidate = env.decision.action_ids[index]
        if not isinstance(candidate, Assignment):
            continue
        alternatives.append({
            "action_index": index,
            "probability": float(probabilities[index].item()),
            "assignment": assignment_dict(candidate),
        })

    display = display_robots(dispatch, task, selected)
    selected_ids = list(dict.fromkeys(item for item in (
        selected.pickup_carter, selected.dog_id,
        selected.receiving_carter) if item))
    plan = {
        "schema_version": "warehouse_stage26_rviz_single_task_v2",
        "scope": "ONE_RANDOM_TASK_ONLY",
        "continuous_task_publication": False,
        "robot_ids": list(ALL_ROBOTS),
        "task_seed": task_seed,
        "task_pool_size": len(pool),
        "task_pool_index": pool_index,
        "task_type": task_type,
        "persistent_snapshot_adjustments": snapshot_adjustments,
        "task": {
            "task_id": task.task_id, "cargo_id": task.cargo_id,
            "priority": task.priority, "deadline": task.deadline,
            "source": asdict(task.source), "target": asdict(task.target),
        },
        "checkpoint_selection": {
            "criterion": "maximum final evaluation mean_reward among passed completed seeds",
            "selected_seed_index": int(weight.parent.name.split("_")[-1]),
            "checkpoint": str(weight),
            "selected_seed_mean_reward": float(
                training_summary["evaluation"]["mean_reward"]),
            "candidate_seeds": seed_candidates,
            "architecture": checkpoint["model_metadata"]["architecture"],
        },
        "policy_decision": {
            "action_index": action,
            "probability": float(probabilities[action].item()),
            "value_estimate": float(value.item()),
            "memory_update_l2": float(torch.linalg.vector_norm(
                next_memory - memory).item()),
            "assignment": assignment_dict(selected),
            "rule_reference": assignment_dict(rule),
            "agrees_with_rule_mode": (
                selected.transport_mode == rule.transport_mode),
            "top_legal_actions": alternatives,
        },
        "selected_robots": selected_ids,
        "display_robots": display,
        "robot_initial_poses": {
            name: list(dispatch.robots[name].xyz)
            for name in ALL_ROBOTS},
        "cargo_pickup_xyz": list(task.source.xyz),
        "cargo_delivery_xyz": list(task.target.xyz),
        "segments": build_segments(dispatch, task, selected),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "plan": str(output), "task_seed": task_seed,
        "task_type": task_type, "source": task.source.point_id,
        "target": task.target.point_id,
        "selected_seed_index": plan["checkpoint_selection"][
            "selected_seed_index"],
        "transport_mode": selected.transport_mode,
        "selected_robots": selected_ids,
        "policy_probability": plan["policy_decision"]["probability"],
        "agrees_with_rule_mode": plan["policy_decision"][
            "agrees_with_rule_mode"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
