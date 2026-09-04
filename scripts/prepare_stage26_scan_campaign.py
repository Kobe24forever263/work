#!/usr/bin/env python3
"""Freeze ten sequential Stage 26 decisions and SCAN waypoint files."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from math import dist
from pathlib import Path
import random
import secrets
import sys

import numpy as np  # Bind Apple's numeric runtime before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment, AsyncTaskQueue, QueuedTask
from warehouse_core.stage14_training import build_balanced_arrivals
from warehouse_core.stage25_causal_observation import (
    CausalReleasedWarehouseDispatchGymEnv)
from warehouse_core.stage26_recurrent_ppo import load_recurrent_checkpoint

from prepare_stage26_random_single_task import (
    assignment_dict, best_completed_seed, corridor_path, stair_path,
    unique_path)


ALL_ROBOTS = (
    "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4",
    "dog_1", "dog_2", "dog_3", "dog_4",
    "car_f2_1", "car_f2_2",
)


def _route(start, target):
    return unique_path(corridor_path(tuple(start), tuple(target)))


def _retrace_to_standby(*completed_paths, standby):
    """Return over already validated legs before entering a standby route.

    SCAN is sensitive to very short first legs followed by a 90-degree turn.
    A cargo-delivery leg has already demonstrated a dynamically feasible exit
    from the platform, so cleanup should reverse that evidence-backed route
    instead of synthesising a new corner next to the platform.
    """
    route = []
    for completed in reversed(completed_paths):
        reversed_path = list(reversed(completed))
        if route and reversed_path and route[-1] == reversed_path[0]:
            route.extend(reversed_path[1:])
        else:
            route.extend(reversed_path)
    if not route:
        raise ValueError("at least one completed route is required")
    if dist(route[-1], standby) >= 1e-4:
        tail = _route(route[-1], standby)
        route.extend(tail[1:])
    return unique_path(route)


def _add_segment(output, label, robot, path, cargo_state, speed):
    path = unique_path(path)
    if len(path) < 2 or dist(path[0], path[-1]) < 1e-4:
        return
    output.append({
        "label": label,
        "robot": robot,
        "path": [list(point) for point in path],
        "cargo_state": cargo_state,
        "speed_mps": float(speed),
    })


def build_execution_segments(dispatch_before, task, assignment, final_poses,
                             success):
    """Map one high-level assignment to sequential real SCAN legs."""
    initial = {name: tuple(runtime.xyz)
               for name, runtime in dispatch_before.robots.items()}
    source = tuple(task.source.xyz)
    target = tuple(task.target.xyz)
    mode = assignment.transport_mode
    segments = []
    participants = []

    if mode == "SINGLE_CAR":
        car = assignment.pickup_carter
        participants = [car]
        approach = _route(initial[car], source)
        delivery = _route(source, target)
        _add_segment(segments, "车辆前往装货平台", car,
                     approach, "PICKUP", 2.0)
        _add_segment(segments, "货物吸附，车辆送往目标平台", car,
                     delivery, car, 2.0)
        _add_segment(segments, "货物卸下，车辆清场", car,
                     _retrace_to_standby(
                         approach, delivery, standby=final_poses[car]),
                     "DELIVERED" if success else "FAILED", 2.0)

    elif mode == "SINGLE_DOG":
        dog = assignment.dog_id
        participants = [dog]
        approach = _route(initial[dog], source)
        _add_segment(segments, "Go2 前往装货平台", dog,
                     approach, "PICKUP", 0.8)
        if task.source.floor == task.target.floor:
            delivery = _route(source, target)
            _add_segment(segments, "货物吸附，Go2 单独送货", dog,
                         delivery, dog, 0.8)
            cleanup = _retrace_to_standby(
                approach, delivery, standby=final_poses[dog])
        else:
            stair = dispatch_before.stairs[assignment.stair_id]
            entry = tuple(stair[f"floor{task.source.floor}_xyz"])
            direction = "up" if task.target.floor > task.source.floor else "down"
            stair_points = stair_path(assignment.stair_id, direction)
            _add_segment(segments, "Go2 携货前往楼梯入口", dog,
                         _route(source, entry), dog, 0.8)
            _add_segment(segments, "Go2 携货通过楼梯", dog,
                         [entry] + stair_points, dog, 0.8)
            delivery = _route(stair_points[-1], target)
            _add_segment(segments, "Go2 离开楼梯并送达平台", dog,
                         delivery, dog, 0.8)
            cleanup = _retrace_to_standby(
                delivery, standby=final_poses[dog])
        _add_segment(segments, "货物卸下，Go2 前往待命位", dog,
                     cleanup,
                     "DELIVERED" if success else "FAILED", 0.8)

    elif mode == "CAR_DOG_CAR":
        source_car = assignment.pickup_carter
        dog = assignment.dog_id
        target_car = assignment.receiving_carter
        participants = [source_car, dog, target_car]
        stair = dispatch_before.stairs[assignment.stair_id]
        entry = tuple(stair[f"floor{task.source.floor}_xyz"])
        direction = "up" if task.target.floor > task.source.floor else "down"
        stair_points = stair_path(assignment.stair_id, direction)
        visual_exit = tuple(stair_points[-1])
        dog_approach = _route(initial[dog], entry)
        target_car_approach = _route(initial[target_car], visual_exit)
        source_car_approach = _route(initial[source_car], source)
        source_car_loaded = _route(source, entry)
        _add_segment(segments, "Go2 预定位至源层楼梯口", dog,
                     dog_approach, "PICKUP", 0.8)
        _add_segment(segments, "目标层车辆预定位至接收点", target_car,
                     target_car_approach, "PICKUP", 2.0)
        _add_segment(segments, "源层车辆前往装货平台", source_car,
                     source_car_approach, "PICKUP", 2.0)
        _add_segment(segments, "货物吸附，源层车辆前往交接点", source_car,
                     source_car_loaded, source_car, 2.0)
        _add_segment(segments, "车—狗交接完成，Go2 通过楼梯", dog,
                     [entry] + stair_points, dog, 0.8)
        if success:
            target_car_delivery = _route(visual_exit, target)
            _add_segment(segments, "狗—车交接完成，目标层车辆送货", target_car,
                         target_car_delivery, target_car, 2.0)
        cleanup_routes = {
            source_car: _retrace_to_standby(
                source_car_approach, source_car_loaded,
                standby=final_poses[source_car]),
            dog: _route(visual_exit, final_poses[dog]),
            target_car: (_retrace_to_standby(
                target_car_approach, target_car_delivery,
                standby=final_poses[target_car]) if success else
                _retrace_to_standby(
                    target_car_approach, standby=final_poses[target_car])),
        }
        for robot in (source_car, dog, target_car):
            _add_segment(segments, f"{robot} 前往待命位", robot,
                         cleanup_routes[robot],
                         "DELIVERED" if success else "FAILED",
                         0.8 if robot.startswith("dog_") else 2.0)
    else:
        raise ValueError(f"unsupported mode {mode}")

    return segments, participants


def write_scan_routes(campaign, route_dir: Path):
    route_dir.mkdir(parents=True, exist_ok=True)
    for task_index, task in enumerate(campaign["tasks"], start=1):
        for segment_index, segment in enumerate(task["segments"], start=1):
            waypoints = segment["path"][1:]
            if not waypoints:
                waypoints = segment["path"][-1:]
            flat = [float(value) for point in waypoints for value in point]
            route = route_dir / (
                f"task_{task_index:02d}_segment_{segment_index:02d}_"
                f"{segment['robot']}.yaml")
            route.write_text(yaml.safe_dump({
                "scan_planner_node": {"ros__parameters": {
                    "fsm.navi_mode": 2,
                    "fsm.waypoints": flat,
                }}}, sort_keys=False), encoding="utf-8")
            segment["scan_route_file"] = str(route.resolve())
            segment["initial_xyz"] = segment["path"][0]
            segment["goal_xyz"] = segment["path"][-1]
            length = sum(dist(a, b) for a, b in zip(
                segment["path"], segment["path"][1:]))
            segment["route_length_m"] = length
            segment["execution_backend"] = (
                "POSE_SYNC" if length < 2.0 else "SCAN")
            segment["timeout_s"] = max(
                75.0, min(420.0, length / segment["speed_mps"] * 4.0 + 45.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-seed", type=int)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "results" / "stage26_recurrent_causal" /
        "rviz_scan_campaign" / "latest_campaign.json"))
    args = parser.parse_args()
    if args.task_count != 10:
        raise ValueError("Stage 26 RViz acceptance is locked to 10 tasks")
    campaign_seed = (args.campaign_seed if args.campaign_seed is not None else
                     76000000 + secrets.randbelow(1_000_000))
    weight, training_summary, seed_candidates = best_completed_seed()
    device = torch.device("cpu")
    torch.set_num_threads(4)
    model, checkpoint = load_recurrent_checkpoint(weight, device)
    model.eval()
    memory = model.initial_memory(device=device).squeeze(0)

    env = CausalReleasedWarehouseDispatchGymEnv(
        seed=campaign_seed, execution_mode="SERIAL",
        arrival_schedule="BALANCED", handover_sampling="TASK_KEYED",
        reward_contract="CONTINUOUS_TIME_V3")
    env.reset(seed=campaign_seed)
    dispatch = env.dispatch
    dispatch.task_limit = 100
    dispatch.time_limit = 100000.0
    pool = build_balanced_arrivals(campaign_seed)
    order = list(range(len(pool)))
    random.Random(campaign_seed + 26).shuffle(order)
    remaining = [pool[index] for index in order]
    campaign_tasks = []
    total_waits = 0

    while remaining and len(campaign_tasks) < args.task_count:
        chosen_position = None
        for position, (queued, task_type) in enumerate(remaining):
            dispatch.queue = AsyncTaskQueue((
                QueuedTask(queued.task, dispatch.now),))
            dispatch.queue.advance(dispatch.now)
            env.task_types = {queued.task.task_id: task_type}
            env.decision = env._encode_decision()
            if any(isinstance(item, Assignment) and legal
                   for item, legal in zip(
                       env.decision.action_ids, env.decision.action_mask)):
                chosen_position = position
                break
        if chosen_position is None:
            raise RuntimeError(
                "no feasible task remains for the persistent fleet state")
        queued, task_type = remaining.pop(chosen_position)
        task = queued.task
        dispatch.queue = AsyncTaskQueue((QueuedTask(task, dispatch.now),))
        dispatch.queue.advance(dispatch.now)
        env.task_types = {task.task_id: task_type}
        waits = 0
        while True:
            env.decision = env._encode_decision()
            observation = env._observation()
            mask = env.action_masks()
            state = torch.as_tensor(
                observation["state"], dtype=torch.float32)
            features = torch.as_tensor(
                observation["action_features"][..., :model.action_width],
                dtype=torch.float32)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
            with torch.no_grad():
                logits, value, next_memory = model(
                    state, features, mask_tensor, memory)
                probabilities = torch.distributions.Categorical(
                    logits=logits).probs
                action = int(torch.argmax(logits).item())
            selected = env.decision.action_ids[action]
            memory_delta = float(torch.linalg.vector_norm(
                next_memory - memory).item())
            memory = next_memory
            if isinstance(selected, Assignment):
                break
            waits += 1
            total_waits += 1
            dispatch.now += dispatch.decision_gap
            if waits >= 20:
                raise RuntimeError(
                    f"policy selected WAIT 20 times for {task.task_id}")

        rule = env.decision.action_ids[env.rule_action()]
        before = deepcopy(dispatch)
        before_completed = dispatch.completed
        before_failed = dispatch.failed
        _, reward, _, _, info = env.step(action)
        success = dispatch.completed > before_completed
        if not success and dispatch.failed <= before_failed:
            raise RuntimeError("dispatch transition did not resolve the task")
        final_poses = {name: tuple(runtime.xyz)
                       for name, runtime in dispatch.robots.items()}
        segments, participants = build_execution_segments(
            before, task, selected, final_poses, success)
        campaign_tasks.append({
            "sequence": len(campaign_tasks) + 1,
            "task_type": task_type,
            "task": {
                "task_id": task.task_id,
                "cargo_id": task.cargo_id,
                "priority": task.priority,
                "deadline": task.deadline,
                "source": asdict(task.source),
                "target": asdict(task.target),
            },
            "policy_decision": {
                "action_index": action,
                "probability": float(probabilities[action].item()),
                "value_estimate": float(value.item()),
                "memory_update_l2": memory_delta,
                "wait_decisions_before_assignment": waits,
                "assignment": assignment_dict(selected),
                "rule_reference": assignment_dict(rule),
                "agrees_with_rule_mode": (
                    selected.transport_mode == rule.transport_mode),
            },
            "participants": participants,
            "outcome": "COMPLETED" if success else "FAILED",
            "simulated_reward": reward,
            "simulated_delta_time": info.get("delta_time", 0.0),
            "initial_robot_poses": {
                name: list(before.robots[name].xyz) for name in ALL_ROBOTS},
            "final_robot_poses": {
                name: list(final_poses[name]) for name in ALL_ROBOTS},
            "segments": segments,
        })

    if len(campaign_tasks) != args.task_count:
        raise RuntimeError(
            f"generated only {len(campaign_tasks)} of {args.task_count} tasks")
    output = args.output.expanduser().resolve()
    route_dir = output.parent / "routes"
    campaign = {
        "schema_version": "warehouse_stage26_scan_campaign_v1",
        "scope": "TEN_SEQUENTIAL_TASKS",
        "task_count": args.task_count,
        "continuous_task_publication": True,
        "maximum_active_tasks": 1,
        "robot_position_reset_between_tasks": False,
        "recurrent_memory_reset_between_tasks": False,
        "campaign_seed": campaign_seed,
        "checkpoint_selection": {
            "criterion": (
                "maximum final evaluation mean_reward among passed "
                "completed Stage 26 seeds"),
            "selected_seed_index": int(weight.parent.name.split("_")[-1]),
            "checkpoint": str(weight),
            "selected_seed_mean_reward": float(
                training_summary["evaluation"]["mean_reward"]),
            "candidate_seeds": seed_candidates,
            "architecture": checkpoint["model_metadata"]["architecture"],
        },
        "robot_ids": list(ALL_ROBOTS),
        "total_policy_wait_decisions": total_waits,
        "tasks": campaign_tasks,
    }
    write_scan_routes(campaign, route_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "campaign": str(output),
        "campaign_seed": campaign_seed,
        "task_count": len(campaign_tasks),
        "selected_seed_index": campaign["checkpoint_selection"][
            "selected_seed_index"],
        "modes": [item["policy_decision"]["assignment"]["transport_mode"]
                  for item in campaign_tasks],
        "task_types": [item["task_type"] for item in campaign_tasks],
        "outcomes": [item["outcome"] for item in campaign_tasks],
        "total_policy_wait_decisions": total_waits,
        "scan_route_count": sum(
            segment["execution_backend"] == "SCAN"
            for item in campaign_tasks for segment in item["segments"]),
        "pose_sync_count": sum(
            segment["execution_backend"] == "POSE_SYNC"
            for item in campaign_tasks for segment in item["segments"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
