#!/usr/bin/env python3
"""Audit the 284-D Stage 20/22 observation for Markov/SMDP sufficiency."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage12_encoding import Stage12Encoder
from warehouse_core.stage14_training import build_stage14_environment


OUTPUT_ROOT = ROOT / "results" / "stage23_state_sufficiency"
JSON_OUTPUT = OUTPUT_ROOT / "stage23_state_sufficiency_audit.json"
CSV_OUTPUT = OUTPUT_ROOT / "stage23_observation_284_variable_map.csv"
V3_CSV_OUTPUT = OUTPUT_ROOT / "stage23_markov_v3_1720_variable_map.csv"
SOURCE = (ROOT / "src" / "warehouse_core" / "warehouse_core" /
          "stage12_encoding.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variable_map() -> list[dict]:
    rows: list[dict] = []
    global_fields = (
        ("normalized_time", "env.now / env.time_limit"),
        ("waiting_count", "len(queue.waiting) / 50"),
        ("pending_arrival_count", "min(len(pending), 20) / 50"),
        ("active_task_ratio", "len(active_tasks) / max_concurrent_tasks"),
        ("is_concurrent", "execution_mode == CONCURRENT"),
        ("completed_ratio", "completed / task_limit"),
        ("failed_ratio", "failed / task_limit"),
        ("decision_gap", "decision_gap / 10"),
    )
    for index, (field, expression) in enumerate(global_fields):
        rows.append({
            "index": index, "block": "global", "slot": "",
            "field": field, "source_expression": expression,
            "causal_scope": "aggregate_or_clock"})

    task_fields = (
        ("present", "1"), ("source_floor", "floor / 2"),
        ("target_floor", "floor / 2"),
        ("cross_floor", "source.floor != target.floor"),
        ("priority", "priority / 3"),
        ("waiting_age", "max(0, now-arrival) / 600"),
        ("deadline_slack", "max(0, deadline-now) / 600"),
        ("source_x", "x / 25"), ("source_y", "y / 25"),
        ("source_z", "z / 5"), ("target_x", "x / 25"),
        ("target_y", "y / 25"), ("target_z", "z / 5"),
    )
    start = Stage12Encoder.GLOBAL_WIDTH
    for slot in range(Stage12Encoder.MAX_TASKS):
        for offset, (field, expression) in enumerate(task_fields):
            rows.append({
                "index": start + slot * Stage12Encoder.TASK_WIDTH + offset,
                "block": "visible_waiting_task", "slot": slot,
                "field": field, "source_expression": expression,
                "causal_scope": "top_8_priority_deadline_age_task_id_order"})

    robot_fields = (
        ("present", "1"), ("is_dog", "robot_type == dog"),
        ("nominal_speed", "speed / 2"),
        ("current_floor", "floor / 2"),
        ("available", "available"), ("faulted", "bool(failure_code)"),
        ("loaded", "bool(cargo_id)"), ("assigned", "bool(task_id)"),
        ("x", "x / 25"), ("y", "y / 25"), ("z", "z / 5"),
        ("distance_total", "distance_total / 1000"),
        ("tasks_completed", "tasks_completed / 50"),
        ("idle_duration", "max(0, now-idle_since) / 600"),
    )
    start += Stage12Encoder.MAX_TASKS * Stage12Encoder.TASK_WIDTH
    for slot in range(Stage12Encoder.MAX_ROBOTS):
        for offset, (field, expression) in enumerate(robot_fields):
            rows.append({
                "index": start + slot * Stage12Encoder.ROBOT_WIDTH + offset,
                "block": "robot", "slot": slot, "field": field,
                "source_expression": expression,
                "causal_scope": "sorted_robot_id"})

    stair_fields = (
        ("present", "1"), ("free", "state == FREE"),
        ("floor1_x", "x / 25"), ("floor1_y", "y / 25"),
        ("floor1_z", "z / 5"), ("floor2_x", "x / 25"),
        ("floor2_y", "y / 25"), ("floor2_z", "z / 5"),
    )
    start += Stage12Encoder.MAX_ROBOTS * Stage12Encoder.ROBOT_WIDTH
    for slot in range(Stage12Encoder.MAX_STAIRS):
        for offset, (field, expression) in enumerate(stair_fields):
            rows.append({
                "index": start + slot * Stage12Encoder.STAIR_WIDTH + offset,
                "block": "stair", "slot": slot, "field": field,
                "source_expression": expression,
                "causal_scope": "sorted_stair_id"})
    if [row["index"] for row in rows] != list(range(284)):
        raise RuntimeError("observation variable map is not exactly 0..283")
    return rows


def markov_v3_variable_map(legacy_rows: list[dict]) -> list[dict]:
    rows = [dict(row) for row in legacy_rows]
    task_fields = (
        "present", "waiting", "pending", "active", "source_floor",
        "target_floor", "cross_floor", "priority", "release_relative_time",
        "deadline_relative_time", "source_x", "source_y", "source_z",
        "target_x", "target_y", "target_z")
    start = 284
    for slot in range(Stage12Encoder.MARKOV_MAX_TASKS):
        for offset, field in enumerate(task_fields):
            rows.append({
                "index": start + slot * Stage12Encoder.MARKOV_TASK_WIDTH + offset,
                "block": "markov_unresolved_task", "slot": slot,
                "field": field, "source_expression": "see Stage12Encoder._markov_v3_context",
                "causal_scope": "all_waiting_pending_active_tasks"})
    active_fields = [
        "present", "task_slot", "remaining_time", "elapsed_time",
        "scheduled_duration", "mode_single_car", "mode_single_dog",
        "mode_car_dog_car", "pickup_robot_slot", "dog_robot_slot",
        "receiving_robot_slot", "stair_slot", "resource_count",
        "participant_count", "travelled", "handover_count",
        "handover_duration_total", "finish_deadline_margin"]
    for role in ("pickup", "dog", "receiver"):
        active_fields.extend([
            f"{role}_final_present", f"{role}_final_floor",
            f"{role}_final_x", f"{role}_final_y", f"{role}_final_z",
            f"{role}_final_tasks_completed", f"{role}_final_standby_slot"])
    start += (Stage12Encoder.MARKOV_MAX_TASKS *
              Stage12Encoder.MARKOV_TASK_WIDTH)
    for slot in range(Stage12Encoder.MARKOV_MAX_ACTIVE):
        for offset, field in enumerate(active_fields):
            rows.append({
                "index": start + slot * Stage12Encoder.MARKOV_ACTIVE_WIDTH + offset,
                "block": "markov_active_assignment", "slot": slot,
                "field": field, "source_expression": "see Stage12Encoder._markov_v3_context",
                "causal_scope": "all_active_completion_events_and_planned_final_states"})
    if [row["index"] for row in rows] != list(range(1720)):
        raise RuntimeError("Markov V3 variable map is not exactly 0..1719")
    return rows


def same_decision(left, right) -> dict:
    return {
        "state_equal": bool(np.array_equal(
            left.observation, right.observation)),
        "action_features_equal": bool(np.array_equal(
            left.action_features, right.action_features)),
        "action_mask_equal": bool(np.array_equal(
            left.action_mask, right.action_mask)),
        "legal_action_ids_equal": [
            str(item) for item, legal in zip(left.action_ids, left.action_mask)
            if legal] == [
            str(item) for item, legal in zip(right.action_ids, right.action_mask)
            if legal],
    }


def active_residual_alias(seed: int) -> dict:
    env, _ = build_stage14_environment(
        seed, execution_mode="CONCURRENT", handover_sampling="TASK_KEYED",
        arrival_schedule="MIXED_CURRICULUM")
    candidates = env.enumerate_candidate_actions()
    mask = env.build_action_mask(candidates)
    legal = [(index, item) for index, (item, allowed) in
             enumerate(zip(candidates, mask)) if allowed]
    action_index, _ = max(legal, key=lambda pair: pair[1].estimated_cost)
    env.step(action_index)
    if not env.active_tasks:
        raise RuntimeError("probe assignment unexpectedly completed immediately")
    env.max_concurrent_tasks = len(env.active_tasks)
    left, right = deepcopy(env), deepcopy(env)
    active_id = sorted(left.active_tasks)[0]
    next_arrival_gap = min(
        (left.queue.pending_arrivals[0].arrival_time - left.now)
        if left.queue.pending_arrivals else 1.0, 1.0)
    small = max(1e-5, next_arrival_gap / 4.0)
    left.active_tasks[active_id].finish_at = left.now + small
    right.active_tasks[active_id].finish_at = left.now + 2.0 * small
    left_decision = Stage12Encoder().encode(left)
    right_decision = Stage12Encoder().encode(right)
    equality = same_decision(left_decision, right_decision)
    v3_equality = same_decision(
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(left),
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(right))
    left_transition = left.wait_for_next_event()
    right_transition = right.wait_for_next_event()
    outcome_differs = bool(
        not np.isclose(left_transition.delta_time, right_transition.delta_time)
        and not np.isclose(left_transition.discount, right_transition.discount))
    return {
        "name": "active_assignment_residual_time_alias",
        "seed": seed,
        "perturbation": (
            "Only ActiveAssignment.finish_at differs; all encoded fields, "
            "resource occupancy, robot flags and current clock are equal."),
        "same_encoded_decision": equality,
        "markov_v3_state_equal": v3_equality["state_equal"],
        "left_remaining_time_s": small,
        "right_remaining_time_s": 2.0 * small,
        "left_transition_delta_time_s": left_transition.delta_time,
        "right_transition_delta_time_s": right_transition.delta_time,
        "left_discount": left_transition.discount,
        "right_discount": right_transition.discount,
        "same_observation_action_has_different_transition": outcome_differs,
        "alias_detected": all(equality.values()) and outcome_differs,
    }


def pending_arrival_alias(seed: int) -> dict:
    env, _ = build_stage14_environment(
        seed, execution_mode="CONCURRENT", handover_sampling="TASK_KEYED",
        arrival_schedule="MIXED_CURRICULUM")
    env.queue.waiting.clear()
    if len(env.queue.pending_arrivals) < 2:
        raise RuntimeError("pending-arrival probe needs at least two arrivals")
    left, right = deepcopy(env), deepcopy(env)
    first_left = left.queue.pending_arrivals[0]
    first_right = right.queue.pending_arrivals[0]
    second_time = min(left.queue.pending_arrivals[1].arrival_time,
                      right.queue.pending_arrivals[1].arrival_time)
    upper = max(1e-3, second_time - env.now)
    first_gap = min(0.25, upper / 4.0)
    second_gap = min(0.75, upper / 2.0)
    if second_gap <= first_gap:
        second_gap = first_gap * 2.0
    left.queue.pending_arrivals[0] = replace(
        first_left, arrival_time=left.now + first_gap)
    right.queue.pending_arrivals[0] = replace(
        first_right, arrival_time=right.now + second_gap)
    left_decision = Stage12Encoder().encode(left)
    right_decision = Stage12Encoder().encode(right)
    equality = same_decision(left_decision, right_decision)
    v3_equality = same_decision(
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(left),
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(right))
    left_transition = left.wait_for_next_event()
    right_transition = right.wait_for_next_event()
    outcome_differs = bool(
        not np.isclose(left_transition.delta_time, right_transition.delta_time)
        and not np.isclose(left_transition.discount, right_transition.discount))
    return {
        "name": "next_arrival_residual_time_alias",
        "seed": seed,
        "perturbation": (
            "Only the first pending arrival timestamp differs; pending count "
            "and every currently encoded field are equal."),
        "same_encoded_decision": equality,
        "markov_v3_state_equal": v3_equality["state_equal"],
        "left_next_arrival_s": first_gap,
        "right_next_arrival_s": second_gap,
        "left_transition_delta_time_s": left_transition.delta_time,
        "right_transition_delta_time_s": right_transition.delta_time,
        "left_discount": left_transition.discount,
        "right_discount": right_transition.discount,
        "same_observation_action_has_different_transition": outcome_differs,
        "alias_detected": all(equality.values()) and outcome_differs,
    }


def hidden_queue_alias(seed: int) -> dict:
    env, _ = build_stage14_environment(
        seed, execution_mode="CONCURRENT", handover_sampling="TASK_KEYED",
        arrival_schedule="MIXED_CURRICULUM")
    arrival_time = env.queue.pending_arrivals[11].arrival_time
    env.now = arrival_time
    env.queue.advance(env.now)
    ranked = env.queue.policy_view(env.now, limit=len(env.queue.waiting))
    if len(ranked) < 9:
        raise RuntimeError("hidden-queue probe needs at least nine waiting tasks")
    hidden = ranked[8]
    left, right = deepcopy(env), deepcopy(env)
    replacement_point = replace(
        hidden.task.target,
        point_id=hidden.task.target.point_id + "_ALIAS",
        xyz=(hidden.task.target.xyz[0] + 7.0,
             hidden.task.target.xyz[1] - 5.0,
             hidden.task.target.xyz[2]))
    replacement_task = replace(hidden.task, target=replacement_point)
    for index, item in enumerate(right.queue.waiting):
        if item.task.task_id == hidden.task.task_id:
            right.queue.waiting[index] = replace(item, task=replacement_task)
            break
    left_decision = Stage12Encoder().encode(left)
    right_decision = Stage12Encoder().encode(right)
    equality = same_decision(left_decision, right_decision)
    v3_equality = same_decision(
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(left),
        Stage12Encoder("MARKOV_CONTEXT_V3").encode(right))
    visible_ids = [item.task.task_id for item in
                   left.queue.policy_view(left.now, Stage12Encoder.MAX_TASKS)]
    for task_id in visible_ids:
        left.queue.remove(task_id)
        right.queue.remove(task_id)
    left_future = Stage12Encoder().encode(left)
    right_future = Stage12Encoder().encode(right)
    future_differs = bool(
        not np.array_equal(left_future.observation, right_future.observation) or
        not np.array_equal(left_future.action_features,
                           right_future.action_features))
    return {
        "name": "top8_hidden_waiting_task_alias",
        "seed": seed,
        "perturbation": (
            "Only task rank 9 target geometry differs; the visible Top-8 and "
            "all counts are equal."),
        "hidden_task_id": hidden.task.task_id,
        "same_encoded_decision": equality,
        "markov_v3_state_equal": v3_equality["state_equal"],
        "future_difference_after_identical_top8_removed": future_differs,
        "alias_detected": all(equality.values()) and future_differs,
    }


def main() -> int:
    rows = variable_map()
    v3_rows = markov_v3_variable_map(rows)
    probes = [
        active_residual_alias(68000001),
        pending_arrival_alias(68000002),
        hidden_queue_alias(68000003),
    ]
    omitted = [
        {
            "field_family": "active_assignment_identity_and_mode",
            "causal_use": "completion result, participant release, mode counts",
            "currently_encoded": False,
        },
        {
            "field_family": "active_assignment_remaining_time",
            "causal_use": "next event time, reward integral, SMDP discount",
            "currently_encoded": False,
        },
        {
            "field_family": "active_participant_and_resource_owner_mapping",
            "causal_use": "future availability and legal action set",
            "currently_encoded": False,
        },
        {
            "field_family": "next_arrival_remaining_time",
            "causal_use": "next event time and SMDP discount",
            "currently_encoded": False,
        },
        {
            "field_family": "pending_arrival_task_attributes",
            "causal_use": "future queue composition and deadlines",
            "currently_encoded": False,
        },
        {
            "field_family": "waiting_task_attributes_beyond_top8",
            "causal_use": "future decisions after visible tasks are resolved",
            "currently_encoded": False,
        },
        {
            "field_family": "robot_previous_result_finish_pose_and_standby_slot_id",
            "causal_use": "persistent history/parking ownership diagnostics",
            "currently_encoded": False,
        },
    ]
    execution_checks = {
        "variable_map_has_exactly_284_unique_indices": bool(
            len(rows) == 284 and len({row["index"] for row in rows}) == 284),
        "all_three_constructive_alias_probes_ran": len(probes) == 3,
        "all_probes_preserve_current_encoded_decision": all(
            all(probe["same_encoded_decision"].values()) for probe in probes),
        "all_probes_expose_future_transition_or_state_difference": all(
            probe["alias_detected"] for probe in probes),
        "markov_v3_distinguishes_all_three_alias_pairs": all(
            not probe["markov_v3_state_equal"] for probe in probes),
    }
    aliases = [probe["name"] for probe in probes if probe["alias_detected"]]
    report = {
        "schema_version": "warehouse_stage23_state_sufficiency_audit_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage20/22 FULL_CONTEXT_V2 and repaired MARKOV_CONTEXT_V3",
        "claim_boundary": (
            "Constructive logical-simulator audit. It tests encoded-state "
            "sufficiency, not learned-policy performance."),
        "source_path": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "observation_shape": 284,
        "block_widths": {
            "global": 8, "top8_waiting_tasks": 104,
            "ten_robots": 140, "four_stairs": 32,
        },
        "variable_map_csv": str(CSV_OUTPUT),
        "markov_v3_variable_map_csv": str(V3_CSV_OUTPUT),
        "omitted_causally_relevant_families": omitted,
        "constructive_alias_probes": probes,
        "audit_execution_gate": {
            "passed": all(execution_checks.values()),
            "checks": execution_checks,
        },
        "scientific_assessment": {
            "full_context_v2_markov_smdp_sufficiency_supported": not aliases,
            "markov_context_v3_resolves_constructive_alias_gate": all(
                not probe["markov_v3_state_equal"] for probe in probes),
            "status": (
                "PASS_MARKOV_CONTEXT_V3_CONSTRUCTIVE_ALIAS_GATE" if
                all(not probe["markov_v3_state_equal"] for probe in probes)
                else "FAIL_MARKOV_CONTEXT_V3_ALIAS_REMAINS"),
            "detected_aliases": aliases,
            "formal_locked_test_blocked_until_observation_contract_is_resolved":
                not all(not probe["markov_v3_state_equal"] for probe in probes),
            "recommended_resolution": (
                "Add active-assignment residual/event/resource descriptors, "
                "next-arrival residual time, and a lossless or sufficient "
                "hidden-queue summary; then rerun interface gates and retrain "
                "the frozen full/ablation cohorts under a new protocol version."),
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with V3_CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(v3_rows[0]))
        writer.writeheader()
        writer.writerows(v3_rows)
    JSON_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "json_output": str(JSON_OUTPUT),
        "csv_output": str(CSV_OUTPUT),
        "markov_v3_csv_output": str(V3_CSV_OUTPUT),
        "audit_execution_passed": report["audit_execution_gate"]["passed"],
        "scientific_status": report["scientific_assessment"]["status"],
        "detected_aliases": aliases,
        "formal_locked_test_blocked": report["scientific_assessment"][
            "formal_locked_test_blocked_until_observation_contract_is_resolved"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["audit_execution_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
