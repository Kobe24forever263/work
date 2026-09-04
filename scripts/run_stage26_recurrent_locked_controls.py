#!/usr/bin/env python3
"""Run locked single-dog and recurrent-memory intervention controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Bind the Apple runtime before torch.
import torch


ROOT = Path(__file__).resolve().parents[1]
LOCKED_ROOT = (ROOT / "results" / "stage26_recurrent_causal" /
               "locked_test_v1")
CONTROL_ROOT = LOCKED_ROOT / "controls"
FREEZE = LOCKED_ROOT / "cohort_freeze_manifest.json"
SCHEMA = "warehouse_stage26_recurrent_locked_control_v1"

sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage26_recurrent_ppo import (
    evaluate_recurrent, load_recurrent_checkpoint)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def same_episode_contract(left: dict, right: dict,
                          expected_seeds: list[int]) -> bool:
    left_episodes = left["episodes"]
    right_episodes = right["episodes"]
    return bool(
        [row["seed"] for row in left_episodes] ==
        [row["seed"] for row in right_episodes] == expected_seeds and
        [row["scheduled_tasks"] for row in left_episodes] ==
        [row["scheduled_tasks"] for row in right_episodes] and
        [row.get("arrival_schedule_metadata", {}) for row in left_episodes] ==
        [row.get("arrival_schedule_metadata", {}) for row in right_episodes])


def valid_control(path: Path, condition: str, checkpoint_hash: str | None,
                  seed_start: int, seed_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        row.get("schema_version") == SCHEMA and
        row.get("condition") == condition and
        row.get("checkpoint_sha256") == checkpoint_hash and
        row.get("locked_test_seed_start") == seed_start and
        row.get("locked_test_seed_count") == seed_count and
        row.get("passed") is True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-locked-controls", action="store_true")
    args = parser.parse_args()
    if not args.confirm_locked_controls:
        parser.error("locked controls require --confirm-locked-controls")
    if not FREEZE.is_file():
        raise FileNotFoundError("run the frozen Stage 26 locked test first")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    seed_start = int(freeze["locked_test_seed_start"])
    seed_count = int(freeze["locked_test_seed_count"])
    seeds = list(range(seed_start, seed_start + seed_count))
    checkpoints = freeze["checkpoints"]
    if len(checkpoints) != 10:
        raise ValueError("locked cohort does not contain ten checkpoints")

    torch.set_num_threads(4)
    device = torch.device("cpu")
    common = dict(
        execution_mode="CONCURRENT", profile="MIXED_CURRICULUM",
        device=device, reward_contract="CONTINUOUS_TIME_V3")
    first_model, _ = load_recurrent_checkpoint(
        checkpoints[0]["checkpoint"], device)

    dog_path = CONTROL_ROOT / "single_dog_only.json"
    if valid_control(
            dog_path, "SINGLE_DOG_ONLY_FOUR_GO2_FLEET", None,
            seed_start, seed_count):
        print("single-dog control: existing result passed; skip", flush=True)
    else:
        print("single-dog control: evaluating 100 locked episodes...",
              flush=True)
        single_dog = evaluate_recurrent(
            first_model, seeds, use_rule=True,
            allowed_transport_modes=("SINGLE_DOG",), **common)
        assertions = {
            "shared_locked_seed_order": (
                [row["seed"] for row in single_dog["episodes"]] == seeds),
            "only_single_dog_mode_used": (
                not (set(single_dog["transport_modes"]) - {"SINGLE_DOG"})),
            "all_tasks_accounted_for": (
                single_dog["completed"] + single_dog["failed"] +
                single_dog["unresolved"] ==
                single_dog["scheduled_task_count"]),
            "zero_illegal_actions": single_dog["illegal_action_count"] == 0,
            "zero_resource_leaks": single_dog["resource_leak_count"] == 0,
        }
        assertions = {key: bool(value) for key, value in assertions.items()}
        dog_result = {
            "schema_version": SCHEMA,
            "stage": 26,
            "condition": "SINGLE_DOG_ONLY_FOUR_GO2_FLEET",
            "claim_boundary": (
                "Fixed blank control with four Go2 robots available; Carter "
                "and handover transport modes are disabled."),
            "checkpoint_sha256": None,
            "locked_test_seed_start": seed_start,
            "locked_test_seed_count": seed_count,
            "allowed_transport_modes": ["SINGLE_DOG"],
            "control": single_dog,
            "assertions": assertions,
            "passed": all(assertions.values()),
        }
        write_json(dog_path, dog_result)
        if not dog_result["passed"]:
            raise RuntimeError("single-dog locked control failed")
        print(
            "single-dog control: pass, "
            f"success={single_dog['success_rate']:.3%}, "
            f"reward={single_dog['mean_reward']:.3f}", flush=True)

    completed = []
    for frozen in checkpoints:
        index = int(frozen["training_seed_index"])
        checkpoint_hash = frozen["checkpoint_sha256"]
        output = (CONTROL_ROOT / "memory_reset_each_decision" /
                  f"seed_{index:02d}.json")
        if valid_control(
                output, "RESET_GRU_MEMORY_EACH_DECISION", checkpoint_hash,
                seed_start, seed_count):
            print(
                f"memory ablation seed {index:02d}/10: existing result "
                "passed; skip", flush=True)
            completed.append(index)
            continue
        checkpoint = Path(frozen["checkpoint"])
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError(f"seed {index:02d} frozen checkpoint changed")
        full_path = LOCKED_ROOT / f"seed_{index:02d}.json"
        full = json.loads(full_path.read_text(encoding="utf-8"))["policy"]
        print(
            f"memory ablation seed {index:02d}/10: evaluating...",
            flush=True)
        model, _ = load_recurrent_checkpoint(checkpoint, device)
        ablation = evaluate_recurrent(
            model, seeds, memory_mode="RESET_EACH_DECISION", **common)
        assertions = {
            "checkpoint_matches_pretest_freeze": (
                sha256(checkpoint) == checkpoint_hash),
            "paired_task_stream_contract_matches_full_policy":
                same_episode_contract(ablation, full, seeds),
            "memory_is_reset_each_decision": (
                ablation.get("memory_mode") == "RESET_EACH_DECISION"),
            "all_tasks_accounted_for": (
                ablation["completed"] + ablation["failed"] +
                ablation["unresolved"] ==
                ablation["scheduled_task_count"]),
            "zero_illegal_actions": ablation["illegal_action_count"] == 0,
            "zero_resource_leaks": ablation["resource_leak_count"] == 0,
        }
        assertions = {key: bool(value) for key, value in assertions.items()}
        result = {
            "schema_version": SCHEMA,
            "stage": 26,
            "condition": "RESET_GRU_MEMORY_EACH_DECISION",
            "claim_boundary": (
                "Inference-time intervention on the same trained recurrent "
                "checkpoint. It isolates carried memory use but is not a "
                "separately trained architecture ablation."),
            "training_seed_index": index,
            "training_base_seed": frozen["base_seed"],
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
            "locked_test_seed_start": seed_start,
            "locked_test_seed_count": seed_count,
            "control": ablation,
            "assertions": assertions,
            "passed": all(assertions.values()),
        }
        write_json(output, result)
        if not result["passed"]:
            raise RuntimeError(
                f"memory ablation seed {index:02d} safety gate failed")
        completed.append(index)
        print(
            f"memory ablation seed {index:02d}/10: pass, "
            f"success={ablation['success_rate']:.3%}, "
            f"reward={ablation['mean_reward']:.3f}", flush=True)

    status = {
        "schema_version": "warehouse_stage26_locked_controls_status_v1",
        "stage": 26,
        "conditions": [
            "SINGLE_DOG_ONLY_FOUR_GO2_FLEET",
            "RESET_GRU_MEMORY_EACH_DECISION"],
        "completed_memory_ablation_seed_indices": completed,
        "passed": completed == list(range(1, 11)) and
                  valid_control(
                      dog_path, "SINGLE_DOG_ONLY_FOUR_GO2_FLEET", None,
                      seed_start, seed_count),
    }
    write_json(CONTROL_ROOT / "control_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
