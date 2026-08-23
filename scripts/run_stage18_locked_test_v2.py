#!/usr/bin/env python3
"""Corrected Stage 18 confirmatory evaluation on fresh paired test seeds."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# NumPy must precede torch in the Apple Silicon Pixi runtime to avoid loading
# two OpenMP runtimes through different dependency paths.
import numpy as np  # noqa: F401
import torch
from tqdm import tqdm
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import evaluate, load_checkpoint


SEED_CONFIG = (WORK_ROOT / "src" / "warehouse_bringup" / "config" /
               "experiment_seeds_stage18_v2.yaml")
OUTPUT_ROOT = WORK_ROOT / "results" / "stage18_locked_test_v2"
STATUS = OUTPUT_ROOT / "stage18_locked_test_v2_campaign.status.json"
SUMMARY_SCRIPT = WORK_ROOT / "scripts" / "summarize_stage18_locked_test_v2.py"
HANDOVER_SAMPLING = "TASK_KEYED"
PROFILE_ROOTS = {
    "MEDIUM": "stage18", "DENSE": "stage18_dense",
    "BURST": "stage18_burst"}
CONDITIONS = {
    "full_context_v2": "FULL_CONTEXT_V2",
    "no_queue_resource": "NO_QUEUE_RESOURCE",
    "no_handover_cues": "NO_HANDOVER_CUES",
    "no_persistent_position": "NO_PERSISTENT_POSITION",
}


def evaluation_jobs() -> list[tuple[str, str, str, int]]:
    jobs = []
    # Matched-load evidence for Full and all three context ablations.
    for training_profile in PROFILE_ROOTS:
        for condition in CONDITIONS:
            for seed in range(1, 11):
                jobs.append((training_profile, condition,
                             training_profile, seed))
    # Cross-load matrix for the same Full model family.  This replaces the
    # old Stage 15 checkpoint when drawing load-generalization figures.
    for training_profile in PROFILE_ROOTS:
        for evaluation_profile in PROFILE_ROOTS:
            if evaluation_profile == training_profile:
                continue
            for seed in range(1, 11):
                jobs.append((training_profile, "full_context_v2",
                             evaluation_profile, seed))
    return jobs


def weight_path(training_profile: str, condition: str, seed: int) -> Path:
    return (WORK_ROOT / "results" / PROFILE_ROOTS[training_profile] /
            condition / f"seed_{seed:02d}" / "stage18_ppo.best.pt")


def output_path(training_profile: str, condition: str,
                evaluation_profile: str, seed: int) -> Path:
    return (OUTPUT_ROOT / f"trained_{training_profile.lower()}" / condition /
            f"tested_{evaluation_profile.lower()}" /
            f"seed_{seed:02d}.json")


def valid(path: Path, training_profile: str, condition: str,
          evaluation_profile: str, seed: int, test_start: int,
          test_count: int, weight_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    policy = row.get("policy", {})
    rule = row.get("rule_baseline", {})
    return bool(
        row.get("schema_version") == "warehouse_stage18_locked_test_v2" and
        row.get("training_profile") == training_profile and
        row.get("condition") == condition and
        row.get("evaluation_profile") == evaluation_profile and
        row.get("training_seed_index") == seed and
        row.get("test_seed_start") == test_start and
        row.get("test_episode_count") == test_count and
        row.get("weight_sha256") == weight_sha256 and
        row.get("evaluation_protocol", {}).get("handover_sampling") ==
        "TASK_KEYED_COMMON_RANDOM_NUMBERS" and
        policy.get("episode_count") == test_count and
        len(policy.get("episodes", [])) == test_count and
        rule.get("episode_count") == test_count and
        len(rule.get("episodes", [])) == test_count and
        policy.get("handover_sampling") == HANDOVER_SAMPLING and
        rule.get("handover_sampling") == HANDOVER_SAMPLING and
        policy.get("illegal_action_count") == 0 and
        policy.get("resource_leak_count") == 0 and
        rule.get("resource_leak_count") == 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_status(rows: list[dict], state: str, test_start: int,
                test_count: int) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": 18,
        "campaign": "CORRECTED_CONFIRMATORY_V2_AND_CROSS_LOAD",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_seed_start": test_start,
        "test_episode_count": test_count,
        "checkpoint_policy": "validation_selected_best_locked_before_test",
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
        "jobs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    protocol = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    test_start = int(protocol["test"]["seed_start"])
    test_count = int(protocol["test"]["seed_count"])
    seeds = list(range(test_start, test_start + test_count))
    device = torch.device("cpu")
    torch.set_num_threads(4)
    rows: list[dict] = []
    save_status(rows, "RUNNING", test_start, test_count)
    jobs = evaluation_jobs()
    print(
        f"Stage18 v2确认性评估：{len(jobs)} jobs，"
        f"每个job包含{test_count}个policy和{test_count}个rule episodes。",
        flush=True)
    progress = tqdm(
        jobs, total=len(jobs), desc="Stage18 v2 TASK_KEYED",
        unit="job", dynamic_ncols=True, mininterval=0.5)
    for training_profile, condition, evaluation_profile, seed_index in progress:
        progress.set_postfix(
            train=training_profile,
            condition=condition.replace("_context_v2", ""),
            test=evaluation_profile,
            seed=f"{seed_index:02d}")
        output = output_path(
            training_profile, condition, evaluation_profile, seed_index)
        weight = weight_path(training_profile, condition, seed_index)
        if not weight.exists():
            raise FileNotFoundError(f"locked checkpoint missing: {weight}")
        weight_sha256 = sha256(weight)
        if valid(output, training_profile, condition, evaluation_profile,
                 seed_index, test_start, test_count, weight_sha256):
            rows.append({
                "training_profile": training_profile,
                "condition": condition,
                "evaluation_profile": evaluation_profile,
                "training_seed_index": seed_index,
                "state": "SKIPPED_ALREADY_COMPLETE",
                "output": str(output)})
            save_status(rows, "RUNNING", test_start, test_count)
            continue
        model, checkpoint = load_checkpoint(weight, device)
        expected_observation = CONDITIONS[condition]
        if (checkpoint.get("arrival_profile") != training_profile or
                checkpoint.get("observation_variant") != expected_observation):
            raise ValueError(f"checkpoint metadata mismatch: {weight}")
        row = {
            "training_profile": training_profile,
            "condition": condition,
            "evaluation_profile": evaluation_profile,
            "training_seed_index": seed_index,
            "state": "RUNNING",
            "weight": str(weight),
            "weight_sha256": weight_sha256,
        }
        rows.append(row)
        save_status(rows, "RUNNING", test_start, test_count)
        policy = evaluate(
            model, seeds, "CONCURRENT", evaluation_profile, device,
            observation_variant=expected_observation,
            handover_sampling=HANDOVER_SAMPLING)
        rule = evaluate(
            model, seeds, "CONCURRENT", evaluation_profile, device,
            use_rule=True, observation_variant=expected_observation,
            handover_sampling=HANDOVER_SAMPLING)
        result = {
            "schema_version": "warehouse_stage18_locked_test_v2",
            "claim_boundary": (
                "Corrected confirmatory result on fresh test seeds. The v1 "
                "locked results were treated as exploratory after audit."),
            "training_profile": training_profile,
            "condition": condition,
            "evaluation_profile": evaluation_profile,
            "training_seed_index": seed_index,
            "weight": str(weight),
            "weight_sha256": weight_sha256,
            "checkpoint_update": checkpoint.get("update"),
            "test_seed_start": test_start,
            "test_episode_count": test_count,
            "evaluation_protocol": {
                "handover_sampling": (
                    "TASK_KEYED_COMMON_RANDOM_NUMBERS"),
                "pairing_key": "(test_seed, task_id, handover_kind)",
                "randomness_pairing": "shared_across_policy_and_baselines",
                "result_version": "v2_non_overwriting",
            },
            "policy": policy,
            "rule_baseline": rule,
            "comparison": {
                "mean_reward_delta": (
                    policy["mean_reward"] - rule["mean_reward"]),
                "success_rate_delta": (
                    policy["success_rate"] - rule["success_rate"]),
                "throughput_delta_tasks_per_hour": (
                    policy["throughput_tasks_per_hour"] -
                    rule["throughput_tasks_per_hour"]),
                "resolved_throughput_delta_tasks_per_hour": (
                    policy["resolved_throughput_tasks_per_hour"] -
                    rule["resolved_throughput_tasks_per_hour"]),
                "successful_throughput_delta_tasks_per_hour": (
                    policy["successful_throughput_tasks_per_hour"] -
                    rule["successful_throughput_tasks_per_hour"]),
            },
            "safety": {
                "policy_actions_all_legal": (
                    policy["illegal_action_count"] == 0),
                "policy_has_no_resource_leak": (
                    policy["resource_leak_count"] == 0),
                "rule_has_no_resource_leak": (
                    rule["resource_leak_count"] == 0),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if not valid(output, training_profile, condition, evaluation_profile,
                     seed_index, test_start, test_count, weight_sha256):
            row["state"] = "FAILED_VALIDATION"
            save_status(rows, "STOPPED_ON_FAILURE", test_start, test_count)
            raise RuntimeError(f"locked-test validation failed: {output}")
        row["state"] = "COMPLETED"
        row["output"] = str(output)
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(rows, "RUNNING", test_start, test_count)
    save_status(rows, "COMPLETED", test_start, test_count)
    subprocess.run([sys.executable, str(SUMMARY_SCRIPT)],
                   cwd=WORK_ROOT, check=True)
    print(json.dumps({
        "status": str(STATUS), "state": "COMPLETED",
        "job_count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
