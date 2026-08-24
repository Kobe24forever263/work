#!/usr/bin/env python3
"""Evaluate frozen Stage 20 and Stage 19 cohorts on fresh mixed curricula."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Stable Apple Silicon load order before torch.
import torch
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))
from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402
from run_stage15_stress_evaluation import (  # noqa: E402
    StressScenario, run_policy)


CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage20_mixed_curriculum.yaml"
)
STAGE20_FREEZE = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json"
)
STAGE19_FREEZE = (
    ROOT / "results" / "stage19_mixed_recovery" /
    "stage19_full_policy_freeze_manifest.json"
)
SCENARIO = StressScenario(
    name="MIXED_CURRICULUM_LOCKED",
    description=(
        "Fresh randomized persistent NORMAL/DENSE/BURST/RECOVERY curriculum"),
    arrival_interval=(30.0, 50.0),
    episode_spec=None,
    arrival_schedule="MIXED_CURRICULUM",
)
PHASES = {"NORMAL", "DENSE", "BURST", "RECOVERY"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safety_and_completeness(result: dict, episode_count: int) -> bool:
    episodes = result.get("episodes", [])
    phase_results = result.get("phase_results", {})
    return bool(
        result.get("episode_count") == episode_count and
        result.get("task_count") == episode_count * 80 and
        result.get("completed", 0) + result.get("failed", 0) ==
        episode_count * 80 and
        result.get("illegal_action_count") == 0 and
        result.get("resource_leak_count") == 0 and
        len(episodes) == episode_count and
        all(episode.get("task_count") == 80 for episode in episodes) and
        all(len(episode.get("tasks", [])) == 80 for episode in episodes) and
        set(phase_results) == PHASES and
        sum(row.get("task_count", 0) for row in phase_results.values()) ==
        episode_count * 80)


def output_valid(path: Path, *, source_group: str, cohort: str,
                 seed_index: int, weight_hash: str, test_seed_start: int,
                 episode_count: int, expected_fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = row.get("result", {})
    return bool(
        row.get("schema_version") == "warehouse_stage20_locked_test_v1" and
        row.get("source_group") == source_group and
        row.get("training_cohort") == cohort and
        row.get("training_seed_index") == seed_index and
        row.get("weight_sha256") == weight_hash and
        row.get("test_seed_start") == test_seed_start and
        row.get("test_episode_count") == episode_count and
        result.get("task_fingerprint") == expected_fingerprint and
        safety_and_completeness(result, episode_count))


def baseline_valid(path: Path, *, test_seed_start: int,
                   episode_count: int) -> tuple[bool, str | None]:
    if not path.exists():
        return False, None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    rule = row.get("time_greedy_rule", {})
    dog = row.get("single_dog_only", {})
    valid = bool(
        row.get("schema_version") ==
        "warehouse_stage20_locked_baselines_v1" and
        row.get("test_seed_start") == test_seed_start and
        row.get("test_episode_count") == episode_count and
        safety_and_completeness(rule, episode_count) and
        safety_and_completeness(dog, episode_count) and
        rule.get("task_fingerprint") == dog.get("task_fingerprint"))
    return valid, rule.get("task_fingerprint") if valid else None


def save_status(out: Path, *, campaign: str, seed_start: int,
                episode_count: int, jobs: list[dict], state: str,
                baseline_path: Path | None = None) -> None:
    payload = {
        "schema_version": "warehouse_stage20_locked_campaign_status_v1",
        "stage": 20,
        "campaign": campaign,
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_seed_start": seed_start,
        "test_episode_count": episode_count,
        "jobs": jobs,
        "baseline_output": str(baseline_path) if baseline_path else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign.status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def stage20_jobs(manifest: dict) -> list[dict]:
    return [{
        "source_group": "STAGE20_MIXED_CURRICULUM",
        "cohort": "MIXED_CURRICULUM",
        "seed_index": int(row["training_seed_index"]),
        "weight": Path(row["best_weight_path"]),
        "weight_hash": row["best_weight_sha256"],
        "training_gate_accepted": bool(row["accepted_by_training_gate"]),
        "freeze_row": row,
    } for row in manifest["runs"]]


def stage19_jobs(manifest: dict) -> list[dict]:
    return [{
        "source_group": "STAGE19_FIXED_PROFILE",
        "cohort": row["training_profile"],
        "seed_index": int(row["training_seed_index"]),
        "weight": Path(row["best_weight_path"]),
        "weight_hash": row["best_weight_sha256"],
        "training_gate_accepted": True,
        "freeze_row": row,
    } for row in manifest["runs"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument(
        "--groups", nargs="+",
        choices=("STAGE20", "STAGE19", "BASELINES"),
        default=("STAGE20", "STAGE19", "BASELINES"))
    args = parser.parse_args()
    if args.execute and not args.smoke and not args.confirm_locked_test:
        parser.error(
            "formal locked test is run-once; add --confirm-locked-test")

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    stage20_manifest = json.loads(
        STAGE20_FREEZE.read_text(encoding="utf-8"))
    stage19_manifest = json.loads(
        STAGE19_FREEZE.read_text(encoding="utf-8"))
    if not stage20_manifest.get("hard_gate", {}).get("passed"):
        raise RuntimeError("Stage 20 freeze gate has not passed")
    if not stage19_manifest.get("hard_gate", {}).get("passed"):
        raise RuntimeError("Stage 19 freeze gate has not passed")

    seed_cfg = protocol["smoke" if args.smoke else "locked_test"]
    seed_start = int(seed_cfg["seed_start"])
    episode_count = 2 if args.smoke else int(seed_cfg["seed_count"])
    seeds = list(range(seed_start, seed_start + episode_count))
    campaign = "smoke" if args.smoke else "locked_test_v1"
    out = ROOT / "results" / "stage20_mixed_curriculum" / campaign
    jobs = []
    if "STAGE20" in args.groups:
        jobs.extend(stage20_jobs(stage20_manifest))
    if "STAGE19" in args.groups:
        jobs.extend(stage19_jobs(stage19_manifest))

    plan = {
        "stage": 20,
        "campaign": campaign,
        "execute": args.execute,
        "groups": list(args.groups),
        "policy_jobs": len(jobs),
        "stage20_jobs": sum(
            job["source_group"] == "STAGE20_MIXED_CURRICULUM"
            for job in jobs),
        "stage19_jobs": sum(
            job["source_group"] == "STAGE19_FIXED_PROFILE"
            for job in jobs),
        "test_seed_range": [seeds[0], seeds[-1]],
        "episodes_per_job": episode_count,
        "tasks_per_episode": 80,
        "total_policy_tasks": len(jobs) * episode_count * 80,
        "baselines": (
            ["TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"]
            if "BASELINES" in args.groups else []),
        "claim_boundary": (
            "Protocol smoke only; locked-test seeds remain untouched."
            if args.smoke else
            "Run-once held-out evaluation of frozen weights."),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict] = []
    save_status(
        out, campaign=campaign, seed_start=seed_start,
        episode_count=episode_count, jobs=status_rows, state="RUNNING")

    baseline_path = out / "baselines.json"
    baseline_ok, expected_fingerprint = baseline_valid(
        baseline_path, test_seed_start=seed_start,
        episode_count=episode_count)
    if "BASELINES" in args.groups and not baseline_ok:
        rule = run_policy(
            None, seeds, SCENARIO, "TIME_GREEDY_RULE", "CONCURRENT")
        single_dog = run_policy(
            None, seeds, SCENARIO, "TIME_GREEDY_RULE", "CONCURRENT",
            allowed_transport_modes=("SINGLE_DOG",))
        if not safety_and_completeness(rule, episode_count):
            raise RuntimeError("time-greedy baseline failed safety/completeness")
        if not safety_and_completeness(single_dog, episode_count):
            raise RuntimeError("single-dog baseline failed safety/completeness")
        if rule["task_fingerprint"] != single_dog["task_fingerprint"]:
            raise RuntimeError("baseline task streams differ")
        expected_fingerprint = rule["task_fingerprint"]
        baseline_path.write_text(json.dumps({
            "schema_version": "warehouse_stage20_locked_baselines_v1",
            "test_seed_start": seed_start,
            "test_episode_count": episode_count,
            "evaluation_protocol": {
                "arrival_schedule": "MIXED_CURRICULUM",
                "state_reset_at_phase_boundary": False,
                "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
            },
            "time_greedy_rule": rule,
            "single_dog_only": single_dog,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif expected_fingerprint is None:
        # A policy-only partial campaign still needs an independently generated
        # common task-stream reference, but it does not persist a baseline.
        reference = run_policy(
            None, seeds, SCENARIO, "TIME_GREEDY_RULE", "CONCURRENT")
        if not safety_and_completeness(reference, episode_count):
            raise RuntimeError("task-stream reference failed")
        expected_fingerprint = reference["task_fingerprint"]

    progress = tqdm(
        jobs, desc=f"Stage20 {campaign}", unit="model", dynamic_ncols=True)
    device = torch.device("cpu")
    for job in progress:
        source = job["source_group"]
        cohort = job["cohort"]
        seed_index = job["seed_index"]
        weight = job["weight"]
        if sha256(weight) != job["weight_hash"]:
            raise RuntimeError(f"frozen weight changed: {weight}")
        target = (
            out / source.lower() / cohort.lower() /
            f"seed_{seed_index:02d}.json")
        if output_valid(
                target, source_group=source, cohort=cohort,
                seed_index=seed_index, weight_hash=job["weight_hash"],
                test_seed_start=seed_start, episode_count=episode_count,
                expected_fingerprint=expected_fingerprint):
            status_rows.append({
                "source_group": source, "cohort": cohort,
                "training_seed_index": seed_index,
                "state": "SKIPPED_ALREADY_COMPLETE", "output": str(target)})
            save_status(
                out, campaign=campaign, seed_start=seed_start,
                episode_count=episode_count, jobs=status_rows,
                state="RUNNING", baseline_path=baseline_path)
            continue

        model, checkpoint = load_checkpoint(weight, device)
        expected_profile = (
            "MIXED_CURRICULUM" if source == "STAGE20_MIXED_CURRICULUM"
            else cohort)
        if (checkpoint.get("arrival_profile") != expected_profile or
                checkpoint.get("observation_variant") != "FULL_CONTEXT_V2"):
            raise RuntimeError(f"checkpoint metadata mismatch: {weight}")
        result = run_policy(
            model, seeds, SCENARIO, "PPO", "CONCURRENT")
        if not safety_and_completeness(result, episode_count):
            raise RuntimeError(
                f"safety/completeness failed: {source}/{cohort}/{seed_index}")
        if result["task_fingerprint"] != expected_fingerprint:
            raise RuntimeError(
                f"task stream mismatch: {source}/{cohort}/{seed_index}")
        output = {
            "schema_version": "warehouse_stage20_locked_test_v1",
            "claim_boundary": (
                "Held-out randomized mixed-curriculum evaluation; no phase "
                "reset and no post-freeze model selection."),
            "source_group": source,
            "training_cohort": cohort,
            "training_seed_index": seed_index,
            "training_gate_accepted": job["training_gate_accepted"],
            "weight_path": str(weight),
            "weight_sha256": job["weight_hash"],
            "test_seed_start": seed_start,
            "test_episode_count": episode_count,
            "evaluation_protocol": {
                "arrival_schedule": "MIXED_CURRICULUM",
                "phase_order_randomized": True,
                "state_reset_at_phase_boundary": False,
                "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
            },
            "result": result,
            "safety_and_completeness_passed": True,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        status_rows.append({
            "source_group": source, "cohort": cohort,
            "training_seed_index": seed_index,
            "state": "COMPLETED", "output": str(target)})
        save_status(
            out, campaign=campaign, seed_start=seed_start,
            episode_count=episode_count, jobs=status_rows,
            state="RUNNING", baseline_path=baseline_path)

    save_status(
        out, campaign=campaign, seed_start=seed_start,
        episode_count=episode_count, jobs=status_rows, state="COMPLETED",
        baseline_path=baseline_path if baseline_path.exists() else None)
    print(json.dumps({
        **plan,
        "state": "COMPLETED",
        "output": str(out),
        "task_fingerprint": expected_fingerprint,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
