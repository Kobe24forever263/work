#!/usr/bin/env python3
"""Plan or run the frozen Stage 23 held-out campaign."""

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
    "experiment_stage23_locked_test.yaml"
)
FREEZE = (
    ROOT / "results" / "stage23_markov_continuous" / "freeze" /
    "stage23_policy_freeze_manifest.json"
)
SCENARIO = StressScenario(
    name="MIXED_CURRICULUM_LOCKED",
    description=(
        "Fresh persistent randomized NORMAL/DENSE/BURST/RECOVERY curriculum"),
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


def safety_and_completeness(result: dict, episode_count: int,
                            tasks_per_episode: int) -> bool:
    episodes = result.get("episodes", [])
    phase_results = result.get("phase_results", {})
    expected_tasks = episode_count * tasks_per_episode
    return bool(
        result.get("episode_count") == episode_count and
        result.get("task_count") == expected_tasks and
        result.get("completed", 0) + result.get("failed", 0) ==
        expected_tasks and
        result.get("illegal_action_count") == 0 and
        result.get("resource_leak_count") == 0 and
        result.get("handover_sampling") ==
        "TASK_KEYED_COMMON_RANDOM_NUMBERS" and
        result.get("handover_potential_fingerprint") and
        len(episodes) == episode_count and
        all(episode.get("task_count") == tasks_per_episode
            for episode in episodes) and
        all(len(episode.get("tasks", [])) == tasks_per_episode
            for episode in episodes) and
        all(all(task.get("task_result") in {"COMPLETED", "FAILED"}
                for task in episode.get("tasks", []))
            for episode in episodes) and
        all(episode.get("handover_potential_fingerprint")
            for episode in episodes) and
        set(phase_results) == PHASES and
        sum(row.get("task_count", 0) for row in phase_results.values()) ==
        expected_tasks)


def output_valid(path: Path, *, seed_index: int, weight_hash: str,
                 test_seed_start: int, episode_count: int,
                 tasks_per_episode: int, task_fingerprint: str,
                 handover_fingerprint: str, config_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = row.get("result", {})
    return bool(
        row.get("schema_version") == "warehouse_stage23_locked_test_v1" and
        row.get("training_seed_index") == seed_index and
        row.get("weight_sha256") == weight_hash and
        row.get("analysis_protocol_sha256") == config_hash and
        row.get("test_seed_start") == test_seed_start and
        row.get("test_episode_count") == episode_count and
        result.get("task_fingerprint") == task_fingerprint and
        result.get("handover_potential_fingerprint") ==
        handover_fingerprint and
        safety_and_completeness(
            result, episode_count, tasks_per_episode))


def baseline_valid(path: Path, *, test_seed_start: int,
                   episode_count: int, tasks_per_episode: int,
                   config_hash: str) -> tuple[bool, str | None, str | None]:
    if not path.exists():
        return False, None, None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None, None
    rule = row.get("time_greedy_rule", {})
    dog = row.get("single_dog_only", {})
    task_fingerprint = rule.get("task_fingerprint")
    handover_fingerprint = rule.get("handover_potential_fingerprint")
    valid = bool(
        row.get("schema_version") ==
        "warehouse_stage23_locked_baselines_v1" and
        row.get("analysis_protocol_sha256") == config_hash and
        row.get("test_seed_start") == test_seed_start and
        row.get("test_episode_count") == episode_count and
        safety_and_completeness(rule, episode_count, tasks_per_episode) and
        safety_and_completeness(dog, episode_count, tasks_per_episode) and
        task_fingerprint == dog.get("task_fingerprint") and
        handover_fingerprint == dog.get("handover_potential_fingerprint"))
    return (
        valid,
        task_fingerprint if valid else None,
        handover_fingerprint if valid else None,
    )


def save_status(out: Path, *, campaign: str, seed_start: int,
                episode_count: int, jobs: list[dict], state: str,
                baseline_path: Path | None = None) -> None:
    payload = {
        "schema_version": "warehouse_stage23_locked_campaign_status_v1",
        "stage": 23,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument(
        "--groups", nargs="+", choices=("POLICY", "BASELINES"),
        default=("POLICY", "BASELINES"))
    args = parser.parse_args()
    if args.execute and not args.smoke and not args.confirm_locked_test:
        parser.error(
            "formal locked test is run-once; add --confirm-locked-test")

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not freeze.get("hard_gate", {}).get("passed"):
        raise RuntimeError("Stage 23 policy freeze gate has not passed")
    config_hash = sha256(CONFIG)
    frozen_analysis = freeze.get("analysis_protocol", {})
    if frozen_analysis.get("sha256") != config_hash:
        raise RuntimeError(
            "locked-test analysis protocol is not part of the freeze "
            "manifest; rerun the freeze gate")
    for relative_path, expected_hash in frozen_analysis.get(
            "evaluation_source_sha256", {}).items():
        source_path = ROOT / relative_path
        if not source_path.exists() or sha256(source_path) != expected_hash:
            raise RuntimeError(
                f"frozen evaluation source changed: {source_path}; "
                "rerun protocol review and the freeze gate")

    seed_cfg = protocol["smoke" if args.smoke else "locked_test"]
    seed_start = int(seed_cfg["seed_start"])
    episode_count = int(seed_cfg["seed_count"])
    seeds = list(range(seed_start, seed_start + episode_count))
    scenario_cfg = protocol["scenario"]
    method = protocol["method"]
    tasks_per_episode = int(scenario_cfg["tasks_per_episode"])
    campaign = "locked_protocol_smoke" if args.smoke else "locked_test_v1"
    out = ROOT / "results" / "stage23_markov_continuous" / campaign
    jobs = []
    if "POLICY" in args.groups:
        jobs = [{
            "seed_index": int(row["training_seed_index"]),
            "weight": Path(row["selected_checkpoint"]["path"]),
            "weight_hash": row["selected_checkpoint"]["sha256"],
            "selected_update": int(row["selected_checkpoint"]["update"]),
        } for row in freeze["runs"]]

    plan = {
        "stage": 23,
        "campaign": campaign,
        "execute": args.execute,
        "groups": list(args.groups),
        "policy_jobs": len(jobs),
        "test_seed_range": [seeds[0], seeds[-1]],
        "episodes_per_job": episode_count,
        "tasks_per_episode": tasks_per_episode,
        "total_policy_tasks": len(jobs) * episode_count * tasks_per_episode,
        "baselines": (["TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"]
                      if "BASELINES" in args.groups else []),
        "analysis_protocol": str(CONFIG),
        "analysis_protocol_sha256": config_hash,
        "claim_boundary": (
            protocol["smoke"]["claim_boundary"] if args.smoke else
            "Run-once held-out evaluation of frozen checkpoints."),
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
    baseline_ok, task_fingerprint, handover_fingerprint = baseline_valid(
        baseline_path, test_seed_start=seed_start,
        episode_count=episode_count, tasks_per_episode=tasks_per_episode,
        config_hash=config_hash)
    run_arguments = {
        "scenario": SCENARIO,
        "execution_mode": method["execution_mode"],
        "observation_variant": method["observation_variant"],
        "reward_contract": method["reward_contract"],
    }
    if "BASELINES" in args.groups and not baseline_ok:
        rule = run_policy(
            None, seeds, policy_kind="TIME_GREEDY_RULE",
            **run_arguments)
        single_dog = run_policy(
            None, seeds, policy_kind="TIME_GREEDY_RULE",
            allowed_transport_modes=("SINGLE_DOG",),
            **run_arguments)
        if not safety_and_completeness(
                rule, episode_count, tasks_per_episode):
            raise RuntimeError("time-greedy baseline failed completeness")
        if not safety_and_completeness(
                single_dog, episode_count, tasks_per_episode):
            raise RuntimeError("single-dog baseline failed completeness")
        if rule["task_fingerprint"] != single_dog["task_fingerprint"]:
            raise RuntimeError("baseline task streams differ")
        if (rule["handover_potential_fingerprint"] !=
                single_dog["handover_potential_fingerprint"]):
            raise RuntimeError("baseline handover CRN streams differ")
        task_fingerprint = rule["task_fingerprint"]
        handover_fingerprint = rule["handover_potential_fingerprint"]
        baseline_path.write_text(json.dumps({
            "schema_version": "warehouse_stage23_locked_baselines_v1",
            "analysis_protocol": str(CONFIG),
            "analysis_protocol_sha256": config_hash,
            "test_seed_start": seed_start,
            "test_episode_count": episode_count,
            "evaluation_protocol": {
                "arrival_schedule": scenario_cfg["arrival_schedule"],
                "state_reset_at_phase_boundary":
                scenario_cfg["state_reset_at_phase_boundary"],
                "handover_sampling": scenario_cfg["handover_sampling"],
                "observation_variant": method["observation_variant"],
                "reward_contract": method["reward_contract"],
            },
            "time_greedy_rule": rule,
            "single_dog_only": single_dog,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif task_fingerprint is None or handover_fingerprint is None:
        reference = run_policy(
            None, seeds, policy_kind="TIME_GREEDY_RULE",
            **run_arguments)
        if not safety_and_completeness(
                reference, episode_count, tasks_per_episode):
            raise RuntimeError("task-stream reference failed")
        task_fingerprint = reference["task_fingerprint"]
        handover_fingerprint = reference[
            "handover_potential_fingerprint"]

    progress = tqdm(
        jobs, desc=f"Stage23 {campaign}", unit="model", dynamic_ncols=True)
    device = torch.device("cpu")
    for job in progress:
        seed_index = job["seed_index"]
        weight = job["weight"]
        if sha256(weight) != job["weight_hash"]:
            raise RuntimeError(f"frozen weight changed: {weight}")
        target = out / "policy" / f"seed_{seed_index:02d}.json"
        if output_valid(
                target, seed_index=seed_index,
                weight_hash=job["weight_hash"],
                test_seed_start=seed_start, episode_count=episode_count,
                tasks_per_episode=tasks_per_episode,
                task_fingerprint=task_fingerprint,
                handover_fingerprint=handover_fingerprint,
                config_hash=config_hash):
            status_rows.append({
                "training_seed_index": seed_index,
                "state": "SKIPPED_ALREADY_COMPLETE",
                "output": str(target)})
            save_status(
                out, campaign=campaign, seed_start=seed_start,
                episode_count=episode_count, jobs=status_rows,
                state="RUNNING", baseline_path=baseline_path)
            continue

        model, checkpoint = load_checkpoint(weight, device)
        metadata_ok = all((
            checkpoint.get("observation_variant") ==
            method["observation_variant"],
            checkpoint.get("policy_variant") == method["policy_variant"],
            checkpoint.get("reward_contract") == method["reward_contract"],
            checkpoint.get("execution_mode") == method["execution_mode"],
            checkpoint.get("arrival_profile") == method["arrival_profile"],
            int(checkpoint.get("update", -1)) == job["selected_update"],
        ))
        if not metadata_ok:
            raise RuntimeError(f"checkpoint metadata mismatch: {weight}")
        result = run_policy(
            model, seeds, policy_kind="PPO", **run_arguments)
        if not safety_and_completeness(
                result, episode_count, tasks_per_episode):
            raise RuntimeError(
                f"safety/completeness failed: seed {seed_index:02d}")
        if result["task_fingerprint"] != task_fingerprint:
            raise RuntimeError(
                f"task stream mismatch: seed {seed_index:02d}")
        if (result["handover_potential_fingerprint"] !=
                handover_fingerprint):
            raise RuntimeError(
                f"handover CRN mismatch: seed {seed_index:02d}")
        output = {
            "schema_version": "warehouse_stage23_locked_test_v1",
            "claim_boundary": (
                "Held-out randomized mixed curriculum; no state reset and "
                "no post-freeze model selection."),
            "training_seed_index": seed_index,
            "selected_checkpoint_update": job["selected_update"],
            "weight_path": str(weight),
            "weight_sha256": job["weight_hash"],
            "analysis_protocol": str(CONFIG),
            "analysis_protocol_sha256": config_hash,
            "test_seed_start": seed_start,
            "test_episode_count": episode_count,
            "evaluation_protocol": {
                "arrival_schedule": scenario_cfg["arrival_schedule"],
                "state_reset_at_phase_boundary":
                scenario_cfg["state_reset_at_phase_boundary"],
                "handover_sampling": scenario_cfg["handover_sampling"],
                "observation_variant": method["observation_variant"],
                "reward_contract": method["reward_contract"],
            },
            "result": result,
            "safety_and_completeness_passed": True,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        status_rows.append({
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
        "task_fingerprint": task_fingerprint,
        "handover_potential_fingerprint": handover_fingerprint,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
