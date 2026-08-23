#!/usr/bin/env python3
"""Evaluate frozen policies on one continuous four-phase load trajectory."""
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
sys.path.insert(0, str(ROOT / "src/warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))
from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402
from run_stage15_stress_evaluation import SCENARIOS, run_policy  # noqa: E402

CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage19_mixed_recovery.yaml"
FREEZE = ROOT / "results/stage19_mixed_recovery/stage19_full_policy_freeze_manifest.json"
PHASES = ("NORMAL", "DENSE", "BURST", "RECOVERY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adaptation_summary(result: dict) -> dict:
    phase_results = result["phase_results"]
    mode_shares = {}
    for phase in PHASES:
        row = phase_results[phase]
        denominator = max(row["task_count"], 1)
        mode_shares[phase] = {
            mode: count / denominator
            for mode, count in row["transport_modes"].items()
        }
    early_wait, late_wait = [], []
    for episode in result["episodes"]:
        recovery = sorted(
            (row for row in episode["tasks"] if row["phase"] == "RECOVERY"),
            key=lambda row: row["arrival_time"])
        split = len(recovery) // 2
        early_wait.extend(row["waiting_time"] for row in recovery[:split])
        late_wait.extend(row["waiting_time"] for row in recovery[split:])
    return {
        "transport_mode_shares_by_phase": mode_shares,
        "recovery_early_mean_waiting_time": float(np.mean(early_wait)),
        "recovery_late_mean_waiting_time": float(np.mean(late_wait)),
        "recovery_wait_change_late_minus_early": (
            float(np.mean(late_wait)) - float(np.mean(early_wait))),
        "mode_share_changes": {
            "burst_minus_normal": {
                mode: mode_shares["BURST"].get(mode, 0.0) -
                      mode_shares["NORMAL"].get(mode, 0.0)
                for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")},
            "recovery_minus_burst": {
                mode: mode_shares["RECOVERY"].get(mode, 0.0) -
                      mode_shares["BURST"].get(mode, 0.0)
                for mode in ("SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR")},
        },
    }


def safety_passes(result: dict, episode_count: int) -> bool:
    phases = result.get("phase_results", {})
    return bool(
        result.get("illegal_action_count") == 0 and
        result.get("resource_leak_count") == 0 and
        len(result.get("episodes", [])) == episode_count and
        set(phases) == set(PHASES) and
        all(phases[phase].get("task_count") == episode_count * 20
            for phase in PHASES) and
        result.get("task_count") == episode_count * 80)


def completed_output_is_valid(path: Path, profile: str, training_seed: int,
                              weight_hash: str, seed_start: int,
                              episode_count: int) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        row.get("schema_version") == "warehouse_stage19_mixed_recovery_v1" and
        row.get("training_profile") == profile and
        row.get("training_seed_index") == training_seed and
        row.get("weight_sha256") == weight_hash and
        row.get("test_seed_start") == seed_start and
        row.get("test_episode_count") == episode_count and
        row.get("safety_and_completeness_passed") is True and
        safety_passes(row.get("result", {}), episode_count))


def save_status(out: Path, campaign: str, seed_start: int, count: int,
                rows: list[dict], state: str,
                baseline_output: str | None = None) -> dict:
    status = {
        "schema_version": "warehouse_stage19_campaign_status_v1",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign,
        "test_seed_start": seed_start,
        "test_episode_count": count,
        "jobs": rows,
        "baseline_output": baseline_output,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign.status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=("MEDIUM", "DENSE", "BURST"),
                        default=("MEDIUM", "DENSE", "BURST"))
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not frozen.get("hard_gate", {}).get("passed"):
        raise RuntimeError("Stage 19 policy freeze gate has not passed")
    frozen_rows = {
        (row["training_profile"], int(row["training_seed_index"])): row
        for row in frozen["runs"]}
    seed_cfg = config["smoke" if args.smoke else "test"]
    seed_start, count = int(seed_cfg["seed_start"]), int(seed_cfg["seed_count"])
    seeds = list(range(seed_start, seed_start + count))
    campaign = "smoke" if args.smoke else "locked_test_v1"
    out = ROOT / "results/stage19_mixed_recovery" / campaign
    jobs = [(profile, seed) for profile in args.profiles for seed in range(1, 11)]
    print(json.dumps({
        "stage": 19, "campaign": campaign, "execute": args.execute,
        "profiles": list(args.profiles), "policy_jobs": len(jobs),
        "test_seed_range": [seeds[0], seeds[-1]],
        "episodes_per_job": count, "tasks_per_episode": 80,
        "total_policy_tasks": len(jobs) * count * 80,
        "baselines": ["TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"],
    }, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    device = torch.device("cpu")
    torch.set_num_threads(4)
    scenario = SCENARIOS["MIXED_CONTINUOUS_RECOVERY"]
    status_rows = []
    out.mkdir(parents=True, exist_ok=True)
    save_status(out, campaign, seed_start, count, status_rows, "RUNNING")
    progress = tqdm(jobs, desc=f"Stage19 {campaign}", unit="model", dynamic_ncols=True)
    for profile, training_seed in progress:
        row = frozen_rows[(profile, training_seed)]
        checkpoint = Path(row["best_weight_path"])
        if sha256(checkpoint) != row["best_weight_sha256"]:
            raise RuntimeError(f"frozen weight changed: {checkpoint}")
        target = out / profile.lower() / f"seed_{training_seed:02d}.json"
        if completed_output_is_valid(
                target, profile, training_seed, row["best_weight_sha256"],
                seed_start, count):
            status_rows.append({"profile": profile, "training_seed": training_seed,
                                "state": "SKIPPED_ALREADY_COMPLETE",
                                "output": str(target)})
            save_status(out, campaign, seed_start, count, status_rows, "RUNNING")
            continue
        model, payload = load_checkpoint(checkpoint, device)
        if (payload.get("arrival_profile") != profile or
                payload.get("observation_variant") != "FULL_CONTEXT_V2"):
            raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint}")
        result = run_policy(model, seeds, scenario, "PPO", "CONCURRENT")
        if not safety_passes(result, count):
            raise RuntimeError(f"Stage 19 safety/completeness gate failed: {profile}/{training_seed}")
        output = {
            "schema_version": "warehouse_stage19_mixed_recovery_v1",
            "claim_boundary": "Continuous mixed-load evaluation; no phase reset.",
            "training_profile": profile,
            "training_seed_index": training_seed,
            "weight_path": str(checkpoint),
            "weight_sha256": row["best_weight_sha256"],
            "test_seed_start": seed_start,
            "test_episode_count": count,
            "evaluation_protocol": {
                "arrival_schedule": "NORMAL_DENSE_BURST_RECOVERY_CONTINUOUS",
                "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
                "state_reset_at_phase_boundary": False,
            },
            "result": result,
            "adaptation": adaptation_summary(result),
            "safety_and_completeness_passed": True,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        status_rows.append({"profile": profile, "training_seed": training_seed,
                            "state": "COMPLETED", "output": str(target)})
        save_status(out, campaign, seed_start, count, status_rows, "RUNNING")

    baseline_model = None
    baselines = {
        "time_greedy_rule": run_policy(
            baseline_model, seeds, scenario, "TIME_GREEDY_RULE", "CONCURRENT"),
        "single_dog_only": run_policy(
            baseline_model, seeds, scenario, "TIME_GREEDY_RULE", "CONCURRENT",
            allowed_transport_modes=("SINGLE_DOG",)),
    }
    for name, result in baselines.items():
        if not safety_passes(result, count):
            raise RuntimeError(f"Stage 19 baseline gate failed: {name}")
    baseline_path = out / "baselines.json"
    baseline_path.write_text(json.dumps({
        "schema_version": "warehouse_stage19_mixed_recovery_baselines_v1",
        "test_seed_start": seed_start, "test_episode_count": count,
        "evaluation_protocol": {
            "arrival_schedule": "NORMAL_DENSE_BURST_RECOVERY_CONTINUOUS",
            "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
            "state_reset_at_phase_boundary": False,
        },
        **{name: {"result": result, "adaptation": adaptation_summary(result)}
           for name, result in baselines.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = save_status(out, campaign, seed_start, count, status_rows,
                         "COMPLETED", str(baseline_path))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
