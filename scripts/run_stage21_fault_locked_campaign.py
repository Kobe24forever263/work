#!/usr/bin/env python3
"""Resumable Stage 21 all-weight locked fault-robustness campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Stable Apple Silicon load order before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))

from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402
from warehouse_core.stage21_faults import FAULT_SCENARIOS  # noqa: E402
from run_stage15_stress_evaluation import run_policy  # noqa: E402
from run_stage21_fault_robustness import (  # noqa: E402
    SCENARIO, env_factory)


CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage21_fault_robustness.yaml")
STAGE20_FREEZE = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json")
STAGE21_FREEZE = (
    ROOT / "results" / "stage21_external_validity" /
    "stage21_fault_protocol_freeze_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safety_complete(result: dict, episode_count: int) -> bool:
    task_count = episode_count * SCENARIO.task_count
    return bool(
        result.get("episode_count") == episode_count and
        result.get("task_count") == task_count and
        result.get("completed", 0) + result.get("failed", 0) == task_count and
        result.get("illegal_action_count") == 0 and
        result.get("resource_leak_count") == 0 and
        len(result.get("episodes", [])) == episode_count and
        all(len(episode.get("tasks", [])) == SCENARIO.task_count
            for episode in result.get("episodes", [])))


def output_valid(path: Path, *, method: str, fault: str,
                 episode_count: int, seed_start: int,
                 weight_hash: str = "") -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = row.get("result", {})
    return bool(
        row.get("schema_version") ==
        "warehouse_stage21_fault_locked_job_v1" and
        row.get("method") == method and
        row.get("fault_scenario") == fault and
        row.get("seed_start") == seed_start and
        row.get("episode_count") == episode_count and
        row.get("weight_sha256", "") == weight_hash and
        result.get("task_fingerprint") and
        result.get("fault_fingerprint") and
        safety_complete(result, episode_count))


def save_status(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def progress(label: str):
    def report(index: int, total: int, episode: dict) -> None:
        if index == 1 or index == total or index % 10 == 0:
            print(
                f"{label}: {index}/{total} episodes "
                f"success={episode['success_rate']:.2%} "
                f"wait={episode['mean_waiting_time']:.1f}s",
                flush=True)
    return report


def run_method(model, seeds: list[int], fault: str, method: str,
               observation_variant: str):
    allowed = ("SINGLE_DOG",) if method == "SINGLE_DOG_ONLY" else None
    policy_kind = "PPO" if method == "PPO" else "TIME_GREEDY_RULE"
    result = run_policy(
        model, seeds, SCENARIO, policy_kind, "CONCURRENT",
        observation_variant=observation_variant,
        allowed_transport_modes=allowed,
        env_factory=env_factory(fault),
        progress_callback=progress(f"{fault}/{method}"))
    result["policy_kind"] = method
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument(
        "--groups", nargs="+", choices=("BASELINES", "PPO"),
        default=("BASELINES", "PPO"))
    parser.add_argument("--weight-start", type=int, default=1)
    parser.add_argument("--weight-end", type=int, default=1)
    parser.add_argument(
        "--faults", nargs="+", choices=FAULT_SCENARIOS,
        default=list(FAULT_SCENARIOS))
    args = parser.parse_args()
    if not 1 <= args.weight_start <= args.weight_end <= 10:
        parser.error("weight range must be within 1..10")
    if (args.execute and not args.preflight and
            not args.confirm_locked_test):
        parser.error("formal seeds are run-once; add --confirm-locked-test")

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed_cfg = protocol["preflight" if args.preflight else "locked_test"]
    seed_start = int(seed_cfg["seed_start"])
    episode_count = int(seed_cfg["seed_count"])
    seeds = list(range(seed_start, seed_start + episode_count))
    campaign = "preflight" if args.preflight else "locked_v1"
    out = (
        ROOT / "results" / "stage21_external_validity" /
        "fault_robustness_campaign" / campaign)

    stage20 = json.loads(STAGE20_FREEZE.read_text(encoding="utf-8"))
    if not stage20.get("hard_gate", {}).get("passed"):
        raise RuntimeError("Stage 20 weight freeze gate has not passed")
    if not args.preflight:
        if not STAGE21_FREEZE.exists():
            raise RuntimeError("Stage 21 protocol freeze manifest is missing")
        freeze = json.loads(STAGE21_FREEZE.read_text(encoding="utf-8"))
        if not freeze.get("hard_gate", {}).get("passed"):
            raise RuntimeError("Stage 21 protocol freeze gate has not passed")
        changed = [
            row["path"] for row in freeze.get("frozen_files", [])
            if not Path(row["path"]).exists() or
            sha256(Path(row["path"])) != row["sha256"]]
        if changed:
            raise RuntimeError(
                f"Stage 21 frozen protocol files changed: {changed}")
        if sha256(STAGE20_FREEZE) != freeze.get(
                "stage20_freeze_manifest_sha256"):
            raise RuntimeError("Stage 20 freeze manifest changed after freeze")

    weights = []
    for row in stage20["runs"]:
        index = int(row["training_seed_index"])
        if args.weight_start <= index <= args.weight_end:
            weights.append({
                "index": index,
                "path": Path(row["best_weight_path"]),
                "sha256": row["best_weight_sha256"],
            })
    weights.sort(key=lambda row: row["index"])
    jobs = []
    if "BASELINES" in args.groups:
        jobs.extend({"method": method, "fault": fault}
                    for fault in args.faults
                    for method in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"))
    if "PPO" in args.groups:
        jobs.extend({"method": "PPO", "fault": fault, "weight": weight}
                    for fault in args.faults for weight in weights)
    plan = {
        "stage": 21,
        "campaign": campaign,
        "execute": args.execute,
        "groups": list(args.groups),
        "faults": list(args.faults),
        "weight_range": [args.weight_start, args.weight_end],
        "seed_range": [seeds[0], seeds[-1]],
        "episodes_per_job": episode_count,
        "job_count": len(jobs),
        "task_count": len(jobs) * episode_count * SCENARIO.task_count,
        "output_root": str(out),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0

    torch.set_num_threads(4)
    status_path = out / (
        f"campaign_{'_'.join(args.groups).lower()}_"
        f"{args.weight_start:02d}_{args.weight_end:02d}.status.json")
    status_jobs = []
    model_cache = {}
    for job_index, job in enumerate(jobs, start=1):
        method = job["method"]
        fault = job["fault"]
        weight = job.get("weight")
        suffix = (
            f"ppo_seed_{weight['index']:02d}.json" if weight else
            f"{method.lower()}.json")
        destination = out / fault.lower() / suffix
        weight_hash = weight["sha256"] if weight else ""
        if output_valid(
                destination, method=method, fault=fault,
                episode_count=episode_count, seed_start=seed_start,
                weight_hash=weight_hash):
            state = "SKIPPED_VALID"
            print(f"[{job_index}/{len(jobs)}] {fault}/{suffix}: {state}")
        else:
            if weight:
                actual = sha256(weight["path"])
                if actual != weight_hash:
                    raise RuntimeError(
                        f"frozen weight hash mismatch: {weight['path']}")
                if weight["index"] not in model_cache:
                    model_cache[weight["index"]] = load_checkpoint(
                        weight["path"], torch.device("cpu"))
                model, checkpoint = model_cache[weight["index"]]
            else:
                anchor = weights[0] if weights else {
                    "index": int(stage20["runs"][0]["training_seed_index"]),
                    "path": Path(stage20["runs"][0]["best_weight_path"]),
                }
                if anchor["index"] not in model_cache:
                    model_cache[anchor["index"]] = load_checkpoint(
                        anchor["path"], torch.device("cpu"))
                model, checkpoint = model_cache[anchor["index"]]
            model.eval()
            observation_variant = checkpoint.get(
                "observation_variant", "FULL_CONTEXT_V2")
            print(f"[{job_index}/{len(jobs)}] RUN {fault}/{suffix}")
            result = run_method(
                model, seeds, fault, method, observation_variant)
            row = {
                "schema_version": "warehouse_stage21_fault_locked_job_v1",
                "stage": 21,
                "campaign": campaign,
                "method": method,
                "fault_scenario": fault,
                "training_seed_index": (
                    weight["index"] if weight else None),
                "weight_path": str(weight["path"]) if weight else "",
                "weight_sha256": weight_hash,
                "seed_start": seed_start,
                "episode_count": episode_count,
                "result": result,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            if not output_valid(
                    destination, method=method, fault=fault,
                    episode_count=episode_count, seed_start=seed_start,
                    weight_hash=weight_hash):
                raise RuntimeError(f"new output failed validation: {destination}")
            state = "COMPLETED"
        status_jobs.append({
            "method": method, "fault": fault,
            "training_seed_index": weight["index"] if weight else None,
            "output": str(destination), "state": state})
        save_status(status_path, {
            **plan, "state": "RUNNING", "jobs": status_jobs})
    save_status(status_path, {
        **plan, "state": "COMPLETED", "jobs": status_jobs})
    print(json.dumps({
        "state": "COMPLETED", "status": str(status_path),
        "completed_jobs": len(status_jobs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
