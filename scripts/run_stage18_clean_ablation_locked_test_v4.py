#!/usr/bin/env python3
"""Run fresh TASK_KEYED locked tests for frozen DENSE/BURST cohorts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np  # Stable Apple Silicon OpenMP load order.
import torch
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/warehouse_core"))
from warehouse_core.stage14_ppo import evaluate, load_checkpoint  # noqa: E402

CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage18_clean_v4.yaml"
OUT = ROOT / "results/stage18_clean_ablation_locked_test_v4"
SUMMARY = ROOT / "scripts/summarize_stage18_clean_ablation_v4.py"
CONDITIONS = {
    "full_context_v2": "FULL_CONTEXT_V2",
    "no_position_only": "NO_POSITION_ONLY",
    "no_eta_cost_only": "NO_ETA_COST_ONLY",
    "no_history_only": "NO_HISTORY_ONLY",
    "no_explicit_queue_resource": "NO_EXPLICIT_QUEUE_RESOURCE",
    "no_explicit_handover_risk": "NO_EXPLICIT_HANDOVER_RISK",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_path(profile: str, condition: str, seed: int) -> Path:
    return OUT / profile.lower() / condition / f"seed_{seed:02d}.json"


def valid(path: Path, profile: str, condition: str, seed: int,
          test_start: int, test_count: int, weight_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    policy = row.get("policy", {})
    rule_ok = condition != "full_context_v2" or (
        len(row.get("rule_baseline", {}).get("episodes", [])) == test_count and
        row["rule_baseline"].get("resource_leak_count") == 0)
    return bool(
        row.get("schema_version") == "warehouse_stage18_clean_locked_test_v4" and
        row.get("evaluation_profile") == profile and row.get("condition") == condition and
        row.get("training_seed_index") == seed and row.get("test_seed_start") == test_start and
        row.get("test_episode_count") == test_count and row.get("weight_sha256") == weight_hash and
        row.get("evaluation_protocol", {}).get("handover_sampling") ==
            "TASK_KEYED_COMMON_RANDOM_NUMBERS" and
        len(policy.get("episodes", [])) == test_count and
        policy.get("illegal_action_count") == 0 and policy.get("resource_leak_count") == 0 and
        rule_ok)


def save_status(profile: str, rows: list[dict], state: str, start: int,
                count: int) -> None:
    path = OUT / profile.lower() / "campaign.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "campaign": "CLEAN_ABLATION_LOCKED_TEST_V4", "profile": profile,
        "state": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_seed_start": start, "test_episode_count": count,
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS", "jobs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frozen_hashes(profile: str, cfg: dict) -> dict[tuple[str, int], str]:
    manifest = ROOT / cfg["profiles"][profile]["freeze_manifest"]
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if not report.get("hard_gate", {}).get("passed"):
        raise RuntimeError(f"freeze gate failed: {manifest}")
    return {(row["condition"], int(row["seed_index"])): row["best_weight_sha256"]
            for row in report["runs"]}


def evaluate_single_dog(profile: str, cfg: dict, seeds: list[int],
                        device: torch.device) -> None:
    destination = OUT / profile.lower() / "baselines/single_dog_only.json"
    start, count = seeds[0], len(seeds)
    if destination.exists():
        try:
            old = json.loads(destination.read_text(encoding="utf-8"))
            result = old.get("single_dog_only", {})
            if (old.get("test_seed_start") == start and
                    len(result.get("episodes", [])) == count and
                    result.get("allowed_transport_modes") == ["SINGLE_DOG"] and
                    result.get("illegal_action_count") == 0 and
                    result.get("resource_leak_count") == 0):
                return
        except (OSError, json.JSONDecodeError):
            pass
    weight = ROOT / cfg["profiles"][profile]["training_root"] / \
        "full_context_v2/seed_01/stage18_ppo.best.pt"
    model, _ = load_checkpoint(weight, device)
    result = evaluate(model, seeds, "CONCURRENT", profile, device, use_rule=True,
                      observation_variant="FULL_CONTEXT_V2",
                      handover_sampling="TASK_KEYED",
                      allowed_transport_modes=("SINGLE_DOG",))
    payload = {
        "schema_version": "warehouse_stage18_single_dog_baseline_v4",
        "baseline": "SINGLE_DOG_ONLY_FOUR_GO2_FLEET",
        "evaluation_profile": profile, "test_seed_start": start,
        "test_episode_count": count,
        "evaluation_protocol": {"handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS"},
        "single_dog_only": result,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=("DENSE", "BURST"),
                        default=("DENSE", "BURST"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    count = int(cfg["test"]["seed_count"])
    plan = {profile: {"test_seed_start": cfg["profiles"][profile]["test_seed_start"],
                      "jobs": len(CONDITIONS) * 10} for profile in args.profiles}
    print(json.dumps({"stage": 18, "campaign": "CLEAN_ABLATION_LOCKED_TEST_V4",
                      "profiles": plan, "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    device = torch.device("cpu")
    torch.set_num_threads(4)
    for profile in args.profiles:
        start = int(cfg["profiles"][profile]["test_seed_start"])
        seeds = list(range(start, start + count))
        hashes = frozen_hashes(profile, cfg)
        rows: list[dict] = []
        save_status(profile, rows, "RUNNING", start, count)
        jobs = [(condition, seed) for condition in CONDITIONS for seed in range(1, 11)]
        progress = tqdm(jobs, desc=f"Stage18 v4 {profile} TASK_KEYED", unit="job",
                        dynamic_ncols=True)
        for condition, seed in progress:
            weight = ROOT / cfg["profiles"][profile]["training_root"] / condition / \
                f"seed_{seed:02d}/stage18_ppo.best.pt"
            weight_hash = sha256(weight)
            if weight_hash != hashes[(condition, seed)]:
                raise RuntimeError(f"frozen weight changed: {weight}")
            destination = result_path(profile, condition, seed)
            if valid(destination, profile, condition, seed, start, count, weight_hash):
                rows.append({"condition": condition, "seed": seed,
                             "state": "SKIPPED_ALREADY_COMPLETE"})
                save_status(profile, rows, "RUNNING", start, count)
                continue
            model, payload = load_checkpoint(weight, device)
            if (payload.get("arrival_profile") != profile or
                    payload.get("observation_variant") != CONDITIONS[condition]):
                raise ValueError(f"checkpoint metadata mismatch: {weight}")
            policy = evaluate(model, seeds, "CONCURRENT", profile, device,
                              observation_variant=CONDITIONS[condition],
                              handover_sampling="TASK_KEYED")
            rule = None
            if condition == "full_context_v2":
                rule = evaluate(model, seeds, "CONCURRENT", profile, device, use_rule=True,
                                observation_variant=CONDITIONS[condition],
                                handover_sampling="TASK_KEYED")
            row = {
                "schema_version": "warehouse_stage18_clean_locked_test_v4",
                "claim_boundary": "Fresh v4 test after frozen DENSE/BURST cohorts.",
                "training_profile": profile, "evaluation_profile": profile,
                "condition": condition, "training_seed_index": seed,
                "weight": str(weight), "weight_sha256": weight_hash,
                "checkpoint_update": payload.get("update"), "test_seed_start": start,
                "test_episode_count": count,
                "evaluation_protocol": {
                    "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
                    "pairing_key": "(test_seed, task_id, handover_kind)",
                    "randomness_pairing": "shared_across_policy_and_baselines",
                },
                "policy": policy,
            }
            if rule is not None:
                row["rule_baseline"] = rule
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            if not valid(destination, profile, condition, seed, start, count, weight_hash):
                raise RuntimeError(f"result validation failed: {destination}")
            rows.append({"condition": condition, "seed": seed, "state": "COMPLETED",
                         "output": str(destination)})
            save_status(profile, rows, "RUNNING", start, count)
        evaluate_single_dog(profile, cfg, seeds, device)
        save_status(profile, rows, "COMPLETED", start, count)
    subprocess.run([sys.executable, str(SUMMARY)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
