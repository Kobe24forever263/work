#!/usr/bin/env python3
"""Evaluate a dog-only transport baseline on Stage18 v2 test seeds."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

# Keep NumPy before torch in the Apple Silicon Pixi runtime.
import numpy as np  # noqa: F401
import torch
from tqdm import tqdm
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import evaluate, load_checkpoint


SEED_CONFIG = (WORK_ROOT / "src" / "warehouse_bringup" / "config" /
               "experiment_seeds_stage18_v2.yaml")
OUTPUT_ROOT = (WORK_ROOT / "results" / "stage18_locked_test_v2" /
               "baselines" / "single_dog_only")
STATUS = OUTPUT_ROOT / "single_dog_only.status.json"
REFERENCE_WEIGHT = (WORK_ROOT / "results" / "stage18" /
                    "full_context_v2" / "seed_01" /
                    "stage18_ppo.best.pt")
PROFILES = ("MEDIUM", "DENSE", "BURST")
ALLOWED_MODES = ("SINGLE_DOG",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(profile: str) -> Path:
    return OUTPUT_ROOT / f"tested_{profile.lower()}.json"


def valid(path: Path, profile: str, test_start: int,
          test_count: int) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = row.get("single_dog_only", {})
    return bool(
        row.get("schema_version") ==
        "warehouse_stage18_single_dog_baseline_v1" and
        row.get("evaluation_profile") == profile and
        row.get("test_seed_start") == test_start and
        row.get("test_episode_count") == test_count and
        row.get("allowed_transport_modes") == ["SINGLE_DOG"] and
        result.get("episode_count") == test_count and
        len(result.get("episodes", [])) == test_count and
        result.get("handover_sampling") == "TASK_KEYED" and
        result.get("allowed_transport_modes") == ["SINGLE_DOG"] and
        not (set(result.get("transport_modes", {})) - {"SINGLE_DOG"}) and
        result.get("illegal_action_count") == 0 and
        result.get("resource_leak_count") == 0)


def save_status(state: str, rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": 18,
        "baseline": "SINGLE_DOG_ONLY_FOUR_GO2_FLEET",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    protocol = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    test_start = int(protocol["test"]["seed_start"])
    test_count = int(protocol["test"]["seed_count"])
    seeds = list(range(test_start, test_start + test_count))
    device = torch.device("cpu")
    torch.set_num_threads(4)
    if not REFERENCE_WEIGHT.exists():
        raise FileNotFoundError(REFERENCE_WEIGHT)
    model, _ = load_checkpoint(REFERENCE_WEIGHT, device)
    weight_hash = sha256(REFERENCE_WEIGHT)
    rows: list[dict] = []
    save_status("RUNNING", rows)
    progress = tqdm(
        PROFILES, desc="Stage18 SINGLE_DOG_ONLY", unit="load",
        dynamic_ncols=True)
    for profile in progress:
        progress.set_postfix(profile=profile)
        destination = output_path(profile)
        if valid(destination, profile, test_start, test_count):
            rows.append({
                "evaluation_profile": profile,
                "state": "SKIPPED_ALREADY_COMPLETE",
                "output": str(destination),
            })
            save_status("RUNNING", rows)
            continue
        row = {"evaluation_profile": profile, "state": "RUNNING"}
        rows.append(row)
        save_status("RUNNING", rows)
        result = evaluate(
            model, seeds, "CONCURRENT", profile, device,
            use_rule=True, observation_variant="FULL_CONTEXT_V2",
            handover_sampling="TASK_KEYED",
            allowed_transport_modes=ALLOWED_MODES)
        payload = {
            "schema_version": "warehouse_stage18_single_dog_baseline_v1",
            "claim_boundary": (
                "Fixed dog-only blank control evaluated on the same fresh "
                "Stage18 v2 paired test seeds.  Four Go2 robots remain in "
                "the resource pool; Carter and handover modes are disabled."),
            "baseline": "SINGLE_DOG_ONLY_FOUR_GO2_FLEET",
            "evaluation_profile": profile,
            "execution_mode": "CONCURRENT",
            "allowed_transport_modes": ["SINGLE_DOG"],
            "selection_rule": (
                "highest-ranked executable task, then minimum estimated "
                "SINGLE_DOG cost"),
            "test_seed_start": test_start,
            "test_episode_count": test_count,
            "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS",
            "reference_weight_for_interface_only": str(REFERENCE_WEIGHT),
            "reference_weight_sha256": weight_hash,
            "single_dog_only": result,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if not valid(destination, profile, test_start, test_count):
            row["state"] = "FAILED_VALIDATION"
            save_status("STOPPED_ON_FAILURE", rows)
            raise RuntimeError(f"single-dog validation failed: {destination}")
        row.update({
            "state": "COMPLETED",
            "output": str(destination),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save_status("RUNNING", rows)
    save_status("COMPLETED", rows)
    print(json.dumps({
        "state": "COMPLETED",
        "profiles": list(PROFILES),
        "status": str(STATUS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
