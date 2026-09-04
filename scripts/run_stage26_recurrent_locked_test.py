#!/usr/bin/env python3
"""Freeze and evaluate the ten Stage 26 recurrent policies on locked seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Bind the Apple runtime before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage26_recurrent_causal.yaml")
LONG_ROOT = ROOT / "results" / "stage26_recurrent_causal" / "long"
DEFAULT_OUTPUT = (ROOT / "results" / "stage26_recurrent_causal" /
                  "locked_test_v1")
SCHEMA = "warehouse_stage26_recurrent_locked_test_v1"

sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage25_causal_observation import CAUSAL_RELEASED_V4
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


def checkpoint_inventory(protocol: dict) -> list[dict]:
    expected = protocol["cohort"]["independent_training_base_seeds"]
    inventory = []
    for index, base_seed in enumerate(expected, start=1):
        run_dir = LONG_ROOT / f"seed_{index:02d}"
        checkpoint = run_dir / "stage26_recurrent_ppo.pt"
        summary_path = run_dir / "stage26_recurrent_ppo.summary.json"
        if not checkpoint.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"seed {index:02d} checkpoint or summary is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("passed"):
            raise ValueError(f"seed {index:02d} training summary did not pass")
        if int(summary.get("updates", 0)) != 2000 or \
                int(summary.get("episodes_trained_total", 0)) != 8000:
            raise ValueError(f"seed {index:02d} training budget mismatch")
        if int(summary.get("experiment_run_seed", -1)) != int(base_seed):
            raise ValueError(f"seed {index:02d} base seed mismatch")
        inventory.append({
            "training_seed_index": index,
            "base_seed": int(base_seed),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "training_summary": str(summary_path.resolve()),
        })
    return inventory


def freeze_cohort(output_root: Path, protocol: dict,
                  inventory: list[dict]) -> dict:
    validation = protocol["validation"]
    manifest = {
        "schema_version": "warehouse_stage26_recurrent_freeze_v1",
        "stage": 26,
        "purpose": (
            "Freeze all ten final checkpoints before the formal locked test; "
            "locked-test outcomes must not be used for checkpoint selection."),
        "protocol": protocol["protocol"]["version"],
        "training_seed_count": len(inventory),
        "locked_test_seed_start": int(
            validation["formal_locked_seed_start"]),
        "locked_test_seed_count": int(
            validation["formal_locked_seed_count"]),
        "checkpoints": inventory,
    }
    path = output_root / "cohort_freeze_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "frozen cohort or locked protocol changed; refusing to "
                "overwrite the formal namespace")
    else:
        write_json(path, manifest)
    return manifest


def valid_existing(path: Path, frozen: dict, seed_start: int,
                   seed_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        row.get("schema_version") == SCHEMA and
        row.get("checkpoint_sha256") == frozen["checkpoint_sha256"] and
        row.get("locked_test_seed_start") == seed_start and
        row.get("locked_test_seed_count") == seed_count and
        row.get("passed") is True)


def aggregate_delta(policy: dict, baseline: dict) -> dict:
    return {
        "success_rate": policy["success_rate"] - baseline["success_rate"],
        "mean_reward": policy["mean_reward"] - baseline["mean_reward"],
        "mean_simulated_time": (
            policy["mean_simulated_time"] -
            baseline["mean_simulated_time"]),
        "successful_throughput_tasks_per_hour": (
            policy["successful_throughput_tasks_per_hour"] -
            baseline["successful_throughput_tasks_per_hour"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_locked_test:
        parser.error("formal evaluation requires --confirm-locked-test")

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validation = protocol["validation"]
    seed_start = int(validation["formal_locked_seed_start"])
    seed_count = int(validation["formal_locked_seed_count"])
    test_seeds = list(range(seed_start, seed_start + seed_count))
    output_root = args.output_root.expanduser().resolve()
    inventory = checkpoint_inventory(protocol)
    freeze = freeze_cohort(output_root, protocol, inventory)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    common = dict(
        execution_mode="CONCURRENT", profile="MIXED_CURRICULUM",
        device=device, reward_contract="CONTINUOUS_TIME_V3")

    first_model, first_payload = load_recurrent_checkpoint(
        inventory[0]["checkpoint"], device)
    if first_payload.get("observation_variant") != CAUSAL_RELEASED_V4:
        raise ValueError("seed 01 observation contract mismatch")
    print("Evaluating the shared rule baseline on 100 locked episodes...",
          flush=True)
    baseline = evaluate_recurrent(
        first_model, test_seeds, use_rule=True, **common)

    completed = []
    for frozen in inventory:
        index = frozen["training_seed_index"]
        output = output_root / f"seed_{index:02d}.json"
        if valid_existing(output, frozen, seed_start, seed_count):
            print(f"seed {index:02d}/10: existing locked result passed; skip",
                  flush=True)
            completed.append(index)
            continue
        print(f"seed {index:02d}/10: evaluating recurrent policy...",
              flush=True)
        model, payload = load_recurrent_checkpoint(
            frozen["checkpoint"], device)
        if payload.get("observation_variant") != CAUSAL_RELEASED_V4:
            raise ValueError(f"seed {index:02d} observation mismatch")
        if int(payload.get("base_seed", -1)) != frozen["base_seed"]:
            raise ValueError(f"seed {index:02d} checkpoint seed mismatch")
        policy = evaluate_recurrent(model, test_seeds, **common)
        expected_order = test_seeds
        policy_order = [row["seed"] for row in policy["episodes"]]
        baseline_order = [row["seed"] for row in baseline["episodes"]]
        assertions = {
            "checkpoint_matches_pretest_freeze": (
                sha256(Path(frozen["checkpoint"])) ==
                frozen["checkpoint_sha256"]),
            "shared_test_seed_order_matches": (
                policy_order == baseline_order == expected_order),
            "policy_resolved_all_tasks": policy["unresolved"] == 0,
            "policy_actions_all_legal": (
                policy["illegal_action_count"] == 0),
            "policy_has_no_resource_leak": (
                policy["resource_leak_count"] == 0),
            "baseline_resolved_all_tasks": baseline["unresolved"] == 0,
            "baseline_actions_all_legal": (
                baseline["illegal_action_count"] == 0),
            "baseline_has_no_resource_leak": (
                baseline["resource_leak_count"] == 0),
        }
        assertions = {key: bool(value) for key, value in assertions.items()}
        result = {
            "schema_version": SCHEMA,
            "stage": 26,
            "gate": "RECURRENT_CAUSAL_FORMAL_LOCKED_TEST",
            "claim_boundary": (
                "Final frozen cohort evaluation on previously unseen locked "
                "seeds. No checkpoint selection may use these results."),
            "protocol": freeze["protocol"],
            "training_seed_index": index,
            "training_base_seed": frozen["base_seed"],
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "locked_test_seed_start": seed_start,
            "locked_test_seed_end": seed_start + seed_count - 1,
            "locked_test_seed_count": seed_count,
            "pairing": "TASK_KEYED shared locked episode seeds",
            "policy": policy,
            "rule_baseline": baseline,
            "policy_minus_rule": aggregate_delta(policy, baseline),
            "assertions": assertions,
            "passed": all(assertions.values()),
        }
        write_json(output, result)
        if not result["passed"]:
            raise RuntimeError(f"seed {index:02d} locked safety gate failed")
        completed.append(index)
        print(
            f"seed {index:02d}/10: pass, "
            f"success={policy['success_rate']:.3%}, "
            f"reward_delta={result['policy_minus_rule']['mean_reward']:+.3f}",
            flush=True)

    status = {
        "schema_version": "warehouse_stage26_locked_campaign_status_v1",
        "stage": 26,
        "completed_training_seed_indices": completed,
        "expected_training_seed_count": 10,
        "locked_test_seed_count": seed_count,
        "passed": completed == list(range(1, 11)),
    }
    write_json(output_root / "campaign_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
