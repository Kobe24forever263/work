#!/usr/bin/env python3
"""Freeze the Stage 21 fault protocol before locked-test seed use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage21_fault_robustness.yaml")
STAGE20_FREEZE = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json")
SMOKE = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_smoke.json")
PREFLIGHT = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_campaign" / "preflight")
LOCKED = (
    ROOT / "results" / "stage21_external_validity" /
    "fault_robustness_campaign" / "locked_v1")
OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "stage21_fault_protocol_freeze_manifest.json")
FROZEN_FILES = (
    "src/warehouse_core/warehouse_core/stage14_training.py",
    "src/warehouse_core/warehouse_core/stage21_faults.py",
    "src/warehouse_bringup/config/experiment_seeds_stage21_fault_robustness.yaml",
    "scripts/run_stage15_stress_evaluation.py",
    "scripts/run_stage21_fault_robustness.py",
    "scripts/run_stage21_fault_locked_campaign.py",
    "scripts/summarize_stage21_fault_locked.py",
    "scripts/run_stage21_fault_locked.command",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_range(section: dict) -> set[int]:
    start = int(section["seed_start"])
    return set(range(start, start + int(section["seed_count"])))


def load_preflight(fault: str, name: str) -> dict:
    path = PREFLIGHT / fault.lower() / name
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row["result"]
    if not (
            row.get("campaign") == "preflight" and
            row.get("fault_scenario") == fault and
            row.get("episode_count") == 2 and
            result.get("task_count") == 160 and
            result.get("completed", 0) + result.get("failed", 0) == 160 and
            result.get("illegal_action_count") == 0 and
            result.get("resource_leak_count") == 0):
        raise RuntimeError(f"invalid preflight output: {path}")
    return result


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    stage20 = json.loads(STAGE20_FREEZE.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    faults = protocol["execution"]["fault_scenarios"]
    preflight = {}
    for fault in faults:
        preflight[fault] = {
            "PPO": load_preflight(fault, "ppo_seed_01.json"),
            "TIME_GREEDY_RULE": load_preflight(
                fault, "time_greedy_rule.json"),
            "SINGLE_DOG_ONLY": load_preflight(
                fault, "single_dog_only.json"),
        }

    shared = {
        fault: {
            "task": len({result["task_fingerprint"]
                         for result in methods.values()}) == 1,
            "fault": len({result["fault_fingerprint"]
                          for result in methods.values()}) == 1,
        } for fault, methods in preflight.items()}
    seed_sets = {
        key: integer_range(protocol[key])
        for key in ("smoke", "preflight", "locked_test")}
    seed_disjoint = all(
        seed_sets[left].isdisjoint(seed_sets[right])
        for left, right in (
            ("smoke", "preflight"), ("smoke", "locked_test"),
            ("preflight", "locked_test")))

    frozen_files = [{
        "path": str(ROOT / relative),
        "sha256": sha256(ROOT / relative),
    } for relative in FROZEN_FILES]
    weights = []
    for row in stage20["runs"]:
        path = Path(row["best_weight_path"])
        weights.append({
            "training_seed_index": int(row["training_seed_index"]),
            "path": str(path),
            "expected_sha256": row["best_weight_sha256"],
            "actual_sha256": sha256(path),
        })
    locked_outputs = list(LOCKED.rglob("*.json")) if LOCKED.exists() else []
    assertions = {
        "stage20_freeze_gate_passed": bool(
            stage20.get("hard_gate", {}).get("passed")),
        "ten_frozen_weights_present": len(weights) == 10,
        "all_weight_hashes_match": all(
            row["expected_sha256"] == row["actual_sha256"]
            for row in weights),
        "fault_smoke_gate_passed": bool(smoke.get("passed")),
        "all_preflight_jobs_present": sum(
            len(methods) for methods in preflight.values()) == 12,
        "preflight_task_streams_shared": all(
            item["task"] for item in shared.values()),
        "preflight_fault_schedules_shared": all(
            item["fault"] for item in shared.values()),
        "seed_namespaces_disjoint": seed_disjoint,
        "formal_scope_matches_protocol": (
            protocol["formal_scope"]["stage20_weight_count"] == 10 and
            protocol["formal_scope"]["ppo_job_count"] == 40 and
            protocol["formal_scope"]["baseline_job_count"] == 8 and
            protocol["formal_scope"]["total_task_count"] == 384000),
        "locked_test_seed_outputs_absent_before_freeze": not locked_outputs,
    }
    payload = {
        "schema_version": "warehouse_stage21_fault_protocol_freeze_v1",
        "stage": 21,
        "campaign": protocol["campaign"],
        "protocol": protocol,
        "frozen_files": frozen_files,
        "frozen_weights": weights,
        "stage20_freeze_manifest": str(STAGE20_FREEZE),
        "stage20_freeze_manifest_sha256": sha256(STAGE20_FREEZE),
        "smoke_result": str(SMOKE),
        "smoke_result_sha256": sha256(SMOKE),
        "preflight_root": str(PREFLIGHT),
        "preflight_shared_fingerprints": shared,
        "locked_outputs_seen_before_freeze": [
            str(path) for path in locked_outputs],
        "hard_gate": {
            "assertions": assertions,
            "passed": all(assertions.values()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "assertions": assertions,
        "passed": payload["hard_gate"]["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
