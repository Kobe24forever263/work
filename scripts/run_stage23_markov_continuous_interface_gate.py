#!/usr/bin/env python3
"""Aggregate the corrected Stage 23 interface and provenance gates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python")
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage23_markov_continuous.yaml")
EVENT_AUDIT = (ROOT / "results" / "stage23_event_partition" /
               "stage23_event_partition_audit.json")
STATE_AUDIT = (ROOT / "results" / "stage23_state_sufficiency" /
               "stage23_state_sufficiency_audit.json")
SMOKE_DIR = ROOT / "results" / "stage23_markov_continuous" / "smoke"
WARM_SUMMARY = SMOKE_DIR / "stage23_markov_v3_warm_start.summary.json"
PPO_SUMMARY = SMOKE_DIR / "stage23_markov_v3_warm_ppo_smoke.summary.json"
OUTPUT = (ROOT / "results" / "stage23_markov_continuous" /
          "stage23_markov_continuous_interface_gate.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_set(start: int, count: int) -> set[int]:
    return set(range(start, start + count))


def main() -> int:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    commands = [
        [str(PYTHON), str(ROOT / "scripts" /
                            "run_stage23_event_partition_audit.py")],
        [str(PYTHON), str(ROOT / "scripts" /
                            "run_stage23_state_sufficiency_audit.py")],
    ]
    command_results = []
    for command in commands:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, check=False,
            capture_output=True, text=True)
        command_results.append({
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        })

    test_command = [
        str(PYTHON), "-m", "pytest", "-p", "no:cacheprovider",
        "src/warehouse_core/test/test_core.py", "-q",
    ]
    tests = subprocess.run(
        test_command, cwd=ROOT, env=env, check=False,
        capture_output=True, text=True)
    match = re.search(r"(\d+) passed", tests.stdout)
    test_count = int(match.group(1)) if match else 0

    event = load_json(EVENT_AUDIT)
    state = load_json(STATE_AUDIT)
    warm = load_json(WARM_SUMMARY)
    ppo = load_json(PPO_SUMMARY)
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    method = protocol["method"]

    event_hashes_current = all(
        Path(path).exists() and sha256(Path(path)) == expected
        for path, expected in event["source_sha256"].items())
    state_source = Path(state["source_path"])
    state_hash_current = (
        state_source.exists() and sha256(state_source) ==
        state["source_sha256"])

    training = protocol["training"]
    pilot = protocol["pilot"]
    validation = protocol["validation"]
    locked = protocol["locked_test"]
    training_episode_seeds = set()
    warm_seeds = set()
    for base in training["independent_run_base_seeds"]:
        warm_seeds |= seed_set(
            int(base) + int(training["warm_start_seed_offset"]),
            int(training["warm_start_seed_count"]))
        training_episode_seeds |= seed_set(
            int(base) + int(training["ppo_seed_offset"]),
            int(training["episodes_per_run"]))
    validation_seeds = seed_set(
        int(validation["seed_start"]), int(validation["seed_count"]))
    locked_seeds = seed_set(
        int(locked["seed_start"]), int(locked["seed_count"]))
    pilot_seeds = set()
    for base in pilot["independent_run_base_seeds"]:
        pilot_seeds |= seed_set(
            int(base), int(pilot["warm_start_seed_count"]))
        pilot_seeds |= seed_set(
            int(base) + int(pilot["ppo_seed_offset"]),
            int(pilot["updates"]) * int(pilot["episodes_per_update"]))
    pilot_seeds |= seed_set(
        int(pilot["validation_seed_start"]),
        int(pilot["validation_seed_count"]))
    partitions = {
        "training_episode_seed_count": len(training_episode_seeds),
        "warm_start_seed_count": len(warm_seeds),
        "validation_seed_count": len(validation_seeds),
        "locked_test_seed_count": len(locked_seeds),
        "pilot_seed_count": len(pilot_seeds),
        "all_partitions_disjoint": not any((
            training_episode_seeds & warm_seeds,
            training_episode_seeds & validation_seeds,
            training_episode_seeds & locked_seeds,
            training_episode_seeds & pilot_seeds,
            warm_seeds & validation_seeds,
            warm_seeds & locked_seeds,
            warm_seeds & pilot_seeds,
            validation_seeds & locked_seeds,
            validation_seeds & pilot_seeds,
            locked_seeds & pilot_seeds,
        )),
    }

    warm_weight = str(
        (SMOKE_DIR / "stage23_markov_v3_warm_start.pt").resolve())
    checks = {
        "audit_commands_succeeded": all(
            item["returncode"] == 0 for item in command_results),
        "core_tests_pass": tests.returncode == 0 and test_count >= 55,
        "event_partition_gate_passes": (
            event["audit_execution_gate"]["passed"] and
            event["scientific_assessment"]["status"] ==
            "PASS_CORRECTED_CONTINUOUS_TIME_V3"),
        "event_audit_hashes_match_current_sources": event_hashes_current,
        "markov_alias_gate_passes": (
            state["audit_execution_gate"]["passed"] and
            state["scientific_assessment"]
            ["markov_context_v3_resolves_constructive_alias_gate"]),
        "state_audit_hash_matches_current_source": state_hash_current,
        "warm_start_gate_passes": bool(warm["passed"]),
        "warm_start_covers_all_three_modes": bool(
            warm["assertions"]["heldout_uses_all_three_modes"]),
        "warm_started_ppo_smoke_passes": bool(ppo["passed"]),
        "ppo_uses_markov_context_v3": (
            ppo["observation_variant"] == method["observation_variant"] and
            ppo["model"]["state_width"] == 1720),
        "ppo_uses_continuous_time_contract": (
            ppo["reward_contract"] == method["reward_contract"] and
            ppo["discount_mode"] == method["discount_mode"] and
            ppo["trace_mode"] == method["trace_mode"] and
            ppo["trace_tau_s"] == float(method["trace_tau_s"])),
        "ppo_uses_corrected_return_normalizer": (
            ppo["reward_normalization"]["source"] ==
            "reverse_variable_discounted_return_to_go_std_v2"),
        "ppo_warm_start_provenance_matches": (
            ppo["warm_start_source"] == warm_weight),
        "ppo_resolves_all_tasks": (
            ppo["assertions"]["evaluation_resolved_all_tasks"]),
        "ppo_has_no_illegal_action_or_resource_leak": (
            ppo["assertions"]["evaluation_actions_all_legal"] and
            ppo["assertions"]["evaluation_has_no_resource_leak"]),
        "seed_partitions_are_disjoint": partitions[
            "all_partitions_disjoint"],
    }
    passed = all(checks.values())
    report = {
        "schema_version": "warehouse_stage23_interface_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 23,
        "gate": "MARKOV_CONTINUOUS_INTERFACE_AND_PROVENANCE_GATE",
        "protocol": protocol["protocol"]["version"],
        "claim_boundary": (
            "Interface, deterministic-contract, provenance and safety gate "
            "only; not a convergence or performance claim."),
        "method": method,
        "command_results": command_results,
        "core_tests": {
            "command": test_command,
            "returncode": tests.returncode,
            "passed_count": test_count,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        },
        "seed_partitions": partitions,
        "evidence": {
            "event_audit": str(EVENT_AUDIT),
            "state_audit": str(STATE_AUDIT),
            "warm_start_summary": str(WARM_SUMMARY),
            "warm_ppo_summary": str(PPO_SUMMARY),
            "protocol_config": str(CONFIG),
        },
        "checks": checks,
        "passed": passed,
        "next_gate": (
            "THREE_SEED_OPTIMIZATION_PILOT" if passed else
            "REPAIR_FAILED_INTERFACE_CHECKS"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
