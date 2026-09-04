#!/usr/bin/env python3
"""Adjudicate the manually trained Stage 26 recurrent sentinel."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage26_recurrent_causal.yaml")
RUN_DIR = (ROOT / "results" / "stage26_recurrent_causal" / "long" /
           "seed_01")
WEIGHT = RUN_DIR / "stage26_recurrent_ppo.pt"
SUMMARY = WEIGHT.with_suffix(".summary.json")
VALIDATION = RUN_DIR / "sentinel_fresh_validation.json"
OUTPUT = RUN_DIR / "sentinel_quality_gate.json"


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not WEIGHT.exists() or not SUMMARY.exists():
        raise FileNotFoundError(
            "Stage 26 sentinel weight/summary missing; run seed 01 manually")
    validation = protocol["validation"]
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "run_stage26_recurrent_validation.py"),
        str(WEIGHT),
        "--seed-start", str(validation["online_validation_seed_start"]),
        "--episodes", str(validation["online_validation_episode_count"]),
        "--output", str(VALIDATION),
    ], cwd=ROOT, check=True)
    training = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fresh = json.loads(VALIDATION.read_text(encoding="utf-8"))
    assertions = {
        "training_safety_assertions_passed": training.get("passed") is True,
        "fresh_validation_safety_assertions_passed": fresh.get("passed") is True,
        "fresh_policy_resolved_all_tasks": (
            fresh["policy"]["unresolved"] == 0),
        "fresh_policy_actions_all_legal": (
            fresh["policy"]["illegal_action_count"] == 0),
        "fresh_policy_has_no_resource_leak": (
            fresh["policy"]["resource_leak_count"] == 0),
        "fresh_recurrent_memory_changes": (
            fresh["policy"]["memory_update_l2_mean"] > 0.0),
        "fresh_contextual_switching_gate_passed": (
            fresh["contextual_mode_acceptance"]["passed"] is True),
        "fresh_mean_reward_not_below_rule_baseline": (
            fresh["policy_minus_rule"]["mean_reward"] >= 0.0),
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    passed = all(assertions.values())
    result = {
        "stage": 26,
        "gate": "RECURRENT_CAUSAL_SENTINEL_EXPANSION_GATE",
        "claim_boundary": (
            "Controls seeds 02-10 only; it is not the final locked-test "
            "superiority claim."),
        "training_summary": str(SUMMARY),
        "fresh_validation": str(VALIDATION),
        "fresh_policy_minus_rule": fresh["policy_minus_rule"],
        "fresh_contextual_mode_acceptance":
            fresh["contextual_mode_acceptance"],
        "assertions": assertions,
        "passed": passed,
        "next_action": (
            "ALLOW_STAGE26_SEEDS_02_TO_10" if passed else
            "STOP_AND_REVIEW_RECURRENT_SENTINEL"),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
