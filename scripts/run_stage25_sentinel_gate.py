#!/usr/bin/env python3
"""Run and adjudicate the frozen Stage 25 sentinel quality gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage25_causal_online.yaml")
RUN_DIR = (ROOT / "results" / "stage25_causal_online" / "long" /
           "seed_01")
WEIGHT = RUN_DIR / "stage25_causal_ppo.pt"
TRAINING_SUMMARY = WEIGHT.with_suffix(".summary.json")
VALIDATION = RUN_DIR / "sentinel_fresh_validation.json"
OUTPUT = RUN_DIR / "sentinel_quality_gate.json"


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not WEIGHT.exists() or not TRAINING_SUMMARY.exists():
        raise FileNotFoundError(
            "sentinel weight/summary missing; run seed 01 long training first")
    validation = protocol["validation"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_stage25_causal_validation.py"),
        str(WEIGHT),
        "--seed-start", str(validation["online_validation_seed_start"]),
        "--episodes", str(validation["online_validation_episode_count"]),
        "--output", str(VALIDATION),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    training = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))
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
        "fresh_contextual_switching_gate_passed": (
            fresh["contextual_mode_acceptance"]["passed"] is True),
        "fresh_mean_reward_not_below_rule_baseline": (
            fresh["policy_minus_rule"]["mean_reward"] >= 0.0),
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    result = {
        "stage": 25,
        "gate": "CAUSAL_RELEASED_V4_SENTINEL_EXPANSION_GATE",
        "claim_boundary": (
            "Controls whether seeds 02-10 may start. It is not a formal "
            "locked-test superiority claim."),
        "training_summary": str(TRAINING_SUMMARY),
        "fresh_validation": str(VALIDATION),
        "fresh_policy_minus_rule": fresh["policy_minus_rule"],
        "fresh_contextual_mode_acceptance":
            fresh["contextual_mode_acceptance"],
        "assertions": assertions,
        "passed": all(assertions.values()),
        "next_action": (
            "ALLOW_SEEDS_02_TO_10" if all(assertions.values()) else
            "STOP_AND_TEST_RECURRENT_POLICY"),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

