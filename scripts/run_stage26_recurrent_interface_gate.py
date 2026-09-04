#!/usr/bin/env python3
"""Run the Stage 26 trainability smoke and independent fresh-seed gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "stage26_recurrent_causal" / "smoke" / "v1"
WEIGHT = RUN_DIR / "stage26_recurrent_ppo.pt"
SUMMARY = WEIGHT.with_suffix(".summary.json")
VALIDATION = RUN_DIR / "stage26_recurrent_fresh_validation.json"
OUTPUT = (ROOT / "results" / "stage26_recurrent_causal" / "gates" /
          "stage26_recurrent_interface_gate.json")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train = [
        sys.executable,
        str(ROOT / "scripts" / "run_stage26_recurrent_ppo.py"),
        "--updates", "1",
        "--episodes-per-update", "1",
        "--ppo-epochs", "1",
        "--eval-episodes", "2",
        "--eval-every", "1",
        "--checkpoint-every", "1",
        "--log-every", "1",
        "--seed", "72910000",
        "--experiment-run-seed", "72910000",
        "--validation-seed-start", "72920000",
        "--output", str(WEIGHT),
    ]
    subprocess.run(train, cwd=ROOT, check=True)
    validate = [
        sys.executable,
        str(ROOT / "scripts" / "run_stage26_recurrent_validation.py"),
        str(WEIGHT),
        "--seed-start", "72930000",
        "--episodes", "2",
        "--output", str(VALIDATION),
    ]
    subprocess.run(validate, cwd=ROOT, check=True)
    training = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fresh = json.loads(VALIDATION.read_text(encoding="utf-8"))
    model = training.get("model", {})
    assertions = {
        "training_smoke_passed": training.get("passed") is True,
        "fresh_safety_validation_passed": fresh.get("passed") is True,
        "architecture_is_recurrent": model.get("architecture") ==
        "recurrent_contextual_masked_candidate_actor_critic_v1",
        "memory_resets_at_episode_boundary": (
            model.get("memory_reset_boundary") == "EPISODE" and
            training["memory_diagnostics"]["episode_reset_zero_max"] == 0.0),
        "memory_changes_during_episode": (
            training["memory_diagnostics"]["memory_update_l2_mean"] > 0.0 and
            fresh["policy"]["memory_update_l2_mean"] > 0.0),
        "parameters_updated": training["parameter_delta_l2"] > 0.0,
        "policy_actions_all_legal": (
            fresh["policy"]["illegal_action_count"] == 0),
        "policy_has_no_resource_leak": (
            fresh["policy"]["resource_leak_count"] == 0),
        "policy_resolved_all_tasks": fresh["policy"]["unresolved"] == 0,
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    result = {
        "stage": 26,
        "gate": "RECURRENT_CAUSAL_INTERFACE_AND_TRAINABILITY_GATE",
        "claim_boundary": (
            "Short interface/trainability gate only. Contextual switching "
            "and rule-baseline superiority are diagnostics, not smoke pass "
            "requirements."),
        "training_summary": str(SUMMARY),
        "fresh_validation": str(VALIDATION),
        "fresh_policy_minus_rule": fresh["policy_minus_rule"],
        "fresh_contextual_mode_acceptance":
            fresh["contextual_mode_acceptance"],
        "assertions": assertions,
        "passed": all(assertions.values()),
        "next_action": (
            "ALLOW_MANUAL_STAGE26_SENTINEL_LONG_TRAINING"
            if all(assertions.values()) else
            "STOP_AND_REPAIR_RECURRENT_INTERFACE"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
