#!/usr/bin/env python3
"""Run one-update interface smokes for every Stage 17 ablation condition."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN = WORK_ROOT / "scripts" / "run_stage14_smdp_ppo.py"
OUTPUT = WORK_ROOT / "results" / "stage17" / "smoke"
CONDITIONS = {
    "full_context_v2": {},
    "no_queue_resource": {
        "--observation-variant": "NO_QUEUE_RESOURCE"},
    "no_handover_cues": {
        "--observation-variant": "NO_HANDOVER_CUES"},
    "no_persistent_position": {
        "--observation-variant": "NO_PERSISTENT_POSITION"},
    "fixed_gamma": {"--discount-mode": "FIXED_PER_DECISION"},
    "flat_masked_ppo": {"--policy-variant": "FLAT_MASKED_PPO"},
}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, options in CONDITIONS.items():
        weight = OUTPUT / f"stage17_{name}_smoke.pt"
        command = [
            sys.executable, str(TRAIN),
            "--execution-mode", "CONCURRENT",
            "--arrival-profile", "MEDIUM",
            "--updates", "1",
            "--episodes-per-update", "1",
            "--ppo-epochs", "1",
            "--eval-episodes", "2",
            "--eval-every", "1",
            "--checkpoint-every", "1",
            "--log-every", "1",
            "--seed", "41000020",
            "--validation-seed-start", "42000000",
            "--output", str(weight),
        ]
        for option, value in options.items():
            command.extend((option, value))
        subprocess.run(command, cwd=WORK_ROOT, check=True)
        summary_path = weight.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "condition": name,
            "summary": str(summary_path),
            "observation_variant": summary["observation_variant"],
            "policy_variant": summary["policy_variant"],
            "discount_mode": summary["discount_mode"],
            "action_width": summary["model"]["action_width"],
            "passed": summary["passed"],
        })
    assertions = {
        "all_six_conditions_ran": len(rows) == 6,
        "all_condition_smokes_passed": all(row["passed"] for row in rows),
        "all_use_explicit_30d_candidate_schema": all(
            row["action_width"] == 30 for row in rows),
        "all_use_shared_validation_seeds": all(
            json.loads(Path(row["summary"]).read_text(encoding="utf-8"))[
                "validation_seed_start"] == 42000000 for row in rows),
    }
    report = {
        "stage": 17,
        "gate": "ALL_ABLATION_SHORT_TRAINING_SMOKES",
        "claim_boundary": "One-update interface smokes; no performance claim.",
        "conditions": rows,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = OUTPUT.parent / "stage17_ablation_smokes.summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output), "condition_count": len(rows),
        "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
