#!/usr/bin/env python3
"""Generate the locked 30-run Stage 18 manual-training manifest."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
PLANNER = WORK_ROOT / "scripts" / "run_stage18_train_one.py"


def main() -> int:
    conditions = (
        "full_context_v2", "no_queue_resource", "no_handover_cues",
        "no_persistent_position", "fixed_gamma", "flat_masked_ppo")
    runs = []
    for condition in conditions:
        for seed_index in range(1, 6):
            completed = subprocess.run([
                sys.executable, str(PLANNER), condition, str(seed_index),
                "--phase", "all", "--updates", "2000"],
                cwd=WORK_ROOT, check=True, text=True, capture_output=True)
            plan = json.loads(completed.stdout)
            plan["manual_launch"] = (
                f"{sys.executable} {PLANNER} {condition} {seed_index} "
                "--phase all --updates 2000 --execute")
            runs.append(plan)
    ppo_ranges = [set(range(
        row["ppo_seed_range"][0], row["ppo_seed_range"][1] + 1))
        for row in runs[:5]]
    assertions = {
        "six_conditions_times_five_seeds": len(runs) == 30,
        "all_output_weights_unique": len({
            row["final_weight"] for row in runs}) == 30,
        "five_training_seed_ranges_disjoint": all(
            not (left & right)
            for index, left in enumerate(ppo_ranges)
            for right in ppo_ranges[index + 1:]),
        "all_conditions_share_paired_seed_indices": all(
            [row["experiment_run_seed"] for row in runs
             if row["condition"] == condition] ==
            [row["experiment_run_seed"] for row in runs
             if row["condition"] == "full_context_v2"]
            for condition in conditions),
    }
    manifest = {
        "stage": 18,
        "protocol": "tro_stage18_30run_v1",
        "execution": "MANUAL_ONE_RUN_AT_A_TIME",
        "condition_count": len(conditions),
        "training_seed_count": 5,
        "run_count": len(runs),
        "runs": runs,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = (WORK_ROOT / "results" / "stage18" /
              "stage18_training_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output), "run_count": len(runs),
        "passed": manifest["passed"]}, ensure_ascii=False, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
