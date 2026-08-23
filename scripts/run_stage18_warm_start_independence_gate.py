#!/usr/bin/env python3
"""Prove two Stage 18 warm starts are independently seeded end to end."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
WARM_SCRIPT = WORK_ROOT / "scripts" / "run_stage14_policy_smoke.py"
OUTPUT = WORK_ROOT / "results" / "stage18" / "warm_start_independence_gate"
RUN_SEEDS = (31000000, 31010000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_seed in RUN_SEEDS:
        weight = OUTPUT / f"warm_start_seed_{run_seed}.pt"
        subprocess.run([
            sys.executable, str(WARM_SCRIPT),
            "--epochs", "30", "--batch-size", "32",
            "--learning-rate", "0.001",
            "--execution-mode", "CONCURRENT",
            "--run-seed", str(run_seed),
            "--training-seed-count", "20",
            "--validation-seed-start", "42000000",
            "--validation-episodes", "5",
            "--observation-variant", "FULL_CONTEXT_V2",
            "--policy-variant", "CONTEXT_INTERACTION",
            "--output", str(weight),
        ], cwd=WORK_ROOT, check=True)
        summary_path = weight.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "run_seed": run_seed,
            "training_seed_start": summary["training_seed_start"],
            "training_seed_end": summary["training_seed_end"],
            "validation_seeds": summary["evaluation_seeds"],
            "initial_loss": summary["training_initial"]["loss"],
            "final_loss": summary["training_final"]["loss"],
            "weight": str(weight),
            "weight_sha256": sha256(weight),
            "passed": summary["passed"],
        })
    first_range = set(range(
        rows[0]["training_seed_start"], rows[0]["training_seed_end"] + 1))
    second_range = set(range(
        rows[1]["training_seed_start"], rows[1]["training_seed_end"] + 1))
    assertions = {
        "both_warm_starts_passed": all(row["passed"] for row in rows),
        "training_data_seed_ranges_disjoint": not bool(
            first_range & second_range),
        "validation_seeds_shared": (
            rows[0]["validation_seeds"] == rows[1]["validation_seeds"]),
        "initial_metrics_differ": (
            rows[0]["initial_loss"] != rows[1]["initial_loss"]),
        "trained_weight_hashes_differ": (
            rows[0]["weight_sha256"] != rows[1]["weight_sha256"]),
    }
    report = {
        "stage": 18,
        "gate": "INDEPENDENT_END_TO_END_WARM_START_SEEDS",
        "claim_boundary": "Two-seed independence gate; not a PPO result.",
        "runs": rows,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = OUTPUT / "stage18_warm_start_independence_gate.summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output), "weight_hashes": [
            row["weight_sha256"] for row in rows],
        "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
