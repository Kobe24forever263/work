#!/usr/bin/env python3
"""Verify exact Stage 23 terminal weights across an interrupted resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np  # Bind the macOS OpenMP runtime before torch.
import torch


ROOT = Path(__file__).resolve().parents[1]
PPO = ROOT / "scripts" / "run_stage14_smdp_ppo.py"
BASE = (ROOT / "results" / "stage23_markov_continuous" /
        "resume_equivalence")
WARM = (ROOT / "results" / "stage23_markov_continuous" / "pilot" /
        "seed_01" / "warm_start.pt")
OUTPUT = BASE / "stage23_resume_equivalence_gate.json"


def command(output: Path, updates: int, resume: Path | None = None) -> list[str]:
    result = [
        sys.executable, str(PPO),
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", "MIXED_CURRICULUM",
        "--observation-variant", "MARKOV_CONTEXT_V3",
        "--policy-variant", "CONTEXT_INTERACTION",
        "--discount-mode", "VARIABLE_SMDP",
        "--trace-mode", "DURATION_SCALED",
        "--trace-tau-s", "10.0",
        "--reward-contract", "CONTINUOUS_TIME_V3",
        "--updates", str(updates),
        "--episodes-per-update", "1",
        "--ppo-epochs", "2",
        "--minibatch-size", "32",
        "--seed", "68190000",
        "--experiment-run-seed", "68189999",
        "--validation-seed-start", "68191000",
        "--eval-episodes", "1",
        "--eval-every", "4",
        "--checkpoint-every", "2",
        "--log-every", "2",
        "--allow-ablation-outcome",
        "--output", str(output),
    ]
    if resume is None:
        result.extend(["--warm-start", str(WARM)])
    else:
        result.extend(["--resume", str(resume)])
    return result


def equal_nested(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys() and
                all(equal_nested(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, type(left)) and len(left) == len(right) and
                all(equal_nested(a, b) for a, b in zip(left, right)))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def run(step: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        step, cwd=ROOT, env=env, check=False,
        capture_output=True, text=True)
    return {
        "command": step,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1200:],
        "stderr_tail": completed.stderr[-1200:],
    }


def main() -> int:
    if not WARM.exists():
        raise FileNotFoundError(f"pilot warm-start is missing: {WARM}")
    BASE.mkdir(parents=True, exist_ok=True)
    uninterrupted = BASE / "uninterrupted.pt"
    split = BASE / "split.pt"
    steps = []
    steps.append(run(command(uninterrupted, 4)))
    steps.append(run(command(split, 2)))
    checkpoint = BASE / "checkpoints" / "split_update_00002.pt"
    if steps[-1]["returncode"] == 0 and checkpoint.exists():
        steps.append(run(command(split, 4, checkpoint)))

    commands_pass = len(steps) == 3 and all(
        item["returncode"] == 0 for item in steps)
    checks = {"commands_pass": commands_pass}
    if commands_pass:
        whole = torch.load(
            uninterrupted, map_location="cpu", weights_only=False)
        resumed = torch.load(split, map_location="cpu", weights_only=False)
        checks.update({
            "terminal_update_equal": whole["update"] == resumed["update"] == 4,
            "episodes_trained_equal": (
                whole["episodes_trained"] == resumed["episodes_trained"] == 4),
            "model_state_exact": equal_nested(
                whole["model_state_dict"], resumed["model_state_dict"]),
            "optimizer_state_exact": equal_nested(
                whole["optimizer_state_dict"],
                resumed["optimizer_state_dict"]),
            "reward_normalizer_exact": equal_nested(
                whole["reward_normalizer"], resumed["reward_normalizer"]),
            "numpy_rng_exact": equal_nested(
                whole["numpy_random_state"], resumed["numpy_random_state"]),
            "torch_rng_exact": equal_nested(
                whole["torch_random_state"], resumed["torch_random_state"]),
            "ppo_generator_rng_exact": equal_nested(
                whole["ppo_generator_state"],
                resumed["ppo_generator_state"]),
        })
    passed = all(checks.values())
    report = {
        "schema_version": "warehouse_stage23_resume_equivalence_v1",
        "stage": 23,
        "gate": "EXACT_INTERRUPTED_RESUME_EQUIVALENCE",
        "claim_boundary": (
            "Four-update deterministic engineering test; not a learning "
            "performance claim."),
        "commands": steps,
        "checks": checks,
        "passed": passed,
        "uninterrupted_weight": str(uninterrupted),
        "resumed_weight": str(split),
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
