#!/usr/bin/env python3
"""Plan or execute one Stage 20 mixed-curriculum training seed.

Formal long runs require both --execute and --confirm-long so an automated
smoke command cannot accidentally start a multi-hour campaign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np  # Load the environment BLAS/OpenMP runtime before torch on macOS.
import torch
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
SEED_CONFIG = (
    WORK_ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage20_mixed_curriculum.yaml")
WARM_SCRIPT = WORK_ROOT / "scripts" / "run_stage14_policy_smoke.py"
PPO_SCRIPT = WORK_ROOT / "scripts" / "run_stage14_smdp_ppo.py"


def command_text(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("stage20_ppo_update_*.pt"))
    return checkpoints[-1] if checkpoints else None


def validate_resume(path: Path, *, target_updates: int, run_seed: int,
                    ppo_seed: int, validation_seed: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "execution_mode": "CONCURRENT",
        "arrival_profile": "MIXED_CURRICULUM",
        "observation_variant": "FULL_CONTEXT_V2",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP",
        "experiment_run_seed": run_seed,
        "base_seed": ppo_seed,
        "validation_seed_start": validation_seed,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items() if payload.get(key) != value
    }
    update = int(payload.get("update", -1))
    if not 0 < update < target_updates:
        mismatches["update"] = {
            "expected": f"1..{target_updates - 1}", "actual": update}
    if "reward_normalizer" not in payload:
        mismatches["reward_normalizer"] = {
            "expected": "present", "actual": "missing"}
    metadata = payload.get("model_metadata", {})
    for key, value in {"state_width": 284, "action_width": 30,
                       "hidden_width": 96}.items():
        if metadata.get(key) != value:
            mismatches[f"model_metadata.{key}"] = {
                "expected": value, "actual": metadata.get(key)}
    if mismatches:
        raise ValueError(
            "resume checkpoint is incompatible:\n" +
            json.dumps(mismatches, ensure_ascii=False, indent=2))
    return {
        "checkpoint": str(path.resolve()),
        "update": update,
        "target_updates": target_updates,
        "remaining_updates": target_updates - update,
        "exact_rng_state_available": all(
            payload.get(key) is not None for key in (
                "numpy_random_state", "torch_random_state",
                "ppo_generator_state")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_index", type=int)
    parser.add_argument("--short-gate", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--phase", choices=("warm-start", "ppo", "all"),
                        default="all")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    training = protocol["training"]
    seeds = training["independent_run_base_seeds"]
    if not 1 <= args.seed_index <= len(seeds):
        parser.error(f"seed_index must be in [1, {len(seeds)}]")
    if args.execute and not args.short_gate and not args.confirm_long:
        parser.error("formal execution requires --confirm-long")

    run_seed = int(seeds[args.seed_index - 1])
    warm_seed = run_seed + int(training["warm_start_seed_offset"])
    ppo_seed = run_seed + int(training["ppo_seed_offset"])
    validation_seed = int(
        protocol["smoke"]["seed_start"] if args.short_gate else
        protocol["validation"]["seed_start"])
    updates = int(args.updates if args.updates is not None else
                  5 if args.short_gate else training["formal_updates"])
    if updates <= 0:
        parser.error("updates must be positive")

    lane = "short_gate" if args.short_gate else "long"
    run_dir = (
        WORK_ROOT / "results" / "stage20_mixed_curriculum" / lane /
        f"seed_{args.seed_index:02d}")
    warm_weight = run_dir / "warm_start.pt"
    final_weight = run_dir / "stage20_ppo.pt"
    warm_epochs = 15 if args.short_gate else 30
    warm_seed_count = 5 if args.short_gate else int(
        training["warm_start_seed_count"])
    eval_episodes = 4 if args.short_gate else int(
        protocol["validation"]["seed_count"])
    episodes_per_update = 1 if args.short_gate else int(
        training["episodes_per_update"])
    eval_every = updates if args.short_gate else 50
    checkpoint_every = updates if args.short_gate else 50

    warm_command = [
        sys.executable, str(WARM_SCRIPT),
        "--epochs", str(warm_epochs), "--batch-size", "32",
        "--learning-rate", "0.001",
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", "MIXED_CURRICULUM",
        "--run-seed", str(warm_seed),
        "--training-seed-count", str(warm_seed_count),
        "--validation-seed-start", str(validation_seed),
        "--validation-episodes", str(eval_episodes),
        "--observation-variant", "FULL_CONTEXT_V2",
        "--policy-variant", "CONTEXT_INTERACTION",
        "--output", str(warm_weight),
    ]
    resume_path = latest_checkpoint(run_dir) if args.resume_latest else None
    resume_audit = None
    if resume_path is not None:
        resume_audit = validate_resume(
            resume_path, target_updates=updates, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_seed)
    ppo_command = [
        sys.executable, str(PPO_SCRIPT),
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", "MIXED_CURRICULUM",
        "--observation-variant", "FULL_CONTEXT_V2",
        "--policy-variant", "CONTEXT_INTERACTION",
        "--discount-mode", "VARIABLE_SMDP",
        "--updates", str(updates),
        "--episodes-per-update", str(episodes_per_update),
        "--ppo-epochs", "4", "--minibatch-size", "32",
        "--learning-rate", "0.0003", "--clip-ratio", "0.2",
        "--gae-lambda", "0.95", "--entropy-coefficient", "0.01",
        "--value-coefficient", "0.5",
        "--seed", str(ppo_seed),
        "--experiment-run-seed", str(run_seed),
        "--validation-seed-start", str(validation_seed),
        "--eval-episodes", str(eval_episodes),
        "--eval-every", str(eval_every),
        "--checkpoint-every", str(checkpoint_every),
        "--log-every", str(min(20, updates)),
        "--output", str(final_weight),
    ]
    if resume_path is not None:
        ppo_command.extend(["--resume", str(resume_path)])
    else:
        ppo_command.extend(["--warm-start", str(warm_weight)])

    plan = {
        "stage": 20,
        "gate": "MIXED_CURRICULUM_SHORT_GATE" if args.short_gate else
                "MIXED_CURRICULUM_FORMAL_TRAINING",
        "claim_boundary": (
            "Interface, schedule diversity, safety, and persistence only."
            if args.short_gate else
            "Training only; fresh locked test required after ten-weight freeze."),
        "seed_index": args.seed_index,
        "experiment_run_seed": run_seed,
        "arrival_profile": "MIXED_CURRICULUM",
        "execution_mode": "CONCURRENT",
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "planned_episodes": updates * episodes_per_update,
        "tasks_per_episode": 80,
        "run_dir": str(run_dir),
        "warm_start_weight": str(warm_weight),
        "final_weight": str(final_weight),
        "best_weight": str(final_weight.with_suffix(".best.pt")),
        "history": str(final_weight.with_suffix(".history.json")),
        "resume_audit": resume_audit,
        "warm_start_command": command_text(warm_command),
        "ppo_command": command_text(ppo_command),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if args.phase in {"warm-start", "all"} and resume_path is None:
        subprocess.run(warm_command, cwd=WORK_ROOT, check=True)
    if args.phase in {"ppo", "all"}:
        if resume_path is None and not warm_weight.exists():
            raise FileNotFoundError(f"warm-start weight not found: {warm_weight}")
        subprocess.run(ppo_command, cwd=WORK_ROOT, check=True)

    summary_path = final_weight.with_suffix(".summary.json")
    result = {
        "stage": 20,
        "seed_index": args.seed_index,
        "short_gate": args.short_gate,
        "run_plan": str(run_dir / "run_plan.json"),
        "summary": str(summary_path),
        "completed": summary_path.exists(),
    }
    (run_dir / "stage20_run.status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
