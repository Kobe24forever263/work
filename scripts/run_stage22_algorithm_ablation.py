#!/usr/bin/env python3
"""Plan or execute one Stage 22 algorithm-ablation training seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np  # Bind the macOS OpenMP runtime before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage22_algorithm_ablation.yaml")
WARM = ROOT / "scripts" / "run_stage14_policy_smoke.py"
PPO = ROOT / "scripts" / "run_stage14_smdp_ppo.py"


def command_text(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted(
        (run_dir / "checkpoints").glob("stage22_ppo_update_*.pt"))
    return paths[-1] if paths else None


def validate_resume(path: Path, *, condition: dict, run_seed: int,
                    ppo_seed: int, validation_seed: int,
                    target_updates: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed = sorted(condition["allowed_transport_modes"])
    expected = {
        "execution_mode": "CONCURRENT",
        "arrival_profile": "MIXED_CURRICULUM",
        "observation_variant": condition["observation_variant"],
        "policy_variant": condition["policy_variant"],
        "discount_mode": condition["discount_mode"],
        "allowed_transport_modes": (
            allowed if len(allowed) < 3 else None),
        "experiment_run_seed": run_seed,
        "base_seed": ppo_seed,
        "validation_seed_start": validation_seed,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    update = int(payload.get("update", -1))
    if not 0 < update < target_updates:
        mismatches["update"] = {
            "expected": f"1..{target_updates - 1}", "actual": update}
    if "reward_normalizer" not in payload:
        mismatches["reward_normalizer"] = {
            "expected": "present", "actual": "missing"}
    if mismatches:
        raise ValueError(
            "incompatible Stage 22 resume checkpoint:\n" +
            json.dumps(mismatches, ensure_ascii=False, indent=2))
    return {
        "checkpoint": str(path.resolve()),
        "update": update,
        "remaining_updates": target_updates - update,
        "exact_rng_state_available": all(
            payload.get(key) is not None for key in (
                "numpy_random_state", "torch_random_state",
                "ppo_generator_state")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    conditions = protocol["conditions"]
    parser.add_argument("condition", choices=tuple(conditions))
    parser.add_argument("seed_index", type=int)
    parser.add_argument("--short-gate", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--phase", choices=("warm-start", "ppo", "all"),
                        default="all")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()

    condition = conditions[args.condition]
    training = protocol["training"]
    smoke = protocol["smoke"]
    seeds = (smoke["independent_run_base_seeds"] if args.short_gate else
             training["independent_run_base_seeds"])
    if not 1 <= args.seed_index <= len(seeds):
        parser.error(f"seed_index must be in [1, {len(seeds)}]")
    if args.execute and not args.short_gate and not args.confirm_long:
        parser.error("formal execution requires --confirm-long")
    if args.phase == "warm-start" and not condition["warm_start"]:
        parser.error("no_warm_start has no warm-start phase")

    run_seed = int(seeds[args.seed_index - 1])
    warm_seed = run_seed + int(training["warm_start_seed_offset"])
    ppo_seed = run_seed + int(training["ppo_seed_offset"])
    validation_seed = int(
        smoke["validation_seed_start"] if args.short_gate else
        protocol["validation"]["seed_start"])
    updates = int(args.updates if args.updates is not None else
                  smoke["updates"] if args.short_gate else
                  training["formal_updates"])
    if updates <= 0:
        parser.error("updates must be positive")

    lane = "short_gate" if args.short_gate else "long"
    run_dir = (ROOT / "results" / "stage22_algorithm_ablation" / lane /
               args.condition / f"seed_{args.seed_index:02d}")
    warm_weight = run_dir / "warm_start.pt"
    final_weight = run_dir / "stage22_ppo.pt"
    # The short gate still has to clear the existing imitation-safety gate.
    # Five epochs over three seeds proved too noisy (86.6% mode agreement),
    # so use the smallest stable smoke budget without changing formal runs.
    warm_epochs = 10 if args.short_gate else 30
    warm_seed_count = 5 if args.short_gate else int(
        training["warm_start_seed_count"])
    eval_episodes = int(
        smoke["validation_seed_count"] if args.short_gate else
        protocol["validation"]["seed_count"])
    episodes_per_update = int(
        smoke["episodes_per_update"] if args.short_gate else
        training["episodes_per_update"])
    eval_every = updates if args.short_gate else 50
    checkpoint_every = updates if args.short_gate else 50
    allowed = list(condition["allowed_transport_modes"])
    restricted = len(allowed) < 3

    warm_command = None
    if condition["warm_start"]:
        warm_command = [
            sys.executable, str(WARM),
            "--epochs", str(warm_epochs), "--batch-size", "32",
            "--learning-rate", "0.001",
            "--execution-mode", "CONCURRENT",
            "--arrival-profile", "MIXED_CURRICULUM",
            "--run-seed", str(warm_seed),
            "--training-seed-count", str(warm_seed_count),
            "--validation-seed-start", str(validation_seed),
            "--validation-episodes", str(eval_episodes),
            "--observation-variant", condition["observation_variant"],
            "--policy-variant", condition["policy_variant"],
            "--output", str(warm_weight),
        ]
        if restricted:
            warm_command.extend(["--allowed-transport-modes", *allowed])

    resume_path = latest_checkpoint(run_dir) if args.resume_latest else None
    resume_audit = None
    if resume_path is not None:
        resume_audit = validate_resume(
            resume_path, condition=condition, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_seed,
            target_updates=updates)

    ppo_command = [
        sys.executable, str(PPO),
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", "MIXED_CURRICULUM",
        "--observation-variant", condition["observation_variant"],
        "--policy-variant", condition["policy_variant"],
        "--discount-mode", condition["discount_mode"],
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
        "--allow-ablation-outcome",
        "--output", str(final_weight),
    ]
    if restricted:
        ppo_command.extend(["--allowed-transport-modes", *allowed])
    if resume_path is not None:
        ppo_command.extend(["--resume", str(resume_path)])
    elif condition["warm_start"]:
        ppo_command.extend(["--warm-start", str(warm_weight)])

    plan = {
        "stage": 22,
        "protocol": protocol["protocol"]["version"],
        "gate": ("ALGORITHM_ABLATION_SHORT_GATE" if args.short_gate else
                 "ALGORITHM_ABLATION_FORMAL_TRAINING"),
        "claim_boundary": (
            "Interface and safety only; no performance claim."
            if args.short_gate else
            "Training only; fresh locked test follows cohort freeze."),
        "condition": args.condition,
        "isolates": condition["isolates"],
        "condition_config": condition,
        "seed_index": args.seed_index,
        "experiment_run_seed": run_seed,
        "ppo_seed": ppo_seed,
        "validation_seed_start": validation_seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "planned_episodes": updates * episodes_per_update,
        "tasks_per_episode": 80,
        "warm_start_enabled": bool(condition["warm_start"]),
        "run_dir": str(run_dir),
        "warm_start_weight": (
            str(warm_weight) if condition["warm_start"] else None),
        "final_weight": str(final_weight),
        "best_weight": str(final_weight.with_suffix(".best.pt")),
        "history": str(final_weight.with_suffix(".history.json")),
        "resume_audit": resume_audit,
        "warm_start_command": (
            command_text(warm_command) if warm_command else None),
        "ppo_command": command_text(ppo_command),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if (args.phase in {"warm-start", "all"} and warm_command is not None and
            resume_path is None):
        subprocess.run(warm_command, cwd=ROOT, check=True)
    if args.phase in {"ppo", "all"}:
        if (condition["warm_start"] and resume_path is None and
                not warm_weight.exists()):
            raise FileNotFoundError(
                f"warm-start weight not found: {warm_weight}")
        subprocess.run(ppo_command, cwd=ROOT, check=True)

    summary = final_weight.with_suffix(".summary.json")
    status = {
        "stage": 22,
        "condition": args.condition,
        "seed_index": args.seed_index,
        "short_gate": args.short_gate,
        "run_plan": str(run_dir / "run_plan.json"),
        "summary": str(summary),
        "completed": summary.exists(),
    }
    (run_dir / "stage22_run.status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
