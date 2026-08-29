#!/usr/bin/env python3
"""Plan or manually execute one corrected Stage 23 training seed."""

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
          "experiment_seeds_stage23_markov_continuous.yaml")
WARM = ROOT / "scripts" / "run_stage14_policy_smoke.py"
PPO = ROOT / "scripts" / "run_stage14_smdp_ppo.py"


def command_text(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("stage23_ppo_update_*.pt"))
    return checkpoints[-1] if checkpoints else None


def valid_existing_warm_start(path: Path, *, run_seed: int,
                              validation_seed: int,
                              validation_count: int,
                              method: dict) -> dict | None:
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    interface = summary.get("interface_continuation_assertions", {})
    expected_validation = list(range(
        validation_seed, validation_seed + validation_count))
    valid = bool(
        interface and all(interface.values()) and
        summary.get("run_seed") == run_seed and
        summary.get("evaluation_seeds") == expected_validation and
        summary.get("observation_variant") ==
        method["observation_variant"] and
        summary.get("policy_variant") == method["policy_variant"] and
        summary.get("reward_contract") == method["reward_contract"] and
        summary.get("allowed_transport_modes") ==
        sorted(method["allowed_transport_modes"]))
    if not valid:
        return None
    return {
        "weight": str(path),
        "summary": str(summary_path),
        "strict_imitation_gate_passed": bool(summary.get("passed")),
        "interface_safety_gate_passed": True,
        "mode_agreement": summary.get(
            "evaluation_after", {}).get("mode_agreement"),
    }


def validate_resume(path: Path, *, protocol: dict, run_seed: int,
                    ppo_seed: int, validation_seed: int,
                    target_updates: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    method = protocol["method"]
    expected = {
        "execution_mode": protocol["training"]["execution_mode"],
        "arrival_profile": protocol["training"]["arrival_profile"],
        "observation_variant": method["observation_variant"],
        "policy_variant": method["policy_variant"],
        "discount_mode": method["discount_mode"],
        "trace_mode": method["trace_mode"],
        "trace_tau_s": float(method["trace_tau_s"]),
        "reward_contract": method["reward_contract"],
        "allowed_transport_modes": None,
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
    normalizer = payload.get("reward_normalizer", {})
    if normalizer.get("source") != (
            "reverse_variable_discounted_return_to_go_std_v2"):
        mismatches["reward_normalizer.source"] = {
            "expected": "reverse_variable_discounted_return_to_go_std_v2",
            "actual": normalizer.get("source"),
        }
    if mismatches:
        raise ValueError(
            "incompatible Stage 23 resume checkpoint:\n" +
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
    parser.add_argument("seed_index", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--phase", choices=("warm-start", "ppo", "all"),
                        default="all")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    method = protocol["method"]
    training = protocol["training"]
    lane = protocol["pilot"] if args.pilot else training
    seeds = lane["independent_run_base_seeds"]
    if not 1 <= args.seed_index <= len(seeds):
        parser.error(f"seed_index must be in [1, {len(seeds)}]")
    if args.execute and not args.pilot and not args.confirm_long:
        parser.error("formal execution requires --confirm-long")

    run_seed = int(seeds[args.seed_index - 1])
    warm_seed = run_seed + int(training["warm_start_seed_offset"])
    ppo_seed = run_seed + int(lane.get(
        "ppo_seed_offset", training["ppo_seed_offset"]))
    validation_seed = int(
        lane["validation_seed_start"] if args.pilot else
        protocol["validation"]["seed_start"])
    validation_count = int(
        lane["validation_seed_count"] if args.pilot else
        protocol["validation"]["seed_count"])
    target_updates = int(args.updates if args.updates is not None else
                         lane.get("updates", training["formal_updates"]))
    episodes_per_update = int(lane.get(
        "episodes_per_update", training["episodes_per_update"]))
    warm_seed_count = int(lane.get(
        "warm_start_seed_count", training["warm_start_seed_count"]))
    if target_updates <= 0:
        parser.error("updates must be positive")

    output_lane = "pilot" if args.pilot else "long"
    run_dir = (ROOT / "results" / "stage23_markov_continuous" /
               output_lane / f"seed_{args.seed_index:02d}")
    warm_weight = run_dir / "warm_start.pt"
    final_weight = run_dir / "stage23_ppo.pt"
    warm_epochs = 15 if args.pilot else 30
    eval_every = int(lane["evaluation_every_updates"])
    checkpoint_every = int(lane["checkpoint_every_updates"])

    warm_command = [
        sys.executable, str(WARM),
        "--epochs", str(warm_epochs), "--batch-size", "32",
        "--learning-rate", "0.001",
        "--execution-mode", training["execution_mode"],
        "--arrival-profile", training["arrival_profile"],
        "--run-seed", str(warm_seed),
        "--training-seed-count", str(warm_seed_count),
        "--validation-seed-start", str(validation_seed),
        "--validation-episodes", str(validation_count),
        "--observation-variant", method["observation_variant"],
        "--policy-variant", method["policy_variant"],
        "--reward-contract", method["reward_contract"],
        "--allow-interface-only-continuation",
        "--output", str(warm_weight),
    ]

    existing_warm_audit = valid_existing_warm_start(
        warm_weight, run_seed=warm_seed,
        validation_seed=validation_seed,
        validation_count=validation_count, method=method)

    resume_path = latest_checkpoint(run_dir) if args.resume_latest else None
    resume_audit = None
    if resume_path is not None:
        resume_audit = validate_resume(
            resume_path, protocol=protocol, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_seed,
            target_updates=target_updates)

    ppo = training["ppo"]
    ppo_command = [
        sys.executable, str(PPO),
        "--execution-mode", training["execution_mode"],
        "--arrival-profile", training["arrival_profile"],
        "--observation-variant", method["observation_variant"],
        "--policy-variant", method["policy_variant"],
        "--discount-mode", method["discount_mode"],
        "--trace-mode", method["trace_mode"],
        "--trace-tau-s", str(method["trace_tau_s"]),
        "--reward-contract", method["reward_contract"],
        "--updates", str(target_updates),
        "--episodes-per-update", str(episodes_per_update),
        "--ppo-epochs", str(ppo["epochs"]),
        "--minibatch-size", str(ppo["minibatch_size"]),
        "--learning-rate", str(ppo["learning_rate"]),
        "--clip-ratio", str(ppo["clip_ratio"]),
        "--gae-lambda", str(ppo["gae_lambda"]),
        "--entropy-coefficient", str(ppo["entropy_coefficient"]),
        "--value-coefficient", str(ppo["value_coefficient"]),
        "--seed", str(ppo_seed),
        "--experiment-run-seed", str(run_seed),
        "--validation-seed-start", str(validation_seed),
        "--eval-episodes", str(validation_count),
        "--eval-every", str(eval_every),
        "--checkpoint-every", str(checkpoint_every),
        "--log-every", str(min(20, target_updates)),
        "--allow-ablation-outcome",
        "--output", str(final_weight),
    ]
    if resume_path is not None:
        ppo_command.extend(["--resume", str(resume_path)])
    else:
        ppo_command.extend(["--warm-start", str(warm_weight)])

    plan = {
        "stage": 23,
        "protocol": protocol["protocol"]["version"],
        "gate": ("MARKOV_CONTINUOUS_PILOT" if args.pilot else
                 "MARKOV_CONTINUOUS_FORMAL_TRAINING"),
        "claim_boundary": (
            "Pilot optimization and provenance only; no performance claim."
            if args.pilot else
            "Training only; the fresh locked test follows cohort freeze."),
        "method": method,
        "seed_index": args.seed_index,
        "experiment_run_seed": run_seed,
        "ppo_seed": ppo_seed,
        "validation_seed_start": validation_seed,
        "validation_seed_count": validation_count,
        "updates": target_updates,
        "episodes_per_update": episodes_per_update,
        "planned_episodes": target_updates * episodes_per_update,
        "tasks_per_episode": training["tasks_per_episode"],
        "run_dir": str(run_dir),
        "warm_start_weight": str(warm_weight),
        "final_weight": str(final_weight),
        "best_weight": str(final_weight.with_suffix(".best.pt")),
        "history": str(final_weight.with_suffix(".history.json")),
        "resume_audit": resume_audit,
        "existing_warm_start_audit": existing_warm_audit,
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
    if (args.phase in {"warm-start", "all"} and resume_path is None and
            existing_warm_audit is None):
        subprocess.run(warm_command, cwd=ROOT, check=True)
    if args.phase in {"ppo", "all"}:
        if resume_path is None and not warm_weight.exists():
            raise FileNotFoundError(
                f"warm-start weight not found: {warm_weight}")
        subprocess.run(ppo_command, cwd=ROOT, check=True)

    summary = final_weight.with_suffix(".summary.json")
    status = {
        "stage": 23,
        "seed_index": args.seed_index,
        "pilot": args.pilot,
        "run_plan": str(run_dir / "run_plan.json"),
        "summary": str(summary),
        "completed": summary.exists(),
    }
    (run_dir / "stage23_run.status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
