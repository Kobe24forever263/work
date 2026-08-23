#!/usr/bin/env python3
"""Plan or manually execute one Stage 18 condition/seed training run."""

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
SEED_CONFIG = (WORK_ROOT / "src" / "warehouse_bringup" / "config" /
               "experiment_seeds.yaml")
WARM_SCRIPT = WORK_ROOT / "scripts" / "run_stage14_policy_smoke.py"
PPO_SCRIPT = WORK_ROOT / "scripts" / "run_stage14_smdp_ppo.py"
CONDITIONS = {
    "full_context_v2": {
        "observation_variant": "FULL_CONTEXT_V2",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_queue_resource": {
        "observation_variant": "NO_QUEUE_RESOURCE",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_handover_cues": {
        "observation_variant": "NO_HANDOVER_CUES",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_persistent_position": {
        "observation_variant": "NO_PERSISTENT_POSITION",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_position_only": {
        "observation_variant": "NO_POSITION_ONLY",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_eta_cost_only": {
        "observation_variant": "NO_ETA_COST_ONLY",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_history_only": {
        "observation_variant": "NO_HISTORY_ONLY",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_explicit_queue_resource": {
        "observation_variant": "NO_EXPLICIT_QUEUE_RESOURCE",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "no_explicit_handover_risk": {
        "observation_variant": "NO_EXPLICIT_HANDOVER_RISK",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP"},
    "fixed_gamma": {
        "observation_variant": "FULL_CONTEXT_V2",
        "policy_variant": "CONTEXT_INTERACTION",
        "discount_mode": "FIXED_PER_DECISION"},
    "flat_masked_ppo": {
        "observation_variant": "FULL_CONTEXT_V2",
        "policy_variant": "FLAT_MASKED_PPO",
        "discount_mode": "VARIABLE_SMDP"},
}


def command_text(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("stage18_ppo_update_*.pt"))
    return checkpoints[-1] if checkpoints else None


def completed_ablation_run_is_reusable(run_dir: Path, *, condition: dict,
                                       updates: int) -> bool:
    summary_path = run_dir / "stage18_ppo.summary.json"
    history_path = run_dir / "stage18_ppo.history.json"
    if not summary_path.exists() or not history_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (summary.get("updates") != updates or len(history) != updates or
            history[-1].get("update") != updates or
            summary.get("observation_variant") !=
            condition["observation_variant"]):
        return False
    assertions = summary.get("assertions", {})
    required = (
        "ppo_parameters_updated", "training_metrics_are_finite",
        "discount_contract_is_valid", "evaluation_resolved_all_tasks",
        "evaluation_actions_all_legal", "evaluation_has_no_resource_leak",
        "one_policy_exposes_all_three_transport_modes",
        "context_conditioned_metrics_cover_every_assignment",
        "rule_baseline_has_no_resource_leak",
        "concurrent_mode_overlaps_when_requested")
    return all(assertions.get(key) is True for key in required)


def validate_resume_checkpoint(path: Path, *, condition: dict,
                               arrival_profile: str, run_seed: int,
                               ppo_seed: int, validation_seed: int,
                               target_updates: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "execution_mode": "CONCURRENT",
        "arrival_profile": arrival_profile,
        "observation_variant": condition["observation_variant"],
        "policy_variant": condition["policy_variant"],
        "discount_mode": condition["discount_mode"],
        "experiment_run_seed": run_seed,
        "base_seed": ppo_seed,
        "validation_seed_start": validation_seed,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items() if payload.get(key) != value}
    update = int(payload.get("update", -1))
    episodes = int(payload.get("episodes_trained", -1))
    if not 0 < update < target_updates:
        mismatches["update"] = {
            "expected": f"1..{target_updates - 1}", "actual": update}
    if episodes != update * 4:
        mismatches["episodes_trained"] = {
            "expected": update * 4, "actual": episodes}
    metadata = payload.get("model_metadata", {})
    for key, value in {"state_width": 284, "action_width": 30,
                       "hidden_width": 96}.items():
        if metadata.get(key) != value:
            mismatches[f"model_metadata.{key}"] = {
                "expected": value, "actual": metadata.get(key)}
    if "reward_normalizer" not in payload:
        mismatches["reward_normalizer"] = {
            "expected": "present", "actual": "missing"}
    if mismatches:
        raise ValueError(
            "resume checkpoint is incompatible:\n" +
            json.dumps(mismatches, ensure_ascii=False, indent=2))
    return {
        "checkpoint": str(path.resolve()),
        "update": update,
        "episodes_trained": episodes,
        "target_updates": target_updates,
        "remaining_updates": target_updates - update,
        "exact_rng_state_available": all(
            payload.get(key) is not None for key in (
                "numpy_random_state", "torch_random_state",
                "ppo_generator_state")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument("seed_index", type=int)
    parser.add_argument("--arrival-profile", choices=(
        "MEDIUM", "DENSE", "BURST"),
                        default="MEDIUM")
    parser.add_argument("--phase", choices=("warm-start", "ppo", "all"),
                        default="all")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run; default only prints the plan.")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume the latest compatible partial PPO run.")
    args = parser.parse_args()
    seed_protocol = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    training = seed_protocol["training"]
    seed_count = len(training["independent_run_base_seeds"])
    if not 1 <= args.seed_index <= seed_count:
        parser.error(f"seed_index must be in [1, {seed_count}]")
    run_seed = training["independent_run_base_seeds"][args.seed_index - 1]
    warm_seed = run_seed + int(training["warm_start_seed_offset"])
    ppo_seed = run_seed + int(training["ppo_seed_offset"])
    validation_seed = int(seed_protocol["validation"]["seed_start"])
    condition = CONDITIONS[args.condition]
    profile_root = {
        "MEDIUM": "stage18",
        "DENSE": "stage18_dense",
        "BURST": "stage18_burst",
    }[args.arrival_profile]
    run_dir = (WORK_ROOT / "results" / profile_root / args.condition /
               f"seed_{args.seed_index:02d}")
    warm_weight = run_dir / "warm_start.pt"
    final_weight = run_dir / "stage18_ppo.pt"
    warm_command = [
        sys.executable, str(WARM_SCRIPT),
        "--epochs", "30", "--batch-size", "32",
        "--learning-rate", "0.001",
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", args.arrival_profile,
        "--run-seed", str(warm_seed),
        "--training-seed-count", str(training["warm_start_seed_count"]),
        "--validation-seed-start", str(validation_seed),
        "--validation-episodes", "5",
        "--observation-variant", condition["observation_variant"],
        "--policy-variant", condition["policy_variant"],
        "--output", str(warm_weight),
    ]
    if args.condition.startswith("no_"):
        warm_command.append("--allow-interface-only-continuation")
    resume_path = latest_checkpoint(run_dir) if args.resume_latest else None
    resume_audit = None
    if resume_path is not None:
        resume_audit = validate_resume_checkpoint(
            resume_path, condition=condition,
            arrival_profile=args.arrival_profile, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_seed,
            target_updates=args.updates)
    ppo_command = [
        sys.executable, str(PPO_SCRIPT),
        "--execution-mode", "CONCURRENT",
        "--arrival-profile", args.arrival_profile,
        "--observation-variant", condition["observation_variant"],
        "--policy-variant", condition["policy_variant"],
        "--discount-mode", condition["discount_mode"],
        "--updates", str(args.updates),
        "--episodes-per-update", "4",
        "--ppo-epochs", "4",
        "--minibatch-size", "32",
        "--learning-rate", "0.0003",
        "--clip-ratio", "0.2",
        "--gae-lambda", "0.95",
        "--entropy-coefficient", "0.01",
        "--value-coefficient", "0.5",
        "--seed", str(ppo_seed),
        "--experiment-run-seed", str(run_seed),
        "--validation-seed-start", str(validation_seed),
        "--eval-episodes", "100",
        "--eval-every", "50",
        "--checkpoint-every", "50",
        "--log-every", "20",
        "--output", str(final_weight),
    ]
    if resume_path is not None:
        ppo_command.extend(["--resume", str(resume_path)])
    else:
        ppo_command.extend(["--warm-start", str(warm_weight)])
    plan = {
        "stage": 18,
        "condition": args.condition,
        "arrival_profile": args.arrival_profile,
        "seed_index": args.seed_index,
        "experiment_run_seed": run_seed,
        "warm_start_seed_range": [
            warm_seed, warm_seed + training["warm_start_seed_count"] - 1],
        "ppo_seed_range": [ppo_seed, ppo_seed + args.updates * 4 - 1],
        "validation_seed_range": [
            validation_seed,
            validation_seed + seed_protocol["validation"]["seed_count"] - 1],
        "condition_config": condition,
        "warm_start_weight": str(warm_weight),
        "final_weight": str(final_weight),
        "best_weight": str(final_weight.with_suffix(".best.pt")),
        "history": str(final_weight.with_suffix(".history.json")),
        "warm_start_command": command_text(warm_command),
        "ppo_command": command_text(ppo_command),
        "planned_updates": args.updates,
        "planned_ppo_episodes": args.updates * 4,
        "resume_audit": resume_audit,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if completed_ablation_run_is_reusable(
            run_dir, condition=condition, updates=args.updates):
        print(json.dumps({
            "state": "SKIPPED_COMPLETE_REUSABLE",
            "run_dir": str(run_dir),
            "note": ("Context-mode agreement is a measured ablation outcome, "
                     "not a continuation gate.")}, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if args.phase in {"warm-start", "all"} and resume_path is None:
        subprocess.run(warm_command, cwd=WORK_ROOT, check=True)
    if args.phase in {"ppo", "all"}:
        if resume_path is None and not warm_weight.exists():
            raise FileNotFoundError(
                f"warm-start weight not found: {warm_weight}")
        completed = subprocess.run(ppo_command, cwd=WORK_ROOT, check=False)
        if completed.returncode != 0 and not completed_ablation_run_is_reusable(
                run_dir, condition=condition, updates=args.updates):
            raise subprocess.CalledProcessError(
                completed.returncode, ppo_command)
        if completed.returncode != 0:
            acceptance = {
                "accepted": True,
                "reason": ("Training, numerical, safety and completion gates "
                           "passed; contextual cost-reference agreement is "
                           "retained as an ablation result."),
                "original_exit_code": completed.returncode,
            }
            (run_dir / "ablation_acceptance.json").write_text(
                json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
