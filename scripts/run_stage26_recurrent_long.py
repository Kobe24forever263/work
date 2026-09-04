#!/usr/bin/env python3
"""Plan or manually execute one frozen Stage 26 recurrent seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage26_recurrent_causal.yaml")
TRAINER = ROOT / "scripts" / "run_stage26_recurrent_ppo.py"


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "checkpoints").glob(
        "stage26_recurrent_ppo_update_*.pt"))
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_index", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume-latest", action="store_true")
    resume_group.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seeds = protocol["cohort"]["independent_training_base_seeds"]
    if not 1 <= args.seed_index <= len(seeds):
        parser.error(f"seed_index must be in [1, {len(seeds)}]")
    if args.execute and not args.confirm_long:
        parser.error("long execution requires --confirm-long")
    if args.execute and args.seed_index != protocol["cohort"][
            "sentinel_seed_index"]:
        sentinel = (ROOT / "results" / "stage26_recurrent_causal" /
                    "long" / "seed_01" / "sentinel_quality_gate.json")
        if not sentinel.exists() or not json.loads(
                sentinel.read_text(encoding="utf-8")).get("passed"):
            parser.error(
                "Stage 26 seed 01 gate is absent or failed; cohort expansion "
                "is blocked")

    ppo = protocol["ppo"]
    model = protocol["model"]
    base_seed = int(seeds[args.seed_index - 1])
    validation = protocol["validation"]
    run_dir = (ROOT / "results" / "stage26_recurrent_causal" / "long" /
               f"seed_{args.seed_index:02d}")
    output = run_dir / "stage26_recurrent_ppo.pt"
    command = [
        sys.executable, str(TRAINER),
        "--updates", str(ppo["updates"]),
        "--episodes-per-update", str(ppo["episodes_per_update"]),
        "--ppo-epochs", str(ppo["ppo_epochs"]),
        "--learning-rate", str(ppo["learning_rate"]),
        "--clip-ratio", str(ppo["clip_ratio"]),
        "--gae-lambda", str(ppo["gae_lambda"]),
        "--entropy-coefficient", str(ppo["entropy_coefficient"]),
        "--value-coefficient", str(ppo["value_coefficient"]),
        "--hidden-width", str(model["hidden_width"]),
        "--memory-width", str(model["memory_width"]),
        "--trace-tau-s", str(protocol["environment"]["trace_tau_s"]),
        "--seed", str(base_seed),
        "--experiment-run-seed", str(base_seed),
        "--validation-seed-start", str(
            validation["online_validation_seed_start"]),
        "--eval-episodes", str(
            validation["online_validation_episode_count"]),
        "--eval-every", str(ppo["evaluation_every_updates"]),
        "--checkpoint-every", str(ppo["checkpoint_every_updates"]),
        "--log-every", str(ppo["progress_log_every_updates"]),
        "--output", str(output),
    ]
    resume = None
    if args.resume_checkpoint is not None:
        resume = args.resume_checkpoint.expanduser().resolve()
        if not resume.is_file():
            parser.error(f"resume checkpoint not found: {resume}")
        checkpoint_dir = (run_dir / "checkpoints").resolve()
        if resume.parent != checkpoint_dir:
            parser.error(
                "explicit resume checkpoint must belong to the selected "
                f"seed directory: {checkpoint_dir}")
    elif args.resume_latest:
        resume = latest_checkpoint(run_dir)
    if args.resume_latest and resume is None:
        parser.error(f"no checkpoint found under {run_dir / 'checkpoints'}")
    if resume is not None:
        command.extend(["--resume", str(resume)])
    plan = {
        "stage": 26,
        "protocol": protocol["protocol"]["version"],
        "manual_execution_only": True,
        "seed_index": args.seed_index,
        "base_seed": base_seed,
        "sentinel": args.seed_index == protocol["cohort"][
            "sentinel_seed_index"],
        "updates": ppo["updates"],
        "episodes_per_update": ppo["episodes_per_update"],
        "planned_episodes": ppo["episodes_per_seed"],
        "tasks_per_episode": protocol["environment"]["tasks_per_episode"],
        "planned_task_decisions": (
            ppo["episodes_per_seed"] *
            protocol["environment"]["tasks_per_episode"]),
        "architecture": model["architecture"],
        "memory_reset_boundary": model["memory_reset_boundary"],
        "output": str(output),
        "best_weight": str(output.with_suffix(".best.pt")),
        "history": str(output.with_suffix(".history.json")),
        "summary": str(output.with_suffix(".summary.json")),
        "resume": str(resume) if resume else None,
        "command": command,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
