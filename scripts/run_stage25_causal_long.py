#!/usr/bin/env python3
"""Plan or manually execute one frozen Stage 25 causal training seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_stage25_causal_online.yaml")
TRAINER = ROOT / "scripts" / "run_stage25_causal_ppo.py"


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "checkpoints").glob(
        "stage25_causal_ppo_update_*.pt"))
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_index", type=int)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seeds = protocol["cohort"]["independent_training_base_seeds"]
    if not 1 <= args.seed_index <= len(seeds):
        parser.error(f"seed_index must be in [1, {len(seeds)}]")
    if args.execute and not args.confirm_long:
        parser.error("long execution requires --confirm-long")
    if (args.seed_index != protocol["cohort"]["sentinel_seed_index"] and
            args.execute):
        sentinel = (ROOT / "results" / "stage25_causal_online" / "long" /
                    "seed_01" / "sentinel_quality_gate.json")
        if not sentinel.exists():
            parser.error(
                "seed 01 quality gate is missing; do not expand cohort")
        result = json.loads(sentinel.read_text(encoding="utf-8"))
        if not result.get("passed"):
            parser.error(
                "seed 01 sentinel quality gate failed; do not expand cohort")

    ppo = protocol["ppo"]
    base_seed = int(seeds[args.seed_index - 1])
    validation = protocol["validation"]
    run_dir = (ROOT / "results" / "stage25_causal_online" / "long" /
               f"seed_{args.seed_index:02d}")
    output = run_dir / "stage25_causal_ppo.pt"
    command = [
        sys.executable, str(TRAINER),
        "--execution-mode", protocol["environment"]["execution_mode"],
        "--arrival-profile", protocol["environment"]["arrival_profile"],
        "--observation-variant", protocol["claim_boundary"][
            "policy_observation"],
        "--policy-variant", protocol["model"]["policy_variant"],
        "--discount-mode", protocol["environment"]["discount_mode"],
        "--trace-mode", protocol["environment"]["trace_mode"],
        "--trace-tau-s", str(protocol["environment"]["trace_tau_s"]),
        "--reward-contract", protocol["environment"]["reward_contract"],
        "--updates", str(ppo["updates"]),
        "--episodes-per-update", str(ppo["episodes_per_update"]),
        "--ppo-epochs", str(ppo["ppo_epochs"]),
        "--minibatch-size", str(ppo["minibatch_size"]),
        "--learning-rate", str(ppo["learning_rate"]),
        "--clip-ratio", str(ppo["clip_ratio"]),
        "--gae-lambda", str(ppo["gae_lambda"]),
        "--entropy-coefficient", str(ppo["entropy_coefficient"]),
        "--value-coefficient", str(ppo["value_coefficient"]),
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
    resume = latest_checkpoint(run_dir) if args.resume_latest else None
    if args.resume_latest and resume is None:
        parser.error(f"no checkpoint found under {run_dir / 'checkpoints'}")
    if resume is not None:
        command.extend(["--resume", str(resume)])

    plan = {
        "stage": 25,
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
        "visibility_contract": protocol["claim_boundary"]["visibility"],
        "handover_sampling": protocol["environment"]["handover_sampling"],
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
