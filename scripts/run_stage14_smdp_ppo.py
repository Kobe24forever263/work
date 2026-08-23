#!/usr/bin/env python3
"""Train the masked Stage 14 dispatcher with variable-time SMDP-PPO."""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_policy import MaskedCandidateActorCritic
from warehouse_core.stage12_encoding import Stage12Encoder
from warehouse_core.stage14_ppo import (
    PPOConfig, RunningReturnNormalizer, baseline_relative_score,
    collect_rollouts, evaluate,
    load_checkpoint, ppo_update)


RESULT_SCHEMA_VERSION = "warehouse_stage16_training_v1"
ALGORITHM_NAME = "centralized_masked_smdp_ppo"
REWARD_VERSION = "reward_v2"


def checkpoint_payload(model, optimizer, reward_normalizer, args, update,
                       episodes, best_score, ppo_generator=None):
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "format": "warehouse_stage14_smdp_ppo_reward_v2_context_v2",
        "algorithm": ALGORITHM_NAME,
        "reward_version": REWARD_VERSION,
        "reward_config": {
            "outstanding_time_scale": 30.0,
            "handover_timeout_penalty": 2.0,
            "active_robot_time_penalty": 0.0002,
            "mode_specific_bonus": False,
        },
        "model_metadata": model.metadata(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward_normalizer": reward_normalizer.state_dict(),
        "execution_mode": args.execution_mode,
        "arrival_profile": args.arrival_profile,
        "observation_variant": args.observation_variant,
        "policy_variant": args.policy_variant,
        "discount_mode": args.discount_mode,
        "encoder_metadata": Stage12Encoder(
            args.observation_variant).metadata(),
        "update": update,
        "episodes_trained": episodes,
        "base_seed": args.seed,
        "experiment_run_seed": args.experiment_run_seed,
        "validation_seed_start": args.validation_seed_start,
        "best_evaluation_reward": best_score,
        "warm_start_source": args.warm_start_source,
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "ppo_generator_state": (
            ppo_generator.get_state() if ppo_generator is not None else None),
    }


def save_checkpoint(path, model, optimizer, reward_normalizer, args, update,
                    episodes, best_score, ppo_generator=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(
        model, optimizer, reward_normalizer, args, update, episodes,
        best_score, ppo_generator), path)


def compact(evaluation):
    return {key: value for key, value in evaluation.items()
            if key != "episodes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-mode", choices=(
        "SERIAL", "CONCURRENT", "MIXED"), default="MIXED")
    parser.add_argument("--arrival-profile", choices=(
        "MEDIUM", "DENSE", "BURST"), default="MEDIUM")
    parser.add_argument("--observation-variant", choices=(
        "FULL_CONTEXT_V2", "NO_QUEUE_RESOURCE", "NO_HANDOVER_CUES",
        "NO_PERSISTENT_POSITION", "NO_POSITION_ONLY", "NO_ETA_COST_ONLY",
        "NO_HISTORY_ONLY", "NO_EXPLICIT_QUEUE_RESOURCE",
        "NO_EXPLICIT_HANDOVER_RISK"), default="FULL_CONTEXT_V2")
    parser.add_argument("--policy-variant", choices=(
        "CONTEXT_INTERACTION", "FLAT_MASKED_PPO"),
        default="CONTEXT_INTERACTION")
    parser.add_argument("--discount-mode", choices=(
        "VARIABLE_SMDP", "FIXED_PER_DECISION"), default="VARIABLE_SMDP")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=.2)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--entropy-coefficient", type=float, default=.01)
    parser.add_argument("--value-coefficient", type=float, default=.5)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261801)
    parser.add_argument("--experiment-run-seed", type=int)
    parser.add_argument("--validation-seed-start", type=int,
                        default=42000000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--resume", type=Path)
    start_group.add_argument("--warm-start", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.ppo_epochs,
           args.minibatch_size, args.eval_episodes, args.eval_every,
           args.checkpoint_every, args.log_every) <= 0:
        parser.error("all count arguments must be positive")

    mode = args.execution_mode.lower()
    profile = args.arrival_profile.lower()
    output = args.output or (
        WORK_ROOT / "results" / "stage14" / "ppo" / "smoke" /
        f"stage14_smdp_ppo_{mode}_{profile}.pt")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    best_path = output.with_suffix(".best.pt")
    history_path = output.with_suffix(".history.json")
    summary_path = output.with_suffix(".summary.json")
    checkpoint_dir = output.parent / "checkpoints"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        if payload["execution_mode"] != args.execution_mode or \
                payload["arrival_profile"] != args.arrival_profile:
            raise ValueError("resume checkpoint mode/profile mismatch")
        for key in ("observation_variant", "policy_variant", "discount_mode"):
            if payload.get(key, getattr(args, key)) != getattr(args, key):
                raise ValueError(f"resume checkpoint {key} mismatch")
        start_update = int(payload["update"])
        trained_episodes = int(payload["episodes_trained"])
        best_score = float(payload.get(
            "best_evaluation_reward", -math.inf))
        if "reward_normalizer" not in payload:
            raise ValueError(
                "legacy checkpoint has no reward normalization; restart "
                "Reward V2 training instead of resuming it")
        reward_normalizer = RunningReturnNormalizer.from_state_dict(
            payload["reward_normalizer"])
        args.warm_start_source = payload.get("warm_start_source")
    elif args.warm_start:
        warm_payload = torch.load(
            args.warm_start, map_location=device, weights_only=False)
        metadata = warm_payload.get(
            "model_metadata", warm_payload.get("metadata"))
        if metadata is None:
            raise ValueError("warm-start checkpoint has no model metadata")
        warm_observation_variant = warm_payload.get(
            "observation_variant", "FULL_CONTEXT_V2")
        warm_policy_variant = warm_payload.get(
            "policy_variant", metadata.get(
                "policy_variant", "CONTEXT_INTERACTION"))
        if warm_observation_variant != args.observation_variant:
            raise ValueError("warm-start observation variant mismatch")
        if warm_policy_variant != args.policy_variant:
            raise ValueError("warm-start policy variant mismatch")
        expected = MaskedCandidateActorCritic(
            hidden_width=args.hidden_width,
            policy_variant=args.policy_variant).metadata()
        if (metadata.get("architecture") != expected["architecture"] or
                metadata.get("state_width") != expected["state_width"] or
                metadata.get("action_width") != expected["action_width"] or
                metadata.get("hidden_width") != expected["hidden_width"]):
            raise ValueError(
                "warm-start model is incompatible with contextual policy V2")
        model = MaskedCandidateActorCritic(
            hidden_width=args.hidden_width,
            policy_variant=args.policy_variant).to(device)
        state_dict = warm_payload.get(
            "model_state_dict", warm_payload.get("state_dict"))
        if state_dict is None:
            raise ValueError("warm-start checkpoint has no model state")
        model.load_state_dict(state_dict)
        payload = None
        start_update = 0
        trained_episodes = 0
        best_score = -math.inf
        reward_normalizer = RunningReturnNormalizer()
        args.warm_start_source = str(args.warm_start.resolve())
    else:
        model = MaskedCandidateActorCritic(
            hidden_width=args.hidden_width,
            policy_variant=args.policy_variant).to(device)
        payload = None
        start_update = 0
        trained_episodes = 0
        best_score = -math.inf
        reward_normalizer = RunningReturnNormalizer()
        args.warm_start_source = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if payload and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    config = PPOConfig(
        clip_ratio=args.clip_ratio,
        gae_lambda=args.gae_lambda,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size)
    generator = torch.Generator().manual_seed(args.seed + 14)
    if payload:
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(payload["torch_random_state"])
        if payload.get("ppo_generator_state") is not None:
            generator.set_state(payload["ppo_generator_state"])
    initial_parameters = torch.cat([
        parameter.detach().flatten().cpu() for parameter in model.parameters()
    ]).clone()
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    last_evaluation = None
    # ``--updates`` is the total target budget.  This makes an interrupted
    # 1050/2000 run resume to 2000, rather than accidentally training to 3050.
    final_update = args.updates
    if start_update >= final_update:
        raise ValueError(
            f"checkpoint is already at update {start_update}, which is not "
            f"below target {final_update}")
    episodes_at_resume = trained_episodes
    progress = tqdm(
        range(start_update + 1, final_update + 1),
        initial=start_update, total=final_update,
        desc=f"SMDP-PPO {args.execution_mode}/{args.arrival_profile}",
        unit="update", dynamic_ncols=True)
    for update in progress:
        first_episode = trained_episodes
        seeds = [args.seed + first_episode + index
                 for index in range(args.episodes_per_update)]
        steps, episodes = collect_rollouts(
            model, seeds, args.execution_mode,
            args.arrival_profile, device, args.observation_variant,
            args.discount_mode)
        metrics = ppo_update(
            model, optimizer, steps, config, device, generator,
            reward_normalizer)
        trained_episodes += len(episodes)
        mode_counts = Counter()
        exposed_modes = set()
        for episode in episodes:
            mode_counts.update(episode["transport_modes"])
            exposed_modes.update(episode["exposed_transport_modes"])
        row = {
            "update": update,
            "episodes_trained": trained_episodes,
            "rollout_steps": len(steps),
            "rollout_discount_min": min(step.discount for step in steps),
            "rollout_discount_max": max(step.discount for step in steps),
            "rollout_mean_reward": float(np.mean(
                [episode["reward"] for episode in episodes])),
            "raw_episode_reward_mean": float(np.mean(
                [episode["reward"] for episode in episodes])),
            "rollout_completed": sum(
                episode["completed"] for episode in episodes),
            "rollout_failed": sum(
                episode["failed"] for episode in episodes),
            "rollout_peak_active_tasks": max(
                episode["peak_active_tasks"] for episode in episodes),
            "rollout_execution_modes": dict(Counter(
                episode["execution_mode"] for episode in episodes)),
            "rollout_transport_modes": dict(mode_counts),
            "rollout_exposed_transport_modes": sorted(exposed_modes),
            **metrics,
        }
        if update % args.eval_every == 0 or update == final_update:
            eval_seeds = [args.validation_seed_start + index
                          for index in range(args.eval_episodes)]
            last_evaluation = evaluate(
                model, eval_seeds, args.execution_mode,
                args.arrival_profile, device,
                observation_variant=args.observation_variant)
            score = last_evaluation["mean_reward"]
            row["evaluation"] = compact(last_evaluation)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_path, model, optimizer, reward_normalizer,
                    args, update,
                    trained_episodes, best_score, generator)
        history.append(row)
        if update % args.checkpoint_every == 0:
            history_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            save_checkpoint(
                checkpoint_dir / f"{output.stem}_update_{update:05d}.pt",
                model, optimizer, reward_normalizer, args, update,
                trained_episodes, best_score, generator)
        progress.set_postfix(
            episodes=trained_episodes,
            raw_ep_reward=f"{row['raw_episode_reward_mean']:.2f}",
            norm_step_reward=f"{row['normalized_reward_mean']:.3f}",
            loss=f"{row['loss']:.3f}",
            entropy=f"{row['entropy']:.3f}")
        if update % args.log_every == 0 or update == final_update:
            evaluation = row.get("evaluation", {})
            evaluation_text = (
                f" eval_reward={evaluation['mean_reward']:.3f}"
                f" success={evaluation['success_rate']:.2%}"
                f" modes={evaluation['transport_modes']}"
                if evaluation else "")
            tqdm.write(
                f"update={update}/{final_update} "
                f"episodes={trained_episodes} steps={len(steps)} "
                f"raw_episode_reward={row['raw_episode_reward_mean']:.3f} "
                f"normalized_step_reward="
                f"{row['normalized_reward_mean']:.4f} "
                f"loss={row['loss']:.4f} entropy={row['entropy']:.4f}"
                f"{evaluation_text}")

    final_parameters = torch.cat([
        parameter.detach().flatten().cpu() for parameter in model.parameters()
    ])
    parameter_delta = float(torch.linalg.vector_norm(
        final_parameters - initial_parameters).item())
    eval_seeds = [args.validation_seed_start + index
                  for index in range(args.eval_episodes)]
    if last_evaluation is None:
        last_evaluation = evaluate(
            model, eval_seeds, args.execution_mode,
            args.arrival_profile, device,
            observation_variant=args.observation_variant)
    rule = evaluate(
        model, eval_seeds, args.execution_mode,
        args.arrival_profile, device, use_rule=True,
        observation_variant=args.observation_variant)
    save_checkpoint(
        output, model, optimizer, reward_normalizer, args, final_update,
        trained_episodes, best_score, generator)
    finite_metrics = all(
        math.isfinite(value)
        for row in history for key, value in row.items()
        if key in {"loss", "actor_loss", "value_loss", "entropy",
                   "approximate_kl", "clip_fraction", "gradient_norm"})
    smoke_run = args.updates * args.episodes_per_update <= 20
    transport_modes = {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
    context_matrix = last_evaluation["selected_mode_by_cost_reference"]
    positive_relative_score = baseline_relative_score(
        last_evaluation["mean_reward"], rule["mean_reward"])
    context_match_rates = {
        reference_mode: (
            selected_counts.get(reference_mode, 0) /
            max(sum(selected_counts.values()), 1))
        for reference_mode, selected_counts in context_matrix.items()
    }
    contextual_switching_gate = (
        set(context_matrix) == transport_modes and
        set(last_evaluation["transport_modes"]) == transport_modes and
        all(context_match_rates.get(mode, 0.0) >= .20
            for mode in transport_modes))
    assertions = {
        "ppo_parameters_updated": parameter_delta > 0,
        "training_metrics_are_finite": finite_metrics,
        "discount_contract_is_valid": (
            all(0 < row["rollout_discount_min"] <=
                row["rollout_discount_max"] <= 1 for row in history) and
            ((args.discount_mode == "VARIABLE_SMDP" and any(
                row["rollout_discount_min"] < row["rollout_discount_max"]
                for row in history)) or
             (args.discount_mode == "FIXED_PER_DECISION" and all(
                abs(row["rollout_discount_min"] - .99) < 1e-7 and
                abs(row["rollout_discount_max"] - .99) < 1e-7
                for row in history)))),
        "evaluation_resolved_all_tasks": (
            last_evaluation["task_count"] == args.eval_episodes * 20),
        "evaluation_actions_all_legal": (
            last_evaluation["illegal_action_count"] == 0),
        "evaluation_has_no_resource_leak": (
            last_evaluation["resource_leak_count"] == 0),
        "one_policy_exposes_all_three_transport_modes": (
            set(last_evaluation["exposed_transport_modes"]) ==
            transport_modes),
        "context_conditioned_metrics_cover_every_assignment": (
            last_evaluation["cost_reference_decision_count"] ==
            last_evaluation["task_count"]),
        # A very short run only proves the interface.  A formal run must also
        # show that the same policy switches modes when the per-task context
        # changes, rather than converging to a global mode prior.
        "formal_policy_switches_modes_by_context": (
            smoke_run or contextual_switching_gate),
        "rule_baseline_has_no_resource_leak": (
            rule["resource_leak_count"] == 0),
        "concurrent_mode_overlaps_when_requested": (
            args.execution_mode == "SERIAL" or
            last_evaluation["maximum_active_tasks"] >= 2),
    }
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "algorithm": ALGORITHM_NAME,
        "reward_version": REWARD_VERSION,
        "stage": 14,
        "gate": ("SHORT_SMDP_PPO_REWARD_V2_SMOKE" if smoke_run else
                 "SMDP_PPO_REWARD_V2_TRAINING_RUN"),
        "claim_boundary": (
            "Interface smoke only; not a convergence claim." if smoke_run else
            "Training completed; held-out acceptance is still required."),
        "execution_mode": args.execution_mode,
        "arrival_profile": args.arrival_profile,
        "observation_variant": args.observation_variant,
        "policy_variant": args.policy_variant,
        "discount_mode": args.discount_mode,
        "encoder": Stage12Encoder(args.observation_variant).metadata(),
        "updates": args.updates,
        "episodes_trained_this_run": trained_episodes - episodes_at_resume,
        "episodes_trained_total": trained_episodes,
        "resumed_from_update": start_update if args.resume else None,
        "training_seed_start": args.seed,
        "training_seed_end": args.seed + trained_episodes - 1,
        "validation_seed_start": args.validation_seed_start,
        "validation_seed_end": (
            args.validation_seed_start + args.eval_episodes - 1),
        "experiment_run_seed": args.experiment_run_seed,
        "warm_start_source": args.warm_start_source,
        "model": model.metadata(),
        "ppo": asdict(config),
        "reward_normalization": reward_normalizer.state_dict(),
        "reward_reporting": {
            "evaluation_mean_reward": "raw_environment_episode_sum",
            "rollout_mean_reward": "raw_environment_episode_sum",
            "normalized_reward_mean": "ppo_training_step_reward",
            "positive_relative_score": (
                "reporting_only_rule_baseline_equals_100"),
        },
        "positive_relative_score": positive_relative_score,
        "contextual_mode_acceptance": {
            "principle": (
                "No target mode frequency. Each assignment is evaluated "
                "against the cheapest legal mode in its current task/fleet "
                "context."),
            "selected_mode_by_cost_reference": context_matrix,
            "diagonal_match_rates": context_match_rates,
            "minimum_formal_diagonal_rate": .20,
            "formal_gate_passed": contextual_switching_gate,
        },
        "parameter_delta_l2": parameter_delta,
        "final_weight": str(output),
        "best_weight": str(best_path),
        "history": str(history_path),
        "evaluation": compact(last_evaluation),
        "rule_baseline": compact(rule),
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
