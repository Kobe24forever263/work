#!/usr/bin/env python3
"""Train the Stage 26 GRU-memory policy on the causal online interface."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import baseline_relative_score
from warehouse_core.stage25_causal_observation import (
    CAUSAL_RELEASED_V4, CausalReleasedEncoder)
from warehouse_core.stage26_recurrent_policy import (
    RecurrentMaskedCandidateActorCritic)
from warehouse_core.stage26_recurrent_ppo import (
    PPOConfig, RunningReturnNormalizer, collect_recurrent_rollouts,
    evaluate_recurrent, load_recurrent_checkpoint, recurrent_ppo_update,
    tasks_per_episode)


SCHEMA = "warehouse_stage26_recurrent_training_v1"
ALGORITHM = "centralized_masked_recurrent_causal_smdp_ppo"


def compact(evaluation: dict) -> dict:
    result = {key: value for key, value in evaluation.items()
              if key != "episodes"}
    metadata = [
        episode.get("arrival_schedule_metadata", {})
        for episode in evaluation.get("episodes", [])
        if episode.get("arrival_schedule_metadata")]
    if metadata:
        result["arrival_schedule_metadata"] = metadata
    return result


def checkpoint_payload(model, optimizer, normalizer, args, update,
                       episodes, best_score, generator):
    return {
        "schema_version": SCHEMA,
        "format": "warehouse_stage26_recurrent_causal_smdp_ppo",
        "algorithm": ALGORITHM,
        "stage": 26,
        "model_metadata": model.metadata(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward_normalizer": normalizer.state_dict(),
        "execution_mode": args.execution_mode,
        "arrival_profile": args.arrival_profile,
        "observation_variant": CAUSAL_RELEASED_V4,
        "policy_variant": "RECURRENT_CONTEXT_INTERACTION",
        "discount_mode": "VARIABLE_SMDP",
        "trace_mode": "DURATION_SCALED",
        "trace_tau_s": args.trace_tau_s,
        "reward_contract": "CONTINUOUS_TIME_V3",
        "update": update,
        "episodes_trained": episodes,
        "base_seed": args.seed,
        "experiment_run_seed": args.experiment_run_seed,
        "validation_seed_start": args.validation_seed_start,
        "best_evaluation_reward": best_score,
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "ppo_generator_state": generator.get_state(),
    }


def save_checkpoint(path, model, optimizer, normalizer, args, update,
                    episodes, best_score, generator):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(
        model, optimizer, normalizer, args, update, episodes,
        best_score, generator), path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-mode", choices=("CONCURRENT",),
                        default="CONCURRENT")
    parser.add_argument("--arrival-profile", choices=("MIXED_CURRICULUM",),
                        default="MIXED_CURRICULUM")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=.2)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--entropy-coefficient", type=float, default=.01)
    parser.add_argument("--value-coefficient", type=float, default=.5)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--memory-width", type=int, default=96)
    parser.add_argument("--trace-tau-s", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=72600000)
    parser.add_argument("--experiment-run-seed", type=int,
                        default=72600000)
    parser.add_argument("--validation-seed-start", type=int,
                        default=72800000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "results" / "stage26_recurrent_causal" / "smoke" /
        "stage26_recurrent_ppo.pt"))
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.ppo_epochs,
           args.eval_episodes, args.eval_every,
           args.checkpoint_every, args.log_every,
           args.hidden_width, args.memory_width) <= 0:
        parser.error("count and width arguments must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    best_path = output.with_suffix(".best.pt")
    history_path = output.with_suffix(".history.json")
    summary_path = output.with_suffix(".summary.json")
    checkpoint_dir = output.parent / "checkpoints"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    encoder = CausalReleasedEncoder()

    if args.resume:
        model, payload = load_recurrent_checkpoint(args.resume, device)
        for key, expected in (
                ("execution_mode", args.execution_mode),
                ("arrival_profile", args.arrival_profile),
                ("observation_variant", CAUSAL_RELEASED_V4),
                ("reward_contract", "CONTINUOUS_TIME_V3"),
                ("trace_mode", "DURATION_SCALED")):
            if payload.get(key) != expected:
                raise ValueError(f"resume checkpoint {key} mismatch")
        if model.hidden_width != args.hidden_width or \
                model.memory_width != args.memory_width:
            raise ValueError("resume checkpoint network width mismatch")
        start_update = int(payload["update"])
        trained_episodes = int(payload["episodes_trained"])
        best_score = float(payload.get("best_evaluation_reward", -math.inf))
        normalizer = RunningReturnNormalizer.from_state_dict(
            payload["reward_normalizer"])
        history = (json.loads(history_path.read_text(encoding="utf-8"))
                   if history_path.exists() else [])
        if len(history) != start_update:
            raise ValueError(
                "resume history/checkpoint mismatch: "
                f"history has {len(history)} updates but checkpoint is at "
                f"update {start_update}")
        if history and int(history[-1].get("update", -1)) != start_update:
            raise ValueError(
                "resume history does not end at the checkpoint update: "
                f"last history update={history[-1].get('update')}, "
                f"checkpoint update={start_update}")
    else:
        model = RecurrentMaskedCandidateActorCritic(
            state_width=encoder.OBSERVATION_SIZE,
            action_width=encoder.ACTION_WIDTH,
            hidden_width=args.hidden_width,
            memory_width=args.memory_width).to(device)
        payload = None
        start_update = 0
        trained_episodes = 0
        best_score = -math.inf
        normalizer = RunningReturnNormalizer()
        history = []
    if start_update >= args.updates:
        raise ValueError("resume checkpoint already reached the target budget")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    config = PPOConfig(
        clip_ratio=args.clip_ratio, gae_lambda=args.gae_lambda,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        learning_rate=args.learning_rate, ppo_epochs=args.ppo_epochs,
        minibatch_size=1, trace_mode="DURATION_SCALED",
        trace_tau_s=args.trace_tau_s)
    generator = torch.Generator().manual_seed(args.seed + 26)
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
    episodes_at_resume = trained_episodes
    last_evaluation = None
    last_memory_diagnostics = None
    progress = tqdm(
        range(start_update + 1, args.updates + 1), initial=start_update,
        total=args.updates, desc="Recurrent SMDP-PPO CONCURRENT/MIXED",
        unit="update", dynamic_ncols=True)
    for update in progress:
        seeds = [args.seed + trained_episodes + index
                 for index in range(args.episodes_per_update)]
        episode_steps, episodes, memory_diagnostics = \
            collect_recurrent_rollouts(
                model, seeds, args.execution_mode, args.arrival_profile,
                device, reward_contract="CONTINUOUS_TIME_V3")
        last_memory_diagnostics = memory_diagnostics
        metrics = recurrent_ppo_update(
            model, optimizer, episode_steps, config, device, generator,
            normalizer)
        trained_episodes += len(episodes)
        mode_counts = Counter()
        exposed_modes = set()
        for episode in episodes:
            mode_counts.update(episode["transport_modes"])
            exposed_modes.update(episode["exposed_transport_modes"])
        flat_steps = [step for sequence in episode_steps for step in sequence]
        row = {
            "update": update,
            "episodes_trained": trained_episodes,
            "rollout_steps": len(flat_steps),
            "rollout_discount_min": min(step.discount for step in flat_steps),
            "rollout_discount_max": max(step.discount for step in flat_steps),
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
            "rollout_transport_modes": dict(mode_counts),
            "rollout_exposed_transport_modes": sorted(exposed_modes),
            "memory_diagnostics": memory_diagnostics,
            **metrics,
        }
        if update % args.eval_every == 0 or update == args.updates:
            eval_seeds = [args.validation_seed_start + index
                          for index in range(args.eval_episodes)]
            last_evaluation = evaluate_recurrent(
                model, eval_seeds, args.execution_mode,
                args.arrival_profile, device)
            row["evaluation"] = compact(last_evaluation)
            score = last_evaluation["mean_reward"]
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_path, model, optimizer, normalizer, args, update,
                    trained_episodes, best_score, generator)
        history.append(row)
        if update % args.checkpoint_every == 0:
            history_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            save_checkpoint(
                checkpoint_dir /
                f"{output.stem}_update_{update:05d}.pt",
                model, optimizer, normalizer, args, update,
                trained_episodes, best_score, generator)
        progress.set_postfix(
            episodes=trained_episodes,
            reward=f"{row['raw_episode_reward_mean']:.2f}",
            loss=f"{row['loss']:.3f}", entropy=f"{row['entropy']:.3f}")
        if update % args.log_every == 0 or update == args.updates:
            evaluation = row.get("evaluation", {})
            eval_text = (f" eval_reward={evaluation['mean_reward']:.3f}"
                         f" success={evaluation['success_rate']:.2%}"
                         if evaluation else "")
            tqdm.write(
                f"update={update}/{args.updates} episodes={trained_episodes} "
                f"steps={len(flat_steps)} "
                f"raw_episode_reward={row['raw_episode_reward_mean']:.3f} "
                f"loss={row['loss']:.4f} entropy={row['entropy']:.4f}"
                f"{eval_text}")

    final_parameters = torch.cat([
        parameter.detach().flatten().cpu() for parameter in model.parameters()
    ])
    parameter_delta = float(torch.linalg.vector_norm(
        final_parameters - initial_parameters).item())
    eval_seeds = [args.validation_seed_start + index
                  for index in range(args.eval_episodes)]
    if last_evaluation is None:
        last_evaluation = evaluate_recurrent(
            model, eval_seeds, args.execution_mode,
            args.arrival_profile, device)
    rule = evaluate_recurrent(
        model, eval_seeds, args.execution_mode,
        args.arrival_profile, device, use_rule=True)
    save_checkpoint(
        output, model, optimizer, normalizer, args, args.updates,
        trained_episodes, best_score, generator)

    finite_metrics = all(
        math.isfinite(float(row[key]))
        for row in history
        for key in ("loss", "actor_loss", "value_loss", "entropy",
                    "approximate_kl", "clip_fraction", "gradient_norm"))
    context_matrix = last_evaluation["selected_mode_by_cost_reference"]
    context_rates = {
        reference: selected.get(reference, 0) /
        max(sum(selected.values()), 1)
        for reference, selected in context_matrix.items()}
    expected_modes = {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
    context_gate = (
        set(context_matrix) == expected_modes and
        set(last_evaluation["transport_modes"]) == expected_modes and
        all(context_rates.get(mode, 0.0) >= .20 for mode in expected_modes))
    expected_tasks = args.eval_episodes * tasks_per_episode(
        args.arrival_profile)
    safety_assertions = {
        "ppo_parameters_updated": parameter_delta > 0,
        "training_metrics_are_finite": finite_metrics,
        "episode_memory_resets_to_zero": (
            last_memory_diagnostics["episode_reset_zero_max"] == 0.0),
        "recurrent_memory_changes_with_observation": (
            last_memory_diagnostics["memory_update_l2_mean"] > 0.0),
        "evaluation_resolved_all_tasks": (
            last_evaluation["task_count"] == expected_tasks),
        "evaluation_actions_all_legal": (
            last_evaluation["illegal_action_count"] == 0),
        "evaluation_has_no_resource_leak": (
            last_evaluation["resource_leak_count"] == 0),
        "all_three_modes_exposed": (
            set(last_evaluation["exposed_transport_modes"]) == expected_modes),
        "rule_baseline_has_no_resource_leak": (
            rule["resource_leak_count"] == 0),
    }
    policy_minus_rule = {
        "success_rate": (
            last_evaluation["success_rate"] - rule["success_rate"]),
        "mean_reward": (
            last_evaluation["mean_reward"] - rule["mean_reward"]),
        "successful_throughput_tasks_per_hour": (
            last_evaluation["successful_throughput_tasks_per_hour"] -
            rule["successful_throughput_tasks_per_hour"]),
    }
    summary = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "stage": 26,
        "gate": ("SHORT_RECURRENT_CAUSAL_PPO_SMOKE" if
                 args.updates * args.episodes_per_update <= 20 else
                 "RECURRENT_CAUSAL_PPO_TRAINING_RUN"),
        "claim_boundary": (
            "Interface and trainability smoke only; no superiority claim."
            if args.updates * args.episodes_per_update <= 20 else
            "Training only; fresh sentinel evaluation remains required."),
        "execution_mode": args.execution_mode,
        "arrival_profile": args.arrival_profile,
        "observation_variant": CAUSAL_RELEASED_V4,
        "visibility_contract": "RELEASED_ONLY",
        "decision_process_claim": "CAUSAL_PARTIALLY_OBSERVED_SMDP",
        "policy_variant": "RECURRENT_CONTEXT_INTERACTION",
        "recurrent_training_contract": "FULL_EPISODE_BPTT",
        "memory_reset_boundary": "EPISODE",
        "discount_mode": "VARIABLE_SMDP",
        "trace_mode": "DURATION_SCALED",
        "trace_tau_s": args.trace_tau_s,
        "reward_contract": "CONTINUOUS_TIME_V3",
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
        "encoder": encoder.metadata(),
        "model": model.metadata(),
        "ppo": asdict(config),
        "reward_normalization": normalizer.state_dict(),
        "memory_diagnostics": last_memory_diagnostics,
        "parameter_delta_l2": parameter_delta,
        "final_weight": str(output),
        "best_weight": str(best_path),
        "history": str(history_path),
        "evaluation": compact(last_evaluation),
        "rule_baseline": compact(rule),
        "policy_minus_rule": policy_minus_rule,
        "positive_relative_score": baseline_relative_score(
            last_evaluation["mean_reward"], rule["mean_reward"]),
        "contextual_mode_acceptance": {
            "selected_mode_by_cost_reference": context_matrix,
            "diagonal_match_rates": context_rates,
            "minimum_formal_diagonal_rate": .20,
            "formal_gate_passed": context_gate,
        },
        "safety_assertions": safety_assertions,
        "passed": all(safety_assertions.values()),
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
