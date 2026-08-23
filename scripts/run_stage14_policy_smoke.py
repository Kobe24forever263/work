#!/usr/bin/env python3
"""Warm-start and validate the Stage 14 shared candidate-scoring policy."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.persistent_dispatch import Assignment
from warehouse_core.stage14_policy import MaskedCandidateActorCritic
from warehouse_core.stage14_training import WarehouseDispatchGymEnv
from warehouse_core.stage14_training import (
    BURST_INTERVAL, DENSE_INTERVAL, MEDIUM_INTERVAL)


def arrival_interval(profile):
    return {
        "MEDIUM": MEDIUM_INTERVAL,
        "DENSE": DENSE_INTERVAL,
        "BURST": BURST_INTERVAL,
    }[profile]


def collect_rule_dataset(seeds, execution_mode, observation_variant,
                         arrival_profile):
    samples = []
    label_modes = Counter()
    for seed in seeds:
        env = WarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=arrival_interval(arrival_profile),
            observation_variant=observation_variant)
        observation, info = env.reset(seed=seed)
        done = False
        while not done:
            action = env.rule_action()
            decoded = env.decision.action_ids[action]
            state = observation["state"].copy()
            features = observation["action_features"].copy()
            mask = info["action_mask"].copy()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if not isinstance(decoded, Assignment):
                continue
            legal_indices = np.flatnonzero(mask)
            target_positions = np.flatnonzero(legal_indices == action)
            if target_positions.size != 1:
                raise RuntimeError("rule action is absent from the legal mask")
            valid_count = int(legal_indices.size)
            samples.append({
                "state": state,
                # DENSE may expose a legal WAIT after illegal/padded actions,
                # so compact by mask rather than assuming a legal prefix.
                "features": features[legal_indices],
                "target": int(target_positions[0]),
                "reward": reward,
                "valid_count": valid_count,
            })
            label_modes[decoded.transport_mode] += 1
    max_candidates = max(sample["valid_count"] for sample in samples)
    states = np.stack([sample["state"] for sample in samples])
    features = np.zeros(
        (len(samples), max_candidates, samples[0]["features"].shape[1]),
        dtype=np.float32)
    masks = np.zeros((len(samples), max_candidates), dtype=np.bool_)
    for index, sample in enumerate(samples):
        count = sample["valid_count"]
        features[index, :count] = sample["features"]
        masks[index, :count] = True
    targets = np.asarray([sample["target"] for sample in samples],
                         dtype=np.int64)
    rewards = np.asarray([sample["reward"] for sample in samples],
                         dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(states), torch.from_numpy(features),
        torch.from_numpy(masks), torch.from_numpy(targets),
        torch.from_numpy(rewards))
    return dataset, max_candidates, dict(label_modes)


@torch.no_grad()
def dataset_metrics(model, loader):
    model.eval()
    total_loss = total_actor = total_value = 0.0
    correct = count = 0
    entropy_total = 0.0
    for state, features, mask, target, reward in loader:
        logits, value = model(state.float(), features.float(), mask.bool())
        actor = nn.functional.cross_entropy(logits, target)
        value_loss = nn.functional.mse_loss(value, reward.float())
        probabilities = torch.softmax(logits, dim=1)
        log_probabilities = torch.log_softmax(logits, dim=1)
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        batch = state.shape[0]
        total_actor += actor.item() * batch
        total_value += value_loss.item() * batch
        total_loss += actor.item() * batch
        entropy_total += entropy.item() * batch
        correct += int((torch.argmax(logits, dim=1) == target).sum())
        count += batch
    return {
        "loss": total_loss / count,
        "actor_loss": total_actor / count,
        "value_loss": total_value / count,
        "entropy": entropy_total / count,
        "accuracy": correct / count,
    }


def train(model, dataset, epochs, batch_size, learning_rate, run_seed):
    generator = torch.Generator().manual_seed(run_seed + 17)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=generator)
    audit_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial = dataset_metrics(model, audit_loader)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for state, features, mask, target, reward in loader:
            logits, value = model(state.float(), features.float(), mask.bool())
            actor_loss = nn.functional.cross_entropy(logits, target)
            # Warm-start only the contextual actor.  The PPO critic must learn
            # normalized SMDP returns later; fitting it to one-step raw rewards
            # here would put it on an incompatible scale.
            loss = actor_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = dataset_metrics(model, audit_loader)
        metrics["epoch"] = epoch
        history.append(metrics)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={metrics['loss']:.5f} "
                  f"actor={metrics['actor_loss']:.5f} "
                  f"value={metrics['value_loss']:.5f} "
                  f"entropy={metrics['entropy']:.5f} "
                  f"accuracy={metrics['accuracy']:.3f}")
    return initial, history[-1], history


def _mode(action):
    return action.transport_mode if isinstance(action, Assignment) else "WAIT"


@torch.no_grad()
def evaluate_policy(model, seeds, execution_mode, observation_variant,
                    arrival_profile="MEDIUM", use_rule=False):
    rows = []
    episode_summaries = []
    for seed in seeds:
        env = WarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=arrival_interval(arrival_profile),
            observation_variant=observation_variant)
        observation, info = env.reset(seed=seed)
        done = False
        reward_total = 0.0
        exact = mode_matches = assignment_count = illegal = 0
        modes = Counter()
        while not done:
            rule_action = env.rule_action()
            if use_rule:
                action = rule_action
            else:
                action = model.select_action(
                    observation, info["action_mask"], deterministic=True)
            illegal += int(not info["action_mask"][action])
            selected = env.decision.action_ids[action]
            rule_selected = env.decision.action_ids[rule_action]
            is_assignment = isinstance(selected, Assignment)
            if is_assignment:
                assignment_count += 1
                exact += int(action == rule_action)
                mode_matches += int(_mode(selected) == _mode(rule_selected))
                modes[_mode(selected)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            done = terminated or truncated
            rows.append({
                "seed": seed,
                "action": action,
                "rule_action": rule_action,
                "mode": _mode(selected),
                "rule_mode": _mode(rule_selected),
                "legal": not bool(illegal),
                "reward": reward,
                "task_result": info["selected_action"].get("task_result", ""),
                "failure_reason": info["selected_action"].get(
                    "failure_reason", ""),
                "completed": info["completed"],
                "terminated": terminated,
                "truncated": truncated,
            })
        episode_summaries.append({
            "seed": seed,
            "completed": env.dispatch.completed,
            "failed": env.dispatch.failed,
            "reward_total": reward_total,
            "simulated_time": env.dispatch.now,
            "exact_action_matches": exact,
            "mode_matches": mode_matches,
            "assignment_count": assignment_count,
            "illegal_actions": illegal,
            "transport_modes": dict(modes),
        })
    total_assignments = sum(item["assignment_count"]
                            for item in episode_summaries)
    mode_counts = Counter()
    for item in episode_summaries:
        mode_counts.update(item["transport_modes"])
    return {
        "episode_count": len(episode_summaries),
        "task_count": sum(item["completed"] + item["failed"]
                          for item in episode_summaries),
        "completed_task_count": sum(item["completed"]
                                    for item in episode_summaries),
        "failure_count": sum(item["failed"] for item in episode_summaries),
        "non_handover_failure_count": sum(
            row["task_result"] == "FAILED" and
            row["failure_reason"] != "HANDOVER_TIMEOUT" for row in rows),
        "illegal_action_count": sum(item["illegal_actions"]
                                    for item in episode_summaries),
        "exact_action_agreement": (
            sum(item["exact_action_matches"] for item in episode_summaries) /
            total_assignments),
        "mode_agreement": (
            sum(item["mode_matches"] for item in episode_summaries) /
            total_assignments),
        "mean_episode_reward": (
            sum(item["reward_total"] for item in episode_summaries) /
            len(episode_summaries)),
        "mean_simulated_time": (
            sum(item["simulated_time"] for item in episode_summaries) /
            len(episode_summaries)),
        "transport_modes": dict(mode_counts),
        "per_episode": episode_summaries,
        "rows": rows,
    }


def compact_evaluation(evaluation):
    return {key: value for key, value in evaluation.items()
            if key not in {"per_episode", "rows"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--execution-mode", choices=(
        "SERIAL", "CONCURRENT"), default="CONCURRENT")
    parser.add_argument("--arrival-profile", choices=(
        "MEDIUM", "DENSE", "BURST"),
                        default="MEDIUM")
    parser.add_argument("--run-seed", type=int, default=41000040)
    parser.add_argument("--training-seed-count", type=int, default=20)
    parser.add_argument("--validation-seed-start", type=int,
                        default=42000000)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--observation-variant", choices=(
        "FULL_CONTEXT_V2", "NO_QUEUE_RESOURCE", "NO_HANDOVER_CUES",
        "NO_PERSISTENT_POSITION", "NO_POSITION_ONLY", "NO_ETA_COST_ONLY",
        "NO_HISTORY_ONLY", "NO_EXPLICIT_QUEUE_RESOURCE",
        "NO_EXPLICIT_HANDOVER_RISK"), default="FULL_CONTEXT_V2")
    parser.add_argument("--policy-variant", choices=(
        "CONTEXT_INTERACTION", "FLAT_MASKED_PPO"),
        default="CONTEXT_INTERACTION")
    parser.add_argument(
        "--allow-interface-only-continuation", action="store_true",
        help=("Allow PPO continuation when the ablated observation cannot "
              "fully identify the full-information rule teacher. Quality "
              "assertions remain recorded as failed."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.training_seed_count,
           args.validation_episodes) <= 0:
        parser.error("all count arguments must be positive")

    np.random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    torch.set_num_threads(4)
    training_seeds = list(range(
        args.run_seed, args.run_seed + args.training_seed_count))
    evaluation_seeds = list(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_episodes))
    dataset, max_candidates, label_modes = collect_rule_dataset(
        training_seeds, args.execution_mode, args.observation_variant,
        args.arrival_profile)
    print(f"dataset={len(dataset)} max_candidates={max_candidates} "
          f"labels={label_modes}")

    model = MaskedCandidateActorCritic(
        hidden_width=96, policy_variant=args.policy_variant)
    before = evaluate_policy(
        model, evaluation_seeds, args.execution_mode,
        args.observation_variant, args.arrival_profile)
    rule = evaluate_policy(
        model, evaluation_seeds, args.execution_mode,
        args.observation_variant, args.arrival_profile, use_rule=True)
    initial, final, history = train(
        model, dataset, args.epochs, args.batch_size, args.learning_rate,
        args.run_seed)
    after = evaluate_policy(
        model, evaluation_seeds, args.execution_mode,
        args.observation_variant, args.arrival_profile)

    before_compact = compact_evaluation(before)
    after_compact = compact_evaluation(after)
    rule_compact = compact_evaluation(rule)
    assertions = {
        "training_dataset_has_expected_task_count": (
            len(dataset) == args.training_seed_count * 20),
        "training_labels_cover_three_modes": set(label_modes) == {
            "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"},
        "loss_decreased": final["loss"] < initial["loss"],
        "classification_accuracy_improved": (
            final["accuracy"] > initial["accuracy"] + .25),
        "heldout_policy_resolved_one_hundred_tasks": (
            after["task_count"] == 100),
        "heldout_failures_only_come_from_p95_handover_timeout": (
            after["non_handover_failure_count"] == 0),
        "heldout_actions_all_legal": after["illegal_action_count"] == 0,
        "heldout_uses_all_three_modes": set(after["transport_modes"]) == {
            "SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"},
        "heldout_mode_agreement_at_least_90_percent": (
            after["mode_agreement"] >= .9),
        "heldout_exact_action_agreement_at_least_70_percent": (
            after["exact_action_agreement"] >= .7),
        "heldout_reward_improved_from_initial": (
            after["mean_episode_reward"] > before["mean_episode_reward"]),
        "heldout_reward_within_two_percent_of_rule": (
            after["mean_episode_reward"] >=
            rule["mean_episode_reward"] -
            abs(rule["mean_episode_reward"]) * .02),
    }
    # Reward proximity is diagnostic at warm start, not an admission gate for
    # PPO.  Under DENSE/BURST, small differences in queue timing can move raw
    # episode return by more than 2% even when imitation accuracy, legality,
    # resource safety, and mode coverage are all sound.  Superiority belongs
    # to the paired held-out PPO evaluation, not actor distillation.
    gate_assertions = {
        key: value for key, value in assertions.items()
        if key not in {
            "heldout_reward_improved_from_initial",
            "heldout_reward_within_two_percent_of_rule"}}
    # Minimal interface/safety contract used only by explicitly declared
    # information-removal ablations.  These runs are expected to lose mode
    # discrimination; requiring full imitation would prevent measuring that
    # degradation with PPO.
    interface_continuation_assertions = {
        "training_dataset_has_expected_task_count":
            assertions["training_dataset_has_expected_task_count"],
        "training_labels_cover_three_modes":
            assertions["training_labels_cover_three_modes"],
        "loss_decreased": assertions["loss_decreased"],
        "heldout_policy_resolved_one_hundred_tasks":
            assertions["heldout_policy_resolved_one_hundred_tasks"],
        "heldout_failures_only_come_from_p95_handover_timeout":
            assertions["heldout_failures_only_come_from_p95_handover_timeout"],
        "heldout_actions_all_legal":
            assertions["heldout_actions_all_legal"],
    }
    continuation_assertions = {
        **interface_continuation_assertions,
        "heldout_uses_all_three_modes":
            assertions["heldout_uses_all_three_modes"],
        "heldout_mode_agreement_at_least_90_percent":
            assertions["heldout_mode_agreement_at_least_90_percent"],
        "heldout_exact_action_agreement_at_least_70_percent":
            assertions["heldout_exact_action_agreement_at_least_70_percent"],
    }
    interface_continuation_allowed = (
        args.allow_interface_only_continuation and
        all(interface_continuation_assertions.values()))
    summary = {
        "schema_version": "warehouse_stage18_warm_start_v1",
        "stage": 14,
        "gate": "CONTEXT_CONDITIONED_POLICY_WARM_START",
        "training_type": "contextual_rule_actor_distillation_warm_start",
        "execution_mode": args.execution_mode,
        "arrival_profile": args.arrival_profile,
        "run_seed": args.run_seed,
        "observation_variant": args.observation_variant,
        "policy_variant": args.policy_variant,
        "training_seed_start": training_seeds[0],
        "training_seed_end": training_seeds[-1],
        "evaluation_seeds": evaluation_seeds,
        "training_task_count": len(dataset),
        "maximum_candidates_in_training": max_candidates,
        "training_label_modes": label_modes,
        "model": model.metadata(),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "gradient_clip": 1.0,
            "value_loss_weight": 0.0,
            "critic_note": (
                "Critic intentionally left for normalized SMDP-PPO."),
        },
        "training_initial": initial,
        "training_final": final,
        "evaluation_before": before_compact,
        "evaluation_after": after_compact,
        "rule_baseline": rule_compact,
        "assertions": assertions,
        "gate_assertions": gate_assertions,
        "diagnostics": {
            "heldout_reward_improved_from_initial":
                assertions["heldout_reward_improved_from_initial"],
            "heldout_reward_within_two_percent_of_rule":
                assertions["heldout_reward_within_two_percent_of_rule"],
            "note": (
                "Warm-start reward comparisons are diagnostic only. Formal "
                "acceptance uses teacher imitation accuracy, three-mode "
                "coverage, legality, and resource safety; reward superiority "
                "is tested after PPO with paired held-out seeds."),
        },
        "passed": all(gate_assertions.values()),
        "continuation_policy": (
            "INTERFACE_ONLY_FOR_INFORMATION_REMOVAL_ABLATION"
            if args.allow_interface_only_continuation else
            "IMITATION_SAFETY_GATE_REWARD_DIAGNOSTIC_ONLY"),
        "continuation_assertions": continuation_assertions,
        "interface_continuation_assertions": (
            interface_continuation_assertions),
        "ppo_continuation_allowed": (
            all(gate_assertions.values()) or interface_continuation_allowed),
    }
    output = (args.output or (
        WORK_ROOT / "results" / "stage18" / "warm_start" /
        f"warm_start_{args.observation_variant.lower()}_"
        f"{args.policy_variant.lower()}_seed_{args.run_seed}.pt")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": summary["model"],
        "training_type": summary["training_type"],
        "run_seed": args.run_seed,
        "training_seeds": training_seeds,
        "validation_seeds": evaluation_seeds,
        "observation_variant": args.observation_variant,
        "policy_variant": args.policy_variant,
    }, output)
    output.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary["weight"] = str(output)
    summary["history"] = str(output.with_suffix(".history.json"))
    summary["evaluation_rows"] = str(output.with_suffix(".eval.jsonl"))
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    output.with_suffix(".eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in after["rows"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ppo_continuation_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
