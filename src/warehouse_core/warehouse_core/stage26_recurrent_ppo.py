"""Episode-sequence recurrent PPO utilities for Stage 26."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .persistent_dispatch import Assignment
from .stage14_ppo import (
    PPOConfig, RolloutStep, RunningReturnNormalizer, compute_smdp_gae,
    profile_environment, tasks_per_episode)
from .stage25_causal_observation import (
    CAUSAL_RELEASED_V4, CausalReleasedWarehouseDispatchGymEnv)
from .stage26_recurrent_policy import RecurrentMaskedCandidateActorCritic


def _nested_mode_matrix(counts: Counter) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for (context, selected_mode), count in sorted(counts.items()):
        rows.setdefault(context, {})[selected_mode] = count
    return rows


def _model_forward(model, state, features, mask, memory,
                   elapsed_s: float = 0.0):
    """Preserve Stage 26 behavior while supporting Stage 27 elapsed time."""
    if getattr(model, "uses_elapsed_time", False):
        return model(
            state, features, mask, memory, elapsed_s=elapsed_s)
    return model(state, features, mask, memory)


def _distribution(model, observation, mask, memory, device,
                  elapsed_s: float = 0.0):
    state = torch.as_tensor(
        observation["state"], dtype=torch.float32, device=device)
    features = torch.as_tensor(
        observation["action_features"][..., :model.action_width],
        dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    logits, value, next_memory = _model_forward(
        model, state, features, mask_tensor, memory, elapsed_s)
    return (torch.distributions.Categorical(logits=logits), value,
            next_memory)


def collect_recurrent_rollouts(
        model: RecurrentMaskedCandidateActorCritic, seeds,
        execution_mode: str, profile: str, device: torch.device,
        discount_mode: str = "VARIABLE_SMDP",
        allowed_transport_modes: tuple[str, ...] | None = None,
        reward_contract: str = "CONTINUOUS_TIME_V3"):
    """Collect complete episodes and retain episode sequence boundaries."""
    model.eval()
    episode_steps: list[list[RolloutStep]] = []
    episodes = []
    interval, arrival_schedule, _ = profile_environment(profile)
    memory_update_norms: list[float] = []
    effective_memory_norms: list[float] = []
    effective_memory_saturation: list[float] = []
    episode_reset_maxima: list[float] = []
    for seed in seeds:
        env = CausalReleasedWarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=interval, arrival_schedule=arrival_schedule,
            observation_variant=CAUSAL_RELEASED_V4,
            allowed_transport_modes=allowed_transport_modes,
            reward_contract=reward_contract,
            handover_sampling="TASK_KEYED")
        observation, info = env.reset(seed=seed)
        memory = model.initial_memory(device=device).squeeze(0)
        previous_delta_time = 0.0
        episode_reset_maxima.append(float(memory.abs().max().item()))
        steps: list[RolloutStep] = []
        done = False
        reward_total = 0.0
        mode_counts = Counter()
        modes_by_phase = Counter()
        exposed_modes = set()
        peak_active = 0
        while not done:
            mask = info["action_mask"].copy()
            exposed_modes.update(
                candidate.transport_mode
                for candidate, legal in zip(env.decision.action_ids, mask)
                if legal and isinstance(candidate, Assignment))
            with torch.no_grad():
                if hasattr(model, "effective_memory_for_diagnostics"):
                    diagnostic_state = torch.as_tensor(
                        observation["state"], dtype=torch.float32,
                        device=device)
                    effective_memory = model.effective_memory_for_diagnostics(
                        diagnostic_state, memory.unsqueeze(0),
                        previous_delta_time).squeeze(0)
                else:
                    effective_memory = memory
                distribution, value, next_memory = _distribution(
                    model, observation, mask, memory, device,
                    previous_delta_time)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)
            memory_update_norms.append(float(torch.linalg.vector_norm(
                next_memory - memory).item()))
            effective_memory_norms.append(float(torch.linalg.vector_norm(
                effective_memory).item()))
            effective_memory_saturation.append(float(
                (effective_memory.abs() >= .95).float().mean().item()))
            action = int(action_tensor.item())
            decoded = env.decision.action_ids[action]
            if isinstance(decoded, Assignment):
                mode_counts[decoded.transport_mode] += 1
                phase = env.task_types.get(
                    decoded.task_id, "BALANCED").split(":", 1)[0]
                modes_by_phase[(phase, decoded.transport_mode)] += 1
                peak_active = max(peak_active, 1)
            next_observation, reward, terminated, truncated, next_info = (
                env.step(action))
            done = terminated or truncated
            discount = (float(next_info["discount"])
                        if discount_mode == "VARIABLE_SMDP" else 0.99)
            steps.append(RolloutStep(
                observation["state"].copy(),
                observation["action_features"].copy(), mask, action,
                float(log_probability.item()), float(value.item()),
                float(reward), discount, done,
                float(next_info["delta_time"])))
            reward_total += reward
            peak_active = max(peak_active, next_info["active_task_count"])
            observation, info = next_observation, next_info
            memory = next_memory.detach()
            previous_delta_time = float(next_info["delta_time"])
        episode_steps.append(steps)
        dispatch = env.dispatch
        episodes.append({
            "seed": seed,
            "execution_mode": env.current_execution_mode,
            "completed": dispatch.completed,
            "failed": dispatch.failed,
            "reward": reward_total,
            "simulated_time": dispatch.now,
            "peak_active_tasks": peak_active,
            "transport_modes": dict(mode_counts),
            "transport_modes_by_phase": _nested_mode_matrix(modes_by_phase),
            "exposed_transport_modes": sorted(exposed_modes),
            "arrival_schedule": arrival_schedule,
            "arrival_schedule_metadata": getattr(
                dispatch, "arrival_schedule_metadata", {}),
            "resource_leak": bool(
                dispatch.resource_claims or
                getattr(dispatch, "active_standby_claims", {}) or
                getattr(dispatch, "active_tasks", {}) or
                any(runtime.robot.task_id or runtime.robot.cargo_id
                    for runtime in dispatch.robots.values())),
        })
    diagnostics = {
        "episode_reset_zero_max": max(episode_reset_maxima, default=0.0),
        "memory_update_l2_mean": float(np.mean(memory_update_norms))
        if memory_update_norms else 0.0,
        "memory_update_l2_max": max(memory_update_norms, default=0.0),
        "effective_memory_l2_mean": float(np.mean(effective_memory_norms))
        if effective_memory_norms else 0.0,
        "effective_memory_saturation_fraction_mean": float(np.mean(
            effective_memory_saturation))
        if effective_memory_saturation else 0.0,
    }
    return episode_steps, episodes, diagnostics


def recurrent_ppo_update(
        model: RecurrentMaskedCandidateActorCritic,
        optimizer: torch.optim.Optimizer,
        episode_steps: list[list[RolloutStep]], config: PPOConfig,
        device: torch.device, generator: torch.Generator,
        reward_normalizer: RunningReturnNormalizer | None = None):
    """Update by full episode, preserving BPTT and episode memory resets."""
    flat_steps = [step for episode in episode_steps for step in episode]
    if not flat_steps:
        raise ValueError("recurrent PPO update requires rollout steps")
    raw_rewards = np.asarray(
        [step.reward for step in flat_steps], dtype=np.float32)
    training_rewards = (
        reward_normalizer.normalize_rollout(flat_steps, update=True)
        if reward_normalizer is not None else raw_rewards)
    advantages_np, returns_np = compute_smdp_gae(
        training_rewards, [step.old_value for step in flat_steps],
        [step.discount for step in flat_steps],
        [step.episode_end for step in flat_steps], config.gae_lambda,
        durations=[step.delta_time for step in flat_steps],
        trace_mode=config.trace_mode, trace_tau_s=config.trace_tau_s)
    advantages_np = (advantages_np - advantages_np.mean()) / (
        advantages_np.std() + 1e-8)
    slices = []
    cursor = 0
    for episode in episode_steps:
        slices.append(slice(cursor, cursor + len(episode)))
        cursor += len(episode)

    totals = Counter()
    update_count = 0
    model.train()
    for _ in range(config.ppo_epochs):
        order = torch.randperm(len(episode_steps), generator=generator).tolist()
        for episode_index in order:
            episode = episode_steps[episode_index]
            segment = slices[episode_index]
            memory = model.initial_memory(device=device).squeeze(0)
            previous_delta_time = 0.0
            logits_rows = []
            value_rows = []
            for step in episode:
                state = torch.as_tensor(
                    step.state, dtype=torch.float32, device=device)
                features = torch.as_tensor(
                    step.action_features[..., :model.action_width],
                    dtype=torch.float32, device=device)
                mask = torch.as_tensor(
                    step.action_mask, dtype=torch.bool, device=device)
                logits, value, memory = _model_forward(
                    model, state, features, mask, memory,
                    previous_delta_time)
                logits_rows.append(logits)
                value_rows.append(value)
                previous_delta_time = float(step.delta_time)
            logits_tensor = torch.stack(logits_rows)
            values_now = torch.stack(value_rows)
            actions = torch.as_tensor(
                [step.action for step in episode],
                dtype=torch.long, device=device)
            old_log_probabilities = torch.as_tensor(
                [step.old_log_probability for step in episode],
                dtype=torch.float32, device=device)
            advantages = torch.as_tensor(
                advantages_np[segment], dtype=torch.float32, device=device)
            returns = torch.as_tensor(
                returns_np[segment], dtype=torch.float32, device=device)
            distribution = torch.distributions.Categorical(
                logits=logits_tensor)
            log_probabilities = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio,
                1.0 + config.clip_ratio) * advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values_now, returns)
            loss = (actor_loss + config.value_coefficient * value_loss -
                    config.entropy_coefficient * entropy)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = (
                    old_log_probabilities - log_probabilities).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_ratio)
                    .float().mean())
            totals.update({
                "loss": float(loss.item()),
                "actor_loss": float(actor_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approximate_kl": float(approximate_kl.item()),
                "clip_fraction": float(clip_fraction.item()),
                "gradient_norm": float(gradient_norm.item()),
            })
            update_count += 1
    result = {key: value / update_count for key, value in totals.items()}
    result.update({
        "raw_reward_mean": float(raw_rewards.mean()),
        "normalized_reward_mean": float(training_rewards.mean()),
        "normalized_reward_std": float(training_rewards.std()),
        "return_normalizer_variance": (
            reward_normalizer.variance if reward_normalizer else 1.0),
        "recurrent_update_unit": "FULL_EPISODE_BPTT",
    })
    return result


@torch.no_grad()
def evaluate_recurrent(
        model: RecurrentMaskedCandidateActorCritic, seeds,
        execution_mode: str, profile: str, device: torch.device,
        use_rule: bool = False,
        allowed_transport_modes: tuple[str, ...] | None = None,
        reward_contract: str = "CONTINUOUS_TIME_V3",
        memory_mode: str = "RECURRENT"):
    if memory_mode not in {"RECURRENT", "RESET_EACH_DECISION"}:
        raise ValueError(f"unsupported evaluation memory mode: {memory_mode}")
    model.eval()
    interval, arrival_schedule, _ = profile_environment(profile)
    episodes = []
    total_modes = Counter()
    total_modes_by_phase = Counter()
    total_exposed_modes = set()
    total_mode_by_cost_reference = Counter()
    total_mode_by_floor_relation = Counter()
    total_agreements = 0
    total_decisions = 0
    memory_update_norms: list[float] = []
    effective_memory_norms: list[float] = []
    effective_memory_saturation: list[float] = []
    for seed in seeds:
        env = CausalReleasedWarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=interval, arrival_schedule=arrival_schedule,
            observation_variant=CAUSAL_RELEASED_V4,
            handover_sampling="TASK_KEYED",
            allowed_transport_modes=allowed_transport_modes,
            reward_contract=reward_contract)
        observation, info = env.reset(seed=seed)
        memory = model.initial_memory(device=device).squeeze(0)
        previous_delta_time = 0.0
        done = False
        reward_total = 0.0
        illegal = 0
        peak_active = 0
        modes = Counter()
        modes_by_phase = Counter()
        exposed_modes = set()
        mode_by_cost_reference = Counter()
        mode_by_floor_relation = Counter()
        agreements = 0
        decisions = 0
        while not done:
            exposed_modes.update(
                candidate.transport_mode
                for candidate, legal in zip(
                    env.decision.action_ids, info["action_mask"])
                if legal and isinstance(candidate, Assignment))
            if use_rule:
                action = env.rule_action()
                next_memory = memory
            else:
                decision_memory = (
                    memory if memory_mode == "RECURRENT" else
                    model.initial_memory(device=device).squeeze(0))
                if hasattr(model, "effective_memory_for_diagnostics"):
                    diagnostic_state = torch.as_tensor(
                        observation["state"], dtype=torch.float32,
                        device=device)
                    effective_memory = model.effective_memory_for_diagnostics(
                        diagnostic_state, decision_memory.unsqueeze(0),
                        previous_delta_time).squeeze(0)
                else:
                    effective_memory = decision_memory
                distribution, _, next_memory = _distribution(
                    model, observation, info["action_mask"],
                    decision_memory, device, previous_delta_time)
                action = int(torch.argmax(distribution.logits).item())
                memory_update_norms.append(float(torch.linalg.vector_norm(
                    next_memory - decision_memory).item()))
                effective_memory_norms.append(float(torch.linalg.vector_norm(
                    effective_memory).item()))
                effective_memory_saturation.append(float(
                    (effective_memory.abs() >= .95).float().mean().item()))
            illegal += int(not info["action_mask"][action])
            decoded = env.decision.action_ids[action]
            if isinstance(decoded, Assignment):
                modes[decoded.transport_mode] += 1
                phase = env.task_types.get(
                    decoded.task_id, "BALANCED").split(":", 1)[0]
                modes_by_phase[(phase, decoded.transport_mode)] += 1
                legal_same_task = [
                    candidate for candidate, legal in zip(
                        env.decision.action_ids, info["action_mask"])
                    if legal and isinstance(candidate, Assignment) and
                    candidate.task_id == decoded.task_id]
                if legal_same_task:
                    reference = min(
                        legal_same_task,
                        key=lambda candidate: candidate.estimated_cost)
                    mode_by_cost_reference[
                        (reference.transport_mode,
                         decoded.transport_mode)] += 1
                    agreements += int(
                        reference.transport_mode == decoded.transport_mode)
                    decisions += 1
                task = env.dispatch._task(decoded.task_id).task
                relation = ("CROSS_FLOOR" if
                            task.source.floor != task.target.floor else
                            "SAME_FLOOR")
                mode_by_floor_relation[
                    (relation, decoded.transport_mode)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            peak_active = max(peak_active, info["active_task_count"])
            memory = (
                next_memory.detach() if memory_mode == "RECURRENT" else
                model.initial_memory(device=device).squeeze(0))
            previous_delta_time = float(info["delta_time"])
            done = terminated or truncated
        dispatch = env.dispatch
        scheduled = len(env.task_types)
        unresolved = max(
            scheduled - dispatch.completed - dispatch.failed, 0)
        leak = bool(
            dispatch.resource_claims or
            getattr(dispatch, "active_standby_claims", {}) or
            getattr(dispatch, "active_tasks", {}) or
            any(runtime.robot.task_id or runtime.robot.cargo_id
                for runtime in dispatch.robots.values()))
        total_modes.update(modes)
        total_modes_by_phase.update(modes_by_phase)
        total_exposed_modes.update(exposed_modes)
        total_mode_by_cost_reference.update(mode_by_cost_reference)
        total_mode_by_floor_relation.update(mode_by_floor_relation)
        total_agreements += agreements
        total_decisions += decisions
        episodes.append({
            "seed": seed, "completed": dispatch.completed,
            "failed": dispatch.failed, "scheduled_tasks": scheduled,
            "unresolved_tasks": unresolved, "reward": reward_total,
            "simulated_time": dispatch.now, "illegal_actions": illegal,
            "peak_active_tasks": peak_active,
            "resource_leak": leak,
            "transport_modes": dict(modes),
            "transport_modes_by_phase": _nested_mode_matrix(modes_by_phase),
            "exposed_transport_modes": sorted(exposed_modes),
            "arrival_schedule_metadata": getattr(
                dispatch, "arrival_schedule_metadata", {}),
        })
    total_time = sum(item["simulated_time"] for item in episodes)
    completed = sum(item["completed"] for item in episodes)
    failed = sum(item["failed"] for item in episodes)
    scheduled = sum(item["scheduled_tasks"] for item in episodes)
    unresolved = sum(item["unresolved_tasks"] for item in episodes)
    resolved = completed + failed
    return {
        "episode_count": len(episodes),
        "task_count": resolved,
        "scheduled_task_count": scheduled,
        "completed": completed,
        "failed": failed,
        "unresolved": unresolved,
        "effective_failed": failed + unresolved,
        "resolution_rate": resolved / max(scheduled, 1),
        "success_rate": completed / max(scheduled, 1),
        "mean_reward": float(np.mean(
            [item["reward"] for item in episodes])),
        "mean_simulated_time": float(np.mean(
            [item["simulated_time"] for item in episodes])),
        "throughput_tasks_per_hour": resolved / max(total_time, 1e-9) * 3600,
        "resolved_throughput_tasks_per_hour": (
            resolved / max(total_time, 1e-9) * 3600),
        "successful_throughput_tasks_per_hour": (
            completed / max(total_time, 1e-9) * 3600),
        "handover_sampling": "TASK_KEYED",
        "memory_mode": memory_mode,
        "illegal_action_count": sum(
            item["illegal_actions"] for item in episodes),
        "maximum_active_tasks": max(
            (item["peak_active_tasks"] for item in episodes), default=0),
        "resource_leak_count": sum(
            item["resource_leak"] for item in episodes),
        "transport_modes": dict(total_modes),
        "transport_modes_by_phase": _nested_mode_matrix(total_modes_by_phase),
        "arrival_schedule": arrival_schedule,
        "exposed_transport_modes": sorted(total_exposed_modes),
        "selected_mode_by_cost_reference": _nested_mode_matrix(
            total_mode_by_cost_reference),
        "selected_mode_by_floor_relation": _nested_mode_matrix(
            total_mode_by_floor_relation),
        "cost_reference_mode_agreement": (
            total_agreements / max(total_decisions, 1)),
        "cost_reference_decision_count": total_decisions,
        "memory_update_l2_mean": (
            float(np.mean(memory_update_norms))
            if memory_update_norms else 0.0),
        "effective_memory_l2_mean": (
            float(np.mean(effective_memory_norms))
            if effective_memory_norms else 0.0),
        "effective_memory_saturation_fraction_mean": (
            float(np.mean(effective_memory_saturation))
            if effective_memory_saturation else 0.0),
        "episodes": episodes,
    }


def load_recurrent_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["model_metadata"]
    if metadata.get("architecture") != \
            "recurrent_contextual_masked_candidate_actor_critic_v1":
        raise ValueError("checkpoint is not a Stage 26 recurrent policy")
    model = RecurrentMaskedCandidateActorCritic(
        state_width=int(metadata["state_width"]),
        action_width=int(metadata["action_width"]),
        hidden_width=int(metadata["hidden_width"]),
        memory_width=int(metadata["memory_width"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


__all__ = [
    "PPOConfig", "RunningReturnNormalizer", "tasks_per_episode",
    "collect_recurrent_rollouts", "recurrent_ppo_update",
    "evaluate_recurrent", "load_recurrent_checkpoint",
]
