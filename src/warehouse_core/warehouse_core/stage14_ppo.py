"""Masked SMDP-PPO utilities for the Stage 14 dispatch curriculum."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# NumPy must be imported before torch in the Apple Silicon ROS/Pixi runtime.
import numpy as np
import torch
from torch import nn

from .persistent_dispatch import Assignment
from .stage14_policy import MaskedCandidateActorCritic
from .stage14_training import (
    BURST_INTERVAL, DENSE_INTERVAL, MEDIUM_INTERVAL,
    WarehouseDispatchGymEnv)


def baseline_relative_score(policy_reward: float, rule_reward: float,
                            scale_floor: float = 10.0) -> dict[str, float]:
    """Positive reporting score with the rule baseline fixed at 100.

    This monotone transform is reporting-only.  It must never replace the raw
    environment reward used by PPO, checkpoint selection, or acceptance.
    """
    scale = max(abs(float(rule_reward)), float(scale_floor))
    exponent = float(np.clip(
        (float(policy_reward) - float(rule_reward)) / scale, -50.0, 50.0))
    return {
        "policy_score": float(100.0 * np.exp(exponent)),
        "rule_baseline_score": 100.0,
        "raw_reward_delta": float(policy_reward) - float(rule_reward),
        "normalization_scale": scale,
        "formula": "100*exp((policy_reward-rule_reward)/max(abs(rule_reward),10))",
    }


def _nested_mode_matrix(counts: Counter) -> dict[str, dict[str, int]]:
    """Serialize (context, selected mode) counts as a readable matrix."""
    rows = {}
    for (context, selected_mode), count in sorted(counts.items()):
        rows.setdefault(context, {})[selected_mode] = count
    return rows


@dataclass(frozen=True)
class PPOConfig:
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 1.0
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 32


@dataclass
class RolloutStep:
    state: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
    reward: float
    discount: float
    episode_end: bool


@dataclass
class RunningReturnNormalizer:
    """Normalize rewards by the running std of variable-discount returns.

    The mean is tracked for diagnostics but is not subtracted, matching the
    common PPO reward-normalization convention.  Evaluation continues to
    report raw environment rewards.
    """

    count: float = 1e-4
    mean: float = 0.0
    variance: float = 1.0
    epsilon: float = 1e-8
    clip: float = 10.0

    def _update(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        batch_count = float(samples.size)
        batch_mean = float(samples.mean())
        batch_variance = float(samples.var())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        moment_a = self.variance * self.count
        moment_b = batch_variance * batch_count
        combined = (moment_a + moment_b +
                    delta * delta * self.count * batch_count / total)
        self.mean = new_mean
        self.variance = max(combined / total, self.epsilon)
        self.count = total

    def normalize_rollout(self, steps: list[RolloutStep],
                          update: bool = True) -> np.ndarray:
        discounted_return = 0.0
        returns = []
        for step in steps:
            discounted_return = (
                step.reward + step.discount * discounted_return)
            returns.append(discounted_return)
            if step.episode_end:
                discounted_return = 0.0
        if update:
            self._update(np.asarray(returns, dtype=np.float64))
        scale = float(np.sqrt(self.variance + self.epsilon))
        rewards = np.asarray(
            [step.reward for step in steps], dtype=np.float32) / scale
        return np.clip(rewards, -self.clip, self.clip)

    def state_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "subtract_mean": False,
            "source": "variable_discounted_return_std_v1",
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningReturnNormalizer":
        return cls(
            count=float(state["count"]), mean=float(state["mean"]),
            variance=float(state["variance"]),
            epsilon=float(state.get("epsilon", 1e-8)),
            clip=float(state.get("clip", 10.0)))


def arrival_interval(profile: str) -> tuple[float, float]:
    profile = profile.upper()
    if profile == "MEDIUM":
        return MEDIUM_INTERVAL
    if profile == "DENSE":
        return DENSE_INTERVAL
    if profile == "BURST":
        return BURST_INTERVAL
    if profile == "MIXED_CURRICULUM":
        # Individual phases own their intervals; this value is only the
        # constructor fallback and is not used to generate phase arrivals.
        return MEDIUM_INTERVAL
    raise ValueError(f"unsupported arrival profile: {profile}")


def profile_environment(profile: str) -> tuple[tuple[float, float], str, int]:
    profile = profile.upper()
    if profile == "MIXED_CURRICULUM":
        return MEDIUM_INTERVAL, "MIXED_CURRICULUM", 80
    return arrival_interval(profile), "BALANCED", 20


def tasks_per_episode(profile: str) -> int:
    return profile_environment(profile)[2]


def _distribution(model, observation, mask, device):
    state = torch.as_tensor(
        observation["state"], dtype=torch.float32, device=device)
    features = torch.as_tensor(
        observation["action_features"][..., :model.action_width],
        dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    logits, value = model(state, features, mask_tensor)
    return torch.distributions.Categorical(logits=logits), value


def collect_rollouts(model: MaskedCandidateActorCritic, seeds,
                     execution_mode: str, profile: str,
                     device: torch.device,
                     observation_variant: str = "FULL_CONTEXT_V2",
                     discount_mode: str = "VARIABLE_SMDP"):
    """Collect complete on-policy episodes and preserve SMDP discounts."""
    model.eval()
    steps: list[RolloutStep] = []
    episodes = []
    interval, arrival_schedule, _ = profile_environment(profile)
    for seed in seeds:
        env = WarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=interval,
            arrival_schedule=arrival_schedule,
            observation_variant=observation_variant)
        observation, info = env.reset(seed=seed)
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
                for candidate, legal in zip(
                    env.decision.action_ids, mask)
                if legal and isinstance(candidate, Assignment))
            with torch.no_grad():
                distribution, value = _distribution(
                    model, observation, mask, device)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)
            action = int(action_tensor.item())
            decoded = env.decision.action_ids[action]
            if isinstance(decoded, Assignment):
                mode_counts[decoded.transport_mode] += 1
                phase = env.task_types.get(decoded.task_id, "BALANCED").split(":", 1)[0]
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
                float(reward), discount, done))
            reward_total += reward
            peak_active = max(peak_active, next_info["active_task_count"])
            observation, info = next_observation, next_info
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
    return steps, episodes


def compute_smdp_gae(rewards, values, discounts, episode_ends,
                     gae_lambda: float):
    """Compute GAE using each event transition's variable-time discount."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    discounts = np.asarray(discounts, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.bool_)
    advantages = np.zeros_like(rewards)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if episode_ends[index]:
            next_advantage = 0.0
            next_value = 0.0
        delta = rewards[index] + discounts[index] * next_value - values[index]
        advantages[index] = (
            delta + discounts[index] * gae_lambda * next_advantage)
        next_advantage = float(advantages[index])
        next_value = float(values[index])
    return advantages, advantages + values


def ppo_update(model: MaskedCandidateActorCritic,
               optimizer: torch.optim.Optimizer,
               steps: list[RolloutStep], config: PPOConfig,
               device: torch.device, generator: torch.Generator,
               reward_normalizer: RunningReturnNormalizer | None = None):
    if not steps:
        raise ValueError("PPO update requires rollout steps")
    states = torch.as_tensor(
        np.stack([step.state for step in steps]),
        dtype=torch.float32, device=device)
    features = torch.as_tensor(
        np.stack([step.action_features for step in steps]),
        dtype=torch.float32, device=device)
    masks = torch.as_tensor(
        np.stack([step.action_mask for step in steps]),
        dtype=torch.bool, device=device)
    actions = torch.as_tensor(
        [step.action for step in steps], dtype=torch.long, device=device)
    old_log_probabilities = torch.as_tensor(
        [step.old_log_probability for step in steps],
        dtype=torch.float32, device=device)
    raw_rewards = np.asarray(
        [step.reward for step in steps], dtype=np.float32)
    training_rewards = (
        reward_normalizer.normalize_rollout(steps, update=True)
        if reward_normalizer is not None else raw_rewards)
    advantages_np, returns_np = compute_smdp_gae(
        training_rewards,
        [step.old_value for step in steps],
        [step.discount for step in steps],
        [step.episode_end for step in steps], config.gae_lambda)
    advantages = torch.as_tensor(
        advantages_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(
        returns_np, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8)

    totals = Counter()
    update_count = 0
    model.train()
    for _ in range(config.ppo_epochs):
        order = torch.randperm(len(steps), generator=generator)
        for start in range(0, len(steps), config.minibatch_size):
            indices = order[start:start + config.minibatch_size].to(device)
            logits, values_now = model(
                states[indices], features[indices], masks[indices])
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[indices])
            entropy = distribution.entropy().mean()
            ratio = torch.exp(
                log_probabilities - old_log_probabilities[indices])
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio,
                1.0 + config.clip_ratio) * advantages[indices]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(
                values_now, returns[indices])
            loss = (actor_loss + config.value_coefficient * value_loss -
                    config.entropy_coefficient * entropy)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = (
                    old_log_probabilities[indices] -
                    log_probabilities).mean()
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
    })
    return result


@torch.no_grad()
def evaluate(model: MaskedCandidateActorCritic, seeds,
             execution_mode: str, profile: str, device: torch.device,
             use_rule: bool = False,
             observation_variant: str = "FULL_CONTEXT_V2",
             handover_sampling: str = "SEQUENTIAL",
             allowed_transport_modes: tuple[str, ...] | None = None):
    model.eval()
    interval, arrival_schedule, _ = profile_environment(profile)
    episodes = []
    total_modes = Counter()
    total_modes_by_phase = Counter()
    total_exposed_modes = set()
    total_mode_by_cost_reference = Counter()
    total_mode_by_floor_relation = Counter()
    total_cost_reference_agreements = 0
    total_cost_reference_decisions = 0
    for seed in seeds:
        env = WarehouseDispatchGymEnv(
            seed=seed, execution_mode=execution_mode,
            arrival_interval=interval,
            arrival_schedule=arrival_schedule,
            observation_variant=observation_variant,
            handover_sampling=handover_sampling,
            allowed_transport_modes=allowed_transport_modes)
        observation, info = env.reset(seed=seed)
        done = False
        reward_total = 0.0
        illegal = 0
        peak_active = 0
        modes = Counter()
        modes_by_phase = Counter()
        exposed_modes = set()
        mode_by_cost_reference = Counter()
        mode_by_floor_relation = Counter()
        cost_reference_agreements = 0
        cost_reference_decisions = 0
        while not done:
            exposed_modes.update(
                candidate.transport_mode
                for candidate, legal in zip(
                    env.decision.action_ids, info["action_mask"])
                if legal and isinstance(candidate, Assignment))
            if use_rule:
                action = env.rule_action()
            else:
                distribution, _ = _distribution(
                    model, observation, info["action_mask"], device)
                action = int(torch.argmax(distribution.logits).item())
            illegal += int(not info["action_mask"][action])
            decoded = env.decision.action_ids[action]
            if isinstance(decoded, Assignment):
                modes[decoded.transport_mode] += 1
                phase = env.task_types.get(decoded.task_id, "BALANCED").split(":", 1)[0]
                modes_by_phase[(phase, decoded.transport_mode)] += 1
                peak_active = max(peak_active, 1)
                legal_same_task = [
                    candidate for candidate, legal in zip(
                        env.decision.action_ids, info["action_mask"])
                    if legal and isinstance(candidate, Assignment) and
                    candidate.task_id == decoded.task_id]
                if legal_same_task:
                    cost_reference = min(
                        legal_same_task,
                        key=lambda candidate: candidate.estimated_cost)
                    mode_by_cost_reference[
                        (cost_reference.transport_mode,
                         decoded.transport_mode)] += 1
                    cost_reference_agreements += int(
                        cost_reference.transport_mode ==
                        decoded.transport_mode)
                    cost_reference_decisions += 1
                task = env.dispatch._task(decoded.task_id).task
                floor_relation = (
                    "CROSS_FLOOR" if
                    task.source.floor != task.target.floor else
                    "SAME_FLOOR")
                mode_by_floor_relation[
                    (floor_relation, decoded.transport_mode)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            peak_active = max(peak_active, info["active_task_count"])
            done = terminated or truncated
        dispatch = env.dispatch
        scheduled_tasks = len(env.task_types)
        unresolved_tasks = max(
            scheduled_tasks - dispatch.completed - dispatch.failed, 0)
        terminal_adjusted_reward = (
            reward_total - unresolved_tasks *
            dispatch.reward_config.severe_failure_penalty)
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
        total_cost_reference_agreements += cost_reference_agreements
        total_cost_reference_decisions += cost_reference_decisions
        episodes.append({
            "seed": seed,
            "actual_execution_mode": env.current_execution_mode,
            "completed": dispatch.completed,
            "failed": dispatch.failed,
            "scheduled_tasks": scheduled_tasks,
            "unresolved_tasks": unresolved_tasks,
            "reward": reward_total,
            "terminal_adjusted_reward": terminal_adjusted_reward,
            "simulated_time": dispatch.now,
            "illegal_actions": illegal,
            "peak_active_tasks": peak_active,
            "transport_modes": dict(modes),
            "transport_modes_by_phase": _nested_mode_matrix(modes_by_phase),
            "arrival_schedule": arrival_schedule,
            "arrival_schedule_metadata": getattr(
                dispatch, "arrival_schedule_metadata", {}),
            "exposed_transport_modes": sorted(exposed_modes),
            "selected_mode_by_cost_reference": _nested_mode_matrix(
                mode_by_cost_reference),
            "selected_mode_by_floor_relation": _nested_mode_matrix(
                mode_by_floor_relation),
            "cost_reference_mode_agreement": (
                cost_reference_agreements /
                max(cost_reference_decisions, 1)),
            "resource_leak": leak,
        })
    total_time = sum(item["simulated_time"] for item in episodes)
    completed_count = sum(item["completed"] for item in episodes)
    failed_count = sum(item["failed"] for item in episodes)
    scheduled_count = sum(item["scheduled_tasks"] for item in episodes)
    unresolved_count = sum(item["unresolved_tasks"] for item in episodes)
    task_count = completed_count + failed_count
    resolved_throughput = task_count / max(total_time, 1e-9) * 3600
    successful_throughput = (
        completed_count / max(total_time, 1e-9) * 3600)
    return {
        "episode_count": len(episodes),
        "task_count": task_count,
        "scheduled_task_count": scheduled_count,
        "completed": completed_count,
        "failed": failed_count,
        "unresolved": unresolved_count,
        "effective_failed": failed_count + unresolved_count,
        "resolution_rate": task_count / max(scheduled_count, 1),
        "success_rate": completed_count / max(scheduled_count, 1),
        "reward_units": "raw_environment_episode_sum",
        "mean_reward": float(np.mean([item["reward"] for item in episodes])),
        "mean_terminal_adjusted_reward": float(np.mean([
            item["terminal_adjusted_reward"] for item in episodes])),
        "mean_simulated_time": float(np.mean(
            [item["simulated_time"] for item in episodes])),
        # Backward-compatible alias.  This metric counts both completed and
        # failed tasks, so it is a resolution rate rather than productive
        # delivery throughput.
        "throughput_tasks_per_hour": resolved_throughput,
        "resolved_throughput_tasks_per_hour": resolved_throughput,
        "successful_throughput_tasks_per_hour": successful_throughput,
        "handover_sampling": handover_sampling.upper(),
        "allowed_transport_modes": (
            sorted(mode.upper() for mode in allowed_transport_modes)
            if allowed_transport_modes is not None else None),
        "illegal_action_count": sum(
            item["illegal_actions"] for item in episodes),
        "maximum_active_tasks": max(
            item["peak_active_tasks"] for item in episodes),
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
            total_cost_reference_agreements /
            max(total_cost_reference_decisions, 1)),
        "cost_reference_decision_count": total_cost_reference_decisions,
        "episodes": episodes,
    }


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["model_metadata"]
    policy_variant = metadata.get("policy_variant", "CONTEXT_INTERACTION")
    expected = MaskedCandidateActorCritic(
        action_width=metadata["action_width"],
        policy_variant=policy_variant).metadata()
    if (metadata.get("architecture") != expected["architecture"] or
            metadata.get("state_width") != expected["state_width"] or
            metadata.get("action_width") not in {28, 30}):
        raise ValueError(
            "checkpoint is incompatible with contextual policy V2; "
            "start a new training run")
    model = MaskedCandidateActorCritic(
        state_width=metadata["state_width"],
        action_width=metadata["action_width"],
        hidden_width=metadata["hidden_width"],
        policy_variant=policy_variant).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
