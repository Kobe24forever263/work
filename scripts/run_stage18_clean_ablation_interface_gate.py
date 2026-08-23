#!/usr/bin/env python3
"""Verify semantically isolated Stage 18 clean context ablations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage12_encoding import Stage12Encoder
from warehouse_core.stage14_policy import MaskedCandidateActorCritic
from warehouse_core.stage14_ppo import collect_rollouts, evaluate
from warehouse_core.stage14_training import WarehouseDispatchGymEnv


VARIANTS = (
    "FULL_CONTEXT_V2", "NO_POSITION_ONLY", "NO_ETA_COST_ONLY",
    "NO_HISTORY_ONLY", "NO_EXPLICIT_QUEUE_RESOURCE",
    "NO_EXPLICIT_HANDOVER_RISK")


def encoded(seed: int, variant: str):
    env = WarehouseDispatchGymEnv(
        seed=seed, execution_mode="CONCURRENT",
        observation_variant=variant)
    observation, info = env.reset(seed=seed)
    return env, observation, info


def changed_state_indices(full: np.ndarray, variant: np.ndarray) -> set[int]:
    return set(np.flatnonzero(~np.isclose(full, variant)).tolist())


def changed_action_columns(full: np.ndarray,
                           variant: np.ndarray) -> set[int]:
    changed = ~np.isclose(full, variant)
    return set(np.flatnonzero(changed.any(axis=0)).tolist())


def main() -> int:
    seed = 41000017
    full_env, full_observation, full_info = encoded(seed, "FULL_CONTEXT_V2")
    full_decision = full_env.decision
    checks = {
        "full_state_shape_is_284": (
            full_observation["state"].shape == (284,)),
        "full_action_shape_is_1536x30": (
            full_observation["action_features"].shape == (1536, 30)),
        "explicit_handover_fields_nonzero_for_some_candidates": bool(
            np.any(full_observation["action_features"][:, 28:] > 0)),
    }
    diagnostics = {}
    expected_state = {
        "NO_POSITION_ONLY": set(
            Stage12Encoder.position_only_state_indices()),
        "NO_ETA_COST_ONLY": set(),
        "NO_HISTORY_ONLY": set(
            Stage12Encoder.history_only_state_indices()),
        "NO_EXPLICIT_QUEUE_RESOURCE": set(
            Stage12Encoder.queue_resource_state_indices()),
        "NO_EXPLICIT_HANDOVER_RISK": set(),
    }
    expected_action = {
        "NO_POSITION_ONLY": set(
            Stage12Encoder.position_only_action_indices()),
        "NO_ETA_COST_ONLY": set(
            Stage12Encoder.eta_cost_only_action_indices()),
        "NO_HISTORY_ONLY": set(),
        "NO_EXPLICIT_QUEUE_RESOURCE": set(),
        "NO_EXPLICIT_HANDOVER_RISK": {28, 29},
    }
    rule_reference = evaluate(
        MaskedCandidateActorCritic(hidden_width=16), [seed],
        "CONCURRENT", "MEDIUM", torch.device("cpu"), use_rule=True,
        observation_variant="FULL_CONTEXT_V2")
    for variant in VARIANTS[1:]:
        env, observation, info = encoded(seed, variant)
        state_changed = changed_state_indices(
            full_observation["state"], observation["state"])
        action_changed = changed_action_columns(
            full_observation["action_features"],
            observation["action_features"])
        diagnostics[variant] = {
            "changed_state_indices": sorted(state_changed),
            "changed_action_columns": sorted(action_changed),
            "encoder_metadata": env.encoder.metadata(),
        }
        checks[f"{variant}.action_ids_unchanged"] = (
            env.decision.action_ids == full_decision.action_ids)
        checks[f"{variant}.legal_action_mask_unchanged"] = np.array_equal(
            info["action_mask"], full_info["action_mask"])
        checks[f"{variant}.only_declared_state_fields_changed"] = (
            state_changed <= expected_state[variant])
        checks[f"{variant}.only_declared_action_fields_changed"] = (
            action_changed <= expected_action[variant])
        if expected_state[variant]:
            checks[f"{variant}.declared_state_fields_are_zero"] = bool(
                np.all(observation["state"][list(
                    expected_state[variant])] == 0))
        if expected_action[variant]:
            checks[f"{variant}.declared_action_fields_are_zero"] = bool(
                np.all(observation["action_features"][:, list(
                    expected_action[variant])] == 0))
        rule_variant = evaluate(
            MaskedCandidateActorCritic(hidden_width=16), [seed],
            "CONCURRENT", "MEDIUM", torch.device("cpu"), use_rule=True,
            observation_variant=variant)
        checks[f"{variant}.rule_dynamics_unchanged"] = (
            rule_variant["episodes"] == rule_reference["episodes"])

    torch.manual_seed(seed)
    variable_model = MaskedCandidateActorCritic(hidden_width=16)
    variable_steps, _ = collect_rollouts(
        variable_model, [seed], "CONCURRENT", "MEDIUM",
        torch.device("cpu"), discount_mode="VARIABLE_SMDP")
    torch.manual_seed(seed)
    fixed_model = MaskedCandidateActorCritic(hidden_width=16)
    fixed_steps, _ = collect_rollouts(
        fixed_model, [seed], "CONCURRENT", "MEDIUM",
        torch.device("cpu"), discount_mode="FIXED_PER_DECISION")
    variable_discounts = [step.discount for step in variable_steps]
    fixed_discounts = [step.discount for step in fixed_steps]
    checks["variable_smdp_discount_varies"] = (
        min(variable_discounts) < max(variable_discounts))
    checks["fixed_discount_is_exactly_0.99"] = all(
        abs(value - .99) < 1e-12 for value in fixed_discounts)
    checks["discount_ablation_keeps_rollout_length"] = (
        len(variable_steps) == len(fixed_steps))

    flat = MaskedCandidateActorCritic(
        hidden_width=16, policy_variant="FLAT_MASKED_PPO")
    state = torch.as_tensor(full_observation["state"])
    features = torch.as_tensor(full_observation["action_features"])
    mask = torch.as_tensor(full_info["action_mask"])
    logits, value = flat(state, features, mask)
    checks["flat_policy_has_expected_architecture"] = (
        flat.metadata()["architecture"] ==
        "flat_masked_candidate_actor_critic_v1")
    checks["flat_policy_forward_is_finite"] = bool(
        torch.isfinite(logits[mask]).all() and torch.isfinite(value))
    checks["flat_policy_preserves_action_mask"] = bool(
        torch.all(logits[~mask] == torch.finfo(logits.dtype).min))

    checks["queue_construct_is_explicit_telemetry_only"] = (
        Stage12Encoder("NO_EXPLICIT_QUEUE_RESOURCE").metadata()[
            "construct_boundary"]["NO_EXPLICIT_QUEUE_RESOURCE"] ==
        "removes explicit telemetry; safety action mask remains")
    checks["handover_construct_is_explicit_risk_only"] = (
        Stage12Encoder("NO_EXPLICIT_HANDOVER_RISK").metadata()[
            "construct_boundary"]["NO_EXPLICIT_HANDOVER_RISK"] ==
        "removes risk fields; transport-mode semantics remain")

    output = (WORK_ROOT / "results" / "stage18_clean_ablation" /
              "interface_gate.summary.json")
    report = {
        "stage": 18,
        "gate": "SEMANTICALLY_ISOLATED_ABLATION_INTERFACE",
        "claim_boundary": (
            "Interface gate only; no ablation performance claim."),
        "seed": seed,
        "variants": list(VARIANTS),
        "diagnostics": diagnostics,
        "discount_diagnostics": {
            "variable_min": min(variable_discounts),
            "variable_max": max(variable_discounts),
            "fixed_unique": sorted(set(fixed_discounts)),
        },
        "assertions": checks,
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "assertion_count": len(checks),
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
