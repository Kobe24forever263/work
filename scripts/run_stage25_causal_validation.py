#!/usr/bin/env python3
"""Fresh-seed safety validation for a Stage 25 causal PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np  # Bind the Apple runtime before torch.
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

import warehouse_core.stage14_ppo as ppo_module
from warehouse_core.stage14_ppo import evaluate, load_checkpoint
from warehouse_core.stage25_causal_observation import (
    CAUSAL_RELEASED_V4, CausalReleasedWarehouseDispatchGymEnv)


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "episodes"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, default=72530000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output or checkpoint.with_suffix(".fresh_validation.json")
    output = output.expanduser().resolve()

    device = torch.device("cpu")
    torch.set_num_threads(4)
    model, payload = load_checkpoint(checkpoint, device)
    if payload.get("observation_variant") != CAUSAL_RELEASED_V4:
        raise ValueError("checkpoint is not CAUSAL_RELEASED_V4")
    encoder = payload.get("encoder_metadata", {})
    if encoder.get("visibility_contract") != "RELEASED_ONLY":
        raise ValueError("checkpoint lacks RELEASED_ONLY provenance")

    ppo_module.WarehouseDispatchGymEnv = \
        CausalReleasedWarehouseDispatchGymEnv
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    common = dict(
        execution_mode="CONCURRENT", profile="MIXED_CURRICULUM",
        device=device, observation_variant=CAUSAL_RELEASED_V4,
        handover_sampling="TASK_KEYED",
        reward_contract="CONTINUOUS_TIME_V3")
    policy = evaluate(model, seeds, **common)
    rule = evaluate(model, seeds, use_rule=True, **common)
    expected_modes = {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
    context_matrix = policy["selected_mode_by_cost_reference"]
    context_match_rates = {
        reference_mode: (
            selected_counts.get(reference_mode, 0) /
            max(sum(selected_counts.values()), 1))
        for reference_mode, selected_counts in context_matrix.items()
    }
    contextual_switching_gate = bool(
        set(context_matrix) == expected_modes and
        set(policy["transport_modes"]) == expected_modes and
        all(context_match_rates.get(mode, 0.0) >= .20
            for mode in expected_modes))
    assertions = {
        "fresh_seeds_are_disjoint_from_training": (
            args.seed_start > int(payload.get("base_seed", -1)) +
            int(payload.get("episodes_trained", 0))),
        "policy_resolved_all_tasks": policy["unresolved"] == 0,
        "policy_actions_all_legal": policy["illegal_action_count"] == 0,
        "policy_has_no_resource_leak": policy["resource_leak_count"] == 0,
        "rule_resolved_all_tasks": rule["unresolved"] == 0,
        "rule_actions_all_legal": rule["illegal_action_count"] == 0,
        "rule_has_no_resource_leak": rule["resource_leak_count"] == 0,
        "all_three_modes_were_exposed": set(
            policy["exposed_transport_modes"]) == expected_modes,
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    result = {
        "stage": 25,
        "gate": "CAUSAL_RELEASED_V4_FRESH_SEED_SAFETY_VALIDATION",
        "claim_boundary": (
            "Fresh-seed safety/interface validation only. A smoke-trained "
            "checkpoint is not used for a performance claim."),
        "checkpoint": str(checkpoint),
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "episode_count": args.episodes,
        "observation_variant": CAUSAL_RELEASED_V4,
        "visibility_contract": "RELEASED_ONLY",
        "handover_sampling": "TASK_KEYED",
        "policy": compact(policy),
        "rule_baseline": compact(rule),
        "policy_minus_rule": {
            "success_rate": policy["success_rate"] - rule["success_rate"],
            "mean_reward": policy["mean_reward"] - rule["mean_reward"],
            "successful_throughput_tasks_per_hour": (
                policy["successful_throughput_tasks_per_hour"] -
                rule["successful_throughput_tasks_per_hour"]),
        },
        "contextual_mode_acceptance": {
            "selected_mode_by_cost_reference": context_matrix,
            "diagonal_match_rates": context_match_rates,
            "minimum_diagonal_rate": .20,
            "all_three_modes_selected": (
                set(policy["transport_modes"]) == expected_modes),
            "passed": contextual_switching_gate,
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
