#!/usr/bin/env python3
"""Verify the four Stage 22 ablations change only their declared construct."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_policy import MaskedCandidateActorCritic
from warehouse_core.stage14_ppo import collect_rollouts


CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage22_algorithm_ablation.yaml")
RUNNER = ROOT / "scripts" / "run_stage22_algorithm_ablation.py"
OUTPUT = (ROOT / "results" / "stage22_algorithm_ablation" /
          "interface_gate.summary.json")


def plan(condition: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), condition, "1", "--short-gate"],
        cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    plans = {name: plan(name) for name in protocol["conditions"]}
    checks = {
        "four_conditions_registered": set(plans) == {
            "fixed_gamma", "no_context_interaction", "no_warm_start",
            "no_relay"},
        "all_use_mixed_curriculum_80_task_episodes": all(
            row["tasks_per_episode"] == 80 and
            row["condition_config"]["observation_variant"] ==
            "FULL_CONTEXT_V2" for row in plans.values()),
        "fixed_gamma_changes_only_discount": (
            plans["fixed_gamma"]["condition_config"]["discount_mode"] ==
            "FIXED_PER_DECISION" and
            plans["fixed_gamma"]["condition_config"]["policy_variant"] ==
            "CONTEXT_INTERACTION"),
        "no_context_interaction_uses_flat_scorer": (
            plans["no_context_interaction"]["condition_config"][
                "policy_variant"] == "FLAT_MASKED_PPO"),
        "no_warm_start_has_no_teacher_command_or_weight": (
            not plans["no_warm_start"]["warm_start_enabled"] and
            plans["no_warm_start"]["warm_start_command"] is None and
            plans["no_warm_start"]["warm_start_weight"] is None and
            "--warm-start" not in plans["no_warm_start"]["ppo_command"]),
        "no_relay_declares_exactly_two_modes": (
            plans["no_relay"]["condition_config"][
                "allowed_transport_modes"] == ["SINGLE_CAR", "SINGLE_DOG"] and
            "CAR_DOG_CAR" not in plans["no_relay"]["condition_config"][
                "allowed_transport_modes"]),
        "locked_test_seeds_are_fresh_and_disjoint": (
            protocol["locked_test"]["seed_start"] == 67100000 and
            protocol["locked_test"]["seed_count"] == 100 and
            protocol["locked_test"]["seed_start"] not in set(
                protocol["training"]["independent_run_base_seeds"])),
    }

    seed = int(protocol["smoke"]["independent_run_base_seeds"][0]) + 100
    device = torch.device("cpu")
    torch.manual_seed(seed)
    variable = MaskedCandidateActorCritic(hidden_width=16)
    variable_steps, _ = collect_rollouts(
        variable, [seed], "CONCURRENT", "MIXED_CURRICULUM", device,
        discount_mode="VARIABLE_SMDP")
    torch.manual_seed(seed)
    fixed = MaskedCandidateActorCritic(hidden_width=16)
    fixed_steps, _ = collect_rollouts(
        fixed, [seed], "CONCURRENT", "MIXED_CURRICULUM", device,
        discount_mode="FIXED_PER_DECISION")
    checks.update({
        "variable_discount_varies": (
            min(step.discount for step in variable_steps) <
            max(step.discount for step in variable_steps)),
        "fixed_discount_is_exactly_0_99": all(
            abs(step.discount - 0.99) < 1e-12 for step in fixed_steps),
        "discount_ablation_preserves_rollout_length": (
            len(variable_steps) == len(fixed_steps)),
    })

    contextual = MaskedCandidateActorCritic(hidden_width=16)
    flat = MaskedCandidateActorCritic(
        hidden_width=16, policy_variant="FLAT_MASKED_PPO")
    checks.update({
        "flat_scorer_removes_interaction_parameters": (
            sum(parameter.numel() for parameter in flat.parameters()) <
            sum(parameter.numel() for parameter in contextual.parameters())),
        "both_scorers_keep_state_and_candidate_widths": (
            contextual.state_width == flat.state_width == 284 and
            contextual.action_width == flat.action_width == 30),
    })

    torch.manual_seed(seed)
    relay_free = MaskedCandidateActorCritic(hidden_width=16)
    restricted_steps, restricted_episodes = collect_rollouts(
        relay_free, [seed], "CONCURRENT", "MIXED_CURRICULUM", device,
        allowed_transport_modes=("SINGLE_CAR", "SINGLE_DOG"))
    exposed = set(restricted_episodes[0]["exposed_transport_modes"])
    selected = set(restricted_episodes[0]["transport_modes"])
    checks.update({
        "relay_free_rollout_exposes_only_declared_modes": (
            exposed == {"SINGLE_CAR", "SINGLE_DOG"}),
        "relay_free_rollout_never_selects_relay": (
            "CAR_DOG_CAR" not in selected),
        "relay_free_rollout_still_resolves_all_tasks": (
            restricted_episodes[0]["completed"] +
            restricted_episodes[0]["failed"] == 80),
        "relay_free_rollout_has_no_resource_leak": (
            not restricted_episodes[0]["resource_leak"]),
        "relay_free_rollout_discounts_are_valid": all(
            0 < step.discount <= 1 for step in restricted_steps),
    })

    report = {
        "stage": 22,
        "gate": "ALGORITHM_ABLATION_SEMANTIC_INTERFACE_V1",
        "claim_boundary": "Interface semantics only; no performance claim.",
        "protocol": protocol["protocol"]["version"],
        "plans": plans,
        "diagnostics": {
            "variable_discount_range": [
                min(step.discount for step in variable_steps),
                max(step.discount for step in variable_steps)],
            "fixed_discount_values": sorted({
                step.discount for step in fixed_steps}),
            "contextual_parameter_count": sum(
                parameter.numel() for parameter in contextual.parameters()),
            "flat_parameter_count": sum(
                parameter.numel() for parameter in flat.parameters()),
            "relay_free_exposed_modes": sorted(exposed),
            "relay_free_selected_modes": sorted(selected),
        },
        "assertions": checks,
        "passed": all(checks.values()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "assertion_count": len(checks),
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

