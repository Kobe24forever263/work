#!/usr/bin/env python3
"""Audit duration-aware return and GAE against neutral event insertion."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from math import isclose, log
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import (
    RolloutStep, RunningReturnNormalizer, compute_smdp_gae)


OUTPUT_ROOT = ROOT / "results" / "stage23_event_partition"
OUTPUT = OUTPUT_ROOT / "stage23_event_partition_audit.json"
PPO_SOURCE = (ROOT / "src" / "warehouse_core" / "warehouse_core" /
              "stage14_ppo.py")
ENV_SOURCE = (ROOT / "src" / "warehouse_core" / "warehouse_core" /
              "concurrent_dispatch.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discount(duration: float, gamma0: float = .99,
             tau_s: float = 10.0) -> float:
    return gamma0 ** (duration / tau_s)


def dummy_step(reward: float, factor: float, end: bool) -> RolloutStep:
    return RolloutStep(
        state=np.zeros(1, dtype=np.float32),
        action_features=np.zeros((1, 1), dtype=np.float32),
        action_mask=np.ones(1, dtype=np.bool_), action=0,
        old_log_probability=0.0, old_value=0.0,
        reward=reward, discount=factor, episode_end=end)


def main() -> int:
    gamma0 = .99
    lambda0 = .95
    tau_s = 10.0
    first_duration = 4.0
    second_duration = 7.0
    total_duration = first_duration + second_duration
    gamma_first = discount(first_duration, gamma0, tau_s)
    gamma_second = discount(second_duration, gamma0, tau_s)
    gamma_total = discount(total_duration, gamma0, tau_s)

    discount_semigroup = {
        "unsplit": gamma_total,
        "split_product": gamma_first * gamma_second,
        "absolute_difference": abs(
            gamma_total - gamma_first * gamma_second),
        "invariant": isclose(
            gamma_total, gamma_first * gamma_second,
            rel_tol=0.0, abs_tol=1e-14),
    }

    current_unsplit_trace = gamma_total * lambda0
    current_split_trace = (
        gamma_first * lambda0 * gamma_second * lambda0)
    duration_lambda_first = lambda0 ** (first_duration / tau_s)
    duration_lambda_second = lambda0 ** (second_duration / tau_s)
    duration_lambda_total = lambda0 ** (total_duration / tau_s)
    corrected_unsplit_trace = gamma_total * duration_lambda_total
    corrected_split_trace = (
        gamma_first * duration_lambda_first *
        gamma_second * duration_lambda_second)
    trace_probe = {
        "current_formula": "gamma(delta_t) * lambda_per_event",
        "current_unsplit_trace_coefficient": current_unsplit_trace,
        "current_split_trace_coefficient": current_split_trace,
        "current_absolute_difference": abs(
            current_unsplit_trace - current_split_trace),
        "current_event_partition_invariant": isclose(
            current_unsplit_trace, current_split_trace,
            rel_tol=0.0, abs_tol=1e-14),
        "duration_scaled_formula": (
            "gamma(delta_t) * lambda0**(delta_t/tau_s)"),
        "duration_scaled_unsplit_trace_coefficient": corrected_unsplit_trace,
        "duration_scaled_split_trace_coefficient": corrected_split_trace,
        "duration_scaled_absolute_difference": abs(
            corrected_unsplit_trace - corrected_split_trace),
        "duration_scaled_event_partition_invariant": isclose(
            corrected_unsplit_trace, corrected_split_trace,
            rel_tol=0.0, abs_tol=1e-14),
    }

    rate = -3.0 / 100.0
    current_unsplit_reward = rate * total_duration
    current_split_return = (
        rate * first_duration + gamma_first * rate * second_duration)
    beta = -log(gamma0) / tau_s
    corrected_unsplit_reward = rate * (1.0 - np.exp(
        -beta * total_duration)) / beta
    corrected_first_reward = rate * (
        1.0 - np.exp(-beta * first_duration)) / beta
    corrected_second_reward = rate * (
        1.0 - np.exp(-beta * second_duration)) / beta
    corrected_split_return = (
        corrected_first_reward + gamma_first * corrected_second_reward)
    interval_reward_probe = {
        "physical_cost_rate": rate,
        "current_environment_formula": "rate * elapsed",
        "current_unsplit_reward": current_unsplit_reward,
        "current_split_discounted_return": current_split_return,
        "current_absolute_difference": abs(
            current_unsplit_reward - current_split_return),
        "current_event_partition_invariant": isclose(
            current_unsplit_reward, current_split_return,
            rel_tol=0.0, abs_tol=1e-14),
        "continuous_discount_integral_formula": (
            "rate * (1-exp(-beta*elapsed)) / beta"),
        "corrected_unsplit_reward": corrected_unsplit_reward,
        "corrected_split_discounted_return": corrected_split_return,
        "corrected_absolute_difference": abs(
            corrected_unsplit_reward - corrected_split_return),
        "corrected_event_partition_invariant": isclose(
            corrected_unsplit_reward, corrected_split_return,
            rel_tol=0.0, abs_tol=1e-14),
    }

    normalizer = RunningReturnNormalizer()
    steps = [dummy_step(1.0, .9, False), dummy_step(2.0, .9, True)]
    normalizer.normalize_rollout(steps, update=True)
    expected_return_to_go = np.asarray([2.8, 2.0])
    legacy_forward_samples = np.asarray([1.0, 2.9])
    normalizer_probe = {
        "two_step_rewards": [1.0, 2.0],
        "discounts": [.9, .9],
        "expected_reverse_return_to_go_samples": (
            expected_return_to_go.tolist()),
        "legacy_forward_accumulator_samples": (
            legacy_forward_samples.tolist()),
        "observed_running_mean": normalizer.mean,
        "expected_reverse_return_mean": float(expected_return_to_go.mean()),
        "implementation_matches_reverse_return_to_go": isclose(
            normalizer.mean, float(expected_return_to_go.mean()),
            rel_tol=0.0, abs_tol=1e-3),
        "implementation_matches_legacy_forward_accumulator": isclose(
            normalizer.mean, float(legacy_forward_samples.mean()),
            rel_tol=0.0, abs_tol=1e-3),
        "normalizer_contract": normalizer.state_dict()["source"],
    }

    rewards = np.asarray([0.0, 1.0], dtype=np.float32)
    values = np.zeros(2, dtype=np.float32)
    legacy_advantages, _ = compute_smdp_gae(
        rewards, values, [gamma_first, gamma_second],
        [False, True], lambda0)
    corrected_advantages, _ = compute_smdp_gae(
        rewards, values, [gamma_first, gamma_second],
        [False, True], lambda0, durations=[first_duration, second_duration],
        trace_mode="DURATION_SCALED", trace_tau_s=tau_s)
    legacy_expected_a0 = gamma_first * lambda0
    corrected_expected_a0 = (
        gamma_first * lambda0 ** (first_duration / tau_s))
    implementation_probe = {
        "legacy_per_event_two_step_a0": float(legacy_advantages[0]),
        "legacy_expected_a0": legacy_expected_a0,
        "legacy_mode_reproduced": isclose(
            float(legacy_advantages[0]), legacy_expected_a0,
            rel_tol=0.0, abs_tol=1e-7),
        "corrected_duration_scaled_two_step_a0": float(
            corrected_advantages[0]),
        "corrected_expected_a0": corrected_expected_a0,
        "corrected_mode_matches_duration_formula": isclose(
            float(corrected_advantages[0]), corrected_expected_a0,
            rel_tol=0.0, abs_tol=1e-7),
    }

    execution_checks = {
        "duration_discount_has_semigroup_property": (
            discount_semigroup["invariant"]),
        "legacy_gae_mode_remains_explicitly_reproducible": (
            implementation_probe["legacy_mode_reproduced"]),
        "corrected_gae_mode_matches_duration_formula": (
            implementation_probe[
                "corrected_mode_matches_duration_formula"]),
        "duration_scaled_lambda_restores_trace_semigroup": (
            trace_probe["duration_scaled_event_partition_invariant"]),
        "continuous_discount_integral_restores_reward_partition": (
            interval_reward_probe["corrected_event_partition_invariant"]),
        "normalizer_uses_reverse_return_to_go": (
            normalizer_probe["implementation_matches_reverse_return_to_go"]),
    }
    scientific_failures = []
    if not trace_probe["duration_scaled_event_partition_invariant"]:
        scientific_failures.append("duration_scaled_lambda_is_not_invariant")
    if not interval_reward_probe["corrected_event_partition_invariant"]:
        scientific_failures.append("continuous_interval_reward_is_not_invariant")
    if not normalizer_probe["implementation_matches_reverse_return_to_go"]:
        scientific_failures.append("reward_normalizer_is_not_return_to_go")
    if not implementation_probe["corrected_mode_matches_duration_formula"]:
        scientific_failures.append("corrected_gae_implementation_mismatch")

    report = {
        "schema_version": "warehouse_stage23_event_partition_audit_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Legacy Stage20/22 versus corrected continuous-time V3 contract",
        "claim_boundary": (
            "Deterministic algebraic/unit audit. It does not evaluate learned "
            "policy performance."),
        "source_sha256": {
            str(PPO_SOURCE): sha256(PPO_SOURCE),
            str(ENV_SOURCE): sha256(ENV_SOURCE),
        },
        "parameters": {
            "gamma0": gamma0, "lambda0": lambda0, "tau_s": tau_s,
            "split_durations_s": [first_duration, second_duration],
        },
        "duration_discount_semigroup_probe": discount_semigroup,
        "gae_trace_partition_probe": trace_probe,
        "interval_reward_partition_probe": interval_reward_probe,
        "reward_normalizer_order_probe": normalizer_probe,
        "implementation_formula_probe": implementation_probe,
        "audit_execution_gate": {
            "passed": all(execution_checks.values()),
            "checks": execution_checks,
        },
        "scientific_assessment": {
            "neutral_event_partition_invariance_supported": (
                not scientific_failures),
            "status": (
                "PASS_CORRECTED_CONTINUOUS_TIME_V3" if not scientific_failures else
                "FAIL_EVENT_PARTITION_DEPENDENCE_DETECTED"),
            "failures": scientific_failures,
            "legacy_findings_retained": [
                "PER_EVENT_LEGACY lambda is event-partition dependent.",
                "REWARD_V2_LEGACY interval costs are event-partition dependent.",
                "Stage20/22 normalizer checkpoints contain forward accumulator statistics.",
            ],
            "existing_stage20_stage22_weights_affected": True,
            "formal_locked_test_blocked_until_return_contract_is_resolved":
                bool(scientific_failures),
            "recommended_resolution": [
                "Use lambda(delta_t)=lambda0**(delta_t/tau_s).",
                "Define each SMDP interval reward as the discounted integral "
                "of time-rate costs plus correctly timed event impulses.",
                "Compute reward-normalizer return-to-go samples in reverse "
                "within each episode.",
                "Add neutral-event insertion regression tests before retraining.",
            ],
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "audit_execution_passed": report["audit_execution_gate"]["passed"],
        "scientific_status": report["scientific_assessment"]["status"],
        "failures": scientific_failures,
        "formal_locked_test_blocked": report["scientific_assessment"][
            "formal_locked_test_blocked_until_return_contract_is_resolved"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["audit_execution_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
