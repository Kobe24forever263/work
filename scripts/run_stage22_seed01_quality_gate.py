#!/usr/bin/env python3
"""Audit the four formal Stage 22 seed-01 runs before cohort expansion."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np  # Bind the macOS OpenMP runtime before torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage22_algorithm_ablation.yaml")
RESULT_ROOT = ROOT / "results" / "stage22_algorithm_ablation" / "long"
REFERENCE = (ROOT / "results" / "stage20_mixed_curriculum" / "long" /
             "seed_01" / "stage20_ppo.summary.json")
OUTPUT = (ROOT / "results" / "stage22_algorithm_ablation" /
          "stage22_seed01_quality_gate.summary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def checkpoint_checks(payload: dict, *, condition: dict, run_seed: int,
                      ppo_seed: int, validation_seed: int,
                      model_metadata: dict, final: bool) -> dict:
    update = int(payload.get("update", -1))
    episodes = int(payload.get("episodes_trained", -1))
    modes = sorted(condition["allowed_transport_modes"])
    expected_stored_modes = modes if len(modes) < 3 else None
    checks = {
        "execution_mode": payload.get("execution_mode") == "CONCURRENT",
        "arrival_profile": (
            payload.get("arrival_profile") == "MIXED_CURRICULUM"),
        "observation_variant": (
            payload.get("observation_variant") ==
            condition["observation_variant"]),
        "policy_variant": (
            payload.get("policy_variant") == condition["policy_variant"]),
        "discount_mode": (
            payload.get("discount_mode") == condition["discount_mode"]),
        "allowed_transport_modes": (
            payload.get("allowed_transport_modes") == expected_stored_modes),
        "experiment_run_seed": (
            payload.get("experiment_run_seed") == run_seed),
        "ppo_seed": payload.get("base_seed") == ppo_seed,
        "validation_seed": (
            payload.get("validation_seed_start") == validation_seed),
        "model_metadata": payload.get("model_metadata") == model_metadata,
        "update_range": 0 < update <= 2000,
        "episode_count_matches_update": episodes == update * 4,
        "reward_normalizer_present": bool(payload.get("reward_normalizer")),
        "numpy_rng_present": payload.get("numpy_random_state") is not None,
        "torch_rng_present": payload.get("torch_random_state") is not None,
        "ppo_rng_present": payload.get("ppo_generator_state") is not None,
        "model_state_present": bool(payload.get("model_state_dict")),
    }
    if final:
        checks["final_update"] = update == 2000
        checks["final_episodes"] = episodes == 8000
    else:
        checks["best_checkpoint_on_validation_boundary"] = update % 50 == 0
    return checks


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    conditions = protocol["conditions"]
    training = protocol["training"]
    validation = protocol["validation"]
    run_seed = int(training["independent_run_base_seeds"][0])
    ppo_seed = run_seed + int(training["ppo_seed_offset"])
    validation_start = int(validation["seed_start"])
    expected_validation_updates = list(range(50, 2001, 50))

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_schedule_hash = canonical_hash(
        reference["evaluation"]["arrival_schedule_metadata"])
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    schedule_hashes: set[str] = set()
    final_hashes: set[str] = set()
    best_hashes: set[str] = set()
    ppo_configs: list[dict] = []

    for name, condition in conditions.items():
        directory = RESULT_ROOT / name / "seed_01"
        paths = {
            "run_plan": directory / "run_plan.json",
            "summary": directory / "stage22_ppo.summary.json",
            "history": directory / "stage22_ppo.history.json",
            "final_weight": directory / "stage22_ppo.pt",
            "best_weight": directory / "stage22_ppo.best.pt",
        }
        if condition["warm_start"]:
            paths.update({
                "warm_start_weight": directory / "warm_start.pt",
                "warm_start_summary": directory / "warm_start.summary.json",
            })
        missing = [key for key, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"{name}: missing {missing}")
            continue

        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        history = json.loads(paths["history"].read_text(encoding="utf-8"))
        plan = json.loads(paths["run_plan"].read_text(encoding="utf-8"))
        final_payload = torch.load(
            paths["final_weight"], map_location="cpu", weights_only=False)
        best_payload = torch.load(
            paths["best_weight"], map_location="cpu", weights_only=False)
        evaluation = summary.get("evaluation", {})
        assertions = summary.get("acceptance_assertions", {})
        modes = sorted(condition["allowed_transport_modes"])
        schedule_hash = canonical_hash(
            evaluation.get("arrival_schedule_metadata"))
        schedule_hashes.add(schedule_hash)
        final_hash = sha256(paths["final_weight"])
        best_hash = sha256(paths["best_weight"])
        final_hashes.add(final_hash)
        best_hashes.add(best_hash)
        ppo_configs.append(summary.get("ppo", {}))

        evaluation_updates = [
            int(item["update"]) for item in history if "evaluation" in item]
        history_updates = [int(item.get("update", -1)) for item in history]
        summary_checks = {
            "condition_name": plan.get("condition") == name,
            "protocol_version": (
                plan.get("protocol") == protocol["protocol"]["version"]),
            "execution_mode": summary.get("execution_mode") == "CONCURRENT",
            "arrival_profile": (
                summary.get("arrival_profile") == "MIXED_CURRICULUM"),
            "observation_variant": (
                summary.get("observation_variant") ==
                condition["observation_variant"]),
            "policy_variant": (
                summary.get("policy_variant") == condition["policy_variant"]),
            "discount_mode": (
                summary.get("discount_mode") == condition["discount_mode"]),
            "allowed_transport_modes": (
                sorted(summary.get("allowed_transport_modes", [])) == modes),
            "experiment_run_seed": (
                summary.get("experiment_run_seed") == run_seed),
            "training_seed_range": (
                summary.get("training_seed_start") == ppo_seed and
                summary.get("training_seed_end") == ppo_seed + 7999),
            "validation_seed_range": (
                summary.get("validation_seed_start") == validation_start and
                summary.get("validation_seed_end") == validation_start + 99),
            "training_budget": (
                summary.get("updates") == 2000 and
                summary.get("episodes_trained_total") == 8000),
            "history_is_exactly_2000_ordered_updates": (
                history_updates == list(range(1, 2001))),
            "validation_cadence_is_exactly_every_50_updates": (
                evaluation_updates == expected_validation_updates),
            "summary_and_history_are_finite": (
                all_finite(summary) and all_finite(history)),
            "all_acceptance_assertions_true": (
                bool(assertions) and all(assertions.values())),
            "summary_passed": summary.get("passed") is True,
            "evaluation_resolves_8000_tasks": (
                evaluation.get("scheduled_task_count") == 8000 and
                evaluation.get("unresolved") == 0 and
                evaluation.get("completed", 0) +
                evaluation.get("failed", 0) == 8000),
            "evaluation_has_no_illegal_action": (
                evaluation.get("illegal_action_count") == 0),
            "evaluation_has_no_resource_leak": (
                evaluation.get("resource_leak_count") == 0),
            "shared_validation_schedule_matches_stage20": (
                schedule_hash == reference_schedule_hash),
            "reward_normalization_present": bool(
                summary.get("reward_normalization")),
            "parameter_delta_is_positive_finite": (
                math.isfinite(float(summary.get("parameter_delta_l2", math.nan)))
                and float(summary.get("parameter_delta_l2", 0)) > 0),
            "warm_start_source_semantics": (
                (summary.get("warm_start_source") is not None) ==
                bool(condition["warm_start"])),
        }
        if name == "fixed_gamma":
            summary_checks["fixed_gamma_history_is_exactly_0_99"] = all(
                abs(float(item["rollout_discount_min"]) - 0.99) < 1e-12 and
                abs(float(item["rollout_discount_max"]) - 0.99) < 1e-12
                for item in history)
        else:
            summary_checks["variable_discount_changes_with_duration"] = any(
                float(item["rollout_discount_min"]) <
                float(item["rollout_discount_max"])
                for item in history)
        if name == "no_relay":
            summary_checks["relay_is_absent_from_evaluation"] = (
                "CAR_DOG_CAR" not in evaluation.get("transport_modes", {}) and
                sorted(evaluation.get("exposed_transport_modes", [])) == modes)
        if name == "no_warm_start":
            summary_checks["no_teacher_artifact_created"] = not any(
                (directory / filename).exists() for filename in (
                    "warm_start.pt", "warm_start.summary.json"))

        final_checks = checkpoint_checks(
            final_payload, condition=condition, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_start,
            model_metadata=summary["model"], final=True)
        best_checks = checkpoint_checks(
            best_payload, condition=condition, run_seed=run_seed,
            ppo_seed=ppo_seed, validation_seed=validation_start,
            model_metadata=summary["model"], final=False)
        if not all(summary_checks.values()):
            failures.append(
                f"{name} summary/history: " + json.dumps({
                    key: value for key, value in summary_checks.items()
                    if not value}, sort_keys=True))
        if not all(final_checks.values()):
            failures.append(
                f"{name} final checkpoint: " + json.dumps({
                    key: value for key, value in final_checks.items()
                    if not value}, sort_keys=True))
        if not all(best_checks.values()):
            failures.append(
                f"{name} best checkpoint: " + json.dumps({
                    key: value for key, value in best_checks.items()
                    if not value}, sort_keys=True))

        rows.append({
            "condition": name,
            "isolates": condition["isolates"],
            "checks": summary_checks,
            "final_checkpoint_checks": final_checks,
            "best_checkpoint_checks": best_checks,
            "best_checkpoint_update": int(best_payload["update"]),
            "success_rate": evaluation.get("success_rate"),
            "successful_throughput_tasks_per_hour": evaluation.get(
                "successful_throughput_tasks_per_hour"),
            "transport_modes": evaluation.get("transport_modes"),
            "validation_schedule_sha256": schedule_hash,
            **{f"{key}_path": str(path) for key, path in paths.items()},
            **{f"{key}_sha256": sha256(path)
               for key, path in paths.items()},
        })

    cohort_checks = {
        "all_four_conditions_present": len(rows) == 4,
        "all_validation_schedules_identical": len(schedule_hashes) == 1,
        "all_final_weights_are_distinct": len(final_hashes) == 4,
        "all_best_weights_are_distinct": len(best_hashes) == 4,
        "ppo_hyperparameters_are_identical": (
            len({canonical_hash(item) for item in ppo_configs}) == 1),
        "training_validation_locked_bootstrap_seeds_disjoint": len({
            run_seed, ppo_seed, validation_start,
            int(protocol["locked_test"]["seed_start"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        }) == 5,
    }
    if not all(cohort_checks.values()):
        failures.append("cohort checks: " + json.dumps({
            key: value for key, value in cohort_checks.items() if not value},
            sort_keys=True))

    no_relay_row = next(
        (row for row in rows if row["condition"] == "no_relay"), None)
    if no_relay_row and no_relay_row["success_rate"] == 1.0:
        warnings.append(
            "no_relay seed_01 has 100% validation success because disabling "
            "CAR_DOG_CAR removes handover-timeout exposure. Treat this as a "
            "mechanism diagnostic, not a superiority claim; retain all ten "
            "seeds and use the fresh task-keyed locked test.")

    report = {
        "schema_version": "warehouse_stage22_seed01_quality_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 22,
        "gate": "FORMAL_SEED01_COHORT_EXPANSION_GATE",
        "claim_boundary": (
            "Training provenance, comparability, completeness, and safety "
            "only. No ablation performance claim is made from one seed."),
        "protocol_path": str(CONFIG),
        "protocol_sha256": sha256(CONFIG),
        "stage20_reference_summary": str(REFERENCE),
        "stage20_reference_validation_schedule_sha256": (
            reference_schedule_hash),
        "cohort_checks": cohort_checks,
        "runs": rows,
        "warnings": warnings,
        "hard_gate": {
            "passed": not failures and len(rows) == 4,
            "safe_to_expand_to_seed_02_through_seed_10": (
                not failures and len(rows) == 4),
            "failures": failures,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "conditions": len(rows),
        "passed": report["hard_gate"]["passed"],
        "safe_to_expand": report["hard_gate"][
            "safe_to_expand_to_seed_02_through_seed_10"],
        "failures": failures,
        "warnings": warnings,
        "cohort_checks": cohort_checks,
    }, ensure_ascii=False, indent=2))
    return 0 if report["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
