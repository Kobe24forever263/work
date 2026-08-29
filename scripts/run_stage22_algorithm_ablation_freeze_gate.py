#!/usr/bin/env python3
"""Freeze and audit the complete 4 x 10 Stage 22 ablation cohort.

This gate verifies provenance and protocol compliance only.  It deliberately
does not inspect fresh locked-test outcomes and never drops a preregistered
training seed based on validation performance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np  # Bind the macOS BLAS/OpenMP runtime before importing torch.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage22_algorithm_ablation.yaml")
RESULT_ROOT = ROOT / "results" / "stage22_algorithm_ablation" / "long"
OUTPUT = (ROOT / "results" / "stage22_algorithm_ablation" /
          "stage22_policy_freeze_manifest.json")
STAGE20_MANIFEST = (ROOT / "results" / "stage20_mixed_curriculum" /
                    "stage20_policy_freeze_manifest.json")
STAGE20_REFERENCE = (ROOT / "results" / "stage20_mixed_curriculum" /
                     "long" / "seed_01" / "stage20_ppo.summary.json")

CODE_FILES = (
    ROOT / "scripts" / "run_stage14_policy_smoke.py",
    ROOT / "scripts" / "run_stage14_smdp_ppo.py",
    ROOT / "scripts" / "run_stage22_algorithm_ablation.py",
    ROOT / "scripts" / "run_stage22_algorithm_ablation_campaign.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" / "stage14_ppo.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_training.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage12_encoding.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def all_finite(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def expected_model(condition: dict) -> dict:
    architecture = {
        "CONTEXT_INTERACTION": "contextual_masked_candidate_actor_critic_v2",
        "FLAT_MASKED_PPO": "flat_masked_candidate_actor_critic_v1",
    }[condition["policy_variant"]]
    return {
        "architecture": architecture,
        "policy_variant": condition["policy_variant"],
        "state_width": 284,
        "action_width": 30,
        "hidden_width": 96,
        "index_specific_parameters": False,
    }


def checkpoint_checks(payload: dict, *, condition: dict, run_seed: int,
                      ppo_seed: int, validation_seed: int,
                      final: bool) -> dict:
    update = int(payload.get("update", -1))
    episodes = int(payload.get("episodes_trained", -1))
    allowed = sorted(condition["allowed_transport_modes"])
    stored_allowed = allowed if len(allowed) < 3 else None
    checks = {
        "execution_mode": payload.get("execution_mode") == "CONCURRENT",
        "arrival_profile": payload.get("arrival_profile") == "MIXED_CURRICULUM",
        "observation_variant": (
            payload.get("observation_variant") ==
            condition["observation_variant"]),
        "policy_variant": (
            payload.get("policy_variant") == condition["policy_variant"]),
        "discount_mode": (
            payload.get("discount_mode") == condition["discount_mode"]),
        "allowed_transport_modes": (
            payload.get("allowed_transport_modes") == stored_allowed),
        "experiment_run_seed": payload.get("experiment_run_seed") == run_seed,
        "ppo_seed": payload.get("base_seed") == ppo_seed,
        "validation_seed": payload.get("validation_seed_start") == validation_seed,
        "model_metadata": payload.get("model_metadata") == expected_model(condition),
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
        checks["best_checkpoint_on_validation_boundary"] = (
            0 < update <= 2000 and update % 50 == 0)
    return checks


def ranges_overlap(left: range, right: range) -> bool:
    return max(left.start, right.start) < min(left.stop, right.stop)


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    conditions = protocol["conditions"]
    training = protocol["training"]
    validation = protocol["validation"]
    locked = protocol["locked_test"]
    statistics = protocol["statistics"]
    run_seeds = [int(seed) for seed in training["independent_run_base_seeds"]]
    ppo_offset = int(training["ppo_seed_offset"])
    episodes_per_run = int(training["episodes_per_run"])
    warm_count = int(training["warm_start_seed_count"])
    validation_start = int(validation["seed_start"])
    validation_count = int(validation["seed_count"])
    locked_start = int(locked["seed_start"])
    locked_count = int(locked["seed_count"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    expected_eval_updates = list(range(50, 2001, 50))

    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    schedule_hashes: set[str] = set()
    final_hashes: set[str] = set()
    best_hashes: set[str] = set()
    ppo_hashes: set[str] = set()
    paired_seed_ranges: dict[str, list[list[int]]] = {
        name: [] for name in conditions}

    reference = json.loads(STAGE20_REFERENCE.read_text(encoding="utf-8"))
    reference_schedule_hash = canonical_hash(
        reference["evaluation"]["arrival_schedule_metadata"])

    if not STAGE20_MANIFEST.exists():
        failures.append("Stage 20 full-reference freeze manifest is missing")
        stage20_manifest = None
    else:
        stage20_manifest = json.loads(
            STAGE20_MANIFEST.read_text(encoding="utf-8"))
        if not stage20_manifest.get("hard_gate", {}).get("passed"):
            failures.append("Stage 20 full-reference freeze manifest did not pass")

    for condition_name, condition in conditions.items():
        for seed_index, run_seed in enumerate(run_seeds, start=1):
            ppo_seed = run_seed + ppo_offset
            paired_seed_ranges[condition_name].append(
                [ppo_seed, ppo_seed + episodes_per_run - 1])
            directory = (RESULT_ROOT / condition_name /
                         f"seed_{seed_index:02d}")
            paths = {
                "run_plan": directory / "run_plan.json",
                "summary": directory / "stage22_ppo.summary.json",
                "history": directory / "stage22_ppo.history.json",
                "final_weight": directory / "stage22_ppo.pt",
                "best_weight": directory / "stage22_ppo.best.pt",
            }
            if condition["warm_start"]:
                paths["warm_start_weight"] = directory / "warm_start.pt"
                paths["warm_start_summary"] = (
                    directory / "warm_start.summary.json")
            missing = [key for key, path in paths.items() if not path.exists()]
            if missing:
                failures.append(
                    f"{condition_name}/seed_{seed_index:02d}: missing {missing}")
                continue

            plan = json.loads(paths["run_plan"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            history = json.loads(paths["history"].read_text(encoding="utf-8"))
            final_payload = torch.load(
                paths["final_weight"], map_location="cpu", weights_only=False)
            best_payload = torch.load(
                paths["best_weight"], map_location="cpu", weights_only=False)
            evaluation = summary.get("evaluation", {})
            assertions = summary.get("acceptance_assertions", {})
            allowed = sorted(condition["allowed_transport_modes"])
            schedule_hash = canonical_hash(
                evaluation.get("arrival_schedule_metadata"))
            schedule_hashes.add(schedule_hash)
            ppo_hashes.add(canonical_hash(summary.get("ppo", {})))

            history_updates = [int(item.get("update", -1)) for item in history]
            evaluation_updates = [
                int(item["update"]) for item in history if "evaluation" in item]
            checks = {
                "run_plan_protocol": (
                    plan.get("protocol") == protocol["protocol"]["version"]),
                "run_plan_condition": plan.get("condition") == condition_name,
                "run_plan_condition_config": (
                    plan.get("condition_config") == condition),
                "run_plan_seed_index": plan.get("seed_index") == seed_index,
                "run_plan_experiment_seed": (
                    plan.get("experiment_run_seed") == run_seed),
                "run_plan_ppo_seed": plan.get("ppo_seed") == ppo_seed,
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
                    sorted(summary.get("allowed_transport_modes", [])) == allowed),
                "model_metadata": summary.get("model") == expected_model(condition),
                "experiment_run_seed": (
                    summary.get("experiment_run_seed") == run_seed),
                "training_seed_range": (
                    summary.get("training_seed_start") == ppo_seed and
                    summary.get("training_seed_end") ==
                    ppo_seed + episodes_per_run - 1),
                "validation_seed_range": (
                    summary.get("validation_seed_start") == validation_start and
                    summary.get("validation_seed_end") ==
                    validation_start + validation_count - 1),
                "training_budget": (
                    summary.get("updates") == 2000 and
                    summary.get("episodes_trained_total") == 8000),
                "history_exactly_2000_ordered_updates": (
                    history_updates == list(range(1, 2001))),
                "validation_exactly_every_50_updates": (
                    evaluation_updates == expected_eval_updates),
                "summary_and_history_finite": (
                    all_finite(summary) and all_finite(history)),
                "all_acceptance_assertions_true": (
                    bool(assertions) and all(assertions.values())),
                "summary_passed": summary.get("passed") is True,
                "evaluation_resolves_all_8000_tasks": (
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
                "parameter_delta_positive_finite": (
                    math.isfinite(float(summary.get(
                        "parameter_delta_l2", math.nan))) and
                    float(summary.get("parameter_delta_l2", 0.0)) > 0.0),
                "warm_start_summary_semantics": True,
                "warm_start_artifact_semantics": True,
            }

            if condition["warm_start"]:
                warm_summary = json.loads(
                    paths["warm_start_summary"].read_text(encoding="utf-8"))
                checks["warm_start_summary_semantics"] = bool(
                    warm_summary.get("run_seed") == run_seed and
                    warm_summary.get("training_seed_start") == run_seed and
                    warm_summary.get("training_seed_end") ==
                    run_seed + warm_count - 1 and
                    warm_summary.get("observation_variant") ==
                    condition["observation_variant"] and
                    warm_summary.get("policy_variant") ==
                    condition["policy_variant"] and
                    sorted(warm_summary.get("allowed_transport_modes", [])) ==
                    allowed and warm_summary.get("passed") is True)
                checks["warm_start_artifact_semantics"] = (
                    summary.get("warm_start_source") ==
                    str(paths["warm_start_weight"]))
            else:
                checks["warm_start_artifact_semantics"] = bool(
                    summary.get("warm_start_source") is None and
                    not (directory / "warm_start.pt").exists() and
                    not (directory / "warm_start.summary.json").exists())

            if condition["discount_mode"] == "FIXED_PER_DECISION":
                checks["discount_history_matches_declared_mode"] = all(
                    abs(float(item["rollout_discount_min"]) - 0.99) < 1e-12 and
                    abs(float(item["rollout_discount_max"]) - 0.99) < 1e-12
                    for item in history)
            else:
                checks["discount_history_matches_declared_mode"] = any(
                    float(item["rollout_discount_min"]) <
                    float(item["rollout_discount_max"])
                    for item in history)

            if condition_name == "no_relay":
                checks["relay_absent_from_evaluation"] = bool(
                    "CAR_DOG_CAR" not in evaluation.get("transport_modes", {}) and
                    sorted(evaluation.get("exposed_transport_modes", [])) == allowed)

            final_checks = checkpoint_checks(
                final_payload, condition=condition, run_seed=run_seed,
                ppo_seed=ppo_seed, validation_seed=validation_start, final=True)
            best_checks = checkpoint_checks(
                best_payload, condition=condition, run_seed=run_seed,
                ppo_seed=ppo_seed, validation_seed=validation_start, final=False)

            failed_checks = [key for key, value in checks.items() if not value]
            failed_final = [
                key for key, value in final_checks.items() if not value]
            failed_best = [key for key, value in best_checks.items() if not value]
            if failed_checks:
                failures.append(
                    f"{condition_name}/seed_{seed_index:02d} summary/history: "
                    f"{failed_checks}")
            if failed_final:
                failures.append(
                    f"{condition_name}/seed_{seed_index:02d} final: "
                    f"{failed_final}")
            if failed_best:
                failures.append(
                    f"{condition_name}/seed_{seed_index:02d} best: "
                    f"{failed_best}")

            path_hashes = {key: sha256(path) for key, path in paths.items()}
            final_hashes.add(path_hashes["final_weight"])
            best_hashes.add(path_hashes["best_weight"])
            rows.append({
                "condition": condition_name,
                "isolates": condition["isolates"],
                "seed_index": seed_index,
                "experiment_run_seed": run_seed,
                "ppo_episode_seed_range": [
                    ppo_seed, ppo_seed + episodes_per_run - 1],
                "warm_start_seed_range": (
                    [run_seed, run_seed + warm_count - 1]
                    if condition["warm_start"] else None),
                "retained_without_performance_selection": True,
                "accepted_by_training_gate": summary.get("passed") is True,
                "best_checkpoint_update": int(best_payload["update"]),
                "best_checkpoint_episodes_trained": int(
                    best_payload["episodes_trained"]),
                "validation_success_rate": evaluation.get("success_rate"),
                "validation_successful_throughput_tasks_per_hour": (
                    evaluation.get("successful_throughput_tasks_per_hour")),
                "validation_transport_modes": evaluation.get("transport_modes"),
                "validation_schedule_sha256": schedule_hash,
                "checks": checks,
                "final_checkpoint_checks": final_checks,
                "best_checkpoint_checks": best_checks,
                "paths": {key: str(path) for key, path in paths.items()},
                "sha256": path_hashes,
            })

    ppo_ranges = [range(seed + ppo_offset,
                        seed + ppo_offset + episodes_per_run)
                  for seed in run_seeds]
    validation_range = range(validation_start, validation_start + validation_count)
    locked_range = range(locked_start, locked_start + locked_count)
    within_condition_nonoverlap = all(
        not ranges_overlap(left, right)
        for index, left in enumerate(ppo_ranges)
        for right in ppo_ranges[index + 1:])
    seed_partition_checks = {
        "training_ranges_unique_within_each_condition": within_condition_nonoverlap,
        "conditions_deliberately_share_paired_training_streams": all(
            ranges == paired_seed_ranges[next(iter(conditions))]
            for ranges in paired_seed_ranges.values()),
        "training_vs_validation": all(
            not ranges_overlap(item, validation_range) for item in ppo_ranges),
        "training_vs_locked_test": all(
            not ranges_overlap(item, locked_range) for item in ppo_ranges),
        "validation_vs_locked_test": not ranges_overlap(
            validation_range, locked_range),
        "bootstrap_seed_outside_training_validation_locked": bool(
            all(bootstrap_seed not in item for item in ppo_ranges) and
            bootstrap_seed not in validation_range and
            bootstrap_seed not in locked_range),
    }

    expected_rows = len(conditions) * len(run_seeds)
    cohort_checks = {
        "all_40_preregistered_runs_retained": len(rows) == expected_rows == 40,
        "all_training_gates_accepted": all(
            row["accepted_by_training_gate"] for row in rows),
        "all_validation_schedules_identical": len(schedule_hashes) == 1,
        "all_validation_schedules_match_stage20": (
            schedule_hashes == {reference_schedule_hash}),
        "all_final_weights_distinct": len(final_hashes) == expected_rows,
        "all_best_weights_distinct": len(best_hashes) == expected_rows,
        "ppo_hyperparameters_identical": len(ppo_hashes) == 1,
        "seed_partition_contract": all(seed_partition_checks.values()),
        "stage20_full_reference_is_frozen": bool(
            stage20_manifest and
            stage20_manifest.get("hard_gate", {}).get("passed")),
    }
    if not all(seed_partition_checks.values()):
        failures.append(f"seed partition checks: {seed_partition_checks}")
    if not all(cohort_checks.values()):
        failures.append("cohort checks: " + json.dumps({
            key: value for key, value in cohort_checks.items() if not value},
            sort_keys=True))

    no_relay_rows = [row for row in rows if row["condition"] == "no_relay"]
    if no_relay_rows:
        warnings.append(
            "The no_relay condition removes handover-timeout exposure; its "
            "success rate is not an efficiency or superiority endpoint. "
            "Interpret only through the preregistered multi-metric locked test.")

    source_hashes = {}
    for path in (CONFIG, *CODE_FILES):
        if path.exists():
            source_hashes[str(path)] = sha256(path)
        else:
            failures.append(f"source/protocol file missing: {path}")

    report = {
        "schema_version": "warehouse_stage22_policy_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 22,
        "gate": "FULL_ALGORITHM_ABLATION_COHORT_FREEZE",
        "claim_boundary": (
            "Training provenance, completeness, protocol compliance, and "
            "immutable cohort selection only. Validation metrics are retained "
            "for checkpoint provenance and must not support performance claims. "
            "Fresh locked-test seeds remain untouched."),
        "selection_policy": "RETAIN_ALL_40_PREREGISTERED_RUNS",
        "locked_test_weight": "BEST_SHARED_VALIDATION_CHECKPOINT",
        "protocol_path": str(CONFIG),
        "protocol_sha256": sha256(CONFIG),
        "source_file_sha256": source_hashes,
        "stage20_reference": {
            "manifest_path": str(STAGE20_MANIFEST),
            "manifest_sha256": (
                sha256(STAGE20_MANIFEST) if STAGE20_MANIFEST.exists() else None),
            "manifest_passed": bool(
                stage20_manifest and
                stage20_manifest.get("hard_gate", {}).get("passed")),
            "validation_schedule_sha256": reference_schedule_hash,
        },
        "cohort": {
            "expected_runs": expected_rows,
            "actual_runs": len(rows),
            "condition_count": len(conditions),
            "training_seeds_per_condition": len(run_seeds),
            "training_gate_accepted": sum(
                row["accepted_by_training_gate"] for row in rows),
            "training_gate_not_accepted": sum(
                not row["accepted_by_training_gate"] for row in rows),
        },
        "seed_partitions": {
            "ppo_episode_seed_ranges_by_condition": paired_seed_ranges,
            "validation_seed_range": [
                validation_start, validation_start + validation_count - 1],
            "fresh_locked_test_seed_range": [
                locked_start, locked_start + locked_count - 1],
            "bootstrap_seed": bootstrap_seed,
            "checks": seed_partition_checks,
        },
        "cohort_checks": cohort_checks,
        "warnings": warnings,
        "hard_gate": {
            "passed": not failures and len(rows) == expected_rows,
            "safe_to_start_fresh_locked_test_after_statistics_freeze": (
                not failures and len(rows) == expected_rows),
            "failures": failures,
        },
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "expected_runs": expected_rows,
        "actual_runs": len(rows),
        "accepted": report["cohort"]["training_gate_accepted"],
        "passed": report["hard_gate"]["passed"],
        "safe_to_start_fresh_locked_test_after_statistics_freeze": (
            report["hard_gate"][
                "safe_to_start_fresh_locked_test_after_statistics_freeze"]),
        "failure_count": len(failures),
        "failures": failures,
        "warnings": warnings,
        "cohort_checks": cohort_checks,
        "seed_partition_checks": seed_partition_checks,
    }, ensure_ascii=False, indent=2))
    return 0 if report["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
