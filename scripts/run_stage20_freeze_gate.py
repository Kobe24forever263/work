#!/usr/bin/env python3
"""Freeze all ten Stage 20 mixed-curriculum policies without cherry-picking."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np  # Load the environment BLAS/OpenMP runtime first on macOS.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage20_mixed_curriculum.yaml"
)
RESULT_ROOT = ROOT / "results" / "stage20_mixed_curriculum" / "long"
OUT = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json"
)

NON_SELECTION_ASSERTIONS = (
    "ppo_parameters_updated",
    "training_metrics_are_finite",
    "discount_contract_is_valid",
    "evaluation_resolved_all_tasks",
    "evaluation_actions_all_legal",
    "evaluation_has_no_resource_leak",
    "one_policy_exposes_all_three_transport_modes",
    "context_conditioned_metrics_cover_every_assignment",
    "rule_baseline_has_no_resource_leak",
    "concurrent_mode_overlaps_when_requested",
    "mixed_curriculum_schedule_contract",
)
MODEL_METADATA = {
    "architecture": "contextual_masked_candidate_actor_critic_v2",
    "policy_variant": "CONTEXT_INTERACTION",
    "state_width": 284,
    "action_width": 30,
    "hidden_width": 96,
    "index_specific_parameters": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_checks(payload: dict, *, run_seed: int, ppo_seed: int,
                      validation_seed: int, final: bool) -> dict:
    update = int(payload.get("update", -1))
    episodes = int(payload.get("episodes_trained", -1))
    checks = {
        "execution_mode": payload.get("execution_mode") == "CONCURRENT",
        "arrival_profile": payload.get("arrival_profile") == "MIXED_CURRICULUM",
        "observation_variant": payload.get("observation_variant") == "FULL_CONTEXT_V2",
        "policy_variant": payload.get("policy_variant") == "CONTEXT_INTERACTION",
        "discount_mode": payload.get("discount_mode") == "VARIABLE_SMDP",
        "experiment_run_seed": payload.get("experiment_run_seed") == run_seed,
        "ppo_seed": payload.get("base_seed") == ppo_seed,
        "validation_seed": payload.get("validation_seed_start") == validation_seed,
        "model_metadata": payload.get("model_metadata") == MODEL_METADATA,
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
    training = protocol["training"]
    validation = protocol["validation"]
    locked = protocol["locked_test"]
    statistics = protocol["statistics"]
    run_seeds = [int(value) for value in training["independent_run_base_seeds"]]
    validation_start = int(validation["seed_start"])
    failures: list[str] = []
    rows: list[dict] = []

    used_training_seeds: set[int] = set()
    used_validation_seeds = set(range(
        validation_start, validation_start + int(validation["seed_count"])))
    locked_test_seeds = set(range(
        int(locked["seed_start"]),
        int(locked["seed_start"]) + int(locked["seed_count"])))
    bootstrap_seeds = set(range(
        int(statistics["bootstrap_seed_start"]),
        int(statistics["bootstrap_seed_start"]) + int(statistics["samples"])))

    for seed_index, run_seed in enumerate(run_seeds, start=1):
        ppo_seed = run_seed + int(training["ppo_seed_offset"])
        used_training_seeds.update(range(
            ppo_seed, ppo_seed + int(training["episodes_per_run"])))
        directory = RESULT_ROOT / f"seed_{seed_index:02d}"
        paths = {
            "summary": directory / "stage20_ppo.summary.json",
            "history": directory / "stage20_ppo.history.json",
            "final_weight": directory / "stage20_ppo.pt",
            "best_weight": directory / "stage20_ppo.best.pt",
            "warm_start_weight": directory / "warm_start.pt",
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"seed_{seed_index:02d}: missing {missing}")
            continue

        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        history = json.loads(paths["history"].read_text(encoding="utf-8"))
        final_payload = torch.load(
            paths["final_weight"], map_location="cpu", weights_only=False)
        best_payload = torch.load(
            paths["best_weight"], map_location="cpu", weights_only=False)
        assertions = summary.get("assertions", {})
        context_gate = assertions.get(
            "formal_policy_switches_modes_by_context") is True
        checks = {
            "profile_matches": summary.get("arrival_profile") == "MIXED_CURRICULUM",
            "execution_mode_matches": summary.get("execution_mode") == "CONCURRENT",
            "full_context_v2": summary.get("observation_variant") == "FULL_CONTEXT_V2",
            "context_interaction_policy": summary.get("policy_variant") == "CONTEXT_INTERACTION",
            "variable_smdp_discount": summary.get("discount_mode") == "VARIABLE_SMDP",
            "experiment_run_seed": summary.get("experiment_run_seed") == run_seed,
            "training_seed_start": summary.get("training_seed_start") == ppo_seed,
            "training_seed_end": summary.get("training_seed_end") == ppo_seed + 7999,
            "validation_seed_start": summary.get("validation_seed_start") == validation_start,
            "validation_seed_end": summary.get("validation_seed_end") == validation_start + 99,
            "updates_complete": summary.get("updates") == 2000,
            "episodes_complete": summary.get("episodes_trained_total") == 8000,
            "history_update_count": len(history) == 2000,
            "history_final_update": bool(history) and history[-1].get("update") == 2000,
            "non_selection_assertions": all(
                assertions.get(name) is True for name in NON_SELECTION_ASSERTIONS),
            "summary_pass_label_consistent": summary.get("passed") is context_gate,
            "reward_normalization_present": bool(summary.get("reward_normalization")),
            "model_metadata": summary.get("model") == MODEL_METADATA,
        }
        final_checks = checkpoint_checks(
            final_payload, run_seed=run_seed, ppo_seed=ppo_seed,
            validation_seed=validation_start, final=True)
        best_checks = checkpoint_checks(
            best_payload, run_seed=run_seed, ppo_seed=ppo_seed,
            validation_seed=validation_start, final=False)
        if not all(checks.values()):
            failures.append(f"seed_{seed_index:02d} summary/history: {checks}")
        if not all(final_checks.values()):
            failures.append(f"seed_{seed_index:02d} final checkpoint: {final_checks}")
        if not all(best_checks.values()):
            failures.append(f"seed_{seed_index:02d} best checkpoint: {best_checks}")
        rows.append({
            "training_seed_index": seed_index,
            "experiment_run_seed": run_seed,
            "ppo_seed_start": ppo_seed,
            "accepted_by_training_gate": context_gate,
            "failed_training_assertions": sorted(
                name for name, value in assertions.items() if not value),
            "best_checkpoint_update": int(best_payload["update"]),
            "best_checkpoint_episodes_trained": int(
                best_payload["episodes_trained"]),
            "checks": checks,
            "final_checkpoint_checks": final_checks,
            "best_checkpoint_checks": best_checks,
            **{f"{name}_path": str(path) for name, path in paths.items()},
            **{f"{name}_sha256": sha256(path) for name, path in paths.items()},
        })

    disjoint_checks = {
        "training_vs_validation": not (used_training_seeds & used_validation_seeds),
        "training_vs_locked_test": not (used_training_seeds & locked_test_seeds),
        "training_vs_bootstrap": not (used_training_seeds & bootstrap_seeds),
        "validation_vs_locked_test": not (used_validation_seeds & locked_test_seeds),
        "validation_vs_bootstrap": not (used_validation_seeds & bootstrap_seeds),
        "locked_test_vs_bootstrap": not (locked_test_seeds & bootstrap_seeds),
    }
    if not all(disjoint_checks.values()):
        failures.append(f"seed partition overlap: {disjoint_checks}")

    accepted_count = sum(row["accepted_by_training_gate"] for row in rows)
    payload = {
        "schema_version": "warehouse_stage20_policy_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Training provenance and immutable cohort freeze only. All ten "
            "pre-registered seeds are retained; locked-test performance is "
            "not certified here."
        ),
        "protocol_path": str(CONFIG),
        "protocol_sha256": sha256(CONFIG),
        "cohort": {
            "expected_runs": 10,
            "actual_runs": len(rows),
            "training_gate_accepted": accepted_count,
            "training_gate_not_accepted": len(rows) - accepted_count,
            "selection_policy": "RETAIN_ALL_PRE_REGISTERED_TRAINING_SEEDS",
            "locked_test_weight": "best_validation_checkpoint",
        },
        "seed_partitions": {
            "training_episode_seed_count": len(used_training_seeds),
            "validation_seed_range": [validation_start, validation_start + 99],
            "locked_test_seed_range": [min(locked_test_seeds), max(locked_test_seeds)],
            "bootstrap_seed_range": [min(bootstrap_seeds), max(bootstrap_seeds)],
            "disjoint_checks": disjoint_checks,
        },
        "hard_gate": {
            "passed": not failures and len(rows) == 10,
            "failures": failures,
        },
        "runs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "runs": len(rows),
        "accepted": accepted_count,
        "not_accepted": len(rows) - accepted_count,
        "passed": payload["hard_gate"]["passed"],
        "failures": failures,
        "seed_partitions": payload["seed_partitions"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
