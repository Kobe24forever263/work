#!/usr/bin/env python3
"""Freeze the ten corrected Stage 23 policies before locked evaluation.

This gate audits training completeness and reproduces the checkpoint-selection
rule from the validation history.  It never instantiates an environment and
therefore never reads the formal locked-test seed stream.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np  # Bind the environment BLAS runtime before torch on macOS.
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds_stage23_markov_continuous.yaml"
)
ANALYSIS_CONFIG = (
    ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_stage23_locked_test.yaml"
)
RESULT_ROOT = ROOT / "results" / "stage23_markov_continuous" / "long"
OUT = (
    ROOT / "results" / "stage23_markov_continuous" / "freeze" /
    "stage23_policy_freeze_manifest.json"
)
INTERFACE_GATE = (
    ROOT / "results" / "stage23_markov_continuous" /
    "stage23_markov_continuous_interface_gate.json"
)
STATE_GATE = (
    ROOT / "results" / "stage23_state_sufficiency" /
    "stage23_state_sufficiency_audit.json"
)
EVENT_GATE = (
    ROOT / "results" / "stage23_event_partition" /
    "stage23_event_partition_audit.json"
)
RESUME_GATE = (
    ROOT / "results" / "stage23_markov_continuous" /
    "resume_equivalence" / "stage23_resume_equivalence_gate.json"
)
CAMPAIGN_STATUS = (
    RESULT_ROOT / "stage23_campaign_01_05.status.json",
    RESULT_ROOT / "stage23_campaign_06_10.status.json",
)
SOURCE_PATHS = (
    ROOT / "scripts" / "run_stage14_policy_smoke.py",
    ROOT / "scripts" / "run_stage14_smdp_ppo.py",
    ROOT / "scripts" / "run_stage23_markov_continuous.py",
    ROOT / "scripts" / "run_stage23_markov_continuous_campaign.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage12_encoding.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_policy.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_ppo.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_training.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "persistent_dispatch.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "concurrent_dispatch.py",
)
EVALUATION_SOURCE_PATHS = (
    ROOT / "scripts" / "run_stage23_locked_test.py",
    ROOT / "scripts" / "summarize_stage23_locked_test.py",
    ROOT / "scripts" / "run_stage15_stress_evaluation.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_ppo.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "stage14_training.py",
    ROOT / "src" / "warehouse_core" / "warehouse_core" /
    "handover_timing.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE)
    return completed.stdout.strip()


def all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def contextual_rates(evaluation: dict, modes: list[str]) -> dict[str, float]:
    matrix = evaluation.get("selected_mode_by_cost_reference", {})
    rates = {}
    for mode in modes:
        row = matrix.get(mode, {})
        denominator = sum(int(value) for value in row.values())
        rates[mode] = (
            int(row.get(mode, 0)) / denominator if denominator else 0.0)
    return rates


def best_validation_row(history: list[dict]) -> dict:
    rows = [row for row in history if isinstance(row.get("evaluation"), dict)]
    if not rows:
        raise ValueError("history contains no validation rows")
    # The training code replaces the best checkpoint only on a strict score
    # improvement, so an exact tie is resolved in favour of the earliest row.
    return max(rows, key=lambda row: (
        float(row["evaluation"]["mean_reward"]), -int(row["update"])))


def checkpoint_checks(payload: dict, *, protocol: dict, run_seed: int,
                      ppo_seed: int, expected_update: int,
                      expected_score: float | None, final: bool,
                      summary: dict) -> dict[str, bool]:
    method = protocol["method"]
    training = protocol["training"]
    validation = protocol["validation"]
    update = int(payload.get("update", -1))
    checks = {
        "stage23_schema": payload.get("schema_version") ==
        "warehouse_stage16_training_v1",
        "execution_mode": payload.get("execution_mode") ==
        training["execution_mode"],
        "arrival_profile": payload.get("arrival_profile") ==
        training["arrival_profile"],
        "observation_variant": payload.get("observation_variant") ==
        method["observation_variant"],
        "policy_variant": payload.get("policy_variant") ==
        method["policy_variant"],
        "discount_mode": payload.get("discount_mode") ==
        method["discount_mode"],
        "trace_mode": payload.get("trace_mode") == method["trace_mode"],
        "trace_tau_s": payload.get("trace_tau_s") ==
        float(method["trace_tau_s"]),
        "reward_contract": payload.get("reward_contract") ==
        method["reward_contract"],
        "experiment_run_seed": payload.get("experiment_run_seed") ==
        run_seed,
        "ppo_seed": payload.get("base_seed") == ppo_seed,
        "validation_seed": payload.get("validation_seed_start") ==
        int(validation["seed_start"]),
        "model_metadata": payload.get("model_metadata") ==
        summary.get("model"),
        "encoder_metadata": payload.get("encoder_metadata") ==
        summary.get("encoder"),
        "expected_update": update == expected_update,
        "episode_count": payload.get("episodes_trained") ==
        update * int(training["episodes_per_update"]),
        "reward_normalizer": payload.get(
            "reward_normalizer", {}).get("source") ==
        "reverse_variable_discounted_return_to_go_std_v2",
        "numpy_rng": payload.get("numpy_random_state") is not None,
        "torch_rng": payload.get("torch_random_state") is not None,
        "ppo_rng": payload.get("ppo_generator_state") is not None,
        "model_state": bool(payload.get("model_state_dict")),
        "optimizer_state": bool(payload.get("optimizer_state_dict")),
    }
    if final:
        checks["terminal_update"] = update == int(training["formal_updates"])
    else:
        checks["validation_boundary"] = (
            update % int(training["evaluation_every_updates"]) == 0)
        checks["best_score_matches_history"] = bool(
            expected_score is not None and math.isclose(
                float(payload.get("best_evaluation_reward", math.nan)),
                expected_score, rel_tol=0.0, abs_tol=1e-10))
    return checks


def supporting_gate(path: Path) -> dict:
    payload = load_json(path)
    passed = bool(payload.get("passed"))
    if not passed:
        hard_gate = payload.get("hard_gate", {})
        passed = bool(hard_gate.get("passed"))
    if not passed:
        audit_gate = payload.get("audit_execution_gate", {})
        passed = bool(audit_gate.get("passed"))
    return {"path": str(path), "sha256": sha256(path), "passed": passed}


def main() -> int:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    analysis_protocol = yaml.safe_load(
        ANALYSIS_CONFIG.read_text(encoding="utf-8"))
    training = protocol["training"]
    validation = protocol["validation"]
    locked = protocol["locked_test"]
    statistics = protocol["statistics"]
    method = protocol["method"]
    run_seeds = [int(value) for value in training[
        "independent_run_base_seeds"]]
    expected_modes = sorted(method["allowed_transport_modes"])
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []

    analysis_checks = {
        "training_protocol_version":
        analysis_protocol["protocol"]["training_protocol_version"] ==
        protocol["protocol"]["version"],
        "locked_seed_start":
        int(analysis_protocol["locked_test"]["seed_start"]) ==
        int(locked["seed_start"]),
        "locked_seed_count":
        int(analysis_protocol["locked_test"]["seed_count"]) ==
        int(locked["seed_count"]),
        "handover_sampling":
        analysis_protocol["scenario"]["handover_sampling"] ==
        locked["handover_sampling"],
        "bootstrap_seed":
        int(analysis_protocol["statistics"]["bootstrap_seed"]) ==
        int(statistics["bootstrap_seed"]),
        "bootstrap_samples":
        int(analysis_protocol["statistics"]["bootstrap_samples"]) ==
        int(statistics["bootstrap_samples"]),
        "no_posthoc_noninferiority_claim":
        analysis_protocol["statistics"][
            "noninferiority_claim_allowed"] is False,
    }
    if not all(analysis_checks.values()):
        failures.append(
            f"locked-test analysis protocol mismatch: {analysis_checks}")

    campaign_rows = []
    for path in CAMPAIGN_STATUS:
        if not path.exists():
            failures.append(f"missing campaign status: {path}")
            continue
        payload = load_json(path)
        campaign_rows.append({
            "path": str(path), "sha256": sha256(path),
            "state": payload.get("state"),
        })
        if payload.get("state") != "COMPLETED":
            failures.append(f"campaign is not complete: {path}")

    support = {}
    for name, path in {
            "interface_gate": INTERFACE_GATE,
            "constructive_state_gate": STATE_GATE,
            "event_partition_gate": EVENT_GATE,
            "resume_equivalence_gate": RESUME_GATE}.items():
        if not path.exists():
            failures.append(f"missing supporting gate: {path}")
            continue
        support[name] = supporting_gate(path)
        if not support[name]["passed"]:
            failures.append(f"supporting gate did not pass: {name}")

    used_training_seeds: set[int] = set()
    used_warm_seeds: set[int] = set()
    selected_hashes: list[str] = []
    final_hashes: list[str] = []
    for seed_index, run_seed in enumerate(run_seeds, start=1):
        ppo_seed = run_seed + int(training["ppo_seed_offset"])
        warm_seed = run_seed + int(training["warm_start_seed_offset"])
        used_training_seeds.update(range(
            ppo_seed, ppo_seed + int(training["episodes_per_run"])))
        used_warm_seeds.update(range(
            warm_seed, warm_seed + int(training["warm_start_seed_count"])))
        directory = RESULT_ROOT / f"seed_{seed_index:02d}"
        paths = {
            "summary": directory / "stage23_ppo.summary.json",
            "history": directory / "stage23_ppo.history.json",
            "final_weight": directory / "stage23_ppo.pt",
            "best_weight": directory / "stage23_ppo.best.pt",
            "warm_start_weight": directory / "warm_start.pt",
            "warm_start_summary": directory / "warm_start.summary.json",
            "run_plan": directory / "run_plan.json",
            "run_status": directory / "stage23_run.status.json",
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"seed_{seed_index:02d}: missing {missing}")
            continue

        summary = load_json(paths["summary"])
        history = load_json(paths["history"])
        warm = load_json(paths["warm_start_summary"])
        run_plan = load_json(paths["run_plan"])
        run_status = load_json(paths["run_status"])
        best_row = best_validation_row(history)
        best_update = int(best_row["update"])
        best_evaluation = best_row["evaluation"]
        best_score = float(best_evaluation["mean_reward"])
        final_payload = torch.load(
            paths["final_weight"], map_location="cpu", weights_only=False)
        best_payload = torch.load(
            paths["best_weight"], map_location="cpu", weights_only=False)

        eval_rows = [row for row in history if row.get("evaluation")]
        expected_eval_updates = list(range(
            int(training["evaluation_every_updates"]),
            int(training["formal_updates"]) + 1,
            int(training["evaluation_every_updates"])))
        checkpoint_paths = sorted(
            (directory / "checkpoints").glob("stage23_ppo_update_*.pt"))
        checkpoint_updates = [
            int(path.stem.rsplit("_", 1)[1]) for path in checkpoint_paths]
        expected_checkpoint_updates = list(range(
            int(training["checkpoint_every_updates"]),
            int(training["formal_updates"]) + 1,
            int(training["checkpoint_every_updates"])))
        rates = contextual_rates(best_evaluation, expected_modes)
        acceptance_assertions = summary.get("acceptance_assertions", {})
        checks = {
            "summary_stage": summary.get("stage") == 23,
            "training_gate": summary.get("gate") ==
            "MARKOV_CONTINUOUS_SMDP_PPO_V3_TRAINING_RUN",
            "summary_passed": summary.get("passed") is True,
            "all_nonselection_acceptance_assertions": bool(
                acceptance_assertions) and all(
                acceptance_assertions.values()),
            "method_metadata": all((
                summary.get("observation_variant") ==
                method["observation_variant"],
                summary.get("policy_variant") == method["policy_variant"],
                summary.get("discount_mode") == method["discount_mode"],
                summary.get("trace_mode") == method["trace_mode"],
                summary.get("trace_tau_s") == float(method["trace_tau_s"]),
                summary.get("reward_contract") == method["reward_contract"],
            )),
            "run_seed": summary.get("experiment_run_seed") == run_seed,
            "training_seed_range": (
                summary.get("training_seed_start") == ppo_seed and
                summary.get("training_seed_end") ==
                ppo_seed + int(training["episodes_per_run"]) - 1),
            "validation_seed_range": (
                summary.get("validation_seed_start") ==
                int(validation["seed_start"]) and
                summary.get("validation_seed_end") ==
                int(validation["seed_start"]) +
                int(validation["seed_count"]) - 1),
            "updates_complete": summary.get("updates") ==
            int(training["formal_updates"]),
            "episodes_complete": summary.get("episodes_trained_total") ==
            int(training["episodes_per_run"]),
            "history_length": len(history) ==
            int(training["formal_updates"]),
            "history_updates_contiguous": [
                int(row.get("update", -1)) for row in history] ==
            list(range(1, int(training["formal_updates"]) + 1)),
            "history_finite": all_finite(history),
            "validation_schedule": [int(row["update"]) for row in eval_rows]
            == expected_eval_updates,
            "checkpoint_schedule": checkpoint_updates ==
            expected_checkpoint_updates,
            "validation_tasks_complete": all(
                row["evaluation"].get("episode_count") ==
                int(validation["seed_count"]) and
                row["evaluation"].get("scheduled_task_count") ==
                int(validation["seed_count"]) *
                int(training["tasks_per_episode"]) and
                row["evaluation"].get("unresolved") == 0 and
                row["evaluation"].get("illegal_action_count") == 0 and
                row["evaluation"].get("resource_leak_count") == 0
                for row in eval_rows),
            "best_contextual_soft_gate": all(
                rates[mode] >= 0.20 for mode in expected_modes),
            "best_exposes_three_modes": sorted(
                best_evaluation.get("exposed_transport_modes", [])) ==
            expected_modes,
            "warm_start_interface_safe": bool(
                warm.get("interface_continuation_assertions")) and all(
                warm["interface_continuation_assertions"].values()),
            "run_plan_protocol": run_plan.get("protocol") ==
            protocol["protocol"]["version"],
            "run_status_complete": run_status.get("completed") is True,
        }
        final_checks = checkpoint_checks(
            final_payload, protocol=protocol, run_seed=run_seed,
            ppo_seed=ppo_seed,
            expected_update=int(training["formal_updates"]),
            expected_score=None, final=True, summary=summary)
        best_checks = checkpoint_checks(
            best_payload, protocol=protocol, run_seed=run_seed,
            ppo_seed=ppo_seed, expected_update=best_update,
            expected_score=best_score, final=False, summary=summary)
        if not all(checks.values()):
            failures.append(
                f"seed_{seed_index:02d} summary/history: {checks}")
        if not all(final_checks.values()):
            failures.append(
                f"seed_{seed_index:02d} final checkpoint: {final_checks}")
        if not all(best_checks.values()):
            failures.append(
                f"seed_{seed_index:02d} best checkpoint: {best_checks}")
        if not warm.get("passed"):
            warnings.append(
                f"seed_{seed_index:02d}: strict imitation diagnostic did "
                "not pass; PPO admission used the predeclared interface-"
                "safety continuation rule")

        selected_hash = sha256(paths["best_weight"])
        final_hash = sha256(paths["final_weight"])
        selected_hashes.append(selected_hash)
        final_hashes.append(final_hash)
        rows.append({
            "training_seed_index": seed_index,
            "experiment_run_seed": run_seed,
            "warm_start_seed_range": [
                warm_seed,
                warm_seed + int(training["warm_start_seed_count"]) - 1],
            "ppo_episode_seed_range": [
                ppo_seed,
                ppo_seed + int(training["episodes_per_run"]) - 1],
            "selected_checkpoint": {
                "selection_rule": (
                    "MAX_VALIDATION_MEAN_RAW_EPISODE_REWARD_"
                    "EARLIEST_EXACT_TIE"),
                "update": best_update,
                "episodes_trained": int(best_row["episodes_trained"]),
                "validation_mean_reward": best_score,
                "validation_success_rate": best_evaluation["success_rate"],
                "validation_successful_throughput_tasks_per_hour":
                best_evaluation["successful_throughput_tasks_per_hour"],
                "validation_cost_reference_mode_agreement":
                best_evaluation.get("cost_reference_mode_agreement"),
                "contextual_diagonal_rates": rates,
                "path": str(paths["best_weight"]),
                "sha256": selected_hash,
            },
            "terminal_checkpoint": {
                "update": int(training["formal_updates"]),
                "path": str(paths["final_weight"]),
                "sha256": final_hash,
            },
            "strict_imitation_diagnostic_passed": bool(warm.get("passed")),
            "interface_safety_admission_passed": checks[
                "warm_start_interface_safe"],
            "checks": checks,
            "best_checkpoint_checks": best_checks,
            "terminal_checkpoint_checks": final_checks,
            "evidence": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in paths.items()
                if name not in {"best_weight", "final_weight"}
            },
        })

    validation_seeds = set(range(
        int(validation["seed_start"]),
        int(validation["seed_start"]) + int(validation["seed_count"])))
    locked_seeds = set(range(
        int(locked["seed_start"]),
        int(locked["seed_start"]) + int(locked["seed_count"])))
    bootstrap_seeds = {int(statistics["bootstrap_seed"])}
    partition_checks = {
        "warm_vs_ppo": not (used_warm_seeds & used_training_seeds),
        "training_vs_validation": not (
            used_training_seeds & validation_seeds),
        "training_vs_locked_test": not (
            used_training_seeds & locked_seeds),
        "training_vs_bootstrap": not (
            used_training_seeds & bootstrap_seeds),
        "validation_vs_locked_test": not (
            validation_seeds & locked_seeds),
        "validation_vs_bootstrap": not (
            validation_seeds & bootstrap_seeds),
        "locked_test_vs_bootstrap": not (
            locked_seeds & bootstrap_seeds),
    }
    if not all(partition_checks.values()):
        failures.append(f"seed partition overlap: {partition_checks}")
    if len(set(selected_hashes)) != len(run_seeds):
        failures.append("selected checkpoint hashes are not unique")
    if len(set(final_hashes)) != len(run_seeds):
        failures.append("terminal checkpoint hashes are not unique")
    if len(rows) != len(run_seeds):
        failures.append(
            f"expected {len(run_seeds)} audited runs, found {len(rows)}")

    source_fingerprints = {}
    for path in SOURCE_PATHS:
        if not path.exists():
            failures.append(f"missing training source: {path}")
            continue
        source_fingerprints[str(path.relative_to(ROOT))] = sha256(path)
    evaluation_source_fingerprints = {}
    for path in EVALUATION_SOURCE_PATHS:
        if not path.exists():
            failures.append(f"missing evaluation source: {path}")
            continue
        evaluation_source_fingerprints[
            str(path.relative_to(ROOT))] = sha256(path)
    payload = {
        "schema_version": "warehouse_stage23_policy_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Training provenance and validation-selected cohort freeze only. "
            "No formal locked-test seed is read and no held-out performance "
            "claim is certified here."),
        "protocol": {
            "path": str(CONFIG),
            "version": protocol["protocol"]["version"],
            "sha256": sha256(CONFIG),
        },
        "analysis_protocol": {
            "path": str(ANALYSIS_CONFIG),
            "version": analysis_protocol["protocol"]["version"],
            "sha256": sha256(ANALYSIS_CONFIG),
            "checks": analysis_checks,
            "evaluation_source_sha256": evaluation_source_fingerprints,
        },
        "cohort": {
            "expected_runs": len(run_seeds),
            "actual_runs": len(rows),
            "retention_policy": "RETAIN_ALL_PREDECLARED_TRAINING_SEEDS",
            "locked_test_weight": "BEST_VALIDATION_CHECKPOINT",
            "checkpoint_selection_rule": (
                "MAX_VALIDATION_MEAN_RAW_EPISODE_REWARD_"
                "EARLIEST_EXACT_TIE"),
            "validation_frequency_updates": int(
                training["evaluation_every_updates"]),
            "selected_update_range": [
                min((row["selected_checkpoint"]["update"] for row in rows),
                    default=None),
                max((row["selected_checkpoint"]["update"] for row in rows),
                    default=None),
            ],
        },
        "formal_analysis_boundary": {
            "primary_estimand": (
                "mean_across_shared_test_streams_of_episode_p95_waiting_time"),
            "tail_sensitivity_estimand": "pooled_task_p95_waiting_time",
            "independent_training_units": len(run_seeds),
            "shared_locked_test_streams": int(locked["seed_count"]),
            "tasks_per_stream_are_clustered_not_independent": True,
            "bootstrap_design": statistics["design"],
            "bootstrap_samples": int(statistics["bootstrap_samples"]),
            "bootstrap_seed": int(statistics["bootstrap_seed"]),
            "guardrails": statistics["guardrails"],
            "guardrail_inference": (
                "DESCRIPTIVE_CI_ONLY_NO_NONINFERIORITY_MARGIN_"
                "WAS_PREREGISTERED"),
            "multiplicity": statistics["multiplicity"],
            "locked_test_handover_sampling": locked["handover_sampling"],
        },
        "seed_partitions": {
            "training_episode_seed_count": len(used_training_seeds),
            "warm_start_seed_count": len(used_warm_seeds),
            "validation_seed_range": [min(validation_seeds),
                                      max(validation_seeds)],
            "locked_test_seed_range": [min(locked_seeds), max(locked_seeds)],
            "bootstrap_seed": int(statistics["bootstrap_seed"]),
            "disjoint_checks": partition_checks,
        },
        "provenance": {
            "git_head": git_text("rev-parse", "HEAD"),
            "git_status_porcelain": git_text("status", "--porcelain"),
            "source_sha256": source_fingerprints,
            "campaign_status": campaign_rows,
            "supporting_gates": support,
        },
        "warnings": warnings,
        "hard_gate": {
            "passed": not failures,
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
        "selected_updates": [
            row["selected_checkpoint"]["update"] for row in rows],
        "strict_imitation_diagnostic_failures": sum(
            not row["strict_imitation_diagnostic_passed"] for row in rows),
        "warnings": warnings,
        "passed": payload["hard_gate"]["passed"],
        "failures": failures,
        "locked_test_seed_stream_touched": False,
    }, ensure_ascii=False, indent=2))
    return 0 if payload["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
