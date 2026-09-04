#!/usr/bin/env python3
"""Pilot gate for the causal Stage 24 rolling-horizon optimizer.

This file deliberately reuses the Stage 15 evaluator instead of changing any
Stage 23 frozen evaluation source.  ``run_policy`` still sees the rule-policy
entry point, while an ``env_factory`` supplies an environment whose
``rule_action`` delegates to the Stage 24 optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterable

# Keep NumPy ahead of modules that import torch in the Apple Silicon Pixi env.
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))

from warehouse_core.stage14_training import MEDIUM_INTERVAL  # noqa: E402
from warehouse_core.stage24_rolling_optimizer import (  # noqa: E402
    Stage24OptimizerConfig, Stage24OptimizerGymEnv)
from run_stage15_stress_evaluation import (  # noqa: E402
    StressScenario, run_policy)


SCHEMA_VERSION = "warehouse_stage24_optimizer_smoke_v1"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "stage24_rolling_optimizer" / "pilot")
MIXED_CURRICULUM = StressScenario(
    name="STAGE24_MIXED_CURRICULUM_PILOT",
    description=(
        "Non-formal persistent randomized NORMAL/DENSE/BURST/RECOVERY "
        "curriculum shared by the optimizer and time-greedy baseline"),
    arrival_interval=MEDIUM_INTERVAL,
    episode_spec=None,
    arrival_schedule="MIXED_CURRICULUM",
)


def percentile(values: Iterable[float], quantile: float) -> float:
    rows = np.asarray(tuple(values), dtype=np.float64)
    return float(np.percentile(rows, quantile)) if rows.size else 0.0


def _config_field_names() -> set[str]:
    if is_dataclass(Stage24OptimizerConfig):
        return {field.name for field in fields(Stage24OptimizerConfig)}
    return set(inspect.signature(Stage24OptimizerConfig).parameters)


def build_optimizer_config(args: argparse.Namespace):
    """Map the public CLI to the optimizer config without hiding drift.

    The aliases make this pilot tolerant of the final descriptive field names
    (events vs decisions, budget vs wall-clock budget).  Exactly one alias for
    every requested control must exist; otherwise the pilot stops before any
    evidence is produced.
    """
    available = _config_field_names()
    requested = {
        "budget": (
            ("wall_clock_budget_ms", "budget_ms"), float(args.budget_ms)),
        "horizon": (
            ("horizon_events", "horizon_decisions"),
            int(args.horizon_events)),
        "visibility": (
            ("visibility", "visibility_mode"),
            str(args.visibility).upper()),
        "max_root_rollouts": (
            ("max_root_rollouts",), int(args.max_root_rollouts)),
        "budget_reserve": (
            ("budget_reserve_ms",), min(
                float(args.budget_reserve_ms),
                max(0.0, float(args.budget_ms) * 0.1))),
    }
    kwargs: dict[str, Any] = {}
    resolved: dict[str, str] = {}
    for public_name, (aliases, value) in requested.items():
        matches = [name for name in aliases if name in available]
        if len(matches) != 1:
            raise RuntimeError(
                f"Stage24OptimizerConfig must expose exactly one {public_name} "
                f"field from {aliases}; found {matches}")
        kwargs[matches[0]] = value
        resolved[public_name] = matches[0]
    return Stage24OptimizerConfig(**kwargs), resolved


def optimizer_env_factory(config, environment_sink: list):
    """Return a fresh-env factory and retain each env for diagnostics."""
    def factory(*args, **kwargs):
        env = Stage24OptimizerGymEnv(
            *args, optimizer_config=config, **kwargs)
        environment_sink.append(env)
        return env
    return factory


def _first(row: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def collect_optimizer_diagnostics(environments: list) -> list[dict]:
    records: list[dict] = []
    for episode_index, env in enumerate(environments, start=1):
        rows = list(getattr(env, "optimizer_diagnostics", []))
        for decision_index, item in enumerate(rows, start=1):
            if not isinstance(item, dict):
                raise TypeError("optimizer diagnostics must be dictionaries")
            row = dict(item)
            row.setdefault("episode_index", episode_index)
            row.setdefault("decision_index", decision_index)
            records.append(row)
    return records


def summarize_optimizer_diagnostics(records: list[dict], budget_ms: float):
    latencies = []
    missing_latency = 0
    for row in records:
        value = _first(row, (
            "latency_ms", "wall_clock_ms", "solve_time_ms",
            "decision_latency_ms"))
        if value is None:
            missing_latency += 1
        else:
            latencies.append(float(value))

    def flag_count(names: tuple[str, ...], *, derived=None) -> int:
        count = 0
        for row in records:
            value = _first(row, names)
            if value is None and derived is not None:
                value = derived(row)
            count += int(bool(value))
        return count

    timed_out = flag_count(("timed_out", "timeout", "budget_timeout"))
    fallback = flag_count(("fallback_used", "used_fallback", "fallback"))
    budget_compliant = flag_count(
        ("budget_compliant", "within_budget"),
        derived=lambda row: (
            float(_first(row, (
                "latency_ms", "wall_clock_ms", "solve_time_ms",
                "decision_latency_ms"), float("inf"))) <= budget_ms))
    exact_root = flag_count((
        "exact_root_enumeration", "root_enumeration_exact",
        "enumerated_all_root_actions"))
    search_truncated = flag_count(("search_truncated",))
    gap_reporting_complete = sum(
        int(
            row.get("certified_gap_status") in {
                "EXACT_ZERO", "UNKNOWN_PRUNED", "UNKNOWN_TIMEOUT"} and
            ((row.get("exact_gap") == 0.0) if
             row.get("certified_gap_status") == "EXACT_ZERO" else
             row.get("exact_gap") is None))
        for row in records)
    root_candidates = [int(value) for value in (
        _first(row, ("root_candidate_count", "root_candidates"))
        for row in records) if value is not None]
    root_evaluated = [int(value) for value in (
        _first(row, ("evaluated_root_count", "root_evaluated_count"))
        for row in records) if value is not None]
    exact_gaps = [float(value) for value in (
        _first(row, ("exact_gap",)) for row in records)
        if value is not None]
    count = len(records)
    return {
        "decision_count": count,
        "latency_sample_count": len(latencies),
        "missing_latency_count": missing_latency,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "maximum": max(latencies, default=0.0),
        },
        "wall_clock_budget_ms": budget_ms,
        "budget_compliant_count": budget_compliant,
        "budget_compliance_rate": budget_compliant / max(count, 1),
        "timeout_count": timed_out,
        "timeout_rate": timed_out / max(count, 1),
        "fallback_count": fallback,
        "fallback_rate": fallback / max(count, 1),
        "exact_root_enumeration_count": exact_root,
        "exact_root_enumeration_rate": exact_root / max(count, 1),
        "search_truncated_count": search_truncated,
        "search_truncated_rate": search_truncated / max(count, 1),
        "gap_reporting_complete_count": gap_reporting_complete,
        "gap_reporting_complete_rate": (
            gap_reporting_complete / max(count, 1)),
        "root_candidate_count": {
            "minimum": min(root_candidates, default=0),
            "maximum": max(root_candidates, default=0),
            "mean": float(np.mean(root_candidates))
            if root_candidates else 0.0,
        },
        "evaluated_root_count": {
            "minimum": min(root_evaluated, default=0),
            "maximum": max(root_evaluated, default=0),
            "mean": float(np.mean(root_evaluated))
            if root_evaluated else 0.0,
        },
        "exact_gap": {
            "reported_count": len(exact_gaps),
            "nonzero_count": sum(abs(value) > 1e-12
                                 for value in exact_gaps),
            "maximum_absolute": max(
                (abs(value) for value in exact_gaps), default=0.0),
        },
    }


PERFORMANCE_KEYS = (
    "success_rate", "mean_episode_reward",
    "successful_throughput_tasks_per_hour", "mean_waiting_time",
    "p95_waiting_time", "mean_flow_time", "p95_flow_time",
    "distance_per_task", "maximum_active_tasks", "illegal_action_count",
    "resource_leak_count", "transport_modes", "phase_results",
)


def performance_summary(result: dict) -> dict:
    return {key: result[key] for key in PERFORMANCE_KEYS}


def performance_delta(optimizer: dict, baseline: dict) -> dict:
    keys = (
        "success_rate", "mean_episode_reward",
        "successful_throughput_tasks_per_hour", "mean_waiting_time",
        "p95_waiting_time", "mean_flow_time", "p95_flow_time",
        "distance_per_task")
    return {
        key: float(optimizer[key]) - float(baseline[key]) for key in keys}


def fingerprints_match(left: dict, right: dict, aggregate_key: str,
                       episode_key: str | None = None) -> bool:
    episode_key = episode_key or aggregate_key
    return bool(
        left.get(aggregate_key) and
        left.get(aggregate_key) == right.get(aggregate_key) and
        len(left.get("episodes", [])) == len(right.get("episodes", [])) and
        all(a.get(episode_key) == b.get(episode_key) for a, b in zip(
            left.get("episodes", []), right.get("episodes", []))))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: getattr(value, field.name)
                for field in fields(value)}
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 24 rolling-optimizer non-formal pilot gate")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=71310000)
    parser.add_argument("--budget-ms", type=float, default=1000.0)
    parser.add_argument("--horizon-events", type=int, default=3)
    parser.add_argument("--max-root-rollouts", type=int, default=24)
    parser.add_argument("--budget-reserve-ms", type=float, default=25.0)
    parser.add_argument("--visibility", default="RELEASED_ONLY")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.budget_ms <= 0.0:
        parser.error("--budget-ms must be positive")
    if args.horizon_events <= 0:
        parser.error("--horizon-events must be positive")
    if args.max_root_rollouts <= 0:
        parser.error("--max-root-rollouts must be positive")
    if args.budget_reserve_ms < 0.0:
        parser.error("--budget-reserve-ms must be non-negative")

    config, field_mapping = build_optimizer_config(args)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "stage": 24,
        "gate": "ROLLING_OPTIMIZER_NONFORMAL_SMOKE",
        "claim_boundary": (
            "Non-formal pilot only. Seeds may be used for implementation "
            "debugging and must not be reused for the formal locked test."),
        "execute": args.execute,
        "scenario": MIXED_CURRICULUM.name,
        "arrival_schedule": MIXED_CURRICULUM.arrival_schedule,
        "execution_mode": "CONCURRENT",
        "observation_variant": "MARKOV_CONTEXT_V3",
        "reward_contract": "CONTINUOUS_TIME_V3",
        "episode_count": args.episodes,
        "tasks_per_episode": MIXED_CURRICULUM.task_count,
        "seed_range": [seeds[0], seeds[-1]],
        "optimizer_config": config,
        "config_field_mapping": field_mapping,
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2,
                     default=_json_default))
    if not args.execute:
        return 0

    environment_sink: list = []
    factory = optimizer_env_factory(config, environment_sink)
    common_arguments = {
        "seeds": seeds,
        "scenario": MIXED_CURRICULUM,
        "policy_kind": "TIME_GREEDY_RULE",
        "execution_mode": "CONCURRENT",
        "observation_variant": "MARKOV_CONTEXT_V3",
        "reward_contract": "CONTINUOUS_TIME_V3",
    }

    def progress(label):
        def callback(index, total, episode):
            print(
                f"{label} episode {index}/{total} seed={episode['seed']} "
                f"success={episode['success_rate']:.2%} "
                f"p95_wait={episode['p95_waiting_time']:.3f}s",
                flush=True)
        return callback

    optimizer_result = run_policy(
        None, env_factory=factory,
        progress_callback=progress("optimizer"), **common_arguments)
    baseline_result = run_policy(
        None, progress_callback=progress("time-greedy"),
        **common_arguments)
    diagnostics = collect_optimizer_diagnostics(environment_sink)
    diagnostic_summary = summarize_optimizer_diagnostics(
        diagnostics, args.budget_ms)

    expected_tasks = args.episodes * MIXED_CURRICULUM.task_count
    assertions = {
        "task_fingerprints_identical": fingerprints_match(
            optimizer_result, baseline_result, "task_fingerprint",
            "task_stream_fingerprint"),
        "handover_fingerprints_identical": fingerprints_match(
            optimizer_result, baseline_result,
            "handover_potential_fingerprint"),
        "optimizer_resolved_all_tasks": (
            optimizer_result["task_count"] == expected_tasks and
            optimizer_result["completed"] + optimizer_result["failed"] ==
            expected_tasks),
        "time_greedy_resolved_all_tasks": (
            baseline_result["task_count"] == expected_tasks and
            baseline_result["completed"] + baseline_result["failed"] ==
            expected_tasks),
        "optimizer_actions_all_legal": (
            optimizer_result["illegal_action_count"] == 0),
        "optimizer_has_no_resource_leak": (
            optimizer_result["resource_leak_count"] == 0),
        "time_greedy_actions_all_legal": (
            baseline_result["illegal_action_count"] == 0),
        "time_greedy_has_no_resource_leak": (
            baseline_result["resource_leak_count"] == 0),
        "optimizer_diagnostics_present": (
            diagnostic_summary["decision_count"] > 0),
        "optimizer_latency_complete": (
            diagnostic_summary["missing_latency_count"] == 0),
        "optimizer_gap_status_is_complete": (
            diagnostic_summary["gap_reporting_complete_rate"] == 1.0),
        "optimizer_meets_99pct_budget_compliance": (
            diagnostic_summary["budget_compliance_rate"] >= 0.99),
    }
    passed = all(assertions.values())
    output = {
        **plan,
        "execute": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_labels": {
            "optimizer": "STAGE24_ROLLING_HORIZON_OPTIMIZER",
            "baseline": "TIME_GREEDY_RULE",
            "run_policy_entry_for_optimizer": "TIME_GREEDY_RULE",
        },
        "optimizer_diagnostic_summary": diagnostic_summary,
        "performance": {
            "optimizer": performance_summary(optimizer_result),
            "time_greedy": performance_summary(baseline_result),
            "optimizer_minus_time_greedy": performance_delta(
                optimizer_result, baseline_result),
        },
        "fingerprints": {
            "optimizer_task": optimizer_result["task_fingerprint"],
            "time_greedy_task": baseline_result["task_fingerprint"],
            "optimizer_handover": optimizer_result[
                "handover_potential_fingerprint"],
            "time_greedy_handover": baseline_result[
                "handover_potential_fingerprint"],
        },
        "assertions": assertions,
        "passed": passed,
        "optimizer_diagnostics": diagnostics,
        "raw_results": {
            "optimizer": optimizer_result,
            "time_greedy": baseline_result,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"stage24_optimizer_smoke_{args.episodes}ep_"
        f"seed{args.seed_start}.json")
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2,
                   default=_json_default) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "output": str(output_path.resolve()),
        "optimizer_diagnostic_summary": diagnostic_summary,
        "performance": output["performance"],
        "assertions": assertions,
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
