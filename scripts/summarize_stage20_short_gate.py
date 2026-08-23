#!/usr/bin/env python3
"""Validate and summarize the three Stage 20 mixed-curriculum smoke seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


WORK_ROOT = Path(__file__).resolve().parents[1]


def load_summary(seed_index: int) -> tuple[Path, dict]:
    path = (
        WORK_ROOT / "results" / "stage20_mixed_curriculum" / "short_gate" /
        f"seed_{seed_index:02d}" / "stage20_ppo.summary.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=3)
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")

    rows: list[dict] = []
    all_orders: set[tuple[str, ...]] = set()
    for seed_index in range(1, args.seed_count + 1):
        path, summary = load_summary(seed_index)
        evaluation = summary["evaluation"]
        assertions = summary.get("assertions", {})
        schedules = evaluation.get("arrival_schedule_metadata", [])
        for schedule in schedules:
            order = schedule.get("phase_order")
            if order:
                all_orders.add(tuple(order))
        history_path = Path(summary["history"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            for update in history:
                for order_text in update.get(
                        "rollout_curriculum_phase_orders", {}):
                    if order_text and order_text != "FIXED":
                        all_orders.add(tuple(order_text.split("->")))
        run_plan_path = path.parent / "run_plan.json"
        run_plan = (
            json.loads(run_plan_path.read_text(encoding="utf-8"))
            if run_plan_path.exists() else {})
        rows.append({
            "seed_index": seed_index,
            "summary": str(path),
            "passed": bool(summary.get("passed")),
            "updates": int(summary.get("updates", 0)),
            "episodes_trained": int(
                summary.get("episodes_trained_this_run", 0)),
            "parameter_delta_l2": float(summary["parameter_delta_l2"]),
            "task_count": int(evaluation["task_count"]),
            "completed": int(evaluation["completed"]),
            "failed": int(evaluation["failed"]),
            "success_rate": float(evaluation["success_rate"]),
            "mean_reward": float(evaluation["mean_reward"]),
            "throughput_tasks_per_hour": float(
                evaluation["throughput_tasks_per_hour"]),
            "illegal_action_count": int(evaluation["illegal_action_count"]),
            "resource_leak_count": int(evaluation["resource_leak_count"]),
            "maximum_active_tasks": int(evaluation["maximum_active_tasks"]),
            "transport_modes": evaluation["transport_modes"],
            "transport_modes_by_phase": evaluation[
                "transport_modes_by_phase"],
            "resume_audit": run_plan.get("resume_audit"),
            "all_assertions_passed": bool(assertions) and all(
                assertions.values()),
        })

    required_modes = {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}
    acceptance = {
        "three_independent_seeds_present": len(rows) == 3,
        "every_seed_passed": all(row["passed"] for row in rows),
        "every_seed_accounts_for_320_evaluation_tasks": all(
            row["task_count"] == 320 for row in rows),
        "every_seed_has_no_illegal_action": all(
            row["illegal_action_count"] == 0 for row in rows),
        "every_seed_has_no_resource_leak": all(
            row["resource_leak_count"] == 0 for row in rows),
        "every_seed_exposes_all_three_modes": all(
            required_modes <= set(row["transport_modes"]) for row in rows),
        "every_seed_observes_concurrency": all(
            row["maximum_active_tasks"] >= 2 for row in rows),
        "every_seed_updates_parameters": all(
            row["parameter_delta_l2"] > 0.0 for row in rows),
        "every_internal_assertion_passed": all(
            row["all_assertions_passed"] for row in rows),
        "multiple_phase_orders_observed": len(all_orders) >= 2,
        "checkpoint_resume_exact_rng_state_verified": any(
            (row["resume_audit"] or {}).get("exact_rng_state_available")
            for row in rows),
    }
    report = {
        "schema_version": "warehouse_stage20_short_gate_summary_v1",
        "stage": 20,
        "gate": "MIXED_CURRICULUM_THREE_SEED_SHORT_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Interface, persistence, schedule diversity, safety, and PPO "
            "update smoke only; this is not convergence or superiority evidence."
        ),
        "seed_count": len(rows),
        "distinct_phase_orders_observed": len(all_orders),
        "aggregate": {
            "mean_success_rate": statistics.fmean(
                row["success_rate"] for row in rows),
            "mean_throughput_tasks_per_hour": statistics.fmean(
                row["throughput_tasks_per_hour"] for row in rows),
            "mean_raw_episode_reward": statistics.fmean(
                row["mean_reward"] for row in rows),
            "total_evaluation_tasks": sum(
                row["task_count"] for row in rows),
            "total_completed": sum(row["completed"] for row in rows),
            "total_failed": sum(row["failed"] for row in rows),
        },
        "runs": rows,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    output = (
        WORK_ROOT / "results" / "stage20_mixed_curriculum" / "short_gate" /
        "stage20_short_gate.summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "aggregate": report["aggregate"],
        "acceptance": acceptance,
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
