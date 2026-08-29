#!/usr/bin/env python3
"""Run one paired short training gate for every Stage 22 condition."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_stage22_algorithm_ablation.py"
OUT = ROOT / "results" / "stage22_algorithm_ablation" / "short_gate"
CONDITIONS = (
    "fixed_gamma", "no_context_interaction", "no_warm_start", "no_relay")


def main() -> int:
    rows = []
    for condition in CONDITIONS:
        subprocess.run([
            sys.executable, str(RUNNER), condition, "1",
            "--short-gate", "--execute"], cwd=ROOT, check=True)
        summary_path = (
            OUT / condition / "seed_01" / "stage22_ppo.summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "condition": condition,
            "summary": str(summary_path),
            "passed": summary["passed"],
            "discount_mode": summary["discount_mode"],
            "policy_variant": summary["policy_variant"],
            "allowed_transport_modes": summary["allowed_transport_modes"],
            "warm_start_source": summary["warm_start_source"],
            "illegal_action_count": summary["evaluation"][
                "illegal_action_count"],
            "resource_leak_count": summary["evaluation"][
                "resource_leak_count"],
        })
    by_condition = {row["condition"]: row for row in rows}
    assertions = {
        "all_four_short_runs_completed": len(rows) == 4,
        "all_short_safety_gates_passed": all(row["passed"] for row in rows),
        "no_illegal_actions": all(
            row["illegal_action_count"] == 0 for row in rows),
        "no_resource_leaks": all(
            row["resource_leak_count"] == 0 for row in rows),
        "fixed_gamma_summary_is_fixed": (
            by_condition["fixed_gamma"]["discount_mode"] ==
            "FIXED_PER_DECISION"),
        "no_context_summary_is_flat": (
            by_condition["no_context_interaction"]["policy_variant"] ==
            "FLAT_MASKED_PPO"),
        "no_warm_start_summary_has_no_source": (
            by_condition["no_warm_start"]["warm_start_source"] is None),
        "no_relay_summary_has_two_modes": (
            by_condition["no_relay"]["allowed_transport_modes"] ==
            ["SINGLE_CAR", "SINGLE_DOG"]),
    }
    report = {
        "stage": 22,
        "gate": "ALGORITHM_ABLATION_SHORT_TRAINING_V1",
        "claim_boundary": "Two-update interface gate; no performance claim.",
        "conditions": rows,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = OUT / "stage22_algorithm_ablation_smokes.summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output), "passed": report["passed"]},
        ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

