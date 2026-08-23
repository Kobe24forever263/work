#!/usr/bin/env python3
"""Manually launch three context ablations with ten seeds each."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage18_train_one.py"
CONDITIONS = ("no_queue_resource", "no_handover_cues",
              "no_persistent_position")


def summary_path(profile_root: str, condition: str, seed: int) -> Path:
    return (WORK_ROOT / "results" / profile_root / condition /
            f"seed_{seed:02d}" / "stage18_ppo.summary.json")


def completion_state(profile_root: str, condition: str, seed: int) -> str | None:
    path = summary_path(profile_root, condition, seed)
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    complete_budget = (
        summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000)
    if not complete_budget:
        return None
    if summary.get("passed"):
        return "PASSED"
    if condition == "no_persistent_position":
        assertions = summary.get("assertions", {})
        required = {
            key: value for key, value in assertions.items()
            if key != "formal_policy_switches_modes_by_context"}
        if (required and all(required.values()) and
                assertions.get("formal_policy_switches_modes_by_context") is False):
            return "EXPECTED_INFORMATION_ABLATION_DEGRADATION"
    return None


def save_status(status: Path, condition: str, arrival_profile: str,
                rows: list[dict], state: str) -> None:
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({
        "stage": 18,
        "batch": "ONE_CONTEXT_ABLATION_TEN_SEEDS",
        "condition": condition,
        "arrival_profile": arrival_profile,
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": 10,
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--arrival-profile", choices=(
        "MEDIUM", "DENSE", "BURST"),
                        default="MEDIUM")
    args = parser.parse_args()
    profile_root = {
        "MEDIUM": "stage18",
        "DENSE": "stage18_dense",
        "BURST": "stage18_burst",
    }[args.arrival_profile]
    runs = [(args.condition, seed) for seed in range(1, 11)]
    status = (WORK_ROOT / "results" / profile_root /
              f"{profile_root}_{args.condition}_ten_seed_batch.status.json")
    rows = []
    save_status(status, args.condition, args.arrival_profile, rows, "RUNNING")
    for condition, seed in runs:
        prior_state = completion_state(profile_root, condition, seed)
        if prior_state is not None:
            rows.append({
                "condition": condition, "seed_index": seed,
                "state": ("SKIPPED_ALREADY_COMPLETE" if prior_state == "PASSED"
                          else "SKIPPED_COMPLETE_WITH_EXPECTED_DEGRADATION"),
                         "summary": str(summary_path(profile_root, condition, seed))})
            save_status(status, args.condition, args.arrival_profile, rows, "RUNNING")
            continue
        row = {
            "condition": condition, "seed_index": seed,
            "state": "RUNNING",
            "started_at_utc": datetime.now(timezone.utc).isoformat()}
        rows.append(row)
        save_status(status, args.condition, args.arrival_profile, rows, "RUNNING")
        try:
            subprocess.run([
                sys.executable, str(TRAIN_ONE), condition, str(seed),
                "--arrival-profile", args.arrival_profile,
                "--phase", "all", "--updates", "2000",
                "--resume-latest", "--execute"],
                cwd=WORK_ROOT, check=True)
        except BaseException as error:
            completed_state = completion_state(profile_root, condition, seed)
            if completed_state == "EXPECTED_INFORMATION_ABLATION_DEGRADATION":
                row["state"] = "COMPLETED_WITH_EXPECTED_DEGRADATION"
                row["summary"] = str(summary_path(
                    profile_root, condition, seed))
                row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                save_status(status, args.condition, args.arrival_profile,
                            rows, "RUNNING")
                continue
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save_status(status, args.condition, args.arrival_profile, rows, "STOPPED_ON_FAILURE")
            raise
        completed_state = completion_state(profile_root, condition, seed)
        if completed_state is None:
            row["state"] = "FAILED_VALIDATION"
            save_status(status, args.condition, args.arrival_profile, rows, "STOPPED_ON_FAILURE")
            raise RuntimeError(f"completed run failed validation: {condition}/{seed}")
        row["state"] = ("COMPLETED" if completed_state == "PASSED"
                        else "COMPLETED_WITH_EXPECTED_DEGRADATION")
        row["summary"] = str(summary_path(profile_root, condition, seed))
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(status, args.condition, args.arrival_profile, rows, "RUNNING")
    save_status(status, args.condition, args.arrival_profile, rows, "COMPLETED")
    print(json.dumps({
        "status": str(status), "run_count": len(runs),
        "state": "COMPLETED"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
