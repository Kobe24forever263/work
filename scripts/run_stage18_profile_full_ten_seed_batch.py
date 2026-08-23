#!/usr/bin/env python3
"""Run ten Full Context seeds for one high-load Stage 18 profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage18_train_one.py"


def root_for(profile: str) -> str:
    return {"DENSE": "stage18_dense", "BURST": "stage18_burst"}[profile]


def summary_path(profile: str, seed: int) -> Path:
    return (WORK_ROOT / "results" / root_for(profile) / "full_context_v2" /
            f"seed_{seed:02d}" / "stage18_ppo.summary.json")


def valid(profile: str, seed: int) -> bool:
    path = summary_path(profile, seed)
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        summary.get("passed") and summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000 and
        summary.get("arrival_profile") == profile)


def save(status: Path, profile: str, rows: list[dict], state: str) -> None:
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({
        "stage": 18,
        "batch": f"{profile}_FULL_CONTEXT_TEN_SEEDS",
        "arrival_profile": profile,
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": 10,
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrival-profile", choices=("DENSE", "BURST"),
                        required=True)
    args = parser.parse_args()
    profile = args.arrival_profile
    status = (WORK_ROOT / "results" / root_for(profile) /
              f"{root_for(profile)}_full_ten_seed_batch.status.json")
    rows: list[dict] = []
    save(status, profile, rows, "RUNNING")
    for seed in range(1, 11):
        if valid(profile, seed):
            rows.append({
                "seed_index": seed, "state": "SKIPPED_ALREADY_COMPLETE",
                "summary": str(summary_path(profile, seed))})
            save(status, profile, rows, "RUNNING")
            continue
        row = {
            "seed_index": seed, "state": "RUNNING",
            "started_at_utc": datetime.now(timezone.utc).isoformat()}
        rows.append(row)
        save(status, profile, rows, "RUNNING")
        try:
            subprocess.run([
                sys.executable, str(TRAIN_ONE), "full_context_v2", str(seed),
                "--arrival-profile", profile, "--phase", "all",
                "--updates", "2000", "--resume-latest", "--execute"],
                cwd=WORK_ROOT, check=True)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save(status, profile, rows, "STOPPED_ON_FAILURE")
            raise
        if not valid(profile, seed):
            row["state"] = "FAILED_VALIDATION"
            save(status, profile, rows, "STOPPED_ON_FAILURE")
            raise RuntimeError(
                f"{profile} Full seed {seed} failed validation")
        row.update(
            state="COMPLETED", summary=str(summary_path(profile, seed)),
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(status, profile, rows, "RUNNING")
    save(status, profile, rows, "COMPLETED")
    print(json.dumps({"status": str(status), "state": "COMPLETED"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
