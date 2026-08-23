#!/usr/bin/env python3
"""Parallel lane B: run BURST Full and the three context ablations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
FULL = WORK_ROOT / "scripts" / "run_stage18_profile_full_ten_seed_batch.py"
ABLATION = WORK_ROOT / "scripts" / "run_stage18_context_ablation_batch.py"
STATUS = WORK_ROOT / "results" / "stage18_burst_lane.status.json"
CONDITIONS = (
    "no_queue_resource", "no_handover_cues", "no_persistent_position")


def save(rows: list[dict], state: str) -> None:
    STATUS.write_text(json.dumps({
        "stage": 18, "lane": "B_BURST_FULL_AND_ABLATIONS",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(rows: list[dict], name: str, command: list[str]) -> None:
    row = {"name": name, "state": "RUNNING", "command": command,
           "started_at_utc": datetime.now(timezone.utc).isoformat()}
    rows.append(row); save(rows, "RUNNING")
    try:
        subprocess.run(command, cwd=WORK_ROOT, check=True)
    except BaseException as error:
        row["state"] = "FAILED_OR_INTERRUPTED"; row["error"] = repr(error)
        save(rows, "STOPPED_ON_FAILURE"); raise
    row["state"] = "COMPLETED"
    row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(rows, "RUNNING")


def main() -> int:
    rows: list[dict] = []
    save(rows, "RUNNING")
    run(rows, "BURST_full_context_v2", [
        sys.executable, str(FULL), "--arrival-profile", "BURST"])
    for condition in CONDITIONS:
        run(rows, f"BURST_{condition}", [
            sys.executable, str(ABLATION), condition,
            "--arrival-profile", "BURST"])
    save(rows, "COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
