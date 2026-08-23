#!/usr/bin/env python3
"""Parallel lane A: repair histories, run DENSE ablations, then test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


WORK_ROOT = Path(__file__).resolve().parents[1]
REPAIR = WORK_ROOT / "scripts" / "run_stage18_history_repair.py"
ABLATION = WORK_ROOT / "scripts" / "run_stage18_context_ablation_batch.py"
LOCKED_TEST = WORK_ROOT / "scripts" / "run_stage18_locked_test_campaign.py"
STATUS = WORK_ROOT / "results" / "stage18_dense_lane.status.json"
BURST_STATUS = WORK_ROOT / "results" / "stage18_burst_lane.status.json"
CONDITIONS = (
    "no_queue_resource", "no_handover_cues", "no_persistent_position")


def save(rows: list[dict], state: str) -> None:
    STATUS.write_text(json.dumps({
        "stage": 18, "lane": "A_HISTORY_REPAIR_AND_DENSE_ABLATIONS",
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


def burst_complete() -> bool:
    if not BURST_STATUS.exists():
        return False
    try:
        return json.loads(BURST_STATUS.read_text(
            encoding="utf-8")).get("state") == "COMPLETED"
    except (json.JSONDecodeError, OSError):
        return False


def main() -> int:
    rows: list[dict] = []
    save(rows, "RUNNING")
    run(rows, "HISTORY_AND_RNG_REPAIR", [sys.executable, str(REPAIR)])
    for condition in CONDITIONS:
        run(rows, f"DENSE_{condition}", [
            sys.executable, str(ABLATION), condition,
            "--arrival-profile", "DENSE"])
    wait_row = {
        "name": "WAIT_FOR_BURST_LANE", "state": "WAITING",
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    rows.append(wait_row); save(rows, "WAITING_FOR_BURST")
    while not burst_complete():
        time.sleep(60)
    wait_row["state"] = "COMPLETED"
    wait_row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(rows, "RUNNING")
    run(rows, "LOCKED_TEST_AND_STATISTICS", [
        sys.executable, str(LOCKED_TEST)])
    save(rows, "COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
