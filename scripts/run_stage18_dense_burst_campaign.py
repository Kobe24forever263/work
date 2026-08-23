#!/usr/bin/env python3
"""Run requested DENSE ablations, then BURST Full and ablations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
ABLATION = WORK_ROOT / "scripts" / "run_stage18_context_ablation_batch.py"
FULL = WORK_ROOT / "scripts" / "run_stage18_profile_full_ten_seed_batch.py"
HISTORY_REPAIR = WORK_ROOT / "scripts" / "run_stage18_history_repair.py"
LOCKED_TEST = WORK_ROOT / "scripts" / "run_stage18_locked_test_campaign.py"
STATUS = (WORK_ROOT / "results" / "stage18_dense_burst_campaign.status.json")
CONDITIONS = (
    "no_queue_resource", "no_handover_cues", "no_persistent_position")


def phases() -> list[tuple[str, list[str]]]:
    rows = [("DATA_LIMIT_HISTORY_AND_RNG_REPAIR", [
        sys.executable, str(HISTORY_REPAIR)])]
    for condition in CONDITIONS:
        rows.append((f"DENSE_{condition}", [
            sys.executable, str(ABLATION), condition,
            "--arrival-profile", "DENSE"]))
    rows.append(("BURST_full_context_v2", [
        sys.executable, str(FULL), "--arrival-profile", "BURST"]))
    for condition in CONDITIONS:
        rows.append((f"BURST_{condition}", [
            sys.executable, str(ABLATION), condition,
            "--arrival-profile", "BURST"]))
    rows.append(("LOCKED_TEST_AND_CROSS_LOAD_GENERALIZATION", [
        sys.executable, str(LOCKED_TEST)]))
    return rows


def save(rows: list[dict], state: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": 18,
        "campaign": "DENSE_ABLATIONS_THEN_BURST_FULL_AND_ABLATIONS",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "SERIAL_RESUMABLE",
        "runs_per_phase": 10,
        "phases": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    save(rows, "RUNNING")
    for name, command in phases():
        row = {
            "name": name, "state": "RUNNING",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command}
        rows.append(row)
        save(rows, "RUNNING")
        try:
            subprocess.run(command, cwd=WORK_ROOT, check=True)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save(rows, "STOPPED_ON_FAILURE")
            raise
        row["state"] = "COMPLETED"
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(rows, "RUNNING")
    save(rows, "COMPLETED")
    print(json.dumps({"status": str(STATUS), "state": "COMPLETED"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
