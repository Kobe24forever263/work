#!/usr/bin/env python3
"""Manually run Full Context on DENSE for ten independent seeds."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage18_train_one.py"
GATE = WORK_ROOT / "scripts" / "run_stage18_dense_full_continuation_gate.py"
STATUS = (WORK_ROOT / "results" / "stage18_dense" /
          "stage18_dense_full_ten_seed_batch.status.json")


def summary_path(seed: int) -> Path:
    return (WORK_ROOT / "results" / "stage18_dense" / "full_context_v2" /
            f"seed_{seed:02d}" / "stage18_ppo.summary.json")


def valid(seed: int) -> bool:
    path = summary_path(seed)
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(summary.get("passed") and summary.get("updates") == 2000 and
                summary.get("episodes_trained_total") == 8000 and
                summary.get("arrival_profile") == "DENSE")


def save(rows: list[dict], state: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": 18, "batch": "DENSE_FULL_CONTEXT_TEN_SEEDS",
        "arrival_profile": "DENSE", "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": 10, "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    save(rows, "RUNNING")
    for seed in range(1, 11):
        if valid(seed):
            rows.append({"seed_index": seed, "state": "SKIPPED_ALREADY_COMPLETE",
                         "summary": str(summary_path(seed))})
            save(rows, "RUNNING")
            continue
        row = {"seed_index": seed, "state": "RUNNING",
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
        rows.append(row); save(rows, "RUNNING")
        try:
            subprocess.run([
                sys.executable, str(TRAIN_ONE), "full_context_v2", str(seed),
                "--arrival-profile", "DENSE", "--phase", "all",
                "--updates", "2000", "--resume-latest", "--execute"],
                cwd=WORK_ROOT, check=True)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"; row["error"] = repr(error)
            save(rows, "STOPPED_ON_FAILURE"); raise
        if not valid(seed):
            row["state"] = "FAILED_VALIDATION"; save(rows, "STOPPED_ON_FAILURE")
            raise RuntimeError(f"DENSE Full seed {seed} failed validation")
        row.update(state="COMPLETED", summary=str(summary_path(seed)),
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(rows, "RUNNING")
    subprocess.run([sys.executable, str(GATE)], cwd=WORK_ROOT, check=True)
    save(rows, "COMPLETED")
    print(json.dumps({"status": str(STATUS), "state": "COMPLETED"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
