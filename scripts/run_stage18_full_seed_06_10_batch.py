#!/usr/bin/env python3
"""Manually launch Full Context seeds 06-10, then run the ten-seed gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage18_train_one.py"
FULL_GATE = WORK_ROOT / "scripts" / "run_stage18_full_continuation_gate.py"
STATUS = (WORK_ROOT / "results" / "stage18" /
          "stage18_full_seed_06_10_batch.status.json")
RUNS = [("full_context_v2", seed) for seed in range(6, 11)]


def summary_path(seed: int) -> Path:
    return (WORK_ROOT / "results" / "stage18" / "full_context_v2" /
            f"seed_{seed:02d}" / "stage18_ppo.summary.json")


def valid_completed_run(seed: int) -> bool:
    path = summary_path(seed)
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(summary.get("passed") and summary.get("updates") == 2000 and
                summary.get("episodes_trained_total") == 8000)


def save_status(rows: list[dict], state: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": 18,
        "batch": "FULL_CONTEXT_SEEDS_06_TO_10",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": len(RUNS),
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    save_status(rows, "RUNNING")
    for condition, seed in RUNS:
        if valid_completed_run(seed):
            rows.append({"condition": condition, "seed_index": seed,
                         "state": "SKIPPED_ALREADY_COMPLETE",
                         "summary": str(summary_path(seed))})
            save_status(rows, "RUNNING")
            continue
        row = {"condition": condition, "seed_index": seed, "state": "RUNNING",
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
        rows.append(row)
        save_status(rows, "RUNNING")
        try:
            subprocess.run([
                sys.executable, str(TRAIN_ONE), condition, str(seed),
                "--phase", "all", "--updates", "2000", "--execute"],
                cwd=WORK_ROOT, check=True)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save_status(rows, "STOPPED_ON_FAILURE")
            raise
        if not valid_completed_run(seed):
            row["state"] = "FAILED_VALIDATION"
            save_status(rows, "STOPPED_ON_FAILURE")
            raise RuntimeError(f"completed run failed validation: seed {seed}")
        row["state"] = "COMPLETED"
        row["summary"] = str(summary_path(seed))
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(rows, "RUNNING")
    subprocess.run([sys.executable, str(FULL_GATE)], cwd=WORK_ROOT, check=True)
    save_status(rows, "COMPLETED")
    print(json.dumps({"status": str(STATUS), "run_count": len(RUNS),
                      "state": "COMPLETED"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
