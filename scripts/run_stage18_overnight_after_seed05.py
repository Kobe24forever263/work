#!/usr/bin/env python3
"""Wait for Full seed 05, then run Full 06-10 and one ten-seed ablation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


WORK_ROOT = Path(__file__).resolve().parents[1]
SEED05_SUMMARY = (WORK_ROOT / "results" / "stage18" / "full_context_v2" /
                  "seed_05" / "stage18_ppo.summary.json")
FULL_BATCH = WORK_ROOT / "scripts" / "run_stage18_full_seed_06_10_batch.py"
ABLATION_BATCH = WORK_ROOT / "scripts" / "run_stage18_context_ablation_batch.py"
STATUS = (WORK_ROOT / "results" / "stage18" /
          "stage18_overnight_after_seed05.status.json")


def seed05_complete() -> bool:
    if not SEED05_SUMMARY.exists():
        return False
    try:
        summary = json.loads(SEED05_SUMMARY.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(summary.get("passed") and summary.get("updates") == 2000 and
                summary.get("episodes_trained_total") == 8000)


def save(state: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": 18, "batch": "OVERNIGHT_AFTER_FULL_SEED_05",
               "state": state,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               **extra}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


def main() -> int:
    save("WAITING_FOR_FULL_SEED_05")
    deadline = time.monotonic() + 2 * 60 * 60
    while not seed05_complete():
        if time.monotonic() >= deadline:
            save("STOPPED_SEED_05_TIMEOUT")
            raise TimeoutError("Full seed 05 did not complete within two hours")
        time.sleep(30)
    save("RUNNING_FULL_SEEDS_06_TO_10")
    subprocess.run([sys.executable, str(FULL_BATCH)], cwd=WORK_ROOT, check=True)
    save("RUNNING_NO_QUEUE_RESOURCE_TEN_SEEDS")
    subprocess.run([sys.executable, str(ABLATION_BATCH), "no_queue_resource"],
                   cwd=WORK_ROOT, check=True)
    save("COMPLETED")
    print(json.dumps({"status": str(STATUS), "state": "COMPLETED"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
