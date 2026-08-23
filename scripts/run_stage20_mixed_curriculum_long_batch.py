#!/usr/bin/env python3
"""Run or resume the ten formal Stage 20 mixed-curriculum training seeds.

This launcher is intentionally manual and requires --confirm-long. It stops on
the first failed seed, records progress after every state change, skips already
valid runs, and asks the per-seed runner to resume from its latest checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage20_mixed_curriculum.py"
RESULT_ROOT = WORK_ROOT / "results" / "stage20_mixed_curriculum" / "long"


def summary_path(seed_index: int) -> Path:
    return RESULT_ROOT / f"seed_{seed_index:02d}" / "stage20_ppo.summary.json"


def valid(seed_index: int) -> bool:
    path = summary_path(seed_index)
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    evaluation = summary.get("evaluation", {})
    assertions = summary.get("assertions", {})
    return bool(
        summary.get("passed") and
        summary.get("arrival_profile") == "MIXED_CURRICULUM" and
        summary.get("execution_mode") == "CONCURRENT" and
        summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000 and
        evaluation.get("illegal_action_count") == 0 and
        evaluation.get("resource_leak_count") == 0 and
        assertions and all(assertions.values())
    )


def save(status: Path, rows: list[dict], state: str,
         seed_start: int, seed_end: int) -> None:
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({
        "schema_version": "warehouse_stage20_long_batch_status_v1",
        "stage": 20,
        "batch": "MIXED_CURRICULUM_TEN_SEED_FORMAL_TRAINING",
        "arrival_profile": "MIXED_CURRICULUM",
        "execution_mode": "CONCURRENT",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_start": seed_start,
        "seed_end": seed_end,
        "planned_run_count": seed_end - seed_start + 1,
        "formal_updates_per_seed": 2000,
        "episodes_per_seed": 8000,
        "claim_boundary": (
            "Training only; fresh locked-test seeds are required after all "
            "ten weights are frozen."
        ),
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_start <= args.seed_end <= 10:
        parser.error("seed range must satisfy 1 <= start <= end <= 10")
    if not args.confirm_long:
        parser.error(
            "formal long training is manual; add --confirm-long to start")

    status = (
        WORK_ROOT / "results" / "stage20_mixed_curriculum" /
        "stage20_long_batch.status.json"
    )
    rows: list[dict] = []
    save(status, rows, "RUNNING", args.seed_start, args.seed_end)
    for seed_index in range(args.seed_start, args.seed_end + 1):
        if valid(seed_index):
            rows.append({
                "seed_index": seed_index,
                "state": "SKIPPED_ALREADY_COMPLETE",
                "summary": str(summary_path(seed_index)),
            })
            save(status, rows, "RUNNING", args.seed_start, args.seed_end)
            continue

        row = {
            "seed_index": seed_index,
            "state": "RUNNING",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        save(status, rows, "RUNNING", args.seed_start, args.seed_end)
        print(f"[Stage 20] seed {seed_index}/{args.seed_end} starting", flush=True)
        try:
            subprocess.run([
                sys.executable, str(TRAIN_ONE), str(seed_index),
                "--updates", "2000", "--resume-latest", "--execute",
                "--confirm-long",
            ], cwd=WORK_ROOT, check=True)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save(status, rows, "STOPPED_ON_FAILURE",
                 args.seed_start, args.seed_end)
            raise
        if not valid(seed_index):
            row["state"] = "FAILED_VALIDATION"
            save(status, rows, "STOPPED_ON_FAILURE",
                 args.seed_start, args.seed_end)
            raise RuntimeError(
                f"Stage 20 mixed-curriculum seed {seed_index} failed validation")
        row.update({
            "state": "COMPLETED",
            "summary": str(summary_path(seed_index)),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save(status, rows, "RUNNING", args.seed_start, args.seed_end)
        print(f"[Stage 20] seed {seed_index}/{args.seed_end} completed", flush=True)

    save(status, rows, "COMPLETED", args.seed_start, args.seed_end)
    print(json.dumps({
        "status": str(status),
        "state": "COMPLETED",
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
