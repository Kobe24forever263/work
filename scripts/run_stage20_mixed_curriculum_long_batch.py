#!/usr/bin/env python3
"""Run or resume the ten formal Stage 20 mixed-curriculum training seeds.

This launcher is intentionally manual and requires --confirm-long. It records
progress after every state change, skips every already completed formal run,
retains completed gate failures as evidence, and only stops when a run is
actually incomplete or interrupted. Incomplete runs resume from their latest
checkpoint.
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


def load_summary(seed_index: int) -> dict | None:
    path = summary_path(seed_index)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def formal_run_complete(summary: dict | None) -> bool:
    if summary is None:
        return False
    evaluation = summary.get("evaluation", {})
    return bool(
        summary.get("arrival_profile") == "MIXED_CURRICULUM" and
        summary.get("execution_mode") == "CONCURRENT" and
        summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000 and
        summary.get("validation_seed_start") == 62000000 and
        summary.get("validation_seed_end") == 62000099 and
        evaluation.get("episode_count") == 100 and
        evaluation.get("scheduled_task_count") == 8000
    )


def accepted(summary: dict) -> bool:
    assertions = summary.get("assertions", {})
    return bool(summary.get("passed") and assertions and
                all(assertions.values()))


def result_row(seed_index: int, summary: dict, *, skipped: bool) -> dict:
    assertions = summary.get("assertions", {})
    evaluation = summary.get("evaluation", {})
    gate_passed = accepted(summary)
    return {
        "seed_index": seed_index,
        "state": (
            "SKIPPED_ALREADY_ACCEPTED" if skipped and gate_passed else
            "SKIPPED_COMPLETED_NOT_ACCEPTED" if skipped else
            "COMPLETED_ACCEPTED" if gate_passed else
            "COMPLETED_NOT_ACCEPTED"),
        "accepted": gate_passed,
        "failed_assertions": sorted(
            key for key, value in assertions.items() if not value),
        "summary": str(summary_path(seed_index)),
        "success_rate": evaluation.get("success_rate"),
        "illegal_action_count": evaluation.get("illegal_action_count"),
        "resource_leak_count": evaluation.get("resource_leak_count"),
    }


def save(status: Path, rows: list[dict], state: str,
         seed_start: int, seed_end: int) -> None:
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({
        "schema_version": "warehouse_stage20_long_batch_status_v2",
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
        f"stage20_long_batch_{args.seed_start:02d}_{args.seed_end:02d}.status.json"
    )
    rows: list[dict] = []
    save(status, rows, "RUNNING", args.seed_start, args.seed_end)
    for seed_index in range(args.seed_start, args.seed_end + 1):
        existing = load_summary(seed_index)
        if formal_run_complete(existing):
            row = result_row(seed_index, existing, skipped=True)
            rows.append(row)
            save(status, rows, "RUNNING", args.seed_start, args.seed_end)
            print(
                f"[Stage 20] seed {seed_index}/{args.seed_end} "
                f"{row['state'].lower()}", flush=True)
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
            process = subprocess.run([
                sys.executable, str(TRAIN_ONE), str(seed_index),
                "--updates", "2000", "--resume-latest", "--execute",
                "--confirm-long",
            ], cwd=WORK_ROOT, check=False)
        except BaseException as error:
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["error"] = repr(error)
            save(status, rows, "STOPPED_ON_FAILURE",
                 args.seed_start, args.seed_end)
            raise
        completed = load_summary(seed_index)
        if not formal_run_complete(completed):
            row["state"] = "FAILED_OR_INTERRUPTED"
            row["return_code"] = process.returncode
            row["error"] = "formal summary is missing or incomplete"
            save(status, rows, "STOPPED_ON_FAILURE",
                 args.seed_start, args.seed_end)
            raise RuntimeError(
                f"Stage 20 mixed-curriculum seed {seed_index} is incomplete")
        row.update(result_row(seed_index, completed, skipped=False))
        row["return_code"] = process.returncode
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(status, rows, "RUNNING", args.seed_start, args.seed_end)
        print(
            f"[Stage 20] seed {seed_index}/{args.seed_end} "
            f"{row['state'].lower()}", flush=True)

    final_state = (
        "COMPLETED" if all(row.get("accepted") for row in rows) else
        "COMPLETED_WITH_GATE_FAILURES")
    save(status, rows, final_state, args.seed_start, args.seed_end)
    print(json.dumps({
        "status": str(status),
        "state": final_state,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "completed_not_accepted": sum(
            not bool(row.get("accepted")) for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
