#!/usr/bin/env python3
"""Replace two resumed Stage 18 runs with complete from-update-1 reruns."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ONE = WORK_ROOT / "scripts" / "run_stage18_train_one.py"
ARCHIVE_ROOT = WORK_ROOT / "results" / "stage18_repair_archive"
STATUS = WORK_ROOT / "results" / "stage18_history_repair.status.json"
CASES = (
    {
        "profile": "MEDIUM", "profile_root": "stage18",
        "condition": "no_persistent_position", "seed": 2,
        "expected_degradation": True,
    },
    {
        "profile": "DENSE", "profile_root": "stage18_dense",
        "condition": "full_context_v2", "seed": 2,
        "expected_degradation": False,
    },
)


def run_dir(case: dict) -> Path:
    return (WORK_ROOT / "results" / case["profile_root"] /
            case["condition"] / f"seed_{case['seed']:02d}")


def archive_dir(case: dict) -> Path:
    return (ARCHIVE_ROOT / case["profile_root"] / case["condition"] /
            f"seed_{case['seed']:02d}_resumed_legacy")


def completion_audit(case: dict) -> dict:
    directory = run_dir(case)
    history_path = directory / "stage18_ppo.history.json"
    summary_path = directory / "stage18_ppo.summary.json"
    final_path = directory / "stage18_ppo.pt"
    if not all(path.exists() for path in (
            history_path, summary_path, final_path)):
        return {"complete": False, "reason": "missing_outputs"}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    assertions = summary.get("assertions", {})
    safety = all((
        assertions.get("evaluation_actions_all_legal") is True,
        assertions.get("evaluation_has_no_resource_leak") is True,
        assertions.get("rule_baseline_has_no_resource_leak") is True,
    ))
    quality = bool(summary.get("passed"))
    if case["expected_degradation"]:
        quality = bool(
            assertions and safety and
            assertions.get("formal_policy_switches_modes_by_context") is False and
            all(value for key, value in assertions.items()
                if key != "formal_policy_switches_modes_by_context"))
    complete = bool(
        len(history) == 2000 and history[0].get("update") == 1 and
        history[-1].get("update") == 2000 and
        summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000 and safety and quality and
        all(checkpoint.get(key) is not None for key in (
            "numpy_random_state", "torch_random_state",
            "ppo_generator_state")))
    return {
        "complete": complete,
        "history_length": len(history),
        "history_first_update": history[0].get("update") if history else None,
        "history_last_update": history[-1].get("update") if history else None,
        "episodes_trained_total": summary.get("episodes_trained_total"),
        "resumed_from_update": summary.get("resumed_from_update"),
        "resume_classification": (
            "EXACT_NEW_LINEAGE_CHECKPOINT_RESUME"
            if summary.get("resumed_from_update") is not None and
            len(history) == 2000 and history[0].get("update") == 1 and
            all(checkpoint.get(key) is not None for key in (
                "numpy_random_state", "torch_random_state",
                "ppo_generator_state")) else
            "UNINTERRUPTED_FROM_UPDATE_1"
            if summary.get("resumed_from_update") is None else
            "INCOMPLETE_OR_LEGACY_RESUME"),
        "safety_passed": safety,
        "quality_passed_or_expected_degradation": quality,
        "rng_state_complete": all(checkpoint.get(key) is not None for key in (
            "numpy_random_state", "torch_random_state",
            "ppo_generator_state")),
        "summary": str(summary_path),
    }


def save(rows: list[dict], state: str) -> None:
    STATUS.write_text(json.dumps({
        "stage": 18,
        "repair": "COMPLETE_HISTORY_AND_RNG_FROM_UPDATE_1",
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "archive_resumed_run_then_rerun_same_seed_from_scratch",
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    save(rows, "RUNNING")
    for case in CASES:
        audit = completion_audit(case)
        row = {**case, "pre_audit": audit, "state": "RUNNING"}
        rows.append(row)
        save(rows, "RUNNING")
        if audit.get("complete"):
            row["state"] = "SKIPPED_ALREADY_REPAIRED"
            save(rows, "RUNNING")
            continue
        source = run_dir(case)
        archive = archive_dir(case)
        if source.exists() and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive))
            row["archive"] = str(archive)
            save(rows, "RUNNING")
        elif source.exists() and archive.exists():
            # A previous repair attempt was interrupted after the legacy run
            # had already been archived. Continue the new canonical run from
            # its latest complete checkpoint.
            row["archive"] = str(archive)
            row["repair_resume"] = True
            save(rows, "RUNNING")
        command = [
            sys.executable, str(TRAIN_ONE), case["condition"],
            str(case["seed"]), "--arrival-profile", case["profile"],
            "--phase", "all", "--updates", "2000", "--resume-latest",
            "--execute"]
        try:
            subprocess.run(command, cwd=WORK_ROOT, check=True)
        except subprocess.CalledProcessError:
            # NO_PERSISTENT_POSITION is expected to finish its full budget but
            # fail only the contextual mode-switch quality assertion.
            if not case["expected_degradation"]:
                raise
        post = completion_audit(case)
        row["post_audit"] = post
        if not post.get("complete"):
            row["state"] = "FAILED_VALIDATION"
            save(rows, "STOPPED_ON_FAILURE")
            raise RuntimeError(f"history repair failed: {case}")
        row["state"] = "COMPLETED"
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(rows, "RUNNING")
    save(rows, "COMPLETED")
    print(json.dumps({"status": str(STATUS), "state": "COMPLETED"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
