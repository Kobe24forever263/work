#!/usr/bin/env python3
"""Plan or manually launch a guarded Stage 23 formal-training seed range."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_stage23_markov_continuous.py"
RESULT_ROOT = ROOT / "results" / "stage23_markov_continuous" / "long"
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage23_markov_continuous.yaml")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def accepted(summary: dict | None, *, protocol: dict,
             seed_index: int) -> bool:
    if summary is None:
        return False
    method = protocol["method"]
    training = protocol["training"]
    validation = protocol["validation"]
    run_seed = int(training["independent_run_base_seeds"][seed_index - 1])
    assertions = summary.get("acceptance_assertions", {})
    evaluation = summary.get("evaluation", {})
    return bool(
        summary.get("passed") and assertions and all(assertions.values()) and
        summary.get("experiment_run_seed") == run_seed and
        summary.get("observation_variant") == method["observation_variant"] and
        summary.get("policy_variant") == method["policy_variant"] and
        summary.get("discount_mode") == method["discount_mode"] and
        summary.get("trace_mode") == method["trace_mode"] and
        summary.get("trace_tau_s") == float(method["trace_tau_s"]) and
        summary.get("reward_contract") == method["reward_contract"] and
        summary.get("updates") == int(training["formal_updates"]) and
        summary.get("episodes_trained_total") ==
        int(training["episodes_per_run"]) and
        summary.get("validation_seed_start") ==
        int(validation["seed_start"]) and
        summary.get("validation_seed_end") ==
        int(validation["seed_start"]) + int(validation["seed_count"]) - 1 and
        evaluation.get("episode_count") == int(validation["seed_count"]) and
        evaluation.get("scheduled_task_count") ==
        int(validation["seed_count"]) * int(training["tasks_per_episode"]) and
        summary.get("warm_start_source") is not None)


def save(path: Path, report: dict) -> None:
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-long", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_start <= args.seed_end <= 10:
        parser.error("seed range must satisfy 1 <= start <= end <= 10")
    if args.execute and not args.confirm_long:
        parser.error("formal execution requires --confirm-long")

    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    jobs = []
    for seed_index in range(args.seed_start, args.seed_end + 1):
        run_dir = RESULT_ROOT / f"seed_{seed_index:02d}"
        summary_path = run_dir / "stage23_ppo.summary.json"
        summary = load_json(summary_path)
        checkpoints = sorted(
            (run_dir / "checkpoints").glob("stage23_ppo_update_*.pt"))
        if accepted(summary, protocol=protocol, seed_index=seed_index):
            state = "SKIP_COMPLETE_ACCEPTED"
        elif checkpoints:
            state = "RESUME_CHECKPOINT"
        else:
            state = "START_NEW"
        jobs.append({
            "seed_index": seed_index,
            "status": state,
            "summary": str(summary_path),
            "resume_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        })

    training = protocol["training"]
    report = {
        "stage": 23,
        "gate": "MARKOV_CONTINUOUS_FORMAL_TRAINING_CAMPAIGN",
        "protocol": protocol["protocol"]["version"],
        "claim_boundary": (
            "Training only. Locked test seeds remain forbidden until all "
            "ten terminal weights are frozen."),
        "seed_range": [args.seed_start, args.seed_end],
        "jobs": jobs,
        "job_count": len(jobs),
        "start_new_count": sum(
            row["status"] == "START_NEW" for row in jobs),
        "resume_count": sum(
            row["status"] == "RESUME_CHECKPOINT" for row in jobs),
        "skip_complete_count": sum(
            row["status"] == "SKIP_COMPLETE_ACCEPTED" for row in jobs),
        "updates_per_run": training["formal_updates"],
        "episodes_per_run": training["episodes_per_run"],
        "evaluation_every_updates": training["evaluation_every_updates"],
        "checkpoint_every_updates": training["checkpoint_every_updates"],
        "execution_requested": args.execute,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    status_path = RESULT_ROOT / (
        f"stage23_campaign_{args.seed_start:02d}_{args.seed_end:02d}.status.json")
    report["state"] = "RUNNING"
    save(status_path, report)
    for ordinal, job in enumerate(jobs, start=1):
        if job["status"] == "SKIP_COMPLETE_ACCEPTED":
            print(
                f"[{ordinal}/{len(jobs)}] skip accepted "
                f"seed={job['seed_index']:02d}", flush=True)
            continue
        command = [
            sys.executable, str(RUNNER), str(job["seed_index"]),
            "--execute", "--confirm-long",
        ]
        if job["status"] == "RESUME_CHECKPOINT":
            command.append("--resume-latest")
        print(
            f"[{ordinal}/{len(jobs)}] {job['status'].lower()} "
            f"seed={job['seed_index']:02d}", flush=True)
        job["status"] = "RUNNING"
        job["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(status_path, report)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException as error:
            job["status"] = "FAILED_OR_INTERRUPTED"
            job["error"] = repr(error)
            report["state"] = "STOPPED_ON_FAILURE"
            save(status_path, report)
            raise
        summary = load_json(Path(job["summary"]))
        if not accepted(
                summary, protocol=protocol,
                seed_index=int(job["seed_index"])):
            job["status"] = "FAILED_OR_INCOMPLETE"
            report["state"] = "STOPPED_ON_INCOMPLETE_RUN"
            save(status_path, report)
            raise RuntimeError(
                f"incomplete Stage 23 seed {job['seed_index']:02d}")
        job["status"] = "COMPLETED_ACCEPTED"
        job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(status_path, report)

    report["state"] = "COMPLETED"
    save(status_path, report)
    print(json.dumps({
        "status": str(status_path),
        "state": report["state"],
        "accepted": len(jobs),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
