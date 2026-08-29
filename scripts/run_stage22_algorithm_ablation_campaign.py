#!/usr/bin/env python3
"""Plan or manually launch the frozen Stage 22 ablation campaign.

Formal training is intentionally guarded by both ``--execute`` and
``--confirm-long``.  Completed seeds are skipped, while an incomplete seed
with a compatible checkpoint is resumed through the single-run validator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_stage22_algorithm_ablation.py"
RESULT_ROOT = ROOT / "results" / "stage22_algorithm_ablation" / "long"
CONFIG = (ROOT / "src" / "warehouse_bringup" / "config" /
          "experiment_seeds_stage22_algorithm_ablation.yaml")
CONDITIONS = (
    "fixed_gamma",
    "no_context_interaction",
    "no_warm_start",
    "no_relay",
)


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def formal_run_complete(summary: dict | None, *, condition: dict,
                        run_seed: int) -> bool:
    if summary is None:
        return False
    evaluation = summary.get("evaluation", {})
    expected_modes = sorted(condition["allowed_transport_modes"])
    return bool(
        summary.get("arrival_profile") == "MIXED_CURRICULUM" and
        summary.get("execution_mode") == "CONCURRENT" and
        summary.get("observation_variant") ==
        condition["observation_variant"] and
        summary.get("policy_variant") == condition["policy_variant"] and
        summary.get("discount_mode") == condition["discount_mode"] and
        sorted(summary.get("allowed_transport_modes", [])) ==
        expected_modes and
        summary.get("experiment_run_seed") == run_seed and
        summary.get("updates") == 2000 and
        summary.get("episodes_trained_total") == 8000 and
        summary.get("validation_seed_start") == 62000000 and
        summary.get("validation_seed_end") == 62000099 and
        evaluation.get("episode_count") == 100 and
        evaluation.get("scheduled_task_count") == 8000 and
        ((summary.get("warm_start_source") is not None) ==
         bool(condition["warm_start"])))


def accepted(summary: dict) -> bool:
    assertions = summary.get("acceptance_assertions", {})
    return bool(summary.get("passed") and assertions and
                all(assertions.values()))


def save_status(path: Path, report: dict) -> None:
    payload = dict(report)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS,
        default=list(CONDITIONS))
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
    condition_specs = protocol["conditions"]
    run_seeds = protocol["training"]["independent_run_base_seeds"]
    jobs = []
    for condition in args.conditions:
        for seed_index in range(args.seed_start, args.seed_end + 1):
            run_dir = RESULT_ROOT / condition / f"seed_{seed_index:02d}"
            summary = run_dir / "stage22_ppo.summary.json"
            summary_payload = load_summary(summary)
            checkpoints = sorted(
                (run_dir / "checkpoints").glob("stage22_ppo_update_*.pt"))
            if formal_run_complete(
                    summary_payload,
                    condition=condition_specs[condition],
                    run_seed=int(run_seeds[seed_index - 1])):
                status = (
                    "SKIP_COMPLETE_ACCEPTED" if accepted(summary_payload) else
                    "SKIP_COMPLETE_GATE_FAILURE")
            elif checkpoints:
                status = "RESUME_CHECKPOINT"
            else:
                status = "START_NEW"
            jobs.append({
                "condition": condition,
                "seed_index": seed_index,
                "status": status,
                "summary": str(summary),
                "accepted": (
                    accepted(summary_payload) if
                    status.startswith("SKIP_COMPLETE") else None),
                "resume_checkpoint": (
                    str(checkpoints[-1]) if checkpoints else None),
            })

    report = {
        "stage": 22,
        "gate": "ALGORITHM_ABLATION_FORMAL_CAMPAIGN_PLAN",
        "claim_boundary": (
            "Training only; fresh locked test is forbidden until all "
            "requested weights are frozen."),
        "conditions": list(args.conditions),
        "seed_range": [args.seed_start, args.seed_end],
        "job_count": len(jobs),
        "start_new_count": sum(
            row["status"] == "START_NEW" for row in jobs),
        "resume_count": sum(
            row["status"] == "RESUME_CHECKPOINT" for row in jobs),
        "skip_complete_count": sum(
            row["status"].startswith("SKIP_COMPLETE") for row in jobs),
        "updates_per_run": 2000,
        "episodes_per_run": 8000,
        "jobs": jobs,
        "execution_requested": args.execute,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    status_path = (
        RESULT_ROOT /
        ("stage22_campaign_" + "-".join(args.conditions) +
         f"_{args.seed_start:02d}_{args.seed_end:02d}.status.json"))
    report["state"] = "RUNNING"
    save_status(status_path, report)
    for ordinal, job in enumerate(jobs, start=1):
        if job["status"].startswith("SKIP_COMPLETE"):
            print(
                f"[{ordinal}/{len(jobs)}] skip complete "
                f"{job['condition']} seed={job['seed_index']:02d}",
                flush=True)
            continue
        command = [
            sys.executable,
            str(RUNNER),
            job["condition"],
            str(job["seed_index"]),
            "--execute",
            "--confirm-long",
        ]
        if job["status"] == "RESUME_CHECKPOINT":
            command.append("--resume-latest")
        print(
            f"[{ordinal}/{len(jobs)}] {job['status'].lower()} "
            f"{job['condition']} seed={job['seed_index']:02d}",
            flush=True)
        job["status"] = "RUNNING"
        job["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(status_path, report)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException as error:
            job["status"] = "FAILED_OR_INTERRUPTED"
            job["error"] = repr(error)
            report["state"] = "STOPPED_ON_FAILURE"
            save_status(status_path, report)
            raise
        summary_payload = load_summary(Path(job["summary"]))
        run_seed = int(run_seeds[job["seed_index"] - 1])
        if not formal_run_complete(
                summary_payload,
                condition=condition_specs[job["condition"]],
                run_seed=run_seed):
            report["state"] = "STOPPED_ON_INCOMPLETE_RUN"
            job["status"] = "FAILED_OR_INCOMPLETE"
            save_status(status_path, report)
            raise RuntimeError(
                f"incomplete formal run: {job['condition']} "
                f"seed={job['seed_index']:02d}")
        job["accepted"] = accepted(summary_payload)
        job["status"] = (
            "COMPLETED_ACCEPTED" if job["accepted"] else
            "COMPLETED_GATE_FAILURE")
        job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(status_path, report)

    report["state"] = (
        "COMPLETED" if all(job.get("accepted") for job in jobs) else
        "COMPLETED_WITH_GATE_FAILURES")
    save_status(status_path, report)
    print(json.dumps({
        "status": str(status_path),
        "state": report["state"],
        "accepted": sum(bool(job.get("accepted")) for job in jobs),
        "job_count": len(jobs),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
