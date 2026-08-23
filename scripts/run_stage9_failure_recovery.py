#!/usr/bin/env python3
"""Stage 9 deterministic seven-fault recovery acceptance gate."""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.domain import Cargo, CargoPhase, Robot
from warehouse_core.recovery import (
    FailureRecoveryCoordinator, FaultType, RecoveryDisposition)
from warehouse_core.reservations import ReservationBook
from warehouse_core.spatial import StairManager


EXPECTED = {
    FaultType.PLANNING_FAILURE: RecoveryDisposition.RETRY,
    FaultType.HANDOVER_TIMEOUT: RecoveryDisposition.RETRY,
    FaultType.ROBOT_FAILURE: RecoveryDisposition.REASSIGN,
    FaultType.STAIR_TIMEOUT: RecoveryDisposition.RETRY,
    FaultType.REGION_LOCK_TIMEOUT: RecoveryDisposition.RETRY,
    FaultType.SENSOR_FAILURE: RecoveryDisposition.RETRY,
    FaultType.TASK_CANCELLED: RecoveryDisposition.CANCELLED,
}


def run_case(trial: int, fault: FaultType, repeat: int) -> dict:
    now = [float(trial)]
    clock = lambda: now[0]
    task_id = f"task_{trial:04d}"
    sender = Robot("car_f1_1", "car", 1, "F1_NE", 1, "F1_NE")
    receiver = Robot("dog_1", "dog", 1, "F1_NE", 1, "F1_NE")
    next_dog = Robot("dog_2", "dog", 1, "F1_NW", 1, "F1_NW")
    cargo = Cargo(f"cargo_{trial:04d}", task_id,
                  CargoPhase.HANDOVER_VERIFY, receiver.robot_id, "partial")
    receiver.cargo_id = cargo.cargo_id

    reservations = ReservationBook(clock=clock)
    stairs = StairManager(["STAIR_NE"],
                          {receiver.robot_id: "dog", next_dog.robot_id: "dog"},
                          clock=clock)
    for resource in ("task", "cargo", "region:F1_NE", "handover:NE"):
        assert reservations.reserve(resource, sender.robot_id, task_id, 10.0)
    stair_lease, reason = stairs.reserve(
        "STAIR_NE", receiver.robot_id, task_id, 10.0)
    assert stair_lease and reason == "ACCEPTED"
    stairs.enter("STAIR_NE", receiver.robot_id)

    coordinator = FailureRecoveryCoordinator(
        reservations, stairs, clock, recovery_limit_s=5.0)
    result = coordinator.recover(
        task_id, fault, cargo, (sender, receiver), sender)

    # A different task must immediately be able to acquire every resource.
    reacquired = []
    for resource in ("task", "cargo", "region:F1_NE", "handover:NE"):
        reacquired.append(bool(reservations.reserve(
            resource, "next_robot", "task_next", 10.0)))
    next_stair, next_reason = stairs.reserve(
        "STAIR_NE", next_dog.robot_id, "task_next", 10.0)
    all_resources_reusable = all(reacquired) and bool(next_stair) and \
        next_reason == "ACCEPTED"
    unique_owner = (cargo.owner_id == sender.robot_id and
                    sender.cargo_id == cargo.cargo_id and
                    not receiver.cargo_id)
    test_pass = all((
        result.disposition == EXPECTED[fault],
        result.released_general_locks == 4,
        result.released_stair_claims == 1,
        result.ownership_valid, unique_owner,
        result.completed_within_limit, all_resources_reusable,
    ))
    return {
        "trial": trial, "fault": fault.value, "repeat": repeat,
        "test_pass": test_pass, "disposition": result.disposition.value,
        "released_general_locks": result.released_general_locks,
        "released_stair_claims": result.released_stair_claims,
        "ownership_valid": result.ownership_valid,
        "completed_within_limit": result.completed_within_limit,
        "all_resources_reusable": all_resources_reusable,
        "robot_marked_failed": bool(sender.failure_code),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-fault", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = [(fault, repeat) for fault in FaultType
                for repeat in range(1, args.per_fault + 1)]
    random.Random(args.seed).shuffle(schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial, (fault, repeat) in enumerate(schedule, 1):
            try:
                row = run_case(trial, fault, repeat)
            except Exception as exc:
                row = {"trial": trial, "fault": fault.value, "repeat": repeat,
                       "test_pass": False,
                       "failure_code": f"{type(exc).__name__}: {exc}"}
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{trial}/{len(schedule)}] {fault.value}: "
                  f"{'PASS' if row['test_pass'] else 'FAIL'}")
            if not row["test_pass"]:
                break

    passes = Counter(r["fault"] for r in results if r["test_pass"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "trial_count": len(results),
        "planned_trial_count": len(schedule),
        "test_pass_count": sum(r["test_pass"] for r in results),
        "test_failure_count": sum(not r["test_pass"] for r in results),
        "per_fault": {fault.value: {"trials": args.per_fault,
                                     "passes": passes[fault.value]}
                      for fault in FaultType},
        "all_locks_released": all(r.get("all_resources_reusable", False)
                                  for r in results),
        "all_ownership_recovered": all(r.get("ownership_valid", False)
                                       for r in results),
        "all_within_time_limit": all(r.get("completed_within_limit", False)
                                     for r in results),
        "results_jsonl": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    accepted = (len(results) == len(schedule) and
                not summary["test_failure_count"] and
                summary["all_locks_released"] and
                summary["all_ownership_recovered"] and
                summary["all_within_time_limit"])
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

