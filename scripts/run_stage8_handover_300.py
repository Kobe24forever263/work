#!/usr/bin/env python3
"""Stage 8 handover transaction acceptance gate (three types x 100)."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.domain import Cargo, CargoPhase, Robot
from warehouse_core.handover import Alignment, HandoverPhase, HandoverTransaction


TYPES = {
    "car_to_dog": ("car_f1_1", "car", 1, "dog_1", "dog", 1),
    "dog_to_car": ("dog_1", "dog", 2, "car_f2_2", "car", 2),
    "same_floor": ("car_f2_1", "car", 2, "car_f2_2", "car", 2),
}

VALID_ALIGNMENT = Alignment(
    True, True, .50, .35, .75, 0.0, .20, 0.0, 0.0, .03, .05, 2.0, 1.5)
INVALID_ALIGNMENT = Alignment(
    True, True, 1.20, .35, .75, 0.0, .20, 0.0, 0.0, .03, .05, 2.0, 1.5)


def make_robot(robot_id: str, kind: str, floor: int) -> Robot:
    region = "F1_NE" if floor == 1 else "F2_EAST"
    return Robot(robot_id, kind, floor, region, floor, region)


def ownership_valid(cargo: Cargo, sender: Robot, receiver: Robot,
                    expected_owner: str) -> bool:
    holders = [r.robot_id for r in (sender, receiver)
               if r.cargo_id == cargo.cargo_id]
    return cargo.owner_id == expected_owner and holders == [expected_owner]


def run_case(trial: int, handover_type: str, repeat: int, scenario: str) -> dict:
    sid, skind, sfloor, rid, rkind, rfloor = TYPES[handover_type]
    sender = make_robot(sid, skind, sfloor)
    receiver = make_robot(rid, rkind, rfloor)
    cargo = Cargo(f"cargo_{trial:04d}", f"task_{trial:04d}",
                  CargoPhase.CARRIED_BY_DOG if skind == "dog"
                  else CargoPhase.CARRIED_BY_CAR,
                  sender.robot_id, f"constraint:{sender.robot_id}")
    sender.cargo_id = cargo.cargo_id
    tx = HandoverTransaction(cargo, sender, receiver)
    transfer_accepted = False
    committed = False
    expected_handling = False

    if scenario == "commit":
        tx.transfer(VALID_ALIGNMENT, f"constraint:{receiver.robot_id}")
        transfer_accepted = True
        committed = tx.verify_and_commit(True, True, True, True)
        expected_handling = (committed and tx.phase == HandoverPhase.COMMIT and
                             ownership_valid(cargo, sender, receiver,
                                             receiver.robot_id))
    elif scenario == "reject_alignment":
        try:
            tx.transfer(INVALID_ALIGNMENT, f"constraint:{receiver.robot_id}")
        except ValueError:
            pass
        expected_handling = (
            not transfer_accepted and tx.phase == HandoverPhase.ROLLBACK and
            ownership_valid(cargo, sender, receiver, sender.robot_id) and
            cargo.constraint_id == f"constraint:{sender.robot_id}")
    elif scenario == "rollback_verify":
        tx.transfer(VALID_ALIGNMENT, f"constraint:{receiver.robot_id}")
        transfer_accepted = True
        committed = tx.verify_and_commit(True, False, True, True)
        expected_handling = (
            not committed and tx.phase == HandoverPhase.ROLLBACK and
            cargo.phase == CargoPhase.RECOVERY and
            ownership_valid(cargo, sender, receiver, sender.robot_id) and
            cargo.constraint_id == f"constraint:{sender.robot_id}")
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return {
        "trial": trial, "handover_type": handover_type, "repeat": repeat,
        "scenario": scenario, "test_pass": expected_handling,
        "handover_committed": committed,
        "phase": tx.phase.value, "cargo_phase": cargo.phase.value,
        "owner_id": cargo.owner_id,
        "sender_cargo_id": sender.cargo_id,
        "receiver_cargo_id": receiver.cargo_id,
        "unique_ownership_valid": ownership_valid(
            cargo, sender, receiver,
            receiver.robot_id if committed else sender.robot_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_type < 25:
        raise SystemExit("--per-type must be at least 25")

    schedule = []
    for handover_type in TYPES:
        scenarios = (["commit"] * (args.per_type - 4) +
                     ["reject_alignment"] * 2 + ["rollback_verify"] * 2)
        schedule.extend((handover_type, i + 1, scenario)
                        for i, scenario in enumerate(scenarios))
    random.Random(args.seed).shuffle(schedule)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial, (kind, repeat, scenario) in enumerate(schedule, 1):
            try:
                result = run_case(trial, kind, repeat, scenario)
            except Exception as exc:
                result = {
                    "trial": trial, "handover_type": kind, "repeat": repeat,
                    "scenario": scenario, "test_pass": False,
                    "handover_committed": False,
                    "failure_code": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{trial}/{len(schedule)}] {kind} {scenario}: "
                  f"{'PASS' if result['test_pass'] else 'FAIL'}")
            if not result["test_pass"]:
                break

    by_type = defaultdict(Counter)
    for row in results:
        by_type[row["handover_type"]]["tests"] += 1
        by_type[row["handover_type"]]["test_passes"] += int(row["test_pass"])
        by_type[row["handover_type"]]["commits"] += int(
            row.get("handover_committed", False))
        if row.get("scenario") == "reject_alignment" and row["test_pass"]:
            by_type[row["handover_type"]]["safe_rejections"] += 1
        if row.get("scenario") == "rollback_verify" and row["test_pass"]:
            by_type[row["handover_type"]]["safe_rollbacks"] += 1

    summary_types = {}
    for kind in TYPES:
        values = by_type[kind]
        operational_rate = values["commits"] / args.per_type
        summary_types[kind] = {
            "tests": values["tests"],
            "test_passes": values["test_passes"],
            "commits": values["commits"],
            "operational_success_rate": operational_rate,
            "safe_rejections": values["safe_rejections"],
            "safe_rollbacks": values["safe_rollbacks"],
            "meets_95_percent": operational_rate >= .95,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "trial_count": len(results),
        "planned_trial_count": len(schedule),
        "test_pass_count": sum(r["test_pass"] for r in results),
        "test_failure_count": sum(not r["test_pass"] for r in results),
        "by_type": summary_types,
        "all_ownership_checks_passed": all(
            r.get("unique_ownership_valid", False) for r in results),
        "results_jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    complete = len(results) == len(schedule)
    accepted = (complete and not summary["test_failure_count"] and
                summary["all_ownership_checks_passed"] and
                all(x["meets_95_percent"] for x in summary_types.values()))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

