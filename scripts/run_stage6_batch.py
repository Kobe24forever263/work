#!/usr/bin/env python3
"""Run reproducible Carter SCAN trials and save JSONL evidence.

The manifest controls initial jitter.  Stage 6 currently sets it to zero so
the accepted fixed routes are stabilized before robustness perturbations.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import yaml


WORK = Path("/Users/lab4099/Desktop/Mujoco/work")
ROS2 = Path("/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/ros2")
MANIFEST = (
    WORK / "src/warehouse_bringup/config/scan_planner/"
    "stage6_carter_manifest.yaml")
WAITER = WORK / "scripts/wait_for_scan_pose.py"


def make_schedule(robots, seed, trials_per_robot):
    schedule = []
    for robot, spec in robots.items():
        routes = sorted(spec["routes"])
        for repeat in range(trials_per_robot):
            route = routes[repeat % len(routes)]
            schedule.append({
                "robot": robot, "route": route, "repeat": repeat,
                "goal_xyz": spec["routes"][route]["goal_xyz"],
                "floor": spec["floor"],
            })
    random.Random(seed).shuffle(schedule)
    for index, trial in enumerate(schedule, 1):
        trial["trial"] = index
    return schedule


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def run_trial(trial, rng, policy, log_dir):
    jitter = float(policy["initial_jitter_xy"])
    jitter_x = rng.uniform(-jitter, jitter)
    jitter_y = rng.uniform(-jitter, jitter)
    timeout = float(policy["timeout_seconds"])
    launch_command = [
        str(ROS2), "launch", "warehouse_bringup",
        "single_carter_scan_validation.launch.py",
        f"robot:={trial['robot']}", f"stair:={trial['route']}",
        "publish_global_map:=false",
        f"initial_offset_x:={jitter_x:.4f}",
        f"initial_offset_y:={jitter_y:.4f}",
    ]
    goal = trial["goal_xyz"]
    wait_command = [
        sys.executable, str(WAITER), trial["robot"],
        *(str(value) for value in goal),
        "--tolerance", str(policy["goal_tolerance"]),
        "--stable", str(policy["stable_seconds"]),
        "--timeout", str(timeout),
    ]
    log_path = log_dir / (
        f"{trial['trial']:04d}_{trial['robot']}_{trial['route']}.log")
    started = time.monotonic()
    failure = ""
    distance = None
    env = os.environ.copy()
    bringup_prefix = str(WORK / "install/warehouse_bringup")
    current_prefix = env.get("AMENT_PREFIX_PATH", "")
    if bringup_prefix not in current_prefix.split(os.pathsep):
        env["AMENT_PREFIX_PATH"] = (
            bringup_prefix + (os.pathsep + current_prefix
                              if current_prefix else ""))
    with log_path.open("w", encoding="utf-8") as log:
        planner = subprocess.Popen(
            launch_command, stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, env=env)
        try:
            # Configuration/package errors exit immediately. Detect them before
            # starting the goal waiter so a broken environment never consumes
            # the full navigation timeout.
            time.sleep(2.0)
            if planner.poll() is not None:
                failure = f"PLANNER_EXIT_{planner.returncode}"
                return {
                    **trial,
                    "success": False,
                    "failure_code": failure,
                    "duration_s": round(time.monotonic() - started, 3),
                    "final_error_m": None,
                    "initial_jitter_xy": [
                        round(jitter_x, 4), round(jitter_y, 4)],
                    "log": str(log_path),
                }
            waiter = subprocess.run(
                wait_command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout + 10, env=env)
            log.write(waiter.stdout)
            if waiter.returncode:
                failure = "GOAL_TIMEOUT"
            else:
                for token in waiter.stdout.split():
                    if token.startswith("distance="):
                        distance = float(token.split("=", 1)[1])
        except subprocess.TimeoutExpired:
            failure = "WAITER_TIMEOUT"
        finally:
            stop_group(planner)
    return {
        **trial,
        "success": not failure,
        "failure_code": failure,
        "duration_s": round(time.monotonic() - started, 3),
        "final_error_m": distance,
        "initial_jitter_xy": [round(jitter_x, 4), round(jitter_y, 4)],
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--trials-per-robot", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop immediately after the first failed trial")
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=WORK / "results/stage6/stage6_full.jsonl")
    args = parser.parse_args()
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    policy = data["test_policy"]
    count = args.trials_per_robot or int(policy["trials_per_robot"])
    if args.smoke:
        count = 1
    trials = make_schedule(data["robots"], args.seed, count)
    coverage = Counter((item["robot"], item["route"]) for item in trials)
    print(
        f"Stage 6 schedule: {len(trials)} trials, seed={args.seed}, "
        f"coverage={dict(coverage)}")
    if args.schedule_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / f"{args.output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial in trials:
            result = run_trial(trial, rng, policy, log_dir)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{result['trial']}/{len(trials)}] {result['robot']} "
                f"{result['route']}: "
                f"{'PASS' if result['success'] else result['failure_code']}")
            if args.fail_fast and not result["success"]:
                print("Stage 6 fail-fast: stopping after the first failed trial")
                return 1

    by_robot = defaultdict(lambda: {"trials": 0, "passes": 0})
    for result in results:
        row = by_robot[result["robot"]]
        row["trials"] += 1
        row["passes"] += int(result["success"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "trial_count": len(results),
        "success_count": sum(item["success"] for item in results),
        "failure_count": sum(not item["success"] for item in results),
        "by_robot": dict(by_robot),
        "results_jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
