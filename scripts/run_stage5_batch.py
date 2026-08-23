#!/usr/bin/env python3
"""Run fixed Go2 stair trials with odometry-based pass/fail evidence."""

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
ROS2 = Path(
    "/Users/lab4099/Desktop/Mujoco/task2_handover/"
    ".pixi/envs/default/bin/ros2")
CONFIG = WORK / "src/warehouse_bringup/config/scan_planner"
MANIFEST = CONFIG / "four_go2_manifest.yaml"
WAITER = WORK / "scripts/wait_for_scan_pose.py"
DIRECTIONS = ("up", "down")


def route_goal(robot_spec, direction):
    route_path = WORK / robot_spec[f"{direction}_route_file"]
    values = yaml.safe_load(route_path.read_text(encoding="utf-8"))[
        "scan_planner_node"]["ros__parameters"]["fsm.waypoints"]
    return [float(value) for value in values[-3:]]


def schedule(robots, seed, repeats, smoke):
    count = 1 if smoke else repeats
    trials = []
    for robot, spec in robots.items():
        for direction in DIRECTIONS:
            goal = route_goal(spec, direction)
            for repeat in range(count):
                trials.append({
                    "robot": robot,
                    "direction": direction,
                    "repeat": repeat,
                    "goal_xyz": goal,
                })
    random.Random(seed).shuffle(trials)
    for index, trial in enumerate(trials, 1):
        trial["trial"] = index
    return trials


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def run_trial(trial, timeout, tolerance, stable, jitter, rng, log_dir):
    jitter_x = rng.uniform(-jitter, jitter)
    jitter_y = rng.uniform(-jitter, jitter)
    launch_command = [
        str(ROS2), "launch", "warehouse_bringup",
        "single_go2_stair_validation.launch.py",
        f"robot:={trial['robot']}",
        f"direction:={trial['direction']}",
        "show_rviz:=false",
        "gazebo_bridge:=false",
        "publish_robot_state:=false",
        "publish_global_map:=false",
        f"initial_offset_x:={jitter_x:.4f}",
        f"initial_offset_y:={jitter_y:.4f}",
    ]
    goal = trial["goal_xyz"]
    wait_command = [
        sys.executable, str(WAITER), trial["robot"],
        *(str(value) for value in goal),
        "--tolerance", str(tolerance),
        "--stable", str(stable),
        "--timeout", str(timeout),
    ]
    log_path = log_dir / (
        f"{trial['trial']:04d}_{trial['robot']}_"
        f"{trial['direction']}.log")
    started = time.monotonic()
    failure = ""
    distance = None
    env = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as log:
        planner = subprocess.Popen(
            launch_command, stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, env=env)
        try:
            time.sleep(2.0)
            if planner.poll() is not None:
                failure = f"PLANNER_EXIT_{planner.returncode}"
            else:
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
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--stable", type=float, default=1.0)
    parser.add_argument("--initial-jitter", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=WORK / "results/stage5/stage5_fixed_160.jsonl")
    args = parser.parse_args()

    robots = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["robots"]
    trials = schedule(robots, args.seed, args.repeats, args.smoke)
    coverage = Counter(
        (item["robot"], item["direction"]) for item in trials)
    print(
        f"Stage 5 schedule: {len(trials)} trials, seed={args.seed}, "
        f"jitter={args.initial_jitter}, coverage={dict(coverage)}")
    if args.schedule_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / f"{args.output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for trial in trials:
            result = run_trial(
                trial, args.timeout, args.tolerance, args.stable,
                args.initial_jitter, rng, log_dir)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{result['trial']}/{len(trials)}] "
                f"{result['robot']} {result['direction']}: "
                f"{'PASS' if result['success'] else result['failure_code']}")
            if args.fail_fast and not result["success"]:
                print("Stage 5 fail-fast: stopping after first failure")
                return 1

    by_route = defaultdict(lambda: {"trials": 0, "passes": 0})
    for result in results:
        row = by_route[f"{result['robot']}:{result['direction']}"]
        row["trials"] += 1
        row["passes"] += int(result["success"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "initial_jitter": args.initial_jitter,
        "trial_count": len(results),
        "success_count": sum(item["success"] for item in results),
        "failure_count": sum(not item["success"] for item in results),
        "by_route": dict(by_route),
        "results_jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
