#!/usr/bin/env python3
"""Execute a frozen ten-task Stage 26 campaign with real SCAN processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(
    "/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python")


def stop_process(process, timeout=8.0):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3.0)
    handle = getattr(process, "stage26_log_handle", None)
    if handle is not None and not handle.closed:
        handle.close()


def publish_string(topic, value):
    subprocess.run([
        "ros2", "topic", "pub", "--once", topic,
        "std_msgs/msg/String", "{data: " + json.dumps(value) + "}",
    ], check=True, stdout=subprocess.DEVNULL)


def cargo_value(state, task):
    if state == "PICKUP":
        x, y, z = task["task"]["source"]["xyz"]
        return f"PICKUP:{x}:{y}:{z + 0.06}"
    if state in ("DELIVERED", "FAILED"):
        key = "target" if state == "DELIVERED" else "source"
        x, y, z = task["task"][key]["xyz"]
        return f"{state}:{x}:{y}:{z + 0.06}"
    return state


def start_scan(segment, publish_global_map, log_file):
    x, y, z = segment["initial_xyz"]
    robot = segment["robot"]
    robot_type = "dog" if robot.startswith("dog_") else "car"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen([
        "ros2", "launch", "warehouse_bringup",
        "stage26_scan_segment.launch.py",
        f"robot:={robot}", f"robot_type:={robot_type}",
        f"route_file:={segment['scan_route_file']}",
        f"init_x:={x}", f"init_y:={y}", f"init_z:={z}",
        "publish_global_map:=" + ("true" if publish_global_map else "false"),
    ], start_new_session=True, stdout=handle, stderr=subprocess.STDOUT)
    process.stage26_log_handle = handle
    return process


def wait_for_goal(segment):
    x, y, z = segment["goal_xyz"]
    return subprocess.run([
        str(PYTHON_BIN), str(ROOT / "scripts" / "wait_for_scan_pose.py"),
        segment["robot"], str(x), str(y), str(z),
        "--timeout", str(segment["timeout_s"]),
    ]).returncode == 0


def run_pose_sync(segment):
    return subprocess.run([
        str(PYTHON_BIN), str(ROOT / "scripts" / "stage26_pose_sync.py"),
        "--robot", segment["robot"],
        "--path-json", json.dumps(segment["path"]),
        "--speed", str(segment["speed_mps"]),
    ]).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-gap", type=float, default=2.0)
    parser.add_argument("--start-task", type=int, default=1)
    parser.add_argument("--start-segment", type=int, default=1)
    parser.add_argument("--keep-rviz", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    campaign_file = args.campaign.expanduser().resolve()
    campaign = json.loads(campaign_file.read_text(encoding="utf-8"))
    if (campaign.get("scope") != "TEN_SEQUENTIAL_TASKS" or
            len(campaign.get("tasks", ())) != 10):
        raise ValueError("campaign must contain exactly ten sequential tasks")
    if not 1 <= args.start_task <= len(campaign["tasks"]):
        raise ValueError("--start-task must be within the frozen campaign")
    first_task_segments = campaign["tasks"][args.start_task - 1]["segments"]
    if not 1 <= args.start_segment <= len(first_task_segments):
        raise ValueError(
            "--start-segment must be within the selected start task")

    rviz = None
    scan = None
    log_root = campaign_file.parent / "runtime_logs" / str(
        campaign["campaign_seed"])
    try:
        rviz = subprocess.Popen([
            "ros2", "launch", "warehouse_bringup",
            "stage26_scan_campaign_rviz.launch.py",
            f"campaign_file:={campaign_file}",
        ], start_new_session=True)
        time.sleep(5.0)
        if rviz.poll() is not None:
            raise RuntimeError("RViz campaign launch exited during startup")
        first_scan = True
        if args.start_task == 1 and args.start_segment == 1:
            start_text = "从第 1 个任务开始"
        else:
            start_text = (
                f"从任务 {args.start_task} 的第 {args.start_segment} 段续跑")
        print(
            "Stage 26: 连续发布 10 个任务；每次仅执行一个任务，"
            f"机器人位置与 GRU 记忆均不重置；{start_text}。",
            flush=True)
        for task_number in range(args.start_task, len(campaign["tasks"]) + 1):
            task = campaign["tasks"][task_number - 1]
            assignment = task["policy_decision"]["assignment"]
            print(
                f"\n[TASK {task_number}/10] {task['task_type']} | "
                f"{task['task']['source']['point_id']} -> "
                f"{task['task']['target']['point_id']} | "
                f"policy={assignment['transport_mode']}", flush=True)
            publish_string(
                "/task2/stage26_scan/control", f"TASK:{task_number}")
            publish_string(
                "/task2/cargo_owner", cargo_value("PICKUP", task))
            for segment_number, segment in enumerate(
                    task["segments"], start=1):
                if (task_number == args.start_task and
                        segment_number < args.start_segment):
                    continue
                backend = segment["execution_backend"]
                publish_string(
                    "/task2/stage26_scan/control",
                    f"ACTIVE:{backend}:{segment['robot']}:{segment_number}")
                publish_string(
                    "/task2/cargo_owner",
                    cargo_value(segment["cargo_state"], task))
                print(
                    f"  [{backend} {segment_number}/{len(task['segments'])}] "
                    f"{segment['robot']} | {segment['label']} | "
                    f"{segment['route_length_m']:.1f} m", flush=True)
                if backend == "POSE_SYNC":
                    passed = run_pose_sync(segment)
                else:
                    scan_log = log_root / (
                        f"task_{task_number:02d}_segment_"
                        f"{segment_number:02d}_{segment['robot']}.log")
                    scan = start_scan(segment, first_scan, scan_log)
                    first_scan = False
                    passed = wait_for_goal(segment)
                    stop_process(scan)
                    scan = None
                    time.sleep(0.7)
                if not passed:
                    reason = (
                        f"task {task_number} segment {segment_number} "
                        f"{segment['robot']} SCAN timeout")
                    publish_string(
                        "/task2/stage26_scan/control", f"FAILED:{reason}")
                    print(f"STAGE 26 SCAN CAMPAIGN FAIL: {reason}", flush=True)
                    if backend == "SCAN":
                        print(f"SCAN log: {scan_log}", flush=True)
                    return 1
            final_state = task["outcome"]
            publish_string(
                "/task2/cargo_owner", cargo_value(
                    "DELIVERED" if final_state == "COMPLETED" else "FAILED",
                    task))
            publish_string(
                "/task2/stage26_scan/control",
                f"TASK_COMPLETE:{task_number}")
            print(
                f"[TASK {task_number}/10] {final_state}", flush=True)
            time.sleep(max(0.0, args.task_gap))
            args.start_segment = 1
        publish_string(
            "/task2/stage26_scan/control", "CAMPAIGN_COMPLETE")
        print(
            "STAGE 26 SCAN CAMPAIGN PASS: 10/10 tasks resolved; "
            "no robot-position reset.", flush=True)
        if args.keep_rviz:
            print("RViz 将保持打开；关闭窗口即可结束全部进程。", flush=True)
            return rviz.wait()
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(scan)
        stop_process(rviz)


if __name__ == "__main__":
    raise SystemExit(main())
