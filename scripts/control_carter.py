#!/usr/bin/env python3
"""Safe keyboard teleoperation for the six Carter robots in Gazebo Sim."""

import argparse
import os
import select
import sys
import termios
import time
import tty


ROBOTS = (
    "car_f1_1", "car_f1_2", "car_f1_3",
    "car_f1_4", "car_f2_1", "car_f2_2",
)


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    parser = argparse.ArgumentParser(description="Keyboard control for one Carter")
    parser.add_argument("--robot", choices=ROBOTS, default="car_f1_1")
    parser.add_argument("--partition", default="warehouse_preview")
    args = parser.parse_args()

    # Must be set before Gazebo Transport creates its first Node.
    os.environ["GZ_PARTITION"] = args.partition
    os.environ.setdefault("GZ_IP", "127.0.0.1")
    from gz.msgs10.twist_pb2 import Twist
    from gz.transport13 import Node

    node = Node()
    publishers = {
        robot: node.advertise(f"/model/{robot}/cmd_vel", Twist)
        for robot in ROBOTS
    }
    selected = args.robot
    linear = 0.0
    angular = 0.0
    last_motion_key = 0.0
    old_settings = termios.tcgetattr(sys.stdin)

    def publish(robot, vx, wz):
        message = Twist()
        message.linear.x = vx
        message.angular.z = wz
        publishers[robot].publish(message)

    def stop(robot):
        publish(robot, 0.0, 0.0)

    def status():
        print(
            f"\r控制: {selected:<8}  前进速度 {linear:+.2f} m/s  "
            f"转向速度 {angular:+.2f} rad/s     ",
            end="", flush=True,
        )

    print("Carter 键盘控制")
    print("1–6 切换车辆 | W/S 前进后退 | A/D 左右转 | 空格急停 | Q 退出")
    print("安全机制：松开按键约 0.7 秒后自动停车；需要移动时请持续按键。")
    status()
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                key = sys.stdin.read(1).lower()
                now = time.monotonic()
                if key == "q":
                    break
                if key in "123456":
                    stop(selected)
                    selected = ROBOTS[int(key) - 1]
                    linear = angular = 0.0
                elif key == "w":
                    linear = clamp(linear + 0.10, -0.50, 0.60)
                    last_motion_key = now
                elif key == "s":
                    linear = clamp(linear - 0.10, -0.50, 0.60)
                    last_motion_key = now
                elif key == "a":
                    angular = clamp(angular + 0.20, -1.2, 1.2)
                    last_motion_key = now
                elif key == "d":
                    angular = clamp(angular - 0.20, -1.2, 1.2)
                    last_motion_key = now
                elif key in (" ", "x"):
                    linear = angular = 0.0
                    stop(selected)
                status()
            if (linear or angular) and time.monotonic() - last_motion_key > 0.7:
                linear = angular = 0.0
                stop(selected)
                status()
            publish(selected, linear, angular)
    finally:
        for robot in ROBOTS:
            stop(robot)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\n全部 Carter 已停车，控制程序退出。")


if __name__ == "__main__":
    main()
