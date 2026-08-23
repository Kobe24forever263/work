#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_scan_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_four_go2
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$WORK_ROOT/src"

source "$SCAN_ROOT/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
# The macOS Pixi/colcon combination does not add resource-only CMake packages
# to AMENT_PREFIX_PATH reliably, so expose the bringup prefix explicitly.
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

echo "正在启动四只 Go2 的 SCAN-Planner 上楼验证..."
echo "保持此窗口打开；按 Control+C 停止。"
WORLD="$WORK_ROOT/src/warehouse_bringup/worlds/two_floor_warehouse.sdf"
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -s -r -v 3 "$WORLD" &
SERVER_PID=$!
sleep 3
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -g -v 3 --render-engine-gui-api-backend opengl &
GUI_PID=$!
trap 'kill "$GUI_PID" "$SERVER_PID" 2>/dev/null' EXIT INT TERM
sleep 3
ros2 launch warehouse_bringup four_go2_scan_validation.launch.py
