#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"
ROBOT="${1:-dog_1}"
DIRECTION="${2:-up}"

if [[ ! "$ROBOT" =~ '^dog_[1-4]$' ]] || [[ "$DIRECTION" != "up" && "$DIRECTION" != "down" ]]; then
  echo "用法: $0 dog_1|dog_2|dog_3|dog_4 up|down"
  exit 2
fi

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_single_go2_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_single_go2
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$WORK_ROOT/src"

source "$SCAN_ROOT/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

echo "正在启动 $ROBOT ${DIRECTION} 楼梯单机验证..."
WORLD="$WORK_ROOT/src/warehouse_bringup/worlds/two_floor_warehouse.sdf"
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -s -r -v 2 "$WORLD" &
SERVER_PID=$!
sleep 3
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -g -v 2 --render-engine-gui-api-backend opengl &
GUI_PID=$!
trap 'kill "$GUI_PID" "$SERVER_PID" 2>/dev/null' EXIT INT TERM
sleep 3
ros2 launch warehouse_bringup single_go2_stair_validation.launch.py \
  robot:="$ROBOT" direction:="$DIRECTION" show_rviz:=true
