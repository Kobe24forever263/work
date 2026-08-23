#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_ne_handover_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_ne_handover
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$WORK_ROOT/src"

source "$SCAN_ROOT/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

WORLD="$WORK_ROOT/src/warehouse_bringup/worlds/two_floor_warehouse.sdf"

# Headless Gazebo server: physics remains active, but no Gazebo GUI is opened.
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -s -r -v 1 "$WORLD" &
SERVER_PID=$!
sleep 3

ros2 launch warehouse_bringup carter_interfaces.launch.py robot:=car_f1_1 &
INTERFACE_PID=$!
sleep 4
ros2 launch warehouse_bringup single_carter_nav.launch.py robot:=car_f1_1 &
NAV_PID=$!
sleep 12

ROUTE_PID=0
RVIZ_PID=0
DOG_PID=0
trap '[[ "$DOG_PID" -gt 0 ]] && kill -INT "$DOG_PID" 2>/dev/null; [[ "$ROUTE_PID" -gt 0 ]] && kill -INT "$ROUTE_PID" 2>/dev/null; [[ "$RVIZ_PID" -gt 0 ]] && kill -INT "$RVIZ_PID" 2>/dev/null; kill -INT "$NAV_PID" "$INTERFACE_PID" "$SERVER_PID" 2>/dev/null' EXIT INT TERM

echo "阶段 1：car_f1_1 进入东侧外圈，并沿边缘驶向东北楼梯。"
ros2 launch warehouse_bringup heterogeneous_rviz.launch.py &
RVIZ_PID=$!
sleep 5
python3 "$WORK_ROOT/scripts/navigate_carter_outer_handover.py" &
ROUTE_PID=$!
wait "$ROUTE_PID"
echo "阶段 2：车辆到站并站稳，交接确认；现在启动 dog_1 的 SCAN-Planner。"
ros2 launch warehouse_bringup single_go2_stair_validation.launch.py \
  robot:=dog_1 direction:=up show_rviz:=false gazebo_bridge:=true &
DOG_PID=$!
echo "阶段 3：dog_1 正在通过 STAIR_NE 上楼；保持精简 RViz，Gazebo GUI 未启动。"
wait "$DOG_PID"
