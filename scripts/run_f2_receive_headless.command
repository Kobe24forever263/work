#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
ROBOT="${1:-car_f2_2}"
cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_f2_receive_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_f2_receive
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$WORK_ROOT/src"
export CARTER_F2_ROBOT="$ROBOT"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

WORLD="$WORK_ROOT/src/warehouse_bringup/worlds/two_floor_warehouse.sdf"
RVIZ_CONFIG="$WORK_ROOT/src/warehouse_bringup/config/carter_route_minimal.rviz"

env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -s -r -v 1 "$WORLD" &
SERVER_PID=$!
sleep 3
ros2 launch warehouse_bringup carter_interfaces.launch.py robot:="$ROBOT" &
INTERFACE_PID=$!
sleep 4
ros2 launch warehouse_bringup single_carter_nav.launch.py robot:="$ROBOT" &
NAV_PID=$!
ROUTE_PID=0
RVIZ_PID=0
trap '[[ "$ROUTE_PID" -gt 0 ]] && kill -INT "$ROUTE_PID" 2>/dev/null; [[ "$RVIZ_PID" -gt 0 ]] && kill -INT "$RVIZ_PID" 2>/dev/null; kill -INT "$NAV_PID" "$INTERFACE_PID" "$SERVER_PID" 2>/dev/null' EXIT INT TERM
sleep 12
echo "阶段 6：$ROBOT 从二楼分区车位驶向对应楼梯接收停车线。"
python3 "$WORK_ROOT/scripts/navigate_carter_floor2_receive.py" &
ROUTE_PID=$!
sleep 8
ros2 run rviz2 rviz2 --ros-args -r __ns:="/$ROBOT" \
  -r /tf:=tf -r /tf_static:=tf_static -- -d "$RVIZ_CONFIG" &
RVIZ_PID=$!
wait "$ROUTE_PID"
echo "阶段 6 PASS：$ROBOT 已到达对应楼梯安全停车线。"
wait "$NAV_PID"
