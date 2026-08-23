 #!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
ROBOT="${1:-car_f1_1}"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_carter_rviz_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_carter_rviz
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$WORK_ROOT/src"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

WORLD="$WORK_ROOT/src/warehouse_bringup/worlds/two_floor_warehouse.sdf"
RVIZ_CONFIG="$WORK_ROOT/src/warehouse_bringup/config/carter_nav2.rviz"

env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -s -r -v 2 "$WORLD" &
SERVER_PID=$!
sleep 3
env -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
  pixi run gz sim -g -v 2 --render-engine-gui-api-backend opengl &
GUI_PID=$!
sleep 3
ros2 launch warehouse_bringup carter_interfaces.launch.py &
INTERFACE_PID=$!
sleep 5
ros2 launch warehouse_bringup single_carter_nav.launch.py robot:="$ROBOT" &
NAV_PID=$!
sleep 8
ros2 run rviz2 rviz2 --ros-args -r __ns:=/"$ROBOT" \
  -r /tf:=tf -r /tf_static:=tf_static -- -d "$RVIZ_CONFIG" &
RVIZ_PID=$!

trap 'kill -INT "$RVIZ_PID" "$NAV_PID" "$INTERFACE_PID" "$GUI_PID" "$SERVER_PID" 2>/dev/null' EXIT INT TERM
echo "RViz 已打开：点击顶部 Nav2 Goal，随后在地图上拖动设定目标与朝向。"
wait "$RVIZ_PID"
