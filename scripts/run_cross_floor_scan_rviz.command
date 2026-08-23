#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_cross_floor_scan_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
# A fresh DDS domain prevents retained robot_description/TF samples from an
# earlier interrupted demo from contaminating this RViz session.
export ROS_DOMAIN_ID=$((80 + $$ % 120))
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Fast DDS shared-memory lock files can survive an interrupted macOS run and
# then prevent newly started nodes from discovering each other.  Cyclone DDS
# uses the loopback transport here and is deterministic across repeated runs.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source "$SCAN_ROOT/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"
START_STAGE=${TASK2_START_STAGE:-1}
MISSION=${1:-ne}
SHOW_RVIZ=${TASK2_SHOW_RVIZ:-1}

case "$MISSION" in
  ne)
    F1_CAR=car_f1_1; DOG=dog_1; F2_CAR=car_f2_2
    F1_X=20.3; F1_Y=14.0; DOG_X=13.2; DOG_Y=10.2
    RECEIVE_X=12.6; RECEIVE_Y=9.6; F2_X=-7.0; F2_Y=-6.5
    PICK_X=12.0; PICK_Y=9.5; PICK_Z=0.51
    DROP_X=-7.0; DROP_Y=-5.0; DROP_Z=4.51; LABEL="东北上楼—西南卸货"
    ;;
  nw)
    F1_CAR=car_f1_2; DOG=dog_2; F2_CAR=car_f2_1
    F1_X=-20.3; F1_Y=14.0; DOG_X=-14.2929; DOG_Y=11.2929
    RECEIVE_X=-9.5; RECEIVE_Y=6.5; F2_X=-7.0; F2_Y=6.5
    PICK_X=-12.0; PICK_Y=9.5; PICK_Z=0.51
    DROP_X=-7.0; DROP_Y=5.0; DROP_Z=4.51; LABEL="西北"
    ;;
  sw)
    F1_CAR=car_f1_3; DOG=dog_3; F2_CAR=car_f2_1
    F1_X=-20.3; F1_Y=-14.0; DOG_X=-14.2929; DOG_Y=-11.2929
    RECEIVE_X=-9.5; RECEIVE_Y=-6.5; F2_X=-7.0; F2_Y=-6.5
    PICK_X=-12.0; PICK_Y=-9.5; PICK_Z=0.51
    DROP_X=-7.0; DROP_Y=-5.0; DROP_Z=4.51; LABEL="西南"
    ;;
  se)
    F1_CAR=car_f1_4; DOG=dog_4; F2_CAR=car_f2_2
    F1_X=20.3; F1_Y=-14.0; DOG_X=14.2929; DOG_Y=-11.2929
    RECEIVE_X=9.5; RECEIVE_Y=-6.5; F2_X=7.0; F2_Y=-6.5
    PICK_X=12.0; PICK_Y=-9.5; PICK_Z=0.51
    DROP_X=7.0; DROP_Y=-5.0; DROP_Z=4.51; LABEL="东南"
    ;;
  *)
    echo "用法: $0 [ne|nw|sw|se]"
    exit 2
    ;;
esac
echo "Task-isolated ROS domain: $ROS_DOMAIN_ID"
echo "任务：${LABEL}楼梯 ${F1_CAR} -> ${DOG} -> ${F2_CAR}"

RVIZ_PID=0
STAGE_PID=0
cleanup() {
  [[ "$STAGE_PID" -gt 0 ]] && kill -TERM "$STAGE_PID" 2>/dev/null
  [[ "$RVIZ_PID" -gt 0 ]] && kill -TERM "$RVIZ_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

if [[ "$SHOW_RVIZ" == "1" ]]; then
  ros2 launch warehouse_bringup heterogeneous_rviz.launch.py mission:="$MISSION" &
  RVIZ_PID=$!
  sleep 5
  echo "[货物] C_DEMO 位于 ${LABEL}取货台，等待车辆装载"
  ros2 topic pub --once /task2/cargo_owner std_msgs/msg/String \
    "{data: 'PICKUP:${PICK_X}:${PICK_Y}:${PICK_Z}'}" >/dev/null
fi

if [[ "$START_STAGE" -le 1 ]]; then
  if [[ "$MISSION" == "ne" ]]; then
    echo "[1/6] SCAN Carter ${F1_CAR}：停车位 -> ${LABEL}取货平台"
    ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
      robot:="$F1_CAR" stair:="$MISSION" route_phase:=pickup publish_global_map:=true &
    STAGE_PID=$!
    if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
      "$F1_CAR" 12.0 8.0 0.45 --timeout 180; then
      echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：一楼车辆未到达取货台"
      exit 1
    fi
    kill -TERM "$STAGE_PID" 2>/dev/null
    wait "$STAGE_PID" 2>/dev/null || true
    STAGE_PID=0
    echo "[装货确认] ${F1_CAR} 停稳，立即吸附 C_DEMO"
    ros2 topic pub --once /task2/cargo_owner std_msgs/msg/String \
      "{data: '${F1_CAR}'}" >/dev/null
    echo "[2/6] C_DEMO 已吸附；${F1_CAR} 从取货台驶向楼梯交接线"
    ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
      robot:="$F1_CAR" stair:="$MISSION" route_phase:=send_loaded publish_global_map:=true &
  else
    echo "[1/5] SCAN Carter ${F1_CAR}：停车位 -> ${LABEL}装货平台 -> 外圈 -> 楼梯交接线"
    ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
      robot:="$F1_CAR" stair:="$MISSION" publish_global_map:=true &
  fi
  STAGE_PID=$!
  if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
    "$F1_CAR" "$F1_X" "$F1_Y" 0.45 --timeout 360; then
    echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：一楼车辆未到达交接线"
    exit 1
  fi
  kill -TERM "$STAGE_PID" 2>/dev/null
  wait "$STAGE_PID" 2>/dev/null || true
  STAGE_PID=0
  echo "[交接 1] ${F1_CAR} 已到站，${DOG} 接收货物"
  ros2 topic pub --once /task2/cargo_owner std_msgs/msg/String \
    "{data: '${DOG}'}" >/dev/null
fi

if [[ "$START_STAGE" -le 2 ]]; then
  echo "[2/3] SCAN Go2 ${DOG}：${LABEL}楼梯一楼 -> 二楼"
  ros2 launch warehouse_bringup single_go2_stair_validation.launch.py \
    robot:="$DOG" direction:=up show_rviz:=false gazebo_bridge:=false \
    publish_robot_state:=false &
  STAGE_PID=$!
  if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
    "$DOG" "$DOG_X" "$DOG_Y" 4.56 --timeout 300; then
    echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：Go2 未到达二楼"
    exit 1
  fi
  kill -TERM "$STAGE_PID" 2>/dev/null
  wait "$STAGE_PID" 2>/dev/null || true
  STAGE_PID=0
  echo "[交接 2] ${DOG} 已到二楼，${F2_CAR} 开始接收"
fi

echo "[3/5] SCAN Carter ${F2_CAR}：二楼停车位 -> ${LABEL}楼梯接收线"
ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
  robot:="$F2_CAR" stair:="$MISSION" route_phase:=receive publish_global_map:=true &
STAGE_PID=$!
if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
  "$F2_CAR" "$RECEIVE_X" "$RECEIVE_Y" 4.70 --timeout 240; then
  echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：二楼车辆未到达接收点"
  exit 1
fi
kill -TERM "$STAGE_PID" 2>/dev/null
wait "$STAGE_PID" 2>/dev/null || true
STAGE_PID=0
echo "[交接 2] ${F2_CAR} 已在楼梯口与 ${DOG} 对齐，立即切换货物所有权"
ros2 topic pub --once /task2/cargo_owner std_msgs/msg/String \
  "{data: '${F2_CAR}'}" >/dev/null
echo "[4/5] SCAN Carter ${F2_CAR}：交接完成 -> ${LABEL}卸货平台"
ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
  robot:="$F2_CAR" stair:="$MISSION" route_phase:=deliver publish_global_map:=true &
STAGE_PID=$!
if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
    "$F2_CAR" "$F2_X" "$F2_Y" 4.70 --timeout 300; then
  echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：二楼车辆未到达卸货平台"
  exit 1
fi
kill -TERM "$STAGE_PID" 2>/dev/null
wait "$STAGE_PID" 2>/dev/null || true
STAGE_PID=0
echo "[卸货确认] ${F2_CAR} 到站，立即将 C_DEMO 放到实体平台并解绑"
ros2 topic pub --once /task2/cargo_owner std_msgs/msg/String \
  "{data: 'DELIVERED:${DROP_X}:${DROP_Y}:${DROP_Z}'}" >/dev/null

if [[ "$MISSION" == "ne" ]]; then
  echo "[5/5] 货物已送达；${F2_CAR} 清空卸货位 -> 最近空闲待命泊位"
  ros2 launch warehouse_bringup single_carter_scan_validation.launch.py \
    robot:="$F2_CAR" stair:="$MISSION" route_phase:=standby publish_global_map:=false &
  STAGE_PID=$!
  if ! python3 "$WORK_ROOT/scripts/wait_for_scan_pose.py" \
    "$F2_CAR" -5.5 -9.5 4.70 --timeout 180; then
    echo "CROSS-FLOOR SCAN TASK FAIL (${MISSION})：二楼车辆未清场至待命泊位"
    exit 1
  fi
  kill -TERM "$STAGE_PID" 2>/dev/null
  wait "$STAGE_PID" 2>/dev/null || true
  STAGE_PID=0
fi

echo "CROSS-FLOOR SCAN TASK PASS (${MISSION})：${F1_CAR} -> ${DOG} -> ${F2_CAR}"
if [[ "$SHOW_RVIZ" == "1" ]]; then
  echo "RViz 将保持打开；关闭 RViz 即结束任务。"
  wait "$RVIZ_PID"
fi
