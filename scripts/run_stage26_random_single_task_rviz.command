#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"
PYTHON_BIN="$TASK_ROOT/.pixi/envs/default/bin/python"
PLAN_FILE="$WORK_ROOT/results/stage26_recurrent_causal/rviz_single_task/latest_plan.json"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_stage26_single_task_rviz_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export ROS_DOMAIN_ID=$((100 + $$ % 100))
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source "$SCAN_ROOT/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

PREPARE=(
  "$PYTHON_BIN"
  "$WORK_ROOT/scripts/prepare_stage26_random_single_task.py"
  --output "$PLAN_FILE"
)
if [[ -n "${1:-}" ]]; then
  PREPARE+=(--task-seed "$1")
fi

echo "[Stage 26] 随机抽取且只发布一个任务……"
"${PREPARE[@]}"
echo "[Stage 26] 计划文件：$PLAN_FILE"
echo "[Stage 26] RViz 将显示全部 10 个智能体和一次完整执行；未选中智能体原地待命。"
echo "[Stage 26] 关闭 RViz 窗口即可结束全部进程。"

ros2 launch warehouse_bringup stage26_random_single_task_rviz.launch.py \
  plan_file:="$PLAN_FILE" time_scale:="${TASK2_TIME_SCALE:-1.0}"
