#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
SCAN_ROOT="$TASK_ROOT/SCAN-Planner-Ros2"
PYTHON_BIN="$TASK_ROOT/.pixi/envs/default/bin/python"
CAMPAIGN_FILE="$WORK_ROOT/results/stage26_recurrent_causal/rviz_scan_campaign/latest_campaign.json"

cd "$TASK_ROOT"
export ROS_LOG_DIR=/private/tmp/warehouse_stage26_scan_campaign_logs
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
  "$WORK_ROOT/scripts/prepare_stage26_scan_campaign.py"
  --task-count 10
  --output "$CAMPAIGN_FILE"
)
if [[ -n "${1:-}" ]]; then
  PREPARE+=(--campaign-seed "$1")
fi

echo "[Stage 26] 生成 10 个连续随机任务，并用最佳已完成种子决策……"
"${PREPARE[@]}"
echo "[Stage 26] 启动真实 SCAN-Planner；RViz 将显示活动机器人的局部雷达点云。"
"$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage26_scan_campaign.py" \
  --campaign "$CAMPAIGN_FILE" --task-gap "${TASK2_TASK_GAP:-2.0}" \
  --start-task "${TASK2_START_TASK:-1}" \
  --start-segment "${TASK2_START_SEGMENT:-1}"
