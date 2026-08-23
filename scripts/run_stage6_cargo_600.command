#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
OUTPUT="$WORK_ROOT/results/stage6/stage6_cargo_fixed_600.jsonl"
PYTHON="$TASK_ROOT/.pixi/envs/default/bin/python"

cd "$TASK_ROOT"
eval "$(/opt/homebrew/bin/pixi shell-hook -s zsh)"
source "$TASK_ROOT/SCAN-Planner-Ros2/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"

export ROS_LOG_DIR=/private/tmp/warehouse_stage6_cargo_600_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export ROS_DOMAIN_ID=176
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "[环境自检] scan_planner 与 warehouse_bringup"
"$TASK_ROOT/.pixi/envs/default/bin/ros2" pkg prefix scan_planner >/dev/null
"$TASK_ROOT/.pixi/envs/default/bin/ros2" pkg prefix warehouse_bringup >/dev/null

echo "[预检 1/2] 检查八条货运路线"
"$PYTHON" "$WORK_ROOT/scripts/validate_stage6_routes.py"
echo "[预检 2/2] 检查 600 次任务分配"
"$PYTHON" "$WORK_ROOT/scripts/run_stage6_batch.py" --schedule-only

if [[ -e "$OUTPUT" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  backup="${OUTPUT%.jsonl}_interrupted_${stamp}.jsonl"
  mv "$OUTPUT" "$backup"
  echo "已把旧的未完成结果保存为：$backup"
fi

echo "开始 600 次固定货运路线测试；按 Control+C 可安全停止。"
"$PYTHON" -u "$WORK_ROOT/scripts/run_stage6_batch.py" \
  --trials-per-robot 100 \
  --fail-fast \
  --output "$OUTPUT"
