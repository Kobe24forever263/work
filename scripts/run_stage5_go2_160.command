#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
OUTPUT="$WORK_ROOT/results/stage5/stage5_go2_fixed_160.jsonl"
PYTHON="$TASK_ROOT/.pixi/envs/default/bin/python"

cd "$TASK_ROOT"
eval "$(/opt/homebrew/bin/pixi shell-hook -s zsh)"
source "$TASK_ROOT/SCAN-Planner-Ros2/install/setup.zsh"
source "$WORK_ROOT/install/setup.zsh"
export AMENT_PREFIX_PATH="$WORK_ROOT/install/warehouse_bringup:$AMENT_PREFIX_PATH"
export ROS_LOG_DIR=/private/tmp/warehouse_stage5_go2_160_logs
export DYLD_LIBRARY_PATH="$TASK_ROOT/.pixi/envs/default/lib"
export DYLD_INSERT_LIBRARIES="$TASK_ROOT/py312_shim.dylib"
export ROS_DOMAIN_ID=177
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "[环境自检] scan_planner 与 warehouse_bringup"
ros2 pkg prefix scan_planner >/dev/null
ros2 pkg prefix warehouse_bringup >/dev/null
echo "[路线自检] PCD 与八条 Go2 上下楼路线"
"$PYTHON" "$WORK_ROOT/scripts/validate_scan_assets.py"
echo "[计划自检] 4 条楼梯 × 2 个方向 × 20 次"
"$PYTHON" "$WORK_ROOT/scripts/run_stage5_batch.py" \
  --repeats 20 --initial-jitter 0.0 --schedule-only

if [[ -e "$OUTPUT" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "$OUTPUT" "${OUTPUT%.jsonl}_interrupted_${stamp}.jsonl"
fi

"$PYTHON" -u "$WORK_ROOT/scripts/run_stage5_batch.py" \
  --repeats 20 --initial-jitter 0.0 --fail-fast --output "$OUTPUT"
