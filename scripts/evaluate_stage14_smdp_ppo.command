#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
WEIGHT_PATH="${1:?请把权重文件路径作为第一个参数}"
EXECUTION_MODE="${2:-AUTO}"
ARRIVAL_PROFILE="${3:-AUTO}"
EPISODES="${4:-30}"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/evaluate_stage14_smdp_ppo.py" \
  --weights "$WEIGHT_PATH" \
  --execution-mode "$EXECUTION_MODE" \
  --arrival-profile "$ARRIVAL_PROFILE" \
  --episodes "$EPISODES"

