#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
DEFAULT_WEIGHT="$WORK_ROOT/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt"
WEIGHT_PATH="${1:-$DEFAULT_WEIGHT}"
EPISODES="${2:-30}"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/evaluate_stage14_smdp_ppo.py" \
  --weights "$WEIGHT_PATH" \
  --execution-mode CONCURRENT \
  --arrival-profile MEDIUM \
  --episodes "$EPISODES"
