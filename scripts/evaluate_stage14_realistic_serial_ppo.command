#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
DEFAULT_WEIGHT="$WORK_ROOT/results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/stage14_realistic_serial_medium_reward_v2_context_v2.best.pt"
WEIGHT="${1:-$DEFAULT_WEIGHT}"
EPISODES="${2:-30}"
OUTPUT="$WORK_ROOT/results/stage15/stage15_serial_medium_three_way_${EPISODES}ep.json"

echo "SERIAL三方评估：PPO / 时间规则 / 风险规则"
echo "每个episode 20个异步任务，最多1个任务在途"
echo "输出：$OUTPUT"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage15_stress_evaluation.py" \
  --weights "$WEIGHT" \
  --execution-mode SERIAL \
  --episodes "$EPISODES" \
  --seed 20266401 \
  --scenarios MEDIUM \
  --output "$OUTPUT"
