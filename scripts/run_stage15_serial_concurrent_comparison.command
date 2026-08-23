#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
SERIAL_WEIGHT="$WORK_ROOT/results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/stage14_realistic_serial_medium_reward_v2_context_v2.best.pt"
CONCURRENT_WEIGHT="$WORK_ROOT/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt"
EPISODES="${1:-100}"
OUTPUT="$WORK_ROOT/results/stage15/stage15_serial_vs_concurrent_${EPISODES}ep.json"

echo "SERIAL与CONCURRENT PPO同任务流配对消融"
echo "场景：MEDIUM + DENSE + BURST（每个episode 20任务）"
echo "输出：$OUTPUT"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage15_serial_concurrent_comparison.py" \
  --serial-weights "$SERIAL_WEIGHT" \
  --concurrent-weights "$CONCURRENT_WEIGHT" \
  --episodes "$EPISODES" \
  --seed 20266501 \
  --scenarios MEDIUM DENSE BURST \
  --output "$OUTPUT"
