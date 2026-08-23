#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
WEIGHT="${1:-$WORK_ROOT/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt}"
EPISODES="${2:-100}"
OUTPUT="$WORK_ROOT/results/stage15/stage15_three_way_${EPISODES}ep.json"

echo "Stage 15预注册场景：MEDIUM / DENSE / CROSS_HEAVY / BURST"
echo "每场景：$EPISODES episode；PPO、时间规则和风险规则使用同任务指纹"
echo "权重：$WEIGHT"
echo "输出：$OUTPUT"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage15_stress_evaluation.py" \
  --weights "$WEIGHT" \
  --episodes "$EPISODES" \
  --seed 20266001 \
  --scenarios MEDIUM DENSE CROSS_HEAVY BURST \
  --output "$OUTPUT"
