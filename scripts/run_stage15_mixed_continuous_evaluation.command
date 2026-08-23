#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
WEIGHT="${1:-$WORK_ROOT/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt}"
EPISODES="${2:-100}"
OUTPUT="$WORK_ROOT/results/stage15/stage15_mixed_continuous_${EPISODES}ep.json"

echo "Stage 15连续混合负载：NORMAL 20任务 -> DENSE 20任务 -> BURST 20任务"
echo "每个episode内机器人、队列和资源状态均不重置"
echo "三方对照：PPO / 时间规则 / 风险规则"
echo "输出：$OUTPUT"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage15_stress_evaluation.py" \
  --weights "$WEIGHT" \
  --episodes "$EPISODES" \
  --seed 20266201 \
  --scenarios MIXED_CONTINUOUS \
  --output "$OUTPUT"
