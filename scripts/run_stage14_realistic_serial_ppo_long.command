#!/bin/zsh
set -euo pipefail

# SERIAL正式训练入口：同一时间最多一个配送任务在途，但队列异步到达且位置不重置。
WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
RESUME_PATH="${1:-}"
UPDATE_COUNT="${2:-2000}"
OUTPUT_DIR="$WORK_ROOT/results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2"
OUTPUT_WEIGHT="$OUTPUT_DIR/stage14_realistic_serial_medium_reward_v2_context_v2.pt"
WARM_START_WEIGHT="$WORK_ROOT/results/stage14/context_warm_start/stage14_context_policy_warm_start.pt"

echo "正式场景：SERIAL / MEDIUM"
echo "每个episode仍有20个异步任务；最多1个任务在途，机器人状态跨任务保留"
echo "同一策略共同选择 SINGLE_CAR、SINGLE_DOG、CAR_DOG_CAR"
echo "最终权重：$OUTPUT_WEIGHT"
echo "最佳权重：${OUTPUT_WEIGHT:r}.best.pt"

RESUME_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
  echo "从检查点继续：$RESUME_PATH（本次追加 $UPDATE_COUNT 个update）"
  RESUME_ARGS=(--resume "$RESUME_PATH")
else
  if [[ ! -f "$WARM_START_WEIGHT" ]]; then
    echo "缺少上下文策略预热权重：$WARM_START_WEIGHT"
    exit 1
  fi
  echo "从与并发策略相同的Context V2预热权重开始"
  RESUME_ARGS=(--warm-start "$WARM_START_WEIGHT")
fi

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage14_smdp_ppo.py" \
  --execution-mode SERIAL \
  --arrival-profile MEDIUM \
  --updates "$UPDATE_COUNT" \
  --episodes-per-update 4 \
  --ppo-epochs 4 \
  --minibatch-size 32 \
  --learning-rate 0.0003 \
  --eval-every 20 \
  --eval-episodes 10 \
  --checkpoint-every 20 \
  --log-every 20 \
  --seed 20263001 \
  "${RESUME_ARGS[@]}" \
  --output "$OUTPUT_WEIGHT"
