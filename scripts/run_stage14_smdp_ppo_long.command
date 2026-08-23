#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
EXECUTION_MODE="${1:-MIXED}"
ARRIVAL_PROFILE="${2:-MEDIUM}"
RESUME_PATH="${3:-}"
UPDATE_COUNT="${4:-250}"

echo "注意：这是可选消融入口。正式场景请运行 run_stage14_realistic_ppo_long.command。"

case "$EXECUTION_MODE" in
  SERIAL|CONCURRENT|MIXED) ;;
  *) echo "执行模式必须是 SERIAL、CONCURRENT 或 MIXED" >&2; exit 2 ;;
esac
case "$ARRIVAL_PROFILE" in
  MEDIUM|DENSE) ;;
  *) echo "负载必须是 MEDIUM 或 DENSE" >&2; exit 2 ;;
esac

MODE_LOWER="${EXECUTION_MODE:l}"
PROFILE_LOWER="${ARRIVAL_PROFILE:l}"
OUTPUT_DIR="$WORK_ROOT/results/stage14/ppo/long/${MODE_LOWER}_${PROFILE_LOWER}"
OUTPUT_WEIGHT="$OUTPUT_DIR/stage14_smdp_ppo_${MODE_LOWER}_${PROFILE_LOWER}.pt"

echo "开始长时间训练：$EXECUTION_MODE / $ARRIVAL_PROFILE"
echo "最终权重：$OUTPUT_WEIGHT"
echo "最佳权重：${OUTPUT_WEIGHT:r}.best.pt"

RESUME_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
  echo "从检查点继续：$RESUME_PATH（本次追加 $UPDATE_COUNT 个update）"
  RESUME_ARGS=(--resume "$RESUME_PATH")
fi

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage14_smdp_ppo.py" \
  --execution-mode "$EXECUTION_MODE" \
  --arrival-profile "$ARRIVAL_PROFILE" \
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
