#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"
WARM_START_WEIGHT="$WORK_ROOT/results/stage14/context_warm_start/stage14_context_policy_warm_start.pt"
OUTPUT_DIR="$WORK_ROOT/results/stage14/ppo/smoke_serial_context_v2"
OUTPUT_WEIGHT="$OUTPUT_DIR/stage14_serial_context_v2_smoke.pt"

echo "SERIAL短门禁：5 updates × 2 episodes = 10 episodes"
echo "该结果只验证接口，不代表收敛或正式性能"

exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage14_smdp_ppo.py" \
  --execution-mode SERIAL \
  --arrival-profile MEDIUM \
  --updates 5 \
  --episodes-per-update 2 \
  --ppo-epochs 4 \
  --minibatch-size 32 \
  --learning-rate 0.0003 \
  --eval-every 1 \
  --eval-episodes 5 \
  --checkpoint-every 5 \
  --log-every 1 \
  --seed 20264001 \
  --warm-start "$WARM_START_WEIGHT" \
  --output "$OUTPUT_WEIGHT"
