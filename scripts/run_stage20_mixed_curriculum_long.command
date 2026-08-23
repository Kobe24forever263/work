#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"

cd "$WORK_ROOT"
exec "$PYTHON_BIN" scripts/run_stage20_mixed_curriculum_long_batch.py \
  --confirm-long "$@"
