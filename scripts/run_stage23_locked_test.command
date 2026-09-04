#!/bin/zsh
set -euo pipefail

ROOT="/Users/lab4099/Desktop/Mujoco/work"
ENV_ROOT="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default"
TORCH_LIB="$ENV_ROOT/lib/python3.12/site-packages/torch/lib"

export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

cd "$ROOT"
exec "$ENV_ROOT/bin/python" scripts/run_stage23_locked_test.py "$@"
