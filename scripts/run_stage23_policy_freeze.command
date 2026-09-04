#!/bin/zsh
set -euo pipefail

ROOT="/Users/lab4099/Desktop/Mujoco/work"
ENV_ROOT="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default"
TORCH_LIB="$ENV_ROOT/lib/python3.12/site-packages/torch/lib"

# This environment contains both a conda libomp and the wheel-bundled torch
# libomp.  Resolve both @rpath lookups to the same image instead of enabling
# the unsafe KMP_DUPLICATE_LIB_OK workaround.
export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

cd "$ROOT"
exec "$ENV_ROOT/bin/python" scripts/run_stage23_policy_freeze_gate.py "$@"
