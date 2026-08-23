#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"

cd "$WORK_ROOT"
exec "$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage18_locked_test_v2.py"
