#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON_BIN="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"

cd "$WORK_ROOT"
"$PYTHON_BIN" "$WORK_ROOT/scripts/run_stage18_single_dog_baseline.py"
"$PYTHON_BIN" "$WORK_ROOT/scripts/summarize_stage18_locked_test_v2.py"
