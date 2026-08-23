#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
PYTHON="$TASK_ROOT/.pixi/envs/default/bin/python"
OUTPUT="$WORK_ROOT/results/stage9/stage9_failure_recovery_fixed_350.jsonl"

cd "$TASK_ROOT"
eval "$(/opt/homebrew/bin/pixi shell-hook -s zsh)"
source "$WORK_ROOT/install/setup.zsh"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"$PYTHON" -m pytest -q -p no:cacheprovider "$WORK_ROOT/src/warehouse_core/test"
"$PYTHON" -u "$WORK_ROOT/scripts/run_stage9_failure_recovery.py" \
  --per-fault 50 --seed 20260804 --output "$OUTPUT"

