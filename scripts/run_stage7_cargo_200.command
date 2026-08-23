#!/bin/zsh
set -e

TASK_ROOT=/Users/lab4099/Desktop/Mujoco/task2_handover
WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
OUTPUT="$WORK_ROOT/results/stage7/stage7_cargo_ownership_fixed_200.jsonl"
PYTHON="$TASK_ROOT/.pixi/envs/default/bin/python"

cd "$TASK_ROOT"
eval "$(/opt/homebrew/bin/pixi shell-hook -s zsh)"
source "$WORK_ROOT/install/setup.zsh"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PYTHON" -m pytest -q "$WORK_ROOT/src/warehouse_core/test"
"$PYTHON" -u "$WORK_ROOT/scripts/run_stage7_cargo_ownership.py" \
  --repeats 50 --seed 20260804 --output "$OUTPUT"
