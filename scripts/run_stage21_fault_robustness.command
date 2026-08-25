#!/bin/zsh
set -euo pipefail

cd /Users/lab4099/Desktop/Mujoco/work
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK21_FAULT_PYTHON=/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python

"$TASK21_FAULT_PYTHON" scripts/run_stage21_fault_robustness.py "$@"
