#!/bin/zsh
set -euo pipefail

cd /Users/lab4099/Desktop/Mujoco/work
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
TASK21_PYTHON=/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python

"$TASK21_PYTHON" scripts/run_stage21_latency_scaling.py "$@"
