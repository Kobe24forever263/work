#!/bin/zsh
set -euo pipefail

cd /Users/lab4099/Desktop/Mujoco/work
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK21_LOCKED_PYTHON=/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python

"$TASK21_LOCKED_PYTHON" scripts/run_stage21_fault_locked_campaign.py \
  --execute --confirm-locked-test --groups BASELINES \
  --weight-start 1 --weight-end 1
"$TASK21_LOCKED_PYTHON" scripts/run_stage21_fault_locked_campaign.py \
  --execute --confirm-locked-test --groups PPO \
  --weight-start 1 --weight-end 10
"$TASK21_LOCKED_PYTHON" scripts/summarize_stage21_fault_locked.py
