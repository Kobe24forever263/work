#!/bin/zsh
set -euo pipefail

WORK_ROOT="/Users/lab4099/Desktop/Mujoco/work"
PYTHON="/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python"

cd "$WORK_ROOT"

if [[ "${1:-}" == "--smoke" ]]; then
  "$PYTHON" scripts/run_stage22_algorithm_ablation_interface_gate.py
  "$PYTHON" scripts/run_stage22_algorithm_ablation_smokes.py
  exit 0
fi

echo "Formal training is manual:"
echo "$PYTHON scripts/run_stage22_algorithm_ablation.py CONDITION SEED_INDEX --execute --confirm-long"
echo "CONDITION: fixed_gamma | no_context_interaction | no_warm_start | no_relay"
echo "SEED_INDEX: 1..10"
echo ""
echo "Guarded campaign launcher (plans unless both confirmation flags are present):"
echo "$PYTHON scripts/run_stage22_algorithm_ablation_campaign.py"
echo "$PYTHON scripts/run_stage22_algorithm_ablation_campaign.py --conditions CONDITION [CONDITION ...] --seed-start 1 --seed-end 10 --execute --confirm-long"
