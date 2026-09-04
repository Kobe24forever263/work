#!/bin/zsh
set -euo pipefail

WORK_ROOT=/Users/lab4099/Desktop/Mujoco/work
PYTHON_BIN=/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python
LONG_RUNNER="$WORK_ROOT/scripts/run_stage26_recurrent_long.py"
VALIDATOR="$WORK_ROOT/scripts/run_stage26_recurrent_validation.py"
RESULT_ROOT="$WORK_ROOT/results/stage26_recurrent_causal/long"
LOG_ROOT="$WORK_ROOT/results/stage26_recurrent_causal/cohort_04_10_logs"
DRY_RUN="${TASK2_DRY_RUN:-0}"

seed_state() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
summary_path = run_dir / "stage26_recurrent_ppo.summary.json"
weight_path = run_dir / "stage26_recurrent_ppo.pt"
validation_path = run_dir / "fresh_validation.json"
history_path = run_dir / "stage26_recurrent_ppo.history.json"
checkpoints = sorted((run_dir / "checkpoints").glob(
    "stage26_recurrent_ppo_update_*.pt"))

complete = False
if summary_path.is_file() and weight_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        complete = bool(
            summary.get("passed") is True and
            int(summary.get("updates", 0)) >= 2000 and
            int(summary.get("episodes_trained_total", 0)) >= 8000)
    except (OSError, ValueError, TypeError):
        complete = False

validated = False
if validation_path.is_file():
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validated = validation.get("passed") is True
    except (OSError, ValueError, TypeError):
        validated = False

history_updates = 0
try:
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            last_update = int(history[-1].get("update", -1))
            if len(history) == last_update:
                history_updates = len(history)
except (OSError, ValueError, TypeError, KeyError):
    history_updates = 0

matching_checkpoint = run_dir / "checkpoints" / (
    f"stage26_recurrent_ppo_update_{history_updates:05d}.pt")

if complete and history_updates < 2000 and matching_checkpoint.is_file():
    print(f"REPAIR_HISTORY|{matching_checkpoint}")
elif complete and validated and history_updates >= 2000:
    print("COMPLETE_VALIDATED")
elif complete and history_updates >= 2000:
    print("COMPLETE_NEEDS_VALIDATION")
elif checkpoints:
    print("PARTIAL")
else:
    print("NEW")
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "Stage 26 cohort error: Python environment not found: $PYTHON_BIN"
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_ROOT"
  if pgrep -f "$WORK_ROOT/scripts/run_stage26_recurrent_ppo.py" >/dev/null 2>&1; then
    print -u2 "Stage 26 cohort error: another recurrent PPO trainer is running."
    print -u2 "Stop it or wait for it to finish before starting seed 03 recovery and seeds 04-10."
    exit 3
  fi
fi

cd "$WORK_ROOT"
export PYTHONUNBUFFERED=1

print "Stage 26 recurrent cohort: repair seed 03, then seeds 04-10"
print "Execution: sequential; seed 03 resumes from its matching history checkpoint."
print "Budget after seed 03 recovery: 7 seeds x 2000 updates x 4 episodes/update."
if [[ "$DRY_RUN" == "1" ]]; then
  print "Mode: DRY RUN (no training or validation will execute)."
fi

for seed in {3..10}; do
  seed_tag=$(printf "%02d" "$seed")
  run_dir="$RESULT_ROOT/seed_$seed_tag"
  current_state=$(seed_state "$run_dir")
  print ""
  print "[seed $seed_tag/10] state=$current_state"

  if [[ "$current_state" == "COMPLETE_VALIDATED" ]]; then
    print "[seed $seed_tag/10] training and fresh validation already passed; skip."
    continue
  fi

  if [[ "$current_state" != "COMPLETE_NEEDS_VALIDATION" ]]; then
    resume_args=()
    if [[ "$current_state" == REPAIR_HISTORY\|* ]]; then
      repair_checkpoint="${current_state#REPAIR_HISTORY|}"
      resume_args=(--resume-checkpoint "$repair_checkpoint")
      print "[seed $seed_tag/10] truncated history detected; resume from $repair_checkpoint."
    elif [[ "$current_state" == "PARTIAL" ]]; then
      resume_args=(--resume-latest)
      print "[seed $seed_tag/10] latest checkpoint found; resume enabled."
    else
      print "[seed $seed_tag/10] starting a new independent run."
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      print "$PYTHON_BIN $LONG_RUNNER $seed ${resume_args[*]} --execute --confirm-long"
    else
      train_log="$LOG_ROOT/seed_${seed_tag}_train.log"
      print "[seed $seed_tag/10] training log: $train_log"
      "$PYTHON_BIN" "$LONG_RUNNER" "$seed" \
        "${resume_args[@]}" --execute --confirm-long 2>&1 | tee "$train_log"
    fi
  else
    print "[seed $seed_tag/10] long training already complete; validation only."
  fi

  if [[ "$seed" == "3" && "$DRY_RUN" != "1" ]]; then
    recovery_weight="$run_dir/recovery_before_resume_20260902/stage26_recurrent_ppo.pt"
    reconstructed_weight="$run_dir/stage26_recurrent_ppo.pt"
    if [[ -f "$recovery_weight" ]]; then
      recovery_hash=$(shasum -a 256 "$recovery_weight" | awk '{print $1}')
      reconstructed_hash=$(shasum -a 256 "$reconstructed_weight" | awk '{print $1}')
      if [[ "$recovery_hash" != "$reconstructed_hash" ]]; then
        print -u2 "[seed 03/10] deterministic recovery hash mismatch."
        print -u2 "Original backup remains at: $recovery_weight"
        exit 5
      fi
      print "[seed 03/10] deterministic final-weight hash matches the original backup."
    fi
  fi

  validation_output="$run_dir/fresh_validation.json"
  if [[ "$DRY_RUN" == "1" ]]; then
    print "$PYTHON_BIN $VALIDATOR $run_dir/stage26_recurrent_ppo.pt --seed-start 73250000 --episodes 40 --output $validation_output"
  else
    validation_log="$LOG_ROOT/seed_${seed_tag}_validation.log"
    print "[seed $seed_tag/10] fresh validation log: $validation_log"
    "$PYTHON_BIN" "$VALIDATOR" \
      "$run_dir/stage26_recurrent_ppo.pt" \
      --seed-start 73250000 --episodes 40 \
      --output "$validation_output" 2>&1 | tee "$validation_log"
    verified_state=$(seed_state "$run_dir")
    if [[ "$verified_state" != "COMPLETE_VALIDATED" ]]; then
      print -u2 "[seed $seed_tag/10] validation did not produce a passed result."
      exit 4
    fi
    print "[seed $seed_tag/10] COMPLETE_VALIDATED"
  fi
done

print ""
if [[ "$DRY_RUN" == "1" ]]; then
  print "Stage 26 seed 03 recovery and seeds 04-10 dry-run completed."
else
  print "Stage 26 seed 03 recovery and seeds 04-10 training and fresh validation completed."
  print "Weights: $RESULT_ROOT/seed_03 through seed_10"
  print "Logs: $LOG_ROOT"
fi
