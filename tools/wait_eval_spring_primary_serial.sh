#!/usr/bin/env bash
# Evaluate corrected Spring temporal arms serially while respecting GPU
# ownership. Each arm has a durable scheduler receipt, so a failed evaluator
# is terminal and the compose watcher cannot wait forever for metrics.json.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/spring_seed42_primary"
ARMS="$RUN/corrected_plan/arms"
PYTHON_BIN="${PYTHON_BIN:-/home/CNF2026527811/miniconda3/envs/trtllm/bin/python}"
POLL_SEC="${POLL_SEC:-30}"
# The full evaluator needs more headroom than the temporal trainer.
MIN_FREE_MIB="${MIN_FREE_MIB:-15000}"
LOG_DIR="$RUN/corrected_plan/logs"
LOG_PATH="$LOG_DIR/serial_eval_queue.log"
QUEUE_FAILURE="$RUN/corrected_plan/serial_eval_queue_failure.json"

source "$ROOT/tools/spring_scheduler_lib.sh"
mkdir -p "$LOG_DIR" "$ARMS"
exec >>"$LOG_PATH" 2>&1
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

# One queue lock avoids two independently started watchers evaluating the same
# arm. This is separate from the per-GPU leases, which coordinate this queue
# with the S6 dynamic launcher.
exec 11>"$RUN/corrected_plan/serial_eval_queue.lock"
if ! flock -n 11; then
  log "Another serial evaluation queue owns the run lock; exiting."
  exit 0
fi

CURRENT_ARM=""
CURRENT_GPU=""
FAILURE_WRITTEN=0

write_current_failure() {
  local rc="$1" path="$ARMS/$CURRENT_ARM/eval/failure_receipt.json"
  [[ -n "$CURRENT_ARM" ]] || return 0
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$path" "FAILED" "eval" "$rc" "$CURRENT_GPU" \
    "$LOG_PATH" "serial evaluator exited with code $rc" "" || true
  FAILURE_WRITTEN=1
}

on_exit() {
  local rc=$?
  spring_release_gpu_lease || true
  if (( rc != 0 )) && (( FAILURE_WRITTEN == 0 )) && [[ -n "$CURRENT_ARM" ]]; then
    write_current_failure "$rc"
  fi
  exit "$rc"
}
trap on_exit EXIT

derived_root_for() {
  case "$1" in
    S2|S3) printf '%s\n' "$RUN/cache/validation/derived_gt_no_depth" ;;
    S4) printf '%s\n' "$RUN/cache/validation/derived_gt_pose_vggt_depth" ;;
    S5) printf '%s\n' "$RUN/cache/validation/derived_vggt_pose_depth" ;;
    *) return 1 ;;
  esac
}

mark_blocked_by_training() {
  local arm="$1"
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "BLOCKED" "eval" 2 "" \
    "$LOG_PATH" "training failure receipt blocks evaluation" "" || true
}

for arm in S2 S3 S4 S5; do
  while [[ ! -s "$ARMS/$arm/train/final.pt" ]]; do
    if spring_receipt_is_terminal_failure "$PYTHON_BIN" "$ARMS/$arm/train/failure_receipt.json"; then
      log "$arm training failure receipt found; marking evaluation BLOCKED."
      mark_blocked_by_training "$arm"
      break
    fi
    log "Waiting for $arm final checkpoint."
    sleep "$POLL_SEC"
  done
done

failures=0
for arm in S2 S3 S4 S5; do
  CURRENT_ARM="$arm"
  CURRENT_GPU=""
  # Re-arm the unexpected-error trap for each independent arm. Explicitly
  # handled failures below set this back to one only until the arm is cleared.
  FAILURE_WRITTEN=0
  metrics="$ARMS/$arm/eval/metrics.json"
  eval_dir="$ARMS/$arm/eval"

  if spring_metrics_ready "$PYTHON_BIN" "$metrics"; then
    log "$arm metrics already exist and are valid; skipping."
    CURRENT_ARM=""
    continue
  fi
  if spring_receipt_is_terminal_failure "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json"; then
    log "$arm has a terminal evaluation failure receipt; skipping retry."
    failures=$((failures + 1))
    CURRENT_ARM=""
    continue
  fi
  if [[ -e "$metrics" || -e "$ARMS/$arm/eval/metrics.csv" ]]; then
    log "$arm evaluation output is partial/corrupt; refusing to overwrite it."
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "FAILED" "eval" 2 "" \
      "$LOG_PATH" "metrics artifact exists but is not valid JSON" "" || true
    failures=$((failures + 1))
    CURRENT_ARM=""
    continue
  fi

  gpu=""
  while [[ -z "$gpu" ]]; do
    gpu="$(spring_select_gpu "$ROOT" "$MIN_FREE_MIB" || true)"
    if [[ -z "$gpu" ]]; then
      log "No arm-free GPU with >=${MIN_FREE_MIB} MiB free for $arm; retrying."
      sleep "$POLL_SEC"
      continue
    fi
    if ! spring_acquire_gpu_lease "$RUN" "$gpu" "${arm}-eval"; then
      log "GPU $gpu lease is held by another watcher; reselecting for $arm."
      gpu=""
      sleep 1
      continue
    fi
    # Lease acquisition serializes cooperating schedulers; this second query
    # catches manual/legacy processes which do not participate in the lease.
    if ! spring_gpu_is_safe "$ROOT" "$gpu" "$MIN_FREE_MIB"; then
      log "GPU $gpu became occupied after lease acquisition; reselecting."
      spring_release_gpu_lease
      gpu=""
      sleep 1
      continue
    fi
  done
  CURRENT_GPU="$gpu"

  arm_log="$LOG_DIR/${arm}_eval_serial.log"
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "RUNNING" "eval" 0 "$gpu" \
    "$arm_log" "$arm evaluation started" "" || true
  log "Evaluating $arm on physical GPU $gpu."
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/eval.py" \
      --config "$ROOT/configs/spring_corrected/$arm.yaml" \
      --checkpoint "$ARMS/$arm/train/final.pt" \
      --spatial-checkpoint "$ARMS/S1/train/final.pt" \
      --manifest "$RUN/manifests/validation.jsonl" \
      --observation-cache-root "$RUN/cache/validation/observation" \
      --teacher-cache-root "$RUN/cache/validation/teacher" \
      --derived-cache-root "$(derived_root_for "$arm")" \
      --output "$eval_dir" --device cuda --batch-size 1 --num-workers 0 \
      --visualization-samples 0 --failure-samples-per-criterion 0 \
      --crop-mode full --allow-non-holdout-smoke --limit 1350 --spring-native-metrics \
      >>"$arm_log" 2>&1; then
    :
  else
    rc=$?
    if spring_metrics_ready "$PYTHON_BIN" "$metrics"; then
      # A legacy/manual evaluator may have won the output race just as this
      # command exited (typically FileExistsError). Treat the valid artifact
      # as authoritative instead of recording a false failure.
      log "$arm command exited $rc, but a valid metrics.json appeared; accepting the artifact."
      spring_release_gpu_lease
      CURRENT_ARM=""
      CURRENT_GPU=""
      continue
    fi
    log "$arm evaluation failed with exit code $rc; see $arm_log."
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "FAILED" "eval" "$rc" "$gpu" \
      "$arm_log" "$arm evaluator command failed" "" || true
    failures=$((failures + 1))
    FAILURE_WRITTEN=1
    spring_release_gpu_lease
    CURRENT_ARM=""
    CURRENT_GPU=""
    continue
  fi
  spring_release_gpu_lease
  if spring_metrics_ready "$PYTHON_BIN" "$metrics"; then
    log "$arm evaluation completed."
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "COMPLETE" "eval" 0 "$gpu" \
      "$arm_log" "$arm metrics.json is valid" "" || true
  else
    log "$arm evaluator exited successfully but produced no valid metrics.json."
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$ARMS/$arm/eval/failure_receipt.json" "FAILED" "eval" 1 "$gpu" \
      "$arm_log" "$arm metrics.json is missing or malformed" "" || true
    failures=$((failures + 1))
  fi
  CURRENT_ARM=""
  CURRENT_GPU=""
done

if (( failures > 0 )); then
  log "Serial temporal evaluation queue finished with $failures failed/blocked arm(s)."
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$QUEUE_FAILURE" "FAILED" "queue" 1 "" "$LOG_PATH" \
    "$failures arm(s) failed or were blocked" "" || true
  FAILURE_WRITTEN=1
  exit 1
fi
log "Serial temporal evaluation queue finished successfully."
