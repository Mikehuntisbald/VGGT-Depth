#!/usr/bin/env bash
# Launch (or resume) the bounded Spring Stage-C S6 arm only when a safe GPU is
# available. The scheduler is deliberately idempotent: an existing final.pt
# skips training and goes straight to evaluation; an existing latest.pt is
# resumed. No process is signalled by this script.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/spring_seed42_primary"
ARM_ROOT="$RUN_ROOT/corrected_plan/arms"
CACHE_ROOT="$RUN_ROOT/cache"
PYTHON_BIN="${PYTHON_BIN:-/home/CNF2026527811/miniconda3/envs/trtllm/bin/python}"
POLL_SEC="${POLL_SEC:-30}"
TRAIN_MIN_FREE_MIB="${TRAIN_MIN_FREE_MIB:-12000}"
# Stage-C evaluation has a larger temporary footprint than training. Keep a
# 12-GiB floor so a concurrent non-Spring process cannot consume the margin.
EVAL_MIN_FREE_MIB="${EVAL_MIN_FREE_MIB:-12000}"

S5_FINAL="$ARM_ROOT/S5/train/final.pt"
S6_ROOT="$ARM_ROOT/S6"
S6_TRAIN="$S6_ROOT/train"
S6_EVAL="$S6_ROOT/eval"
S6_TRAIN_FAILURE="$S6_TRAIN/failure_receipt.json"
S6_EVAL_FAILURE="$S6_EVAL/failure_receipt.json"
LOG_PATH="$S6_ROOT/dynamic_launcher.log"

source "$ROOT/tools/spring_scheduler_lib.sh"
mkdir -p "$S6_TRAIN" "$S6_EVAL" "$(dirname -- "$LOG_PATH")"
# Keep every launcher message and child stdout/stderr after a shell exits or
# the terminal disconnects. Python is explicitly unbuffered when redirected.
exec >>"$LOG_PATH" 2>&1
export PYTHONUNBUFFERED=1

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

# Prevent two copies of this watcher from racing on the same output tree.
# Existing pre-hardening copies do not hold this lock; GPU occupancy and the
# immediate lease recheck below still protect the active run.
exec 10>"$RUN_ROOT/corrected_plan/s6_dynamic.lock"
if ! flock -n 10; then
  log "Another S6 dynamic launcher owns the run lock; exiting without changes."
  exit 0
fi

CURRENT_STAGE="preflight"
CURRENT_GPU=""
FAILURE_WRITTEN=0

write_failure() {
  local rc="$1" path="$S6_TRAIN_FAILURE"
  [[ "$CURRENT_STAGE" == "eval" ]] && path="$S6_EVAL_FAILURE"
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$path" "FAILED" "$CURRENT_STAGE" "$rc" \
    "$CURRENT_GPU" "$LOG_PATH" "scheduler exited with code $rc" "" \
    || true
  FAILURE_WRITTEN=1
}

on_exit() {
  local rc=$?
  spring_release_gpu_lease || true
  if (( rc != 0 )) && (( FAILURE_WRITTEN == 0 )); then
    write_failure "$rc"
  fi
  exit "$rc"
}
trap on_exit EXIT

# If both terminal artifacts are already valid, this is a no-op. A malformed
# metrics file is not silently overwritten: it gets a durable failure receipt
# so an operator can inspect the partial output before retrying.
if [[ -s "$S6_TRAIN/final.pt" ]] && spring_metrics_ready "$PYTHON_BIN" "$S6_EVAL/metrics.json"; then
  log "S6 final.pt and valid metrics.json already exist; nothing to launch."
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$S6_EVAL_FAILURE" "COMPLETE" "eval" 0 "" "$LOG_PATH" \
    "outputs already complete" "" || true
  exit 0
fi
if [[ -e "$S6_EVAL/metrics.json" ]] && ! spring_metrics_ready "$PYTHON_BIN" "$S6_EVAL/metrics.json"; then
  log "S6 metrics.json exists but is not valid JSON; refusing to overwrite it."
  CURRENT_STAGE="eval"
  write_failure 2
  exit 2
fi

log "Waiting for S5 final checkpoint: $S5_FINAL"
while [[ ! -s "$S5_FINAL" ]]; do
  if [[ -s "$ARM_ROOT/S5/train/failure_receipt.json" ]]; then
    log "S5 training failure receipt found; S6 cannot start."
    CURRENT_STAGE="train"
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$S6_TRAIN_FAILURE" "BLOCKED" "train" 2 "" "$LOG_PATH" \
      "S5 failure receipt blocks S6" "" || true
    FAILURE_WRITTEN=1
    exit 2
  fi
  sleep "$POLL_SEC"
done

# Let a just-finished S5 process release its CUDA context and let nvidia-smi
# converge before selecting a card. This does not kill or alter that process.
sleep 10

TRAIN_CHECKPOINT="$S6_TRAIN/final.pt"
if [[ -s "$S6_TRAIN/final.pt" ]]; then
  log "S6 final.pt exists; skipping S6 training and evaluating that checkpoint."
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$S6_TRAIN_FAILURE" "COMPLETE" "train" 0 "" "$LOG_PATH" \
    "final checkpoint already existed; training skipped" "" || true
else
  CURRENT_STAGE="train"
  TRAIN_ARGS=(--init-from "$S5_FINAL")
  TRAIN_STEPS=5000
  if [[ -s "$S6_TRAIN/latest.pt" ]]; then
    TRAIN_ARGS=(--resume "$S6_TRAIN/latest.pt")
    latest_step="$(spring_checkpoint_step "$PYTHON_BIN" "$S6_TRAIN/latest.pt" || true)"
    configured_steps="$(spring_checkpoint_configured_steps "$PYTHON_BIN" "$S6_TRAIN/latest.pt" || true)"
    [[ "$configured_steps" =~ ^[0-9]+$ ]] && TRAIN_STEPS="$configured_steps"
    if [[ ! "$latest_step" =~ ^[0-9]+$ ]] || (( latest_step >= TRAIN_STEPS )); then
      log "S6 latest.pt has invalid/completed step ($latest_step/$TRAIN_STEPS); refusing to guess a resume segment."
      write_failure 2
      exit 2
    fi
    TRAIN_STEPS=$((TRAIN_STEPS - latest_step))
    log "S6 latest.pt exists at step $latest_step/$((latest_step + TRAIN_STEPS)); resuming for $TRAIN_STEPS step(s)."
  else
    log "No S6 checkpoint exists; starting from S5 final.pt."
  fi

  TRAIN_GPU=""
  while [[ -z "$TRAIN_GPU" ]]; do
    TRAIN_GPU="$(spring_select_gpu "$ROOT" "$TRAIN_MIN_FREE_MIB" || true)"
    if [[ -z "$TRAIN_GPU" ]]; then
      log "No arm-free GPU with >=${TRAIN_MIN_FREE_MIB} MiB free; retrying S6 train selection."
      sleep "$POLL_SEC"
      continue
    fi
    if ! spring_acquire_gpu_lease "$RUN_ROOT" "$TRAIN_GPU" "S6-train"; then
      log "GPU $TRAIN_GPU lease is held by another Spring watcher; reselecting."
      TRAIN_GPU=""
      sleep 1
      continue
    fi
    # Query again after acquiring the lease: the lease serializes cooperating
    # watchers, while this check catches legacy/manual processes that predate
    # the lease protocol.
    if ! spring_gpu_is_safe "$ROOT" "$TRAIN_GPU" "$TRAIN_MIN_FREE_MIB"; then
      log "GPU $TRAIN_GPU became occupied after lease acquisition; reselecting."
      spring_release_gpu_lease
      TRAIN_GPU=""
      sleep 1
      continue
    fi
  done

  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$S6_TRAIN_FAILURE" "RUNNING" "train" 0 "$TRAIN_GPU" "$LOG_PATH" \
    "S6 training started" "" || true
  log "Launching S6 train on physical GPU $TRAIN_GPU"
  if CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" \
      "$ROOT/tools/train_spring_epipolar.py" \
      --config "$ROOT/configs/spring_corrected/S6.yaml" \
      "${TRAIN_ARGS[@]}" \
      --manifest "$RUN_ROOT/manifests/train.jsonl" \
      --observation-cache-root "$CACHE_ROOT/train/observation" \
      --teacher-cache-root "$CACHE_ROOT/train/teacher" \
      --derived-cache-root "$CACHE_ROOT/train/derived_vggt_pose_depth" \
      --rectification-audit "$ROOT/reports/spring_epipolar_rectification_primary.json" \
      --output "$S6_TRAIN" --device cuda --spring-screening --run-steps "$TRAIN_STEPS" \
      >>"$LOG_PATH" 2>&1; then
    :
  else
    rc=$?
    log "S6 train failed with exit code $rc; evaluation will not be started."
    spring_write_scheduler_receipt \
      "$PYTHON_BIN" "$S6_TRAIN_FAILURE" "FAILED" "train" "$rc" "$TRAIN_GPU" "$LOG_PATH" \
      "S6 train command failed" "" || true
    FAILURE_WRITTEN=1
    spring_release_gpu_lease
    exit "$rc"
  fi
  spring_release_gpu_lease
  [[ -s "$S6_TRAIN/final.pt" ]] || {
    log "S6 train exited without final.pt; evaluation will not be started."
    CURRENT_STAGE="train"
    write_failure 1
    exit 1
  }
  TRAIN_CHECKPOINT="$S6_TRAIN/final.pt"
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$S6_TRAIN_FAILURE" "COMPLETE" "train" 0 "$TRAIN_GPU" "$LOG_PATH" \
    "S6 training completed" "" || true
fi

# A second invocation may arrive after S6 eval started but before metrics.json
# was visible. The per-GPU lease and occupancy check make it wait for that
# process instead of launching a duplicate evaluator.
if spring_metrics_ready "$PYTHON_BIN" "$S6_EVAL/metrics.json"; then
  log "S6 metrics appeared while scheduling; skipping duplicate evaluation."
  exit 0
fi

CURRENT_STAGE="eval"
CURRENT_GPU=""
EVAL_GPU=""
while [[ -z "$EVAL_GPU" ]]; do
  EVAL_GPU="$(spring_select_gpu "$ROOT" "$EVAL_MIN_FREE_MIB" || true)"
  if [[ -z "$EVAL_GPU" ]]; then
    log "No arm-free GPU with >=${EVAL_MIN_FREE_MIB} MiB free for S6 eval; retrying."
    sleep "$POLL_SEC"
    continue
  fi
  if ! spring_acquire_gpu_lease "$RUN_ROOT" "$EVAL_GPU" "S6-eval"; then
    log "GPU $EVAL_GPU lease is held by another Spring watcher; reselecting."
    EVAL_GPU=""
    sleep 1
    continue
  fi
  if ! spring_gpu_is_safe "$ROOT" "$EVAL_GPU" "$EVAL_MIN_FREE_MIB"; then
    log "GPU $EVAL_GPU became occupied after lease acquisition; reselecting."
    spring_release_gpu_lease
    EVAL_GPU=""
    sleep 1
    continue
  fi
done

spring_write_scheduler_receipt \
  "$PYTHON_BIN" "$S6_EVAL_FAILURE" "RUNNING" "eval" 0 "$EVAL_GPU" "$LOG_PATH" \
  "S6 evaluation started" "" || true
log "Launching S6 eval on physical GPU $EVAL_GPU"
if CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" \
    "$ROOT/tools/eval_spring_epipolar.py" \
    --config "$ROOT/configs/spring_corrected/S6.yaml" \
    --checkpoint "$TRAIN_CHECKPOINT" \
    --base-checkpoint "$S5_FINAL" \
    --manifest "$RUN_ROOT/manifests/validation.jsonl" \
    --observation-cache-root "$CACHE_ROOT/validation/observation" \
    --teacher-cache-root "$CACHE_ROOT/validation/teacher" \
    --derived-cache-root "$CACHE_ROOT/validation/derived_vggt_pose_depth" \
    --rectification-audit "$ROOT/reports/spring_epipolar_rectification_primary.json" \
    --output "$S6_EVAL" --device cuda --batch-size 1 --num-workers 0 \
    --visualization-samples 0 --limit 1350 --spring-screening \
    >>"$LOG_PATH" 2>&1; then
  :
else
  rc=$?
  if spring_metrics_ready "$PYTHON_BIN" "$S6_EVAL/metrics.json"; then
    # A manual/legacy evaluator may have won the output race as this command
    # exited (for example, the canonical writer noticed an existing output).
    log "S6 eval exited $rc, but a valid metrics.json appeared; accepting it."
    spring_release_gpu_lease
    exit 0
  fi
  log "S6 eval failed with exit code $rc."
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$S6_EVAL_FAILURE" "FAILED" "eval" "$rc" "$EVAL_GPU" "$LOG_PATH" \
    "S6 eval command failed" "" || true
  FAILURE_WRITTEN=1
  spring_release_gpu_lease
  exit "$rc"
fi
spring_release_gpu_lease

if ! spring_metrics_ready "$PYTHON_BIN" "$S6_EVAL/metrics.json"; then
  log "S6 eval exited successfully but did not produce valid metrics.json."
  write_failure 1
  exit 1
fi
spring_write_scheduler_receipt \
  "$PYTHON_BIN" "$S6_EVAL_FAILURE" "COMPLETE" "eval" 0 "$EVAL_GPU" "$LOG_PATH" \
  "S6 training/evaluation completed" "" || true
log "S6 train/eval completed."
