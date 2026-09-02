#!/usr/bin/env bash
# Wait for corrected Spring arm evaluation terminals, then compose the report.
# A FAILED/BLOCKED scheduler receipt is terminal just like metrics.json, so a
# broken arm produces an explicit partial report instead of an infinite wait.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/spring_seed42_primary"
PYTHON_BIN="${PYTHON_BIN:-/home/CNF2026527811/miniconda3/envs/trtllm/bin/python}"
POLL_SEC="${POLL_SEC:-60}"
LOG_DIR="$RUN/corrected_plan/logs"
LOG_PATH="$LOG_DIR/compose_watcher.log"
COMPOSE_FAILURE="$RUN/corrected_plan/compose_failure_receipt.json"

source "$ROOT/tools/spring_scheduler_lib.sh"
mkdir -p "$LOG_DIR" "$RUN/corrected_plan"
exec >>"$LOG_PATH" 2>&1
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

exec 12>"$RUN/corrected_plan/compose_watcher.lock"
if ! flock -n 12; then
  log "Another compose watcher owns the run lock; exiting."
  exit 0
fi

terminal_for_arm() {
  local arm="$1" metrics="$RUN/corrected_plan/arms/$arm/eval/metrics.json"
  local eval_receipt="$RUN/corrected_plan/arms/$arm/eval/failure_receipt.json"
  local train_receipt="$RUN/corrected_plan/arms/$arm/train/failure_receipt.json"
  if spring_metrics_ready "$PYTHON_BIN" "$metrics"; then
    return 0
  fi
  # An evaluation receipt marked COMPLETE without metrics is an inconsistent
  # terminal state; do not wait forever for an artifact that cannot appear.
  if spring_receipt_is_terminal "$PYTHON_BIN" "$eval_receipt"; then
    return 0
  fi
  if [[ -e "$metrics" ]]; then
    # A malformed/partial artifact is terminal unless the evaluator explicitly
    # recorded RUNNING (which can be a zero-byte file race at startup).
    if spring_receipt_is_running "$PYTHON_BIN" "$eval_receipt"; then
      return 1
    fi
    return 0
  fi
  if spring_receipt_is_terminal_failure "$PYTHON_BIN" "$train_receipt"; then
    return 0
  fi
  return 1
}

log "Waiting for S2-S6 evaluation terminals (metrics or failure receipts)."
failures=0
while :; do
  ready=1
  failures=0
  for arm in S2 S3 S4 S5 S6; do
    if ! terminal_for_arm "$arm"; then
      ready=0
    fi
    if spring_receipt_is_terminal_failure "$PYTHON_BIN" "$RUN/corrected_plan/arms/$arm/eval/failure_receipt.json" || \
       spring_receipt_is_terminal_failure "$PYTHON_BIN" "$RUN/corrected_plan/arms/$arm/train/failure_receipt.json" || \
       { [[ -e "$RUN/corrected_plan/arms/$arm/eval/metrics.json" ]] &&
         ! spring_metrics_ready "$PYTHON_BIN" "$RUN/corrected_plan/arms/$arm/eval/metrics.json" &&
         ! spring_receipt_is_running "$PYTHON_BIN" "$RUN/corrected_plan/arms/$arm/eval/failure_receipt.json"; }; then
      failures=$((failures + 1))
    fi
  done
  (( ready )) && break
  log "Still waiting; terminal failures observed so far: $failures."
  sleep "$POLL_SEC"
done

if (( failures > 0 )); then
  log "All arm terminals present, including $failures failed/blocked arm(s); composing partial report."
else
  log "All corrected arm metrics receipts present; composing report."
fi

if "$PYTHON_BIN" "$ROOT/tools/compose_spring_screening_report.py" \
    --output-json "$ROOT/reports/spring_seed42_primary_corrected.json" \
    --output-md "$ROOT/reports/spring_seed42_primary_corrected.md" \
    --s0 "$RUN/corrected_plan/arms/S0/eval/metrics.json" \
    --arm-root "$RUN/corrected_plan/arms" \
    --train-manifest "$RUN/manifests/train.jsonl" \
    --manifest "$RUN/manifests/validation.jsonl" \
    --blocked "$RUN/corrected_plan/spring_seed42_summary.json" \
    >>"$LOG_PATH" 2>&1; then
  :
else
  rc=$?
  log "Report composition failed with exit code $rc."
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$COMPOSE_FAILURE" "FAILED" "compose" "$rc" "" "$LOG_PATH" \
    "compose_spring_screening_report.py failed" "" || true
  exit "$rc"
fi

if (( failures > 0 )); then
  # The report is useful and auditable, but the watcher still returns nonzero
  # so automation cannot mistake a partial matrix for a clean completion.
  spring_write_scheduler_receipt \
    "$PYTHON_BIN" "$COMPOSE_FAILURE" "BLOCKED" "compose" 1 "" "$LOG_PATH" \
    "$failures arm(s) failed or were blocked; report is partial" "" || true
  log "Partial report composed; returning failure to preserve the incomplete status."
  exit 1
fi
spring_write_scheduler_receipt \
  "$PYTHON_BIN" "$COMPOSE_FAILURE" "COMPLETE" "compose" 0 "" "$LOG_PATH" \
  "all arm metrics composed" "" || true
log "Report composition complete."
