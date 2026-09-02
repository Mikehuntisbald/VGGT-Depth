#!/usr/bin/env bash
# Small, dependency-light helpers shared by the Spring scheduler watchers.
#
# This file is sourced by bash launchers. It deliberately does not start,
# stop, or signal any process. GPU occupancy is checked with nvidia-smi and a
# per-run advisory lease closes the query-to-launch race between watchers.

SPRING_GPU_LEASE_FD=""
SPRING_GPU_LEASE_PATH=""

spring_arm_pids_on_gpu() {
  local root="$1" gpu="$2"
  local uuid row_uuid pid cmd spring_cmd
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null \
    | sed -n "$((gpu + 1))p" | tr -d '[:space:]')"
  [[ -n "$uuid" ]] || return 0
  while IFS=',' read -r row_uuid pid; do
    row_uuid="${row_uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    [[ "$row_uuid" == "$uuid" && "$pid" =~ ^[0-9]+$ ]] || continue
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    # Restrict generic train.py/eval.py matches to this run or to the explicit
    # corrected Spring config. The config test also catches legacy/manual
    # launchers that used relative paths (and therefore omitted ``$root``).
    spring_cmd=0
    [[ "$cmd" == *"$root"* || "$cmd" == *"configs/spring_corrected/"* ]] && spring_cmd=1
    if (( spring_cmd )) && [[ "$cmd" == *"train.py"* || \
          "$cmd" == *"eval.py"* || \
          "$cmd" == *"train_spring_epipolar.py"* || \
          "$cmd" == *"eval_spring_epipolar.py"* ]]; then
      printf '%s\n' "$pid"
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid \
    --format=csv,noheader,nounits 2>/dev/null || true)
}

spring_gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    2>/dev/null \
    | awk -F',' -v wanted="$gpu" '{
        gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2);
        if ($1 == wanted) print $2
      }'
}

spring_gpu_is_safe() {
  local root="$1" gpu="$2" minimum="$3" free
  free="$(spring_gpu_free_mib "$gpu" || true)"
  [[ "$free" =~ ^[0-9]+$ ]] || return 1
  (( free >= minimum )) || return 1
  [[ -z "$(spring_arm_pids_on_gpu "$root" "$gpu")" ]]
}

spring_select_gpu() {
  local root="$1" minimum="$2"
  local best=-1 best_free=-1 idx free
  while IFS=',' read -r idx free; do
    idx="${idx//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    [[ "$idx" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
    (( free >= minimum )) || continue
    [[ -z "$(spring_arm_pids_on_gpu "$root" "$idx")" ]] || continue
    if (( free > best_free )); then
      best="$idx"
      best_free="$free"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits 2>/dev/null || true)
  (( best >= 0 )) && printf '%s\n' "$best"
}

# Acquire a per-run/per-physical-GPU advisory lease. Callers must run
# spring_gpu_is_safe again while this lease is held immediately before exec.
spring_acquire_gpu_lease() {
  local run_root="$1" gpu="$2" owner="${3:-spring-scheduler}"
  local lease_root="$run_root/corrected_plan/gpu_leases"
  mkdir -p "$lease_root"
  SPRING_GPU_LEASE_PATH="$lease_root/gpu-${gpu}.lock"
  exec {SPRING_GPU_LEASE_FD}>"$SPRING_GPU_LEASE_PATH"
  if ! flock -n "$SPRING_GPU_LEASE_FD"; then
    eval "exec ${SPRING_GPU_LEASE_FD}>&-"
    SPRING_GPU_LEASE_FD=""
    SPRING_GPU_LEASE_PATH=""
    return 1
  fi
  printf 'pid=%s owner=%s gpu=%s started_utc=%s\n' \
    "$$" "$owner" "$gpu" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    >&"$SPRING_GPU_LEASE_FD"
  return 0
}

spring_release_gpu_lease() {
  if [[ -n "${SPRING_GPU_LEASE_FD:-}" ]]; then
    flock -u "$SPRING_GPU_LEASE_FD" 2>/dev/null || true
    eval "exec ${SPRING_GPU_LEASE_FD}>&-" 2>/dev/null || true
  fi
  SPRING_GPU_LEASE_FD=""
  SPRING_GPU_LEASE_PATH=""
}

# Write a durable, atomic scheduler receipt. ``status`` may be RUNNING,
# COMPLETE, FAILED, or BLOCKED; compose treats only FAILED/BLOCKED as terminal
# failures. Keeping RUNNING in the same receipt path makes a retry supersede
# a stale failure without deleting its directory or racing a watcher.
spring_write_scheduler_receipt() {
  local python_bin="$1" path="$2" receipt_status="$3" stage="$4" exit_code="$5"
  local gpu="${6:-}" log_path="${7:-}" detail="${8:-}" command="${9:-}"
  SPRING_RECEIPT_PATH="$path" \
  SPRING_RECEIPT_STATUS="$receipt_status" \
  SPRING_RECEIPT_STAGE="$stage" \
  SPRING_RECEIPT_EXIT_CODE="$exit_code" \
  SPRING_RECEIPT_GPU="$gpu" \
  SPRING_RECEIPT_LOG="$log_path" \
  SPRING_RECEIPT_DETAIL="$detail" \
  SPRING_RECEIPT_COMMAND="$command" \
  SPRING_RECEIPT_PID="$$" \
  "$python_bin" - <<'PY'
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

path = Path(os.environ["SPRING_RECEIPT_PATH"]).expanduser().resolve()
path.parent.mkdir(parents=True, exist_ok=True)
prior = None
if path.is_file():
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(candidate, dict) and candidate.get("status") in {"FAILED", "BLOCKED"}:
            prior = candidate
    except (OSError, json.JSONDecodeError):
        prior = None
try:
    exit_code = int(os.environ.get("SPRING_RECEIPT_EXIT_CODE", "0"))
except ValueError:
    exit_code = 1
payload = {
    "schema_version": 1,
    "component": "spring-scheduler",
    "status": os.environ.get("SPRING_RECEIPT_STATUS", "FAILED"),
    "stage": os.environ.get("SPRING_RECEIPT_STAGE", "unknown"),
    "exit_code": exit_code,
    "pid": int(os.environ.get("SPRING_RECEIPT_PID", "0")),
    "gpu": os.environ.get("SPRING_RECEIPT_GPU") or None,
    "log": os.environ.get("SPRING_RECEIPT_LOG") or None,
    "detail": os.environ.get("SPRING_RECEIPT_DETAIL") or None,
    "command": os.environ.get("SPRING_RECEIPT_COMMAND") or None,
    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
if prior is not None and os.environ.get("SPRING_RECEIPT_STATUS") not in {"FAILED", "BLOCKED"}:
    payload["previous_receipt"] = prior
fd, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY
}

spring_metrics_ready() {
  local python_bin="$1" path="$2"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(value, dict) or not isinstance(value.get("status"), str):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

spring_receipt_is_terminal_failure() {
  local python_bin="$1" path="$2"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(value, dict) and value.get("status") in {"FAILED", "BLOCKED"} else 1)
PY
}

spring_receipt_is_terminal() {
  local python_bin="$1" path="$2"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(value, dict) and value.get("status") in {"COMPLETE", "FAILED", "BLOCKED"} else 1)
PY
}

spring_receipt_is_running() {
  local python_bin="$1" path="$2"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(value, dict) and value.get("status") == "RUNNING" else 1)
PY
}

spring_checkpoint_step() {
  local python_bin="$1" path="$2"
  "$python_bin" - "$path" <<'PY'
import sys
from pathlib import Path

try:
    import torch
    payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
    step = payload.get("step") if isinstance(payload, dict) else None
    if isinstance(step, int) and step >= 0:
        print(step)
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
}

spring_checkpoint_configured_steps() {
  local python_bin="$1" path="$2"
  "$python_bin" - "$path" <<'PY'
import sys
from pathlib import Path

try:
    import torch
    payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
    config = payload.get("config") if isinstance(payload, dict) else None
    train = config.get("train") if isinstance(config, dict) else None
    steps = train.get("steps_epipolar") if isinstance(train, dict) else None
    if isinstance(steps, int) and steps > 0:
        print(steps)
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
}
