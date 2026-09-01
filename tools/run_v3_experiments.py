#!/usr/bin/env python3
"""Sequential, resumable orchestration for calibration-conditioning v3.

The formal order is fixed per seed::

    A0 -> A1 -> A2 -> A3 -> B0 -> B1

Both Stage-B arms initialize from that seed's selected A3 ``final.pt``.  Every
training arm first attempts micro-batch 4 / accumulation 2 and falls back to
2 / 4 only after a non-zero process exit whose captured log contains a
specific CUDA OOM signature.  Attempts never share output directories.

Seed 42 is always run first.  Seeds 43 and 44 are run only when requested or
when the aggregate seed-42 compute screen says that at least one identifiable
candidate warrants replication.  Aggregate screening is never promoted to a
final GO: the companion decision tool requires all three seeds and paired
per-record bootstrap inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.decide_v3_conditioning import (
        STAGE_A_RECORDS,
        STAGE_B_RECORDS,
        _atomic_json,
        _canonical_sha256,
        _validation_record_sets,
        decide_manifest,
    )
except ModuleNotFoundError:  # Direct ``python tools/run_v3_experiments.py``.
    from decide_v3_conditioning import (  # type: ignore[no-redef]
        STAGE_A_RECORDS,
        STAGE_B_RECORDS,
        _atomic_json,
        _canonical_sha256,
        _validation_record_sets,
        decide_manifest,
    )


SCHEMA_VERSION = 1
COMPONENT = "v3-experiment-orchestrator"
DECISION_INPUT_COMPONENT = "v3-experiment-decision-inputs"
SEED_ORDER = (42, 43, 44)
ARM_ORDER = ("A0", "A1", "A2", "A3", "B0", "B1")
STAGE_B_ARMS = frozenset(("B0", "B1"))
PROFILES = {
    "4x2": {"micro_batch_size": 4, "grad_accumulation": 2},
    "2x4": {"micro_batch_size": 2, "grad_accumulation": 4},
}
ARM_CONFIGS = {
    "A0": "configs/ablations/v3_a0_control.yaml",
    "A1": "configs/ablations/v3_a1_rays.yaml",
    "A2": "configs/ablations/v3_a2_stereo_pose.yaml",
    "A3": "configs/ablations/v3_a3_rays_stereo_pose.yaml",
    "B0": "configs/ablations/v3_b0_temporal_pose_off.yaml",
    "B1": "configs/ablations/v3_b1_temporal_pose_on.yaml",
}
CUDA_OOM_PATTERN = re.compile(
    r"(?:CUDA out of memory|CUDA error:\s*out of memory|"
    r"CUDNN_STATUS_ALLOC_FAILED)",
    re.IGNORECASE,
)


class OrchestrationError(RuntimeError):
    """A fail-closed orchestration or subprocess error."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int


ProcessExecutor = Callable[
    [Sequence[str], Path, Callable[[int], None], Callable[[str], None]],
    ProcessResult,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise OrchestrationError(f"{name} must be a JSON object: {path}")
    return value


def _pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8")
    except OSError:
        # Fail closed on platforms without inspectable procfs.
        return True
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split() if closing >= 0 else []
    return not fields or fields[0] != "Z"


def _live_recorded_child(active_process_path: Path) -> Mapping[str, Any] | None:
    if not active_process_path.is_file():
        return None
    active = _strict_json(active_process_path, "active process receipt")
    if active.get("status") != "RUNNING":
        return None
    child_pid = active.get("child_pid")
    if (
        isinstance(child_pid, bool)
        or not isinstance(child_pid, int)
        or not _pid_is_live(child_pid)
    ):
        return None
    recorded_command = active.get("command")
    if not isinstance(recorded_command, list) or any(
        not isinstance(value, str) for value in recorded_command
    ):
        raise OrchestrationError("live active-process receipt has malformed command")
    cmdline_path = Path(f"/proc/{child_pid}/cmdline")
    try:
        actual_command = [
            value.decode("utf-8", errors="surrogateescape")
            for value in cmdline_path.read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        actual_command = []
    # A live but uninspectable PID is still unsafe. A clearly reused PID with a
    # different command is not the orphaned child recorded by this runner.
    if actual_command and actual_command != recorded_command:
        return None
    return active


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    """Hash executable model/evaluator/config bytes, including dirty files."""

    candidates = [project_root / "train.py", project_root / "eval.py"]
    candidates.extend(sorted((project_root / "src").rglob("*.py")))
    candidates.extend(sorted((project_root / "third_party").rglob("*.py")))
    pending_configs = [project_root / relative for relative in ARM_CONFIGS.values()]
    seen_configs: set[Path] = set()
    while pending_configs:
        config_path = pending_configs.pop().resolve()
        if config_path in seen_configs:
            continue
        seen_configs.add(config_path)
        if not config_path.is_file():
            raise OrchestrationError(f"source/config file is missing: {config_path}")
        for line in config_path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"\s*defaults_from:\s*([^#\s]+)\s*(?:#.*)?", line)
            if match is None:
                continue
            inherited = Path(match.group(1)).expanduser()
            if not inherited.is_absolute():
                project_candidate = project_root / inherited
                inherited = (
                    project_candidate
                    if project_candidate.exists()
                    else config_path.parent / inherited
                )
            pending_configs.append(inherited)
    candidates.extend(seen_configs)
    candidates.extend(
        (
            project_root / "tools" / "run_v3_experiments.py",
            project_root / "tools" / "decide_v3_conditioning.py",
            project_root / "tools" / "audit_v3_temporal_pose.py",
        )
    )
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in sorted(set(candidates)):
        if not path.is_file():
            raise OrchestrationError(f"source/config file is missing: {path}")
        relative = path.relative_to(project_root).as_posix()
        file_hash = _sha256(path)
        files[relative] = file_hash
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\n")
    return {
        "sha256": combined.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _path_identity(path: Path, *, directory: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if directory:
        if not resolved.is_dir():
            raise OrchestrationError(f"required directory is missing: {resolved}")
        receipt = resolved / "run_receipt.json"
        cache_manifest = resolved / "cache_manifest.jsonl"
        return {
            "path": str(resolved),
            "run_receipt_path": str(receipt) if receipt.is_file() else None,
            "run_receipt_sha256": _sha256(receipt) if receipt.is_file() else None,
            "cache_manifest_path": (
                str(cache_manifest) if cache_manifest.is_file() else None
            ),
            "cache_manifest_sha256": (
                _sha256(cache_manifest) if cache_manifest.is_file() else None
            ),
        }
    if not resolved.is_file():
        raise OrchestrationError(f"required file is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _input_identities(args: argparse.Namespace) -> dict[str, Any]:
    calibration_receipt = args.validation_calibration_receipt
    if calibration_receipt is None:
        calibration_receipt = args.validation_calibration_sidecar.with_suffix(
            ".receipt.json"
        )
    identities = {
        "train_manifest": _path_identity(args.train_manifest),
        "train_observation_cache_root": _path_identity(
            args.train_observation_cache_root, directory=True
        ),
        "train_teacher_cache_root": _path_identity(
            args.train_teacher_cache_root, directory=True
        ),
        "train_derived_cache_root": _path_identity(
            args.train_derived_cache_root, directory=True
        ),
        "train_calibration_sidecar": _path_identity(
            args.train_calibration_sidecar
        ),
        "validation_manifest": _path_identity(args.validation_manifest),
        "validation_observation_cache_root": _path_identity(
            args.validation_observation_cache_root, directory=True
        ),
        "validation_teacher_cache_root": _path_identity(
            args.validation_teacher_cache_root, directory=True
        ),
        "validation_derived_cache_root": _path_identity(
            args.validation_derived_cache_root, directory=True
        ),
        "validation_calibration_sidecar": _path_identity(
            args.validation_calibration_sidecar
        ),
        "validation_calibration_receipt": (
            _path_identity(calibration_receipt)
            if calibration_receipt.is_file()
            else {"path": str(calibration_receipt.resolve()), "sha256": None}
        ),
    }
    temporal_pose_audit = getattr(args, "temporal_pose_audit", None)
    if temporal_pose_audit is not None:
        identities["temporal_pose_variation_audit"] = _path_identity(
            temporal_pose_audit
        )
    return identities


def _default_executor(
    command: Sequence[str],
    cwd: Path,
    on_started: Callable[[int], None],
    emit: Callable[[str], None],
) -> ProcessResult:
    watched_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise
    try:
        # A signal arriving between spawn and handler installation remains
        # pending and is delivered only inside this protected block.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        on_started(process.pid)
        assert process.stdout is not None
        for line in process.stdout:
            emit(line)
        return ProcessResult(exit_code=process.wait())
    except BaseException:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _next_attempt_paths(job_directory: Path) -> tuple[Path, Path, Path]:
    job_directory.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in job_directory.glob("process_attempt_*"):
        match = re.match(r"process_attempt_(\d+)(?:\.|$)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    prefix = job_directory / f"process_attempt_{number:03d}"
    return (
        Path(f"{prefix}.log"),
        Path(f"{prefix}.log.partial"),
        job_directory / f"process_attempt_{number:03d}.receipt.json",
    )


def _last_process_receipt(job_directory: Path) -> Mapping[str, Any] | None:
    receipts = sorted(job_directory.glob("process_attempt_*.receipt.json"))
    return _strict_json(receipts[-1], "process receipt") if receipts else None


def _verified_cuda_oom_receipt(receipt: Mapping[str, Any] | None) -> bool:
    if (
        receipt is None
        or receipt.get("status") != "FAILED"
        or not isinstance(receipt.get("exit_code"), int)
        or receipt.get("exit_code") == 0
        or receipt.get("cuda_oom_detected") is not True
    ):
        return False
    log_value = receipt.get("log_path")
    log_sha256 = receipt.get("log_sha256")
    receipt_value = receipt.get("receipt_path")
    command = receipt.get("command")
    if (
        not isinstance(log_value, str)
        or not isinstance(log_sha256, str)
        or not isinstance(receipt_value, str)
        or not isinstance(command, list)
        or "train.micro_batch_size=4" not in command
        or "train.grad_accumulation=2" not in command
    ):
        return False
    log_path = Path(log_value).expanduser().resolve()
    receipt_path = Path(receipt_value).expanduser().resolve()
    if (
        not log_path.is_file()
        or not receipt_path.is_file()
        or _sha256(log_path) != log_sha256
    ):
        return False
    return CUDA_OOM_PATTERN.search(
        log_path.read_text(encoding="utf-8", errors="replace")
    ) is not None


def _invoke(
    command: Sequence[str],
    *,
    cwd: Path,
    job_directory: Path,
    active_process_path: Path,
    executor: ProcessExecutor,
) -> Mapping[str, Any]:
    log_path, partial_log_path, receipt_path = _next_attempt_paths(job_directory)
    started_at = _utc_now()
    started = time.monotonic()
    command_list = [str(value) for value in command]
    child_pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    with partial_log_path.open("w", encoding="utf-8") as handle:

        def on_started(pid: int) -> None:
            nonlocal child_pid
            child_pid = int(pid)
            _atomic_json(
                active_process_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "component": COMPONENT,
                    "status": "RUNNING",
                    "runner_pid": os.getpid(),
                    "child_pid": child_pid,
                    "started_at": started_at,
                    "cwd": str(cwd),
                    "command": command_list,
                    "log_partial": str(partial_log_path),
                },
            )

        def emit(text: str) -> None:
            handle.write(text)
            handle.flush()
            print(text, end="", flush=True)

        try:
            result = executor(command_list, cwd, on_started, emit)
            exit_code = int(result.exit_code)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            emit(error + "\n")
        finally:
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial_log_path, log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    oom = exit_code not in (None, 0) and CUDA_OOM_PATTERN.search(log_text) is not None
    status = "SUCCESS" if exit_code == 0 and error is None else "FAILED"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "component": "v3-orchestrated-process",
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "runner_pid": os.getpid(),
        "child_pid": child_pid,
        "cwd": str(cwd),
        "command": command_list,
        "command_shell": shlex.join(command_list),
        "exit_code": exit_code,
        "exception": error,
        "cuda_oom_detected": oom,
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "receipt_path": str(receipt_path),
    }
    _atomic_json(receipt_path, receipt)
    _atomic_json(
        active_process_path,
        {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": status,
            "runner_pid": os.getpid(),
            "child_pid": child_pid,
            "finished_at": receipt["finished_at"],
            "process_receipt": str(receipt_path),
        },
    )
    if error is not None:
        raise OrchestrationError(error)
    return receipt


def _config_path(args: argparse.Namespace, arm: str) -> Path:
    return (args.project_root / ARM_CONFIGS[arm]).resolve()


def build_train_command(
    args: argparse.Namespace,
    *,
    arm: str,
    seed: int,
    output_directory: Path,
    profile: str,
    initialization_checkpoint: Path | None,
    resume_checkpoint: Path | None,
) -> list[str]:
    if profile not in PROFILES:
        raise OrchestrationError(f"unsupported training profile: {profile}")
    stage_b = arm in STAGE_B_ARMS
    command = [
        str(args.python),
        str(args.project_root / "train.py"),
        "--config",
        str(_config_path(args, arm)),
        "--manifest",
        str(args.train_manifest),
        "--observation-cache-root",
        str(args.train_observation_cache_root),
        "--teacher-cache-root",
        str(args.train_teacher_cache_root),
        "--calibration-sidecar",
        str(args.train_calibration_sidecar),
        "--output-dir",
        str(output_directory),
        "--device",
        str(args.device),
    ]
    if stage_b:
        command.extend(("--derived-cache-root", str(args.train_derived_cache_root)))
        if resume_checkpoint is None:
            if initialization_checkpoint is None:
                raise OrchestrationError(f"{arm} requires the same-seed A3 final.pt")
            command.extend(("--init-from", str(initialization_checkpoint)))
    elif initialization_checkpoint is not None:
        raise OrchestrationError(f"{arm} cannot have a Stage-A initializer")
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    profile_values = PROFILES[profile]
    command.extend(
        (
            f"seed={seed}",
            f"train.micro_batch_size={profile_values['micro_batch_size']}",
            f"train.grad_accumulation={profile_values['grad_accumulation']}",
            "train.effective_batch_size=8",
            *args.train_override,
        )
    )
    return command


def build_eval_command(
    args: argparse.Namespace,
    *,
    arm: str,
    seed: int,
    checkpoint: Path,
    output_directory: Path,
    spatial_checkpoint: Path | None,
) -> list[str]:
    stage_b = arm in STAGE_B_ARMS
    command = [
        str(args.python),
        str(args.project_root / "eval.py"),
        "--config",
        str(_config_path(args, arm)),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(args.validation_manifest),
        "--observation-cache-root",
        str(args.validation_observation_cache_root),
        "--teacher-cache-root",
        str(args.validation_teacher_cache_root),
        "--calibration-sidecar",
        str(args.validation_calibration_sidecar),
        "--output-dir",
        str(output_directory),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.eval_batch_size),
        "--crop-mode",
        "full",
    ]
    if stage_b:
        if spatial_checkpoint is None:
            raise OrchestrationError(f"{arm} evaluation requires same-seed A3")
        command.extend(
            (
                "--derived-cache-root",
                str(args.validation_derived_cache_root),
                "--spatial-checkpoint",
                str(spatial_checkpoint),
            )
        )
    elif spatial_checkpoint is not None:
        raise OrchestrationError(f"{arm} cannot use --spatial-checkpoint")
    command.extend((f"seed={seed}", *args.eval_override))
    return command


def _dry_run_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for seed in SEED_ORDER:
        for arm in ARM_ORDER:
            plan.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "stage": "B" if arm in STAGE_B_ARMS else "A",
                    "depends_on": "same-seed A3 final.pt" if arm in STAGE_B_ARMS else None,
                    "training_profiles": ["4x2", "2x4_on_cuda_oom_only"],
                    "then": "full-resolution evaluation",
                }
            )
    return plan


def _calibration_count(args: argparse.Namespace) -> tuple[int | None, str | None]:
    receipt_path = args.validation_calibration_receipt
    if receipt_path is None:
        receipt_path = args.validation_calibration_sidecar.with_suffix(".receipt.json")
    if not receipt_path.is_file():
        return None, None
    receipt = _strict_json(receipt_path, "validation calibration receipt")
    counts = receipt.get("counts")
    count = counts.get("unique_calibrations") if isinstance(counts, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise OrchestrationError(
            f"validation calibration receipt has invalid unique count: {receipt_path}"
        )
    return count, str(receipt_path.resolve())


def _validated_temporal_pose_audit(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    audit_path = getattr(args, "temporal_pose_audit", None)
    if audit_path is None:
        return None
    audit = _strict_json(audit_path, "temporal-pose variation audit")
    if audit.get("schema_version") != 1 or audit.get("component") != (
        "v3-temporal-pose-variation-audit"
    ):
        raise OrchestrationError("temporal-pose audit schema/component mismatch")
    if audit.get("status") != "PASS" or audit.get("temporal_pose_varies") is not True:
        raise OrchestrationError(
            "temporal-pose audit must explicitly PASS with temporal_pose_varies=true"
        )
    counts = audit.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("formal_temporal_endpoints") != STAGE_B_RECORDS
        or counts.get("formal_windows") != STAGE_B_RECORDS
        or isinstance(counts.get("formal_pose_valid_windows"), bool)
        or not isinstance(counts.get("formal_pose_valid_windows"), int)
        or int(counts["formal_pose_valid_windows"]) < 30
    ):
        raise OrchestrationError(
            "temporal-pose audit lacks exact/adequate formal endpoint coverage"
        )
    ages = audit.get("ages")
    if not isinstance(ages, Mapping) or any(
        not isinstance(ages.get(str(age)), Mapping)
        or ages[str(age)].get("varies") is not True
        for age in (1, 2)
    ):
        raise OrchestrationError("temporal-pose audit must pass both age-1 and age-2")
    inputs = audit.get("inputs")
    if not isinstance(inputs, Mapping):
        raise OrchestrationError("temporal-pose audit input lineage is missing")
    validation_identity = inputs.get("validation_manifest")
    if (
        not isinstance(validation_identity, Mapping)
        or Path(str(validation_identity.get("path"))).expanduser().resolve()
        != args.validation_manifest.resolve()
        or validation_identity.get("sha256") != _sha256(args.validation_manifest)
        or validation_identity.get("records") != STAGE_A_RECORDS
    ):
        raise OrchestrationError("temporal-pose audit validation manifest mismatch")
    derived_root = args.validation_derived_cache_root.resolve()
    if Path(str(inputs.get("derived_root"))).expanduser().resolve() != derived_root:
        raise OrchestrationError("temporal-pose audit derived root mismatch")
    expected_receipt = derived_root / "run_receipt.json"
    expected_manifest = derived_root / "cache_manifest.jsonl"
    if inputs.get("run_receipt_sha256") != _sha256(expected_receipt):
        raise OrchestrationError("temporal-pose audit derived receipt hash mismatch")
    if inputs.get("cache_manifest_sha256") != _sha256(expected_manifest):
        raise OrchestrationError("temporal-pose audit cache-manifest hash mismatch")
    receipt = _strict_json(expected_receipt, "audited derived receipt")
    output = receipt.get("output")
    if not isinstance(output, Mapping) or output.get(
        "cache_manifest_sha256"
    ) != inputs.get("cache_manifest_sha256"):
        raise OrchestrationError("audited derived receipt/manifest binding mismatch")
    _, formal_records = _validation_record_sets(args.validation_manifest)
    expected_ids = [
        identity.record_id
        for identity in sorted(
            formal_records.values(), key=lambda value: value.manifest_index
        )
    ]
    binding = audit.get("formal_endpoint_binding")
    valid_ids = binding.get("pose_valid_record_ids") if isinstance(binding, Mapping) else None
    if (
        not isinstance(binding, Mapping)
        or binding.get("available") is not True
        or binding.get("record_ids") != expected_ids
        or binding.get("record_ids_sha256")
        != _canonical_sha256(expected_ids, "formal endpoint IDs")
        or not isinstance(valid_ids, list)
        or any(not isinstance(value, str) for value in valid_ids)
        or len(set(valid_ids)) != len(valid_ids)
        or not set(valid_ids).issubset(set(expected_ids))
        or len(valid_ids) != int(counts["formal_pose_valid_windows"])
    ):
        raise OrchestrationError("temporal-pose audit formal endpoint binding mismatch")
    return {
        "path": str(audit_path),
        "sha256": _sha256(audit_path),
        "component": audit["component"],
        "status": audit["status"],
        "pose_valid_windows": int(counts["formal_pose_valid_windows"]),
        "derived_root": str(derived_root),
        "run_receipt_sha256": str(inputs["run_receipt_sha256"]),
        "cache_manifest_sha256": str(inputs["cache_manifest_sha256"]),
        "formal_endpoint_ids_sha256": str(binding["record_ids_sha256"]),
    }


def _execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python": str(args.python),
        "host": platform.node(),
        "device": str(args.device),
        "eval_batch_size": args.eval_batch_size,
        "temporal_pose_varies": args.temporal_pose_varies,
        "temporal_pose_identifiability_source": (
            "bound_audit_receipt"
            if getattr(args, "temporal_pose_audit_identity", None) is not None
            else "operator_assertion_or_unavailable"
        ),
        "per_record_filename": args.per_record_filename,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_random_seed": args.bootstrap_random_seed,
        "train_overrides": list(args.train_override),
        "eval_overrides": list(args.eval_override),
        "training_profiles": PROFILES,
    }


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.resume and args.dry_run:
        raise OrchestrationError("--resume and --dry-run are mutually exclusive")
    if not str(args.device).startswith("cuda"):
        raise OrchestrationError("formal 4x2/2x4 orchestration requires a CUDA device")
    if args.eval_batch_size <= 0:
        raise OrchestrationError("--eval-batch-size must be positive")
    if args.bootstrap_replicates < 200:
        raise OrchestrationError("--bootstrap-replicates must be at least 200")
    if args.temporal_pose_varies and getattr(
        args, "temporal_pose_audit_identity", None
    ) is None:
        raise OrchestrationError(
            "bare --temporal-pose-varies is not formal evidence; pass a hash-bound "
            "--temporal-pose-audit"
        )
    per_record = Path(args.per_record_filename)
    if (
        not args.per_record_filename
        or per_record.name != args.per_record_filename
        or per_record.is_absolute()
    ):
        raise OrchestrationError("--per-record-filename must be one plain filename")
    # Formal treatment/metric identity is config-owned. Only operational keys
    # that cannot change model, optimizer, loss, schedule, or metric domains
    # are accepted here.
    allowed_train = {
        "train.num_workers",
        "train.pin_memory",
        "train.persistent_workers",
        "train.log_interval",
        "train.checkpoint_interval",
    }
    allowed_eval = {
        "eval.num_workers",
        "eval.visualization_samples",
        "eval.temporal_flicker_video",
        "eval.temporal_flicker_video_fps",
        "eval.failure_samples_per_criterion",
    }
    for kind, overrides, allowed in (
        ("train", args.train_override, allowed_train),
        ("eval", args.eval_override, allowed_eval),
    ):
        for override in overrides:
            key = override.split("=", 1)[0]
            if "=" not in override or key not in allowed:
                raise OrchestrationError(
                    f"--{kind}-override key {key!r} is not in the benign allowlist"
                )


class Orchestrator:
    def __init__(
        self, args: argparse.Namespace, *, executor: ProcessExecutor = _default_executor
    ) -> None:
        self.args = args
        self.executor = executor
        self.output_root = args.output_root.resolve()
        self.state_path = self.output_root / "orchestration_state.json"
        self.pid_path = self.output_root / "runner_pid.json"
        self.active_process_path = self.output_root / "active_process.json"
        self.decision_inputs_path = self.output_root / "decision_inputs.json"
        self.decision_path = self.output_root / "decision.json"
        self.run_receipt_path = self.output_root / "run_receipt.json"
        self.source_snapshot = _source_snapshot(args.project_root)
        self.unique_calibrations, self.calibration_receipt = _calibration_count(args)
        self.temporal_pose_audit = getattr(
            args, "temporal_pose_audit_identity", None
        )
        self.input_identities = _input_identities(args)
        self.execution_contract = _execution_contract(args)
        self.state: dict[str, Any] = {}
        self.selected_train_dirs: dict[tuple[int, str], Path] = {}
        self.selected_train_hashes: dict[tuple[int, str], str] = {}

    def _publish_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, self.state)

    def _set_job(self, seed: int, arm: str, phase: str, **values: Any) -> None:
        jobs = self.state.setdefault("jobs", {})
        key = f"seed_{seed}.{arm}.{phase}"
        current = jobs.setdefault(key, {})
        current.update(values)
        current["updated_at"] = _utc_now()
        self._publish_state()

    def _assert_source_unchanged(self) -> None:
        current = _source_snapshot(self.args.project_root)
        if current["sha256"] != self.source_snapshot["sha256"]:
            raise OrchestrationError(
                "executable source/config bytes changed during orchestration"
            )
        if _input_identities(self.args) != self.input_identities:
            raise OrchestrationError(
                "manifest, sidecar, or cache-receipt identity changed during orchestration"
            )
        if self.temporal_pose_audit is not None:
            current_audit = _validated_temporal_pose_audit(self.args)
            if current_audit != self.temporal_pose_audit:
                raise OrchestrationError(
                    "temporal-pose audit or its derived/manifest lineage changed"
                )

    def _train_artifact_hashes(self, directory: Path) -> dict[str, str]:
        return {
            "final_pt_sha256": _sha256(directory / "final.pt"),
            "run_summary_sha256": _sha256(directory / "run_summary.json"),
        }

    def _bind_selected_training(
        self, seed: int, arm: str, directory: Path
    ) -> dict[str, str]:
        hashes = self._train_artifact_hashes(directory)
        self.selected_train_dirs[(seed, arm)] = directory
        self.selected_train_hashes[(seed, arm)] = hashes["final_pt_sha256"]
        return hashes

    def _assert_selected_checkpoint_unchanged(
        self, seed: int, arm: str, checkpoint: Path
    ) -> None:
        expected = self.selected_train_hashes.get((seed, arm))
        if expected is None or _sha256(checkpoint) != expected:
            raise OrchestrationError(
                f"selected checkpoint changed after completion: seed {seed} {arm}"
            )

    def _initialize(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            if not self.args.resume:
                raise OrchestrationError(
                    f"orchestration state exists; pass --resume: {self.state_path}"
                )
            live_child = _live_recorded_child(self.active_process_path)
            if live_child is not None:
                raise OrchestrationError(
                    "resume refused because the recorded child process is still live: "
                    f"pid={live_child.get('child_pid')} command="
                    f"{shlex.join(list(live_child.get('command', [])))}"
                )
            previous = _strict_json(self.state_path, "orchestration state")
            if previous.get("component") != COMPONENT:
                raise OrchestrationError("existing orchestration component mismatch")
            previous_snapshot = previous.get("source_snapshot")
            if not isinstance(previous_snapshot, Mapping) or (
                previous_snapshot.get("sha256") != self.source_snapshot["sha256"]
            ):
                raise OrchestrationError("resume source/config snapshot changed")
            if previous.get("input_identities") != self.input_identities:
                raise OrchestrationError("resume input identities changed")
            if previous.get("execution_contract") != self.execution_contract:
                raise OrchestrationError("resume execution contract changed")
            self.state = dict(previous)
            self.state["status"] = "RUNNING"
            self.state["resumed_at"] = _utc_now()
        else:
            non_lock_entries = [
                path for path in self.output_root.iterdir() if path.name != "runner.lock"
            ]
            if self.args.resume:
                raise OrchestrationError(
                    f"--resume requires existing orchestration state: {self.state_path}"
                )
            if non_lock_entries:
                raise OrchestrationError(
                    f"output root is non-empty without state: {self.output_root}"
                )
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "status": "RUNNING",
                "created_at": _utc_now(),
                "runner_pid": os.getpid(),
                "host": platform.node(),
                "project_root": str(self.args.project_root),
                "output_root": str(self.output_root),
                "additional_seeds": self.args.additional_seeds,
                "source_snapshot": self.source_snapshot,
                "input_identities": self.input_identities,
                "execution_contract": self.execution_contract,
                "plan": _dry_run_plan(self.args),
                "jobs": {},
                "completed_seeds": [],
            }
        self.state["runner_pid"] = os.getpid()
        self._publish_state()
        _atomic_json(
            self.pid_path,
            {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "status": "RUNNING",
                "pid": os.getpid(),
                "host": platform.node(),
                "started_at": _utc_now(),
                "state_path": str(self.state_path),
            },
        )

    def _train_arm(self, seed: int, arm: str, a3_final: Path | None) -> Path:
        arm_root = self.output_root / f"seed_{seed}" / arm
        if self.args.resume:
            for profile in ("4x2", "2x4"):
                directory = arm_root / f"train_{profile}"
                if (directory / "final.pt").is_file() and (
                    directory / "run_summary.json"
                ).is_file():
                    hashes = self._train_artifact_hashes(directory)
                    previous_job = self.state.get("jobs", {}).get(
                        f"seed_{seed}.{arm}.train", {}
                    )
                    for name, value in hashes.items():
                        recorded = (
                            previous_job.get(name)
                            if isinstance(previous_job, Mapping)
                            else None
                        )
                        if recorded is not None and recorded != value:
                            raise OrchestrationError(
                                f"resume {seed}/{arm} training artifact hash changed"
                            )
                    self._bind_selected_training(seed, arm, directory)
                    self._set_job(
                        seed,
                        arm,
                        "train",
                        status="RECOVERED_COMPLETE",
                        selected_profile=profile,
                        selected_directory=str(directory),
                        **hashes,
                    )
                    return directory

        high_directory = arm_root / "train_4x2"
        high_receipt = _last_process_receipt(high_directory)
        high_already_oom = bool(
            self.args.resume
            and _verified_cuda_oom_receipt(high_receipt)
        )
        if not high_already_oom:
            resume_checkpoint = (
                high_directory / "latest.pt"
                if self.args.resume and (high_directory / "latest.pt").is_file()
                else None
            )
            command = build_train_command(
                self.args,
                arm=arm,
                seed=seed,
                output_directory=high_directory,
                profile="4x2",
                initialization_checkpoint=a3_final,
                resume_checkpoint=resume_checkpoint,
            )
            self._assert_source_unchanged()
            self._set_job(seed, arm, "train", status="RUNNING_4x2")
            high_receipt = _invoke(
                command,
                cwd=self.args.project_root,
                job_directory=high_directory,
                active_process_path=self.active_process_path,
                executor=self.executor,
            )
            if high_receipt["status"] == "SUCCESS":
                if not (high_directory / "final.pt").is_file() or not (
                    high_directory / "run_summary.json"
                ).is_file():
                    raise OrchestrationError(
                        f"successful {seed}/{arm}/4x2 lacks final completion artifacts"
                    )
                hashes = self._bind_selected_training(seed, arm, high_directory)
                self._set_job(
                    seed,
                    arm,
                    "train",
                    status="COMPLETE",
                    selected_profile="4x2",
                    selected_directory=str(high_directory),
                    **hashes,
                )
                return high_directory
            if not _verified_cuda_oom_receipt(high_receipt):
                raise OrchestrationError(
                    f"{seed}/{arm}/4x2 failed without an auditable CUDA OOM"
                )

        fallback_directory = arm_root / "train_2x4"
        resume_checkpoint = (
            fallback_directory / "latest.pt"
            if self.args.resume and (fallback_directory / "latest.pt").is_file()
            else None
        )
        command = build_train_command(
            self.args,
            arm=arm,
            seed=seed,
            output_directory=fallback_directory,
            profile="2x4",
            initialization_checkpoint=a3_final,
            resume_checkpoint=resume_checkpoint,
        )
        self._assert_source_unchanged()
        self._set_job(
            seed,
            arm,
            "train",
            status="RUNNING_2x4_AFTER_CUDA_OOM",
            high_vram_receipt=(
                None
                if high_receipt is None
                else high_receipt.get("receipt_path")
            ),
            high_vram_receipt_sha256=(
                None
                if high_receipt is None
                else _sha256(Path(str(high_receipt["receipt_path"])))
            ),
        )
        fallback_receipt = _invoke(
            command,
            cwd=self.args.project_root,
            job_directory=fallback_directory,
            active_process_path=self.active_process_path,
            executor=self.executor,
        )
        if fallback_receipt["status"] != "SUCCESS":
            raise OrchestrationError(f"{seed}/{arm}/2x4 fallback failed")
        if not (fallback_directory / "final.pt").is_file() or not (
            fallback_directory / "run_summary.json"
        ).is_file():
            raise OrchestrationError(
                f"successful {seed}/{arm}/2x4 lacks final completion artifacts"
            )
        hashes = self._bind_selected_training(seed, arm, fallback_directory)
        self._set_job(
            seed,
            arm,
            "train",
            status="COMPLETE",
            selected_profile="2x4",
            selected_directory=str(fallback_directory),
            fallback_trigger="verified_cuda_oom",
            **hashes,
        )
        return fallback_directory

    def _evaluate_arm(
        self, seed: int, arm: str, train_directory: Path, a3_final: Path | None
    ) -> Path:
        output_directory = self.output_root / f"seed_{seed}" / arm / "eval"
        metrics_path = output_directory / "metrics.json"
        self._assert_selected_checkpoint_unchanged(
            seed, arm, train_directory / "final.pt"
        )
        if arm in STAGE_B_ARMS:
            assert a3_final is not None
            self._assert_selected_checkpoint_unchanged(seed, "A3", a3_final)
        if self.args.resume and metrics_path.is_file():
            _strict_json(metrics_path, f"seed {seed} {arm} metrics")
            current_metrics_sha = _sha256(metrics_path)
            per_record_path = output_directory / self.args.per_record_filename
            current_per_record_sha = (
                _sha256(per_record_path) if per_record_path.is_file() else None
            )
            previous_job = self.state.get("jobs", {}).get(
                f"seed_{seed}.{arm}.eval", {}
            )
            if isinstance(previous_job, Mapping):
                recorded_metrics = previous_job.get("metrics_sha256")
                if recorded_metrics is not None and recorded_metrics != current_metrics_sha:
                    raise OrchestrationError(
                        f"resume {seed}/{arm} metrics.json hash changed"
                    )
                if "per_record_jsonl_sha256" in previous_job and (
                    previous_job.get("per_record_jsonl_sha256")
                    != current_per_record_sha
                ):
                    raise OrchestrationError(
                        f"resume {seed}/{arm} per-record metrics presence/hash changed"
                    )
            self._set_job(
                seed,
                arm,
                "eval",
                status="RECOVERED_COMPLETE",
                metrics_json=str(metrics_path),
                metrics_sha256=current_metrics_sha,
                per_record_jsonl_sha256=current_per_record_sha,
            )
            return metrics_path
        command = build_eval_command(
            self.args,
            arm=arm,
            seed=seed,
            checkpoint=train_directory / "final.pt",
            output_directory=output_directory,
            spatial_checkpoint=a3_final if arm in STAGE_B_ARMS else None,
        )
        self._assert_source_unchanged()
        self._set_job(seed, arm, "eval", status="RUNNING")
        receipt = _invoke(
            command,
            cwd=self.args.project_root,
            job_directory=output_directory,
            active_process_path=self.active_process_path,
            executor=self.executor,
        )
        if receipt["status"] != "SUCCESS" or not metrics_path.is_file():
            raise OrchestrationError(f"{seed}/{arm} evaluation failed or lacks metrics.json")
        _strict_json(metrics_path, f"seed {seed} {arm} metrics")
        per_record_path = output_directory / self.args.per_record_filename
        self._set_job(
            seed,
            arm,
            "eval",
            status="COMPLETE",
            metrics_json=str(metrics_path),
            metrics_sha256=_sha256(metrics_path),
            per_record_jsonl_sha256=(
                _sha256(per_record_path) if per_record_path.is_file() else None
            ),
            process_receipt=receipt["receipt_path"],
        )
        return metrics_path

    def _run_seed(self, seed: int) -> None:
        a3_final: Path | None = None
        for arm in ARM_ORDER:
            if arm in STAGE_B_ARMS:
                a3_directory = self.selected_train_dirs.get((seed, "A3"))
                if a3_directory is None:
                    raise OrchestrationError(f"seed {seed} Stage-B lacks selected A3")
                a3_final = a3_directory / "final.pt"
                self._assert_selected_checkpoint_unchanged(seed, "A3", a3_final)
            train_directory = self._train_arm(seed, arm, a3_final)
            if arm == "A3":
                a3_final = train_directory / "final.pt"
            self._evaluate_arm(seed, arm, train_directory, a3_final)
        completed = set(int(value) for value in self.state.get("completed_seeds", []))
        completed.add(seed)
        self.state["completed_seeds"] = sorted(completed)
        self._publish_state()

    def _write_decision_inputs(self) -> Path:
        seeds: dict[str, Any] = {}
        for seed in SEED_ORDER:
            arms: dict[str, Any] = {}
            for arm in ARM_ORDER:
                metrics = self.output_root / f"seed_{seed}" / arm / "eval" / "metrics.json"
                if not metrics.is_file():
                    continue
                metrics_sha = _sha256(metrics)
                per_record = metrics.parent / self.args.per_record_filename
                per_record_sha = _sha256(per_record) if per_record.is_file() else None
                job = self.state.get("jobs", {}).get(f"seed_{seed}.{arm}.eval", {})
                if isinstance(job, Mapping):
                    if job.get("metrics_sha256") != metrics_sha or (
                        job.get("per_record_jsonl_sha256") != per_record_sha
                    ):
                        raise OrchestrationError(
                            f"seed {seed} {arm} evaluation artifacts changed before decision"
                        )
                arms[arm] = {
                    "metrics_json": str(metrics),
                    "metrics_sha256": metrics_sha,
                    "per_record_jsonl": str(per_record),
                    "per_record_jsonl_sha256": per_record_sha,
                }
            if arms:
                seeds[str(seed)] = arms
        payload = {
            "schema_version": SCHEMA_VERSION,
            "component": DECISION_INPUT_COMPONENT,
            "expected_counts": {
                "stage_a_records": STAGE_A_RECORDS,
                "stage_b_windows": STAGE_B_RECORDS,
            },
            "identifiability": {
                "unique_static_stereo_calibrations": self.unique_calibrations,
                "validation_calibration_receipt": self.calibration_receipt,
                "temporal_pose_varies": self.args.temporal_pose_varies,
                "temporal_pose_variation_assertion": (
                    "validated and hash-bound audit receipt"
                    if self.temporal_pose_audit is not None
                    else (
                        "operator asserted audited variation"
                        if self.args.temporal_pose_varies
                        else "not asserted; fail closed as NOT_IDENTIFIABLE"
                    )
                ),
                "temporal_pose_variation_audit": self.temporal_pose_audit,
            },
            "validation_manifest": self.input_identities["validation_manifest"],
            "seeds": seeds,
        }
        _atomic_json(self.decision_inputs_path, payload)
        return self.decision_inputs_path

    def _decide(self) -> Mapping[str, Any]:
        self._assert_source_unchanged()
        manifest = self._write_decision_inputs()
        decision = decide_manifest(
            manifest,
            bootstrap_replicates=self.args.bootstrap_replicates,
            bootstrap_random_seed=self.args.bootstrap_random_seed,
        )
        _atomic_json(self.decision_path, decision)
        return decision

    def _finish(self, *, status: str, decision: Mapping[str, Any] | None) -> None:
        self.state["status"] = status
        self.state["finished_at"] = _utc_now()
        if decision is not None:
            self.state["scientific_decision"] = decision.get("decision")
        self._publish_state()
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": status,
            "finished_at": self.state["finished_at"],
            "state_path": str(self.state_path),
            "state_sha256": _sha256(self.state_path),
            "source_snapshot": self.source_snapshot,
            "input_identities": self.input_identities,
            "execution_contract": self.execution_contract,
            "completed_seeds": self.state.get("completed_seeds", []),
            "decision_path": str(self.decision_path) if decision is not None else None,
            "decision_sha256": (
                _sha256(self.decision_path) if decision is not None else None
            ),
            "decision_inputs_path": (
                str(self.decision_inputs_path) if decision is not None else None
            ),
            "decision_inputs_sha256": (
                _sha256(self.decision_inputs_path) if decision is not None else None
            ),
            "decision": decision.get("decision") if decision is not None else None,
            "claim_boundary": (
                "Process completion and scientific GO are separate. Missing paired "
                "per-record metrics or any required final seed fails closed in decision.json."
            ),
        }
        _atomic_json(self.run_receipt_path, receipt)
        _atomic_json(
            self.pid_path,
            {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "status": status,
                "pid": os.getpid(),
                "host": platform.node(),
                "finished_at": self.state["finished_at"],
                "run_receipt": str(self.run_receipt_path),
            },
        )

    def run(self) -> Mapping[str, Any] | None:
        self._initialize()
        if self.args.dry_run:
            self._finish(status="DRY_RUN", decision=None)
            return None
        try:
            self._run_seed(42)
            screening = self._decide()
            continue_additional = bool(
                screening.get("screening", {}).get("continue_additional_seeds")
            )
            if self.args.additional_seeds == "always":
                continue_additional = True
            elif self.args.additional_seeds == "never":
                continue_additional = False
            self.state["seed42_screening"] = screening["screening"]
            self.state["continue_additional_seeds"] = continue_additional
            self._publish_state()
            if continue_additional:
                self._run_seed(43)
                self._run_seed(44)
            decision = self._decide()
            self._finish(status="COMPLETE", decision=decision)
            return decision
        except BaseException as exc:
            self.state["status"] = "FAILED"
            self.state["failure"] = f"{type(exc).__name__}: {exc}"
            self.state["finished_at"] = _utc_now()
            self._publish_state()
            _atomic_json(
                self.pid_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "component": COMPONENT,
                    "status": "FAILED",
                    "pid": os.getpid(),
                    "finished_at": self.state["finished_at"],
                    "failure": self.state["failure"],
                },
            )
            raise


@contextlib.contextmanager
def _exclusive_lock(output_root: Path) -> Any:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "runner.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OrchestrationError(
                f"another v3 orchestrator holds {lock_path}"
            ) from exc
        yield


def run_orchestration(
    args: argparse.Namespace, *, executor: ProcessExecutor = _default_executor
) -> Mapping[str, Any] | None:
    args.project_root = args.project_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for name in (
        "train_manifest",
        "train_observation_cache_root",
        "train_teacher_cache_root",
        "train_derived_cache_root",
        "train_calibration_sidecar",
        "validation_manifest",
        "validation_observation_cache_root",
        "validation_teacher_cache_root",
        "validation_derived_cache_root",
        "validation_calibration_sidecar",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.validation_calibration_receipt is not None:
        args.validation_calibration_receipt = (
            args.validation_calibration_receipt.expanduser().resolve()
        )
    temporal_pose_audit = getattr(args, "temporal_pose_audit", None)
    if temporal_pose_audit is not None:
        args.temporal_pose_audit = temporal_pose_audit.expanduser().resolve()
        args.temporal_pose_audit_identity = _validated_temporal_pose_audit(args)
        args.temporal_pose_varies = True
    else:
        args.temporal_pose_audit_identity = None
    _validate_arguments(args)
    with _exclusive_lock(args.output_root):
        return Orchestrator(args, executor=executor).run()


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-observation-cache-root", type=Path, required=True)
    parser.add_argument("--train-teacher-cache-root", type=Path, required=True)
    parser.add_argument("--train-derived-cache-root", type=Path, required=True)
    parser.add_argument("--train-calibration-sidecar", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-observation-cache-root", type=Path, required=True
    )
    parser.add_argument("--validation-teacher-cache-root", type=Path, required=True)
    parser.add_argument("--validation-derived-cache-root", type=Path, required=True)
    parser.add_argument(
        "--validation-calibration-sidecar", type=Path, required=True
    )
    parser.add_argument(
        "--validation-calibration-receipt",
        type=Path,
        help="audited sidecar receipt; defaults to SIDECAR.with_suffix('.receipt.json')",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--additional-seeds", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument(
        "--temporal-pose-varies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "legacy synthetic-only assertion; formal runs reject it unless "
            "--temporal-pose-audit is also supplied"
        ),
    )
    parser.add_argument(
        "--temporal-pose-audit",
        type=Path,
        help=(
            "preferred formal evidence from audit_v3_temporal_pose.py; binds "
            "the PASS receipt and forces temporal-pose identifiability true"
        ),
    )
    parser.add_argument(
        "--per-record-filename", default="per_record_metrics.jsonl"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-random-seed", type=int, default=20260901)
    parser.add_argument("--train-override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = run_orchestration(args)
    print(
        json.dumps(
            {
                "status": "DRY_RUN" if decision is None else "COMPLETE",
                "output_root": str(args.output_root),
                "decision": None if decision is None else decision["decision"],
            },
            sort_keys=True,
        )
    )
    # Scientific NO-GO is an ordinary completed experiment. Process failures
    # raise and return non-zero; decision.json owns the scientific status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "CUDA_OOM_PATTERN",
    "OrchestrationError",
    "ProcessResult",
    "SEED_ORDER",
    "build_eval_command",
    "build_parser",
    "build_train_command",
    "run_orchestration",
]
