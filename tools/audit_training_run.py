#!/usr/bin/env python3
"""Read-only, machine-readable audit of one FFS-Omega-TSR training run.

``run_summary.json`` is the sole completion receipt.  A directory without that
file is reported as ``IN_PROGRESS`` even if a final checkpoint happens to be
visible during the short interval before the trainer publishes its summary.

The checkpoint loader intentionally uses ``weights_only=False`` because local
training checkpoints contain Python and NumPy RNG state.  This tool is only for
checkpoints produced by this repository; do not point it at untrusted files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


AUDIT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EXPECTED_LOSS_TERMS = (
    "disparity",
    "epipolar",
    "gate_regularizer",
    "gradient",
    "measurement",
    "temporal",
    "total",
    "uncertainty_nll",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HASH_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|unknown)$")


class TrainingAuditError(RuntimeError):
    """Raised when a training artifact violates the audit contract."""


@dataclass(frozen=True)
class AuditExpectations:
    """Optional independently supplied identities for a formal audit."""

    stage: str | None = None
    steps: int | None = None
    git_hash: str | None = None
    config_fingerprint: str | None = None
    checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Validated checkpoint identity without retaining the model payload."""

    path: Path
    sha256: str
    byte_size: int
    step: int
    stage: str
    configured_steps: int
    git_hash: str
    parameter_count: int
    model_state_numel: int
    config_fingerprint: str
    learning_rate: float
    scheduler_last_epoch: int
    checkpoint_interval: int
    gradient_clip: float
    config: Mapping[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": self.step,
            "stage": self.stage,
            "configured_steps": self.configured_steps,
            "git_hash": self.git_hash,
            "parameter_count": self.parameter_count,
            "model_state_numel": self.model_state_numel,
            "config_fingerprint": self.config_fingerprint,
            "optimizer_learning_rate": self.learning_rate,
            "scheduler_last_epoch": self.scheduler_last_epoch,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingAuditError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise TrainingAuditError(f"strict JSON contains non-finite constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingAuditError(f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, name: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except TrainingAuditError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingAuditError(f"cannot parse strict JSON {name}: {exc}") from exc


def _finite_tree(value: Any, name: str) -> None:
    """Reject non-finite numeric leaves in JSON data and checkpoint state."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"{name} contains non-finite tensor values")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            _require(bool(np.isfinite(value).all()), f"{name} contains non-finite ndarray values")
        return
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.number):
            _require(bool(np.isfinite(value)), f"{name} contains a non-finite NumPy scalar")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite float")
        return
    if isinstance(value, int):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
        return
    raise TrainingAuditError(f"{name} contains unsupported value type {type(value).__name__}")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingAuditError(f"checkpoint config is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _configured_stage_and_steps(config: Mapping[str, Any]) -> tuple[str, int]:
    train = config.get("train")
    _require(isinstance(train, Mapping), "checkpoint config.train is missing")
    stage = train.get("stage")
    _require(stage in {"spatial", "temporal"}, "checkpoint config.train.stage is invalid")
    steps_key = "steps_spatial" if stage == "spatial" else "steps"
    configured_steps = train.get(steps_key)
    _require(
        _is_int(configured_steps) and configured_steps > 0,
        f"checkpoint config.train.{steps_key} must be a positive integer",
    )
    return str(stage), int(configured_steps)


def _learning_rate_multiplier(update_index: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and update_index < warmup_steps:
        return float(update_index + 1) / float(warmup_steps)
    decay_updates = total_steps - warmup_steps
    if decay_updates <= 1:
        return 1.0
    progress = min(max(update_index - warmup_steps, 0), decay_updates - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress / (decay_updates - 1)))


def _expected_learning_rate(config: Mapping[str, Any], step: int) -> float:
    train = config["train"]
    stage, total_steps = _configured_stage_and_steps(config)
    del stage
    base = _finite_float(train.get("learning_rate"), "config.train.learning_rate")
    _require(base > 0.0, "config.train.learning_rate must be positive")
    warmup = train.get("warmup_steps")
    _require(_is_int(warmup) and warmup >= 0, "config.train.warmup_steps is invalid")
    return base * _learning_rate_multiplier(step, total_steps, int(warmup))


def _validate_sha256(value: str | None, name: str) -> None:
    if value is not None:
        _require(bool(SHA256_PATTERN.fullmatch(value)), f"{name} must be a lowercase SHA-256")


def _load_checkpoint(path: Path, label: str) -> CheckpointSnapshot:
    _require(path.is_file(), f"{label} checkpoint is missing: {path}")
    payload_bytes = path.read_bytes()
    try:
        payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - convert a corrupt artifact into an audit failure
        raise TrainingAuditError(f"cannot load {label} checkpoint: {exc}") from exc
    _require(isinstance(payload, Mapping), f"{label} checkpoint is not a mapping")
    required = {
        "schema_version",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "config",
        "git_hash",
        "parameter_count",
        "rng_states",
    }
    missing = sorted(required.difference(payload))
    _require(not missing, f"{label} checkpoint fields are missing: {missing}")
    _require(
        payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION,
        f"{label} checkpoint schema mismatch",
    )
    _finite_tree(payload, f"checkpoint.{label}")

    step = payload["step"]
    _require(_is_int(step) and step >= 0, f"{label} checkpoint step is invalid")
    parameter_count = payload["parameter_count"]
    _require(
        _is_int(parameter_count) and parameter_count > 0,
        f"{label} checkpoint parameter_count is invalid",
    )
    git_hash = payload["git_hash"]
    _require(
        isinstance(git_hash, str) and GIT_HASH_PATTERN.fullmatch(git_hash) is not None,
        f"{label} checkpoint git_hash is invalid",
    )
    config = payload["config"]
    _require(isinstance(config, Mapping), f"{label} checkpoint config is malformed")
    stage, configured_steps = _configured_stage_and_steps(config)
    _require(step <= configured_steps, f"{label} checkpoint step exceeds configured steps")
    fingerprint = _config_fingerprint(config)

    model_state = payload["model"]
    _require(isinstance(model_state, Mapping) and model_state, f"{label} model state is malformed")
    _require(
        all(isinstance(value, torch.Tensor) for value in model_state.values()),
        f"{label} model state contains non-tensor values",
    )
    model_numel = sum(int(value.numel()) for value in model_state.values())
    _require(
        model_numel == parameter_count,
        f"{label} model-state numel {model_numel} != parameter_count {parameter_count}",
    )

    optimizer = payload["optimizer"]
    scheduler = payload["scheduler"]
    _require(isinstance(optimizer, Mapping), f"{label} optimizer state is malformed")
    _require(isinstance(scheduler, Mapping), f"{label} scheduler state is malformed")
    parameter_groups = optimizer.get("param_groups")
    _require(isinstance(parameter_groups, list) and parameter_groups, f"{label} optimizer has no parameter groups")
    optimizer_lrs = [
        _finite_float(group.get("lr"), f"{label} optimizer group learning rate")
        for group in parameter_groups
        if isinstance(group, Mapping)
    ]
    _require(len(optimizer_lrs) == len(parameter_groups), f"{label} optimizer group is malformed")
    _require(all(value >= 0.0 for value in optimizer_lrs), f"{label} optimizer learning rate is negative")
    _require(
        max(optimizer_lrs) - min(optimizer_lrs) <= 1e-15,
        f"{label} optimizer groups have inconsistent learning rates",
    )
    base_lrs = scheduler.get("base_lrs")
    last_lrs = scheduler.get("_last_lr")
    last_epoch = scheduler.get("last_epoch")
    _require(
        isinstance(base_lrs, list) and len(base_lrs) == len(parameter_groups),
        f"{label} scheduler base_lrs is malformed",
    )
    _require(
        isinstance(last_lrs, list) and len(last_lrs) == len(parameter_groups),
        f"{label} scheduler _last_lr is malformed",
    )
    _require(_is_int(last_epoch) and last_epoch == step, f"{label} scheduler last_epoch != checkpoint step")
    for index, (optimizer_lr, last_lr) in enumerate(zip(optimizer_lrs, last_lrs, strict=True)):
        last_lr_value = _finite_float(last_lr, f"{label} scheduler _last_lr[{index}]")
        _require(last_lr_value >= 0.0, f"{label} scheduler learning rate is negative")
        _require(
            math.isclose(optimizer_lr, last_lr_value, rel_tol=1e-9, abs_tol=1e-12),
            f"{label} optimizer and scheduler learning rates differ",
        )
    expected_lr = _expected_learning_rate(config, int(step))
    _require(
        math.isclose(optimizer_lrs[0], expected_lr, rel_tol=1e-8, abs_tol=1e-12),
        f"{label} learning rate {optimizer_lrs[0]} != expected schedule {expected_lr}",
    )

    train = config["train"]
    checkpoint_interval = train.get("checkpoint_interval")
    _require(
        _is_int(checkpoint_interval) and checkpoint_interval > 0,
        "config.train.checkpoint_interval must be positive",
    )
    gradient_clip = _finite_float(train.get("gradient_clip"), "config.train.gradient_clip")
    _require(gradient_clip > 0.0, "config.train.gradient_clip must be positive")
    return CheckpointSnapshot(
        path=path.resolve(),
        sha256=_sha256_bytes(payload_bytes),
        byte_size=len(payload_bytes),
        step=int(step),
        stage=stage,
        configured_steps=configured_steps,
        git_hash=str(git_hash),
        parameter_count=int(parameter_count),
        model_state_numel=model_numel,
        config_fingerprint=fingerprint,
        learning_rate=optimizer_lrs[0],
        scheduler_last_epoch=int(last_epoch),
        checkpoint_interval=int(checkpoint_interval),
        gradient_clip=gradient_clip,
        config=config,
    )


def _parse_training_log(path: Path, *, complete: bool) -> tuple[list[dict[str, Any]], bytes, list[str]]:
    _require(path.is_file(), f"training log is missing: {path}")
    payload = path.read_bytes()
    warnings: list[str] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingAuditError(f"training log is not UTF-8: {exc}") from exc
    if text and not text.endswith("\n"):
        if complete:
            raise TrainingAuditError("completed training log has an incomplete final line")
        prefix, separator, suffix = text.rpartition("\n")
        _require(bool(separator), "in-progress training log has no complete JSONL record")
        text = prefix + "\n"
        warnings.append(
            f"ignored one unterminated in-progress JSONL suffix ({len(suffix.encode('utf-8'))} bytes)"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"training log line {line_number} is blank")
        record = _strict_json_loads(line, f"train.jsonl line {line_number}")
        _require(isinstance(record, dict), f"training log line {line_number} is not an object")
        records.append(record)
    _require(bool(records), "training log contains no complete records")
    return records, payload, warnings


def _validate_training_log(
    records: Sequence[Mapping[str, Any]], checkpoint: CheckpointSnapshot
) -> dict[str, Any]:
    required = {"step", "stage", "learning_rate", "gradient_norm", "elapsed_seconds", "loss"}
    previous_step = 0
    previous_elapsed: float | None = None
    resume_boundaries: list[int] = []
    elapsed_values: list[float] = []
    learning_rates: list[float] = []
    gradient_norms: list[float] = []
    losses: dict[str, list[float]] = {name: [] for name in EXPECTED_LOSS_TERMS}

    for line_number, record in enumerate(records, start=1):
        missing = sorted(required.difference(record))
        _require(not missing, f"training log line {line_number} fields are missing: {missing}")
        _finite_tree(record, f"train.jsonl[{line_number}]")
        step = record["step"]
        _require(_is_int(step) and step == previous_step + 1, f"training steps are not continuous at line {line_number}")
        _require(record["stage"] == checkpoint.stage, f"training stage mismatch at line {line_number}")
        learning_rate = _finite_float(record["learning_rate"], f"line {line_number} learning_rate")
        gradient_norm = _finite_float(record["gradient_norm"], f"line {line_number} gradient_norm")
        elapsed = _finite_float(record["elapsed_seconds"], f"line {line_number} elapsed_seconds")
        _require(learning_rate >= 0.0, f"negative learning rate at line {line_number}")
        _require(gradient_norm >= 0.0, f"negative gradient norm at line {line_number}")
        _require(elapsed > 0.0, f"non-positive elapsed_seconds at line {line_number}")
        expected_lr = _expected_learning_rate(checkpoint.config, int(step))
        _require(
            math.isclose(learning_rate, expected_lr, rel_tol=1e-8, abs_tol=1e-12),
            f"learning rate at step {step} is {learning_rate}, expected {expected_lr}",
        )
        loss = record["loss"]
        _require(isinstance(loss, Mapping), f"loss at line {line_number} is not a mapping")
        _require(
            set(loss) == set(EXPECTED_LOSS_TERMS),
            f"loss schema mismatch at line {line_number}: {sorted(loss)}",
        )
        for name in EXPECTED_LOSS_TERMS:
            losses[name].append(_finite_float(loss[name], f"line {line_number} loss.{name}"))
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            resume_boundaries.append(int(step))
        previous_step = int(step)
        previous_elapsed = elapsed
        elapsed_values.append(elapsed)
        learning_rates.append(learning_rate)
        gradient_norms.append(gradient_norm)

    _require(len(records) == previous_step, "training row count does not equal final logged step")
    _require(previous_step <= checkpoint.configured_steps, "training log exceeds configured steps")
    _require(checkpoint.step <= previous_step, "latest checkpoint is ahead of the training log")
    checkpoint_lr = learning_rates[checkpoint.step - 1] if checkpoint.step > 0 else _expected_learning_rate(checkpoint.config, 0)
    _require(
        math.isclose(checkpoint.learning_rate, checkpoint_lr, rel_tol=1e-8, abs_tol=1e-12),
        "latest checkpoint learning rate disagrees with its log record",
    )
    return {
        "last_step": previous_step,
        "elapsed_seconds": elapsed_values,
        "learning_rates": learning_rates,
        "gradient_norms": gradient_norms,
        "losses": losses,
        "resume_boundaries": resume_boundaries,
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _rolling_summary(values: Sequence[float], requested_window: int) -> dict[str, Any]:
    window = min(requested_window, len(values))
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    means = [
        (prefix[index + window] - prefix[index]) / float(window)
        for index in range(len(values) - window + 1)
    ]
    return {
        "window_size": window,
        "window_count": len(means),
        "first_mean": float(means[0]),
        "last_mean": float(means[-1]),
        "minimum_mean": float(min(means)),
        "maximum_mean": float(max(means)),
        "last_minus_first": float(means[-1] - means[0]),
    }


def _series_statistics(values: Sequence[float], rolling_window: int) -> dict[str, Any]:
    _require(bool(values), "cannot summarize an empty numeric series")
    _require(all(math.isfinite(value) for value in values), "statistics input is non-finite")
    ordered = sorted(values)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 3.0 * iqr
    upper_fence = q3 + 3.0 * iqr
    outlier_count = sum(value < lower_fence or value > upper_fence for value in values)
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean": float(mean),
        "population_std": float(statistics.pstdev(values)),
        "minimum": float(ordered[0]),
        "maximum": float(ordered[-1]),
        "quantiles": {
            "p01": _percentile(ordered, 0.01),
            "p05": _percentile(ordered, 0.05),
            "p25": q1,
            "p50": _percentile(ordered, 0.50),
            "p75": q3,
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
        },
        "outlier_policy": "outside_q1_q3_plus_or_minus_3_iqr",
        "outlier_lower_fence": float(lower_fence),
        "outlier_upper_fence": float(upper_fence),
        "outlier_count": int(outlier_count),
        "rolling": _rolling_summary(values, rolling_window),
    }


def _throughput_series(elapsed_values: Sequence[float], resume_boundaries: Sequence[int]) -> list[float]:
    boundaries = set(resume_boundaries)
    result: list[float] = []
    previous_elapsed: float | None = None
    for one_based_step, elapsed in enumerate(elapsed_values, start=1):
        if previous_elapsed is None or one_based_step in boundaries:
            delta = elapsed
        else:
            delta = elapsed - previous_elapsed
        _require(delta > 0.0, f"non-positive elapsed delta at step {one_based_step}")
        result.append(1.0 / delta)
        previous_elapsed = elapsed
    return result


def _load_summary(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingAuditError(f"run summary is not UTF-8: {exc}") from exc
    summary = _strict_json_loads(text, "run_summary.json")
    _require(isinstance(summary, dict), "run summary is not an object")
    _finite_tree(summary, "run_summary")
    return summary, payload


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    final_checkpoint: CheckpointSnapshot,
    last_logged_step: int,
    last_logged_elapsed: float,
) -> None:
    required = {
        "stage",
        "status",
        "steps",
        "run_steps",
        "elapsed_seconds",
        "steps_per_second",
        "device",
        "device_name",
        "torch_version",
        "cuda_version",
        "git_hash",
        "config_fingerprint",
        "final_checkpoint",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    }
    missing = sorted(required.difference(summary))
    _require(not missing, f"run summary fields are missing: {missing}")
    _require(summary["status"] == "TRAINING_COMPLETE", "run summary does not claim TRAINING_COMPLETE")
    _require(summary["stage"] == final_checkpoint.stage, "run summary stage mismatch")
    _require(summary["steps"] == final_checkpoint.step == last_logged_step, "run summary/checkpoint/log step mismatch")
    _require(final_checkpoint.step == final_checkpoint.configured_steps, "completed run did not reach configured steps")
    run_steps = summary["run_steps"]
    _require(_is_int(run_steps) and 0 < run_steps <= final_checkpoint.step, "run summary run_steps is invalid")
    elapsed = _finite_float(summary["elapsed_seconds"], "run summary elapsed_seconds")
    rate = _finite_float(summary["steps_per_second"], "run summary steps_per_second")
    _require(elapsed > 0.0 and rate > 0.0, "run summary elapsed/throughput must be positive")
    _require(elapsed >= last_logged_elapsed, "run summary elapsed time precedes the final log record")
    _require(
        math.isclose(rate, run_steps / elapsed, rel_tol=1e-9, abs_tol=1e-12),
        "run summary steps_per_second is inconsistent",
    )
    _require(summary["git_hash"] == final_checkpoint.git_hash, "run summary git hash mismatch")
    _require(
        summary["config_fingerprint"] == final_checkpoint.config_fingerprint,
        "run summary config fingerprint mismatch",
    )
    final_identity = summary["final_checkpoint"]
    _require(isinstance(final_identity, Mapping), "run summary final_checkpoint is malformed")
    expected_path = Path(str(final_identity.get("path", ""))).expanduser().resolve()
    _require(expected_path == final_checkpoint.path, "run summary final checkpoint path mismatch")
    _require(final_identity.get("sha256") == final_checkpoint.sha256, "run summary final checkpoint SHA mismatch")
    for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        value = summary[name]
        _require(value is None or (_is_int(value) and value >= 0), f"run summary {name} is invalid")


def _validate_expectations(
    expectations: AuditExpectations,
    *,
    checkpoint: CheckpointSnapshot,
    final_checkpoint: CheckpointSnapshot | None,
) -> None:
    if expectations.stage is not None:
        _require(expectations.stage in {"spatial", "temporal"}, "expected stage must be spatial or temporal")
        _require(checkpoint.stage == expectations.stage, "stage differs from the expected stage")
    if expectations.steps is not None:
        _require(_is_int(expectations.steps) and expectations.steps > 0, "expected steps must be positive")
        _require(checkpoint.configured_steps == expectations.steps, "configured steps differ from expected steps")
    if expectations.git_hash is not None:
        _require(GIT_HASH_PATTERN.fullmatch(expectations.git_hash) is not None, "expected git hash is invalid")
        _require(checkpoint.git_hash == expectations.git_hash, "checkpoint git hash differs from expected")
    _validate_sha256(expectations.config_fingerprint, "expected config fingerprint")
    if expectations.config_fingerprint is not None:
        _require(
            checkpoint.config_fingerprint == expectations.config_fingerprint,
            "checkpoint config fingerprint differs from expected",
        )
    _validate_sha256(expectations.checkpoint_sha256, "expected checkpoint SHA")
    if expectations.checkpoint_sha256 is not None:
        _require(final_checkpoint is not None, "expected final checkpoint SHA cannot be checked before completion")
        _require(
            final_checkpoint.sha256 == expectations.checkpoint_sha256,
            "final checkpoint SHA differs from expected",
        )


def _same_checkpoint_identity(left: CheckpointSnapshot, right: CheckpointSnapshot) -> None:
    for name in (
        "step",
        "stage",
        "configured_steps",
        "git_hash",
        "parameter_count",
        "model_state_numel",
        "config_fingerprint",
    ):
        _require(
            getattr(left, name) == getattr(right, name),
            f"latest/final checkpoint {name} mismatch",
        )
    _require(
        math.isclose(left.learning_rate, right.learning_rate, rel_tol=1e-9, abs_tol=1e-12),
        "latest/final checkpoint learning-rate mismatch",
    )


def audit_training_run(
    output_dir: str | Path,
    *,
    expectations: AuditExpectations = AuditExpectations(),
    rolling_window: int = 100,
) -> dict[str, Any]:
    """Audit a training directory without modifying it.

    Args:
        output_dir: Directory containing ``train.jsonl`` and checkpoints.
        expectations: Optional independently known experiment identities.
        rolling_window: Number of optimizer updates per rolling statistic.

    Returns:
        Strict-JSON-compatible audit report.  Its status is ``PASS`` only when
        a valid completion receipt exists, otherwise ``IN_PROGRESS``.
    """

    root = Path(output_dir).expanduser().resolve()
    _require(root.is_dir(), f"training output directory does not exist: {root}")
    _require(_is_int(rolling_window) and rolling_window > 0, "rolling_window must be positive")
    summary_path = root / "run_summary.json"
    final_path = root / "final.pt"
    latest_path = root / "latest.pt"
    log_path = root / "train.jsonl"
    complete = summary_path.is_file()
    if complete:
        _require(final_path.is_file(), "run summary exists but final.pt is missing")

    latest = _load_checkpoint(latest_path, "latest")
    final = _load_checkpoint(final_path, "final") if final_path.is_file() else None
    if final is not None:
        _same_checkpoint_identity(latest, final)
    log_records, log_bytes, warnings = _parse_training_log(log_path, complete=complete)
    log_validation = _validate_training_log(log_records, latest)
    last_step = int(log_validation["last_step"])

    checkpoint_lag = last_step - latest.step
    _require(checkpoint_lag >= 0, "latest checkpoint is ahead of the log")
    if not complete:
        _require(
            checkpoint_lag < latest.checkpoint_interval or latest.step == 0,
            "in-progress latest checkpoint lags by at least one checkpoint interval",
        )
    else:
        _require(final is not None, "completed run has no final checkpoint")
        _require(latest.step == last_step, "completed latest checkpoint does not match final logged step")

    summary: dict[str, Any] | None = None
    summary_bytes: bytes | None = None
    if complete:
        summary, summary_bytes = _load_summary(summary_path)
        assert final is not None
        _validate_summary(
            summary,
            final_checkpoint=final,
            last_logged_step=last_step,
            last_logged_elapsed=float(log_validation["elapsed_seconds"][-1]),
        )
    _validate_expectations(expectations, checkpoint=latest, final_checkpoint=final if complete else None)

    throughput = _throughput_series(
        log_validation["elapsed_seconds"], log_validation["resume_boundaries"]
    )
    loss_statistics = {
        name: _series_statistics(values, rolling_window)
        for name, values in log_validation["losses"].items()
    }
    gradient_statistics = _series_statistics(log_validation["gradient_norms"], rolling_window)
    above_clip = sum(
        value > latest.gradient_clip for value in log_validation["gradient_norms"]
    )
    gradient_statistics["configured_clip"] = latest.gradient_clip
    gradient_statistics["above_configured_clip_count"] = above_clip
    gradient_statistics["interpretation"] = (
        "gradient_norm is measured before clipping; values above the clip are diagnostic, not audit failures"
    )
    throughput_statistics = _series_statistics(throughput, rolling_window)
    total_outliers = (
        sum(statistic["outlier_count"] for statistic in loss_statistics.values())
        + gradient_statistics["outlier_count"]
        + throughput_statistics["outlier_count"]
    )
    files: dict[str, Any] = {
        "train_log": {
            "path": str(log_path),
            "sha256": _sha256_bytes(log_bytes),
            "byte_size": len(log_bytes),
            "records": len(log_records),
        },
        "latest_checkpoint": latest.to_report(),
        "final_checkpoint": None if final is None else final.to_report(),
        "run_summary": None,
    }
    if summary_bytes is not None:
        files["run_summary"] = {
            "path": str(summary_path),
            "sha256": _sha256_bytes(summary_bytes),
            "byte_size": len(summary_bytes),
        }

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": "training-run-audit",
        "status": "PASS" if complete else "IN_PROGRESS",
        "training_status": "TRAINING_COMPLETE" if complete else "IN_PROGRESS",
        "read_only": True,
        "output_dir": str(root),
        "stage": latest.stage,
        "configured_steps": latest.configured_steps,
        "logged_steps": last_step,
        "latest_checkpoint_step": latest.step,
        "checkpoint_lag_steps": checkpoint_lag,
        "git_hash": latest.git_hash,
        "parameter_count": latest.parameter_count,
        "config_fingerprint": latest.config_fingerprint,
        "files": files,
        "validation": {
            "strict_json": True,
            "continuous_steps_from_one": True,
            "all_numeric_values_finite": True,
            "stage_consistent": True,
            "checkpoint_schema_valid": True,
            "checkpoint_state_finite": True,
            "checkpoint_identity_consistent": final is None or complete,
            "learning_rates_nonnegative": True,
            "learning_rate_schedule_exact": True,
            "completion_receipt_valid": complete,
            "resume_boundaries": log_validation["resume_boundaries"],
        },
        "statistics": {
            "rolling_window_requested": rolling_window,
            "loss": loss_statistics,
            "gradient_norm_pre_clip": gradient_statistics,
            "optimizer_steps_per_second": throughput_statistics,
            "total_metric_outlier_count": int(total_outliers),
        },
        "warnings": warnings,
    }
    _finite_tree(report, "audit_report")
    # Prove strict JSON serializability before returning a formal receipt.
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-stage", choices=("spatial", "temporal"))
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-git-hash")
    parser.add_argument("--expected-config-fingerprint")
    parser.add_argument("--expected-checkpoint-sha256", help="expected completed final.pt SHA-256")
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--json-out", type=Path, help="write report outside the audited training directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    json_out = None if args.json_out is None else args.json_out.expanduser().resolve()
    try:
        if json_out is not None:
            _require(
                not _path_is_within(json_out, output_dir),
                "--json-out must be outside --output-dir to preserve read-only auditing",
            )
        report = audit_training_run(
            output_dir,
            expectations=AuditExpectations(
                stage=args.expected_stage,
                steps=args.expected_steps,
                git_hash=args.expected_git_hash,
                config_fingerprint=args.expected_config_fingerprint,
                checkpoint_sha256=args.expected_checkpoint_sha256,
            ),
            rolling_window=args.rolling_window,
        )
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if json_out is None:
            sys.stdout.write(encoded)
        else:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(encoded, encoding="utf-8")
            print(f"audit:training-run: {report['status']}")
            print(f"receipt: {json_out}")
        return 0
    except (OSError, TrainingAuditError) as exc:
        failure = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "component": "training-run-audit",
            "status": "FAIL",
            "output_dir": str(output_dir),
            "error": str(exc),
        }
        sys.stderr.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
