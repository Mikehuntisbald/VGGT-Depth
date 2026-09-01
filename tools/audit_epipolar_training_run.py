#!/usr/bin/env python3
"""Read-only audit for a formal or in-progress Stage-C training directory.

The auditor deliberately accepts only the canonical ``latest.pt``, ``final.pt``,
``train.jsonl`` and ``run_summary.json`` artifacts.  Torch files are read with
``weights_only=True`` and a minimal NumPy allow-list needed by the recorded RNG
state; arbitrary pickle globals are never enabled.

``run_summary.json`` is the completion receipt.  A valid checkpoint at step
5,000 without that atomically-published receipt remains ``IN_PROGRESS``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct as numpy_reconstruct
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometry.epipolar import EPIPOLAR_GEOMETRY_CONTRACT  # noqa: E402
from models.epipolar_refiner import HREpipolarRefiner  # noqa: E402
from tools.audit_d025_evaluation import (  # noqa: E402
    D025EvaluationAuditError,
    audit_d025_evaluation,
)
from train_epipolar import (  # noqa: E402
    validate_stage_c_high_vram_preflight_receipt,
)


AUDIT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
RUN_SUMMARY_SCHEMA_VERSION = 1
STAGE_C_COMPONENT = "ffs-omega-tsr-epipolar-stage-c"
STAGE_C_MODEL_COMPONENT = "hr_epipolar_refiner"
RUN_COMPONENT = "ffs-omega-tsr-epipolar-training-run"
AUDIT_COMPONENT = "ffs-omega-tsr-epipolar-training-audit"
EXPECTED_REFINER_PARAMETERS = 69_905
FORMAL_STAGE_B_STEPS = 15_000
FORMAL_STAGE_C_STEPS = 5_000
STRICT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
STAGE_C_D025_POSITIVITY_PROTOCOL = "d025_stage_c_physical_positivity_v1"
CANONICAL_STAGE_C_ROLE = "CANONICAL_STAGE_C"
CONTROLLED_D025_STAGE_C_ROLE = "CONTROLLED_D025_STAGE_C_ABLATION"
ARCHITECTURE_V2_STAGE_C_ROLE = "ARCHITECTURE_V2_STAGE_C"
PHYSICAL_OUTPUT_V2_PROTOCOL = "explicit_valid_completion_nonnegative_v2"
STAGE_C_PHYSICAL_OUTPUT_V2_PROTOCOL = "base_aware_noop_nonnegative_v2"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_RUNTIME_FIELDS = {
    "device",
    "device_type",
    "device_name",
    "device_capability",
    "torch_version",
    "cuda_version",
    "cuda_available",
    "bf16_supported",
    "autocast_enabled",
    "autocast_dtype",
    "deterministic_algorithms_enabled",
    "deterministic_algorithms_warn_only",
    "cublas_workspace_config",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "strict_determinism_eligible",
    "formal_cuda_bf16_eligible",
}
RUNTIME_ROOT_FILES = (
    "train_epipolar.py",
    "eval_epipolar.py",
    "train.py",
    "eval.py",
    "configs/epipolar_x2.yaml",
    "configs/temporal_x2.yaml",
    "configs/mvp_x2.yaml",
    "pyproject.toml",
)
RUNTIME_GIT_SCOPES = (*RUNTIME_ROOT_FILES, "src")
CONTROLLED_RUNTIME_ADDITIONS = (
    "configs/ablations/d025_positivity_t3.yaml",
    "configs/ablations/d025_stage_c_positivity.yaml",
    "tools/audit_d025_evaluation.py",
)
HIGH_VRAM_RUNTIME_ADDITIONS = (
    "configs/ablations/d025_stage_c_positivity_high_vram.yaml",
)
ARCHITECTURE_V2_RUNTIME_ADDITIONS = (
    "configs/mvp_x2_v2.yaml",
    "configs/temporal_x2_v2.yaml",
    "configs/epipolar_x2_v2.yaml",
)
CONTROLLED_RUNTIME_ROOT_FILES = (
    *RUNTIME_ROOT_FILES[:-1],
    *CONTROLLED_RUNTIME_ADDITIONS,
    RUNTIME_ROOT_FILES[-1],
)
CONTROLLED_RUNTIME_GIT_SCOPES = (*CONTROLLED_RUNTIME_ROOT_FILES, "src")
HIGH_VRAM_RUNTIME_ROOT_FILES = (
    *CONTROLLED_RUNTIME_ROOT_FILES[:-1],
    *HIGH_VRAM_RUNTIME_ADDITIONS,
    CONTROLLED_RUNTIME_ROOT_FILES[-1],
)
HIGH_VRAM_RUNTIME_GIT_SCOPES = (*HIGH_VRAM_RUNTIME_ROOT_FILES, "src")


class EpipolarTrainingAuditError(RuntimeError):
    """Raised when any Stage-C training artifact violates the contract."""


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    path: Path
    sha256: str
    byte_size: int
    step: int
    checkpoint_interval: int
    learning_rate: float
    elapsed_seconds: float
    git_hash: str
    config_sha256: str
    config: Mapping[str, Any]
    payload: Mapping[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "step": self.step,
            "configured_steps": FORMAL_STAGE_C_STEPS,
            "checkpoint_interval": self.checkpoint_interval,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "git_hash": self.git_hash,
            "config_sha256": self.config_sha256,
            "parameter_count": EXPECTED_REFINER_PARAMETERS,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpipolarTrainingAuditError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str) -> int:
    _require(_is_int(value) and int(value) > 0, f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    _require(
        _is_int(value) and int(value) >= 0,
        f"{name} must be a non-negative integer",
    )
    return int(value)


def _finite_float(value: object, name: str) -> float:
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


def _require_sha256(value: object, name: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return str(value)


def _reject_json_constant(value: str) -> None:
    raise EpipolarTrainingAuditError(
        f"strict JSON contains non-finite constant {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, name: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except EpipolarTrainingAuditError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EpipolarTrainingAuditError(
            f"cannot parse strict JSON {name}: {exc}"
        ) from exc


def _finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Tensor):
        if value.is_floating_point() or value.is_complex():
            _require(
                bool(torch.isfinite(value).all().item()),
                f"{name} contains non-finite tensor values",
            )
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            _require(
                bool(np.isfinite(value).all()),
                f"{name} contains non-finite ndarray values",
            )
        return
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.number):
            _require(bool(np.isfinite(value)), f"{name} is a non-finite NumPy scalar")
        return
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
        return
    raise EpipolarTrainingAuditError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _read_regular_local_file(path: Path, label: str) -> bytes:
    """Read a non-symlink regular file without following a last-moment link."""

    _require(path.exists(), f"{label} is missing: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EpipolarTrainingAuditError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size)
            == (after.st_dev, after.st_ino, after.st_size),
            f"{label} changed while it was being read",
        )
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"{label} was read incompletely")
        return payload
    finally:
        os.close(descriptor)


def _safe_torch_load(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    """Safely load tensor/RNG-only local state with no arbitrary pickle globals."""

    payload_bytes = _read_regular_local_file(path, label)
    numpy_uint32_dtype = type(np.dtype(np.uint32))
    safe_globals = [
        numpy_reconstruct,
        np.ndarray,
        np.dtype,
        numpy_uint32_dtype,
    ]
    try:
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(
                io.BytesIO(payload_bytes),
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:  # noqa: BLE001 - normalize corrupt/unsafe artifact errors
        raise EpipolarTrainingAuditError(
            f"cannot safe-load {label}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), f"{label} is not a mapping")
    _finite_tree(payload, label)
    return payload, payload_bytes


def _canonical_config_sha256(config: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EpipolarTrainingAuditError(
            f"checkpoint config is not canonical finite JSON: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_stage_c_positivity_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the isolated Stage-C positivity role without changing baseline defaults."""

    section = config.get("stage_c_positivity_ablation")
    if section is None:
        enabled = False
    else:
        _require(
            isinstance(section, Mapping),
            "checkpoint stage_c_positivity_ablation is not a mapping",
        )
        enabled_value = section.get("enabled")
        _require(
            isinstance(enabled_value, bool),
            "checkpoint stage_c_positivity_ablation.enabled is not boolean",
        )
        enabled = bool(enabled_value)

    base_positivity = config.get("positivity_ablation")
    base_enabled = bool(
        isinstance(base_positivity, Mapping)
        and base_positivity.get("enabled") is True
    )
    _require(
        base_enabled == enabled,
        "Stage-C and frozen-base positivity opt-ins are not enabled together",
    )
    if not enabled:
        return {
            "enabled": False,
            "experiment_role": CANONICAL_STAGE_C_ROLE,
        }

    assert isinstance(section, Mapping)
    _require(
        section.get("protocol_version") == STAGE_C_D025_POSITIVITY_PROTOCOL
        and section.get("requires_passing_d025_base") is True,
        "Stage-C positivity protocol/prerequisite config differs",
    )
    floor = _finite_float(
        section.get("correction_lower_bound_hr_px"),
        "stage_c_positivity_ablation.correction_lower_bound_hr_px",
    )
    _require(
        floor == 0.0,
        "Stage-C positivity lower bound must be exactly zero",
    )
    penalty_weight = _finite_float(
        section.get("pre_lower_bound_negative_penalty_weight"),
        "stage_c_positivity_ablation.pre_lower_bound_negative_penalty_weight",
    )
    _require(
        penalty_weight > 0.0,
        "Stage-C positivity penalty weight must be positive",
    )
    _require(
        "d025_evaluation_metrics_path" not in section,
        "Stage-C positivity config must not trust raw D-025 metrics directly",
    )
    for name in ("d025_training_audit_path", "d025_evaluation_audit_path"):
        value = section.get(name)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"stage_c_positivity_ablation.{name} is not a bound path",
        )
    assert isinstance(base_positivity, Mapping)
    _require(
        base_positivity.get("sanitize_invalid_sources") is True
        and _finite_float(
            base_positivity.get("lower_bound_hr_px"),
            "positivity_ablation.lower_bound_hr_px",
        )
        == 0.0,
        "Stage-C positivity config does not retain the D-025 physical base path",
    )
    protocol = config.get("ablation_protocol")
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("name") == "stage_c_physical_positivity_from_passing_d025"
        and protocol.get("required_base")
        == "full_stage_b_d025_15000_and_holdout_pass"
        and protocol.get("canonical_stage_c_replacement") is False,
        "Stage-C positivity controlled-ablation protocol differs",
    )
    return {
        "enabled": True,
        "experiment_role": CONTROLLED_D025_STAGE_C_ROLE,
        "protocol_version": STAGE_C_D025_POSITIVITY_PROTOCOL,
        "correction_lower_bound_hr_px": floor,
        "pre_lower_bound_negative_penalty_weight": penalty_weight,
    }


def _validate_stage_c_architecture_v2_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the paired physical-output opt-ins as a separate lineage."""

    sections: list[tuple[str, object, str]] = [
        ("physical_output_v2", config.get("physical_output_v2"), PHYSICAL_OUTPUT_V2_PROTOCOL),
        (
            "stage_c_physical_output_v2",
            config.get("stage_c_physical_output_v2"),
            STAGE_C_PHYSICAL_OUTPUT_V2_PROTOCOL,
        ),
    ]
    enabled: list[bool] = []
    for name, section, protocol in sections:
        if section is None:
            enabled.append(False)
            continue
        _require(isinstance(section, Mapping), f"checkpoint {name} is not a mapping")
        enabled_value = section.get("enabled")
        _require(
            isinstance(enabled_value, bool),
            f"checkpoint {name}.enabled is not boolean",
        )
        is_enabled = bool(enabled_value)
        if is_enabled:
            _require(
                section.get("protocol_version") == protocol,
                f"checkpoint {name} protocol differs",
            )
        enabled.append(is_enabled)
    _require(
        enabled[0] == enabled[1],
        "physical_output_v2 and stage_c_physical_output_v2 must be enabled together",
    )
    if not enabled[0]:
        return {
            "enabled": False,
            "experiment_role": CANONICAL_STAGE_C_ROLE,
        }
    return {
        "enabled": True,
        "experiment_role": ARCHITECTURE_V2_STAGE_C_ROLE,
        "physical_output_protocol_version": PHYSICAL_OUTPUT_V2_PROTOCOL,
        "stage_c_protocol_version": STAGE_C_PHYSICAL_OUTPUT_V2_PROTOCOL,
    }


def _stage_c_experiment_role(config: Mapping[str, Any]) -> str:
    positivity = _validate_stage_c_positivity_config(config)
    architecture_v2 = _validate_stage_c_architecture_v2_config(config)
    _require(
        not (positivity["enabled"] and architecture_v2["enabled"]),
        "architecture-v2 and controlled D-025 Stage-C roles are separate",
    )
    if positivity["enabled"]:
        return CONTROLLED_D025_STAGE_C_ROLE
    if architecture_v2["enabled"]:
        return ARCHITECTURE_V2_STAGE_C_ROLE
    return CANONICAL_STAGE_C_ROLE


def _validate_stage_c_high_vram_config(
    config: Mapping[str, Any],
    *,
    controlled_ablation: bool,
) -> dict[str, Any]:
    section = config.get("stage_c_high_vram")
    if section is None:
        return {"enabled": False}
    _require(isinstance(section, Mapping), "stage_c_high_vram is not a mapping")
    enabled = section.get("enabled")
    _require(isinstance(enabled, bool), "stage_c_high_vram.enabled is not boolean")
    if not enabled:
        return {"enabled": False}
    _require(controlled_ablation, "high-VRAM Stage C is not a controlled D-025 arm")
    _require(
        section.get("protocol_version")
        == "d025_stage_c_high_vram_cuda_preflight_v1"
        and section.get("requires_cuda_memory_preflight") is True,
        "Stage-C high-VRAM protocol differs",
    )
    minimum = section.get("minimum_headroom_bytes")
    _require(
        _is_int(minimum) and int(minimum) > 0,
        "Stage-C high-VRAM minimum headroom is malformed",
    )
    receipt_path = section.get("preflight_receipt_path")
    _require(
        isinstance(receipt_path, str) and bool(receipt_path.strip()),
        "Stage-C high-VRAM preflight receipt path is not bound",
    )
    _require(
        section.get("oom_fallback")
        == {
            "micro_batch_size": 2,
            "grad_accumulation": 4,
            "effective_batch_size": 8,
        },
        "Stage-C high-VRAM OOM fallback differs",
    )
    train = config.get("train")
    _require(
        isinstance(train, Mapping)
        and {
            "micro_batch_size": train.get("micro_batch_size"),
            "grad_accumulation": train.get("grad_accumulation"),
            "effective_batch_size": train.get("effective_batch_size"),
        }
        == {
            "micro_batch_size": 4,
            "grad_accumulation": 2,
            "effective_batch_size": 8,
        },
        "Stage-C high-VRAM schedule is not 4x2=8",
    )
    return {
        "enabled": True,
        "protocol_version": "d025_stage_c_high_vram_cuda_preflight_v1",
        "preflight_receipt_path": str(receipt_path),
        "minimum_headroom_bytes": int(minimum),
    }


def _learning_rate_multiplier(
    update_index: int, *, total_steps: int, warmup_steps: int
) -> float:
    if warmup_steps and update_index < warmup_steps:
        return float(update_index + 1) / float(warmup_steps)
    decay_updates = total_steps - warmup_steps
    if decay_updates <= 1:
        return 1.0
    progress = min(max(update_index - warmup_steps, 0), decay_updates - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress / (decay_updates - 1)))


def _validate_config(config: object) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _require(isinstance(config, Mapping), "checkpoint config is not a mapping")
    positivity = _validate_stage_c_positivity_config(config)
    architecture_v2 = _validate_stage_c_architecture_v2_config(config)
    _require(
        not (positivity["enabled"] and architecture_v2["enabled"]),
        "architecture-v2 and controlled D-025 Stage-C roles are separate",
    )
    high_vram = _validate_stage_c_high_vram_config(
        config,
        controlled_ablation=bool(positivity["enabled"]),
    )
    data = config.get("data")
    model = config.get("model")
    train = config.get("train")
    _require(isinstance(data, Mapping), "checkpoint config.data is missing")
    _require(isinstance(model, Mapping), "checkpoint config.model is missing")
    _require(isinstance(train, Mapping), "checkpoint config.train is missing")
    _require(train.get("stage") == "epipolar", "checkpoint is not Stage-C epipolar")
    _require(
        train.get("steps_epipolar") == FORMAL_STAGE_C_STEPS,
        "formal Stage-C config must contain exactly 5000 optimizer steps",
    )
    _require(train.get("precision") == "bf16", "formal Stage-C precision must be bf16")
    _require(
        str(train.get("optimizer", "")).lower() == "adamw",
        "formal Stage-C optimizer must be AdamW",
    )
    _require(
        train.get("compile_model") is False,
        "formal Stage-C baseline must keep torch.compile disabled",
    )
    _require(
        (train.get("micro_batch_size"), train.get("grad_accumulation"))
        in ({(4, 2)} if high_vram["enabled"] else {(2, 4), (1, 8)}),
        "formal Stage-C batch schedule differs from its execution profile",
    )
    _require(
        train.get("effective_batch_size") == 8
        and int(train["micro_batch_size"]) * int(train["grad_accumulation"]) == 8,
        "formal Stage-C effective batch size must be exactly 8",
    )
    _require(
        train.get("log_interval") == 1,
        "formal Stage-C train.jsonl must record every optimizer step",
    )
    checkpoint_interval = _positive_int(
        train.get("checkpoint_interval"), "config.train.checkpoint_interval"
    )
    _require(
        checkpoint_interval == 500,
        "formal Stage-C checkpoint interval must be 500 steps",
    )
    canonical_numeric = {
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "correction_regularizer_weight": 0.01,
    }
    for name, expected in canonical_numeric.items():
        value = _finite_float(train.get(name), f"config.train.{name}")
        _require(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15),
            f"formal Stage-C config.train.{name} differs from {expected}",
        )
    _require(train.get("warmup_steps") == 500, "formal Stage-C warmup must be 500")
    _require(data.get("scale") == 2, "formal Stage-C spatial scale must be x2")
    _require(data.get("sequence_length") == 3, "formal Stage-C must use causal T=3")
    _require(
        data.get("vggt_context_pairs") == 5,
        "formal Stage-C must retain five causal VGGT pairs",
    )
    _require(
        data.get("hr_crop") == [384, 768] and data.get("crop_mode") == "random",
        "formal Stage-C must train on random 384x768 HR crops",
    )
    _require(model.get("epipolar_refinement") is True, "epipolar refiner is disabled")
    _require(model.get("use_history") is True, "formal Stage-C history is disabled")
    _require(model.get("use_vggt_pose") is True, "formal Stage-C VGGT pose is disabled")
    _require(
        model.get("epipolar_vertical_geometry")
        == EPIPOLAR_GEOMETRY_CONTRACT["version"],
        "Stage-C vertical geometry contract differs",
    )
    architecture = {
        "feature_channels": model.get("epipolar_feature_channels"),
        "correlation_groups": model.get("epipolar_correlation_groups"),
        "candidate_offsets_hr_px": model.get("epipolar_offsets_hr_px"),
        "correction_limit_hr_px": model.get("epipolar_correction_limit_hr_px"),
        "confidence_temperature": model.get("epipolar_confidence_temperature"),
        "head_channels": model.get("epipolar_head_channels"),
    }
    _require(
        architecture
        == {
            "feature_channels": 32,
            "correlation_groups": 8,
            "candidate_offsets_hr_px": [-2, -1, 0, 1, 2],
            "correction_limit_hr_px": 2.0,
            "confidence_temperature": 1.0,
            "head_channels": 48,
        },
        "formal Stage-C refiner architecture/search config differs",
    )
    _finite_tree(config, "checkpoint.config")
    return config, {
        "checkpoint_interval": checkpoint_interval,
        "base_learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]),
        "warmup_steps": int(train["warmup_steps"]),
        "grad_accumulation": int(train["grad_accumulation"]),
        "stage_c_positivity_ablation": positivity,
        "stage_c_architecture_v2": architecture_v2,
        "experiment_role": _stage_c_experiment_role(config),
        "stage_c_high_vram": high_vram,
    }


def _build_refiner(config: Mapping[str, Any]) -> HREpipolarRefiner:
    model = config["model"]
    refiner = HREpipolarRefiner(
        feature_channels=int(model["epipolar_feature_channels"]),
        correlation_groups=int(model["epipolar_correlation_groups"]),
        candidate_offsets_hr_px=tuple(
            float(value) for value in model["epipolar_offsets_hr_px"]
        ),
        correction_limit_hr_px=float(model["epipolar_correction_limit_hr_px"]),
        confidence_temperature=float(model["epipolar_confidence_temperature"]),
        head_channels=int(model["epipolar_head_channels"]),
    )
    _require(
        refiner.trainable_parameter_count == EXPECTED_REFINER_PARAMETERS,
        "reconstructed refiner does not contain 69,905 trainable parameters",
    )
    return refiner


def _validate_optimizer_scheduler_rng(
    payload: Mapping[str, Any],
    *,
    refiner: HREpipolarRefiner,
    step: int,
    config_values: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    scaler = payload.get("scaler")
    rng = payload.get("rng_states")
    _require(isinstance(optimizer, Mapping), "Stage-C optimizer state is malformed")
    _require(isinstance(scheduler, Mapping), "Stage-C scheduler state is malformed")
    _require(scaler == {}, "Stage-C native BF16 scaler must be empty")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    _require(isinstance(state, Mapping), "Stage-C AdamW state is malformed")
    _require(
        isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], Mapping),
        "Stage-C AdamW must contain exactly one parameter group",
    )
    group = groups[0]
    parameter_ids = group.get("params")
    parameters = list(refiner.parameters())
    _require(
        isinstance(parameter_ids, list)
        and len(parameter_ids) == len(parameters)
        and len(set(parameter_ids)) == len(parameters),
        "Stage-C AdamW does not cover every refiner parameter tensor",
    )
    _require(
        set(state) == set(parameter_ids),
        "Stage-C AdamW state and parameter group differ",
    )
    for parameter_id, parameter in zip(parameter_ids, parameters, strict=True):
        item = state[parameter_id]
        _require(isinstance(item, Mapping), "Stage-C AdamW parameter state is malformed")
        optimizer_step = item.get("step")
        if isinstance(optimizer_step, Tensor):
            _require(optimizer_step.numel() == 1, "Stage-C AdamW step is not scalar")
            optimizer_step = float(optimizer_step.item())
        _require(
            isinstance(optimizer_step, (int, float))
            and not isinstance(optimizer_step, bool)
            and math.isfinite(float(optimizer_step))
            and float(optimizer_step) == float(step),
            "Stage-C AdamW progress differs from checkpoint step",
        )
        for name in ("exp_avg", "exp_avg_sq"):
            value = item.get(name)
            _require(
                isinstance(value, Tensor)
                and value.shape == parameter.shape
                and value.dtype == parameter.dtype
                and bool(torch.isfinite(value).all().item()),
                f"Stage-C AdamW {name} is missing, incompatible, or non-finite",
            )
    expected_lr = float(config_values["base_learning_rate"]) * (
        _learning_rate_multiplier(
            step,
            total_steps=FORMAL_STAGE_C_STEPS,
            warmup_steps=int(config_values["warmup_steps"]),
        )
    )
    numeric_group = {
        "lr": expected_lr,
        "initial_lr": float(config_values["base_learning_rate"]),
        "weight_decay": float(config_values["weight_decay"]),
        "eps": 1e-8,
    }
    for name, expected in numeric_group.items():
        actual = _finite_float(group.get(name), f"optimizer.param_groups[0].{name}")
        _require(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
            f"Stage-C AdamW {name} differs from config/schedule",
        )
    _require(group.get("betas") == (0.9, 0.999), "Stage-C AdamW betas differ")
    _require(
        group.get("amsgrad") is False and group.get("maximize") is False,
        "Stage-C AdamW mode differs",
    )
    _require(
        scheduler.get("last_epoch") == step
        and scheduler.get("_step_count") == step + 1,
        "Stage-C scheduler progress differs from checkpoint step",
    )
    _require(
        scheduler.get("base_lrs") == [float(config_values["base_learning_rate"])],
        "Stage-C scheduler base LR differs",
    )
    last_lr = scheduler.get("_last_lr")
    _require(
        isinstance(last_lr, list)
        and len(last_lr) == 1
        and math.isclose(
            _finite_float(last_lr[0], "scheduler._last_lr[0]"),
            expected_lr,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "Stage-C scheduler LR differs from exact warmup/cosine schedule",
    )
    _require(
        isinstance(rng, Mapping)
        and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "Stage-C RNG state schema is malformed",
    )
    python_rng = rng["python"]
    numpy_rng = rng["numpy"]
    cpu_rng = rng["torch_cpu"]
    cuda_rng = rng["torch_cuda"]
    _require(
        isinstance(python_rng, tuple)
        and len(python_rng) == 3
        and isinstance(python_rng[0], int)
        and isinstance(python_rng[1], tuple),
        "Stage-C Python RNG state is malformed",
    )
    _require(
        isinstance(numpy_rng, tuple)
        and len(numpy_rng) == 5
        and isinstance(numpy_rng[0], str)
        and isinstance(numpy_rng[1], np.ndarray)
        and numpy_rng[1].dtype == np.uint32
        and numpy_rng[1].ndim == 1
        and isinstance(numpy_rng[2], int),
        "Stage-C NumPy RNG state is malformed",
    )
    _require(
        isinstance(cpu_rng, Tensor)
        and cpu_rng.dtype == torch.uint8
        and cpu_rng.ndim == 1
        and cpu_rng.numel() > 0,
        "Stage-C CPU Torch RNG state is malformed",
    )
    _require(
        isinstance(cuda_rng, list)
        and len(cuda_rng) > 0
        and all(
            isinstance(value, Tensor)
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.numel() > 0
            for value in cuda_rng
        ),
        "formal Stage-C CUDA RNG state is missing or malformed",
    )
    return {
        "optimizer": "AdamW",
        "optimizer_parameter_tensors": len(parameters),
        "optimizer_steps_consistent": True,
        "scheduler_progress_consistent": True,
        "scheduler_learning_rate_consistent": True,
        "expected_learning_rate": expected_lr,
        "rng_schema_consistent": True,
    }


def _validate_data_cursor(
    value: object,
    *,
    step: int,
    grad_accumulation: int,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Stage-C data cursor is missing")
    batches_per_epoch = _positive_int(
        value.get("batches_per_epoch"), "data_cursor.batches_per_epoch"
    )
    completed_micro_steps = step * grad_accumulation
    epoch, offset = divmod(completed_micro_steps, batches_per_epoch)
    expected = {
        "completed_micro_steps": completed_micro_steps,
        "batches_per_epoch": batches_per_epoch,
        "epoch": epoch,
        "batch_offset_in_epoch": offset,
        "grad_accumulation": grad_accumulation,
        "drop_last": True,
    }
    _require(dict(value) == expected, "Stage-C data cursor is inconsistent with step")
    return expected


def _validate_training_runtime(value: object) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == EXPECTED_RUNTIME_FIELDS,
        "Stage-C training_runtime fields are missing or malformed",
    )
    booleans = (
        "cuda_available",
        "bf16_supported",
        "autocast_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "strict_determinism_eligible",
        "formal_cuda_bf16_eligible",
    )
    _require(
        all(type(value.get(name)) is bool for name in booleans),
        "Stage-C training_runtime boolean fields are malformed",
    )
    capability = value.get("device_capability")
    device_name = value.get("device_name")
    cuda_version = value.get("cuda_version")
    torch_version = value.get("torch_version")
    device_value = value.get("device")
    try:
        parsed_device = torch.device(device_value) if isinstance(device_value, str) else None
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EpipolarTrainingAuditError(
            "formal Stage-C training device receipt is malformed"
        ) from exc
    _require(
        value.get("device_type") == "cuda"
        and isinstance(device_value, str)
        and parsed_device is not None
        and parsed_device.type == "cuda",
        "formal Stage-C was not produced on CUDA",
    )
    _require(
        isinstance(device_name, str) and "5090" in device_name.lower(),
        "formal Stage-C training device is not an RTX 5090",
    )
    _require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(_is_int(item) and item >= 0 for item in capability)
        and tuple(capability) >= (12, 0),
        "formal Stage-C training capability is not Blackwell-class",
    )
    _require(
        isinstance(torch_version, str) and bool(torch_version),
        "Stage-C Torch version receipt is malformed",
    )
    _require(
        isinstance(cuda_version, str) and bool(cuda_version),
        "Stage-C CUDA version receipt is malformed",
    )
    try:
        cuda_parts = tuple(int(part) for part in cuda_version.split(".")[:2])
    except ValueError as exc:
        raise EpipolarTrainingAuditError(
            "Stage-C CUDA version receipt is malformed"
        ) from exc
    _require(cuda_parts >= (12, 8), "formal Stage-C requires CUDA 12.8+")
    deterministic = bool(
        value["deterministic_algorithms_enabled"]
        and not value["deterministic_algorithms_warn_only"]
        and value.get("cublas_workspace_config") == STRICT_CUBLAS_WORKSPACE_CONFIG
        and value["cudnn_deterministic"]
        and not value["cudnn_benchmark"]
    )
    native_bf16 = bool(
        value["cuda_available"]
        and value["bf16_supported"]
        and value["autocast_enabled"]
        and value.get("autocast_dtype") == "torch.bfloat16"
    )
    _require(deterministic, "Stage-C strict deterministic runtime receipt is false")
    _require(native_bf16, "Stage-C native CUDA BF16 runtime receipt is false")
    _require(
        value["strict_determinism_eligible"] == deterministic
        and value["formal_cuda_bf16_eligible"] is True,
        "Stage-C runtime eligibility flags are inconsistent",
    )
    return {
        "device": device_value,
        "device_name": device_name,
        "device_capability": list(capability),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "native_cuda_bf16": True,
        "strict_determinism": True,
        "formal_cuda_bf16_eligible": True,
    }


def _validate_base_checkpoint(
    value: object, base_completion: object
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "sha256", "step"},
        "Stage-C frozen-base reference must contain path/sha256/step",
    )
    _require(value.get("step") == FORMAL_STAGE_B_STEPS, "Stage-C base is not step 15000")
    expected_sha = _require_sha256(value.get("sha256"), "base checkpoint SHA-256")
    path_value = value.get("path")
    _require(isinstance(path_value, str) and bool(path_value), "base checkpoint path is malformed")
    base_path = Path(path_value).expanduser().resolve()
    base_payload, base_bytes = _safe_torch_load(base_path, "frozen Stage-B checkpoint")
    _require(_sha256_bytes(base_bytes) == expected_sha, "frozen Stage-B SHA-256 mismatch")
    _require(
        base_payload.get("step") == FORMAL_STAGE_B_STEPS,
        "frozen Stage-B payload is not step 15000",
    )
    base_config = base_payload.get("config")
    base_train = base_config.get("train") if isinstance(base_config, Mapping) else None
    _require(
        isinstance(base_train, Mapping)
        and base_train.get("stage") == "temporal"
        and base_train.get("steps") == FORMAL_STAGE_B_STEPS
        and base_train.get("steps_temporal") == FORMAL_STAGE_B_STEPS,
        "frozen Stage-B checkpoint is not a completed canonical temporal run",
    )
    _require(isinstance(base_completion, Mapping), "Stage-C base completion receipt is missing")
    expected_completion = {
        "actual_step": FORMAL_STAGE_B_STEPS,
        "configured_steps": FORMAL_STAGE_B_STEPS,
        "declared_temporal_steps": FORMAL_STAGE_B_STEPS,
        "required_steps": FORMAL_STAGE_B_STEPS,
        "canonical_required_steps": FORMAL_STAGE_B_STEPS,
        "complete": True,
        "required_for_this_run": True,
    }
    _require(
        dict(base_completion) == expected_completion,
        "Stage-C base completion receipt is not canonical",
    )
    return {
        "path": str(base_path),
        "sha256": expected_sha,
        "step": FORMAL_STAGE_B_STEPS,
        "completion": expected_completion,
    }


def _validate_rectification_audit(
    recorded: object,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(recorded, Mapping), "Stage-C rectification audit is missing")
    required = {
        "path",
        "sha256",
        "schema_version",
        "component",
        "status",
        "contract_version",
        "manifest_sha256",
        "algorithm",
        "thresholds",
        "counts",
        "pixel_evidence",
        "metadata_vs_pixels",
        "sample_identity_sha256",
    }
    _require(set(recorded) == required, "Stage-C rectification audit fields differ")
    _require(
        recorded.get("schema_version") == 1
        and recorded.get("component") == "pixel-level-epipolar-rectification-audit"
        and recorded.get("status") == "PASS"
        and recorded.get("contract_version") == EPIPOLAR_GEOMETRY_CONTRACT["version"],
        "Stage-C rectification audit contract/status differs",
    )
    path_value = recorded.get("path")
    _require(
        isinstance(path_value, str) and bool(path_value),
        "rectification audit path is malformed",
    )
    receipt_path = Path(path_value).expanduser().resolve()
    receipt_bytes = _read_regular_local_file(receipt_path, "rectification audit receipt")
    recorded_sha = _require_sha256(recorded.get("sha256"), "rectification audit SHA-256")
    _require(_sha256_bytes(receipt_bytes) == recorded_sha, "rectification audit SHA-256 mismatch")
    try:
        receipt_text = receipt_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpipolarTrainingAuditError("rectification audit is not UTF-8") from exc
    receipt = _strict_json_loads(receipt_text, "rectification audit")
    _require(isinstance(receipt, Mapping), "rectification audit receipt is not an object")
    _finite_tree(receipt, "rectification_audit_receipt")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("component") == recorded["component"]
        and receipt.get("status") == "PASS"
        and receipt.get("published_contract") == recorded["contract_version"],
        "current rectification audit receipt differs from checkpoint",
    )
    checks = receipt.get("threshold_checks")
    _require(
        isinstance(checks, list)
        and bool(checks)
        and all(isinstance(check, Mapping) and check.get("passed") is True for check in checks),
        "rectification audit threshold checks did not all pass",
    )
    manifests = receipt.get("manifests")
    recorded_manifests = recorded.get("manifest_sha256")
    _require(
        isinstance(manifests, Mapping)
        and isinstance(manifests.get("train"), Mapping)
        and isinstance(manifests.get("validation"), Mapping)
        and manifests.get("train_validation_sequence_disjoint") is True
        and recorded_manifests
        == {
            "train": manifests["train"].get("sha256"),
            "validation": manifests["validation"].get("sha256"),
        },
        "rectification audit manifest/isolation binding differs",
    )
    for name in ("train", "validation"):
        _require_sha256(recorded_manifests[name], f"rectification {name} manifest SHA")
    global_result = receipt.get("global")
    counts = global_result.get("counts") if isinstance(global_result, Mapping) else None
    dy = global_result.get("dy_right_minus_left_px") if isinstance(global_result, Mapping) else None
    absolute = dy.get("absolute") if isinstance(dy, Mapping) else None
    signed = dy.get("signed") if isinstance(dy, Mapping) else None
    _require(
        isinstance(counts, Mapping)
        and isinstance(absolute, Mapping)
        and isinstance(signed, Mapping),
        "rectification audit aggregate pixel evidence is malformed",
    )
    expected_counts = {
        "sampled_frames": global_result.get("sampled_frames"),
        "covered_frames": global_result.get("covered_frames"),
        "ratio_matches": counts.get("ratio_matches"),
        "ransac_inliers": counts.get("ransac_inliers"),
    }
    expected_pixels = {
        "coverage_fraction": global_result.get("coverage_fraction"),
        "median_right_y_minus_left_y_px": signed.get("p50"),
        "p95_abs_right_y_minus_left_y_px": absolute.get("p95"),
    }
    _require(recorded.get("counts") == expected_counts, "rectification audit counts differ")
    _require(
        recorded.get("pixel_evidence") == expected_pixels,
        "rectification audit pixel evidence differs",
    )
    sampling = receipt.get("sampling")
    _require(
        isinstance(sampling, Mapping)
        and recorded.get("sample_identity_sha256")
        == sampling.get("sample_identity_sha256"),
        "rectification audit sample identity differs",
    )
    _require_sha256(recorded.get("sample_identity_sha256"), "rectification sample identity")
    data = config.get("data")
    _require(isinstance(data, Mapping), "Stage-C data config is missing")
    configured_path = data.get("epipolar_rectification_audit_path")
    _require(
        isinstance(configured_path, str)
        and Path(configured_path).expanduser().resolve() == receipt_path
        and data.get("epipolar_rectification_audit") == recorded,
        "Stage-C config is not bound to the exact rectification audit",
    )
    return {
        "path": str(receipt_path),
        "sha256": recorded_sha,
        "status": "PASS",
        "contract_version": recorded["contract_version"],
        "manifest_sha256": dict(recorded_manifests),
        "counts": dict(expected_counts),
        "pixel_evidence": dict(expected_pixels),
    }


def _git_tree_contains(git_hash: str, relative: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{git_hash}:{relative}"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def _git_tree_declares_runtime_additions(git_hash: str) -> tuple[bool, ...]:
    shown = subprocess.run(
        ["git", "show", f"{git_hash}:train_epipolar.py"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(
        shown.returncode == 0,
        "Stage-C checkpoint Git tree lacks train_epipolar.py",
    )
    return tuple(
        relative.encode("utf-8") in shown.stdout
        for relative in CONTROLLED_RUNTIME_ADDITIONS
    )


def _git_tree_runtime_python_paths(git_hash: str) -> tuple[str, ...]:
    """Enumerate runtime Python files from ``git_hash``, never the worktree.

    A checkpoint binds a committed source tree.  Using ``SRC_ROOT.rglob`` here
    would silently compare an old checkpoint against files that happened to be
    added to the auditor's current checkout after that checkpoint was written.
    ``git ls-tree`` makes both the membership and canonical ordering properties
    of the checkpoint commit itself.
    """

    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", git_hash, "--", "src"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _require(
        listed.returncode == 0,
        "cannot enumerate runtime sources from the Stage-C checkpoint Git tree",
    )
    paths = tuple(
        line
        for line in listed.stdout.splitlines()
        if line.startswith("src/") and line.endswith(".py")
    )
    _require(bool(paths), "Stage-C checkpoint Git tree contains no src Python files")
    _require(
        paths == tuple(sorted(paths)) and len(paths) == len(set(paths)),
        "Stage-C checkpoint Git tree returned non-canonical src paths",
    )
    return paths


def _git_tree_declares_additions(
    git_hash: str,
    additions: Sequence[str],
) -> tuple[bool, ...]:
    shown = subprocess.run(
        ["git", "show", f"{git_hash}:train_epipolar.py"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(
        shown.returncode == 0,
        "Stage-C checkpoint Git tree lacks train_epipolar.py",
    )
    return tuple(relative.encode("utf-8") in shown.stdout for relative in additions)


def _runtime_source_contract(
    git_hash: str,
    *,
    controlled_ablation: bool,
    high_vram: bool = False,
    physical_v2: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    _require(
        not high_vram or controlled_ablation,
        "high-VRAM runtime requires controlled D-025 Stage C",
    )
    _require(
        not physical_v2 or not controlled_ablation,
        "architecture-v2 and controlled D-025 runtime roles are separate",
    )
    if physical_v2:
        additions_declared = _git_tree_declares_additions(
            git_hash,
            ARCHITECTURE_V2_RUNTIME_ADDITIONS,
        )
        _require(
            all(additions_declared) or not any(additions_declared),
            "Stage-C producer declares only part of the architecture-v2 runtime bundle",
        )
        _require(
            all(additions_declared),
            "architecture-v2 Stage-C producer does not declare its runtime additions",
        )
        _require(
            all(
                _git_tree_contains(git_hash, relative)
                for relative in ARCHITECTURE_V2_RUNTIME_ADDITIONS
            ),
            "Stage-C architecture-v2 runtime file is absent from the checkpoint Git tree",
        )
        root_files = (
            *RUNTIME_ROOT_FILES[:-1],
            *ARCHITECTURE_V2_RUNTIME_ADDITIONS,
            RUNTIME_ROOT_FILES[-1],
        )
        scopes = (*root_files, "src")
        return (
            root_files,
            scopes,
            len(root_files) + len(_git_tree_runtime_python_paths(git_hash)),
        )
    if not controlled_ablation:
        # The canonical/default producer deliberately ignores opt-in files even
        # when they coexist in the same Git tree.  Its src membership, however,
        # is always the membership of the checkpoint commit itself.
        return (
            RUNTIME_ROOT_FILES,
            RUNTIME_GIT_SCOPES,
            len(RUNTIME_ROOT_FILES) + len(_git_tree_runtime_python_paths(git_hash)),
        )
    additions_declared = _git_tree_declares_runtime_additions(git_hash)
    _require(
        all(additions_declared) or not any(additions_declared),
        "Stage-C producer declares only part of the controlled runtime bundle",
    )
    _require(
        all(additions_declared),
        "controlled Stage-C producer does not declare its runtime additions",
    )
    _require(
        all(
            _git_tree_contains(git_hash, relative)
            for relative in CONTROLLED_RUNTIME_ADDITIONS
        ),
        "Stage-C controlled runtime file is absent from the checkpoint Git tree",
    )
    if high_vram:
        _require(
            all(
                _git_tree_contains(git_hash, relative)
                for relative in HIGH_VRAM_RUNTIME_ADDITIONS
            ),
            "Stage-C high-VRAM runtime file is absent from the checkpoint Git tree",
        )
        return (
            HIGH_VRAM_RUNTIME_ROOT_FILES,
            HIGH_VRAM_RUNTIME_GIT_SCOPES,
            len(HIGH_VRAM_RUNTIME_ROOT_FILES)
            + len(_git_tree_runtime_python_paths(git_hash)),
        )
    return (
        CONTROLLED_RUNTIME_ROOT_FILES,
        CONTROLLED_RUNTIME_GIT_SCOPES,
        len(CONTROLLED_RUNTIME_ROOT_FILES)
        + len(_git_tree_runtime_python_paths(git_hash)),
    )


def _expected_runtime_paths(
    *,
    git_hash: str | None = None,
    controlled_ablation: bool = False,
    high_vram: bool = False,
    physical_v2: bool = False,
) -> tuple[str, ...]:
    root_files = (
        (
            HIGH_VRAM_RUNTIME_ROOT_FILES
            if high_vram
            else (
                CONTROLLED_RUNTIME_ROOT_FILES
                if controlled_ablation
                else RUNTIME_ROOT_FILES
            )
        )
        if git_hash is None
        else _runtime_source_contract(
            git_hash,
            controlled_ablation=controlled_ablation,
            high_vram=high_vram,
            physical_v2=physical_v2,
        )[0]
    )
    if git_hash is None and physical_v2:
        _require(
            not controlled_ablation and not high_vram,
            "architecture-v2 and D-025 runtime roles are separate",
        )
        root_files = (
            *RUNTIME_ROOT_FILES[:-1],
            *ARCHITECTURE_V2_RUNTIME_ADDITIONS,
            RUNTIME_ROOT_FILES[-1],
        )
    source_paths = (
        tuple(
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        )
        if git_hash is None
        else _git_tree_runtime_python_paths(git_hash)
    )
    return (
        *root_files,
        *source_paths,
    )


def _validate_source_bundle(
    value: object,
    *,
    git_hash: str,
    controlled_ablation: bool,
    high_vram: bool = False,
    physical_v2: bool = False,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Stage-C runtime source bundle is missing")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{git_hash}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _require(result.returncode == 0, "Stage-C source Git commit is unavailable locally")
    _, expected_scopes, expected_file_count = _runtime_source_contract(
        git_hash,
        controlled_ablation=controlled_ablation,
        high_vram=high_vram,
        physical_v2=physical_v2,
    )
    required = {
        "schema_version",
        "git_head",
        "relevant_paths_clean",
        "git_scopes",
        "files",
        "bundle_sha256",
    }
    _require(
        set(value) == required
        and value.get("schema_version") == 1
        and value.get("git_head") == git_hash
        and value.get("relevant_paths_clean") is True,
        "Stage-C runtime source bundle header is malformed",
    )
    _require(
        value.get("git_scopes") == list(expected_scopes),
        "Stage-C runtime source Git scopes are non-canonical",
    )
    files = value.get("files")
    _require(
        isinstance(files, list) and len(files) == expected_file_count,
        "Stage-C runtime source bundle has the wrong file count for its Git tree",
    )
    expected_paths = _expected_runtime_paths(
        git_hash=git_hash,
        controlled_ablation=controlled_ablation,
        high_vram=high_vram,
        physical_v2=physical_v2,
    )
    _require(
        len(expected_paths) == expected_file_count,
        "auditor Stage-C runtime path count differs from its Git-tree contract",
    )
    actual_paths = [item.get("path") if isinstance(item, Mapping) else None for item in files]
    _require(
        actual_paths == list(expected_paths),
        "Stage-C runtime source bundle is truncated, reordered, or non-canonical",
    )
    encoded = json.dumps(
        {"git_head": git_hash, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    bundle_sha = _require_sha256(value.get("bundle_sha256"), "runtime bundle SHA-256")
    _require(_sha256_bytes(encoded) == bundle_sha, "runtime source bundle SHA-256 differs")
    for record, relative in zip(files, expected_paths, strict=True):
        _require(isinstance(record, Mapping), f"runtime source record is malformed: {relative}")
        recorded_sha = _require_sha256(record.get("sha256"), f"runtime source {relative}")
        shown = subprocess.run(
            ["git", "show", f"{git_hash}:{relative}"],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _require(shown.returncode == 0, f"checkpoint Git tree lacks runtime source: {relative}")
        _require(
            _sha256_bytes(shown.stdout) == recorded_sha,
            f"runtime source hash differs from checkpoint Git tree: {relative}",
        )
    return {
        "git_hash": git_hash,
        "bundle_sha256": bundle_sha,
        "file_count": len(files),
        "all_files_match_checkpoint_git_tree": True,
    }


def _validate_d025_prerequisite(
    value: object,
    *,
    base_checkpoint: Mapping[str, Any],
    expected_formal_audit_path: str,
) -> dict[str, Any]:
    """Validate the embedded, already-controlled D-025 Stage-B prerequisite."""

    _require(isinstance(value, Mapping), "Stage-C D-025 prerequisite is missing")
    _require(
        value.get("schema_version") == 1
        and value.get("component")
        == "d025-stage-b-prerequisite-for-stage-c-positivity"
        and value.get("status") == "PASS"
        and value.get("protocol_version") == STAGE_C_D025_POSITIVITY_PROTOCOL
        and value.get("canonical_stage_c_replacement") is False,
        "Stage-C D-025 prerequisite did not record a controlled PASS",
    )
    prerequisite_base = value.get("base_checkpoint")
    _require(
        isinstance(prerequisite_base, Mapping)
        and prerequisite_base.get("path") == base_checkpoint.get("path")
        and prerequisite_base.get("sha256") == base_checkpoint.get("sha256")
        and prerequisite_base.get("step") == FORMAL_STAGE_B_STEPS,
        "Stage-C D-025 prerequisite does not bind the frozen base",
    )
    formal_audit = value.get("formal_evaluation_audit")
    _require(
        isinstance(formal_audit, Mapping)
        and set(formal_audit)
        == {"path", "sha256", "component", "status", "final_gate"}
        and isinstance(formal_audit.get("path"), str)
        and bool(str(formal_audit["path"]).strip())
        and formal_audit.get("path") == expected_formal_audit_path
        and formal_audit.get("component")
        == "d025-positivity-final-evaluation-audit"
        and formal_audit.get("status")
        == "D025_FINAL_CONTROLLED_COMPARISON_PASS",
        "Stage-C D-025 prerequisite lacks the formal evaluation PASS audit",
    )
    _require_sha256(
        formal_audit.get("sha256"),
        "Stage-C D-025 formal evaluation audit SHA-256",
    )
    final_gate = formal_audit.get("final_gate")
    _require(
        isinstance(final_gate, Mapping)
        and final_gate.get("eligible") is True
        and final_gate.get("result") == "PASS"
        and final_gate.get("limited_or_intermediate_cannot_pass") is True,
        "Stage-C D-025 formal evaluation audit final gate did not pass",
    )
    formal_audit_path = Path(str(formal_audit["path"])).expanduser().resolve()
    _require(
        formal_audit_path == Path(expected_formal_audit_path).expanduser().resolve(),
        "Stage-C D-025 formal evaluation audit resolved path differs",
    )
    formal_audit_payload = _read_regular_local_file(
        formal_audit_path, "Stage-C D-025 formal evaluation audit"
    )
    _require(
        _sha256_bytes(formal_audit_payload) == formal_audit["sha256"],
        "Stage-C D-025 formal evaluation audit content SHA-256 differs",
    )
    try:
        live_formal_audit = _strict_json_loads(
            formal_audit_payload.decode("utf-8"),
            "Stage-C D-025 formal evaluation audit",
        )
    except UnicodeDecodeError as exc:
        raise EpipolarTrainingAuditError(
            "Stage-C D-025 formal evaluation audit is not UTF-8"
        ) from exc
    _require(
        isinstance(live_formal_audit, Mapping)
        and live_formal_audit.get("schema_version") == 1
        and live_formal_audit.get("component") == formal_audit["component"]
        and live_formal_audit.get("status") == formal_audit["status"]
        and live_formal_audit.get("read_only") is True
        and live_formal_audit.get("final_gate") == dict(final_gate),
        "Stage-C D-025 live formal evaluation audit differs from its checkpoint identity",
    )
    artifacts = live_formal_audit.get("artifacts")
    _require(
        isinstance(artifacts, Mapping),
        "Stage-C D-025 formal evaluation audit artifacts are missing",
    )

    def artifact_path(name: str) -> Path:
        identity = artifacts.get(name)
        _require(
            isinstance(identity, Mapping)
            and isinstance(identity.get("path"), str)
            and bool(str(identity["path"]).strip()),
            f"Stage-C D-025 formal audit artifact path is missing: {name}",
        )
        return Path(str(identity["path"])).expanduser().resolve()

    try:
        recomputed_formal_audit = audit_d025_evaluation(
            artifact_path("d025_training_audit"),
            artifact_path("d025_metrics").parent,
            artifact_path("canonical_stage_b_report"),
            artifact_path("canonical_metrics").parent,
            artifact_path("d025_preflight"),
        )
    except D025EvaluationAuditError as exc:
        raise EpipolarTrainingAuditError(
            f"Stage-C D-025 formal evaluation audit no longer verifies: {exc}"
        ) from exc
    _require(
        dict(recomputed_formal_audit) == dict(live_formal_audit),
        "Stage-C D-025 formal evaluation audit differs from full recomputation",
    )
    return {
        "status": "PASS",
        "protocol_version": STAGE_C_D025_POSITIVITY_PROTOCOL,
        "base_checkpoint": {
            "path": str(base_checkpoint["path"]),
            "sha256": str(base_checkpoint["sha256"]),
            "step": FORMAL_STAGE_B_STEPS,
        },
        "formal_evaluation_audit": {
            "path": str(formal_audit["path"]),
            "sha256": str(formal_audit["sha256"]),
            "component": str(formal_audit["component"]),
            "status": str(formal_audit["status"]),
            "final_gate": dict(final_gate),
        },
        "canonical_stage_c_replacement": False,
    }


def _validate_completion(
    value: object,
    *,
    step: int,
    controlled_ablation: bool,
    architecture_v2: bool = False,
    high_vram: bool = False,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Stage-C completion receipt is missing")
    expected = {
        "actual_step": step,
        "configured_steps": FORMAL_STAGE_C_STEPS,
        "execution_complete": step == FORMAL_STAGE_C_STEPS,
        "canonical_schedule": True,
        "base_complete": True,
        "cuda_bf16_eligible": True,
        "strict_determinism_eligible": True,
        "formal_training_complete": (
            step == FORMAL_STAGE_C_STEPS
            and not controlled_ablation
            and not architecture_v2
        ),
    }
    if controlled_ablation:
        expected.update(
            {
                "controlled_ablation_training_complete": (
                    step == FORMAL_STAGE_C_STEPS
                ),
                "canonical_stage_c_replacement": False,
            }
        )
    if architecture_v2:
        expected.update(
            {
                "architecture_v2_training_complete": (
                    step == FORMAL_STAGE_C_STEPS
                ),
                "canonical_stage_c_replacement": False,
            }
        )
    if high_vram:
        expected["high_vram_preflight_passed"] = True
    _require(dict(value) == expected, "Stage-C completion receipt is inconsistent")
    return expected


def _validate_high_vram_preflight(
    value: object,
    *,
    config: Mapping[str, Any],
    base_checkpoint: Mapping[str, Any],
    d025_prerequisite: Mapping[str, Any] | None,
    runtime_source_bundle: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        _require(value is None, "standard Stage-C checkpoint declares high-VRAM preflight")
        return None
    _require(
        isinstance(value, Mapping) and d025_prerequisite is not None,
        "high-VRAM Stage-C checkpoint lacks its preflight/D-025 receipt",
    )
    section = config.get("stage_c_high_vram")
    path = section.get("preflight_receipt_path") if isinstance(section, Mapping) else None
    _require(
        isinstance(path, str) and bool(path),
        "high-VRAM Stage-C preflight path is missing",
    )
    try:
        recomputed = validate_stage_c_high_vram_preflight_receipt(
            path=path,
            config=config,
            base_checkpoint={
                "path": base_checkpoint["path"],
                "sha256": base_checkpoint["sha256"],
                "step": base_checkpoint["step"],
            },
            d025_prerequisite=d025_prerequisite,
            runtime_source_bundle=runtime_source_bundle,
            training_runtime=training_runtime,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise EpipolarTrainingAuditError(
            f"Stage-C high-VRAM preflight no longer verifies: {exc}"
        ) from exc
    _require(
        dict(recomputed) == dict(value),
        "Stage-C embedded high-VRAM preflight differs from recomputation",
    )
    return dict(recomputed)


def _validate_checkpoint(path: Path, label: str) -> CheckpointSnapshot:
    payload, payload_bytes = _safe_torch_load(path, f"{label} Stage-C checkpoint")
    required = {
        "schema_version",
        "component",
        "model_component",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "config",
        "git_hash",
        "rng_states",
        "data_cursor",
        "base_checkpoint",
        "base_lineage",
        "raw_lineage",
        "base_completion",
        "geometry_contract",
        "rectification_audit",
        "runtime_source_bundle",
        "training_runtime",
        "supervision",
        "parameter_count",
        "trainable_refiner_parameter_count",
        "loss",
        "elapsed_seconds",
        "completion",
    }
    missing = sorted(required.difference(payload))
    _require(not missing, f"{label} Stage-C checkpoint fields are missing: {missing}")
    _require(
        payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and payload.get("component") == STAGE_C_COMPONENT
        and payload.get("model_component") == STAGE_C_MODEL_COMPONENT,
        f"{label} Stage-C checkpoint component/schema mismatch",
    )
    step = _positive_int(payload.get("step"), f"{label} checkpoint step")
    _require(step <= FORMAL_STAGE_C_STEPS, f"{label} checkpoint exceeds 5000 steps")
    _require(
        payload.get("parameter_count") == EXPECTED_REFINER_PARAMETERS
        and payload.get("trainable_refiner_parameter_count")
        == EXPECTED_REFINER_PARAMETERS,
        f"{label} Stage-C checkpoint does not declare 69,905 trainable parameters",
    )
    config, config_values = _validate_config(payload.get("config"))
    git_hash = payload.get("git_hash")
    _require(
        isinstance(git_hash, str) and GIT_HASH_PATTERN.fullmatch(git_hash) is not None,
        f"{label} Stage-C git hash is malformed",
    )
    model_state = payload.get("model")
    _require(isinstance(model_state, Mapping), f"{label} Stage-C model state is malformed")
    refiner = _build_refiner(config)
    try:
        refiner.load_state_dict(model_state, strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise EpipolarTrainingAuditError(
            f"{label} Stage-C model state is incompatible: {exc}"
        ) from exc
    for name, value in list(refiner.named_parameters()) + list(refiner.named_buffers()):
        if value.is_floating_point() or value.is_complex():
            _require(
                bool(torch.isfinite(value).all().item()),
                f"{label} Stage-C refiner state is non-finite: {name}",
            )
    training_state = _validate_optimizer_scheduler_rng(
        payload,
        refiner=refiner,
        step=step,
        config_values=config_values,
    )
    cursor = _validate_data_cursor(
        payload.get("data_cursor"),
        step=step,
        grad_accumulation=int(config_values["grad_accumulation"]),
    )
    runtime = _validate_training_runtime(payload.get("training_runtime"))
    base = _validate_base_checkpoint(
        payload.get("base_checkpoint"), payload.get("base_completion")
    )
    positivity = config_values["stage_c_positivity_ablation"]
    controlled_ablation = bool(positivity["enabled"])
    architecture_v2 = config_values["stage_c_architecture_v2"]
    physical_v2 = bool(architecture_v2["enabled"])
    high_vram_validation = config_values["stage_c_high_vram"]
    high_vram = bool(high_vram_validation["enabled"])
    if controlled_ablation:
        _require(
            payload.get("experiment_role") == CONTROLLED_D025_STAGE_C_ROLE,
            f"{label} Stage-C checkpoint lacks the controlled-ablation role",
        )
        d025_prerequisite = _validate_d025_prerequisite(
            payload.get("d025_prerequisite"),
            base_checkpoint=base,
            expected_formal_audit_path=str(
                config["stage_c_positivity_ablation"]["d025_evaluation_audit_path"]
            ),
        )
    elif physical_v2:
        _require(
            payload.get("experiment_role") == ARCHITECTURE_V2_STAGE_C_ROLE
            and "d025_prerequisite" not in payload,
            f"{label} architecture-v2 Stage-C checkpoint role/lineage differs",
        )
        d025_prerequisite = None
    else:
        _require(
            "experiment_role" not in payload and "d025_prerequisite" not in payload,
            f"{label} canonical Stage-C checkpoint unexpectedly declares D-025 lineage",
        )
        d025_prerequisite = None
    _require(
        payload.get("geometry_contract") == EPIPOLAR_GEOMETRY_CONTRACT,
        f"{label} Stage-C epipolar geometry contract differs",
    )
    rectification = _validate_rectification_audit(
        payload.get("rectification_audit"), config=config
    )
    source = _validate_source_bundle(
        payload.get("runtime_source_bundle"),
        git_hash=git_hash,
        controlled_ablation=controlled_ablation,
        high_vram=high_vram,
        physical_v2=physical_v2,
    )
    high_vram_preflight = _validate_high_vram_preflight(
        payload.get("high_vram_preflight"),
        config=config,
        base_checkpoint=base,
        d025_prerequisite=(
            payload.get("d025_prerequisite")
            if controlled_ablation
            else None
        ),
        runtime_source_bundle=payload["runtime_source_bundle"],
        training_runtime=payload["training_runtime"],
        enabled=high_vram,
    )
    completion = _validate_completion(
        payload.get("completion"),
        step=step,
        controlled_ablation=controlled_ablation,
        architecture_v2=physical_v2,
        high_vram=high_vram,
    )
    _require(isinstance(payload.get("base_lineage"), Mapping), "Stage-C base lineage is missing")
    _require(isinstance(payload.get("raw_lineage"), Mapping), "Stage-C raw lineage is missing")
    elapsed = _finite_float(payload.get("elapsed_seconds"), "checkpoint elapsed_seconds")
    _require(elapsed > 0.0, "checkpoint elapsed_seconds must be positive")
    loss = payload.get("loss")
    _require(isinstance(loss, Mapping), "Stage-C checkpoint loss is malformed")
    _finite_tree(loss, "checkpoint.loss")
    payload_report = {
        "training_state": training_state,
        "data_cursor": cursor,
        "training_runtime": runtime,
        "base_checkpoint": base,
        "geometry_contract": dict(EPIPOLAR_GEOMETRY_CONTRACT),
        "rectification_audit": rectification,
        "runtime_source_bundle": source,
        "completion": completion,
        "stage_c_positivity_ablation": {
            **dict(positivity),
            "d025_prerequisite": d025_prerequisite,
        },
        "stage_c_architecture_v2": dict(architecture_v2),
        "stage_c_high_vram": {
            **dict(high_vram_validation),
            "preflight": high_vram_preflight,
        },
    }
    # Store audit-only validation results outside the producer payload namespace.
    audited_payload = dict(payload)
    audited_payload["__audit_validation__"] = payload_report
    return CheckpointSnapshot(
        path=path.resolve(),
        sha256=_sha256_bytes(payload_bytes),
        byte_size=len(payload_bytes),
        step=step,
        checkpoint_interval=int(config_values["checkpoint_interval"]),
        learning_rate=float(training_state["expected_learning_rate"]),
        elapsed_seconds=elapsed,
        git_hash=git_hash,
        config_sha256=_canonical_config_sha256(config),
        config=config,
        payload=audited_payload,
    )


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(torch.equal(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def _same_latest_final(latest: CheckpointSnapshot, final: CheckpointSnapshot) -> None:
    _require(
        _tree_equal(latest.payload, final.payload),
        "latest.pt and final.pt do not contain identical training state",
    )


def _parse_training_log(
    path: Path, *, complete: bool
) -> tuple[list[dict[str, Any]], bytes, list[str]]:
    payload = _read_regular_local_file(path, "Stage-C train.jsonl")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpipolarTrainingAuditError("Stage-C train.jsonl is not UTF-8") from exc
    warnings: list[str] = []
    if text and not text.endswith("\n"):
        if complete:
            raise EpipolarTrainingAuditError(
                "completed Stage-C train.jsonl has an incomplete final line"
            )
        prefix, separator, suffix = text.rpartition("\n")
        _require(bool(separator), "in-progress train.jsonl has no complete JSONL row")
        text = prefix + "\n"
        warnings.append(
            "ignored one unterminated in-progress JSONL suffix "
            f"({len(suffix.encode('utf-8'))} bytes)"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"train.jsonl line {line_number} is blank")
        record = _strict_json_loads(line, f"train.jsonl line {line_number}")
        _require(isinstance(record, dict), f"train.jsonl line {line_number} is not an object")
        records.append(record)
    _require(bool(records), "Stage-C train.jsonl contains no complete rows")
    return records, payload, warnings


def _validate_training_log(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint: CheckpointSnapshot,
) -> dict[str, Any]:
    required = {
        "step",
        "stage",
        "learning_rate",
        "gradient_norm",
        "elapsed_seconds",
        "loss",
    }
    expected_loss = {
        "total",
        "disparity",
        "correction_regularizer",
        "valid_pixel_count",
    }
    positivity_validation = checkpoint.payload["__audit_validation__"][
        "stage_c_positivity_ablation"
    ]
    controlled_ablation = bool(positivity_validation["enabled"])
    architecture_v2 = bool(
        checkpoint.payload["__audit_validation__"]["stage_c_architecture_v2"][
            "enabled"
        ]
    )
    numeric_loss_terms = ["total", "disparity", "correction_regularizer"]
    if controlled_ablation or architecture_v2:
        expected_loss.add("positivity_penalty")
        numeric_loss_terms.append("positivity_penalty")
    train = checkpoint.config["train"]
    previous_elapsed: float | None = None
    resume_boundaries: list[int] = []
    elapsed_values: list[float] = []
    for expected_step, record in enumerate(records, start=1):
        missing = sorted(required.difference(record))
        _require(not missing, f"train.jsonl step {expected_step} fields are missing: {missing}")
        _finite_tree(record, f"train.jsonl[{expected_step}]")
        _require(
            record.get("step") == expected_step,
            f"Stage-C train.jsonl steps are not continuous at row {expected_step}",
        )
        _require(
            record.get("stage") == "epipolar",
            f"train.jsonl stage differs at step {expected_step}",
        )
        actual_lr = _finite_float(
            record.get("learning_rate"),
            f"step {expected_step} learning_rate",
        )
        expected_lr = float(train["learning_rate"]) * _learning_rate_multiplier(
            expected_step,
            total_steps=FORMAL_STAGE_C_STEPS,
            warmup_steps=int(train["warmup_steps"]),
        )
        _require(
            math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12),
            f"learning rate at step {expected_step} differs from exact schedule",
        )
        gradient = _finite_float(record.get("gradient_norm"), f"step {expected_step} gradient_norm")
        _require(gradient >= 0.0, f"gradient norm at step {expected_step} is negative")
        elapsed = _finite_float(
            record.get("elapsed_seconds"),
            f"step {expected_step} elapsed_seconds",
        )
        _require(elapsed > 0.0, f"elapsed time at step {expected_step} is non-positive")
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            resume_boundaries.append(expected_step)
        previous_elapsed = elapsed
        elapsed_values.append(elapsed)
        loss = record.get("loss")
        _require(
            isinstance(loss, Mapping) and set(loss) == expected_loss,
            f"Stage-C loss schema differs at step {expected_step}",
        )
        for name in numeric_loss_terms:
            value = _finite_float(loss.get(name), f"step {expected_step} loss.{name}")
            _require(value >= 0.0, f"loss.{name} is negative at step {expected_step}")
        _positive_int(loss.get("valid_pixel_count"), f"step {expected_step} valid_pixel_count")
    last_step = len(records)
    _require(last_step <= FORMAL_STAGE_C_STEPS, "train.jsonl exceeds 5000 steps")
    _require(checkpoint.step <= last_step, "latest.pt is ahead of complete train.jsonl rows")
    checkpoint_record = records[checkpoint.step - 1]
    _require(
        math.isclose(
            float(checkpoint_record["learning_rate"]),
            checkpoint.learning_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "latest.pt learning rate differs from its train.jsonl row",
    )
    checkpoint_loss = checkpoint.payload.get("loss")
    _require(
        isinstance(checkpoint_loss, Mapping)
        and dict(checkpoint_loss) == dict(checkpoint_record["loss"]),
        "latest.pt loss differs from its train.jsonl row",
    )
    return {
        "records": last_step,
        "first_step": 1,
        "last_step": last_step,
        "steps_continuous": True,
        "learning_rate_schedule_exact": True,
        "finite": True,
        "loss_schema": {
            "stage_c_positivity_ablation_enabled": controlled_ablation,
            **(
                {"stage_c_architecture_v2_enabled": True}
                if architecture_v2
                else {}
            ),
            "terms": sorted(numeric_loss_terms),
            "valid_pixel_count": True,
        },
        "elapsed_seconds_last": elapsed_values[-1],
        "elapsed_resume_boundaries": resume_boundaries,
    }


def _load_summary(path: Path) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_regular_local_file(path, "Stage-C run_summary.json")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpipolarTrainingAuditError("Stage-C run summary is not UTF-8") from exc
    value = _strict_json_loads(text, "run_summary.json")
    _require(isinstance(value, Mapping), "Stage-C run summary is not an object")
    _finite_tree(value, "run_summary")
    return value, payload


def _summary_artifact(
    value: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    name: str,
) -> None:
    _require(isinstance(value, Mapping), f"run summary {name} is malformed")
    path_value = value.get("path")
    _require(isinstance(path_value, str), f"run summary {name} path is malformed")
    _require(
        Path(path_value).expanduser().resolve() == expected_path.resolve(),
        f"run summary {name} path differs",
    )
    _require(value.get("sha256") == expected_sha256, f"run summary {name} SHA-256 differs")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    latest: CheckpointSnapshot,
    final: CheckpointSnapshot,
    log_path: Path,
    log_sha256: str,
    last_logged_step: int,
    last_logged_elapsed_seconds: float,
) -> dict[str, Any]:
    positivity = final.payload["__audit_validation__"][
        "stage_c_positivity_ablation"
    ]
    controlled_ablation = bool(positivity["enabled"])
    architecture_v2 = bool(
        final.payload["__audit_validation__"]["stage_c_architecture_v2"][
            "enabled"
        ]
    )
    high_vram_validation = final.payload["__audit_validation__"][
        "stage_c_high_vram"
    ]
    high_vram = bool(high_vram_validation["enabled"])
    expected_role = (
        CONTROLLED_D025_STAGE_C_ROLE
        if controlled_ablation
        else (
            ARCHITECTURE_V2_STAGE_C_ROLE
            if architecture_v2
            else CANONICAL_STAGE_C_ROLE
        )
    )
    _require(
        summary.get("schema_version") == RUN_SUMMARY_SCHEMA_VERSION
        and summary.get("component") == RUN_COMPONENT
        and summary.get("status") == "TRAINING_COMPLETE"
        and summary.get("stage") == "epipolar",
        "Stage-C run summary component/schema/status differs",
    )
    if controlled_ablation:
        _require(
            summary.get("experiment_role") == expected_role
            and isinstance(summary.get("d025_prerequisite"), Mapping)
            and dict(summary["d025_prerequisite"])
            == dict(final.payload["d025_prerequisite"]),
            "controlled Stage-C run summary D-025 identity differs from final.pt",
        )
    elif architecture_v2:
        _require(
            summary.get("experiment_role") == expected_role
            and summary.get("d025_prerequisite") is None
            and summary.get("controlled_ablation_training_complete", False)
            is False,
            "architecture-v2 Stage-C run summary role/lineage differs from final.pt",
        )
    else:
        _require(
            summary.get("experiment_role", CANONICAL_STAGE_C_ROLE)
            == CANONICAL_STAGE_C_ROLE
            and summary.get("d025_prerequisite") is None
            and summary.get("controlled_ablation_training_complete", False)
            is False,
            "canonical Stage-C run summary unexpectedly declares D-025 completion",
        )
    if high_vram:
        _require(
            isinstance(summary.get("high_vram_preflight"), Mapping)
            and dict(summary["high_vram_preflight"])
            == dict(final.payload["high_vram_preflight"]),
            "high-VRAM run summary preflight differs from final.pt",
        )
    else:
        _require(
            "high_vram_preflight" not in summary,
            "standard Stage-C run summary unexpectedly declares high-VRAM preflight",
        )
    _require(
        summary.get("steps") == FORMAL_STAGE_C_STEPS
        and summary.get("configured_steps") == FORMAL_STAGE_C_STEPS
        and last_logged_step == FORMAL_STAGE_C_STEPS
        and latest.step == FORMAL_STAGE_C_STEPS
        and final.step == FORMAL_STAGE_C_STEPS,
        "Stage-C summary/checkpoints/log did not all reach step 5000",
    )
    _require(summary.get("git_hash") == final.git_hash, "run summary git hash differs")
    _require(summary.get("config_sha256") == final.config_sha256, "run summary config SHA differs")
    _require(
        summary.get("training_runtime") == final.payload.get("training_runtime"),
        "run summary training runtime differs from final.pt",
    )
    _require(
        summary.get("base_checkpoint") == final.payload.get("base_checkpoint")
        and summary.get("base_completion") == final.payload.get("base_completion"),
        "run summary frozen-base receipt differs from final.pt",
    )
    bundle = final.payload["runtime_source_bundle"]
    _require(
        summary.get("runtime_source_bundle_sha256") == bundle.get("bundle_sha256"),
        "run summary runtime source bundle SHA differs",
    )
    if controlled_ablation:
        _require(
            summary.get("formal_training_complete") is False
            and final.payload["completion"].get("formal_training_complete") is False
            and summary.get("controlled_ablation_training_complete") is True
            and final.payload["completion"].get(
                "controlled_ablation_training_complete"
            )
            is True,
            "controlled Stage-C completion flags are inconsistent",
        )
    elif architecture_v2:
        _require(
            summary.get("formal_training_complete") is False
            and final.payload["completion"].get("formal_training_complete") is False
            and summary.get("architecture_v2_training_complete") is True
            and final.payload["completion"].get(
                "architecture_v2_training_complete"
            )
            is True,
            "architecture-v2 Stage-C completion flags are inconsistent",
        )
    else:
        _require(
            summary.get("formal_training_complete") is True
            and final.payload["completion"].get("formal_training_complete") is True,
            "run summary/final checkpoint do not certify formal_training_complete",
        )
    _summary_artifact(
        summary.get("final_checkpoint"),
        expected_path=final.path,
        expected_sha256=final.sha256,
        name="final_checkpoint",
    )
    _summary_artifact(
        summary.get("latest_checkpoint"),
        expected_path=latest.path,
        expected_sha256=latest.sha256,
        name="latest_checkpoint",
    )
    _summary_artifact(
        summary.get("training_log"),
        expected_path=log_path,
        expected_sha256=log_sha256,
        name="training_log",
    )
    run_steps = _positive_int(summary.get("run_steps"), "run summary run_steps")
    _require(run_steps <= FORMAL_STAGE_C_STEPS, "run summary run_steps exceeds 5000")
    elapsed = _finite_float(summary.get("elapsed_seconds"), "run summary elapsed_seconds")
    segment_elapsed = _finite_float(
        summary.get("segment_elapsed_seconds"), "run summary segment_elapsed_seconds"
    )
    segment_rate = _finite_float(
        summary.get("segment_steps_per_second"), "run summary segment_steps_per_second"
    )
    _require(
        elapsed > 0.0
        and segment_elapsed > 0.0
        and segment_elapsed <= elapsed
        and segment_rate > 0.0
        and elapsed == final.elapsed_seconds
        and elapsed >= last_logged_elapsed_seconds
        and math.isclose(
            segment_rate,
            run_steps / segment_elapsed,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "run summary segment throughput is inconsistent",
    )
    for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        value = summary.get(name)
        _require(
            value is None or (_is_int(value) and int(value) >= 0),
            f"run summary {name} is malformed",
        )
    resume_start = FORMAL_STAGE_C_STEPS - run_steps
    return {
        "valid": True,
        "run_steps_in_final_segment": run_steps,
        "final_segment_start_step": resume_start,
        "resume_declared_by_summary": resume_start > 0,
        "elapsed_seconds": elapsed,
        "segment_elapsed_seconds": segment_elapsed,
        "segment_steps_per_second": segment_rate,
        "experiment_role": expected_role,
        "formal_training_complete": not controlled_ablation and not architecture_v2,
        "controlled_ablation_training_complete": controlled_ablation,
        "architecture_v2_training_complete": architecture_v2,
        "high_vram": high_vram,
    }


def audit_epipolar_training_run(output_dir: str | Path) -> dict[str, Any]:
    """Audit one Stage-C run directory without changing any inspected artifact."""

    original_root = Path(output_dir).expanduser()
    _require(original_root.exists(), f"Stage-C output directory is missing: {original_root}")
    _require(not original_root.is_symlink(), "Stage-C output directory must not be a symlink")
    root = original_root.resolve()
    _require(root.is_dir(), f"Stage-C output path is not a directory: {root}")
    latest_path = root / "latest.pt"
    final_path = root / "final.pt"
    log_path = root / "train.jsonl"
    summary_path = root / "run_summary.json"
    summary_present = summary_path.exists()
    _require(not summary_path.is_symlink(), "run_summary.json must not be a symlink")
    if summary_present:
        _require(final_path.exists(), "run_summary.json exists but final.pt is missing")

    latest = _validate_checkpoint(latest_path, "latest")
    final = _validate_checkpoint(final_path, "final") if final_path.exists() else None
    if final is not None:
        _same_latest_final(latest, final)
    records, log_bytes, warnings = _parse_training_log(
        log_path, complete=summary_present
    )
    log_validation = _validate_training_log(records, checkpoint=latest)
    last_step = int(log_validation["last_step"])
    lag = last_step - latest.step
    _require(lag >= 0, "latest.pt is ahead of train.jsonl")
    _require(
        lag <= latest.checkpoint_interval,
        "latest.pt lags train.jsonl by more than checkpoint_interval",
    )
    summary_validation: dict[str, Any] | None = None
    summary_bytes: bytes | None = None
    if summary_present:
        _require(final is not None, "completed Stage-C run has no final.pt")
        _require(lag == 0, "completed Stage-C latest.pt lags train.jsonl")
        summary, summary_bytes = _load_summary(summary_path)
        summary_validation = _validate_summary(
            summary,
            latest=latest,
            final=final,
            log_path=log_path,
            log_sha256=_sha256_bytes(log_bytes),
            last_logged_step=last_step,
            last_logged_elapsed_seconds=float(
                log_validation["elapsed_seconds_last"]
            ),
        )

    resume_boundaries = list(log_validation["elapsed_resume_boundaries"])
    summary_resume = bool(
        summary_validation
        and summary_validation.get("resume_declared_by_summary") is True
    )
    complete = summary_validation is not None
    status = "PASS" if complete else "IN_PROGRESS"
    checkpoint_validation = latest.payload["__audit_validation__"]
    positivity_validation = checkpoint_validation["stage_c_positivity_ablation"]
    controlled_ablation = bool(positivity_validation["enabled"])
    architecture_v2_validation = checkpoint_validation["stage_c_architecture_v2"]
    architecture_v2 = bool(architecture_v2_validation["enabled"])
    experiment_role = (
        CONTROLLED_D025_STAGE_C_ROLE
        if controlled_ablation
        else (
            ARCHITECTURE_V2_STAGE_C_ROLE
            if architecture_v2
            else CANONICAL_STAGE_C_ROLE
        )
    )
    formal_training_complete = bool(
        complete and not controlled_ablation and not architecture_v2
    )
    controlled_ablation_training_complete = bool(
        complete and controlled_ablation
    )
    architecture_v2_training_complete = bool(complete and architecture_v2)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": AUDIT_COMPONENT,
        "status": status,
        "training_status": "TRAINING_COMPLETE" if complete else "IN_PROGRESS",
        "experiment_role": experiment_role,
        "read_only": True,
        "safe_load": {
            "torch_weights_only": True,
            "arbitrary_pickle_globals_enabled": False,
            "symlink_artifacts_allowed": False,
        },
        "output_dir": str(root),
        "files": {
            "latest_checkpoint": latest.report(),
            "final_checkpoint": final.report() if final is not None else None,
            "training_log": {
                "path": str(log_path),
                "sha256": _sha256_bytes(log_bytes),
                "byte_size": len(log_bytes),
                "complete_records": len(records),
            },
            "run_summary": (
                None
                if summary_bytes is None
                else {
                    "path": str(summary_path),
                    "sha256": _sha256_bytes(summary_bytes),
                    "byte_size": len(summary_bytes),
                }
            ),
        },
        "checkpoint_validation": checkpoint_validation,
        "log_validation": {
            **log_validation,
            "latest_checkpoint_lag_steps": lag,
            "maximum_allowed_lag_steps": latest.checkpoint_interval,
            "warnings": warnings,
        },
        "resume": {
            "detected": bool(resume_boundaries or summary_resume),
            "elapsed_reset_boundaries": resume_boundaries,
            "declared_by_completion_summary": summary_resume,
            "final_segment_start_step": (
                summary_validation.get("final_segment_start_step")
                if summary_validation is not None
                else None
            ),
        },
        "completion": {
            "receipt_present": summary_present,
            "receipt_valid": complete,
            "formal_training_complete": formal_training_complete,
            "controlled_ablation_training_complete": (
                controlled_ablation_training_complete
            ),
            "architecture_v2_training_complete": (
                architecture_v2_training_complete
            ),
            "summary": summary_validation,
        },
    }


def _json_outside_run(path: Path, run_root: Path) -> Path:
    output = path.expanduser().resolve()
    root = run_root.expanduser().resolve()
    _require(
        output != root and root not in output.parents,
        "--json-out must not write inside the audited Stage-C directory",
    )
    return output


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only formal/in-progress audit of one Stage-C training run."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.output_dir.expanduser().resolve()
        json_out = (
            None
            if args.json_out is None
            else _json_outside_run(args.json_out, root)
        )
        report = audit_epipolar_training_run(args.output_dir)
        if json_out is not None:
            _write_json_atomic(json_out, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (EpipolarTrainingAuditError, OSError, subprocess.SubprocessError) as exc:
        error = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "component": AUDIT_COMPONENT,
            "status": "FAIL",
            "error": str(exc),
        }
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
