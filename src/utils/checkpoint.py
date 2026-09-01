"""Atomic, provenance-bearing training checkpoints."""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from data.cache_dataset import sha256_file


CHECKPOINT_SCHEMA_VERSION = 1

# These fields were introduced only to select the opt-in architecture-v3
# lineage.  Historical v1/v2 checkpoints predate them.  Resume comparison may
# therefore treat an absent field as its exact legacy-safe value, but must not
# normalize any enabled/non-default v3 value (or any unrelated config field).
_LEGACY_SAFE_CALIBRATION_V3 = {
    "enabled": False,
    "protocol_version": "disabled",
    "use_rays": False,
    "use_stereo_pose": False,
    "use_temporal_pose": False,
}
_LEGACY_SAFE_DATA_V3_DEFAULTS = {
    "calibration_sidecar_path": None,
    "derived_contract": "legacy_v1",
    "calibration_sidecar_lineage": None,
}


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint cannot safely resume the current run."""


def _plain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip a resolved config through strict canonical JSON."""

    try:
        encoded = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("config must contain only finite JSON values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise TypeError("config must resolve to a mapping")
    return decoded


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return the canonical JSON representation used for exact comparison."""

    return json.dumps(
        _plain_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _resume_compatibility_fingerprint(config: Mapping[str, Any]) -> str:
    """Fingerprint with only the declared v3-disabled legacy aliases filled.

    This deliberately is not the public/config provenance fingerprint.  It is
    used only when resuming checkpoints written before v3 added optional config
    keys.  Existing keys are never overwritten or partially repaired, so a
    malformed or enabled v3 treatment remains an exact mismatch.
    """

    compatible = _plain_config(config)
    if "calibration_conditioning_v3" not in compatible:
        compatible["calibration_conditioning_v3"] = dict(
            _LEGACY_SAFE_CALIBRATION_V3
        )
    data = compatible.get("data")
    if isinstance(data, dict):
        for name, value in _LEGACY_SAFE_DATA_V3_DEFAULTS.items():
            if name not in data:
                data[name] = value
    return json.dumps(
        compatible,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all available CUDA RNG states."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise CheckpointMismatchError(f"checkpoint RNG state is missing {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointMismatchError(
                "checkpoint has CUDA RNG states but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointMismatchError(
                "checkpoint CUDA RNG device count does not match this host"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def repository_git_hash(repository_root: str | Path) -> str:
    """Return the current Git commit, or ``"unknown"`` outside a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    value = result.stdout.strip()
    return value if len(value) == 40 else "unknown"


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write a Torch payload and atomically replace ``path`` on success."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_step: int,
    config: Mapping[str, Any],
    git_hash: str,
    parameter_count: int,
    scaler: Any | None = None,
) -> None:
    """Atomically save all state required by the Stage-A resume contract."""

    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise TypeError("completed_step must be an integer")
    if completed_step < 0:
        raise ValueError("completed_step must be non-negative")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int):
        raise TypeError("parameter_count must be an integer")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    resolved_config = _plain_config(config)
    scaler_state = {} if scaler is None else scaler.state_dict()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler_state,
        "step": completed_step,
        "config": resolved_config,
        "git_hash": str(git_hash),
        "parameter_count": parameter_count,
        "rng_states": capture_rng_state(),
    }
    atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_config: Mapping[str, Any],
    expected_parameter_count: int,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> int:
    """Validate and restore a checkpoint, returning its completed update step."""

    checkpoint_path = Path(path)
    # This is a locally produced training checkpoint containing Python/NumPy
    # RNG tuples, not an untrusted model artifact.  ``weights_only=False`` is
    # explicit because those tuples are intentionally outside the safe tensor
    # subset.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("checkpoint payload is not a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError(
            "checkpoint schema mismatch: expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    required = {
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
    if missing:
        raise CheckpointMismatchError(f"checkpoint fields are missing: {missing}")
    if payload["parameter_count"] != expected_parameter_count:
        raise CheckpointMismatchError(
            "model parameter count mismatch: expected "
            f"{expected_parameter_count}, got {payload['parameter_count']}"
        )
    if not isinstance(payload["config"], Mapping):
        raise CheckpointMismatchError("checkpoint config is not a mapping")
    if _resume_compatibility_fingerprint(
        payload["config"]
    ) != _resume_compatibility_fingerprint(expected_config):
        raise CheckpointMismatchError(
            "resolved training config differs from the checkpoint config"
        )

    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointMismatchError(f"checkpoint state is incompatible: {exc}") from exc

    completed_step = payload["step"]
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise CheckpointMismatchError("checkpoint step is not an integer")
    if completed_step < 0:
        raise CheckpointMismatchError("checkpoint step is negative")
    if restore_rng:
        rng_states = payload["rng_states"]
        if not isinstance(rng_states, Mapping):
            raise CheckpointMismatchError("checkpoint RNG state is not a mapping")
        restore_rng_state(rng_states)
    return completed_step


def load_model_initialization_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    expected_parameter_count: int,
    required_sequence_length: int = 1,
    required_seed: int | None = None,
    required_calibration_conditioning_v3: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load only model weights from a validated prior-stage checkpoint.

    Optimizer, scheduler, and RNG state are deliberately not restored.  The
    returned mapping is JSON-safe lineage for the downstream temporal run.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("initialization checkpoint is not a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError("initialization checkpoint schema mismatch")
    if payload.get("parameter_count") != expected_parameter_count:
        raise CheckpointMismatchError(
            "model parameter count mismatch: expected "
            f"{expected_parameter_count}, got {payload.get('parameter_count')!r}"
        )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise CheckpointMismatchError("initialization checkpoint config is malformed")
    data_config = config.get("data")
    if not isinstance(data_config, Mapping) or data_config.get(
        "sequence_length"
    ) != required_sequence_length:
        raise CheckpointMismatchError(
            "initialization checkpoint is not from the required spatial T=1 stage"
        )
    source_seed = config.get("seed")
    source_calibration: dict[str, Any] | None = None
    if required_calibration_conditioning_v3 is not None:
        required_treatment = _plain_config(
            {"calibration_conditioning_v3": required_calibration_conditioning_v3}
        )["calibration_conditioning_v3"]
        actual_treatment = config.get("calibration_conditioning_v3")
        if not isinstance(actual_treatment, Mapping) or dict(
            actual_treatment
        ) != required_treatment:
            raise CheckpointMismatchError(
                "initialization checkpoint is not the required calibration-v3 "
                "Stage-A treatment"
            )
        train_config = config.get("train")
        if not isinstance(train_config, Mapping) or str(
            train_config.get("stage", "")
        ).lower() != "spatial":
            raise CheckpointMismatchError(
                "calibration-v3 initialization checkpoint is not Stage A spatial"
            )
        if data_config.get("derived_contract") != "calibrated_stereo_v2":
            raise CheckpointMismatchError(
                "calibration-v3 initialization checkpoint has the wrong derived contract"
            )
        source_calibration = dict(actual_treatment)
    if required_seed is not None:
        if (
            isinstance(required_seed, bool)
            or not isinstance(required_seed, int)
            or required_seed < 0
        ):
            raise ValueError("required_seed must be a non-negative integer")
        if source_seed != required_seed:
            raise CheckpointMismatchError(
                "initialization checkpoint seed differs from the temporal run"
            )
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointMismatchError(
            f"initialization model state is incompatible: {exc}"
        ) from exc
    completed_step = payload.get("step")
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise CheckpointMismatchError("initialization checkpoint step is malformed")
    return {
        "path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completed_step": completed_step,
        "git_hash": str(payload.get("git_hash", "unknown")),
        "source_sequence_length": required_sequence_length,
        "source_seed": source_seed,
        "calibration_conditioning_v3": source_calibration,
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMismatchError",
    "atomic_torch_save",
    "capture_rng_state",
    "config_fingerprint",
    "load_model_initialization_checkpoint",
    "load_training_checkpoint",
    "repository_git_hash",
    "restore_rng_state",
    "save_training_checkpoint",
]
