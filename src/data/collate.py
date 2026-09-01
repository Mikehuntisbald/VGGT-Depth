"""Batch collation for cached FFS training samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .temporal_training_dataset import TemporalTrainingSample
from .training_dataset import FFSTrainingSample


def _field(sample: FFSTrainingSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, FFSTrainingSample):
        return getattr(sample, name)
    if isinstance(sample, Mapping):
        return sample.get(name)
    raise TypeError(
        "samples must be FFSTrainingSample instances or mappings, got "
        f"{type(sample).__name__}"
    )


def _stack_required(
    samples: Sequence[FFSTrainingSample | Mapping[str, Any]], name: str
) -> Tensor:
    values = [_field(sample, name) for sample in samples]
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"required tensor field {name!r} is missing or malformed")
    try:
        return torch.stack(values)  # type: ignore[arg-type]
    except RuntimeError as exc:
        raise ValueError(f"cannot stack field {name!r}: {exc}") from exc


def _stack_optional(
    samples: Sequence[FFSTrainingSample | Mapping[str, Any]], name: str
) -> Tensor | None:
    values = [_field(sample, name) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            f"optional tensor field {name!r} is present for only part of the batch"
        )
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"optional tensor field {name!r} is malformed")
    try:
        return torch.stack(values)  # type: ignore[arg-type]
    except RuntimeError as exc:
        raise ValueError(f"cannot stack optional field {name!r}: {exc}") from exc


def collate_training_samples(
    samples: Sequence[FFSTrainingSample | Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack tensors while retaining per-record provenance as a list.

    Teacher fields may be absent for the whole batch.  Mixed teacher presence
    is rejected because it cannot form a well-defined supervised loss mask.
    The returned aliases ``disparity_ffs_hr_px``, ``confidence_ffs`` and
    ``valid_ffs`` can be passed directly to the model, while the longer
    observation names preserve grid/unit meaning for losses and audits.
    """

    if not samples:
        raise ValueError("cannot collate an empty training batch")
    batch: dict[str, Any] = {
        "rgb_hr": _stack_required(samples, "rgb_hr"),
        "observation_disparity_hr_px": _stack_required(
            samples, "observation_disparity_hr_px"
        ),
        "observation_disparity_lr_px": _stack_required(
            samples, "observation_disparity_lr_px"
        ),
        "observation_confidence": _stack_required(
            samples, "observation_confidence"
        ),
        "observation_valid_mask": _stack_required(
            samples, "observation_valid_mask"
        ),
        "observation_trusted_mask": _stack_required(
            samples, "observation_trusted_mask"
        ),
        "teacher_disparity_hr_px": _stack_optional(
            samples, "teacher_disparity_hr_px"
        ),
        "teacher_confidence": _stack_optional(samples, "teacher_confidence"),
        "teacher_valid_mask": _stack_optional(samples, "teacher_valid_mask"),
        "teacher_trusted_mask": _stack_optional(
            samples, "teacher_trusted_mask"
        ),
        "K_hr": _stack_required(samples, "K_hr"),
        "baseline_m": _stack_required(samples, "baseline_m"),
        "T_right_rectified_from_left_rectified_m": _stack_optional(
            samples, "T_right_rectified_from_left_rectified_m"
        ),
        "sequence_id": [_field(sample, "sequence_id") for sample in samples],
        "frame_id": torch.tensor(
            [_field(sample, "frame_id") for sample in samples], dtype=torch.int64
        ),
        "timestamp": torch.tensor(
            [_field(sample, "timestamp") for sample in samples], dtype=torch.float64
        ),
        "identity_metadata": [
            _field(sample, "identity_metadata") for sample in samples
        ],
    }
    # Direct model/loss aliases; these are references, not tensor copies.
    batch["disparity_ffs_hr_px"] = batch["observation_disparity_hr_px"]
    batch["confidence_ffs"] = batch["observation_confidence"]
    batch["valid_ffs"] = batch["observation_valid_mask"]
    batch["target_disparity_hr_px"] = batch["teacher_disparity_hr_px"]
    batch["target_trusted_mask"] = batch["teacher_trusted_mask"]
    return batch


def _temporal_field(
    sample: TemporalTrainingSample | Mapping[str, Any], name: str
) -> Any:
    if isinstance(sample, TemporalTrainingSample):
        return getattr(sample, name)
    if isinstance(sample, Mapping):
        value = sample.get(name)
        if value is None and name == "gt_extrinsics_camera_from_world_sequence":
            value = sample.get("gt_pose_sequence")
        return value
    raise TypeError(
        "samples must be TemporalTrainingSample instances or mappings, got "
        f"{type(sample).__name__}"
    )


def _stack_temporal_required(
    samples: Sequence[TemporalTrainingSample | Mapping[str, Any]], name: str
) -> Tensor:
    values = [_temporal_field(sample, name) for sample in samples]
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"required temporal tensor field {name!r} is malformed")
    try:
        return torch.stack(values)  # type: ignore[arg-type]
    except RuntimeError as exc:
        raise ValueError(f"cannot stack temporal field {name!r}: {exc}") from exc


def _stack_temporal_optional(
    samples: Sequence[TemporalTrainingSample | Mapping[str, Any]], name: str
) -> Tensor | None:
    values = [_temporal_field(sample, name) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            f"optional temporal tensor field {name!r} is present for only part "
            "of the batch"
        )
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"optional temporal tensor field {name!r} is malformed")
    try:
        return torch.stack(values)  # type: ignore[arg-type]
    except RuntimeError as exc:
        raise ValueError(
            f"cannot stack optional temporal field {name!r}: {exc}"
        ) from exc


def collate_temporal_training_samples(
    samples: Sequence[TemporalTrainingSample | Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate causal T=3 samples to explicit ``[B,T,...]`` tensors.

    The returned source aliases are references, not copies.  In particular,
    ``history_pose_valid_sequence`` is the strict VGGT pose gate a trainer must
    check before running any z-buffer history reprojection.
    """

    if not samples:
        raise ValueError("cannot collate an empty temporal training batch")
    sequence_ids = [_temporal_field(sample, "sequence_id") for sample in samples]
    frame_ids = [_temporal_field(sample, "frame_ids") for sample in samples]
    timestamps = [_temporal_field(sample, "timestamps") for sample in samples]
    manifest_indices = [
        _temporal_field(sample, "manifest_indices") for sample in samples
    ]
    sequence_lengths = {len(value) for value in frame_ids}
    if sequence_lengths != {3}:
        raise ValueError(
            f"temporal samples must all have T=3, got lengths {sequence_lengths}"
        )
    if any(len(value) != 3 for value in timestamps + manifest_indices):
        raise ValueError("timestamps and manifest_indices must also have T=3")

    batch: dict[str, Any] = {
        "rgb_hr_sequence": _stack_temporal_required(samples, "rgb_hr_sequence"),
        "observation_disparity_hr_px_sequence": _stack_temporal_required(
            samples, "observation_disparity_hr_px_sequence"
        ),
        "observation_disparity_lr_px_sequence": _stack_temporal_required(
            samples, "observation_disparity_lr_px_sequence"
        ),
        "observation_confidence_sequence": _stack_temporal_required(
            samples, "observation_confidence_sequence"
        ),
        "observation_valid_mask_sequence": _stack_temporal_required(
            samples, "observation_valid_mask_sequence"
        ),
        "observation_trusted_mask_sequence": _stack_temporal_required(
            samples, "observation_trusted_mask_sequence"
        ),
        "teacher_disparity_hr_px_sequence": _stack_temporal_optional(
            samples, "teacher_disparity_hr_px_sequence"
        ),
        "teacher_confidence_sequence": _stack_temporal_optional(
            samples, "teacher_confidence_sequence"
        ),
        "teacher_valid_mask_sequence": _stack_temporal_optional(
            samples, "teacher_valid_mask_sequence"
        ),
        "teacher_trusted_mask_sequence": _stack_temporal_optional(
            samples, "teacher_trusted_mask_sequence"
        ),
        "K_hr_sequence": _stack_temporal_required(samples, "K_hr_sequence"),
        "baseline_m_sequence": _stack_temporal_required(
            samples, "baseline_m_sequence"
        ),
        "T_right_rectified_from_left_rectified_m_sequence": (
            _stack_temporal_optional(
                samples,
                "T_right_rectified_from_left_rectified_m_sequence",
            )
        ),
        "vggt_disparity_hr_px_sequence": _stack_temporal_required(
            samples, "vggt_disparity_hr_px_sequence"
        ),
        "vggt_confidence_sequence": _stack_temporal_required(
            samples, "vggt_confidence_sequence"
        ),
        "vggt_valid_mask_sequence": _stack_temporal_required(
            samples, "vggt_valid_mask_sequence"
        ),
        "vggt_extrinsics_camera_from_world_metric_sequence": (
            _stack_temporal_required(
                samples,
                "vggt_extrinsics_camera_from_world_metric_sequence",
            )
        ),
        "temporal_pose_valid_sequence": _stack_temporal_required(
            samples, "temporal_pose_valid_sequence"
        ),
        "temporal_pose_quality_score_sequence": _stack_temporal_required(
            samples, "temporal_pose_quality_score_sequence"
        ),
        "static_prior_valid_sequence": _stack_temporal_required(
            samples, "static_prior_valid_sequence"
        ),
        "gt_extrinsics_camera_from_world_sequence": _stack_temporal_optional(
            samples, "gt_extrinsics_camera_from_world_sequence"
        ),
        "sequence_id": sequence_ids,
        "frame_ids": torch.tensor(frame_ids, dtype=torch.int64),
        "timestamps": torch.tensor(timestamps, dtype=torch.float64),
        "manifest_indices": torch.tensor(manifest_indices, dtype=torch.int64),
        "identity_metadata": [
            _temporal_field(sample, "identity_metadata") for sample in samples
        ],
    }
    # Direct model/loss aliases.  These retain the time dimension and are
    # consumed one slice at a time by the causal training loop.
    batch["disparity_ffs_hr_px_sequence"] = batch[
        "observation_disparity_hr_px_sequence"
    ]
    batch["confidence_ffs_sequence"] = batch["observation_confidence_sequence"]
    batch["valid_ffs_sequence"] = batch["observation_valid_mask_sequence"]
    batch["disparity_vggt_hr_px_sequence"] = batch[
        "vggt_disparity_hr_px_sequence"
    ]
    batch["confidence_vggt_sequence"] = batch["vggt_confidence_sequence"]
    batch["valid_vggt_sequence"] = batch["vggt_valid_mask_sequence"]
    batch["history_pose_valid_sequence"] = batch[
        "temporal_pose_valid_sequence"
    ]
    batch["gt_pose_sequence"] = batch[
        "gt_extrinsics_camera_from_world_sequence"
    ]
    batch["target_disparity_hr_px_sequence"] = batch[
        "teacher_disparity_hr_px_sequence"
    ]
    batch["target_trusted_mask_sequence"] = batch[
        "teacher_trusted_mask_sequence"
    ]
    return batch


__all__ = ["collate_temporal_training_samples", "collate_training_samples"]
