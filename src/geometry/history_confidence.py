"""Confidence gating for z-buffered temporal disparity history."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class HistoryConfidenceResult:
    """Safe temporal source confidence on the current HR/LR-aligned grid."""

    confidence: Tensor
    valid_mask: Tensor
    disparity_error_hr_px: Tensor
    rejected_current_ffs_conflict: Tensor


def _same_shape(reference: Tensor, value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def history_confidence(
    warped_previous_confidence: Tensor,
    visibility_mask: Tensor,
    collision_mask: Tensor,
    photometric_residual: Tensor,
    warped_history_disparity_hr_px: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    *,
    photometric_temperature: float = 0.10,
    disparity_temperature_hr_px: float = 2.0,
    trusted_ffs_threshold: float = 0.8,
    reject_conflict_hr_px: float = 2.0,
) -> HistoryConfidenceResult:
    """Apply visibility, residual decay, and trusted-FFS conflict rejection.

    All tensors have the same shape, normally ``[B,1,H,W]``. Disparities are
    in HR pixels even when sampled on the LR grid. Photometric residual is
    assumed to use RGB values in ``[0,1]``.
    """

    if not isinstance(warped_previous_confidence, Tensor) or not (
        warped_previous_confidence.is_floating_point()
    ):
        raise TypeError("warped_previous_confidence must be floating point")
    for name, value in (
        ("visibility_mask", visibility_mask),
        ("collision_mask", collision_mask),
        ("photometric_residual", photometric_residual),
        ("warped_history_disparity_hr_px", warped_history_disparity_hr_px),
        ("current_ffs_disparity_hr_px", current_ffs_disparity_hr_px),
        ("current_ffs_confidence", current_ffs_confidence),
    ):
        _same_shape(warped_previous_confidence, value, name)
    for name, value in (
        ("photometric_temperature", photometric_temperature),
        ("disparity_temperature_hr_px", disparity_temperature_hr_px),
        ("reject_conflict_hr_px", reject_conflict_hr_px),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= trusted_ffs_threshold <= 1.0:
        raise ValueError("trusted_ffs_threshold must lie in [0,1]")

    disparity_error_hr_px = (
        warped_history_disparity_hr_px - current_ffs_disparity_hr_px
    ).abs()
    finite = (
        torch.isfinite(warped_previous_confidence)
        & torch.isfinite(photometric_residual)
        & torch.isfinite(warped_history_disparity_hr_px)
        & (warped_history_disparity_hr_px > 0)
        & torch.isfinite(disparity_error_hr_px)
    )
    visible = (
        visibility_mask.to(dtype=torch.bool)
        & ~collision_mask.to(dtype=torch.bool)
    )
    trusted_current = (
        torch.isfinite(current_ffs_disparity_hr_px)
        & (current_ffs_disparity_hr_px > 0)
        & torch.isfinite(current_ffs_confidence)
        & (current_ffs_confidence > trusted_ffs_threshold)
    )
    rejected_conflict = (
        trusted_current
        & torch.isfinite(disparity_error_hr_px)
        & (disparity_error_hr_px > reject_conflict_hr_px)
    )
    valid = finite & visible & ~rejected_conflict

    previous = torch.where(
        valid,
        warped_previous_confidence.clamp(0.0, 1.0),
        torch.zeros_like(warped_previous_confidence),
    )
    photo = torch.where(
        valid,
        photometric_residual.clamp_min(0.0),
        torch.zeros_like(photometric_residual),
    )
    disparity_error_safe = torch.where(
        valid,
        disparity_error_hr_px,
        torch.zeros_like(disparity_error_hr_px),
    )
    confidence = (
        previous
        * torch.exp(-photo / photometric_temperature)
        * torch.exp(-disparity_error_safe / disparity_temperature_hr_px)
    )
    confidence = torch.where(
        valid & torch.isfinite(confidence),
        confidence.clamp(0.0, 1.0),
        torch.zeros_like(confidence),
    )
    return HistoryConfidenceResult(
        confidence=confidence,
        valid_mask=valid,
        disparity_error_hr_px=disparity_error_hr_px,
        rejected_current_ffs_conflict=rejected_conflict,
    )


__all__ = ["HistoryConfidenceResult", "history_confidence"]
