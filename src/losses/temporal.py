"""Visibility-gated temporal disparity consistency losses."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .disparity import charbonnier, finite_masked_mean


def temporal_consistency_loss(
    prediction_disparity_hr_px: Tensor,
    warped_previous_disparity_hr_px: Tensor,
    *,
    static_mask: Tensor,
    visibility_mask: Tensor,
    collision_mask: Tensor,
    photometric_residual: Tensor,
    max_photometric_residual: float,
    geometry_consistent_mask: Tensor,
    history_confidence: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Compare current/history disparity only where transport is safe."""

    reference_shape = prediction_disparity_hr_px.shape
    values = {
        "warped_previous_disparity_hr_px": warped_previous_disparity_hr_px,
        "static_mask": static_mask,
        "visibility_mask": visibility_mask,
        "collision_mask": collision_mask,
        "photometric_residual": photometric_residual,
        "geometry_consistent_mask": geometry_consistent_mask,
    }
    for name, value in values.items():
        if not isinstance(value, Tensor) or value.shape != reference_shape:
            raise ValueError(f"{name} must have shape {tuple(reference_shape)}")
    if history_confidence is not None and history_confidence.shape != reference_shape:
        raise ValueError(f"history_confidence must have shape {tuple(reference_shape)}")
    if not math.isfinite(max_photometric_residual) or max_photometric_residual < 0:
        raise ValueError("max_photometric_residual must be finite and non-negative")

    usable = (
        static_mask.to(dtype=torch.bool)
        & visibility_mask.to(dtype=torch.bool)
        & ~collision_mask.to(dtype=torch.bool)
        & geometry_consistent_mask.to(dtype=torch.bool)
        & torch.isfinite(photometric_residual)
        & (photometric_residual <= max_photometric_residual)
        & torch.isfinite(prediction_disparity_hr_px)
        & torch.isfinite(warped_previous_disparity_hr_px)
        & (warped_previous_disparity_hr_px > 0)
    )
    error = torch.where(
        usable,
        prediction_disparity_hr_px - warped_previous_disparity_hr_px,
        torch.zeros_like(prediction_disparity_hr_px),
    )
    return finite_masked_mean(charbonnier(error, epsilon), usable, history_confidence)


__all__ = ["temporal_consistency_loss"]
