"""Consistency with the frozen LR Fast-FoundationStereo measurement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .disparity import charbonnier, finite_masked_mean


def sample_hr_at_lr_centers(disparity_hr_px: Tensor, *, scale: int) -> Tensor:
    """Point-sample an HR map at LR pixel centres without area averaging."""

    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if disparity_hr_px.ndim != 4 or not disparity_hr_px.is_floating_point():
        raise ValueError("disparity_hr_px must be floating [B,C,H,W]")
    height_hr, width_hr = disparity_hr_px.shape[-2:]
    if height_hr % scale or width_hr % scale:
        raise ValueError(
            f"HR shape {(height_hr, width_hr)} must be divisible by scale {scale}"
        )
    if scale == 1:
        return disparity_hr_px
    # With align_corners=False, each output centre maps to the continuous
    # centre of its scale-by-scale HR footprint. This is one bilinear point,
    # not an area average across a foreground/background boundary.
    return F.interpolate(
        disparity_hr_px,
        size=(height_hr // scale, width_hr // scale),
        mode="bilinear",
        align_corners=False,
    )


def measurement_consistency_loss(
    prediction_disparity_hr_px: Tensor,
    observation_disparity_lr_px: Tensor,
    trusted_ffs_mask_lr: Tensor,
    *,
    scale: int = 2,
    confidence_ffs_lr: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Compare centre-sampled HR prediction to LR-unit FFS disparity."""

    sampled_prediction_lr_px = sample_hr_at_lr_centers(
        prediction_disparity_hr_px, scale=scale
    ) / float(scale)
    expected_shape = sampled_prediction_lr_px.shape
    for name, value in (
        ("observation_disparity_lr_px", observation_disparity_lr_px),
        ("trusted_ffs_mask_lr", trusted_ffs_mask_lr),
    ):
        if not isinstance(value, Tensor) or value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if confidence_ffs_lr is not None and confidence_ffs_lr.shape != expected_shape:
        raise ValueError(f"confidence_ffs_lr must have shape {tuple(expected_shape)}")

    usable = (
        trusted_ffs_mask_lr.to(dtype=torch.bool)
        & torch.isfinite(sampled_prediction_lr_px)
        & torch.isfinite(observation_disparity_lr_px)
        & (observation_disparity_lr_px > 0)
    )
    error = torch.where(
        usable,
        sampled_prediction_lr_px - observation_disparity_lr_px,
        torch.zeros_like(sampled_prediction_lr_px),
    )
    return finite_masked_mean(
        charbonnier(error, epsilon), usable, confidence_ffs_lr
    )


__all__ = ["measurement_consistency_loss", "sample_hr_at_lr_centers"]
