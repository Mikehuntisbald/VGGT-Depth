"""Uncertainty supervision and FFS source-gate regularisation."""

from __future__ import annotations

import torch
from torch import Tensor

from .disparity import finite_masked_mean


def laplace_uncertainty_nll(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    log_variance: Tensor,
    *,
    valid_mask: Tensor | None = None,
    minimum_log_variance: float = -10.0,
    maximum_log_variance: float = 10.0,
) -> Tensor:
    """Return ``exp(-s) * |error| + s`` for predicted ``s=log(sigma^2)``."""

    if prediction_disparity_hr_px.shape != target_disparity_hr_px.shape:
        raise ValueError("prediction and target disparity shapes must match")
    if log_variance.shape != prediction_disparity_hr_px.shape:
        raise ValueError("log_variance shape must match disparity")
    if minimum_log_variance >= maximum_log_variance:
        raise ValueError("minimum_log_variance must be less than maximum_log_variance")
    usable = (
        torch.isfinite(prediction_disparity_hr_px)
        & torch.isfinite(target_disparity_hr_px)
        & torch.isfinite(log_variance)
        & (target_disparity_hr_px > 0)
    )
    if valid_mask is not None:
        if valid_mask.shape != usable.shape:
            raise ValueError("valid_mask shape must match disparity")
        usable &= valid_mask.to(dtype=torch.bool)
    safe_log_variance = torch.where(
        usable,
        log_variance.clamp(min=minimum_log_variance, max=maximum_log_variance),
        torch.zeros_like(log_variance),
    )
    absolute_error = torch.where(
        usable,
        (prediction_disparity_hr_px - target_disparity_hr_px).abs(),
        torch.zeros_like(prediction_disparity_hr_px),
    )
    nll = torch.exp(-safe_log_variance) * absolute_error + safe_log_variance
    return finite_masked_mean(nll, usable)


def ffs_gate_regularizer(
    source_weights: Tensor,
    confidence_ffs: Tensor,
    valid_ffs_mask: Tensor,
    *,
    trusted_confidence_threshold: float = 0.8,
) -> Tensor:
    """Encourage FFS ownership only where FFS itself is trusted."""

    if source_weights.ndim != 4 or source_weights.shape[1] != 3:
        raise ValueError("source_weights must have shape [B,3,H,W]")
    expected_shape = (source_weights.shape[0], 1, *source_weights.shape[-2:])
    if confidence_ffs.shape != expected_shape or valid_ffs_mask.shape != expected_shape:
        raise ValueError(f"confidence and valid mask must have shape {expected_shape}")
    if not 0.0 <= trusted_confidence_threshold <= 1.0:
        raise ValueError("trusted_confidence_threshold must lie in [0,1]")
    trusted = (
        valid_ffs_mask.to(dtype=torch.bool)
        & torch.isfinite(confidence_ffs)
        & (confidence_ffs > trusted_confidence_threshold)
    )
    ownership_error = 1.0 - source_weights[:, :1]
    return finite_masked_mean(ownership_error, trusted, confidence_ffs)


__all__ = ["ffs_gate_regularizer", "laplace_uncertainty_nll"]
