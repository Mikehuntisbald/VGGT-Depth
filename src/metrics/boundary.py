"""Disparity-boundary masks and boundary-restricted EPE."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .disparity import MetricResult, end_point_error


def disparity_boundary_mask(
    target_disparity_hr_px: Tensor,
    *,
    gradient_threshold_px: float = 1.0,
    radius_px: int = 1,
) -> Tensor:
    """Derive a boolean boundary band from valid GT disparity gradients.

    The last two axes are interpreted as ``[H,W]`` and all leading dimensions
    are preserved. A horizontal/vertical neighbor pair is marked on both
    sides when its absolute HR-pixel disparity jump is at least
    ``gradient_threshold_px``. The result is then dilated by ``radius_px`` in
    Chebyshev distance. Edges touching invalid GT do not become boundaries.
    """

    if (
        not isinstance(target_disparity_hr_px, Tensor)
        or not target_disparity_hr_px.is_floating_point()
    ):
        raise TypeError(
            "target_disparity_hr_px must be a floating-point torch.Tensor"
        )
    if target_disparity_hr_px.ndim < 2:
        raise ValueError("target_disparity_hr_px must have at least [H,W] dimensions")
    if not math.isfinite(gradient_threshold_px) or gradient_threshold_px < 0:
        raise ValueError("gradient_threshold_px must be finite and >= 0")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("radius_px must be a non-negative integer")

    valid = torch.isfinite(target_disparity_hr_px) & (target_disparity_hr_px > 0)
    boundary = torch.zeros_like(valid)
    if target_disparity_hr_px.shape[-1] > 1:
        dx = (
            (
                target_disparity_hr_px[..., 1:]
                - target_disparity_hr_px[..., :-1]
            ).abs()
            >= gradient_threshold_px
        ) & valid[..., 1:] & valid[..., :-1]
        boundary[..., 1:] |= dx
        boundary[..., :-1] |= dx
    if target_disparity_hr_px.shape[-2] > 1:
        dy = (
            (
                target_disparity_hr_px[..., 1:, :]
                - target_disparity_hr_px[..., :-1, :]
            ).abs()
            >= gradient_threshold_px
        ) & valid[..., 1:, :] & valid[..., :-1, :]
        boundary[..., 1:, :] |= dy
        boundary[..., :-1, :] |= dy

    if radius_px == 0 or boundary.numel() == 0:
        return boundary
    height, width = boundary.shape[-2:]
    flattened = boundary.reshape(-1, 1, height, width).to(dtype=torch.float32)
    kernel_size = 2 * radius_px + 1
    dilated = F.max_pool2d(
        flattened,
        kernel_size=kernel_size,
        stride=1,
        padding=radius_px,
    ) > 0
    return dilated.reshape(boundary.shape)


def boundary_epe(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    gradient_threshold_px: float = 1.0,
    radius_px: int = 1,
) -> MetricResult:
    """Return HR-pixel EPE inside the GT-derived disparity boundary band."""

    boundary = disparity_boundary_mask(
        target_disparity_hr_px,
        gradient_threshold_px=gradient_threshold_px,
        radius_px=radius_px,
    )
    if valid_mask is not None:
        if not isinstance(valid_mask, Tensor) or valid_mask.shape != boundary.shape:
            raise ValueError(f"valid_mask must have shape {tuple(boundary.shape)}")
        boundary &= valid_mask.to(dtype=torch.bool)
    return end_point_error(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=boundary,
    )


__all__ = ["boundary_epe", "disparity_boundary_mask"]
