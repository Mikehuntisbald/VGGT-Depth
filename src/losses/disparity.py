"""Robust spatial disparity losses with explicit HR-pixel units."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _require_same_shape(reference: Tensor, other: Tensor, name: str) -> None:
    if not isinstance(other, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if other.shape != reference.shape:
        raise ValueError(
            f"{name} shape must match {tuple(reference.shape)}, got {tuple(other.shape)}"
        )


def charbonnier(error: Tensor, epsilon: float = 1e-3) -> Tensor:
    """Return a smooth-L1 penalty, zero at zero error."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return torch.sqrt(error.square() + epsilon * epsilon) - epsilon


def finite_masked_mean(
    values: Tensor,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Average finite values without allowing masked NaNs to propagate.

    Empty masks return a differentiable scalar zero. ``weights`` must be
    finite and non-negative where used; non-finite or non-positive weights are
    excluded rather than multiplied by a masked NaN.
    """

    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise TypeError("values must be a floating-point torch.Tensor")
    usable = torch.isfinite(values)
    if valid_mask is not None:
        _require_same_shape(values, valid_mask, "valid_mask")
        usable &= valid_mask.to(dtype=torch.bool)
    if weights is None:
        safe_values = torch.where(usable, values, torch.zeros_like(values))
        count = usable.sum()
        return torch.where(
            count > 0,
            safe_values.sum() / count.clamp_min(1).to(values.dtype),
            safe_values.sum(),
        )

    _require_same_shape(values, weights, "weights")
    usable &= torch.isfinite(weights) & (weights > 0)
    safe_weights = torch.where(usable, weights, torch.zeros_like(weights)).to(
        values.dtype
    )
    safe_values = torch.where(usable, values, torch.zeros_like(values))
    denominator = safe_weights.sum()
    return torch.where(
        denominator > 0,
        (safe_values * safe_weights).sum() / denominator.clamp_min(
            torch.finfo(values.dtype).tiny
        ),
        safe_values.sum(),
    )


def disparity_loss(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust supervised disparity loss for equal-shaped HR-pixel tensors."""

    _require_same_shape(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        "target_disparity_hr_px",
    )
    usable = (
        torch.isfinite(prediction_disparity_hr_px)
        & torch.isfinite(target_disparity_hr_px)
        & (target_disparity_hr_px > 0)
    )
    if valid_mask is not None:
        _require_same_shape(prediction_disparity_hr_px, valid_mask, "valid_mask")
        usable &= valid_mask.to(dtype=torch.bool)
    error = torch.where(
        usable,
        prediction_disparity_hr_px - target_disparity_hr_px,
        torch.zeros_like(prediction_disparity_hr_px),
    )
    return finite_masked_mean(charbonnier(error, epsilon), usable, weights)


def gradient_loss(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Match horizontal and vertical HR disparity gradients at valid edges."""

    _require_same_shape(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        "target_disparity_hr_px",
    )
    if prediction_disparity_hr_px.ndim != 4:
        raise ValueError("disparity tensors must have shape [B,C,H,W]")
    finite = torch.isfinite(prediction_disparity_hr_px) & torch.isfinite(
        target_disparity_hr_px
    )
    if valid_mask is not None:
        _require_same_shape(prediction_disparity_hr_px, valid_mask, "valid_mask")
        finite &= valid_mask.to(dtype=torch.bool)

    prediction_safe = torch.where(
        finite, prediction_disparity_hr_px, torch.zeros_like(prediction_disparity_hr_px)
    )
    target_safe = torch.where(
        finite, target_disparity_hr_px, torch.zeros_like(target_disparity_hr_px)
    )
    pred_dx = prediction_safe[..., :, 1:] - prediction_safe[..., :, :-1]
    target_dx = target_safe[..., :, 1:] - target_safe[..., :, :-1]
    mask_dx = finite[..., :, 1:] & finite[..., :, :-1]
    pred_dy = prediction_safe[..., 1:, :] - prediction_safe[..., :-1, :]
    target_dy = target_safe[..., 1:, :] - target_safe[..., :-1, :]
    mask_dy = finite[..., 1:, :] & finite[..., :-1, :]
    return finite_masked_mean((pred_dx - target_dx).abs(), mask_dx) + finite_masked_mean(
        (pred_dy - target_dy).abs(), mask_dy
    )


def lower_bound_penalty(
    disparity_hr_px: Tensor,
    *,
    lower_bound_hr_px: float = 0.0,
) -> Tensor:
    """Penalize finite values below a physical disparity lower bound.

    This is a squared hinge, not an epsilon fill or a positive-valued output
    transform: values at and above the bound receive exactly zero penalty.
    Non-finite values are excluded so the result remains a finite,
    differentiable scalar even for diagnostic inputs.
    """

    if not isinstance(disparity_hr_px, Tensor) or not disparity_hr_px.is_floating_point():
        raise TypeError("disparity_hr_px must be a floating-point torch.Tensor")
    if not math.isfinite(lower_bound_hr_px) or lower_bound_hr_px < 0:
        raise ValueError("lower_bound_hr_px must be finite and non-negative")
    bound = disparity_hr_px.new_tensor(float(lower_bound_hr_px))
    finite = torch.isfinite(disparity_hr_px)
    violation = torch.where(
        finite,
        (bound - disparity_hr_px).clamp_min(0.0),
        torch.zeros_like(disparity_hr_px),
    )
    return finite_masked_mean(violation.square(), finite)


def epipolar_disparity_loss(
    prediction_disparity_hr_px: Tensor,
    epipolar_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust loss to a disparity selected by a local HR epipolar matcher."""

    return disparity_loss(
        prediction_disparity_hr_px,
        epipolar_disparity_hr_px,
        valid_mask=valid_mask,
        epsilon=epsilon,
    )


__all__ = [
    "charbonnier",
    "disparity_loss",
    "epipolar_disparity_loss",
    "finite_masked_mean",
    "gradient_loss",
    "lower_bound_penalty",
]
