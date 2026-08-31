"""Metric depth/disparity conversions with explicit pixel units."""

from __future__ import annotations

from numbers import Real
from typing import Any, TypeVar

import numpy as np

try:  # Torch is supplied by runtime environments, not the base package.
    import torch
except ImportError:  # pragma: no cover - only in minimal installs.
    torch = None  # type: ignore[assignment]


ArrayValue = TypeVar("ArrayValue")


def _positive_finite_scalar(value: Real, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value_float


def _as_floating(value: Any, name: str) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value if value.is_floating_point() else value.to(dtype=torch.float32)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return array.astype(
        array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64,
        copy=False,
    )


def _invalid_filled_like(value: Any, invalid_value: float) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.full_like(value, invalid_value)
    return np.full_like(value, invalid_value)


def valid_disparity_mask(disparity_px: Any) -> Any:
    """Return where disparity is finite and strictly positive.

    The mask has the same shape and backend (NumPy or Torch) as the input.
    Empty tensors/arrays produce an empty boolean mask.
    """

    disparity_float_px = _as_floating(disparity_px, "disparity_px")
    if torch is not None and isinstance(disparity_float_px, torch.Tensor):
        return torch.isfinite(disparity_float_px) & (disparity_float_px > 0)
    return np.isfinite(disparity_float_px) & (disparity_float_px > 0)


def valid_depth_mask(depth_m: Any) -> Any:
    """Return where metric depth is finite and strictly positive."""

    depth_float_m = _as_floating(depth_m, "depth_m")
    if torch is not None and isinstance(depth_float_m, torch.Tensor):
        return torch.isfinite(depth_float_m) & (depth_float_m > 0)
    return np.isfinite(depth_float_m) & (depth_float_m > 0)


def disparity_lr_to_hr_units(disparity_lr_px: ArrayValue, scale: Real) -> ArrayValue:
    """Convert LR-pixel disparity values to HR-pixel disparity units.

    For an HR/LR ratio ``scale``, metric geometry requires
    ``disparity_hr_px = scale * disparity_lr_px``. Shape and device are
    preserved; integer inputs are promoted to floating point.
    """

    scale_float = _positive_finite_scalar(scale, "scale")
    disparity_float_lr_px = _as_floating(disparity_lr_px, "disparity_lr_px")
    return disparity_float_lr_px * scale_float  # type: ignore[return-value]


def disparity_hr_to_lr_units(disparity_hr_px: ArrayValue, scale: Real) -> ArrayValue:
    """Convert HR-pixel disparity values to LR-pixel disparity units."""

    scale_float = _positive_finite_scalar(scale, "scale")
    disparity_float_hr_px = _as_floating(disparity_hr_px, "disparity_hr_px")
    return disparity_float_hr_px / scale_float  # type: ignore[return-value]


def depth_from_disparity(
    disparity_px: ArrayValue,
    focal_length_px: Real,
    baseline_m: Real,
    *,
    invalid_value: float = float("nan"),
) -> ArrayValue:
    """Convert rectified-stereo disparity in pixels to metric depth.

    Computes ``depth_m = focal_length_px * baseline_m / disparity_px``.
    Non-finite or non-positive disparities are replaced by ``invalid_value``
    (NaN by default). Empty inputs are returned unchanged in shape.
    """

    focal_length_float_px = _positive_finite_scalar(
        focal_length_px, "focal_length_px"
    )
    baseline_float_m = _positive_finite_scalar(baseline_m, "baseline_m")
    if not np.isfinite(invalid_value) and not np.isnan(invalid_value):
        raise ValueError("invalid_value must be finite or NaN")
    disparity_float_px = _as_floating(disparity_px, "disparity_px")
    is_valid = valid_disparity_mask(disparity_float_px)
    depth_m = _invalid_filled_like(disparity_float_px, invalid_value)
    if torch is not None and isinstance(disparity_float_px, torch.Tensor):
        depth_m[is_valid] = (
            focal_length_float_px
            * baseline_float_m
            / disparity_float_px[is_valid]
        )
    else:
        np.divide(
            focal_length_float_px * baseline_float_m,
            disparity_float_px,
            out=depth_m,
            where=is_valid,
        )
    return depth_m  # type: ignore[return-value]


def disparity_from_depth(
    depth_m: ArrayValue,
    focal_length_px: Real,
    baseline_m: Real,
    *,
    invalid_value: float = float("nan"),
) -> ArrayValue:
    """Convert positive metric depth to rectified-stereo disparity pixels.

    Computes ``disparity_px = focal_length_px * baseline_m / depth_m``.
    Non-finite or non-positive depths are replaced by ``invalid_value`` (NaN
    by default).
    """

    focal_length_float_px = _positive_finite_scalar(
        focal_length_px, "focal_length_px"
    )
    baseline_float_m = _positive_finite_scalar(baseline_m, "baseline_m")
    if not np.isfinite(invalid_value) and not np.isnan(invalid_value):
        raise ValueError("invalid_value must be finite or NaN")
    depth_float_m = _as_floating(depth_m, "depth_m")
    is_valid = valid_depth_mask(depth_float_m)
    disparity_px = _invalid_filled_like(depth_float_m, invalid_value)
    if torch is not None and isinstance(depth_float_m, torch.Tensor):
        disparity_px[is_valid] = (
            focal_length_float_px * baseline_float_m / depth_float_m[is_valid]
        )
    else:
        np.divide(
            focal_length_float_px * baseline_float_m,
            depth_float_m,
            out=disparity_px,
            where=is_valid,
        )
    return disparity_px  # type: ignore[return-value]


lr_disparity_to_hr_pixels = disparity_lr_to_hr_units
hr_disparity_to_lr_pixels = disparity_hr_to_lr_units
