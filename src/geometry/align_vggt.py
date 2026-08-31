"""Robustly align a metric-scaled VGGT depth prior to FFS disparity.

FFS is the metric disparity owner.  VGGT depth is used only as a prior and is
converted to inverse depth before fitting one positive, zero-intercept scale::

    disparity_ffs_hr_px ~= scale_px_m * (1 / depth_vggt_m)

The default model deliberately has no additive shift.  A weighted-median
initialization followed by Huber IRLS limits the influence of bad VGGT depth or
stereo mismatches.  If too few trustworthy FFS pixels remain, the returned
prior is explicitly invalid (all NaN with a false mask).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

try:  # Torch is installed in runtime environments, but optional for tooling.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ScaleOnlyEstimate:
    """Scalar robust-fit result for ``target ~= scale * predictor``."""

    scale: float
    valid: bool
    sample_count: int
    iterations: int
    converged: bool
    weighted_mean_absolute_residual: float
    failure_reason: str | None


@dataclass(frozen=True)
class VGGTAlignmentResult:
    """VGGT depth expressed as FFS-owned HR-pixel disparity.

    Attributes:
        disparity_vggt_aligned_hr_px: Dense aligned prior, same shape/backend
            as ``depth_vggt_m``.  Invalid pixels and invalid global fits are
            NaN.
        valid_mask: Where the returned dense prior is finite and positive.
        reliable_ffs_mask: Pixels actually used to estimate the scale.
        scale_px_m: Positive scale multiplying inverse metric depth.  Its unit
            is HR-pixels times metres.
        valid: Whether a global alignment was successfully estimated.
    """

    disparity_vggt_aligned_hr_px: Any
    valid_mask: Any
    reliable_ffs_mask: Any
    scale_px_m: float
    valid: bool
    reliable_pixel_count: int
    iterations: int
    converged: bool
    weighted_mean_absolute_residual_hr_px: float
    failure_reason: str | None

    @property
    def aligned_disparity_hr_px(self) -> Any:
        """Short compatibility alias for the aligned disparity prior."""

        return self.disparity_vggt_aligned_hr_px

    @property
    def scale(self) -> float:
        return self.scale_px_m


# Earlier design documents used this shorter dataclass name.
VGGTAlignment = VGGTAlignmentResult


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _as_floating(value: Any, name: str) -> Any:
    if _is_torch(value):
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued")
        if value.is_floating_point():
            return value
        return value.to(dtype=torch.float64)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued numeric data")
    return array.astype(
        array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64,
        copy=False,
    )


def _as_backend_value(value: Any, reference: Any, name: str) -> Any:
    if _is_torch(reference):
        if not _is_torch(value):
            raise TypeError(f"{name} must be a Torch tensor when disparity is Torch")
        return value.to(device=reference.device)
    if _is_torch(value):
        raise TypeError(f"{name} must be NumPy-compatible when disparity is NumPy")
    return np.asarray(value)


def _broadcast_to_shape(value: Any, shape: tuple[int, ...], name: str) -> Any:
    try:
        if _is_torch(value):
            return torch.broadcast_to(value, shape)
        return np.broadcast_to(value, shape)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} is not broadcastable to {shape}"
        ) from exc


def _finite_positive(value: Any) -> Any:
    if _is_torch(value):
        return torch.isfinite(value) & (value > 0)
    return np.isfinite(value) & (value > 0)


def _full_nan_like(value: Any) -> Any:
    if _is_torch(value):
        return torch.full_like(value, torch.nan)
    return np.full_like(value, np.nan)


def _zeros_bool_like(value: Any) -> Any:
    if _is_torch(value):
        return torch.zeros_like(value, dtype=torch.bool)
    return np.zeros_like(value, dtype=bool)


def _count_true(mask: Any) -> int:
    if _is_torch(mask):
        return int(mask.sum().detach().cpu().item())
    return int(np.count_nonzero(mask))


def _to_float(value: Any) -> float:
    if _is_torch(value):
        return float(value.detach().cpu().item())
    return float(np.asarray(value).item())


def _validate_positive_scalar(value: Real, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _validate_nonnegative_scalar(value: Real, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return number


def _weighted_median(values: Any, weights: Any) -> Any:
    """Return a deterministic weighted median for positive finite weights."""

    if _is_torch(values):
        order = torch.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = torch.cumsum(sorted_weights, dim=0)
        cutoff = sorted_weights.sum() * 0.5
        index = torch.searchsorted(cumulative, cutoff, right=False)
        index = index.clamp(max=sorted_values.numel() - 1)
        return sorted_values[index]

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, sorted_weights.sum() * 0.5, side="left"))
    return sorted_values[min(index, sorted_values.size - 1)]


def ffs_trusted_mask(
    disparity_ffs_hr_px: Any,
    confidence_ffs: Any,
    left_right_error_px: Any,
    *,
    confidence_threshold: float = 0.8,
    max_left_right_error_px: float = 1.0,
) -> Any:
    """Build the reliable FFS set used for VGGT scale-only alignment.

    All inputs are broadcast to ``disparity_ffs_hr_px.shape``.  Disparity and
    left-right error must already use the pixel unit named by the caller; for
    this project that is normally HR pixels.  The exact MVP inequalities are
    ``confidence > 0.8``, ``LR error < 1 px``, and ``disparity > 0``.
    Non-finite values are always excluded.
    """

    confidence_threshold = _validate_nonnegative_scalar(
        confidence_threshold, "confidence_threshold"
    )
    if confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be <= 1")
    max_left_right_error_px = _validate_positive_scalar(
        max_left_right_error_px, "max_left_right_error_px"
    )
    disparity = _as_floating(disparity_ffs_hr_px, "disparity_ffs_hr_px")
    confidence = _as_backend_value(confidence_ffs, disparity, "confidence_ffs")
    lr_error = _as_backend_value(left_right_error_px, disparity, "left_right_error_px")
    confidence = _broadcast_to_shape(confidence, tuple(disparity.shape), "confidence_ffs")
    lr_error = _broadcast_to_shape(lr_error, tuple(disparity.shape), "left_right_error_px")

    if _is_torch(disparity):
        return (
            torch.isfinite(disparity)
            & (disparity > 0)
            & torch.isfinite(confidence)
            & (confidence > confidence_threshold)
            & torch.isfinite(lr_error)
            & (lr_error < max_left_right_error_px)
        )
    return (
        np.isfinite(disparity)
        & (disparity > 0)
        & np.isfinite(confidence)
        & (confidence > confidence_threshold)
        & np.isfinite(lr_error)
        & (lr_error < max_left_right_error_px)
    )


def robust_scale_only_irls(
    predictor_per_m: Any,
    target_hr_px: Any,
    *,
    weights: Any | None = None,
    valid_mask: Any | None = None,
    min_samples: int = 32,
    huber_delta_hr_px: float = 1.0,
    max_iterations: int = 20,
    relative_tolerance: float = 1e-6,
    minimum_predictor: float = 1e-12,
) -> ScaleOnlyEstimate:
    """Fit a positive zero-intercept scale with weighted Huber IRLS.

    Args:
        predictor_per_m: Inverse metric depth ``1 / Z`` in ``m^-1``.
        target_hr_px: Trusted FFS disparity in HR pixels.
        weights: Optional non-negative dimensionless confidence weights.
        valid_mask: Optional caller reliability mask.
        min_samples: Minimum finite, positive, non-zero-weight pixels.
        huber_delta_hr_px: Huber transition in target disparity pixels.

    Returns:
        A scalar estimate whose ``scale`` has units ``HR-px * m``.  Invalid
        inputs return a non-throwing ``valid=False`` result with ``scale=NaN``.
        Shape/type/configuration mistakes still raise clear exceptions.
    """

    if isinstance(min_samples, bool) or not isinstance(min_samples, int):
        raise TypeError("min_samples must be an integer")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    huber_delta = _validate_positive_scalar(huber_delta_hr_px, "huber_delta_hr_px")
    tolerance = _validate_nonnegative_scalar(relative_tolerance, "relative_tolerance")
    predictor_floor = _validate_positive_scalar(minimum_predictor, "minimum_predictor")

    predictor = _as_floating(predictor_per_m, "predictor_per_m")
    target = _as_backend_value(target_hr_px, predictor, "target_hr_px")
    target = _as_floating(target, "target_hr_px")
    try:
        if _is_torch(predictor):
            predictor, target = torch.broadcast_tensors(predictor, target)
        else:
            predictor, target = np.broadcast_arrays(predictor, target)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "predictor_per_m and target_hr_px must be broadcastable, got "
            f"{tuple(predictor.shape)} and {tuple(target.shape)}"
        ) from exc

    shape = tuple(predictor.shape)
    if weights is None:
        if _is_torch(predictor):
            fit_weights = torch.ones_like(predictor)
        else:
            fit_weights = np.ones_like(predictor)
    else:
        fit_weights = _as_backend_value(weights, predictor, "weights")
        fit_weights = _as_floating(fit_weights, "weights")
        fit_weights = _broadcast_to_shape(fit_weights, shape, "weights")

    if valid_mask is None:
        if _is_torch(predictor):
            caller_valid = torch.ones_like(predictor, dtype=torch.bool)
        else:
            caller_valid = np.ones_like(predictor, dtype=bool)
    else:
        caller_valid = _as_backend_value(valid_mask, predictor, "valid_mask")
        caller_valid = _broadcast_to_shape(caller_valid, shape, "valid_mask")
        if _is_torch(caller_valid):
            caller_valid = caller_valid.to(dtype=torch.bool)
        else:
            caller_valid = caller_valid.astype(bool, copy=False)

    if _is_torch(predictor):
        finite = (
            torch.isfinite(predictor)
            & torch.isfinite(target)
            & torch.isfinite(fit_weights)
        )
    else:
        finite = np.isfinite(predictor) & np.isfinite(target) & np.isfinite(fit_weights)
    usable = (
        caller_valid
        & finite
        & (predictor > predictor_floor)
        & (target > 0)
        & (fit_weights > 0)
    )
    sample_count = _count_true(usable)
    if sample_count < min_samples:
        return ScaleOnlyEstimate(
            scale=math.nan,
            valid=False,
            sample_count=sample_count,
            iterations=0,
            converged=False,
            weighted_mean_absolute_residual=math.nan,
            failure_reason="insufficient_reliable_pixels",
        )

    x = predictor[usable]
    y = target[usable]
    w = fit_weights[usable]

    # Compute in at least float32 on Torch (float64 on CPU NumPy) so cached
    # float16 inputs do not destabilize the scalar fit.
    if _is_torch(x):
        working_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x = x.to(dtype=working_dtype)
        y = y.to(dtype=working_dtype)
        w = w.to(dtype=working_dtype)
    else:
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        w = w.astype(np.float64, copy=False)

    scale_value = _weighted_median(y / x, w)
    scale_float = _to_float(scale_value)
    if not math.isfinite(scale_float) or scale_float <= 0:
        return ScaleOnlyEstimate(
            scale=math.nan,
            valid=False,
            sample_count=sample_count,
            iterations=0,
            converged=False,
            weighted_mean_absolute_residual=math.nan,
            failure_reason="non_positive_initial_scale",
        )

    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        residual = y - scale_value * x
        absolute_residual = residual.abs() if _is_torch(residual) else np.abs(residual)
        if _is_torch(residual):
            robust_weight = torch.where(
                absolute_residual <= huber_delta,
                torch.ones_like(absolute_residual),
                huber_delta / absolute_residual.clamp_min(torch.finfo(residual.dtype).tiny),
            )
            effective_weight = w * robust_weight
            denominator = (effective_weight * x.square()).sum()
            numerator = (effective_weight * x * y).sum()
        else:
            robust_weight = np.ones_like(absolute_residual)
            outside = absolute_residual > huber_delta
            robust_weight[outside] = huber_delta / absolute_residual[outside]
            effective_weight = w * robust_weight
            denominator = np.sum(effective_weight * np.square(x))
            numerator = np.sum(effective_weight * x * y)

        denominator_float = _to_float(denominator)
        if not math.isfinite(denominator_float) or denominator_float <= 0:
            return ScaleOnlyEstimate(
                scale=math.nan,
                valid=False,
                sample_count=sample_count,
                iterations=iteration,
                converged=False,
                weighted_mean_absolute_residual=math.nan,
                failure_reason="degenerate_weighted_system",
            )
        updated_scale = numerator / denominator
        updated_float = _to_float(updated_scale)
        if not math.isfinite(updated_float) or updated_float <= 0:
            return ScaleOnlyEstimate(
                scale=math.nan,
                valid=False,
                sample_count=sample_count,
                iterations=iteration,
                converged=False,
                weighted_mean_absolute_residual=math.nan,
                failure_reason="non_positive_fitted_scale",
            )
        change = abs(updated_float - scale_float)
        scale_value = updated_scale
        scale_float = updated_float
        iterations = iteration
        if change <= tolerance * max(abs(scale_float), 1.0):
            converged = True
            break

    final_residual = y - scale_value * x
    if _is_torch(final_residual):
        weighted_mae = (w * final_residual.abs()).sum() / w.sum()
    else:
        weighted_mae = np.sum(w * np.abs(final_residual)) / np.sum(w)
    weighted_mae_float = _to_float(weighted_mae)
    if not math.isfinite(weighted_mae_float):
        return ScaleOnlyEstimate(
            scale=math.nan,
            valid=False,
            sample_count=sample_count,
            iterations=iterations,
            converged=converged,
            weighted_mean_absolute_residual=math.nan,
            failure_reason="non_finite_residual",
        )
    return ScaleOnlyEstimate(
        scale=scale_float,
        valid=True,
        sample_count=sample_count,
        iterations=iterations,
        converged=converged,
        weighted_mean_absolute_residual=weighted_mae_float,
        failure_reason=None,
    )


def align_vggt_depth_to_ffs_disparity(
    disparity_ffs_hr_px: Any,
    depth_vggt_m: Any,
    *,
    reliable_ffs_mask: Any | None = None,
    weights: Any | None = None,
    min_reliable_pixels: int = 32,
    huber_delta_hr_px: float = 1.0,
    max_iterations: int = 20,
    relative_tolerance: float = 1e-6,
    epsilon_m: float = 1e-8,
) -> VGGTAlignmentResult:
    """Align metric VGGT depth to trusted FFS HR-pixel disparity.

    The fitted model is exactly ``d_hr_px = scale_px_m / depth_m``.  There is
    no additive shift.  Alignment estimates the scalar only on
    ``reliable_ffs_mask`` but emits a prior over every positive finite VGGT
    depth pixel after a successful fit.
    """

    epsilon = _validate_positive_scalar(epsilon_m, "epsilon_m")
    disparity = _as_floating(disparity_ffs_hr_px, "disparity_ffs_hr_px")
    depth = _as_backend_value(depth_vggt_m, disparity, "depth_vggt_m")
    depth = _as_floating(depth, "depth_vggt_m")
    if tuple(depth.shape) != tuple(disparity.shape):
        raise ValueError(
            "disparity_ffs_hr_px and depth_vggt_m must have identical shapes, got "
            f"{tuple(disparity.shape)} and {tuple(depth.shape)}"
        )

    if reliable_ffs_mask is None:
        reliable = _finite_positive(disparity)
    else:
        reliable = _as_backend_value(
            reliable_ffs_mask, disparity, "reliable_ffs_mask"
        )
        reliable = _broadcast_to_shape(
            reliable, tuple(disparity.shape), "reliable_ffs_mask"
        )
        if _is_torch(reliable):
            reliable = reliable.to(dtype=torch.bool) & _finite_positive(disparity)
        else:
            reliable = reliable.astype(bool, copy=False) & _finite_positive(disparity)

    depth_valid = _finite_positive(depth)
    reliable = reliable & depth_valid
    if _is_torch(depth):
        inverse_depth_per_m = torch.where(
            depth_valid,
            depth.clamp_min(epsilon).reciprocal(),
            torch.zeros_like(depth),
        )
    else:
        inverse_depth_per_m = np.zeros_like(depth)
        np.divide(1.0, np.maximum(depth, epsilon), out=inverse_depth_per_m, where=depth_valid)

    estimate = robust_scale_only_irls(
        inverse_depth_per_m,
        disparity,
        weights=weights,
        valid_mask=reliable,
        min_samples=min_reliable_pixels,
        huber_delta_hr_px=huber_delta_hr_px,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
    )
    if not estimate.valid:
        return VGGTAlignmentResult(
            disparity_vggt_aligned_hr_px=_full_nan_like(depth),
            valid_mask=_zeros_bool_like(depth),
            reliable_ffs_mask=reliable,
            scale_px_m=math.nan,
            valid=False,
            reliable_pixel_count=estimate.sample_count,
            iterations=estimate.iterations,
            converged=estimate.converged,
            weighted_mean_absolute_residual_hr_px=math.nan,
            failure_reason=estimate.failure_reason,
        )

    aligned = _full_nan_like(depth)
    if _is_torch(depth):
        aligned[depth_valid] = estimate.scale / depth[depth_valid].clamp_min(epsilon)
        aligned_valid = depth_valid & torch.isfinite(aligned) & (aligned > 0)
    else:
        np.divide(
            estimate.scale,
            np.maximum(depth, epsilon),
            out=aligned,
            where=depth_valid,
        )
        aligned_valid = depth_valid & np.isfinite(aligned) & (aligned > 0)
    return VGGTAlignmentResult(
        disparity_vggt_aligned_hr_px=aligned,
        valid_mask=aligned_valid,
        reliable_ffs_mask=reliable,
        scale_px_m=estimate.scale,
        valid=True,
        reliable_pixel_count=estimate.sample_count,
        iterations=estimate.iterations,
        converged=estimate.converged,
        weighted_mean_absolute_residual_hr_px=(
            estimate.weighted_mean_absolute_residual
        ),
        failure_reason=None,
    )


# Readable aliases for adapter/integration call sites.
align_vggt_depth_prior = align_vggt_depth_to_ffs_disparity
align_vggt_to_ffs = align_vggt_depth_to_ffs_disparity
build_ffs_trusted_mask = ffs_trusted_mask


__all__ = [
    "ScaleOnlyEstimate",
    "VGGTAlignment",
    "VGGTAlignmentResult",
    "align_vggt_depth_prior",
    "align_vggt_depth_to_ffs_disparity",
    "align_vggt_to_ffs",
    "build_ffs_trusted_mask",
    "ffs_trusted_mask",
    "robust_scale_only_irls",
]
