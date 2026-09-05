"""Losses for a causal metric stereo-video geometry model.

The canonical prediction in this module is inverse depth in ``m^-1``.  Metric
depth and rectified left disparity are never independent heads:

``depth_m = 1 / inverse_depth_m_inv``
``disparity_left_px = fx_px * baseline_m * inverse_depth_m_inv``

The public helpers accept ``[B,1,H,W]`` and ``[B,T,1,H,W]`` geometry tensors.
RGB tensors use the same layouts with three channels.  All reductions are
empty-mask safe and run in float32 so BF16 training does not underflow small
residuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .disparity import charbonnier, finite_masked_mean


TensorPyramid = Tensor | Sequence[Tensor]


def _require_geometry_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim not in (4, 5) or value.shape[-3] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,T,1,H,W]")


def _require_image_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim not in (4, 5) or value.shape[-3] not in (1, 3):
        raise ValueError(f"{name} must have shape [B,C,H,W] or [B,T,C,H,W]")


def _require_same_shape(reference: Tensor, value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the reference tensor device")


def _prediction_finiteness_guard(
    prediction: Tensor,
    name: str,
    domain: Tensor | None = None,
) -> Tensor:
    """Fail on non-finite predictions instead of rewarding them with zero loss.

    CPU callers receive an immediate named exception. On CUDA, multiplying the
    checked prediction by zero propagates NaN/Inf into the scalar loss without
    forcing a host synchronization; the trainer's non-finite loss/gradient
    guard can then abort the step. ``domain`` is reserved for sampled values
    whose invalidity outside an explicit projection mask is expected.
    """

    checked = prediction
    if domain is not None:
        if domain.shape != prediction.shape:
            raise ValueError(f"{name} finiteness domain must match prediction shape")
        checked = torch.where(domain, prediction, torch.zeros_like(prediction))
    if checked.device.type == "cpu" and not bool(torch.isfinite(checked).all().item()):
        raise ValueError(f"{name} contains non-finite predictions")
    return (checked.float() * 0.0).sum()


def _mask_like(reference: Tensor, mask: Tensor | None, name: str) -> Tensor:
    """Return a positive/true mask; non-finite soft mask values are invalid."""

    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    _require_same_shape(reference, mask, name)
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point():
        return mask != 0
    return torch.isfinite(mask) & (mask > 0.5)


def _not_mask_like(reference: Tensor, mask: Tensor | None, name: str) -> Tensor:
    """Return where a negative mask is explicitly false; NaN means unknown."""

    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    _require_same_shape(reference, mask, name)
    if mask.dtype == torch.bool:
        return ~mask
    return torch.isfinite(mask) & (mask <= 0.5)


def _weights_like(
    reference: Tensor, weights: Tensor | None, name: str
) -> Tensor | None:
    if weights is None:
        return None
    _require_same_shape(reference, weights, name)
    # Geometry/validity heads own confidence calibration. Allowing gradients
    # through a residual weight lets the model minimize loss by declaring hard
    # pixels uncertain instead of improving geometry.
    return weights.detach().float()


def _weight_support(reference: Tensor, weights: Tensor | None, name: str) -> Tensor:
    if weights is None:
        return torch.ones_like(reference, dtype=torch.bool)
    _require_same_shape(reference, weights, name)
    return torch.isfinite(weights) & (weights > 0)


def _availability_like(
    reference: Tensor, available: Tensor | None, name: str
) -> Tensor:
    """Broadcast per-batch/per-frame annotation availability to a geometry map."""

    if available is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if not isinstance(available, Tensor):
        raise TypeError(f"{name} must be a Tensor or None")
    if available.device != reference.device:
        raise ValueError(f"{name} must share the geometry device")
    value = available
    leading_shape = reference.shape[:-3]
    if value.ndim <= len(leading_shape) and tuple(value.shape) == tuple(
        leading_shape[: value.ndim]
    ):
        value = value.reshape(*value.shape, *((1,) * (reference.ndim - value.ndim)))
    try:
        torch.broadcast_shapes(value.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} must be per batch/frame or broadcastable to {tuple(reference.shape)}"
        ) from error
    if value.dtype == torch.bool:
        usable = value
    elif value.is_floating_point():
        usable = torch.isfinite(value) & (value > 0.5)
    else:
        usable = value != 0
    return usable.expand_as(reference)


def _positive_metric_parameter(
    value: Real | Tensor,
    reference: Tensor,
    name: str,
) -> Tensor:
    """Convert scalar or leading-dimension calibration to a broadcast tensor."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar or tensor")
    if isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
        return reference.new_tensor(scalar, dtype=torch.float32)
    if not isinstance(value, Tensor) or value.is_complex():
        raise TypeError(f"{name} must be a real scalar or tensor")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the inverse-depth tensor device")
    parameter = value.float()
    if parameter.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    # Calibration is normally validated before transfer by the dataset. Keep
    # exact diagnostics for CPU callers without synchronizing every CUDA step;
    # CUDA non-finite values are still excluded by the downstream loss masks.
    if parameter.device.type == "cpu" and not bool(
        torch.isfinite(parameter).all().item()
    ):
        raise ValueError(f"{name} must contain finite values")
    if parameter.device.type == "cpu" and not bool((parameter > 0).all().item()):
        raise ValueError(f"{name} must be strictly positive")

    leading_shape = reference.shape[:-3]
    if parameter.ndim and parameter.ndim <= len(leading_shape):
        if tuple(parameter.shape) == tuple(leading_shape[: parameter.ndim]):
            parameter = parameter.reshape(
                *parameter.shape, *((1,) * (reference.ndim - parameter.ndim))
            )
    try:
        torch.broadcast_shapes(parameter.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} is not broadcastable to "
            f"{tuple(reference.shape)}"
        ) from error
    return parameter


def metric_factor_px_m(
    inverse_depth_m_inv: Tensor,
    fx_px: Real | Tensor,
    baseline_m: Real | Tensor,
) -> Tensor:
    """Return broadcastable ``fx_px * baseline_m`` in pixel-metres."""

    _require_geometry_tensor(inverse_depth_m_inv, "inverse_depth_m_inv")
    focal = _positive_metric_parameter(fx_px, inverse_depth_m_inv, "fx_px")
    baseline = _positive_metric_parameter(baseline_m, inverse_depth_m_inv, "baseline_m")
    return focal * baseline


def disparity_from_inverse_depth_m_inv(
    inverse_depth_m_inv: Tensor,
    fx_px: Real | Tensor,
    baseline_m: Real | Tensor,
) -> Tensor:
    """Derive rectified disparity in pixels from canonical inverse depth.

    ``fx_px`` must refer to the tensor's image resolution and ``baseline_m``
    must be metres.  Predictions are not clamped: invalid/non-positive values
    remain visible to masks and validity supervision.
    """

    factor = metric_factor_px_m(inverse_depth_m_inv, fx_px, baseline_m)
    return inverse_depth_m_inv.float() * factor


def depth_from_inverse_depth_m_inv(
    inverse_depth_m_inv: Tensor,
    *,
    invalid_value: float = float("nan"),
) -> Tensor:
    """Derive metric depth in metres, preserving invalidity explicitly."""

    _require_geometry_tensor(inverse_depth_m_inv, "inverse_depth_m_inv")
    if not math.isfinite(invalid_value) and not math.isnan(invalid_value):
        raise ValueError("invalid_value must be finite or NaN")
    value = inverse_depth_m_inv.float()
    usable = torch.isfinite(value) & (value > 0)
    safe = torch.where(usable, value, torch.ones_like(value))
    depth = safe.reciprocal()
    fill = torch.full_like(depth, float(invalid_value))
    return torch.where(usable, depth, fill)


def _resize_to(reference: Tensor, value: Tensor, name: str) -> Tensor:
    if value.ndim != reference.ndim or value.shape[:-3] != reference.shape[:-3]:
        raise ValueError(
            f"{name} leading dimensions must match {tuple(reference.shape[:-3])}"
        )
    if value.shape[-3] != reference.shape[-3]:
        raise ValueError(f"{name} channel count must match the reference")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the target tensor device")
    if value.shape[-2:] == reference.shape[-2:]:
        return value.float()
    flat = value.float().reshape(-1, value.shape[-3], *value.shape[-2:])
    resized = F.interpolate(
        flat,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(
        *reference.shape[:-3], reference.shape[-3], *reference.shape[-2:]
    )


def _as_pyramid(values: TensorPyramid, name: str) -> tuple[Tensor, ...]:
    if isinstance(values, Tensor):
        pyramid = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        pyramid = tuple(values)
    else:
        raise TypeError(f"{name} must be a Tensor or a non-empty Tensor sequence")
    if not pyramid:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(pyramid):
        _require_geometry_tensor(value, f"{name}[{index}]")
    return pyramid


def _normalized_scale_weights(
    count: int,
    scale_weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if scale_weights is None:
        # Inputs are coarse-to-fine.  The finest output owns most supervision.
        weights = tuple(0.5 ** (count - index - 1) for index in range(count))
    else:
        if len(scale_weights) != count:
            raise ValueError("scale_weights length must match the prediction pyramid")
        weights = tuple(float(value) for value in scale_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("scale_weights must be finite and non-negative")
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("at least one scale weight must be positive")
    return tuple(value / denominator for value in weights)


def robust_multiscale_disparity_loss(
    prediction_inverse_depths_m_inv: TensorPyramid,
    target_disparity_left_px: Tensor,
    *,
    fx_px: Real | Tensor,
    baseline_m: Real | Tensor,
    valid_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    scale_weights: Sequence[float] | None = None,
    epsilon_px: float = 1e-3,
) -> Tensor:
    """Supervise a coarse-to-fine inverse-depth pyramid in full-res pixels.

    Every inverse-depth map is first bilinearly resized to the target grid and
    then converted with the *target-grid* focal length.  This deliberately
    avoids the common bug of comparing low-resolution pixel disparities to a
    full-resolution target.  Scale weights are normalized to sum to one, so
    adding an auxiliary output cannot silently increase this loss's weight.
    """

    _require_geometry_tensor(target_disparity_left_px, "target_disparity_left_px")
    if not math.isfinite(epsilon_px) or epsilon_px <= 0:
        raise ValueError("epsilon_px must be finite and positive")
    usable = (
        torch.isfinite(target_disparity_left_px)
        & (target_disparity_left_px > 0)
        & _mask_like(target_disparity_left_px, valid_mask, "valid_mask")
    )
    weights = _weights_like(target_disparity_left_px, confidence, "confidence")
    pyramid = _as_pyramid(
        prediction_inverse_depths_m_inv, "prediction_inverse_depths_m_inv"
    )
    normalized_weights = _normalized_scale_weights(len(pyramid), scale_weights)

    safe_target = torch.where(
        torch.isfinite(target_disparity_left_px),
        target_disparity_left_px,
        torch.zeros_like(target_disparity_left_px),
    )
    total = safe_target.float().sum() * 0.0
    for coefficient, prediction in zip(normalized_weights, pyramid, strict=True):
        prediction_guard = _prediction_finiteness_guard(
            prediction, "prediction inverse depth"
        )
        prediction_resized = _resize_to(
            target_disparity_left_px, prediction, "prediction inverse depth"
        )
        prediction_disparity_px = disparity_from_inverse_depth_m_inv(
            prediction_resized, fx_px, baseline_m
        )
        scale_usable = usable & torch.isfinite(prediction_disparity_px)
        error = torch.where(
            scale_usable,
            prediction_disparity_px - target_disparity_left_px.float(),
            torch.zeros_like(prediction_disparity_px),
        )
        total = (
            total
            + prediction_guard
            + coefficient
            * _per_map_masked_mean(
                charbonnier(error, epsilon_px), scale_usable, weights
            )
        )
    return total


def _per_map_masked_mean(
    values: Tensor,
    mask: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Average each batch/time map first, then average non-empty maps."""

    usable = mask & torch.isfinite(values)
    if weights is None:
        safe_weights = usable.to(dtype=values.dtype)
    else:
        _require_same_shape(values, weights, "weights")
        usable &= torch.isfinite(weights) & (weights > 0)
        safe_weights = torch.where(usable, weights, torch.zeros_like(weights)).to(
            values.dtype
        )
    safe_values = torch.where(usable, values, torch.zeros_like(values))
    map_count = math.prod(values.shape[:-3])
    flat_values = safe_values.reshape(map_count, -1)
    flat_weights = safe_weights.reshape(map_count, -1)
    denominator = flat_weights.sum(dim=1)
    numerator = (flat_values * flat_weights).sum(dim=1)
    nonempty = denominator > 0
    means = torch.where(
        nonempty,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(numerator),
    )
    return finite_masked_mean(means, nonempty)


def normalized_log_depth_loss(
    prediction_inverse_depth_m_inv: Tensor,
    target_inverse_depth_m_inv: Tensor,
    *,
    valid_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust, dimensionless log-depth auxiliary loss.

    The residual is ``log(Z_pred / Z_target)``, equivalently
    ``log(rho_target / rho_pred)``.  No per-scene mean is removed because that
    would discard the stereo metric gauge.  Each non-empty frame contributes
    equally, rather than frames with denser ground truth dominating a clip.
    """

    _require_geometry_tensor(
        prediction_inverse_depth_m_inv, "prediction_inverse_depth_m_inv"
    )
    _require_same_shape(
        prediction_inverse_depth_m_inv,
        target_inverse_depth_m_inv,
        "target_inverse_depth_m_inv",
    )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    prediction = prediction_inverse_depth_m_inv.float()
    target = target_inverse_depth_m_inv.float()
    usable = (
        torch.isfinite(prediction)
        & (prediction > 0)
        & torch.isfinite(target)
        & (target > 0)
        & _mask_like(prediction, valid_mask, "valid_mask")
    )
    safe_prediction = torch.where(usable, prediction, torch.ones_like(prediction))
    safe_target = torch.where(usable, target, torch.ones_like(target))
    log_depth_error = safe_target.log() - safe_prediction.log()
    weights = _weights_like(prediction, confidence, "confidence")
    return _prediction_finiteness_guard(
        prediction_inverse_depth_m_inv, "prediction_inverse_depth_m_inv"
    ) + _per_map_masked_mean(charbonnier(log_depth_error, epsilon), usable, weights)


def robust_disparity_loss(
    prediction_disparity_px: Tensor,
    target_disparity_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    epsilon_px: float = 1e-3,
) -> Tensor:
    """Robust single-scale disparity supervision in one declared pixel unit."""

    _require_geometry_tensor(prediction_disparity_px, "prediction_disparity_px")
    _require_same_shape(
        prediction_disparity_px, target_disparity_px, "target_disparity_px"
    )
    if not math.isfinite(epsilon_px) or epsilon_px <= 0:
        raise ValueError("epsilon_px must be finite and positive")
    prediction = prediction_disparity_px.float()
    target = target_disparity_px.float()
    usable = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
        & (target > 0)
        & _mask_like(prediction_disparity_px, valid_mask, "valid_mask")
    )
    error = torch.where(usable, prediction - target, torch.zeros_like(prediction))
    weights = _weights_like(prediction_disparity_px, confidence, "confidence")
    return _prediction_finiteness_guard(
        prediction_disparity_px, "prediction_disparity_px"
    ) + _per_map_masked_mean(charbonnier(error, epsilon_px), usable, weights)


def robust_multiscale_pixel_disparity_loss(
    prediction_disparities_target_px: TensorPyramid,
    target_disparity_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    scale_weights: Sequence[float] | None = None,
    epsilon_px: float = 1e-3,
) -> Tensor:
    """Supervise a coarse-to-fine disparity sequence in target pixel units.

    A prediction may live on a lower-resolution *grid*, but every value must
    already be expressed in pixels of the target image. Spatial interpolation
    therefore does not rescale disparity values. This is the contract used by
    the trainable FFS wrapper after its explicit LR-to-HR unit conversion.
    """

    _require_geometry_tensor(target_disparity_px, "target_disparity_px")
    pyramid = _as_pyramid(
        prediction_disparities_target_px, "prediction_disparities_target_px"
    )
    normalized_weights = _normalized_scale_weights(len(pyramid), scale_weights)
    zero = _zero_from(target_disparity_px)
    total = zero
    for coefficient, prediction in zip(normalized_weights, pyramid, strict=True):
        resized = _resize_to(target_disparity_px, prediction, "predicted disparity")
        total = total + coefficient * robust_disparity_loss(
            resized,
            target_disparity_px,
            valid_mask=valid_mask,
            confidence=confidence,
            epsilon_px=epsilon_px,
        )
    return total


def visibility_aware_temporal_residual_loss(
    current_inverse_depth_m_inv: Tensor,
    warped_previous_inverse_depth_m_inv: Tensor,
    target_current_inverse_depth_m_inv: Tensor,
    warped_target_previous_inverse_depth_m_inv: Tensor,
    *,
    visibility_mask: Tensor,
    dynamic_mask: Tensor,
    occlusion_mask: Tensor,
    collision_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
    geometry_consistent_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Match temporal log-depth residuals only on safely transported surfaces.

    Comparing residuals, rather than forcing consecutive predictions equal,
    preserves real geometry changes caused by camera motion. Dynamic,
    occluded, and optionally colliding z-buffer samples are explicitly
    excluded to prevent temporal dragging.
    """

    _require_geometry_tensor(current_inverse_depth_m_inv, "current_inverse_depth_m_inv")
    for name, value in (
        ("warped_previous_inverse_depth_m_inv", warped_previous_inverse_depth_m_inv),
        ("target_current_inverse_depth_m_inv", target_current_inverse_depth_m_inv),
        (
            "warped_target_previous_inverse_depth_m_inv",
            warped_target_previous_inverse_depth_m_inv,
        ),
    ):
        _require_same_shape(current_inverse_depth_m_inv, value, name)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    tensors = tuple(
        value.float()
        for value in (
            current_inverse_depth_m_inv,
            warped_previous_inverse_depth_m_inv,
            target_current_inverse_depth_m_inv,
            warped_target_previous_inverse_depth_m_inv,
        )
    )
    declared_domain = (
        _mask_like(current_inverse_depth_m_inv, visibility_mask, "visibility_mask")
        & _not_mask_like(current_inverse_depth_m_inv, dynamic_mask, "dynamic_mask")
        & _not_mask_like(current_inverse_depth_m_inv, occlusion_mask, "occlusion_mask")
        & _not_mask_like(current_inverse_depth_m_inv, collision_mask, "collision_mask")
        & _mask_like(current_inverse_depth_m_inv, valid_mask, "valid_mask")
        & _mask_like(
            current_inverse_depth_m_inv,
            geometry_consistent_mask,
            "geometry_consistent_mask",
        )
    )
    usable = declared_domain.clone()
    for value in tensors:
        usable &= torch.isfinite(value) & (value > 0)
    safe = tuple(
        torch.where(usable, value, torch.ones_like(value)) for value in tensors
    )
    prediction_delta = safe[0].log() - safe[1].log()
    target_delta = safe[2].log() - safe[3].log()
    error = prediction_delta - target_delta
    weights = _weights_like(current_inverse_depth_m_inv, confidence, "confidence")
    guard = _prediction_finiteness_guard(
        current_inverse_depth_m_inv, "current_inverse_depth_m_inv"
    ) + _prediction_finiteness_guard(
        warped_previous_inverse_depth_m_inv,
        "warped_previous_inverse_depth_m_inv",
        declared_domain,
    )
    return guard + _per_map_masked_mean(charbonnier(error, epsilon), usable, weights)


def _pixel_mask(image: Tensor, mask: Tensor | None, name: str) -> Tensor:
    pixel_shape = (*image.shape[:-3], 1, *image.shape[-2:])
    reference = image.new_ones(pixel_shape)
    if mask is None:
        return torch.ones(pixel_shape, dtype=torch.bool, device=image.device)
    if not isinstance(mask, Tensor) or mask.shape not in (pixel_shape, image.shape):
        raise ValueError(
            f"{name} must have shape {pixel_shape} or {tuple(image.shape)}"
        )
    if mask.device != image.device:
        raise ValueError(f"{name} must share the image device")
    if mask.shape == image.shape:
        if mask.dtype == torch.bool:
            return mask.all(dim=-3, keepdim=True)
        return (torch.isfinite(mask) & (mask > 0.5)).all(dim=-3, keepdim=True)
    return _mask_like(reference, mask, name)


def _negative_pixel_mask(image: Tensor, mask: Tensor | None, name: str) -> Tensor:
    pixel_shape = (*image.shape[:-3], 1, *image.shape[-2:])
    reference = image.new_ones(pixel_shape)
    if mask is None:
        return torch.ones(pixel_shape, dtype=torch.bool, device=image.device)
    if not isinstance(mask, Tensor) or mask.shape not in (pixel_shape, image.shape):
        raise ValueError(
            f"{name} must have shape {pixel_shape} or {tuple(image.shape)}"
        )
    if mask.device != image.device:
        raise ValueError(f"{name} must share the image device")
    if mask.shape == image.shape:
        if mask.dtype == torch.bool:
            return (~mask).all(dim=-3, keepdim=True)
        return (torch.isfinite(mask) & (mask <= 0.5)).all(dim=-3, keepdim=True)
    return _not_mask_like(reference, mask, name)


def _ssim_dissimilarity(reference: Tensor, reprojected: Tensor) -> Tensor:
    leading = reference.shape[:-3]
    channels, height, width = reference.shape[-3:]
    reference_flat = reference.reshape(-1, channels, height, width)
    reprojected_flat = reprojected.reshape(-1, channels, height, width)

    mu_reference = F.avg_pool2d(reference_flat, 3, stride=1, padding=1)
    mu_reprojected = F.avg_pool2d(reprojected_flat, 3, stride=1, padding=1)
    sigma_reference = (
        F.avg_pool2d(reference_flat.square(), 3, 1, 1) - mu_reference.square()
    )
    sigma_reprojected = (
        F.avg_pool2d(reprojected_flat.square(), 3, 1, 1) - mu_reprojected.square()
    )
    covariance = (
        F.avg_pool2d(reference_flat * reprojected_flat, 3, 1, 1)
        - mu_reference * mu_reprojected
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_reference * mu_reprojected + c1) * (2 * covariance + c2)
    denominator = (mu_reference.square() + mu_reprojected.square() + c1) * (
        sigma_reference + sigma_reprojected + c2
    )
    ssim = numerator / denominator.clamp_min(torch.finfo(reference.dtype).tiny)
    dissimilarity = ((1.0 - ssim) * 0.5).clamp(0.0, 1.0)
    return dissimilarity.reshape(*leading, channels, height, width).mean(
        dim=-3, keepdim=True
    )


def photometric_reprojection_loss(
    reference_image: Tensor,
    reprojected_image: Tensor,
    *,
    valid_mask: Tensor | None = None,
    occlusion_mask: Tensor | None = None,
    dynamic_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    ssim_weight: float = 0.85,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust SSIM/photometric reprojection with explicit safety masks.

    ``occlusion_mask=True`` and ``dynamic_mask=True`` both mean *exclude*.
    Stereo callers normally omit ``dynamic_mask``; temporal callers should
    supply it.  Images are expected to share a value scale (normally [0, 1]).
    """

    _require_image_tensor(reference_image, "reference_image")
    _require_same_shape(reference_image, reprojected_image, "reprojected_image")
    if not 0.0 <= ssim_weight <= 1.0 or not math.isfinite(ssim_weight):
        raise ValueError("ssim_weight must lie in [0,1]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    reference = reference_image.float()
    reprojected = reprojected_image.float()
    declared_domain = (
        torch.isfinite(reference).all(dim=-3, keepdim=True)
        & _pixel_mask(reference_image, valid_mask, "valid_mask")
        & _negative_pixel_mask(reference_image, occlusion_mask, "occlusion_mask")
        & _negative_pixel_mask(reference_image, dynamic_mask, "dynamic_mask")
    )
    guard = _prediction_finiteness_guard(
        reprojected_image,
        "reprojected_image",
        declared_domain.expand_as(reprojected_image),
    )
    finite_pixel = torch.isfinite(reprojected).all(dim=-3, keepdim=True)
    usable = declared_domain & finite_pixel
    channel_usable = usable.expand_as(reference)
    safe_reference = torch.where(channel_usable, reference, torch.zeros_like(reference))
    # Making excluded pixels identical avoids injecting artificial SSIM edges.
    safe_reprojected = torch.where(channel_usable, reprojected, safe_reference.detach())
    photometric = charbonnier(safe_reprojected - safe_reference, epsilon).mean(
        dim=-3, keepdim=True
    )
    if ssim_weight > 0:
        ssim = _ssim_dissimilarity(safe_reference, safe_reprojected)
        per_pixel = (1.0 - ssim_weight) * photometric + ssim_weight * ssim
    else:
        per_pixel = photometric
    pixel_weights = None
    if confidence is not None:
        pixel_shape = usable.shape
        if confidence.shape == reference_image.shape:
            pixel_weights = confidence.detach().float().mean(dim=-3, keepdim=True)
        elif confidence.shape == pixel_shape:
            pixel_weights = confidence.detach().float()
        else:
            raise ValueError(
                f"confidence must have shape {pixel_shape} or {tuple(reference_image.shape)}"
            )
        if confidence.device != reference_image.device:
            raise ValueError("confidence must share the image device")
    return guard + _per_map_masked_mean(per_pixel, usable, pixel_weights)


def _photometric_has_support(
    reference_image: Tensor,
    reprojected_image: Tensor,
    *,
    valid_mask: Tensor | None,
    occlusion_mask: Tensor | None,
    dynamic_mask: Tensor | None,
    confidence: Tensor | None,
) -> Tensor:
    domain = (
        torch.isfinite(reference_image).all(dim=-3, keepdim=True)
        & torch.isfinite(reprojected_image).all(dim=-3, keepdim=True)
        & _pixel_mask(reference_image, valid_mask, "valid_mask")
        & _negative_pixel_mask(reference_image, occlusion_mask, "occlusion_mask")
        & _negative_pixel_mask(reference_image, dynamic_mask, "dynamic_mask")
    )
    if confidence is not None:
        if confidence.shape == reference_image.shape:
            confidence_support = torch.isfinite(confidence).all(
                dim=-3, keepdim=True
            ) & (confidence.mean(dim=-3, keepdim=True) > 0)
        elif confidence.shape == domain.shape:
            confidence_support = torch.isfinite(confidence) & (confidence > 0)
        else:
            raise ValueError(
                f"confidence must have shape {domain.shape} or "
                f"{tuple(reference_image.shape)}"
            )
        domain &= confidence_support
    return domain.any()


def stereo_reprojection_loss(*args: Any, **kwargs: Any) -> Tensor:
    """Named wrapper for left/right image reprojection."""

    if kwargs.get("dynamic_mask") is not None:
        raise ValueError("stereo reprojection does not accept a dynamic_mask")
    return photometric_reprojection_loss(*args, **kwargs)


def temporal_reprojection_loss(*args: Any, **kwargs: Any) -> Tensor:
    """Named wrapper for temporal image reprojection."""

    return photometric_reprojection_loss(*args, **kwargs)


def _sample_horizontal_correspondence(
    source_disparity_px: Tensor,
    reference_disparity_px: Tensor,
    *,
    source_offset_sign: float,
) -> tuple[Tensor, Tensor]:
    """Sample source disparity at ``x_source=x_reference+sign*d_ref``."""

    _require_geometry_tensor(reference_disparity_px, "reference_disparity_px")
    _require_same_shape(
        reference_disparity_px, source_disparity_px, "source_disparity_px"
    )
    if source_offset_sign not in (-1.0, 1.0):
        raise ValueError("source_offset_sign must be -1 or +1")

    reference = reference_disparity_px.float()
    source = source_disparity_px.float()
    leading = reference.shape[:-3]
    height, width = reference.shape[-2:]
    flat_reference = reference.reshape(-1, 1, height, width)
    flat_source = source.reshape(-1, 1, height, width)
    x = torch.arange(width, dtype=reference.dtype, device=reference.device).view(
        1, 1, width
    )
    y = torch.arange(height, dtype=reference.dtype, device=reference.device).view(
        1, height, 1
    )
    x_source = x + source_offset_sign * flat_reference[:, 0]
    y_source = y.expand(flat_reference.shape[0], height, width)
    coordinate_valid = (
        torch.isfinite(x_source)
        & (x_source >= 0)
        & (x_source <= width - 1)
        & torch.isfinite(flat_reference[:, 0])
        & (flat_reference[:, 0] > 0)
    )
    x_safe = torch.where(coordinate_valid, x_source, torch.zeros_like(x_source))
    x_grid = 2.0 * (x_safe + 0.5) / width - 1.0
    y_grid = 2.0 * (y_source + 0.5) / height - 1.0
    grid = torch.stack((x_grid, y_grid), dim=-1)
    finite_source = torch.isfinite(flat_source) & (flat_source > 0)
    sampled = F.grid_sample(
        torch.where(finite_source, flat_source, torch.zeros_like(flat_source)),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled_support = F.grid_sample(
        finite_source.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = (
        coordinate_valid.unsqueeze(1)
        & (sampled_support >= 1.0 - 1e-5)
        & torch.isfinite(sampled)
        & (sampled > 0)
    )
    return (
        sampled.reshape(*leading, 1, height, width),
        valid.reshape(*leading, 1, height, width),
    )


def warp_right_disparity_to_left(
    disparity_left_px: Tensor,
    disparity_right_px: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sample positive right disparity at ``x_right=x_left-d_left``."""

    return _sample_horizontal_correspondence(
        disparity_right_px, disparity_left_px, source_offset_sign=-1.0
    )


def left_right_consistency_loss(
    disparity_left_px: Tensor,
    disparity_right_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
    occlusion_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    epsilon_px: float = 1e-3,
) -> Tensor:
    """Robust left/right consistency for positive HR-pixel disparities."""

    if not math.isfinite(epsilon_px) or epsilon_px <= 0:
        raise ValueError("epsilon_px must be finite and positive")
    guard = _prediction_finiteness_guard(
        disparity_left_px, "disparity_left_px"
    ) + _prediction_finiteness_guard(disparity_right_px, "disparity_right_px")
    sampled_right, correspondence_valid = warp_right_disparity_to_left(
        disparity_left_px, disparity_right_px
    )
    left = disparity_left_px.float()
    usable = (
        correspondence_valid
        & torch.isfinite(left)
        & (left > 0)
        & _mask_like(disparity_left_px, valid_mask, "valid_mask")
        & _not_mask_like(disparity_left_px, occlusion_mask, "occlusion_mask")
    )
    error = torch.where(usable, left - sampled_right, torch.zeros_like(left))
    weights = _weights_like(disparity_left_px, confidence, "confidence")
    return guard + _per_map_masked_mean(charbonnier(error, epsilon_px), usable, weights)


def pose_residual_loss(
    predicted_pose_residual: Tensor,
    target_pose_residual: Tensor | None = None,
    *,
    valid_mask: Tensor | None = None,
    translation_normalizer_m: float = 0.10,
    rotation_normalizer_rad: float = 0.10,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust loss for ``[..., tx,ty,tz, rx,ry,rz]`` pose residuals.

    Translation and axis-angle rotation are divided by explicit physical
    normalizers before reduction, so radians and metres are never added raw.
    With no target, the function regularizes a residual correction toward zero.
    """

    if (
        not isinstance(predicted_pose_residual, Tensor)
        or not predicted_pose_residual.is_floating_point()
    ):
        raise TypeError("predicted_pose_residual must be a floating Tensor")
    if predicted_pose_residual.ndim < 2 or predicted_pose_residual.shape[-1] != 6:
        raise ValueError("predicted_pose_residual must have shape [...,6]")
    for value, name in (
        (translation_normalizer_m, "translation_normalizer_m"),
        (rotation_normalizer_rad, "rotation_normalizer_rad"),
        (epsilon, "epsilon"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if target_pose_residual is None:
        target = torch.zeros_like(predicted_pose_residual)
    else:
        _require_same_shape(
            predicted_pose_residual, target_pose_residual, "target_pose_residual"
        )
        target = target_pose_residual
    prediction = predicted_pose_residual.float()
    target_float = target.float()
    usable = torch.isfinite(prediction) & torch.isfinite(target_float)
    if valid_mask is not None:
        expected = prediction.shape[:-1]
        if valid_mask.shape not in (expected, (*expected, 1), prediction.shape):
            raise ValueError(
                f"valid_mask must have shape {expected}, {(*expected, 1)}, or {tuple(prediction.shape)}"
            )
        if valid_mask.device != prediction.device:
            raise ValueError("valid_mask must share the pose tensor device")
        expanded = valid_mask
        if valid_mask.shape == expected:
            expanded = valid_mask.unsqueeze(-1)
        if expanded.shape[-1] == 1:
            expanded = expanded.expand_as(prediction)
        if expanded.dtype == torch.bool:
            usable &= expanded
        else:
            usable &= torch.isfinite(expanded) & (expanded > 0.5)
    normalizer = prediction.new_tensor(
        [translation_normalizer_m] * 3 + [rotation_normalizer_rad] * 3
    )
    error = torch.where(
        usable, (prediction - target_float) / normalizer, torch.zeros_like(prediction)
    )
    return _prediction_finiteness_guard(
        predicted_pose_residual, "predicted_pose_residual"
    ) + finite_masked_mean(charbonnier(error, epsilon), usable)


def scale_alignment_loss(
    predicted_log_scale: Tensor,
    target_log_scale: Tensor,
    *,
    valid_mask: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Robust scale alignment in dimensionless log space.

    ``log_scale`` is the logarithm of the multiplicative factor applied to an
    unscaled inverse-depth prior.  Targets should be estimated from trusted
    stereo support outside this function.
    """

    if (
        not isinstance(predicted_log_scale, Tensor)
        or not predicted_log_scale.is_floating_point()
    ):
        raise TypeError("predicted_log_scale must be a floating Tensor")
    _require_same_shape(predicted_log_scale, target_log_scale, "target_log_scale")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    prediction = predicted_log_scale.float()
    target = target_log_scale.float()
    usable = torch.isfinite(prediction) & torch.isfinite(target)
    if valid_mask is not None:
        _require_same_shape(predicted_log_scale, valid_mask, "valid_mask")
        usable &= _mask_like(predicted_log_scale, valid_mask, "valid_mask")
    error = torch.where(usable, prediction - target, torch.zeros_like(prediction))
    return _prediction_finiteness_guard(
        predicted_log_scale, "predicted_log_scale"
    ) + finite_masked_mean(charbonnier(error, epsilon), usable)


def inverse_depth_scale_alignment_loss(
    unscaled_inverse_depth: Tensor,
    target_inverse_depth_m_inv: Tensor,
    predicted_log_scale: Tensor,
    *,
    valid_mask: Tensor | None = None,
    confidence: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """Align a scale-ambiguous inverse-depth prior to stereo metric geometry.

    ``exp(predicted_log_scale)`` multiplies ``unscaled_inverse_depth``.  The
    loss is evaluated as a dimensionless log ratio on trusted stereo support.
    ``predicted_log_scale`` may be per clip/frame (``[B]`` or ``[B,T]``), may
    include trailing singleton dimensions, or may be dense like the geometry.
    """

    _require_geometry_tensor(unscaled_inverse_depth, "unscaled_inverse_depth")
    _require_same_shape(
        unscaled_inverse_depth,
        target_inverse_depth_m_inv,
        "target_inverse_depth_m_inv",
    )
    if (
        not isinstance(predicted_log_scale, Tensor)
        or not predicted_log_scale.is_floating_point()
    ):
        raise TypeError("predicted_log_scale must be a floating Tensor")
    if predicted_log_scale.device != unscaled_inverse_depth.device:
        raise ValueError("predicted_log_scale must share the geometry device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    log_scale = predicted_log_scale.float()
    leading_shape = unscaled_inverse_depth.shape[:-3]
    if log_scale.ndim and log_scale.ndim <= len(leading_shape):
        if tuple(log_scale.shape) == tuple(leading_shape[: log_scale.ndim]):
            log_scale = log_scale.reshape(
                *log_scale.shape,
                *((1,) * (unscaled_inverse_depth.ndim - log_scale.ndim)),
            )
    try:
        torch.broadcast_shapes(log_scale.shape, unscaled_inverse_depth.shape)
    except RuntimeError as error:
        raise ValueError(
            "predicted_log_scale must be per batch/frame or broadcastable to "
            f"{tuple(unscaled_inverse_depth.shape)}"
        ) from error

    source = unscaled_inverse_depth.float()
    target = target_inverse_depth_m_inv.float()
    usable = (
        torch.isfinite(source)
        & (source > 0)
        & torch.isfinite(target)
        & (target > 0)
        & torch.isfinite(log_scale)
        & _mask_like(unscaled_inverse_depth, valid_mask, "valid_mask")
    )
    safe_source = torch.where(usable, source, torch.ones_like(source))
    safe_target = torch.where(usable, target, torch.ones_like(target))
    safe_log_scale = torch.where(
        usable, log_scale.expand_as(source), torch.zeros_like(source)
    )
    log_ratio = safe_source.log() + safe_log_scale - safe_target.log()
    pixel_weights = _weights_like(unscaled_inverse_depth, confidence, "confidence")
    guard = _prediction_finiteness_guard(
        unscaled_inverse_depth, "unscaled_inverse_depth"
    ) + _prediction_finiteness_guard(predicted_log_scale, "predicted_log_scale")
    return guard + _per_map_masked_mean(
        charbonnier(log_ratio, epsilon), usable, pixel_weights
    )


def laplace_inverse_depth_uncertainty_loss(
    prediction_inverse_depth_m_inv: Tensor,
    target_inverse_depth_m_inv: Tensor,
    log_variance: Tensor,
    *,
    valid_mask: Tensor | None = None,
    detach_geometry_residual: bool = True,
    minimum_log_variance: float = -8.0,
    maximum_log_variance: float = 8.0,
) -> Tensor:
    """Laplace NLL for dimensionless log-depth residual uncertainty.

    ``log_variance`` means exactly ``log(var(error))``.  Up to an additive
    constant the Laplace NLL is ``sqrt(2)|e|exp(-s/2) + s/2``.  By default
    ``e`` is detached: the robust disparity/log-depth terms own geometry,
    while this term calibrates only the uncertainty head and therefore does
    not count the same supervised residual twice.
    """

    _require_geometry_tensor(
        prediction_inverse_depth_m_inv, "prediction_inverse_depth_m_inv"
    )
    for name, value in (
        ("target_inverse_depth_m_inv", target_inverse_depth_m_inv),
        ("log_variance", log_variance),
    ):
        _require_same_shape(prediction_inverse_depth_m_inv, value, name)
    if not math.isfinite(minimum_log_variance) or not math.isfinite(
        maximum_log_variance
    ):
        raise ValueError("log-variance bounds must be finite")
    if minimum_log_variance >= maximum_log_variance:
        raise ValueError("minimum_log_variance must be less than maximum_log_variance")

    prediction = prediction_inverse_depth_m_inv.float()
    target = target_inverse_depth_m_inv.float()
    log_variance_float = log_variance.float()
    usable = (
        torch.isfinite(prediction)
        & (prediction > 0)
        & torch.isfinite(target)
        & (target > 0)
        & torch.isfinite(log_variance_float)
        & _mask_like(prediction_inverse_depth_m_inv, valid_mask, "valid_mask")
    )
    safe_prediction = torch.where(usable, prediction, torch.ones_like(prediction))
    safe_target = torch.where(usable, target, torch.ones_like(target))
    residual = (safe_target.log() - safe_prediction.log()).abs()
    if detach_geometry_residual:
        residual = residual.detach()
    safe_log_variance = torch.where(
        usable,
        log_variance_float.clamp(minimum_log_variance, maximum_log_variance),
        torch.zeros_like(log_variance_float),
    )
    nll = (
        math.sqrt(2.0) * residual * torch.exp(-0.5 * safe_log_variance)
        + 0.5 * safe_log_variance
    )
    guarded_prediction = (
        prediction_inverse_depth_m_inv.detach()
        if detach_geometry_residual
        else prediction_inverse_depth_m_inv
    )
    guard = _prediction_finiteness_guard(
        guarded_prediction, "prediction_inverse_depth_m_inv"
    ) + _prediction_finiteness_guard(log_variance, "log_variance")
    return guard + _per_map_masked_mean(nll, usable)


def validity_classification_loss(
    valid_logits: Tensor,
    target_valid: Tensor,
    *,
    supervision_mask: Tensor | None = None,
) -> Tensor:
    """Empty-safe BCE for physical validity, distinct from uncertainty."""

    _require_geometry_tensor(valid_logits, "valid_logits")
    _require_same_shape(valid_logits, target_valid, "target_valid")
    logits = valid_logits.float()
    if target_valid.dtype == torch.bool:
        target = target_valid.float()
        finite_target = torch.ones_like(target_valid)
    else:
        target = target_valid.float()
        finite_target = torch.isfinite(target)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    usable = (
        finite_target
        & torch.isfinite(logits)
        & _mask_like(valid_logits, supervision_mask, "supervision_mask")
    )
    safe_logits = torch.where(usable, logits, torch.zeros_like(logits))
    safe_target = torch.where(usable, target, torch.zeros_like(target))
    bce = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    return _prediction_finiteness_guard(
        valid_logits, "valid_logits"
    ) + _per_map_masked_mean(bce, usable)


@dataclass(frozen=True, slots=True)
class MetricStereoVideoLossWeights:
    """Conservative first-run coefficients for the joint objective."""

    disparity: float = 1.00
    depth: float = 0.15
    temporal: float = 0.10
    reprojection: float = 0.10
    left_right_consistency: float = 0.05
    pose_scale: float = 0.01
    uncertainty: float = 0.03
    validity: float = 0.05

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, field.name)) for field in fields(self))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("loss weights must be finite and non-negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class MetricStereoVideoLossBreakdown:
    """Weighted total, grouped objectives, and unweighted diagnostics."""

    total: Tensor
    disparity: Tensor
    depth: Tensor
    temporal: Tensor
    reprojection: Tensor
    left_right_consistency: Tensor
    pose_scale: Tensor
    uncertainty: Tensor
    validity: Tensor
    stereo_reprojection: Tensor
    temporal_reprojection: Tensor
    pose: Tensor
    scale: Tensor
    active_terms: tuple[str, ...]

    def detached_scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in (
                "total",
                "disparity",
                "depth",
                "temporal",
                "reprojection",
                "left_right_consistency",
                "pose_scale",
                "uncertainty",
                "validity",
                "stereo_reprojection",
                "temporal_reprojection",
                "pose",
                "scale",
            )
        }


def _activity_scalar(
    activity: Tensor | bool | None, reference: Tensor, name: str
) -> Tensor:
    if activity is None:
        return torch.ones_like(reference)
    if isinstance(activity, bool):
        return torch.ones_like(reference) if activity else torch.zeros_like(reference)
    if not isinstance(activity, Tensor) or activity.numel() != 1:
        raise ValueError(f"{name} must be a scalar Tensor, bool, or None")
    if activity.device != reference.device:
        raise ValueError(f"{name} must share the loss device")
    return activity.to(dtype=reference.dtype).reshape_as(reference)


def _mean_active(
    components: Sequence[tuple[Tensor | None, Tensor | bool | None]],
    zero: Tensor,
) -> Tensor:
    numerator = zero
    denominator = torch.zeros_like(zero)
    for index, (value, activity) in enumerate(components):
        if value is None:
            continue
        active = _activity_scalar(activity, zero, f"component_activity[{index}]")
        numerator = numerator + value * active
        denominator = denominator + active
    return numerator / denominator.clamp_min(1.0)


def combine_metric_stereo_video_losses(
    *,
    disparity: Tensor,
    depth: Tensor,
    temporal: Tensor,
    stereo_reprojection: Tensor | None = None,
    temporal_reprojection: Tensor | None = None,
    left_right_consistency: Tensor,
    pose: Tensor | None = None,
    scale: Tensor | None = None,
    uncertainty: Tensor,
    validity: Tensor,
    stereo_reprojection_active: Tensor | bool | None = None,
    temporal_reprojection_active: Tensor | bool | None = None,
    pose_active: Tensor | bool | None = None,
    scale_active: Tensor | bool | None = None,
    weights: MetricStereoVideoLossWeights = MetricStereoVideoLossWeights(),
    active_terms: Sequence[str] = (),
) -> MetricStereoVideoLossBreakdown:
    """Combine objectives without doubling grouped optional supervision."""

    required = {
        "disparity": disparity,
        "depth": depth,
        "temporal": temporal,
        "left_right_consistency": left_right_consistency,
        "uncertainty": uncertainty,
        "validity": validity,
    }
    optional = {
        "stereo_reprojection": stereo_reprojection,
        "temporal_reprojection": temporal_reprojection,
        "pose": pose,
        "scale": scale,
    }
    for name, value in (*required.items(), *optional.items()):
        if value is not None and (not isinstance(value, Tensor) or value.numel() != 1):
            raise ValueError(f"{name} must be a scalar tensor or None")
    reference = disparity
    zero = (
        torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
        * 0.0
    )
    reprojection = _mean_active(
        (
            (stereo_reprojection, stereo_reprojection_active),
            (temporal_reprojection, temporal_reprojection_active),
        ),
        zero,
    )
    pose_scale = _mean_active(((pose, pose_active), (scale, scale_active)), zero)
    grouped = {
        **required,
        "reprojection": reprojection,
        "pose_scale": pose_scale,
    }
    # Preserve useful term-level diagnostics in CPU tests without adding eight
    # host/device synchronizations to every distributed CUDA training step.
    if all(value.device.type == "cpu" for value in grouped.values()):
        bad = [
            name
            for name, value in grouped.items()
            if not bool(torch.isfinite(value.detach()).all().item())
        ]
        if bad:
            raise ValueError(f"non-finite loss terms: {bad}")
    total = sum(
        grouped[name] * float(getattr(weights, name))
        for name in (
            "disparity",
            "depth",
            "temporal",
            "reprojection",
            "left_right_consistency",
            "pose_scale",
            "uncertainty",
            "validity",
        )
    )
    return MetricStereoVideoLossBreakdown(
        total=total,
        disparity=disparity,
        depth=depth,
        temporal=temporal,
        reprojection=reprojection,
        left_right_consistency=left_right_consistency,
        pose_scale=pose_scale,
        uncertainty=uncertainty,
        validity=validity,
        stereo_reprojection=(
            stereo_reprojection if stereo_reprojection is not None else zero
        ),
        temporal_reprojection=(
            temporal_reprojection if temporal_reprojection is not None else zero
        ),
        pose=pose if pose is not None else zero,
        scale=scale if scale is not None else zero,
        active_terms=tuple(active_terms),
    )


def _required(mapping: Mapping[str, Any], key: str, owner: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise KeyError(f"{owner} must contain {key!r}") from error


def _zero_from(reference: Tensor) -> Tensor:
    safe = torch.where(
        torch.isfinite(reference), reference, torch.zeros_like(reference)
    )
    return safe.float().sum() * 0.0


def metric_stereo_video_loss(
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    geometry: Mapping[str, Any],
    masks: Mapping[str, Tensor] | None = None,
    *,
    weights: MetricStereoVideoLossWeights = MetricStereoVideoLossWeights(),
) -> MetricStereoVideoLossBreakdown:
    """Compute the complete joint loss from explicit named mappings.

    Required fields are:

    - ``predictions['inverse_depth_m_inv']``: canonical finest HR output.
    - ``targets['disparity_left_px']`` and ``targets['valid']``.
    - ``geometry['fx_px']`` and ``geometry['baseline_m']``.

    Optional fields activate their term only as a complete set:

    - A prediction ``inverse_depth_pyramid_m_inv`` is coarse-to-fine and does
      not include the canonical finest output.
    - Right-view supervision consumes prediction ``disparity_right_px`` (HR
      pixels), ``right_inverse_depth_m_inv``, or ``disparity_right_lr_px`` plus
      geometry ``stereo_lr_to_hr_scale``. Targets are ``disparity_right_px``
      and ``valid_right``.
    - Temporal residual uses geometry ``temporal_warped_inverse_depth_m_inv``
      and ``temporal_warped_target_inverse_depth_m_inv`` plus masks
      ``temporal_visibility``, ``temporal_dynamic`` (or dataset-native
      ``dynamic_mask_current``), and ``temporal_occlusion``.
    - Reprojection consumes already sampled images so camera warping and
      z-buffer ownership remain in the geometry layer.

    ``dynamic_mask_available=False`` excludes that sample from temporal
    supervision; missing annotations are never interpreted as static.
    """

    if not isinstance(predictions, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("predictions and targets must be mappings")
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping")
    mask_values: Mapping[str, Tensor] = {} if masks is None else masks
    if not isinstance(mask_values, Mapping):
        raise TypeError("masks must be a mapping or None")

    canonical = _required(predictions, "inverse_depth_m_inv", "predictions")
    _require_geometry_tensor(canonical, "predictions['inverse_depth_m_inv']")
    target_disparity = _required(targets, "disparity_left_px", "targets")
    _require_geometry_tensor(target_disparity, "targets['disparity_left_px']")
    if canonical.shape != target_disparity.shape:
        raise ValueError(
            "canonical inverse depth and target disparity must share HR shape"
        )
    target_valid = _required(targets, "valid", "targets")
    _require_same_shape(target_disparity, target_valid, "targets['valid']")
    fx_px = _required(geometry, "fx_px", "geometry")
    baseline_m = _required(geometry, "baseline_m", "geometry")
    factor = metric_factor_px_m(canonical, fx_px, baseline_m)
    target_inverse_depth = target_disparity.float() / factor
    valid = (
        _mask_like(target_disparity, target_valid, "targets['valid']")
        & torch.isfinite(target_disparity)
        & (target_disparity > 0)
        & torch.isfinite(factor)
        & (factor > 0)
    )
    zero = _zero_from(canonical)
    active: list[str] = ["disparity", "depth"]
    uses_dataset_dynamic_mask = (
        "temporal_dynamic" not in mask_values and "dynamic_mask_current" in mask_values
    )
    if uses_dataset_dynamic_mask and "dynamic_mask_available" not in mask_values:
        raise KeyError(
            "dataset-native dynamic_mask_current requires dynamic_mask_available"
        )
    dynamic_mask = mask_values.get(
        "temporal_dynamic", mask_values.get("dynamic_mask_current")
    )
    dynamic_available = _availability_like(
        canonical,
        mask_values.get("dynamic_mask_available"),
        "dynamic_mask_available",
    )

    auxiliary = predictions.get("inverse_depth_pyramid_m_inv", ())
    if auxiliary is None:
        auxiliary_pyramid = ()
    elif isinstance(auxiliary, Tensor):
        auxiliary_pyramid = (auxiliary,)
    else:
        auxiliary_pyramid = tuple(auxiliary)
    disparity = robust_multiscale_disparity_loss(
        (*auxiliary_pyramid, canonical),
        target_disparity,
        fx_px=fx_px,
        baseline_m=baseline_m,
        valid_mask=valid,
        confidence=targets.get("confidence"),
    )
    stereo_iteration_pyramid = predictions.get(
        "disparity_pyramid_left_hr_px",
        predictions.get("ffs_iteration_disparities_left_hr_px_lr_grid"),
    )
    if stereo_iteration_pyramid is not None:
        stereo_iteration_disparity = robust_multiscale_pixel_disparity_loss(
            stereo_iteration_pyramid,
            target_disparity,
            valid_mask=valid,
            confidence=targets.get("confidence"),
        )
        # This is another left-view prediction branch, not another objective.
        disparity = 0.5 * (disparity + stereo_iteration_disparity)
        active.append("stereo_iteration_disparity")
    depth = normalized_log_depth_loss(
        canonical,
        target_inverse_depth,
        valid_mask=valid,
        confidence=targets.get("confidence"),
    )

    # A redundant disparity alias is permitted for model diagnostics, but it
    # must obey the canonical metric conversion contract.
    if "disparity_left_px" in predictions:
        alias = predictions["disparity_left_px"]
        _require_same_shape(canonical, alias, "predictions['disparity_left_px']")
        disparity = disparity + _prediction_finiteness_guard(
            alias, "predictions['disparity_left_px']"
        )
        derived = disparity_from_inverse_depth_m_inv(canonical, fx_px, baseline_m)
        # Numerical alias validation is a CPU diagnostic. On CUDA the derived
        # canonical value is always used, avoiding a synchronization per step.
        if alias.device.type == "cpu":
            finite = torch.isfinite(alias) & torch.isfinite(derived)
            if bool(finite.any().item()) and not bool(
                torch.allclose(
                    alias[finite].float(), derived[finite], rtol=5e-3, atol=1e-3
                )
            ):
                raise ValueError(
                    "predictions['disparity_left_px'] violates "
                    "d=fx*baseline*inverse_depth"
                )

    temporal: Tensor = zero
    if "temporal_warped_inverse_depth_m_inv" in geometry:
        if dynamic_mask is None:
            raise KeyError(
                "masks must contain 'temporal_dynamic' or 'dynamic_mask_current'"
            )
        temporal_valid = (
            _mask_like(
                canonical,
                mask_values.get("temporal_valid", valid),
                "temporal_valid",
            )
            & dynamic_available
        )
        temporal = visibility_aware_temporal_residual_loss(
            canonical,
            geometry["temporal_warped_inverse_depth_m_inv"],
            target_inverse_depth,
            _required(
                geometry,
                "temporal_warped_target_inverse_depth_m_inv",
                "geometry",
            ),
            visibility_mask=_required(mask_values, "temporal_visibility", "masks"),
            dynamic_mask=dynamic_mask,
            occlusion_mask=_required(mask_values, "temporal_occlusion", "masks"),
            collision_mask=mask_values.get("temporal_collision"),
            valid_mask=temporal_valid,
            geometry_consistent_mask=mask_values.get("temporal_geometry_consistent"),
            confidence=mask_values.get("temporal_confidence"),
        )
        active.append("temporal")

    stereo_photo: Tensor | None = None
    stereo_photo_active: Tensor | bool | None = None
    if "stereo_reprojected_left_rgb" in geometry:
        stereo_left_rgb = _required(targets, "left_rgb", "targets")
        stereo_photo_valid = _required(
            mask_values, "stereo_reprojection_valid", "masks"
        )
        stereo_occlusion = _required(mask_values, "stereo_occlusion", "masks")
        stereo_confidence = mask_values.get("stereo_reprojection_confidence")
        stereo_photo = stereo_reprojection_loss(
            stereo_left_rgb,
            geometry["stereo_reprojected_left_rgb"],
            valid_mask=stereo_photo_valid,
            occlusion_mask=stereo_occlusion,
            confidence=stereo_confidence,
        )
        stereo_photo_active = _photometric_has_support(
            stereo_left_rgb,
            geometry["stereo_reprojected_left_rgb"],
            valid_mask=stereo_photo_valid,
            occlusion_mask=stereo_occlusion,
            dynamic_mask=None,
            confidence=stereo_confidence,
        )
        active.append("stereo_reprojection")

    temporal_photo: Tensor | None = None
    temporal_photo_active: Tensor | bool | None = None
    if "temporal_reprojected_left_rgb" in geometry:
        if dynamic_mask is None:
            raise KeyError(
                "masks must contain 'temporal_dynamic' or 'dynamic_mask_current'"
            )
        left_rgb = _required(targets, "left_rgb", "targets")
        temporal_photo_valid = (
            _pixel_mask(
                left_rgb,
                _required(mask_values, "temporal_reprojection_valid", "masks"),
                "temporal_reprojection_valid",
            )
            & dynamic_available
        )
        temporal_photo = temporal_reprojection_loss(
            left_rgb,
            geometry["temporal_reprojected_left_rgb"],
            valid_mask=temporal_photo_valid,
            occlusion_mask=_required(mask_values, "temporal_occlusion", "masks"),
            dynamic_mask=dynamic_mask,
            confidence=mask_values.get("temporal_reprojection_confidence"),
        )
        temporal_photo_active = _photometric_has_support(
            left_rgb,
            geometry["temporal_reprojected_left_rgb"],
            valid_mask=temporal_photo_valid,
            occlusion_mask=_required(mask_values, "temporal_occlusion", "masks"),
            dynamic_mask=dynamic_mask,
            confidence=mask_values.get("temporal_reprojection_confidence"),
        )
        active.append("temporal_reprojection")

    left_right = zero
    right_disparity: Tensor | None = None
    if predictions.get("right_inverse_depth_m_inv") is not None:
        right_inverse_depth = predictions["right_inverse_depth_m_inv"]
        _require_same_shape(canonical, right_inverse_depth, "right_inverse_depth_m_inv")
        right_disparity = disparity_from_inverse_depth_m_inv(
            right_inverse_depth, fx_px, baseline_m
        )
    elif predictions.get("disparity_right_px") is not None:
        right_disparity = predictions["disparity_right_px"]
    elif predictions.get("disparity_right_lr_px") is not None:
        right_lr = predictions["disparity_right_lr_px"]
        _require_geometry_tensor(right_lr, "predictions['disparity_right_lr_px']")
        right_disparity = _resize_to(
            canonical, right_lr, "predictions['disparity_right_lr_px']"
        )
        lr_to_hr_scale = _required(geometry, "stereo_lr_to_hr_scale", "geometry")
        right_disparity = right_disparity * _positive_metric_parameter(
            lr_to_hr_scale, canonical, "stereo_lr_to_hr_scale"
        )
    if right_disparity is not None:
        _require_same_shape(canonical, right_disparity, "right disparity prediction")
        if targets.get("disparity_right_px") is not None:
            target_right = targets["disparity_right_px"]
            _require_geometry_tensor(target_right, "targets['disparity_right_px']")
            _require_same_shape(
                canonical, target_right, "targets['disparity_right_px']"
            )
            valid_right_target = _required(targets, "valid_right", "targets")
            _require_same_shape(
                target_right, valid_right_target, "targets['valid_right']"
            )
            right_supervision_mask = (
                _mask_like(target_right, valid_right_target, "targets['valid_right']")
                & torch.isfinite(target_right)
                & (target_right > 0)
                & torch.isfinite(right_disparity)
                & _weight_support(
                    target_right,
                    targets.get("confidence_right"),
                    "targets['confidence_right']",
                )
            )
            right_supervised = robust_disparity_loss(
                right_disparity,
                target_right,
                valid_mask=right_supervision_mask,
                confidence=targets.get("confidence_right"),
            )
            right_is_active = right_supervision_mask.any().to(disparity.dtype)
            left_is_active = (
                (
                    valid
                    & torch.isfinite(canonical)
                    & _weight_support(
                        target_disparity,
                        targets.get("confidence"),
                        "targets['confidence']",
                    )
                )
                .any()
                .to(disparity.dtype)
            )
            active_view_count = left_is_active + right_is_active
            disparity = (
                left_is_active * disparity + right_is_active * right_supervised
            ) / active_view_count.clamp_min(1.0)
            active.append("right_disparity_supervision")
        derived_left = disparity_from_inverse_depth_m_inv(canonical, fx_px, baseline_m)
        left_right = left_right_consistency_loss(
            derived_left,
            right_disparity,
            valid_mask=mask_values.get("left_right_valid", valid),
            occlusion_mask=mask_values.get("stereo_occlusion"),
            confidence=mask_values.get("left_right_confidence"),
        )
        active.append("left_right_consistency")

    pose: Tensor | None = None
    pose_is_active: Tensor | bool | None = None
    if "pose_residual" in predictions:
        pose_target = targets.get("pose_residual")
        pose = pose_residual_loss(
            predictions["pose_residual"],
            pose_target,
            valid_mask=mask_values.get("pose_valid"),
            translation_normalizer_m=float(
                geometry.get("pose_translation_normalizer_m", 0.10)
            ),
            rotation_normalizer_rad=float(
                geometry.get("pose_rotation_normalizer_rad", 0.10)
            ),
        )
        pose_domain = torch.isfinite(predictions["pose_residual"])
        if pose_target is not None:
            pose_domain &= torch.isfinite(pose_target)
        pose_valid_mask = mask_values.get("pose_valid")
        if pose_valid_mask is not None:
            expected = predictions["pose_residual"].shape[:-1]
            expanded_pose_mask = pose_valid_mask
            if pose_valid_mask.shape == expected:
                expanded_pose_mask = pose_valid_mask.unsqueeze(-1)
            if expanded_pose_mask.shape[-1] == 1:
                expanded_pose_mask = expanded_pose_mask.expand_as(pose_domain)
            if expanded_pose_mask.dtype == torch.bool:
                pose_domain &= expanded_pose_mask
            else:
                pose_domain &= torch.isfinite(expanded_pose_mask) & (
                    expanded_pose_mask > 0.5
                )
        pose_is_active = pose_domain.any()
        active.append("pose")

    scale: Tensor | None = None
    scale_is_active: Tensor | bool | None = None
    if "log_scale" in predictions:
        if "log_scale" in targets:
            scale = scale_alignment_loss(
                predictions["log_scale"],
                targets["log_scale"],
                valid_mask=mask_values.get("scale_valid"),
            )
            scale_domain = (
                torch.isfinite(predictions["log_scale"])
                & torch.isfinite(targets["log_scale"])
                & _mask_like(
                    predictions["log_scale"],
                    mask_values.get("scale_valid"),
                    "scale_valid",
                )
            )
            scale_is_active = scale_domain.any()
        elif "vggt_inverse_depth_unscaled" in geometry:
            scale = inverse_depth_scale_alignment_loss(
                geometry["vggt_inverse_depth_unscaled"],
                target_inverse_depth,
                predictions["log_scale"],
                valid_mask=mask_values.get("scale_alignment_valid", valid),
                confidence=mask_values.get("scale_alignment_confidence"),
            )
            scale_domain = (
                torch.isfinite(geometry["vggt_inverse_depth_unscaled"])
                & (geometry["vggt_inverse_depth_unscaled"] > 0)
                & torch.isfinite(target_inverse_depth)
                & (target_inverse_depth > 0)
                & _mask_like(
                    geometry["vggt_inverse_depth_unscaled"],
                    mask_values.get("scale_alignment_valid", valid),
                    "scale_alignment_valid",
                )
                & _weight_support(
                    geometry["vggt_inverse_depth_unscaled"],
                    mask_values.get("scale_alignment_confidence"),
                    "scale_alignment_confidence",
                )
            )
            scale_is_active = scale_domain.any()
        else:
            raise KeyError(
                "log_scale requires targets['log_scale'] or "
                "geometry['vggt_inverse_depth_unscaled']"
            )
        active.append("scale")

    uncertainty = zero
    if "log_variance" in predictions:
        uncertainty = laplace_inverse_depth_uncertainty_loss(
            canonical,
            target_inverse_depth,
            predictions["log_variance"],
            valid_mask=valid,
            detach_geometry_residual=True,
        )
        active.append("uncertainty")

    validity = zero
    if "valid_logits" in predictions:
        validity = validity_classification_loss(
            predictions["valid_logits"],
            target_valid,
            supervision_mask=mask_values.get("validity_supervision"),
        )
        active.append("validity")

    return combine_metric_stereo_video_losses(
        disparity=disparity,
        depth=depth,
        temporal=temporal,
        stereo_reprojection=stereo_photo,
        temporal_reprojection=temporal_photo,
        stereo_reprojection_active=stereo_photo_active,
        temporal_reprojection_active=temporal_photo_active,
        left_right_consistency=left_right,
        pose=pose,
        scale=scale,
        pose_active=pose_is_active,
        scale_active=scale_is_active,
        uncertainty=uncertainty,
        validity=validity,
        weights=weights,
        active_terms=active,
    )


class MetricStereoVideoLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`metric_stereo_video_loss`."""

    def __init__(
        self,
        weights: MetricStereoVideoLossWeights = MetricStereoVideoLossWeights(),
    ) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        predictions: Mapping[str, Any],
        targets: Mapping[str, Any],
        geometry: Mapping[str, Any],
        masks: Mapping[str, Tensor] | None = None,
    ) -> MetricStereoVideoLossBreakdown:
        return metric_stereo_video_loss(
            predictions, targets, geometry, masks, weights=self.weights
        )


__all__ = [
    "MetricStereoVideoLoss",
    "MetricStereoVideoLossBreakdown",
    "MetricStereoVideoLossWeights",
    "combine_metric_stereo_video_losses",
    "depth_from_inverse_depth_m_inv",
    "disparity_from_inverse_depth_m_inv",
    "inverse_depth_scale_alignment_loss",
    "laplace_inverse_depth_uncertainty_loss",
    "left_right_consistency_loss",
    "metric_factor_px_m",
    "metric_stereo_video_loss",
    "normalized_log_depth_loss",
    "photometric_reprojection_loss",
    "pose_residual_loss",
    "robust_disparity_loss",
    "robust_multiscale_disparity_loss",
    "robust_multiscale_pixel_disparity_loss",
    "scale_alignment_loss",
    "stereo_reprojection_loss",
    "temporal_reprojection_loss",
    "validity_classification_loss",
    "visibility_aware_temporal_residual_loss",
    "warp_right_disparity_to_left",
]
