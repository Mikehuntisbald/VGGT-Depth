"""Temporal and trusted-region disparity evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .disparity import MetricResult, end_point_error


@dataclass(frozen=True)
class TrustedRegionDegradationReport:
    """Candidate-versus-baseline error change on trusted FFS pixels."""

    candidate_epe_px: MetricResult
    baseline_epe_px: MetricResult
    absolute_change_px: float
    relative_change_percent: float
    valid: bool
    relative_valid: bool


def _require_matching(reference: Tensor, value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def legacy_temporal_disparity_error(
    current_disparity_hr_px: Tensor,
    warped_history_disparity_hr_px: Tensor,
    *,
    safe_mask: Tensor,
) -> MetricResult:
    """Legacy mean absolute current/history disparity difference.

    All tensors have identical shape and disparities are in HR pixels. The
    caller owns construction of ``safe_mask`` from visibility, collision,
    photometric, static-scene, and geometry checks. Invalid warped history is
    excluded. A selected non-finite current prediction invalidates the metric
    rather than being silently dropped.
    This is *not* a temporal-residual error against teacher/GT motion.  It
    penalizes real disparity change caused by camera or scene motion and is
    retained only so historical reports remain reproducible.
    """

    if (
        not isinstance(current_disparity_hr_px, Tensor)
        or not current_disparity_hr_px.is_floating_point()
    ):
        raise TypeError(
            "current_disparity_hr_px must be a floating-point torch.Tensor"
        )
    _require_matching(
        current_disparity_hr_px,
        warped_history_disparity_hr_px,
        "warped_history_disparity_hr_px",
    )
    if not warped_history_disparity_hr_px.is_floating_point():
        raise TypeError("warped_history_disparity_hr_px must be floating point")
    _require_matching(current_disparity_hr_px, safe_mask, "safe_mask")
    usable = (
        safe_mask.to(dtype=torch.bool)
        & torch.isfinite(warped_history_disparity_hr_px)
        & (warped_history_disparity_hr_px > 0)
    )
    return end_point_error(
        current_disparity_hr_px,
        warped_history_disparity_hr_px,
        valid_mask=usable,
    )


def temporal_residual_error(
    current_prediction_disparity_hr_px: Tensor,
    warped_previous_prediction_disparity_hr_px: Tensor,
    current_reference_disparity_hr_px: Tensor,
    warped_previous_reference_disparity_hr_px: Tensor,
    *,
    safe_mask: Tensor,
    current_reference_valid_mask: Tensor | None = None,
    warped_previous_reference_valid_mask: Tensor | None = None,
) -> MetricResult:
    """Measure prediction temporal-residual error against teacher/GT.

    The metric, in HR pixels, is

    ``|(d_hat_t - W(d_hat_{t-1})) - (d_star_t - W(d_star_{t-1}))|``.

    ``safe_mask`` is owned by the caller and must encode the common visible,
    static, non-collision, geometry-consistent model-history domain.  The
    reference masks further restrict the denominator to a valid current
    teacher/GT value and a valid z-buffered previous teacher/GT value.  A
    selected non-finite prediction invalidates the result instead of being
    silently omitted.  Reference non-finite/non-positive pixels are excluded
    because they do not define a supervision target.  Empty domains return the
    standard invalid, count-zero :class:`MetricResult`.
    """

    if (
        not isinstance(current_prediction_disparity_hr_px, Tensor)
        or not current_prediction_disparity_hr_px.is_floating_point()
    ):
        raise TypeError(
            "current_prediction_disparity_hr_px must be a floating-point "
            "torch.Tensor"
        )
    reference = current_prediction_disparity_hr_px
    for name, value in (
        (
            "warped_previous_prediction_disparity_hr_px",
            warped_previous_prediction_disparity_hr_px,
        ),
        ("current_reference_disparity_hr_px", current_reference_disparity_hr_px),
        (
            "warped_previous_reference_disparity_hr_px",
            warped_previous_reference_disparity_hr_px,
        ),
    ):
        _require_matching(reference, value, name)
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
    _require_matching(reference, safe_mask, "safe_mask")

    reference_usable = (
        safe_mask.to(dtype=torch.bool)
        & torch.isfinite(current_reference_disparity_hr_px)
        & (current_reference_disparity_hr_px > 0)
        & torch.isfinite(warped_previous_reference_disparity_hr_px)
        & (warped_previous_reference_disparity_hr_px > 0)
    )
    for name, value in (
        ("current_reference_valid_mask", current_reference_valid_mask),
        (
            "warped_previous_reference_valid_mask",
            warped_previous_reference_valid_mask,
        ),
    ):
        if value is not None:
            _require_matching(reference, value, name)
            reference_usable &= value.to(dtype=torch.bool)

    prediction_residual_hr_px = (
        current_prediction_disparity_hr_px
        - warped_previous_prediction_disparity_hr_px
    )
    reference_residual_hr_px = (
        current_reference_disparity_hr_px
        - warped_previous_reference_disparity_hr_px
    )
    # A temporal residual may legitimately be zero or negative, so the
    # disparity EPE helper (which requires a positive target disparity) cannot
    # define this reduction.
    count = int(reference_usable.sum().item())
    if count == 0:
        return MetricResult.invalid()
    selected_error_hr_px = (
        prediction_residual_hr_px - reference_residual_hr_px
    ).abs()[reference_usable]
    if not bool(torch.isfinite(selected_error_hr_px).all().item()):
        return MetricResult.invalid(count=count)
    numerator = float(selected_error_hr_px.to(dtype=torch.float64).sum().item())
    return MetricResult(
        value=numerator / count,
        numerator=numerator,
        count=count,
        valid=True,
    )


# Backward-compatible source API.  The explicit legacy name is preferred in
# new code so it cannot be mistaken for the v2 teacher/GT residual metric.
temporal_disparity_error = legacy_temporal_disparity_error


def trusted_region_degradation(
    candidate_disparity_hr_px: Tensor,
    baseline_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    trusted_mask: Tensor,
) -> TrustedRegionDegradationReport:
    """Compare candidate and baseline EPE on exactly the same trusted domain.

    Positive ``relative_change_percent`` means degradation. Relative change
    is undefined when baseline EPE is zero; this is represented by NaN and
    ``relative_valid=False``. Empty masks make the whole report invalid.
    """

    _require_matching(
        candidate_disparity_hr_px,
        baseline_disparity_hr_px,
        "baseline_disparity_hr_px",
    )
    _require_matching(
        candidate_disparity_hr_px,
        target_disparity_hr_px,
        "target_disparity_hr_px",
    )
    _require_matching(candidate_disparity_hr_px, trusted_mask, "trusted_mask")
    candidate = end_point_error(
        candidate_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=trusted_mask,
    )
    baseline = end_point_error(
        baseline_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=trusted_mask,
    )
    valid = candidate.valid and baseline.valid
    if not valid:
        absolute_change = float("nan")
        relative_change = float("nan")
        relative_valid = False
    else:
        absolute_change = candidate.value - baseline.value
        relative_valid = baseline.value > 0
        relative_change = (
            100.0 * absolute_change / baseline.value
            if relative_valid
            else float("nan")
        )
    return TrustedRegionDegradationReport(
        candidate_epe_px=candidate,
        baseline_epe_px=baseline,
        absolute_change_px=absolute_change,
        relative_change_percent=relative_change,
        valid=valid,
        relative_valid=relative_valid,
    )


__all__ = [
    "TrustedRegionDegradationReport",
    "legacy_temporal_disparity_error",
    "temporal_disparity_error",
    "temporal_residual_error",
    "trusted_region_degradation",
]
