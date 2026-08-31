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


def temporal_disparity_error(
    current_disparity_hr_px: Tensor,
    warped_history_disparity_hr_px: Tensor,
    *,
    safe_mask: Tensor,
) -> MetricResult:
    """Mean absolute current/history disparity error on a supplied safe mask.

    All tensors have identical shape and disparities are in HR pixels. The
    caller owns construction of ``safe_mask`` from visibility, collision,
    photometric, static-scene, and geometry checks. Invalid warped history is
    excluded. A selected non-finite current prediction invalidates the metric
    rather than being silently dropped.
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
    "temporal_disparity_error",
    "trusted_region_degradation",
]
