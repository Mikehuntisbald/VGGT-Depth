"""Disparity evaluation metrics with explicit HR-pixel units.

All public reductions return :class:`MetricResult` instead of a bare scalar.
The stored ``numerator`` and ``count`` make dataset-level aggregation possible
without averaging per-image means. An empty evaluation domain returns
``valid=False`` and NaN, never a misleading zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class MetricResult:
    """One aggregation-friendly scalar metric.

    ``value == numerator / count`` when ``valid`` is true. The numerator is an
    error sum for mean metrics and a positive-event count for rate metrics.
    Values are Python numbers so reports can be serialized directly.
    """

    value: float
    numerator: float
    count: int
    valid: bool

    @classmethod
    def invalid(cls, *, count: int = 0) -> "MetricResult":
        return cls(value=float("nan"), numerator=float("nan"), count=count, valid=False)


@dataclass(frozen=True)
class DisparityMetricReport:
    """Core HR-disparity metrics for one batch."""

    epe_px: MetricResult
    bad_1: MetricResult
    bad_2: MetricResult


@dataclass(frozen=True)
class CompletenessImprovementReport:
    """Positive-disparity completeness inside a caller-supplied hole mask."""

    candidate: MetricResult
    baseline: MetricResult
    absolute_improvement: float
    absolute_improvement_percentage_points: float
    relative_improvement_percent: float
    valid: bool
    relative_valid: bool


@dataclass(frozen=True)
class OutputValidityReport:
    """Rates over an explicit output domain.

    ``invalid`` means non-finite or non-positive disparity. ``negative``,
    ``nan``, ``infinite`` and ``zero`` expose the overlapping/root causes
    needed to diagnose that aggregate rate.
    """

    invalid: MetricResult
    negative: MetricResult
    nan: MetricResult
    infinite: MetricResult
    zero: MetricResult

    @property
    def total_count(self) -> int:
        return self.invalid.count

    @property
    def invalid_rate(self) -> float:
        return self.invalid.value

    @property
    def negative_rate(self) -> float:
        return self.negative.value

    @property
    def nan_rate(self) -> float:
        return self.nan.value


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")


def _require_same_shape(reference: Tensor, other: Tensor, name: str) -> None:
    if not isinstance(other, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if other.shape != reference.shape:
        raise ValueError(
            f"{name} shape must match {tuple(reference.shape)}, got {tuple(other.shape)}"
        )


def _evaluation_domain(
    target_disparity_hr_px: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    """Return valid-target pixels selected by the optional exact-shape mask."""

    domain = torch.isfinite(target_disparity_hr_px) & (target_disparity_hr_px > 0)
    if valid_mask is not None:
        _require_same_shape(target_disparity_hr_px, valid_mask, "valid_mask")
        domain &= valid_mask.to(dtype=torch.bool)
    return domain


def _mean_result(values: Tensor, domain: Tensor) -> MetricResult:
    """Strict masked mean: selected non-finite values invalidate the result."""

    count = int(domain.sum().item())
    if count == 0:
        return MetricResult.invalid()
    selected = values[domain]
    if not bool(torch.isfinite(selected).all().item()):
        return MetricResult.invalid(count=count)
    numerator = float(selected.to(dtype=torch.float64).sum().item())
    return MetricResult(
        value=numerator / count,
        numerator=numerator,
        count=count,
        valid=True,
    )


def _rate_result(events: Tensor, domain: Tensor) -> MetricResult:
    count = int(domain.sum().item())
    if count == 0:
        return MetricResult.invalid()
    numerator = float(events[domain].to(dtype=torch.float64).sum().item())
    return MetricResult(numerator / count, numerator, count, True)


def end_point_error(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> MetricResult:
    """Mean absolute disparity error (EPE), in HR pixels.

    Inputs have identical arbitrary shape. The evaluation domain consists of
    finite, positive target disparities and ``valid_mask`` when supplied. A
    non-finite prediction in that domain invalidates EPE; it is not silently
    dropped. Finite non-positive predictions remain in EPE and should also be
    reported with :func:`invalid_negative_nan_rate`.
    """

    _require_floating_tensor(prediction_disparity_hr_px, "prediction_disparity_hr_px")
    _require_floating_tensor(target_disparity_hr_px, "target_disparity_hr_px")
    _require_same_shape(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        "target_disparity_hr_px",
    )
    domain = _evaluation_domain(target_disparity_hr_px, valid_mask)
    absolute_error_px = (prediction_disparity_hr_px - target_disparity_hr_px).abs()
    return _mean_result(absolute_error_px, domain)


def bad_pixel_rate(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    threshold_px: float,
    *,
    valid_mask: Tensor | None = None,
) -> MetricResult:
    """Fraction of target-valid pixels with absolute error > ``threshold_px``.

    Non-finite predictions count as bad pixels, which prevents an invalid
    output from improving Bad-N by being omitted.
    """

    if not math.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("threshold_px must be finite and > 0")
    _require_floating_tensor(prediction_disparity_hr_px, "prediction_disparity_hr_px")
    _require_floating_tensor(target_disparity_hr_px, "target_disparity_hr_px")
    _require_same_shape(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        "target_disparity_hr_px",
    )
    domain = _evaluation_domain(target_disparity_hr_px, valid_mask)
    finite_prediction = torch.isfinite(prediction_disparity_hr_px)
    absolute_error_px = (prediction_disparity_hr_px - target_disparity_hr_px).abs()
    events = ~finite_prediction | (absolute_error_px > threshold_px)
    return _rate_result(events, domain)


def disparity_metrics(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> DisparityMetricReport:
    """Return EPE, Bad-1, and Bad-2 for equal-shaped HR disparity tensors."""

    return DisparityMetricReport(
        epe_px=end_point_error(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            valid_mask=valid_mask,
        ),
        bad_1=bad_pixel_rate(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            1.0,
            valid_mask=valid_mask,
        ),
        bad_2=bad_pixel_rate(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            2.0,
            valid_mask=valid_mask,
        ),
    )


def low_confidence_region_epe(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    ffs_confidence: Tensor,
    *,
    confidence_threshold: float = 0.8,
    valid_mask: Tensor | None = None,
) -> MetricResult:
    """EPE where finite FFS confidence is below ``confidence_threshold``.

    Confidence is dimensionless and must have exactly the disparity shape.
    """

    if not math.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")
    _require_same_shape(prediction_disparity_hr_px, ffs_confidence, "ffs_confidence")
    _require_floating_tensor(ffs_confidence, "ffs_confidence")
    low_confidence_mask = torch.isfinite(ffs_confidence) & (
        ffs_confidence < confidence_threshold
    )
    if valid_mask is not None:
        _require_same_shape(prediction_disparity_hr_px, valid_mask, "valid_mask")
        low_confidence_mask &= valid_mask.to(dtype=torch.bool)
    return end_point_error(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=low_confidence_mask,
    )


def invalid_region_completeness(
    disparity_hr_px: Tensor,
    invalid_region_mask: Tensor,
    *,
    eligible_mask: Tensor | None = None,
) -> MetricResult:
    """Fraction of requested hole pixels filled by finite positive disparity.

    ``invalid_region_mask`` normally identifies invalid/holes in the original
    FFS observation. ``eligible_mask`` can exclude pixels without evaluable
    image support. The denominator is independent of the candidate output.
    """

    _require_floating_tensor(disparity_hr_px, "disparity_hr_px")
    _require_same_shape(disparity_hr_px, invalid_region_mask, "invalid_region_mask")
    domain = invalid_region_mask.to(dtype=torch.bool)
    if eligible_mask is not None:
        _require_same_shape(disparity_hr_px, eligible_mask, "eligible_mask")
        domain &= eligible_mask.to(dtype=torch.bool)
    filled = torch.isfinite(disparity_hr_px) & (disparity_hr_px > 0)
    return _rate_result(filled, domain)


def invalid_region_completeness_improvement(
    candidate_disparity_hr_px: Tensor,
    baseline_disparity_hr_px: Tensor,
    invalid_region_mask: Tensor,
    *,
    eligible_mask: Tensor | None = None,
) -> CompletenessImprovementReport:
    """Compare output completeness over the same caller-defined hole domain.

    Both absolute fraction and percentage-point improvement are reported.
    Relative improvement is only defined when baseline completeness is > 0.
    """

    _require_same_shape(
        candidate_disparity_hr_px,
        baseline_disparity_hr_px,
        "baseline_disparity_hr_px",
    )
    candidate = invalid_region_completeness(
        candidate_disparity_hr_px,
        invalid_region_mask,
        eligible_mask=eligible_mask,
    )
    baseline = invalid_region_completeness(
        baseline_disparity_hr_px,
        invalid_region_mask,
        eligible_mask=eligible_mask,
    )
    valid = candidate.valid and baseline.valid
    if not valid:
        absolute = float("nan")
        relative = float("nan")
        relative_valid = False
    else:
        absolute = candidate.value - baseline.value
        relative_valid = baseline.value > 0
        relative = 100.0 * absolute / baseline.value if relative_valid else float("nan")
    return CompletenessImprovementReport(
        candidate=candidate,
        baseline=baseline,
        absolute_improvement=absolute,
        absolute_improvement_percentage_points=100.0 * absolute,
        relative_improvement_percent=relative,
        valid=valid,
        relative_valid=relative_valid,
    )


def invalid_negative_nan_rate(
    disparity_hr_px: Tensor,
    *,
    evaluation_mask: Tensor | None = None,
) -> OutputValidityReport:
    """Report invalid, negative, NaN, infinity, and zero output rates.

    The default domain is every tensor element. ``evaluation_mask`` must have
    exactly the disparity shape. Empty domains produce invalid submetrics.
    """

    _require_floating_tensor(disparity_hr_px, "disparity_hr_px")
    if evaluation_mask is None:
        domain = torch.ones_like(disparity_hr_px, dtype=torch.bool)
    else:
        _require_same_shape(disparity_hr_px, evaluation_mask, "evaluation_mask")
        domain = evaluation_mask.to(dtype=torch.bool)
    finite = torch.isfinite(disparity_hr_px)
    return OutputValidityReport(
        invalid=_rate_result(~finite | (disparity_hr_px <= 0), domain),
        negative=_rate_result(finite & (disparity_hr_px < 0), domain),
        nan=_rate_result(torch.isnan(disparity_hr_px), domain),
        infinite=_rate_result(torch.isinf(disparity_hr_px), domain),
        zero=_rate_result(finite & (disparity_hr_px == 0), domain),
    )


# Concise aliases used by evaluation scripts and tables.
epe = end_point_error
bad_n = bad_pixel_rate
output_validity_metrics = invalid_negative_nan_rate


__all__ = [
    "CompletenessImprovementReport",
    "DisparityMetricReport",
    "MetricResult",
    "OutputValidityReport",
    "bad_n",
    "bad_pixel_rate",
    "disparity_metrics",
    "end_point_error",
    "epe",
    "invalid_negative_nan_rate",
    "invalid_region_completeness",
    "invalid_region_completeness_improvement",
    "low_confidence_region_epe",
    "output_validity_metrics",
]
