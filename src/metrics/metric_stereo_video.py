"""Evaluation metrics for the causal metric stereo-video model.

The helpers in this module are intentionally tensor-only and CPU-testable.  A
caller supplies one endpoint batch and receives numerator/count pairs, which
can be reduced exactly across distributed ranks without retaining predictions.
All depth residuals use the dimensionless log-depth error where appropriate;
disparity EPE remains in full-resolution pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MetricValue:
    numerator: float
    count: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.count if self.count else None


def _value(values: Tensor, mask: Tensor) -> MetricValue:
    selected = values.float()[mask.bool()]
    if selected.numel() == 0:
        return MetricValue(0.0, 0)
    finite = torch.isfinite(selected)
    selected = selected[finite]
    return MetricValue(float(selected.double().sum().item()), int(selected.numel()))


def _check_image(name: str, value: Tensor, reference: Tensor) -> None:
    if value.shape != reference.shape or value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must match [B,1,H,W] {tuple(reference.shape)}")


def _finite_positive(value: Tensor) -> Tensor:
    return torch.isfinite(value) & (value > 0)


def _per_sample_scale_error(pred_depth: Tensor, gt_depth: Tensor, mask: Tensor) -> MetricValue:
    values: list[Tensor] = []
    for pred, gt, valid in zip(pred_depth, gt_depth, mask, strict=True):
        selected = valid.bool() & _finite_positive(pred) & _finite_positive(gt)
        if selected.any():
            ratio = (pred[selected] / gt[selected]).float().clamp_min(1e-8)
            values.append(torch.log(ratio).median().abs())
    if not values:
        return MetricValue(0.0, 0)
    stacked = torch.stack(values)
    return MetricValue(float(stacked.double().sum().item()), len(values))


def endpoint_metric_values(
    *,
    predicted_depth_m: Tensor,
    predicted_disparity_px: Tensor,
    predicted_valid_mask: Tensor,
    predicted_valid_probability: Tensor,
    predicted_uncertainty: Tensor,
    gt_disparity_px: Tensor,
    gt_valid_mask: Tensor,
    intrinsics_left: Tensor,
    baseline_m: Tensor,
    dynamic_mask: Tensor | None = None,
    dynamic_available: Tensor | None = None,
    detail_mask: Tensor | None = None,
    matched_mask: Tensor | None = None,
    boundary_mask: Tensor | None = None,
    invalid_penalty_px: float = 10.0,
) -> dict[str, MetricValue]:
    """Compute endpoint metrics without allowing learned validity rejection.

    Legacy/common-valid metrics remain available for diagnosis. Formal spatial
    metrics use every GT-valid pixel. Valid predictions contribute their error
    capped at ``invalid_penalty_px`` and rejected/non-finite predictions incur
    exactly that penalty, so rejecting a hard pixel cannot improve the score.
    """

    for name, value in (
        ("predicted_depth_m", predicted_depth_m),
        ("predicted_disparity_px", predicted_disparity_px),
        ("predicted_valid_mask", predicted_valid_mask),
        ("predicted_valid_probability", predicted_valid_probability),
        ("predicted_uncertainty", predicted_uncertainty),
        ("gt_disparity_px", gt_disparity_px),
        ("gt_valid_mask", gt_valid_mask),
    ):
        _check_image(name, value, predicted_depth_m)
    if intrinsics_left.ndim != 3 or intrinsics_left.shape[0] != predicted_depth_m.shape[0]:
        raise ValueError("intrinsics_left must be [B,3,3]")
    if baseline_m.ndim != 1 or baseline_m.shape[0] != predicted_depth_m.shape[0]:
        raise ValueError("baseline_m must be [B]")
    if not isinstance(invalid_penalty_px, (int, float)) or invalid_penalty_px <= 0:
        raise ValueError("invalid_penalty_px must be positive")
    gt_valid = gt_valid_mask.bool() & _finite_positive(gt_disparity_px)
    pred_valid = predicted_valid_mask.bool() & _finite_positive(predicted_depth_m)
    common = gt_valid & pred_valid
    fx = intrinsics_left[:, 0, 0].float().reshape(-1, 1, 1, 1)
    baseline = baseline_m.float().reshape(-1, 1, 1, 1)
    gt_depth = torch.where(gt_valid, fx * baseline / gt_disparity_px.float().clamp_min(1e-8), torch.zeros_like(gt_disparity_px))
    depth_absrel = (predicted_depth_m.float() - gt_depth).abs() / gt_depth.clamp_min(1e-8)
    depth_sq = (predicted_depth_m.float() - gt_depth).square()
    disp_error = (predicted_disparity_px.float() - gt_disparity_px.float()).abs()
    finite_prediction = _finite_positive(predicted_disparity_px)
    formal_valid = pred_valid & finite_prediction
    penalized_error = torch.where(
        formal_valid,
        disp_error.clamp_max(float(invalid_penalty_px)),
        torch.full_like(disp_error, float(invalid_penalty_px)),
    )
    failed_1px = (~formal_valid) | (disp_error > 1.0)
    failed_3px = (~formal_valid) | (disp_error > 3.0)

    result: dict[str, MetricValue] = {
        "metric_depth_absrel": _value(depth_absrel, common),
        "metric_depth_rmse_squared": _value(depth_sq, common),
        "epe_px": _value(disp_error, common),
        "common_valid_epe_px": _value(disp_error, common),
        "bad1_px": _value((disp_error > 1.0).float(), common),
        "bad3_px": _value((disp_error > 3.0).float(), common),
        "all_gt_penalized_epe_px": _value(penalized_error, gt_valid),
        "all_gt_bad1": _value(failed_1px.float(), gt_valid),
        "all_gt_bad3": _value(failed_3px.float(), gt_valid),
        "completeness_1px": _value((formal_valid & (disp_error <= 1.0)).float(), gt_valid),
        "completeness_2px": _value((formal_valid & (disp_error <= 2.0)).float(), gt_valid),
        "prediction_coverage": _value(formal_valid.float(), gt_valid),
        "prediction_valid_fraction": _value(formal_valid.float(), gt_valid),
        "gt_valid_pixels": MetricValue(float(gt_valid.sum().item()), int(gt_valid.numel())),
        "scale_error_log": _per_sample_scale_error(predicted_depth_m.float(), gt_depth, gt_valid),
    }
    if dynamic_mask is None or dynamic_available is None:
        result["dynamic_epe_px"] = MetricValue(0.0, 0)
        result["static_epe_px"] = _value(disp_error, common)
        result["dynamic_pixels"] = MetricValue(0.0, 0)
        result["static_pixels"] = MetricValue(float(common.sum().item()), int(common.sum().item()))
    else:
        if dynamic_mask.shape != predicted_depth_m.shape or dynamic_available.shape != (predicted_depth_m.shape[0],):
            raise ValueError("dynamic labels have incompatible endpoint shapes")
        available = dynamic_available.bool().reshape(-1, 1, 1, 1)
        dynamic = dynamic_mask.bool() & available
        dynamic_common = common & dynamic
        static_common = common & (~dynamic) & available
        result["dynamic_epe_px"] = _value(disp_error, dynamic_common)
        result["static_epe_px"] = _value(disp_error, static_common)
        result["dynamic_penalized_epe_px"] = _value(
            penalized_error, gt_valid & dynamic
        )
        result["static_penalized_epe_px"] = _value(
            penalized_error, gt_valid & (~dynamic) & available
        )
        result["dynamic_pixels"] = MetricValue(float(dynamic_common.sum().item()), int(dynamic_common.sum().item()))
        result["static_pixels"] = MetricValue(float(static_common.sum().item()), int(static_common.sum().item()))

    log_error = torch.where(common, torch.log(predicted_depth_m.float().clamp_min(1e-8) / gt_depth.clamp_min(1e-8)).abs(), torch.zeros_like(gt_depth))
    variance = predicted_uncertainty.float().clamp_min(1e-8)
    sigma = variance.sqrt()
    log_variance = variance.log()
    nll = 2.0**0.5 * log_error * torch.exp(-0.5 * log_variance) + 0.5 * log_variance
    confidence = predicted_valid_probability.float().clamp(0.0, 1.0)
    correct = (formal_valid & (log_error <= 0.1)).float()
    result["uncertainty_nll"] = _value(nll, common)
    result["uncertainty_1sigma_coverage"] = _value((log_error <= sigma).float(), common)
    result["uncertainty_2sigma_coverage"] = _value((log_error <= 2.0 * sigma).float(), common)
    result["validity_confidence_brier_0p1_log_depth"] = _value(
        (confidence - correct).square(), gt_valid
    )
    # Calibration is evaluated on every GT-valid pixel. A rejected prediction
    # is an incorrect outcome even when its latent depth value happens to be
    # close, so the validity head cannot improve calibration by hiding it.
    ece_sum = torch.zeros((), dtype=torch.float64, device=confidence.device)
    ece_count = 0
    for lower, upper in zip(torch.linspace(0.0, 1.0, 11, device=confidence.device)[:-1], torch.linspace(0.0, 1.0, 11, device=confidence.device)[1:], strict=True):
        selected = gt_valid & (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        count = int(selected.sum().item())
        if count:
            ece_sum = ece_sum + (confidence[selected].double().mean() - correct[selected].double().mean()).abs() * count
            ece_count += count
    result["validity_confidence_ece_0p1_log_depth"] = MetricValue(
        float(ece_sum.item()), ece_count
    )
    # Compatibility aliases retained for the first diagnostic report. These
    # are validity-confidence metrics, not variance calibration metrics.
    result["uncertainty_brier"] = result[
        "validity_confidence_brier_0p1_log_depth"
    ]
    result["uncertainty_ece"] = result[
        "validity_confidence_ece_0p1_log_depth"
    ]

    for name, mask in (
        ("spring_high_detail", detail_mask),
        ("spring_low_detail", None if detail_mask is None else ~detail_mask.bool()),
        ("spring_matched", matched_mask),
        ("spring_boundary", boundary_mask),
    ):
        if mask is None:
            result[f"{name}_epe_px"] = MetricValue(0.0, 0)
            result[f"{name}_bad1"] = MetricValue(0.0, 0)
            continue
        _check_image(f"{name}_mask", mask, predicted_depth_m)
        domain = gt_valid & mask.bool()
        result[f"{name}_epe_px"] = _value(penalized_error, domain)
        result[f"{name}_bad1"] = _value(failed_1px.float(), domain)
    if matched_mask is None:
        result["spring_unmatched_completion_1px"] = MetricValue(0.0, 0)
        result["spring_unmatched_completion_2px"] = MetricValue(0.0, 0)
    else:
        unmatched = gt_valid & ~matched_mask.bool()
        result["spring_unmatched_completion_1px"] = _value(
            (formal_valid & (disp_error <= 1.0)).float(), unmatched
        )
        result["spring_unmatched_completion_2px"] = _value(
            (formal_valid & (disp_error <= 2.0)).float(), unmatched
        )
    return result


def temporal_residual_metric_values(
    *,
    current_prediction_disparity_px: Tensor,
    warped_previous_prediction_disparity_px: Tensor,
    current_gt_disparity_px: Tensor,
    warped_previous_gt_disparity_px: Tensor,
    current_prediction_valid: Tensor,
    warped_prediction_valid: Tensor,
    current_gt_valid: Tensor,
    warped_gt_valid: Tensor,
    dynamic_mask: Tensor,
    dynamic_available: Tensor,
    motion_bucket_masks: Mapping[str, Tensor] | None = None,
    invalid_penalty_px: float = 10.0,
) -> dict[str, MetricValue]:
    """Compare predicted temporal change to GT temporal change.

    Delta residual is defined only where the previous GT surface reaches the
    current frame. Disocclusions have no valid GT delta target and therefore
    report current-frame reconstruction/completion metrics instead.
    """

    reference = current_prediction_disparity_px
    for name, value in (
        ("warped_previous_prediction_disparity_px", warped_previous_prediction_disparity_px),
        ("current_gt_disparity_px", current_gt_disparity_px),
        ("warped_previous_gt_disparity_px", warped_previous_gt_disparity_px),
        ("current_prediction_valid", current_prediction_valid),
        ("warped_prediction_valid", warped_prediction_valid),
        ("current_gt_valid", current_gt_valid),
        ("warped_gt_valid", warped_gt_valid),
        ("dynamic_mask", dynamic_mask),
    ):
        _check_image(name, value, reference)
    if dynamic_available.shape != (reference.shape[0],):
        raise ValueError("dynamic_available must be [B]")
    if invalid_penalty_px <= 0:
        raise ValueError("invalid_penalty_px must be positive")

    gt_current = current_gt_valid.bool() & _finite_positive(current_gt_disparity_px)
    gt_warped = warped_gt_valid.bool() & _finite_positive(
        warped_previous_gt_disparity_px
    )
    gt_matched = gt_current & gt_warped
    pred_current = current_prediction_valid.bool() & _finite_positive(
        current_prediction_disparity_px
    )
    pred_warped = warped_prediction_valid.bool() & _finite_positive(
        warped_previous_prediction_disparity_px
    )
    prediction_supported = pred_current & pred_warped
    common = gt_matched & prediction_supported
    prediction_delta = (
        current_prediction_disparity_px.float()
        - warped_previous_prediction_disparity_px.float()
    )
    gt_delta = (
        current_gt_disparity_px.float()
        - warped_previous_gt_disparity_px.float()
    )
    delta_error = (prediction_delta - gt_delta).abs()
    penalized_delta = torch.where(
        prediction_supported,
        delta_error.clamp_max(float(invalid_penalty_px)),
        torch.full_like(delta_error, float(invalid_penalty_px)),
    )
    available = dynamic_available.bool().reshape(-1, 1, 1, 1)
    dynamic = dynamic_mask.bool() & available
    rigid = ~dynamic & available
    result = {
        "temporal_matched_common_valid_delta_epe_px": _value(delta_error, common),
        "temporal_matched_penalized_delta_epe_px": _value(
            penalized_delta, gt_matched
        ),
        "rigid_temporal_residual_epe_px": _value(
            penalized_delta, gt_matched & rigid
        ),
        "non_rigid_temporal_residual_epe_px": _value(
            penalized_delta, gt_matched & dynamic
        ),
        "temporal_prediction_transport_miss_rate": _value(
            (~prediction_supported).float(), gt_matched
        ),
    }
    current_error = (
        current_prediction_disparity_px.float() - current_gt_disparity_px.float()
    ).abs()
    disocclusion = gt_current & ~gt_warped
    disocclusion_penalized = torch.where(
        pred_current,
        current_error.clamp_max(float(invalid_penalty_px)),
        torch.full_like(current_error, float(invalid_penalty_px)),
    )
    result["temporal_disocclusion_current_epe_px"] = _value(
        disocclusion_penalized, disocclusion
    )
    result["temporal_disocclusion_completion_1px"] = _value(
        (pred_current & (current_error <= 1.0)).float(), disocclusion
    )
    result["temporal_disocclusion_completion_2px"] = _value(
        (pred_current & (current_error <= 2.0)).float(), disocclusion
    )
    for name, mask in (motion_bucket_masks or {}).items():
        _check_image(f"motion_bucket_masks[{name!r}]", mask, reference)
        result[f"temporal_{name}_penalized_delta_epe_px"] = _value(
            penalized_delta, gt_matched & mask.bool()
        )
    return result


def scalar_metric(values: Tensor, mask: Tensor) -> MetricValue:
    """Public empty-safe scalar reduction used by the evaluator and tests."""

    return _value(values, mask)


@dataclass(slots=True)
class MetricAccumulator:
    """Exact numerator/count reduction for one variant."""

    values: dict[str, list[float | int]] = field(default_factory=dict)

    def update(self, metrics: Mapping[str, MetricValue]) -> None:
        for name, metric in metrics.items():
            if metric.count < 0 or not torch.isfinite(torch.tensor(metric.numerator)):
                raise ValueError(f"invalid metric value for {name}")
            item = self.values.setdefault(name, [0.0, 0])
            item[0] = float(item[0]) + float(metric.numerator)
            item[1] = int(item[1]) + int(metric.count)

    def merge(self, other: "MetricAccumulator") -> None:
        self.update({name: MetricValue(float(value[0]), int(value[1])) for name, value in other.values.items()})

    def finalize(self) -> dict[str, dict[str, float | int | bool | None]]:
        result: dict[str, dict[str, float | int | bool | None]] = {}
        for name, (numerator, count) in sorted(self.values.items()):
            count_int = int(count)
            result[name] = {
                "value": float(numerator) / count_int if count_int else None,
                "numerator": float(numerator),
                "count": count_int,
                "valid": count_int > 0,
            }
        # RMSE is defined from the globally aggregated squared residual.
        rmse = result.get("metric_depth_rmse_squared")
        if rmse is not None:
            value = rmse["value"]
            result["metric_depth_rmse"] = {
                "value": float(value) ** 0.5 if value is not None else None,
                "numerator": rmse["numerator"],
                "count": rmse["count"],
                "valid": rmse["valid"],
            }
        return result


@dataclass(slots=True)
class AccuracyCoverageHistogram:
    """Fixed-bin global accuracy/coverage reduction ranked by confidence."""

    bins: int
    device: torch.device
    error_sum: Tensor = field(init=False)
    count: Tensor = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins < 2:
            raise ValueError("coverage histogram bins must be at least two")
        self.error_sum = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.count = torch.zeros(self.bins, dtype=torch.float64, device=self.device)

    def update(
        self,
        confidence: Tensor,
        disparity_error_px: Tensor,
        gt_valid_mask: Tensor,
        *,
        error_cap_px: float,
    ) -> None:
        if confidence.shape != disparity_error_px.shape or confidence.shape != gt_valid_mask.shape:
            raise ValueError("coverage tensors must share one shape")
        domain = gt_valid_mask.bool()
        if not domain.any():
            return
        selected_confidence = torch.nan_to_num(
            confidence.float()[domain], nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        selected_error = torch.nan_to_num(
            disparity_error_px.float()[domain],
            nan=float(error_cap_px),
            posinf=float(error_cap_px),
            neginf=float(error_cap_px),
        ).clamp(0.0, float(error_cap_px))
        index = torch.floor(selected_confidence * (self.bins - 1)).long()
        self.count += torch.bincount(index, minlength=self.bins).to(torch.float64)
        self.error_sum += torch.bincount(
            index, weights=selected_error.to(torch.float64), minlength=self.bins
        )

    def all_reduce_(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.error_sum)
            torch.distributed.all_reduce(self.count)

    def finalize(self, coverage_points: Sequence[float]) -> dict[str, object]:
        for coverage in coverage_points:
            if not 0.0 < float(coverage) <= 1.0:
                raise ValueError("coverage points must lie in (0,1]")
        counts = self.count.detach().cpu().flip(0)
        errors = self.error_sum.detach().cpu().flip(0)
        total = float(counts.sum().item())
        curve: list[dict[str, float | int | None]] = []
        if total == 0:
            return {"histogram_bins": self.bins, "gt_valid_count": 0, "points": curve}
        cumulative_count = torch.cumsum(counts, dim=0)
        cumulative_error = torch.cumsum(errors, dim=0)
        for requested in coverage_points:
            target = float(requested) * total
            bin_index = int(torch.searchsorted(cumulative_count, torch.tensor(target)).item())
            bin_index = min(bin_index, self.bins - 1)
            previous_count = float(cumulative_count[bin_index - 1].item()) if bin_index else 0.0
            previous_error = float(cumulative_error[bin_index - 1].item()) if bin_index else 0.0
            take = max(0.0, target - previous_count)
            bin_count = float(counts[bin_index].item())
            bin_error = float(errors[bin_index].item())
            error = previous_error + (take / bin_count * bin_error if bin_count else 0.0)
            threshold_bin = self.bins - 1 - bin_index
            curve.append(
                {
                    "requested_coverage": float(requested),
                    "effective_count": int(round(target)),
                    "epe_px": error / target if target else None,
                    "minimum_confidence_approx": threshold_bin / (self.bins - 1),
                }
            )
        return {
            "histogram_bins": self.bins,
            "gt_valid_count": int(round(total)),
            "points": curve,
        }


__all__ = [
    "AccuracyCoverageHistogram",
    "MetricAccumulator",
    "MetricValue",
    "endpoint_metric_values",
    "scalar_metric",
    "temporal_residual_metric_values",
]
