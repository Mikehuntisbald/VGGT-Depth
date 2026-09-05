"""Evaluation metrics for the causal metric stereo-video model.

The helpers in this module are intentionally tensor-only and CPU-testable.  A
caller supplies one endpoint batch and receives numerator/count pairs, which
can be reduced exactly across distributed ranks without retaining predictions.
All depth residuals use the dimensionless log-depth error where appropriate;
disparity EPE remains in full-resolution pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
) -> dict[str, MetricValue]:
    """Compute endpoint spatial, depth, scale and uncertainty metrics."""

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
    gt_valid = gt_valid_mask.bool() & _finite_positive(gt_disparity_px)
    pred_valid = predicted_valid_mask.bool() & _finite_positive(predicted_depth_m)
    common = gt_valid & pred_valid
    fx = intrinsics_left[:, 0, 0].float().reshape(-1, 1, 1, 1)
    baseline = baseline_m.float().reshape(-1, 1, 1, 1)
    gt_depth = torch.where(gt_valid, fx * baseline / gt_disparity_px.float().clamp_min(1e-8), torch.zeros_like(gt_disparity_px))
    depth_absrel = (predicted_depth_m.float() - gt_depth).abs() / gt_depth.clamp_min(1e-8)
    depth_sq = (predicted_depth_m.float() - gt_depth).square()
    disp_error = (predicted_disparity_px.float() - gt_disparity_px.float()).abs()

    result: dict[str, MetricValue] = {
        "metric_depth_absrel": _value(depth_absrel, common),
        "metric_depth_rmse_squared": _value(depth_sq, common),
        "epe_px": _value(disp_error, common),
        "bad1_px": _value((disp_error > 1.0).float(), common),
        "bad3_px": _value((disp_error > 3.0).float(), common),
        "prediction_valid_fraction": _value(pred_valid.float(), gt_valid),
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
        result["dynamic_pixels"] = MetricValue(float(dynamic_common.sum().item()), int(dynamic_common.sum().item()))
        result["static_pixels"] = MetricValue(float(static_common.sum().item()), int(static_common.sum().item()))

    log_error = torch.where(common, torch.log(predicted_depth_m.float().clamp_min(1e-8) / gt_depth.clamp_min(1e-8)).abs(), torch.zeros_like(gt_depth))
    variance = predicted_uncertainty.float().clamp_min(1e-8)
    sigma = variance.sqrt()
    log_variance = variance.log()
    nll = 2.0**0.5 * log_error * torch.exp(-0.5 * log_variance) + 0.5 * log_variance
    confidence = predicted_valid_probability.float().clamp(0.0, 1.0)
    correct = (log_error <= 0.1).float()
    result["uncertainty_nll"] = _value(nll, common)
    result["uncertainty_1sigma_coverage"] = _value((log_error <= sigma).float(), common)
    result["uncertainty_2sigma_coverage"] = _value((log_error <= 2.0 * sigma).float(), common)
    result["uncertainty_brier"] = _value((confidence - correct).square(), common)
    # ECE is accumulated as |confidence - accuracy| over fixed confidence bins.
    ece_sum = torch.zeros((), dtype=torch.float64, device=confidence.device)
    ece_count = 0
    for lower, upper in zip(torch.linspace(0.0, 1.0, 11, device=confidence.device)[:-1], torch.linspace(0.0, 1.0, 11, device=confidence.device)[1:], strict=True):
        selected = common & (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        count = int(selected.sum().item())
        if count:
            ece_sum = ece_sum + (confidence[selected].double().mean() - correct[selected].double().mean()).abs() * count
            ece_count += count
    result["uncertainty_ece"] = MetricValue(float(ece_sum.item()), ece_count)
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


__all__ = ["MetricAccumulator", "MetricValue", "endpoint_metric_values", "scalar_metric"]
