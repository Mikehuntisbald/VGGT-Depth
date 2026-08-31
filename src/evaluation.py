"""Stage-A evaluation primitives with aggregation-safe metric accounting.

This module deliberately evaluates only the spatial ``T=1`` model.  The
reference target is the trusted subset of an HR FFS teacher cache, so results
produced here are pseudo-GT engineering measurements rather than paper
accuracy.  All disparities passed to this module are expressed in HR pixels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from metrics.boundary import boundary_epe
from metrics.disparity import (
    MetricResult,
    disparity_metrics,
    invalid_negative_nan_rate,
    invalid_region_completeness,
    low_confidence_region_epe,
)
from utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointMismatchError


PSEUDO_GT_LABEL = "trusted_hr_ffs_teacher_pseudo_gt"
POINT_TO_PLANE_NOT_AVAILABLE = {
    "status": "NOT_AVAILABLE",
    "reason": "target point normals and explicit correspondences are unavailable",
}


@dataclass(frozen=True, slots=True)
class AggregateMetric:
    """A JSON-safe dataset reduction.

    ``value`` and ``numerator`` are ``None`` for an empty domain or when a
    selected non-finite value makes a mean undefined.  ``count`` still records
    the complete selected denominator in the latter case.
    """

    value: float | None
    numerator: float | None
    count: int
    valid: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "count": self.count,
            "valid": self.valid,
        }


@dataclass(slots=True)
class MetricAccumulator:
    """Combine metric numerators and counts without averaging image means."""

    numerator: float = 0.0
    count: int = 0
    invalid_selected_count: int = 0

    def update(self, result: MetricResult) -> None:
        if not isinstance(result, MetricResult):
            raise TypeError("result must be MetricResult")
        if result.count < 0:
            raise ValueError("metric count must be non-negative")
        if result.valid:
            if result.count <= 0:
                raise ValueError("a valid metric must have a positive count")
            if not torch.isfinite(torch.tensor(result.numerator)):
                raise ValueError("a valid metric must have a finite numerator")
            self.numerator += float(result.numerator)
            self.count += int(result.count)
        elif result.count > 0:
            # The metric implementation uses this state for selected invalid
            # predictions.  Do not silently discard that part of the domain.
            self.invalid_selected_count += int(result.count)

    def finalize(self) -> AggregateMetric:
        total_count = self.count + self.invalid_selected_count
        if total_count == 0:
            return AggregateMetric(None, None, 0, False)
        if self.invalid_selected_count:
            return AggregateMetric(None, None, total_count, False)
        return AggregateMetric(
            value=self.numerator / self.count,
            numerator=self.numerator,
            count=self.count,
            valid=True,
        )


def aggregate_metric_results(results: Iterable[MetricResult]) -> AggregateMetric:
    """Pure convenience wrapper around :class:`MetricAccumulator`."""

    accumulator = MetricAccumulator()
    for result in results:
        accumulator.update(result)
    return accumulator.finalize()


@dataclass(slots=True)
class MethodMetricAccumulator:
    """Named metric accumulators for one evaluated method."""

    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)

    def update(self, sample_metrics: Mapping[str, MetricResult]) -> None:
        for name, result in sample_metrics.items():
            self.metrics.setdefault(name, MetricAccumulator()).update(result)

    def finalize(self) -> dict[str, AggregateMetric]:
        return {name: accumulator.finalize() for name, accumulator in self.metrics.items()}


def upsample_ffs_inputs_to_hr(
    disparity_ffs_hr_px_lr_grid: Tensor,
    confidence_ffs_lr: Tensor,
    valid_ffs_lr: Tensor,
    trusted_ffs_lr: Tensor,
    *,
    output_size_hw: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Upsample LR-grid FFS fields while preserving their semantics.

    Disparity already has HR-pixel units and is bilinearly interpolated without
    an additional scale multiplier. Confidence is also bilinear. Boolean masks
    are nearest-neighbor sampled so validity is never invented by interpolation.
    """

    reference_shape = disparity_ffs_hr_px_lr_grid.shape
    if disparity_ffs_hr_px_lr_grid.ndim != 4 or reference_shape[1] != 1:
        raise ValueError("disparity_ffs_hr_px_lr_grid must be [B,1,H,W]")
    for name, tensor in (
        ("confidence_ffs_lr", confidence_ffs_lr),
        ("valid_ffs_lr", valid_ffs_lr),
        ("trusted_ffs_lr", trusted_ffs_lr),
    ):
        if tensor.shape != reference_shape:
            raise ValueError(f"{name} must have shape {tuple(reference_shape)}")
    safe_disparity = torch.nan_to_num(
        disparity_ffs_hr_px_lr_grid, nan=0.0, posinf=0.0, neginf=0.0
    )
    disparity_hr_px = functional.interpolate(
        safe_disparity,
        size=output_size_hw,
        mode="bilinear",
        align_corners=False,
    )
    confidence_hr = functional.interpolate(
        torch.nan_to_num(confidence_ffs_lr, nan=0.0, posinf=0.0, neginf=0.0),
        size=output_size_hw,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    def nearest_mask(mask: Tensor) -> Tensor:
        return functional.interpolate(
            mask.to(dtype=torch.float32), size=output_size_hw, mode="nearest"
        ).to(dtype=torch.bool)

    return (
        disparity_hr_px,
        confidence_hr,
        nearest_mask(valid_ffs_lr),
        nearest_mask(trusted_ffs_lr),
    )


def compute_sample_metrics(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    target_trusted_mask: Tensor,
    ffs_confidence_hr: Tensor,
    ffs_valid_mask_hr: Tensor,
    ffs_trusted_mask_hr: Tensor,
    low_confidence_threshold: float = 0.8,
    boundary_gradient_threshold_px: float = 1.0,
    boundary_radius_px: int = 1,
) -> dict[str, MetricResult]:
    """Compute one method's Stage-A metrics on explicit HR-grid domains."""

    if prediction_disparity_hr_px.shape != target_disparity_hr_px.shape:
        raise ValueError("prediction and target disparity shapes must match")
    expected_shape = prediction_disparity_hr_px.shape
    for name, value in (
        ("target_trusted_mask", target_trusted_mask),
        ("ffs_confidence_hr", ffs_confidence_hr),
        ("ffs_valid_mask_hr", ffs_valid_mask_hr),
        ("ffs_trusted_mask_hr", ffs_trusted_mask_hr),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")

    trusted_target = target_trusted_mask.to(dtype=torch.bool)
    core = disparity_metrics(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=trusted_target,
    )
    validity = invalid_negative_nan_rate(prediction_disparity_hr_px)
    # Boundary construction itself must not inspect untrusted teacher values;
    # merely intersecting the resulting band afterward would allow an
    # untrusted neighbor to create a boundary on a trusted pixel.
    trusted_boundary_target = torch.where(
        trusted_target,
        target_disparity_hr_px,
        torch.full_like(target_disparity_hr_px, float("nan")),
    )
    # Trusted-region degradation is computed dataset-wide from this exact EPE
    # for the baseline and candidate, rather than averaging per-image ratios.
    trusted_ffs_target = trusted_target & ffs_trusted_mask_hr.to(dtype=torch.bool)
    return {
        "epe_px": core.epe_px,
        "bad_1": core.bad_1,
        "bad_2": core.bad_2,
        "boundary_epe_px": boundary_epe(
            prediction_disparity_hr_px,
            trusted_boundary_target,
            valid_mask=trusted_target,
            gradient_threshold_px=boundary_gradient_threshold_px,
            radius_px=boundary_radius_px,
        ),
        "low_confidence_epe_px": low_confidence_region_epe(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            ffs_confidence_hr,
            confidence_threshold=low_confidence_threshold,
            valid_mask=trusted_target,
        ),
        "invalid_region_completeness": invalid_region_completeness(
            prediction_disparity_hr_px,
            ~ffs_valid_mask_hr.to(dtype=torch.bool),
            eligible_mask=trusted_target,
        ),
        "trusted_region_epe_px": disparity_metrics(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            valid_mask=trusted_ffs_target,
        ).epe_px,
        "output_invalid_rate": validity.invalid,
        "output_negative_rate": validity.negative,
        "output_nan_rate": validity.nan,
        "output_infinite_rate": validity.infinite,
        "output_zero_rate": validity.zero,
    }


def comparison_from_aggregates(
    baseline: Mapping[str, AggregateMetric],
    candidate: Mapping[str, AggregateMetric],
) -> dict[str, Any]:
    """Compute aggregate-only Stage-A go/no-go comparison values."""

    def change(metric_name: str) -> dict[str, Any]:
        base = baseline[metric_name]
        cand = candidate[metric_name]
        valid = base.valid and cand.valid
        absolute = cand.value - base.value if valid else None  # type: ignore[operator]
        relative_valid = valid and base.value is not None and base.value != 0.0
        relative = (
            100.0 * absolute / base.value  # type: ignore[operator]
            if relative_valid and absolute is not None
            else None
        )
        return {
            "baseline": base.to_dict(),
            "candidate": cand.to_dict(),
            "absolute_change": absolute,
            "relative_change_percent": relative,
            "valid": valid,
            "relative_valid": relative_valid,
        }

    return {
        "trusted_region_degradation": change("trusted_region_epe_px"),
        "low_confidence_epe_change": change("low_confidence_epe_px"),
        "invalid_region_completeness_change": change(
            "invalid_region_completeness"
        ),
    }


def load_model_for_evaluation(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    expected_parameter_count: int,
) -> dict[str, Any]:
    """Strictly load the model member of a local Stage-A training checkpoint.

    The training resume loader also requires an optimizer and scheduler, which
    evaluation intentionally does not create.  This loader applies the same
    schema and parameter-count checks before a strict model-state load.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("checkpoint payload is not a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError(
            "checkpoint schema mismatch: expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    for key in ("model", "parameter_count", "step", "config", "git_hash"):
        if key not in payload:
            raise CheckpointMismatchError(f"checkpoint field is missing: {key}")
    if payload["parameter_count"] != expected_parameter_count:
        raise CheckpointMismatchError(
            "model parameter count mismatch: expected "
            f"{expected_parameter_count}, got {payload['parameter_count']}"
        )
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (TypeError, RuntimeError) as exc:
        raise CheckpointMismatchError(
            f"checkpoint model state is incompatible: {exc}"
        ) from exc
    return {
        "path": str(path),
        "step": int(payload["step"]),
        "parameter_count": int(payload["parameter_count"]),
        "git_hash": str(payload["git_hash"]),
        "training_config": payload["config"],
    }


__all__ = [
    "AggregateMetric",
    "MethodMetricAccumulator",
    "MetricAccumulator",
    "POINT_TO_PLANE_NOT_AVAILABLE",
    "PSEUDO_GT_LABEL",
    "aggregate_metric_results",
    "comparison_from_aggregates",
    "compute_sample_metrics",
    "load_model_for_evaluation",
    "upsample_ffs_inputs_to_hr",
]
