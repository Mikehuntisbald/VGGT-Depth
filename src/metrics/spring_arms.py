"""Metrics and diagnostics for the Spring seven-arm screening protocol.

The functions here intentionally accept NumPy-like arrays so they can be used
by a lightweight post-processing runner without importing the training stack.
All disparity values are in Spring's full-HD pixel units (the 4K ``.dsp5``
files are sampled with ``[::2, ::2]`` and are *not* divided by two).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _valid_gt(gt: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & (gt > 0)


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    selected = value[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return float("nan")
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if selected.size else float("nan")


def disparity_metrics(
    prediction: Any,
    ground_truth: Any,
    *,
    valid_mask: Any | None = None,
    detail_mask: Any | None = None,
    match_mask: Any | None = None,
    ffs_trusted_mask: Any | None = None,
    ffs_prediction: Any | None = None,
    boundary_mask: Any | None = None,
) -> dict[str, float | int]:
    """Compute the required per-frame Spring disparity metrics."""

    pred = _array(prediction)
    gt = _array(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/ground_truth shape mismatch: {pred.shape} vs {gt.shape}")
    # ``valid`` is the support used by EPE/bad-pixel metrics: a prediction
    # must be finite, while the GT must be a positive finite disparity.  Keep
    # the GT support separate below for completion metrics, because an
    # invalid/zero prediction is a *failed completion*, not a pixel to remove
    # from the denominator.
    gt_support = _valid_gt(gt)
    if valid_mask is not None:
        gt_support &= _array(valid_mask, dtype=bool)
    valid = gt_support & np.isfinite(pred)
    error = np.abs(pred - gt)
    detail = np.ones_like(valid, dtype=bool) if detail_mask is None else _array(detail_mask, dtype=bool)
    matched = np.ones_like(valid, dtype=bool) if match_mask is None else _array(match_mask, dtype=bool)
    unmatched = ~matched
    boundary = np.zeros_like(valid, dtype=bool) if boundary_mask is None else _array(boundary_mask, dtype=bool)

    def epe(mask: np.ndarray) -> float:
        return _masked_mean(error, valid & mask)

    def bad(mask: np.ndarray, threshold: float) -> float:
        selected = valid & mask
        if not selected.any():
            return float("nan")
        return float((error[selected] > threshold).mean())

    # The unmatched domain is defined entirely by valid GT.  In particular,
    # zero, negative, NaN, and +/-Inf predictions remain in the denominator
    # and count as unsuccessful completion.  This prevents a model from
    # inflating completion@N by emitting invalid values in FFS holes.
    completion_domain = gt_support & unmatched
    completion_valid = completion_domain & np.isfinite(pred) & (pred > 0)
    completion_error = np.abs(pred - gt)
    completion_count = int(completion_domain.sum())

    def completion(threshold: float) -> float:
        if completion_count == 0:
            return float("nan")
        success = completion_valid & (completion_error <= threshold)
        return float(success.sum() / completion_count)

    out: dict[str, float | int] = {
        "overall_epe": epe(np.ones_like(valid, dtype=bool)),
        "overall_1px": bad(np.ones_like(valid, dtype=bool), 1.0),
        "high_detail_epe": epe(detail),
        "high_detail_1px": bad(detail, 1.0),
        "low_detail_epe": epe(~detail),
        "low_detail_1px": bad(~detail, 1.0),
        "matched_epe": epe(matched),
        "matched_1px": bad(matched, 1.0),
        "unmatched_completion_1px": completion(1.0),
        "unmatched_completion_2px": completion(2.0),
        "boundary_epe": epe(boundary) if boundary.any() else float("nan"),
        "valid_count": int(valid.sum()),
        "unmatched_count": completion_count,
    }
    if ffs_trusted_mask is not None and ffs_prediction is not None:
        ffs = _array(ffs_prediction)
        # FFS trusted-measurement error evaluates the frozen observation on
        # its own support; it must not disappear merely because the student
        # prediction at that pixel is NaN.
        trusted = gt_support & _array(ffs_trusted_mask, dtype=bool) & np.isfinite(ffs)
        out["ffs_trusted_measurement_error"] = _masked_mean(np.abs(ffs - gt), trusted)
        out["ffs_trusted_count"] = int(trusted.sum())
    else:
        out["ffs_trusted_measurement_error"] = float("nan")
        out["ffs_trusted_count"] = 0
    finite = np.isfinite(pred)
    out["negative_rate"] = float((finite & (pred < 0)).mean())
    out["zero_rate"] = float((finite & (pred == 0)).mean())
    out["invalid_rate"] = float((~finite).mean())
    return out


def temporal_residual_metrics(
    prediction_residual: Any,
    reference_residual: Any,
    *,
    valid_mask: Any,
    rigid_mask: Any | None = None,
) -> dict[str, float | int]:
    """Compute rigid/non-rigid temporal residual error."""

    pred = _array(prediction_residual)
    ref = _array(reference_residual)
    mask = _array(valid_mask, dtype=bool) & np.isfinite(pred) & np.isfinite(ref)
    error = np.abs(pred - ref)
    rigid = np.ones_like(mask, dtype=bool) if rigid_mask is None else _array(rigid_mask, dtype=bool)
    return {
        "rigid_temporal_residual_error": _masked_mean(error, mask & rigid),
        "non_rigid_temporal_residual_error": _masked_mean(error, mask & ~rigid),
        "temporal_residual_valid_count": int(mask.sum()),
    }


def topk_diagnostics(
    *,
    age: Any,
    phase: Any,
    depth: Any,
    weights: Any,
    valid: Any,
    age2_available: Any | None = None,
    fractional_phase_bucket_gain: Mapping[str, Sequence[float]] | None = None,
    camera_motion_bucket_gain: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float | int | dict[str, float]]:
    """Summarize candidate complementarity fields from the top-K transport tap.

    Expected candidate layout is ``[K,H,W]`` (a leading batch dimension is
    accepted and removed). ``phase`` may be ``[K,2,H,W]`` or ``[K,H,W]``.
    ``age2_available`` optionally supplies the pre-truncation age-2
    denominator; when omitted, valid target pixels are used.
    """

    def squeeze(value: Any) -> np.ndarray:
        arr = _array(value)
        if arr.ndim >= 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    ages = squeeze(age)
    phases = squeeze(phase)
    depths = squeeze(depth)
    ws = squeeze(weights)
    vm = squeeze(valid).astype(bool)
    if ages.shape != ws.shape or ages.shape != vm.shape or depths.shape != ws.shape:
        raise ValueError("top-K scalar fields must share [K,H,W] shape")
    if phases.ndim == 4 and phases.shape[1] == 2:
        # Preserve the two sampling-cell components.  Circular variance is
        # computed per component, matching the transport diagnostic contract;
        # taking an L2 magnitude would make wrap-around phases look different
        # even when they refer to the same bilinear cell.
        phase_components = phases
    elif phases.shape == ws.shape:
        phase_components = phases[:, None, ...]
    else:
        raise ValueError("phase must have shape [K,H,W] or [K,2,H,W]")
    finite = vm & np.isfinite(ws) & np.isfinite(ages) & np.isfinite(depths)
    finite &= np.isfinite(phase_components).all(axis=1)
    if not finite.any():
        result: dict[str, float | int | dict[str, float]] = {
            "age_2_survival_rate": float("nan"),
            "unique_age_fraction": float("nan"),
            "phase_variance": float("nan"),
            "candidate_depth_spread": float("nan"),
            "attention_entropy": float("nan"),
            "topk_valid_count": 0,
        }
    else:
        # A target pixel is valid when at least one complete candidate is
        # valid.  Fractions below use this pixel-level denominator rather than
        # H*W, so padded/invalid regions cannot dilute the diagnostic.
        pixel_valid = finite.any(axis=0)
        rounded_age = np.rint(np.where(np.isfinite(ages), ages, 0.0)).astype(
            np.int64, copy=False
        )

        # A pixel survives age-2 if any retained candidate has age >= 2.  The
        # optional availability mask lets callers use the pre-truncation
        # front-layer population as denominator, as in the v3.1 contract.
        age2 = np.any(finite & (rounded_age >= 2), axis=0)
        if age2_available is None:
            age2_domain = pixel_valid
        else:
            age2_domain = squeeze(age2_available).astype(bool)
            # Common tensor exports carry a singleton batch/channel axis.
            # Remove those axes without accepting an ambiguous non-singleton
            # shape.
            while age2_domain.ndim > 2 and 1 in age2_domain.shape[:2]:
                if age2_domain.shape[0] == 1:
                    age2_domain = age2_domain[0]
                elif age2_domain.shape[1] == 1:
                    age2_domain = age2_domain[:, 0]
            if age2_domain.shape != pixel_valid.shape:
                if age2_domain.shape == vm.shape:
                    age2_domain = age2_domain.any(axis=0)
                else:
                    raise ValueError(
                        "age2_available must have [H,W] or [K,H,W] shape"
                    )

        # Count distinct temporal ages (not merely the number of populated
        # slots).  Two candidates from the same age are not evidence of
        # multi-frame complementarity.
        unique_count = np.zeros_like(pixel_valid, dtype=np.int64)
        for rank in range(ages.shape[0]):
            current = finite[rank] & (rounded_age[rank] >= 1)
            if rank:
                duplicate = np.any(
                    finite[:rank]
                    & (rounded_age[:rank] == rounded_age[rank][None, ...]),
                    axis=0,
                )
                current &= ~duplicate
            unique_count += current.astype(np.int64)
        unique_age = unique_count > 1

        # Normalize non-negative attention weights per target pixel for a
        # conventional probability entropy in nats.  Pixels with no positive
        # weight are excluded from the entropy denominator.
        positive_weights = np.where(
            finite, np.maximum(np.nan_to_num(ws, nan=0.0), 0.0), 0.0
        )
        denom = np.sum(positive_weights, axis=0, keepdims=True)
        probs = np.divide(
            positive_weights,
            denom,
            out=np.zeros_like(positive_weights),
            where=denom > 0,
        )
        entropy = -np.sum(
            np.where(
                probs > 0,
                probs * np.log(np.maximum(probs, 1e-12)),
                0.0,
            ),
            axis=0,
        )

        # Circular sampling-cell variance, averaged over available phase
        # components and target pixels.
        phase_variances: list[np.ndarray] = []
        phase_domains: list[np.ndarray] = []
        for component in range(phase_components.shape[1]):
            component_values = phase_components[:, component]
            component_valid = finite & np.isfinite(component_values)
            count = component_valid.sum(axis=0)
            safe_count = np.maximum(count, 1)
            wrapped = np.remainder(component_values, 1.0)
            angle = wrapped * (2.0 * np.pi)
            mean_cos = np.where(
                count > 0,
                np.sum(np.where(component_valid, np.cos(angle), 0.0), axis=0)
                / safe_count,
                0.0,
            )
            mean_sin = np.where(
                count > 0,
                np.sum(np.where(component_valid, np.sin(angle), 0.0), axis=0)
                / safe_count,
                0.0,
            )
            phase_variances.append(
                1.0 - np.sqrt(np.clip(mean_cos * mean_cos + mean_sin * mean_sin, 0.0, 1.0))
            )
            phase_domains.append(count > 0)
        phase_stack = np.stack(phase_variances, axis=0)
        phase_domain_stack = np.stack(phase_domains, axis=0)

        # Per-pixel candidate depth range, then mean over valid target pixels.
        depth_min = np.min(np.where(finite, depths, np.inf), axis=0)
        depth_max = np.max(np.where(finite, depths, -np.inf), axis=0)
        depth_spread = depth_max - depth_min
        depth_spread_domain = pixel_valid & np.isfinite(depth_spread)

        entropy_domain = pixel_valid & (denom[0] > 0) & np.isfinite(entropy)
        age2_domain = age2_domain & np.isfinite(age2)
        result = {
            "age_2_survival_rate": (
                float(age2[age2_domain].mean()) if age2_domain.any() else float("nan")
            ),
            "unique_age_fraction": (
                float(unique_age[pixel_valid].mean()) if pixel_valid.any() else float("nan")
            ),
            "phase_variance": (
                float(phase_stack[phase_domain_stack].mean())
                if phase_domain_stack.any()
                else float("nan")
            ),
            "candidate_depth_spread": (
                float(depth_spread[depth_spread_domain].mean())
                if depth_spread_domain.any()
                else float("nan")
            ),
            "attention_entropy": (
                float(entropy[entropy_domain].mean())
                if entropy_domain.any()
                else float("nan")
            ),
            "topk_valid_count": int(finite.sum()),
        }
    result["gain_by_fractional_phase_bucket"] = _bucket_mean(fractional_phase_bucket_gain)
    result["gain_by_camera_motion_bucket"] = _bucket_mean(camera_motion_bucket_gain)
    return result


def _bucket_mean(value: Mapping[str, Sequence[float]] | None) -> dict[str, float]:
    if not value:
        return {}
    result: dict[str, float] = {}
    for key, values in value.items():
        arr = _array(values)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            result[str(key)] = float(finite.mean())
    return result


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame metric rows by valid-count weighting where possible."""

    if not rows:
        raise ValueError("cannot aggregate an empty metric row sequence")
    keys = sorted({key for row in rows for key in row if isinstance(row[key], (int, float))})
    result: dict[str, Any] = {"frames": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if finite.size else float("nan")
    return result


__all__ = [
    "aggregate_metric_rows",
    "disparity_metrics",
    "temporal_residual_metrics",
    "topk_diagnostics",
]
