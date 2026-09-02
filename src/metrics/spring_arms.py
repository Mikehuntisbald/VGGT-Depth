"""Metrics and diagnostics for the Spring seven-arm screening protocol.

The functions here intentionally accept NumPy-like arrays so they can be used
by a lightweight post-processing runner without importing the training stack.
All disparity values are in Spring's full-HD pixel units (the 4K ``.dsp5``
files are sampled with ``[::2, ::2]`` and are *not* divided by two).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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


def _spring_boundary_mask(
    ground_truth: np.ndarray,
    *,
    gradient_threshold_px: float = 1.0,
    radius_px: int = 1,
) -> np.ndarray:
    """Derive a GT-only disparity boundary band on a NumPy image grid.

    This mirrors :func:`metrics.boundary.disparity_boundary_mask` without
    importing Torch into the lightweight native Spring post-processing path.
    Edges touching invalid/zero GT are deliberately excluded.
    """

    gt = _array(ground_truth)
    if gt.ndim != 2:
        raise ValueError(f"ground_truth must be 2-D, got {gt.shape}")
    if not math.isfinite(float(gradient_threshold_px)) or gradient_threshold_px < 0:
        raise ValueError("gradient_threshold_px must be finite and >= 0")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("radius_px must be a non-negative integer")
    valid = _valid_gt(gt)
    boundary = np.zeros_like(valid, dtype=bool)
    if gt.shape[1] > 1:
        edge = (
            np.abs(gt[:, 1:] - gt[:, :-1]) >= float(gradient_threshold_px)
        ) & valid[:, 1:] & valid[:, :-1]
        boundary[:, 1:] |= edge
        boundary[:, :-1] |= edge
    if gt.shape[0] > 1:
        edge = (
            np.abs(gt[1:, :] - gt[:-1, :]) >= float(gradient_threshold_px)
        ) & valid[1:, :] & valid[:-1, :]
        boundary[1:, :] |= edge
        boundary[:-1, :] |= edge
    if radius_px == 0 or not boundary.any():
        return boundary
    # Chebyshev dilation, equivalent to the max-pool implementation used by
    # the Torch metric.  The explicit shifts avoid a SciPy dependency.
    dilated = np.zeros_like(boundary, dtype=bool)
    height, width = boundary.shape
    for dy in range(-radius_px, radius_px + 1):
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        for dx in range(-radius_px, radius_px + 1):
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            dilated[dst_y0:dst_y1, dst_x0:dst_x1] |= boundary[
                src_y0:src_y1, src_x0:src_x1
            ]
    return dilated


def _masked_stats(
    value: np.ndarray,
    mask: np.ndarray,
    *,
    event_threshold: float | None = None,
) -> tuple[float, float, int]:
    """Return ``(mean/rate, numerator, count)`` for one metric domain.

    Keeping the numerator and denominator beside every native Spring scalar
    prevents a second-stage reducer from averaging per-frame means.  The
    legacy scalar remains unchanged; consumers that understand the contract
    should use the ``*_numerator``/``*_count`` pair.
    """

    selected = np.asarray(mask, dtype=bool).copy()
    if event_threshold is None:
        selected &= np.isfinite(value)
        count = int(selected.sum())
        if count == 0:
            return float("nan"), float("nan"), 0
        numerator = float(np.asarray(value, dtype=np.float64)[selected].sum())
    else:
        selected &= np.isfinite(value)
        count = int(selected.sum())
        if count == 0:
            return float("nan"), float("nan"), 0
        numerator = float(
            (np.asarray(value, dtype=np.float64)[selected] > event_threshold).sum()
        )
    return numerator / count, numerator, count


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
        valid_mask_array = _array(valid_mask, dtype=bool)
        if valid_mask_array.shape != gt.shape:
            raise ValueError(
                f"valid_mask shape mismatch: {valid_mask_array.shape} vs {gt.shape}"
            )
        gt_support &= valid_mask_array
    valid = gt_support & np.isfinite(pred)
    error = np.abs(pred - gt)
    detail = (
        np.ones_like(valid, dtype=bool)
        if detail_mask is None
        else _array(detail_mask, dtype=bool)
    )
    matched = (
        np.ones_like(valid, dtype=bool)
        if match_mask is None
        else _array(match_mask, dtype=bool)
    )
    if detail.shape != valid.shape:
        raise ValueError(
            f"detail_mask shape mismatch: {detail.shape} vs {valid.shape}"
        )
    if matched.shape != valid.shape:
        raise ValueError(
            f"match_mask shape mismatch: {matched.shape} vs {valid.shape}"
        )
    unmatched = ~matched
    boundary = (
        np.zeros_like(valid, dtype=bool)
        if boundary_mask is None
        else _array(boundary_mask, dtype=bool)
    )
    if boundary.shape != valid.shape:
        raise ValueError(
            f"boundary_mask shape mismatch: {boundary.shape} vs {valid.shape}"
        )

    def epe_stats(mask: np.ndarray) -> tuple[float, float, int]:
        return _masked_stats(error, valid & mask)

    def bad_stats(mask: np.ndarray, threshold: float) -> tuple[float, float, int]:
        return _masked_stats(error, valid & mask, event_threshold=threshold)

    # The unmatched domain is defined entirely by valid GT.  In particular,
    # zero, negative, NaN, and +/-Inf predictions remain in the denominator
    # and count as unsuccessful completion.  This prevents a model from
    # inflating completion@N by emitting invalid values in FFS holes.
    completion_domain = gt_support & unmatched
    completion_valid = completion_domain & np.isfinite(pred) & (pred > 0)
    completion_error = np.abs(pred - gt)
    completion_count = int(completion_domain.sum())

    def completion_stats(threshold: float) -> tuple[float, float, int]:
        if completion_count == 0:
            return float("nan"), float("nan"), 0
        success = completion_valid & (completion_error <= threshold)
        numerator = float(success.sum())
        return numerator / completion_count, numerator, completion_count

    overall_epe, overall_epe_num, overall_epe_count = epe_stats(
        np.ones_like(valid, dtype=bool)
    )
    overall_1px, overall_1px_num, overall_1px_count = bad_stats(
        np.ones_like(valid, dtype=bool), 1.0
    )
    high_detail_epe, high_detail_epe_num, high_detail_epe_count = epe_stats(detail)
    high_detail_1px, high_detail_1px_num, high_detail_1px_count = bad_stats(detail, 1.0)
    low_detail_epe, low_detail_epe_num, low_detail_epe_count = epe_stats(~detail)
    low_detail_1px, low_detail_1px_num, low_detail_1px_count = bad_stats(~detail, 1.0)
    matched_epe, matched_epe_num, matched_epe_count = epe_stats(matched)
    matched_1px, matched_1px_num, matched_1px_count = bad_stats(matched, 1.0)
    unmatched_1px, unmatched_1px_num, unmatched_1px_count = completion_stats(1.0)
    unmatched_2px, unmatched_2px_num, unmatched_2px_count = completion_stats(2.0)
    boundary_epe_value, boundary_epe_num, boundary_epe_count = epe_stats(boundary)

    out: dict[str, float | int] = {
        "overall_epe": overall_epe,
        "overall_epe_numerator": overall_epe_num,
        "overall_epe_count": overall_epe_count,
        "overall_1px": overall_1px,
        "overall_1px_numerator": overall_1px_num,
        "overall_1px_count": overall_1px_count,
        "high_detail_epe": high_detail_epe,
        "high_detail_epe_numerator": high_detail_epe_num,
        "high_detail_epe_count": high_detail_epe_count,
        "high_detail_1px": high_detail_1px,
        "high_detail_1px_numerator": high_detail_1px_num,
        "high_detail_1px_count": high_detail_1px_count,
        "low_detail_epe": low_detail_epe,
        "low_detail_epe_numerator": low_detail_epe_num,
        "low_detail_epe_count": low_detail_epe_count,
        "low_detail_1px": low_detail_1px,
        "low_detail_1px_numerator": low_detail_1px_num,
        "low_detail_1px_count": low_detail_1px_count,
        "matched_epe": matched_epe,
        "matched_epe_numerator": matched_epe_num,
        "matched_epe_count": matched_epe_count,
        "matched_1px": matched_1px,
        "matched_1px_numerator": matched_1px_num,
        "matched_1px_count": matched_1px_count,
        "unmatched_completion_1px": unmatched_1px,
        "unmatched_completion_1px_numerator": unmatched_1px_num,
        "unmatched_completion_1px_count": unmatched_1px_count,
        "unmatched_completion_2px": unmatched_2px,
        "unmatched_completion_2px_numerator": unmatched_2px_num,
        "unmatched_completion_2px_count": unmatched_2px_count,
        "boundary_epe": boundary_epe_value,
        "boundary_epe_numerator": boundary_epe_num,
        "boundary_epe_count": boundary_epe_count,
        "valid_count": int(valid.sum()),
        "unmatched_count": completion_count,
    }
    if ffs_trusted_mask is not None and ffs_prediction is not None:
        ffs = _array(ffs_prediction)
        # FFS trusted-measurement error evaluates the frozen observation on
        # its own support; it must not disappear merely because the student
        # prediction at that pixel is NaN.
        trusted_mask = _array(ffs_trusted_mask, dtype=bool)
        if trusted_mask.shape != gt.shape:
            raise ValueError(
                "ffs_trusted_mask shape mismatch: "
                f"{trusted_mask.shape} vs {gt.shape}"
            )
        if ffs.shape != gt.shape:
            raise ValueError(
                f"ffs_prediction shape mismatch: {ffs.shape} vs {gt.shape}"
            )
        trusted = gt_support & trusted_mask & np.isfinite(ffs)
        trusted_value, trusted_num, trusted_count = _masked_stats(
            np.abs(ffs - gt), trusted
        )
        out["ffs_trusted_measurement_error"] = trusted_value
        out["ffs_trusted_measurement_error_numerator"] = trusted_num
        out["ffs_trusted_measurement_error_count"] = trusted_count
        out["ffs_trusted_count"] = trusted_count
    else:
        out["ffs_trusted_measurement_error"] = float("nan")
        out["ffs_trusted_measurement_error_numerator"] = float("nan")
        out["ffs_trusted_measurement_error_count"] = 0
        out["ffs_trusted_count"] = 0
    finite = np.isfinite(pred)
    output_count = int(pred.size)
    negative_num = float((finite & (pred < 0)).sum())
    zero_num = float((finite & (pred == 0)).sum())
    invalid_num = float((~finite).sum())
    out["negative_rate"] = negative_num / output_count if output_count else float("nan")
    out["negative_rate_numerator"] = negative_num
    out["negative_rate_count"] = output_count
    out["zero_rate"] = zero_num / output_count if output_count else float("nan")
    out["zero_rate_numerator"] = zero_num
    out["zero_rate_count"] = output_count
    out["invalid_rate"] = invalid_num / output_count if output_count else float("nan")
    out["invalid_rate_numerator"] = invalid_num
    out["invalid_rate_count"] = output_count
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
    if pred.shape != ref.shape:
        raise ValueError("temporal residual prediction/reference shapes must match")
    valid_array = _array(valid_mask, dtype=bool)
    if valid_array.shape != pred.shape:
        raise ValueError("temporal residual valid_mask shape must match inputs")
    rigid = (
        np.ones_like(pred, dtype=bool)
        if rigid_mask is None
        else _array(rigid_mask, dtype=bool)
    )
    if rigid.shape != pred.shape:
        raise ValueError("temporal residual rigid_mask shape must match inputs")
    mask = valid_array & np.isfinite(pred) & np.isfinite(ref)
    error = np.abs(pred - ref)
    rigid_value, rigid_num, rigid_count = _masked_stats(error, mask & rigid)
    nonrigid_value, nonrigid_num, nonrigid_count = _masked_stats(
        error, mask & ~rigid
    )
    return {
        "rigid_temporal_residual_error": rigid_value,
        "rigid_temporal_residual_error_numerator": rigid_num,
        "rigid_temporal_residual_error_count": rigid_count,
        "non_rigid_temporal_residual_error": nonrigid_value,
        "non_rigid_temporal_residual_error_numerator": nonrigid_num,
        "non_rigid_temporal_residual_error_count": nonrigid_count,
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
    "SPRING_NATIVE_FIELDS",
    "SpringNativeMapError",
    "aggregate_spring_rows",
    "spring_disparity_row",
    "spring_map_bundle",
]


# ---------------------------------------------------------------------------
# Explicit native-Spring side-channel helpers.
# ---------------------------------------------------------------------------

SPRING_NATIVE_FIELDS = (
    "overall_epe",
    "overall_1px",
    "high_detail_epe",
    "high_detail_1px",
    "low_detail_epe",
    "matched_epe",
    "unmatched_completion_1px",
    "unmatched_completion_2px",
    "rigid_temporal_residual_error",
    "non_rigid_temporal_residual_error",
    "boundary_epe",
    "ffs_trusted_measurement_error",
    "negative_rate",
    "zero_rate",
    "invalid_rate",
)


class SpringNativeMapError(ValueError):
    """Raised when a required Spring auxiliary map is absent or malformed."""


def _spring_resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _spring_record_value(record: Mapping[str, Any], name: str) -> Any:
    if name in record:
        return record[name]
    source = record.get("source")
    if isinstance(source, Mapping) and name in source:
        return source[name]
    manifest_record = record.get("manifest_record")
    if isinstance(manifest_record, Mapping) and name in manifest_record:
        return manifest_record[name]
    return None


def _spring_map_path(
    record: Mapping[str, Any],
    name: str,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    gt_value = _spring_record_value(record, "gt_disparity_path")
    if not isinstance(gt_value, str) or not gt_value.strip():
        raise SpringNativeMapError("Spring manifest record has no gt_disparity_path")
    base = (
        None
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve().parent
    )
    gt_path = _spring_resolve_path(gt_value, base=base)
    sequence_root = gt_path.parent.parent
    try:
        frame_id = int(_spring_record_value(record, "frame_id"))
    except (TypeError, ValueError) as exc:
        raise SpringNativeMapError("Spring manifest record has invalid frame_id") from exc
    return sequence_root / "maps" / name / f"{name}_{frame_id:04d}.png"


def _spring_block_reduce_2x(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    if height % 2 or width % 2:
        raise SpringNativeMapError(
            f"4K Spring map shape must be even, got {(height, width)}"
        )
    return mask.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)) >= 0.5


def _spring_resize_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(target_hw):
        return mask.astype(bool, copy=False)
    height, width = target_hw
    image = Image.fromarray(mask.astype(np.uint8, copy=False) * 255, mode="L")
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _spring_read_mask(
    path: Path,
    *,
    target_hw: tuple[int, int],
    crop_hr_xywh: tuple[int, int, int, int] | None,
    kind: str,
) -> np.ndarray:
    if not path.is_file():
        raise SpringNativeMapError(f"required Spring map is missing: {path}")
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if kind == "match":
        if array.ndim != 3 or array.shape[-1] < 2:
            raise SpringNativeMapError(
                f"Spring match map must have at least two channels: {path}"
            )
        array = (array[..., 0] > 0) | (array[..., 1] > 0)
    else:
        if array.ndim == 3:
            array = array[..., 0]
        array = array > 0
    if array.ndim != 2:
        raise SpringNativeMapError(f"Spring map is not 2-D: {path}")
    # Detail/match maps are already image-resolution.  Rigid maps are 4K and
    # follow the official 2x2 majority reduction before any crop.
    if (
        kind == "rigid"
        and array.shape[0] >= 2 * target_hw[0]
        and array.shape[1] >= 2 * target_hw[1]
    ):
        array = _spring_block_reduce_2x(array)
    if crop_hr_xywh is not None and tuple(array.shape) != tuple(target_hw):
        x, y, width, height = crop_hr_xywh
        if y < 0 or x < 0 or y + height > array.shape[0] or x + width > array.shape[1]:
            raise SpringNativeMapError(
                f"Spring map crop {(x, y, width, height)} exceeds map shape {array.shape}: {path}"
            )
        array = array[y : y + height, x : x + width]
    if tuple(array.shape) != tuple(target_hw):
        array = _spring_resize_nearest(array, target_hw)
    return array.astype(bool, copy=False)


def spring_map_bundle(
    record: Mapping[str, Any],
    *,
    target_hw: tuple[int, int],
    manifest_path: str | Path | None = None,
    crop_hr_xywh: tuple[int, int, int, int] | None = None,
    require_rigid: bool = False,
) -> dict[str, Any]:
    """Load native Spring detail/match/(optional) rigid maps on target grid."""

    detail_name = "detailmap_disp1_left"
    match_name = "matchmap_disp1_left"
    detail_path = _spring_map_path(record, detail_name, manifest_path=manifest_path)
    match_path = _spring_map_path(record, match_name, manifest_path=manifest_path)
    detail = _spring_read_mask(
        detail_path,
        target_hw=target_hw,
        crop_hr_xywh=crop_hr_xywh,
        kind="detail",
    )
    matched = _spring_read_mask(
        match_path,
        target_hw=target_hw,
        crop_hr_xywh=crop_hr_xywh,
        kind="match",
    )
    rigid_name = "rigidmap_BW_left"
    rigid_path = _spring_map_path(record, rigid_name, manifest_path=manifest_path)
    if rigid_path.is_file():
        rigid = _spring_read_mask(
            rigid_path,
            target_hw=target_hw,
            crop_hr_xywh=crop_hr_xywh,
            kind="rigid",
        )
    elif require_rigid:
        raise SpringNativeMapError(f"required Spring rigid map is missing: {rigid_path}")
    else:
        rigid = None
    return {
        "detail": detail,
        "matched": matched,
        "rigid": rigid,
        "paths": {
            "detail": str(detail_path),
            "match": str(match_path),
            "rigid": str(rigid_path),
        },
    }


def spring_disparity_row(
    prediction: Any,
    ground_truth: Any,
    *,
    detail_mask: Any,
    match_mask: Any,
    ffs_trusted_mask: Any,
    ffs_prediction: Any,
    boundary_epe: float | None = None,
    boundary_mask: Any | None = None,
    boundary_gradient_threshold_px: float = 1.0,
    boundary_radius_px: int = 1,
) -> dict[str, Any]:
    """Compute one native Spring row using the native-GT metric contract.

    ``boundary_mask`` is optional for callers that already loaded the exact
    Spring mask.  When omitted, it is derived directly from ``ground_truth``;
    this keeps the native side channel independent from the FFS pseudo-GT
    boundary used by the canonical evaluator.  ``boundary_epe`` is retained
    as a deprecated compatibility argument but is intentionally ignored when
    a GT-derived mask is available (and never passed by the production
    evaluator).
    """

    if boundary_mask is None:
        boundary_mask = _spring_boundary_mask(
            _array(ground_truth),
            gradient_threshold_px=boundary_gradient_threshold_px,
            radius_px=boundary_radius_px,
        )

    result = disparity_metrics(
        prediction,
        ground_truth,
        detail_mask=detail_mask,
        match_mask=match_mask,
        ffs_trusted_mask=ffs_trusted_mask,
        ffs_prediction=ffs_prediction,
        boundary_mask=boundary_mask,
    )
    # Do not let a legacy pseudo-GT scalar overwrite the native GT-derived
    # value.  Retain a provenance marker so old callers can see that their
    # override was ignored rather than silently changing the metric domain.
    result["boundary_source"] = "spring_gt_disparity_boundary_mask"
    if boundary_epe is not None:
        result["boundary_override_ignored"] = True
    return result


def aggregate_spring_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce native rows without averaging per-frame means.

    New rows carry an explicit ``<field>_numerator``/``<field>_count`` pair
    for every metric domain.  Legacy rows (and hand-written fixtures) are
    still accepted with a frame-mean fallback, but the aggregation receipt
    makes that fallback visible so it cannot be mistaken for a global
    numerator/count reduction.
    """

    if not rows:
        raise ValueError("cannot aggregate an empty Spring metric row sequence")
    result: dict[str, Any] = {
        "frames": len(rows),
        "aggregation": (
            "global_numerator_count_pixel_weighted_with_legacy_frame_mean_fallback"
        ),
    }
    used_fallback = False
    for name in SPRING_NATIVE_FIELDS:
        numerator_name = f"{name}_numerator"
        count_name = f"{name}_count"
        numerator = 0.0
        count = 0
        pair_available = False
        pair_invalid = False
        missing_pair = False
        for row in rows:
            raw_num = row.get(numerator_name)
            raw_count = row.get(count_name)
            if raw_num is None and raw_count is None:
                missing_pair = True
                continue
            pair_available = True
            # Empty per-frame domains are expected for some Spring subsets
            # (for example a frame with no matched or boundary pixels).  They
            # contribute no numerator and must not poison a valid dataset
            # aggregate.  A positive count with a non-finite numerator,
            # however, is a real malformed/invalid metric and remains
            # fail-closed below.
            if (
                isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count == 0
                and (
                    raw_num is None
                    or (
                        isinstance(raw_num, (int, float))
                        and not isinstance(raw_num, bool)
                        and (
                            not math.isfinite(float(raw_num))
                            or float(raw_num) == 0.0
                        )
                    )
                )
            ):
                continue
            if (
                isinstance(raw_num, (int, float))
                and not isinstance(raw_num, bool)
                and math.isfinite(float(raw_num))
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count > 0
            ):
                numerator += float(raw_num)
                count += int(raw_count)
            else:
                pair_invalid = True
        if pair_available and not missing_pair:
            if pair_invalid or count <= 0:
                result[name] = None
                result[numerator_name] = None
                result[count_name] = count
            else:
                result[name] = numerator / count
                result[numerator_name] = numerator
                result[count_name] = count
            continue
        # Compatibility path for older sidecars that only stored frame means.
        # Do not silently omit legacy rows when a sidecar set mixes old and
        # new schemas; use the scalar fallback across all rows and mark it.
        used_fallback = True
        values = []
        for row in rows:
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value):
                    values.append(value)
        result[name] = None if not values else float(np.mean(values))
        result[numerator_name] = None
        result[count_name] = 0
    pixels = sum(int(row.get("image_pixel_count", 0) or 0) for row in rows)
    result["image_pixel_count"] = pixels
    for name in ("negative_rate", "zero_rate", "invalid_rate"):
        numerator = 0.0
        count = 0
        used_rate_fallback = False
        for row in rows:
            raw_num = row.get(f"{name}_numerator")
            raw_count = row.get(f"{name}_count")
            if (
                isinstance(raw_num, (int, float))
                and not isinstance(raw_num, bool)
                and math.isfinite(float(raw_num))
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count > 0
            ):
                numerator += float(raw_num)
                count += int(raw_count)
                continue
            pixels_for_row = int(row.get("image_pixel_count", 0) or 0)
            value = row.get(name)
            if (
                pixels_for_row > 0
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                numerator += float(value) * pixels_for_row
                count += pixels_for_row
                used_rate_fallback = True
        if count > 0:
            result[name] = numerator / count
            result[f"{name}_numerator"] = numerator
            result[f"{name}_count"] = count
            if used_rate_fallback:
                used_fallback = True
        else:
            result[name] = None
            result[f"{name}_numerator"] = None
            result[f"{name}_count"] = 0
    result["aggregation_fallback_used"] = used_fallback
    for name in ("valid_count", "unmatched_count", "ffs_trusted_count"):
        result[name] = int(sum(int(row.get(name, 0) or 0) for row in rows))
    return result
