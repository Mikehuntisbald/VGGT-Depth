#!/usr/bin/env python3
"""Derive one safe metric-geometry cache from matching VGGT and FFS records.

The raw backbone caches remain immutable.  This CPU-only stage validates their
identities and exact target-image lineage, recovers metric scale from the
calibrated stereo baseline, aligns the current-left depth prior to trusted FFS
disparity, and evaluates strict temporal-pose quality gates.

An invalid pose is retained only as diagnostics: the tensor exposed for
temporal warping is zero-filled and accompanied by ``temporal_pose_valid=False``.
The aligned static depth prior has its own validity bit and may remain usable
when only the photometric temporal-pose gate fails.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.stereo_calibration import (
    RECTIFIED_CALIBRATION_CONTRACT,
    RectifiedCalibrationIndex,
    RectifiedCalibrationRecord,
    calibration_window_sha256,
    load_rectified_calibration_sidecar,
)
from geometry.align_vggt import (
    align_vggt_depth_to_ffs_disparity,
    ffs_trusted_mask,
)
from geometry.pose_quality import (
    CURRENT_LEFT_VIEW_INDEX,
    PREVIOUS_LEFT_VIEW_INDEX,
    adjacent_left_photometric_reprojection,
    combine_pose_quality,
    depth_disparity_consistency,
    validate_raw_cache_pair,
)
from geometry.pose_scale import (
    estimate_baseline_metric_scale,
    scale_vggt_translations_and_depth,
)


DERIVED_SCHEMA_VERSION = 1
CALIBRATED_DERIVED_SCHEMA_VERSION = 2
CALIBRATED_DERIVED_COMPONENT = (
    "vggt-ffs-derived-geometry-calibrated-stereo-v2"
)
CALIBRATED_DERIVED_ALGORITHM = (
    "baseline_metric_scale+scale_only_alignment+strict_pose_quality+"
    "calibrated_stereo_constraint_v2"
)


@dataclass(frozen=True, slots=True)
class GeometryThresholds:
    """All explicit gates participating in derived-cache identity."""

    max_baseline_cv: float = 0.10
    max_stereo_rotation_error_deg: float = 5.0
    ffs_confidence_threshold: float = 0.8
    max_left_right_error_lr_px: float = 1.0
    min_alignment_pixels: int = 256
    alignment_huber_delta_hr_px: float = 1.0
    max_depth_weighted_mae_hr_px: float = 2.0
    max_depth_median_absolute_error_hr_px: float = 2.0
    # Diagnostic by default. Near-zero disparity makes pointwise relative
    # error ill-conditioned; pass a value explicitly for a strict ablation.
    max_depth_median_relative_error: float | None = None
    min_photometric_samples: int = 1024
    min_photometric_valid_fraction: float = 0.50
    max_photometric_median_absolute_rgb: float = 0.12

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def validate(self) -> None:
        if self.min_alignment_pixels <= 0 or self.min_photometric_samples <= 0:
            raise ValueError("minimum sample counts must be positive")
        if not 0.0 <= self.ffs_confidence_threshold <= 1.0:
            raise ValueError("ffs_confidence_threshold must be in [0,1]")
        if not 0.0 <= self.min_photometric_valid_fraction <= 1.0:
            raise ValueError("min_photometric_valid_fraction must be in [0,1]")
        nonnegative = (
            self.max_baseline_cv,
            self.max_stereo_rotation_error_deg,
            self.max_depth_weighted_mae_hr_px,
            self.max_depth_median_absolute_error_hr_px,
            self.max_photometric_median_absolute_rgb,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("quality thresholds must be finite and non-negative")
        if self.max_depth_median_relative_error is not None and (
            not math.isfinite(self.max_depth_median_relative_error)
            or self.max_depth_median_relative_error < 0
        ):
            raise ValueError(
                "max_depth_median_relative_error must be finite/non-negative or None"
            )
        if (
            not math.isfinite(self.max_left_right_error_lr_px)
            or self.max_left_right_error_lr_px <= 0
            or not math.isfinite(self.alignment_huber_delta_hr_px)
            or self.alignment_huber_delta_hr_px <= 0
        ):
            raise ValueError("LR-error and Huber thresholds must be finite and positive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive metric VGGT geometry from one exact raw VGGT/FFS cache pair."
    )
    parser.add_argument("--vggt-cache", type=Path, required=True)
    parser.add_argument("--ffs-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Exact output .pt path")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Exact JSON receipt path (default: OUTPUT.receipt.json)",
    )
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--max-baseline-cv", type=float, default=0.10)
    parser.add_argument("--max-stereo-rotation-error-deg", type=float, default=5.0)
    parser.add_argument("--ffs-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-left-right-error-lr-px", type=float, default=1.0)
    parser.add_argument("--min-alignment-pixels", type=int, default=256)
    parser.add_argument("--alignment-huber-delta-hr-px", type=float, default=1.0)
    parser.add_argument("--max-depth-weighted-mae-hr-px", type=float, default=2.0)
    parser.add_argument(
        "--max-depth-median-absolute-error-hr-px", type=float, default=2.0
    )
    parser.add_argument(
        "--max-depth-median-relative-error",
        type=float,
        default=None,
        help="Optional strict gate; default records relative error as diagnostic only",
    )
    parser.add_argument("--min-photometric-samples", type=int, default=1024)
    parser.add_argument("--min-photometric-valid-fraction", type=float, default=0.50)
    parser.add_argument(
        "--max-photometric-median-absolute-rgb", type=float, default=0.12
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rectified-calibration-sidecar",
        type=Path,
        help=(
            "Opt in to calibrated-stereo-v2 using an immutable sidecar. "
            "Legacy derivation is unchanged when omitted."
        ),
    )
    parser.add_argument(
        "--rectified-calibration-receipt",
        type=Path,
        help="Explicit sidecar receipt (default: SIDECAR with .receipt.json suffix)",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _require_tensor(
    tensors: Mapping[str, Any], name: str, *, ndim: int | None = None
) -> torch.Tensor:
    value = tensors.get(name)
    if not isinstance(value, torch.Tensor):
        raise CacheMismatchError(f"required cache tensor {name!r} is missing")
    if ndim is not None and value.ndim != ndim:
        raise CacheMismatchError(
            f"cache tensor {name!r} must have ndim={ndim}, got {value.ndim}"
        )
    return value


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _validate_grid_and_calibration(
    vggt_payload: Mapping[str, Any], ffs_payload: Mapping[str, Any]
) -> None:
    """Ensure dense grids and calibrated coordinate transforms truly agree."""

    vggt_tensors = vggt_payload["tensors"]
    ffs_tensors = ffs_payload["tensors"]
    depth = _require_tensor(vggt_tensors, "vggt_depth_current_left_arbitrary", ndim=3)
    disparity = _require_tensor(ffs_tensors, "observation_disparity_hr_px", ndim=4)
    if disparity.shape[:2] != (1, 1) or depth.shape[0] != 1:
        raise CacheMismatchError("derived geometry requires singleton batch/channel caches")
    if tuple(depth.shape[-2:]) != tuple(disparity.shape[-2:]):
        raise CacheMismatchError(
            "VGGT/FFS dense grid mismatch: "
            f"depth={tuple(depth.shape)}, disparity={tuple(disparity.shape)}"
        )
    ffs_source = ffs_payload["metadata"]["source"]
    input_shape = ffs_source.get("ffs_input_shape_bchw")
    if input_shape != [1, 3, depth.shape[-2], depth.shape[-1]]:
        raise CacheMismatchError(
            f"FFS input shape does not match dense grid: {input_shape!r}"
        )
    ffs_config = ffs_payload["metadata"].get("config", {})
    if ffs_config.get("scale") != 2 or not ffs_config.get("right_left_check"):
        raise CacheMismatchError(
            "M2 requires x2 FFS observation cache with right-left check enabled"
        )

    intrinsics_original = _require_tensor(
        vggt_tensors, "calibrated_intrinsics_original_px", ndim=3
    )
    intrinsics_model = _require_tensor(
        vggt_tensors, "calibrated_intrinsics_model_px", ndim=3
    )
    original_to_model = _require_tensor(
        vggt_tensors, "original_to_model_transform", ndim=3
    )
    if intrinsics_original.shape != (10, 3, 3) or intrinsics_model.shape != (10, 3, 3):
        raise CacheMismatchError("VGGT calibrated intrinsics must be [10,3,3]")
    expected_model = original_to_model.float() @ intrinsics_original.float()
    if not torch.allclose(expected_model, intrinsics_model.float(), atol=1e-3, rtol=1e-5):
        raise CacheMismatchError(
            "recorded original/model intrinsics and transforms are inconsistent"
        )


def _validate_live_photometric_sources(linkage: Mapping[str, Any]) -> None:
    for label in ("previous_left", "current_left"):
        source = linkage[label]
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{label} image is missing: {path}")
        digest = sha256_file(path)
        if digest != source["sha256"]:
            raise CacheMismatchError(
                f"{label} live image SHA mismatch: expected {source['sha256']}, got {digest}"
            )


def _source_metadata(
    *,
    vggt_cache: Path,
    ffs_cache: Path,
    vggt_cache_sha256: str,
    ffs_cache_sha256: str,
    linkage: Mapping[str, Any],
    rectified_calibration_index: RectifiedCalibrationIndex | None = None,
    rectified_calibration_window: Sequence[RectifiedCalibrationRecord] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "vggt_cache_path": str(vggt_cache.resolve()),
        "vggt_cache_sha256": vggt_cache_sha256,
        "ffs_cache_path": str(ffs_cache.resolve()),
        "ffs_cache_sha256": ffs_cache_sha256,
        "linkage": dict(linkage),
    }
    if (rectified_calibration_index is None) != (
        rectified_calibration_window is None
    ):
        raise ValueError(
            "calibration index and window must be supplied together or omitted"
        )
    if rectified_calibration_index is not None:
        assert rectified_calibration_window is not None
        result["rectified_stereo_calibration"] = {
            "component": CALIBRATED_DERIVED_COMPONENT,
            "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
            "sidecar_path": str(rectified_calibration_index.sidecar_path),
            "sidecar_sha256": rectified_calibration_index.sidecar_sha256,
            "receipt_path": str(rectified_calibration_index.receipt_path),
            "receipt_sha256": rectified_calibration_index.receipt_sha256,
            "pixel_audit_path": str(rectified_calibration_index.pixel_audit_path),
            "pixel_audit_sha256": rectified_calibration_index.pixel_audit_sha256,
            "window_sha256": calibration_window_sha256(
                rectified_calibration_window
            ),
            "ordered_record_sha256": [
                record.calibration_record_sha256
                for record in rectified_calibration_window
            ],
        }
    return result


def _homogeneous_camera_from_world(extrinsic: torch.Tensor) -> torch.Tensor:
    if extrinsic.shape != (3, 4):
        raise ValueError("camera-from-world extrinsic must have shape [3,4]")
    result = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)
    result[:3] = extrinsic
    return result


def constrain_metric_stereo_extrinsics(
    extrinsics_camera_from_world_metric: torch.Tensor,
    calibration_window: Sequence[RectifiedCalibrationRecord],
    *,
    pose_valid: bool,
) -> torch.Tensor:
    """Compose exact calibrated right poses from the five VGGT left poses.

    The input remains the raw metric-scaled VGGT estimate and continues to own
    all pre-constraint quality gates.  This function changes only the exposed
    right pose after those gates pass; it never turns a rejected pose into a
    valid one.
    """

    if extrinsics_camera_from_world_metric.shape != (10, 3, 4):
        raise ValueError("metric VGGT extrinsics must have shape [10,3,4]")
    if len(calibration_window) != 5:
        raise ValueError("calibrated stereo constraint requires five records")
    if not pose_valid:
        return torch.zeros_like(extrinsics_camera_from_world_metric)
    if not bool(torch.isfinite(extrinsics_camera_from_world_metric).all()):
        raise ValueError("metric VGGT extrinsics contain NaN or infinity")
    constrained = extrinsics_camera_from_world_metric.clone()
    for pair_index, calibration in enumerate(calibration_window):
        left_index = 2 * pair_index
        right_index = left_index + 1
        transform_right_left = calibration.as_tensor(
            dtype=constrained.dtype, device=constrained.device
        )
        if transform_right_left.shape != (4, 4) or not bool(
            torch.isfinite(transform_right_left).all()
        ):
            raise ValueError("rectified stereo transform must be finite [4,4]")
        expected_rotation = torch.eye(
            3, dtype=constrained.dtype, device=constrained.device
        )
        if not torch.allclose(
            transform_right_left[:3, :3], expected_rotation, atol=1e-7, rtol=0.0
        ):
            raise ValueError("runtime rectified stereo rotation must be identity")
        right = transform_right_left @ _homogeneous_camera_from_world(
            constrained[left_index]
        )
        constrained[right_index] = right[:3]
    if not bool(torch.isfinite(constrained).all()):
        raise RuntimeError("calibrated stereo composition became non-finite")
    return constrained


def _validate_calibration_window_against_raw_source(
    vggt_payload: Mapping[str, Any],
    calibration_window: Sequence[RectifiedCalibrationRecord],
) -> None:
    if len(calibration_window) != 5:
        raise CacheMismatchError("rectified calibration window must contain five rows")
    metadata = vggt_payload.get("metadata")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    records = source.get("manifest_records") if isinstance(source, Mapping) else None
    if not isinstance(records, list) or len(records) != 5:
        raise CacheMismatchError("raw VGGT source has no five-row manifest window")
    for source_record, calibration in zip(records, calibration_window, strict=True):
        if not isinstance(source_record, Mapping):
            raise CacheMismatchError("raw VGGT manifest record is malformed")
        if calibration.source_record_sha256 != canonical_json_sha256(
            dict(source_record)
        ):
            raise CacheMismatchError("rectified calibration record/source hash mismatch")
        if (
            calibration.sequence_id != source_record.get("sequence_id")
            or calibration.frame_id != source_record.get("frame_id")
            or calibration.timestamp != source_record.get("timestamp")
        ):
            raise CacheMismatchError("rectified calibration record/source identity mismatch")
        baseline = source_record.get("baseline_m")
        if (
            isinstance(baseline, bool)
            or not isinstance(baseline, (int, float))
            or not math.isfinite(float(baseline))
            or float(baseline) <= 0
        ):
            raise CacheMismatchError("raw source baseline is malformed")
        transform = calibration.as_tensor(dtype=torch.float64)
        if not math.isclose(
            float(-transform[0, 3]), float(baseline), abs_tol=1e-9, rel_tol=0.0
        ):
            raise CacheMismatchError("rectified calibration/source baseline mismatch")


def derive_geometry(
    vggt_payload: Mapping[str, Any],
    ffs_payload: Mapping[str, Any],
    *,
    thresholds: GeometryThresholds,
    cache_dtype: torch.dtype = torch.float32,
    rectified_calibration_window: Sequence[RectifiedCalibrationRecord] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Derive metric tensors and strict pose-quality metadata in memory."""

    thresholds.validate()
    if cache_dtype not in {torch.float16, torch.float32}:
        raise ValueError("cache_dtype must be float16 or float32")
    linkage = validate_raw_cache_pair(vggt_payload, ffs_payload)
    if rectified_calibration_window is not None:
        _validate_calibration_window_against_raw_source(
            vggt_payload, rectified_calibration_window
        )
    _validate_grid_and_calibration(vggt_payload, ffs_payload)
    _validate_live_photometric_sources(linkage)
    vggt = vggt_payload["tensors"]
    ffs = ffs_payload["tensors"]

    extrinsics_arbitrary = _require_tensor(
        vggt, "vggt_extrinsics_camera_from_world", ndim=3
    ).float()
    depth_arbitrary = _require_tensor(
        vggt, "vggt_depth_current_left_arbitrary", ndim=3
    ).float()
    baselines_m = _require_tensor(vggt, "stereo_baseline_m_by_pair", ndim=1).float()
    target_baseline_m = float(linkage["target_manifest_record"]["baseline_m"])
    if baselines_m.numel() != 5 or not bool(torch.isfinite(baselines_m).all()):
        raise CacheMismatchError("stereo_baseline_m_by_pair must be five finite values")
    if not torch.allclose(
        baselines_m,
        torch.full_like(baselines_m, target_baseline_m),
        atol=1e-7,
        rtol=1e-6,
    ):
        raise CacheMismatchError("calibrated baseline changed within the raw VGGT window")

    baseline_scale = estimate_baseline_metric_scale(
        extrinsics_arbitrary,
        target_baseline_m,
        calibrated_rotation_right_from_left=torch.eye(3),
        expected_pairs=5,
        max_baseline_cv=thresholds.max_baseline_cv,
        max_stereo_rotation_error_deg=thresholds.max_stereo_rotation_error_deg,
    )
    metric_geometry_available = baseline_scale.valid
    if metric_geometry_available:
        extrinsics_metric_diagnostic, depth_metric_m = scale_vggt_translations_and_depth(
            extrinsics_arbitrary,
            depth_arbitrary,
            baseline_scale.alpha_m_per_vggt_unit,
        )
        depth_metric_valid = torch.isfinite(depth_metric_m) & (depth_metric_m > 0)
    else:
        extrinsics_metric_diagnostic = torch.zeros_like(extrinsics_arbitrary)
        depth_metric_m = torch.zeros_like(depth_arbitrary)
        depth_metric_valid = torch.zeros_like(depth_arbitrary, dtype=torch.bool)

    disparity_ffs_hr_px = _require_tensor(
        ffs, "observation_disparity_hr_px", ndim=4
    )[0].float()
    confidence_ffs = _require_tensor(ffs, "observation_confidence", ndim=4)[0].float()
    left_right_error_lr_px = _require_tensor(
        ffs, "observation_left_right_error_lr_px", ndim=4
    )[0].float()
    trusted = ffs_trusted_mask(
        disparity_ffs_hr_px,
        confidence_ffs,
        left_right_error_lr_px,
        confidence_threshold=thresholds.ffs_confidence_threshold,
        max_left_right_error_px=thresholds.max_left_right_error_lr_px,
    )
    cached_trusted = _require_tensor(ffs, "observation_trusted_mask", ndim=4)[0].bool()
    # M1's stored mask is an upstream convenience mask.  Recompute the M2
    # contract because it additionally and explicitly requires disparity > 0.
    # The stricter set may remove non-positive pixels, but it may never invent
    # a pixel that the raw cache marked untrusted.
    if bool((trusted & ~cached_trusted).any().item()):
        raise CacheMismatchError(
            "recomputed trusted FFS mask contains pixels rejected by raw cache"
        )

    confidence_vggt_unbounded = _require_tensor(
        vggt, "vggt_depth_conf_current_left_unbounded", ndim=3
    ).float()
    # Upstream defines confidence as 1 + exp(logit); this inverse transform is
    # exactly sigmoid(logit), a bounded proxy suitable for fusion weights.
    confidence_vggt_probability = torch.where(
        torch.isfinite(confidence_vggt_unbounded)
        & (confidence_vggt_unbounded > 1.0),
        1.0 - confidence_vggt_unbounded.reciprocal(),
        torch.zeros_like(confidence_vggt_unbounded),
    ).clamp(0.0, 1.0)
    alignment_weights = confidence_ffs * confidence_vggt_probability

    if metric_geometry_available:
        alignment = align_vggt_depth_to_ffs_disparity(
            disparity_ffs_hr_px,
            depth_metric_m,
            reliable_ffs_mask=trusted,
            weights=alignment_weights,
            min_reliable_pixels=thresholds.min_alignment_pixels,
            huber_delta_hr_px=thresholds.alignment_huber_delta_hr_px,
        )
        aligned_disparity = alignment.disparity_vggt_aligned_hr_px.float()
        aligned_valid = alignment.valid_mask.bool()
    else:
        alignment = None
        aligned_disparity = torch.zeros_like(depth_metric_m)
        aligned_valid = torch.zeros_like(depth_metric_m, dtype=torch.bool)

    depth_consistency = depth_disparity_consistency(
        disparity_ffs_hr_px,
        aligned_disparity,
        trusted_mask=trusted & aligned_valid,
        weights=alignment_weights,
        min_samples=thresholds.min_alignment_pixels,
        max_weighted_mae_hr_px=thresholds.max_depth_weighted_mae_hr_px,
        max_median_absolute_error_hr_px=(
            thresholds.max_depth_median_absolute_error_hr_px
        ),
        max_median_relative_error=thresholds.max_depth_median_relative_error,
    )

    intrinsics_model = _require_tensor(
        vggt, "calibrated_intrinsics_model_px", ndim=3
    ).float()
    model_to_original = _require_tensor(
        vggt, "model_to_original_transform", ndim=3
    ).float()
    if metric_geometry_available:
        photometric = adjacent_left_photometric_reprojection(
            depth_metric_m,
            extrinsic_previous_camera_from_world_metric=extrinsics_metric_diagnostic[
                PREVIOUS_LEFT_VIEW_INDEX
            ],
            extrinsic_current_camera_from_world_metric=extrinsics_metric_diagnostic[
                CURRENT_LEFT_VIEW_INDEX
            ],
            intrinsics_previous_model_px=intrinsics_model[PREVIOUS_LEFT_VIEW_INDEX],
            intrinsics_current_model_px=intrinsics_model[CURRENT_LEFT_VIEW_INDEX],
            previous_model_to_original_transform=model_to_original[
                PREVIOUS_LEFT_VIEW_INDEX
            ],
            current_model_to_original_transform=model_to_original[
                CURRENT_LEFT_VIEW_INDEX
            ],
            previous_rgb_path=linkage["previous_left"]["path"],
            current_rgb_path=linkage["current_left"]["path"],
            valid_depth_mask=depth_metric_valid,
            min_samples=thresholds.min_photometric_samples,
            min_valid_fraction=thresholds.min_photometric_valid_fraction,
            max_median_absolute_rgb_residual=(
                thresholds.max_photometric_median_absolute_rgb
            ),
        )
    else:
        photometric = adjacent_left_photometric_reprojection(
            None,
            extrinsic_previous_camera_from_world_metric=None,
            extrinsic_current_camera_from_world_metric=None,
            intrinsics_previous_model_px=None,
            intrinsics_current_model_px=None,
            previous_model_to_original_transform=None,
            current_model_to_original_transform=None,
            previous_rgb_path=None,
            current_rgb_path=None,
            min_samples=thresholds.min_photometric_samples,
            min_valid_fraction=thresholds.min_photometric_valid_fraction,
            max_median_absolute_rgb_residual=(
                thresholds.max_photometric_median_absolute_rgb
            ),
        )
    complete_quality = combine_pose_quality(
        baseline_scale, photometric, depth_consistency
    )
    temporal_extrinsics = (
        extrinsics_metric_diagnostic
        if complete_quality.pose_valid
        else torch.zeros_like(extrinsics_metric_diagnostic)
    )
    constrained_temporal_extrinsics: torch.Tensor | None = None
    target_rectified_transform: torch.Tensor | None = None
    if rectified_calibration_window is not None:
        constrained_temporal_extrinsics = constrain_metric_stereo_extrinsics(
            extrinsics_metric_diagnostic,
            rectified_calibration_window,
            pose_valid=complete_quality.pose_valid,
        )
        target_rectified_transform = rectified_calibration_window[-1].as_tensor(
            dtype=torch.float32
        )
    static_prior_valid = bool(
        alignment is not None
        and alignment.valid
        and depth_consistency.available
        and depth_consistency.valid
    )
    aligned_confidence = torch.where(
        aligned_valid,
        confidence_vggt_probability,
        torch.zeros_like(confidence_vggt_probability),
    )
    if not static_prior_valid:
        aligned_disparity = torch.zeros_like(aligned_disparity)
        aligned_valid = torch.zeros_like(aligned_valid)
        aligned_confidence = torch.zeros_like(aligned_confidence)

    float_tensors = {
        "vggt_extrinsics_camera_from_world_metric_diagnostic_only": (
            extrinsics_metric_diagnostic
        ),
        "vggt_extrinsics_camera_from_world_metric_temporal": temporal_extrinsics,
        "vggt_depth_current_left_metric_m": depth_metric_m,
        "vggt_disparity_current_left_aligned_hr_px": aligned_disparity,
        "vggt_aligned_confidence": aligned_confidence,
    }
    if constrained_temporal_extrinsics is not None:
        assert target_rectified_transform is not None
        float_tensors.update(
            {
                "T_right_rectified_from_left_rectified_m": (
                    target_rectified_transform
                ),
                "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained": (
                    constrained_temporal_extrinsics
                ),
            }
        )
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in float_tensors.items():
        value = tensor.to(cache_dtype)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"derived tensor {name!r} is non-finite at {cache_dtype}; use float32"
            )
        converted[name] = value
    converted.update(
        {
            "vggt_depth_metric_valid_mask": depth_metric_valid,
            "vggt_aligned_valid_mask": aligned_valid,
            "ffs_trusted_mask": trusted,
            "temporal_pose_valid": torch.tensor(complete_quality.pose_valid),
            "static_prior_valid": torch.tensor(static_prior_valid),
        }
    )
    if rectified_calibration_window is not None:
        converted["stereo_calibration_valid"] = torch.tensor(True)

    pose_quality = complete_quality.as_dict()
    pose_quality["baseline"] = {
        "alpha_m_per_vggt_unit": _finite_or_none(
            baseline_scale.alpha_m_per_vggt_unit
        ),
        "calibrated_baseline_m": baseline_scale.calibrated_baseline_m,
        "median_predicted_baseline_vggt_unit": _finite_or_none(
            baseline_scale.median_predicted_baseline_vggt_unit
        ),
        "baseline_coefficient_of_variation": _finite_or_none(
            baseline_scale.quality.baseline_coefficient_of_variation
        ),
        "stereo_rotation_error_max_deg": _finite_or_none(
            baseline_scale.quality.stereo_rotation_error_deg
        ),
        "stereo_rotation_error_median_deg": _finite_or_none(
            baseline_scale.quality.stereo_rotation_error_median_deg
        ),
        "predicted_baselines_vggt_unit": [
            float(value)
            for value in baseline_scale.quality.predicted_baselines_vggt_unit.tolist()
        ],
        "stereo_rotation_errors_deg": [
            float(value)
            for value in baseline_scale.quality.stereo_rotation_errors_deg.tolist()
        ],
        "valid": baseline_scale.valid,
        "failure_reason": baseline_scale.failure_reason,
    }
    pose_quality["alignment"] = {
        "valid": bool(alignment is not None and alignment.valid),
        "static_prior_valid": static_prior_valid,
        "scale_px_m": (
            _finite_or_none(alignment.scale_px_m) if alignment is not None else None
        ),
        "reliable_pixel_count": (
            alignment.reliable_pixel_count if alignment is not None else 0
        ),
        "iterations": alignment.iterations if alignment is not None else 0,
        "converged": alignment.converged if alignment is not None else False,
        "weighted_mean_absolute_residual_hr_px": (
            _finite_or_none(alignment.weighted_mean_absolute_residual_hr_px)
            if alignment is not None
            else None
        ),
        "failure_reason": alignment.failure_reason if alignment is not None else "invalid_pose_scale",
        "model": "scale_only; no additive shift",
    }
    metadata = {
        "target": {
            "sequence_id": linkage["target_sequence_id"],
            "frame_id": linkage["target_frame_id"],
            "timestamp": linkage["target_timestamp"],
        },
        "pose_quality": pose_quality,
        "tensor_semantics": {
            "metric_pose_diagnostic_only": (
                "scaled pose retained for audit; never consume for temporal warp"
            ),
            "metric_pose_temporal": (
                "zero-filled unless temporal_pose_valid is true"
            ),
            "metric_depth": (
                "zero-filled unless vggt_depth_metric_valid_mask is true"
            ),
            "aligned_disparity": (
                "FFS-owned HR pixel units; zero-filled unless static_prior_valid"
            ),
            "aligned_confidence": (
                "1 - reciprocal(raw confidence), since raw confidence is 1+exp(logit)"
            ),
        },
    }
    if rectified_calibration_window is not None:
        metadata["stereo_calibration"] = {
            "component": CALIBRATED_DERIVED_COMPONENT,
            "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
            "extrinsics_convention": (
                "right-camera-from-left-camera; X_right=T_right_left@X_left"
            ),
            "window_sha256": calibration_window_sha256(
                rectified_calibration_window
            ),
            "ordered_record_sha256": [
                record.calibration_record_sha256
                for record in rectified_calibration_window
            ],
            "target_record_sha256": (
                rectified_calibration_window[-1].calibration_record_sha256
            ),
            "quality_policy": (
                "raw VGGT stereo residuals own metric-scale and pose-valid gates; "
                "post-constraint exact stereo geometry is not quality evidence"
            ),
            "hybrid_pose_valid": complete_quality.pose_valid,
        }
        metadata["tensor_semantics"].update(
            {
                "rectified_stereo_extrinsic": (
                    "target right-from-left [4,4] metres in stored rectified "
                    "virtual-camera coordinates"
                ),
                "metric_pose_temporal_stereo_constrained": (
                    "left poses retain metric-scaled VGGT values; each right pose "
                    "is exact calibrated T_right_left @ E_left; zero-filled unless "
                    "the original raw-pose quality gate passes"
                ),
            }
        )
    return converted, metadata


def main() -> int:
    args = _parse_args()
    for name, path in {"VGGT cache": args.vggt_cache, "FFS cache": args.ffs_cache}.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.output.suffix != ".pt":
        raise ValueError("--output must be an exact .pt path")
    receipt_path = args.receipt or args.output.with_suffix(".receipt.json")
    if receipt_path.resolve() == args.output.resolve():
        raise ValueError("--receipt and --output must be different paths")
    thresholds = GeometryThresholds(
        max_baseline_cv=args.max_baseline_cv,
        max_stereo_rotation_error_deg=args.max_stereo_rotation_error_deg,
        ffs_confidence_threshold=args.ffs_confidence_threshold,
        max_left_right_error_lr_px=args.max_left_right_error_lr_px,
        min_alignment_pixels=args.min_alignment_pixels,
        alignment_huber_delta_hr_px=args.alignment_huber_delta_hr_px,
        max_depth_weighted_mae_hr_px=args.max_depth_weighted_mae_hr_px,
        max_depth_median_absolute_error_hr_px=(
            args.max_depth_median_absolute_error_hr_px
        ),
        max_depth_median_relative_error=args.max_depth_median_relative_error,
        min_photometric_samples=args.min_photometric_samples,
        min_photometric_valid_fraction=args.min_photometric_valid_fraction,
        max_photometric_median_absolute_rgb=(
            args.max_photometric_median_absolute_rgb
        ),
    )
    thresholds.validate()
    vggt_payload = load_cache_record(args.vggt_cache)
    ffs_payload = load_cache_record(args.ffs_cache)
    linkage = validate_raw_cache_pair(vggt_payload, ffs_payload)
    calibration_index: RectifiedCalibrationIndex | None = None
    calibration_window: tuple[RectifiedCalibrationRecord, ...] | None = None
    if args.rectified_calibration_receipt is not None and (
        args.rectified_calibration_sidecar is None
    ):
        raise ValueError(
            "--rectified-calibration-receipt requires --rectified-calibration-sidecar"
        )
    if args.rectified_calibration_sidecar is not None:
        calibration_index = load_rectified_calibration_sidecar(
            args.rectified_calibration_sidecar,
            receipt_path=args.rectified_calibration_receipt,
        )
        vggt_source = vggt_payload["metadata"]["source"]
        calibration_window = calibration_index.records_for_vggt_source(vggt_source)
    vggt_cache_sha256 = sha256_file(args.vggt_cache)
    ffs_cache_sha256 = sha256_file(args.ffs_cache)
    source = _source_metadata(
        vggt_cache=args.vggt_cache,
        ffs_cache=args.ffs_cache,
        vggt_cache_sha256=vggt_cache_sha256,
        ffs_cache_sha256=ffs_cache_sha256,
        linkage=linkage,
        rectified_calibration_index=calibration_index,
        rectified_calibration_window=calibration_window,
    )
    calibrated = calibration_window is not None
    config: dict[str, Any] = {
        "schema_version": (
            CALIBRATED_DERIVED_SCHEMA_VERSION if calibrated else DERIVED_SCHEMA_VERSION
        ),
        "algorithm": (
            CALIBRATED_DERIVED_ALGORITHM
            if calibrated
            else "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
        ),
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": PREVIOUS_LEFT_VIEW_INDEX,
        "current_left_view_index": CURRENT_LEFT_VIEW_INDEX,
        "thresholds": thresholds.as_dict(),
        "cache_dtype": args.cache_dtype,
        "missing_diagnostic_policy": "invalid",
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    if calibration_index is not None:
        config["rectified_stereo_calibration"] = {
            "component": CALIBRATED_DERIVED_COMPONENT,
            "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
            "sidecar_sha256": calibration_index.sidecar_sha256,
            "receipt_sha256": calibration_index.receipt_sha256,
            "pixel_audit_sha256": calibration_index.pixel_audit_sha256,
        }
    vggt_identity = vggt_payload["identity"]
    ffs_identity = ffs_payload["identity"]
    identity = CacheIdentity(
        component=(
            CALIBRATED_DERIVED_COMPONENT
            if calibrated
            else "vggt-ffs-derived-geometry"
        ),
        upstream_commit=canonical_json_sha256(
            {
                "vggt": vggt_identity["upstream_commit"],
                "ffs": ffs_identity["upstream_commit"],
            }
        ),
        checkpoint_sha256=canonical_json_sha256(
            {
                "vggt_raw_cache_sha256": vggt_cache_sha256,
                "ffs_raw_cache_sha256": ffs_cache_sha256,
                **(
                    {
                        "rectified_calibration_window_sha256": (
                            calibration_window_sha256(calibration_window)
                        )
                    }
                    if calibration_window is not None
                    else {}
                ),
            }
        ),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(config),
    )

    status: str
    if args.output.exists() and not args.overwrite:
        existing = load_cache_record(args.output, expected_identity=identity)
        if existing["metadata"].get("source") != source:
            raise CacheMismatchError(
                "derived cache source mismatch despite identity match; refusing reuse"
            )
        metadata = existing["metadata"]
        status = "reused_identity_and_source_match"
    else:
        cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
        tensors, derived_metadata = derive_geometry(
            vggt_payload,
            ffs_payload,
            thresholds=thresholds,
            cache_dtype=cache_dtype,
            rectified_calibration_window=calibration_window,
        )
        metadata = {
            "source": source,
            "config": config,
            **derived_metadata,
        }
        save_cache_record(
            args.output,
            tensors=tensors,
            metadata=metadata,
            identity=identity,
        )
        status = "written"

    receipt = {
        "schema_version": config["schema_version"],
        "status": status,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "identity": identity.to_dict(),
        "source": source,
        "config": config,
        "target": metadata["target"],
        "pose_quality": metadata["pose_quality"],
    }
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output.resolve()),
                "receipt": str(receipt_path.resolve()),
                "pose_valid": metadata["pose_quality"]["pose_valid"],
                "static_prior_valid": metadata["pose_quality"]["alignment"][
                    "static_prior_valid"
                ],
                "failure_reasons": metadata["pose_quality"]["failure_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
