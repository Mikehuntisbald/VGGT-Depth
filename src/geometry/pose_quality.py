"""End-to-end quality gates for metric VGGT geometry caches.

This module deliberately keeps *diagnostics* separate from *availability*.
Missing photometric or depth/disparity evidence is not a pass: temporal VGGT
pose may only be exposed after the stereo-baseline, stereo-rotation,
photometric, and FFS-consistency gates have all been measured and passed.

Extrinsics follow the project convention ``camera-from-world``.  Dense depth
and image coordinates in the reprojection diagnostic live on the recorded
VGGT model grid; calibrated intrinsics and the recorded model/original
homographies are used explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data.cache_dataset import CacheMismatchError
from geometry.pose_scale import BaselineScaleEstimate


PREVIOUS_LEFT_VIEW_INDEX = 6
CURRENT_LEFT_VIEW_INDEX = 8


@dataclass(frozen=True, slots=True)
class DepthDisparityConsistency:
    """Agreement between aligned VGGT prior and trusted FFS disparity."""

    available: bool
    valid: bool
    sample_count: int
    weighted_mae_hr_px: float | None
    median_absolute_error_hr_px: float | None
    median_relative_error: float | None
    failure_reasons: tuple[str, ...]
    relative_error_gate_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "valid": self.valid,
            "sample_count": self.sample_count,
            "weighted_mae_hr_px": self.weighted_mae_hr_px,
            "median_absolute_error_hr_px": self.median_absolute_error_hr_px,
            "median_relative_error": self.median_relative_error,
            "relative_error_gate_enabled": self.relative_error_gate_enabled,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class PhotometricReprojectionDiagnostic:
    """Adjacent-left RGB consistency after current-to-previous reprojection.

    RGB residuals are in ``[0, 1]`` and are the median, over projected pixels,
    of per-pixel mean absolute RGB error.  ``valid_fraction`` is measured over
    positive finite current-depth pixels, not over padded/invalid depth.
    """

    available: bool
    valid: bool
    depth_sample_count: int
    projected_sample_count: int
    valid_fraction: float | None
    median_absolute_rgb_residual: float | None
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "valid": self.valid,
            "depth_sample_count": self.depth_sample_count,
            "projected_sample_count": self.projected_sample_count,
            "valid_fraction": self.valid_fraction,
            "median_absolute_rgb_residual": self.median_absolute_rgb_residual,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class CompletePoseQuality:
    """Strict result controlling whether metric pose may feed temporal warp."""

    pose_valid: bool
    baseline_rotation_valid: bool
    photometric: PhotometricReprojectionDiagnostic
    depth_consistency: DepthDisparityConsistency
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_valid": self.pose_valid,
            "baseline_rotation_valid": self.baseline_rotation_valid,
            "photometric": self.photometric.as_dict(),
            "depth_consistency": self.depth_consistency.as_dict(),
            "failure_reasons": list(self.failure_reasons),
        }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheMismatchError(f"{name} is missing or is not a mapping")
    return value


def validate_raw_cache_pair(
    vggt_payload: Mapping[str, Any],
    ffs_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that raw VGGT and FFS records describe one exact target frame.

    The cache loader already validates the outer cache schema.  This function
    additionally checks component identity, target manifest record, image
    paths, and current stereo image hashes before any tensors are combined.

    Returns:
        A JSON-safe source-linkage mapping suitable for the derived record.

    Raises:
        CacheMismatchError: On any identity or source mismatch.
    """

    vggt_identity = _require_mapping(vggt_payload.get("identity"), "VGGT identity")
    ffs_identity = _require_mapping(ffs_payload.get("identity"), "FFS identity")
    if vggt_identity.get("component") != "vggt-omega":
        raise CacheMismatchError(
            "VGGT cache component mismatch: expected 'vggt-omega', got "
            f"{vggt_identity.get('component')!r}"
        )
    if ffs_identity.get("component") != "ffs-observation":
        raise CacheMismatchError(
            "FFS cache component mismatch: expected 'ffs-observation', got "
            f"{ffs_identity.get('component')!r}"
        )

    vggt_metadata = _require_mapping(vggt_payload.get("metadata"), "VGGT metadata")
    ffs_metadata = _require_mapping(ffs_payload.get("metadata"), "FFS metadata")
    vggt_source = _require_mapping(vggt_metadata.get("source"), "VGGT source")
    ffs_source = _require_mapping(ffs_metadata.get("source"), "FFS source")
    manifest_record = _require_mapping(
        ffs_source.get("manifest_record"), "FFS source manifest_record"
    )
    window_records = vggt_source.get("manifest_records")
    ordered_images = vggt_source.get("ordered_images")
    if not isinstance(window_records, list) or len(window_records) != 5:
        raise CacheMismatchError("VGGT source must contain exactly five manifest records")
    if not isinstance(ordered_images, list) or len(ordered_images) != 10:
        raise CacheMismatchError("VGGT source must contain exactly ten ordered images")

    differences: dict[str, dict[str, Any]] = {}

    def compare(name: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            differences[name] = {"expected": expected, "actual": actual}

    compare("target_manifest_record", dict(manifest_record), window_records[-1])
    compare("target_sequence_id", manifest_record.get("sequence_id"), vggt_source.get("target_sequence_id"))
    compare("target_frame_id", manifest_record.get("frame_id"), vggt_source.get("target_frame_id"))
    compare("target_timestamp", manifest_record.get("timestamp"), vggt_source.get("target_timestamp"))

    current_left = _require_mapping(
        ordered_images[CURRENT_LEFT_VIEW_INDEX], "VGGT current-left image source"
    )
    current_right = _require_mapping(
        ordered_images[CURRENT_LEFT_VIEW_INDEX + 1], "VGGT current-right image source"
    )
    compare("current_left_path", manifest_record.get("left_path"), current_left.get("path"))
    compare("current_right_path", manifest_record.get("right_path"), current_right.get("path"))
    compare("current_left_sha256", ffs_source.get("left_sha256"), current_left.get("sha256"))
    compare("current_right_sha256", ffs_source.get("right_sha256"), current_right.get("sha256"))
    if differences:
        raise CacheMismatchError(f"raw cache source mismatch: {differences!r}")

    return {
        "target_sequence_id": manifest_record["sequence_id"],
        "target_frame_id": manifest_record["frame_id"],
        "target_timestamp": manifest_record["timestamp"],
        "target_manifest_record": dict(manifest_record),
        "current_left_sha256": current_left["sha256"],
        "current_right_sha256": current_right["sha256"],
        "previous_left": dict(ordered_images[PREVIOUS_LEFT_VIEW_INDEX]),
        "current_left": dict(current_left),
        "vggt_raw_identity": dict(vggt_identity),
        "ffs_raw_identity": dict(ffs_identity),
    }


def depth_disparity_consistency(
    disparity_ffs_hr_px: torch.Tensor,
    disparity_vggt_aligned_hr_px: torch.Tensor,
    *,
    trusted_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    min_samples: int = 32,
    max_weighted_mae_hr_px: float = 2.0,
    max_median_absolute_error_hr_px: float = 2.0,
    max_median_relative_error: float | None = None,
    relative_epsilon_hr_px: float = 1e-6,
) -> DepthDisparityConsistency:
    """Compute explicit FFS/VGGT disparity residual diagnostics.

    Absolute HR-pixel errors are always gated. Pointwise relative error is
    recorded but is gated only when ``max_median_relative_error`` is supplied,
    because division by near-zero far-range disparity is ill-conditioned.
    """

    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    for name, value in {
        "max_weighted_mae_hr_px": max_weighted_mae_hr_px,
        "max_median_absolute_error_hr_px": max_median_absolute_error_hr_px,
    }.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if relative_epsilon_hr_px <= 0 or not math.isfinite(relative_epsilon_hr_px):
        raise ValueError("relative_epsilon_hr_px must be finite and positive")
    if (
        max_median_relative_error is not None
        and (
            not math.isfinite(float(max_median_relative_error))
            or float(max_median_relative_error) < 0
        )
    ):
        raise ValueError(
            "max_median_relative_error must be finite and non-negative or None"
        )

    disparity_ffs = torch.as_tensor(disparity_ffs_hr_px).float()
    disparity_vggt = torch.as_tensor(disparity_vggt_aligned_hr_px).float()
    trusted = torch.as_tensor(trusted_mask, dtype=torch.bool)
    try:
        disparity_ffs, disparity_vggt, trusted = torch.broadcast_tensors(
            disparity_ffs, disparity_vggt, trusted
        )
    except RuntimeError as exc:
        raise ValueError("disparities and trusted_mask must be broadcastable") from exc
    if weights is None:
        fit_weights = torch.ones_like(disparity_ffs)
    else:
        fit_weights = torch.as_tensor(weights, dtype=torch.float32)
        try:
            fit_weights = torch.broadcast_to(fit_weights, disparity_ffs.shape)
        except RuntimeError as exc:
            raise ValueError("weights must be broadcastable to disparity shape") from exc
    usable = (
        trusted
        & torch.isfinite(disparity_ffs)
        & (disparity_ffs > 0)
        & torch.isfinite(disparity_vggt)
        & (disparity_vggt > 0)
        & torch.isfinite(fit_weights)
        & (fit_weights > 0)
    )
    count = int(usable.sum().item())
    if count < min_samples:
        return DepthDisparityConsistency(
            available=False,
            valid=False,
            sample_count=count,
            weighted_mae_hr_px=None,
            median_absolute_error_hr_px=None,
            median_relative_error=None,
            failure_reasons=("insufficient_depth_consistency_samples",),
            relative_error_gate_enabled=max_median_relative_error is not None,
        )

    target = disparity_ffs[usable]
    prediction = disparity_vggt[usable]
    selected_weights = fit_weights[usable]
    absolute = (prediction - target).abs()
    weighted_mae = float(
        ((absolute * selected_weights).sum() / selected_weights.sum()).item()
    )
    median_absolute = float(absolute.median().item())
    median_relative = float(
        (absolute / target.abs().clamp_min(relative_epsilon_hr_px)).median().item()
    )
    reasons: list[str] = []
    if not all(math.isfinite(value) for value in (weighted_mae, median_absolute, median_relative)):
        reasons.append("non_finite_depth_consistency")
    if weighted_mae > max_weighted_mae_hr_px:
        reasons.append("depth_weighted_mae_exceeds_threshold")
    if median_absolute > max_median_absolute_error_hr_px:
        reasons.append("depth_median_absolute_error_exceeds_threshold")
    if (
        max_median_relative_error is not None
        and median_relative > max_median_relative_error
    ):
        reasons.append("depth_median_relative_error_exceeds_threshold")
    return DepthDisparityConsistency(
        available=True,
        valid=not reasons,
        sample_count=count,
        weighted_mae_hr_px=weighted_mae,
        median_absolute_error_hr_px=median_absolute,
        median_relative_error=median_relative,
        failure_reasons=tuple(reasons),
        relative_error_gate_enabled=max_median_relative_error is not None,
    )


def _homogeneous_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(extrinsic, dtype=torch.float32)
    if tuple(value.shape) == (4, 4):
        return value
    if tuple(value.shape) != (3, 4):
        raise ValueError(f"extrinsic must be [3,4] or [4,4], got {tuple(value.shape)}")
    result = torch.eye(4, dtype=value.dtype, device=value.device)
    result[:3] = value
    return result


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _sample_rgb(rgb_bchw: torch.Tensor, uv_px: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample one image at flattened continuous pixel coordinates."""

    height, width = rgb_bchw.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError("photometric source images must be at least 2x2")
    x = 2.0 * uv_px[:, 0] / float(width - 1) - 1.0
    y = 2.0 * uv_px[:, 1] / float(height - 1) - 1.0
    grid = torch.stack((x, y), dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        rgb_bchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def _apply_pixel_transform(transform_3x3: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
    mapped = (transform_3x3 @ homogeneous.transpose(0, 1)).transpose(0, 1)
    return mapped[:, :2] / mapped[:, 2:].clamp_min(1e-12)


def adjacent_left_photometric_reprojection(
    depth_current_left_m: torch.Tensor | None,
    *,
    extrinsic_previous_camera_from_world_metric: torch.Tensor | None,
    extrinsic_current_camera_from_world_metric: torch.Tensor | None,
    intrinsics_previous_model_px: torch.Tensor | None,
    intrinsics_current_model_px: torch.Tensor | None,
    previous_model_to_original_transform: torch.Tensor | None,
    current_model_to_original_transform: torch.Tensor | None,
    previous_rgb_path: Path | str | None,
    current_rgb_path: Path | str | None,
    valid_depth_mask: torch.Tensor | None = None,
    min_samples: int = 1024,
    min_valid_fraction: float = 0.50,
    max_median_absolute_rgb_residual: float = 0.12,
) -> PhotometricReprojectionDiagnostic:
    """Measure current-left to previous-left photometric pose consistency.

    Current model-grid points are back-projected with current metric depth,
    transformed by ``E_previous @ inverse(E_current)``, and projected into the
    previous calibrated model camera.  Both images are sampled in original
    coordinates via their recorded model-to-original homographies.
    """

    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0,1]")
    if max_median_absolute_rgb_residual < 0 or not math.isfinite(
        max_median_absolute_rgb_residual
    ):
        raise ValueError("max_median_absolute_rgb_residual must be finite and non-negative")
    required = (
        depth_current_left_m,
        extrinsic_previous_camera_from_world_metric,
        extrinsic_current_camera_from_world_metric,
        intrinsics_previous_model_px,
        intrinsics_current_model_px,
        previous_model_to_original_transform,
        current_model_to_original_transform,
        previous_rgb_path,
        current_rgb_path,
    )
    if any(value is None for value in required):
        return PhotometricReprojectionDiagnostic(
            available=False,
            valid=False,
            depth_sample_count=0,
            projected_sample_count=0,
            valid_fraction=None,
            median_absolute_rgb_residual=None,
            failure_reasons=("missing_photometric_inputs",),
        )
    previous_path = Path(previous_rgb_path)  # type: ignore[arg-type]
    current_path = Path(current_rgb_path)  # type: ignore[arg-type]
    if not previous_path.is_file() or not current_path.is_file():
        return PhotometricReprojectionDiagnostic(
            available=False,
            valid=False,
            depth_sample_count=0,
            projected_sample_count=0,
            valid_fraction=None,
            median_absolute_rgb_residual=None,
            failure_reasons=("missing_photometric_image",),
        )

    depth = torch.as_tensor(depth_current_left_m, dtype=torch.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth_current_left_m must reduce to [H,W], got {tuple(depth.shape)}")
    height, width = depth.shape
    depth_valid = torch.isfinite(depth) & (depth > 0)
    if valid_depth_mask is not None:
        mask = torch.as_tensor(valid_depth_mask, dtype=torch.bool).squeeze()
        if tuple(mask.shape) != (height, width):
            raise ValueError("valid_depth_mask must match depth spatial shape")
        depth_valid &= mask
    depth_count = int(depth_valid.sum().item())
    if depth_count == 0:
        return PhotometricReprojectionDiagnostic(
            available=False,
            valid=False,
            depth_sample_count=0,
            projected_sample_count=0,
            valid_fraction=0.0,
            median_absolute_rgb_residual=None,
            failure_reasons=("no_valid_metric_depth",),
        )

    k_previous = torch.as_tensor(intrinsics_previous_model_px, dtype=torch.float32)
    k_current = torch.as_tensor(intrinsics_current_model_px, dtype=torch.float32)
    transform_previous = torch.as_tensor(
        previous_model_to_original_transform, dtype=torch.float32
    )
    transform_current = torch.as_tensor(
        current_model_to_original_transform, dtype=torch.float32
    )
    for name, matrix in {
        "intrinsics_previous_model_px": k_previous,
        "intrinsics_current_model_px": k_current,
        "previous_model_to_original_transform": transform_previous,
        "current_model_to_original_transform": transform_current,
    }.items():
        if tuple(matrix.shape) != (3, 3) or not bool(torch.isfinite(matrix).all()):
            raise ValueError(f"{name} must be finite [3,3]")

    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    uv_current = torch.stack((u[depth_valid], v[depth_valid]), dim=-1)
    pixels_current_h = torch.cat(
        (uv_current, torch.ones((depth_count, 1), dtype=torch.float32)), dim=-1
    )
    points_current = (
        torch.linalg.inv(k_current) @ pixels_current_h.transpose(0, 1)
    ).transpose(0, 1) * depth[depth_valid, None]
    e_previous = _homogeneous_extrinsic(
        torch.as_tensor(extrinsic_previous_camera_from_world_metric)
    )
    e_current = _homogeneous_extrinsic(
        torch.as_tensor(extrinsic_current_camera_from_world_metric)
    )
    relative_previous_from_current = e_previous @ torch.linalg.inv(e_current)
    points_current_h = torch.cat(
        (points_current, torch.ones((depth_count, 1), dtype=torch.float32)), dim=-1
    )
    points_previous = (
        relative_previous_from_current @ points_current_h.transpose(0, 1)
    ).transpose(0, 1)[:, :3]
    projected_previous_h = (k_previous @ points_previous.transpose(0, 1)).transpose(0, 1)
    positive_z = torch.isfinite(points_previous).all(dim=-1) & (points_previous[:, 2] > 1e-8)
    uv_previous = projected_previous_h[:, :2] / projected_previous_h[:, 2:].clamp_min(1e-8)

    previous_rgb = _load_rgb(previous_path)
    current_rgb = _load_rgb(current_path)
    previous_hw = previous_rgb.shape[-2:]
    current_hw = current_rgb.shape[-2:]
    uv_previous_original = _apply_pixel_transform(transform_previous, uv_previous)
    uv_current_original = _apply_pixel_transform(transform_current, uv_current)
    in_previous = (
        (uv_previous_original[:, 0] >= 0)
        & (uv_previous_original[:, 0] <= previous_hw[1] - 1)
        & (uv_previous_original[:, 1] >= 0)
        & (uv_previous_original[:, 1] <= previous_hw[0] - 1)
    )
    in_current = (
        (uv_current_original[:, 0] >= 0)
        & (uv_current_original[:, 0] <= current_hw[1] - 1)
        & (uv_current_original[:, 1] >= 0)
        & (uv_current_original[:, 1] <= current_hw[0] - 1)
    )
    projected_valid = positive_z & torch.isfinite(uv_previous).all(dim=-1) & in_previous & in_current
    projected_count = int(projected_valid.sum().item())
    valid_fraction = projected_count / depth_count
    reasons: list[str] = []
    if projected_count < min_samples:
        reasons.append("insufficient_photometric_samples")
    if valid_fraction < min_valid_fraction:
        reasons.append("photometric_valid_fraction_below_threshold")
    if projected_count == 0:
        return PhotometricReprojectionDiagnostic(
            available=False,
            valid=False,
            depth_sample_count=depth_count,
            projected_sample_count=0,
            valid_fraction=valid_fraction,
            median_absolute_rgb_residual=None,
            failure_reasons=tuple(reasons or ["no_projected_photometric_samples"]),
        )

    sampled_previous = _sample_rgb(
        previous_rgb, uv_previous_original[projected_valid]
    )
    sampled_current = _sample_rgb(current_rgb, uv_current_original[projected_valid])
    per_pixel_rgb_residual = (sampled_previous - sampled_current).abs().mean(dim=-1)
    median_residual = float(per_pixel_rgb_residual.median().item())
    if not math.isfinite(median_residual):
        reasons.append("non_finite_photometric_residual")
    elif median_residual > max_median_absolute_rgb_residual:
        reasons.append("photometric_residual_exceeds_threshold")
    return PhotometricReprojectionDiagnostic(
        available=projected_count >= min_samples,
        valid=not reasons,
        depth_sample_count=depth_count,
        projected_sample_count=projected_count,
        valid_fraction=valid_fraction,
        median_absolute_rgb_residual=median_residual,
        failure_reasons=tuple(reasons),
    )


def combine_pose_quality(
    baseline_scale: BaselineScaleEstimate,
    photometric: PhotometricReprojectionDiagnostic,
    depth_consistency: DepthDisparityConsistency,
) -> CompletePoseQuality:
    """Require every geometry and image diagnostic; missing means invalid."""

    reasons: list[str] = []
    if not baseline_scale.valid:
        reasons.extend(
            f"baseline_rotation:{reason}"
            for reason in (
                baseline_scale.quality.failure_reasons
                or (baseline_scale.failure_reason or "invalid",)
            )
        )
    if not photometric.available:
        reasons.append("photometric:missing_or_insufficient")
    if not photometric.valid:
        reasons.extend(f"photometric:{reason}" for reason in photometric.failure_reasons)
    if not depth_consistency.available:
        reasons.append("depth_consistency:missing_or_insufficient")
    if not depth_consistency.valid:
        reasons.extend(
            f"depth_consistency:{reason}" for reason in depth_consistency.failure_reasons
        )
    # Preserve order while removing duplicate umbrella/detail reasons.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return CompletePoseQuality(
        pose_valid=(
            baseline_scale.valid
            and photometric.available
            and photometric.valid
            and depth_consistency.available
            and depth_consistency.valid
        ),
        baseline_rotation_valid=baseline_scale.valid,
        photometric=photometric,
        depth_consistency=depth_consistency,
        failure_reasons=unique_reasons,
    )


__all__ = [
    "CURRENT_LEFT_VIEW_INDEX",
    "PREVIOUS_LEFT_VIEW_INDEX",
    "CompletePoseQuality",
    "DepthDisparityConsistency",
    "PhotometricReprojectionDiagnostic",
    "adjacent_left_photometric_reprojection",
    "combine_pose_quality",
    "depth_disparity_consistency",
    "validate_raw_cache_pair",
]
