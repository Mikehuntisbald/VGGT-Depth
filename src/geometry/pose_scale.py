"""Recover metric scale for VGGT camera poses from a calibrated stereo rig.

VGGT extrinsics in this project are **camera-from-world** transforms::

    X_camera = R_camera_from_world @ X_world + t_camera_from_world

Consequently, the camera centre expressed in world coordinates is
``C_world = -R.T @ t``.  Stereo inputs are ordered
``L0, R0, L1, R1, ..., L4, R4``.  Their predicted baseline lengths share the
same arbitrary unit as VGGT depth.  A single metric scale is recovered as::

    alpha = calibrated_baseline_m / median(predicted_baseline_vggt_unit)

Both camera translations and depth must be multiplied by ``alpha``.  Scaling
only one of them would make reprojection geometrically inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

try:  # Torch is a runtime dependency in the three execution environments.
    import torch
except ImportError:  # pragma: no cover - permits lightweight metadata tooling.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PoseQuality:
    """Quality diagnostics for one interleaved stereo pose window.

    Attributes:
        baseline_coefficient_of_variation: Population standard deviation of
            the predicted stereo baselines divided by their mean.
        stereo_rotation_error_deg: Maximum rotation-angle error, in degrees,
            between predicted and calibrated right-from-left rotations.
        stereo_rotation_error_median_deg: Median of those per-pair errors.
        predicted_baselines_vggt_unit: One arbitrary-unit baseline per stereo
            pair.  The array/tensor backend matches the extrinsics input.
        stereo_rotation_errors_deg: Per-pair rotation errors in degrees.
        reprojection_residual_px: Optional externally measured reprojection
            residual.  ``None`` means it was not assessed here.
        valid: Whether every enabled quality gate passed.
        failure_reasons: Stable, machine-readable reasons for invalidity.
    """

    baseline_coefficient_of_variation: float
    stereo_rotation_error_deg: float
    stereo_rotation_error_median_deg: float
    predicted_baselines_vggt_unit: Any
    stereo_rotation_errors_deg: Any
    reprojection_residual_px: float | None
    valid: bool
    failure_reasons: tuple[str, ...]

    @property
    def baseline_cv(self) -> float:
        """Short alias used in cache metadata."""

        return self.baseline_coefficient_of_variation

    @property
    def relative_stereo_rotation_error_deg(self) -> float:
        """Alias making the measured rotation convention explicit."""

        return self.stereo_rotation_error_deg


@dataclass(frozen=True)
class BaselineScaleEstimate:
    """Metric scale estimate for one VGGT stereo window."""

    alpha_m_per_vggt_unit: float
    calibrated_baseline_m: float
    median_predicted_baseline_vggt_unit: float
    quality: PoseQuality
    valid: bool
    failure_reason: str | None

    @property
    def alpha(self) -> float:
        """Compatibility alias for the baseline-derived scale."""

        return self.alpha_m_per_vggt_unit


@dataclass(frozen=True)
class MetricScaledVGGTGeometry:
    """VGGT pose/depth after applying one baseline-derived metric scale.

    Scaled values are ``None`` when the pose window fails quality validation;
    callers must disable VGGT temporal pose for such a window.
    """

    extrinsics_camera_from_world_metric: Any | None
    depth_m: Any | None
    scale: BaselineScaleEstimate

    @property
    def valid(self) -> bool:
        return self.scale.valid


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _as_floating(value: Any, name: str) -> Any:
    if _is_torch(value):
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued")
        return value if value.is_floating_point() else value.to(dtype=torch.float64)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued numeric data")
    return array.astype(
        array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64,
        copy=False,
    )


def _extrinsic_parts(extrinsics_camera_from_world: Any) -> tuple[Any, Any]:
    extrinsics = _as_floating(
        extrinsics_camera_from_world, "extrinsics_camera_from_world"
    )
    if extrinsics.ndim < 2 or tuple(extrinsics.shape[-2:]) not in {
        (3, 4),
        (4, 4),
    }:
        raise ValueError(
            "extrinsics_camera_from_world must end in [3,4] or [4,4], got "
            f"{tuple(extrinsics.shape)}"
        )
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return rotation, translation


def _paired_extrinsics(
    extrinsics_camera_from_world: Any,
    *,
    expected_pairs: int | None,
) -> Any:
    extrinsics = _as_floating(
        extrinsics_camera_from_world, "extrinsics_camera_from_world"
    )
    _extrinsic_parts(extrinsics)

    # Accepted unbatched layouts are [2P,3,4]/[2P,4,4] (interleaved) and
    # [P,2,3,4]/[P,2,4,4] (already paired).  A single [2,3,4] pair naturally
    # follows the first branch.
    if extrinsics.ndim == 4 and extrinsics.shape[-3] == 2:
        paired = extrinsics
    elif extrinsics.ndim == 3:
        camera_count = int(extrinsics.shape[0])
        if camera_count == 0 or camera_count % 2:
            raise ValueError(
                "interleaved stereo extrinsics must contain a positive even "
                f"camera count, got {camera_count}"
            )
        paired = extrinsics.reshape(
            camera_count // 2,
            2,
            extrinsics.shape[-2],
            extrinsics.shape[-1],
        )
    else:
        raise ValueError(
            "expected an unbatched interleaved [2P,3,4]/[2P,4,4] or paired "
            f"[P,2,3,4]/[P,2,4,4] pose window, got {tuple(extrinsics.shape)}"
        )

    pair_count = int(paired.shape[0])
    if expected_pairs is not None:
        if isinstance(expected_pairs, bool) or not isinstance(expected_pairs, int):
            raise TypeError("expected_pairs must be an integer or None")
        if expected_pairs <= 0:
            raise ValueError("expected_pairs must be positive")
        if pair_count != expected_pairs:
            raise ValueError(
                f"expected {expected_pairs} stereo pairs, got {pair_count}"
            )
    return paired


def _finite_positive_scalar(value: Real, name: str) -> float:
    value_float = float(value)
    if not math.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value_float


def _finite_nonnegative_scalar(value: Real, name: str) -> float:
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return value_float


def _all_finite(value: Any) -> bool:
    if _is_torch(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _to_float(value: Any) -> float:
    if _is_torch(value):
        return float(value.detach().cpu().item())
    return float(np.asarray(value).item())


def _nan_vector_like(reference: Any, length: int) -> Any:
    if _is_torch(reference):
        return torch.full(
            (length,),
            torch.nan,
            dtype=reference.dtype,
            device=reference.device,
        )
    return np.full((length,), np.nan, dtype=reference.dtype)


def camera_centers_from_extrinsics(extrinsics_camera_from_world: Any) -> Any:
    """Return camera centres in world coordinates for camera-from-world poses.

    Args:
        extrinsics_camera_from_world: ``[...,3,4]`` or homogeneous
            ``[...,4,4]`` camera-from-world matrices.

    Returns:
        Centres with shape ``[...,3]`` in the translation unit of the input.
        The input is not modified.  NaNs propagate so quality assessment can
        reject an invalid window without silently inventing a pose.
    """

    rotation, translation = _extrinsic_parts(extrinsics_camera_from_world)
    if _is_torch(rotation):
        return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    return -np.matmul(np.swapaxes(rotation, -1, -2), translation[..., None])[..., 0]


def stereo_baselines_from_extrinsics(
    extrinsics_camera_from_world: Any,
    *,
    expected_pairs: int | None = 5,
) -> Any:
    """Measure each left/right camera-centre distance in VGGT units.

    The default enforces the MVP contract of five stereo moments (ten images).
    Pass ``expected_pairs=None`` for small diagnostic examples.
    """

    paired = _paired_extrinsics(
        extrinsics_camera_from_world, expected_pairs=expected_pairs
    )
    centers = camera_centers_from_extrinsics(paired)
    delta = centers[:, 1] - centers[:, 0]
    if _is_torch(delta):
        return torch.linalg.vector_norm(delta, dim=-1)
    return np.linalg.norm(delta, axis=-1)


def relative_stereo_rotations(
    extrinsics_camera_from_world: Any,
    *,
    expected_pairs: int | None = 5,
) -> Any:
    """Return predicted right-from-left rotations for every stereo pair.

    For camera-from-world rotations, ``R_right_from_left = R_R @ R_L.T``.
    """

    paired = _paired_extrinsics(
        extrinsics_camera_from_world, expected_pairs=expected_pairs
    )
    rotation, _ = _extrinsic_parts(paired)
    if _is_torch(rotation):
        return rotation[:, 1] @ rotation[:, 0].transpose(-1, -2)
    return rotation[:, 1] @ np.swapaxes(rotation[:, 0], -1, -2)


def rotation_angle_error_deg(
    rotation_predicted: Any,
    rotation_reference: Any,
) -> Any:
    """Return geodesic SO(3) angle error in degrees for ``[...,3,3]`` inputs."""

    predicted = _as_floating(rotation_predicted, "rotation_predicted")
    reference = _as_floating(rotation_reference, "rotation_reference")
    if _is_torch(predicted) != _is_torch(reference):
        raise TypeError("rotation_predicted and rotation_reference must share a backend")
    if tuple(predicted.shape[-2:]) != (3, 3):
        raise ValueError(
            f"rotation_predicted must end in [3,3], got {tuple(predicted.shape)}"
        )
    if tuple(reference.shape[-2:]) != (3, 3):
        raise ValueError(
            f"rotation_reference must end in [3,3], got {tuple(reference.shape)}"
        )

    if _is_torch(predicted):
        reference = reference.to(device=predicted.device, dtype=predicted.dtype)
        relative = predicted @ reference.transpose(-1, -2)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
            -1.0, 1.0
        )
        return torch.rad2deg(torch.acos(cosine))

    predicted, reference = np.broadcast_arrays(predicted, reference)
    relative = predicted @ np.swapaxes(reference, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _rotation_matrices_are_valid(rotation: Any, *, atol: float = 1e-3) -> bool:
    if not _all_finite(rotation):
        return False
    if _is_torch(rotation):
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        gram = rotation.transpose(-1, -2) @ rotation
        orthogonal = torch.allclose(gram, identity.expand_as(gram), atol=atol, rtol=atol)
        determinants = torch.linalg.det(rotation)
        proper = torch.allclose(
            determinants,
            torch.ones_like(determinants),
            atol=atol,
            rtol=atol,
        )
        return bool(orthogonal and proper)
    identity = np.eye(3, dtype=rotation.dtype)
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    return bool(
        np.allclose(gram, identity, atol=atol, rtol=atol)
        and np.allclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=atol)
    )


def assess_pose_quality(
    extrinsics_camera_from_world: Any,
    *,
    calibrated_rotation_right_from_left: Any | None = None,
    expected_pairs: int | None = 5,
    max_baseline_cv: float = 0.10,
    max_stereo_rotation_error_deg: float = 5.0,
    reprojection_residual_px: float | None = None,
    max_reprojection_residual_px: float | None = None,
    minimum_predicted_baseline: float = 1e-8,
) -> PoseQuality:
    """Assess whether a VGGT pose window is safe for metric temporal warping.

    ``calibrated_rotation_right_from_left`` defaults to identity, appropriate
    for rectified stereo.  The maximum (not merely median) per-pair rotation
    error gates the window.  Reprojection gating is optional because that
    residual is computed by the adapter from image data.
    """

    max_baseline_cv = _finite_nonnegative_scalar(max_baseline_cv, "max_baseline_cv")
    max_stereo_rotation_error_deg = _finite_nonnegative_scalar(
        max_stereo_rotation_error_deg, "max_stereo_rotation_error_deg"
    )
    minimum_predicted_baseline = _finite_positive_scalar(
        minimum_predicted_baseline, "minimum_predicted_baseline"
    )
    if max_reprojection_residual_px is not None:
        max_reprojection_residual_px = _finite_nonnegative_scalar(
            max_reprojection_residual_px, "max_reprojection_residual_px"
        )
    if reprojection_residual_px is not None:
        reprojection_residual_px = float(reprojection_residual_px)

    paired = _paired_extrinsics(
        extrinsics_camera_from_world, expected_pairs=expected_pairs
    )
    pair_count = int(paired.shape[0])
    baselines = stereo_baselines_from_extrinsics(
        paired, expected_pairs=expected_pairs
    )
    rotations, _ = _extrinsic_parts(paired)
    failure_reasons: list[str] = []

    if not _all_finite(baselines):
        baseline_cv = math.nan
        failure_reasons.append("non_finite_baseline")
    else:
        if _is_torch(baselines):
            baseline_mean = _to_float(baselines.mean())
            baseline_std = _to_float(baselines.std(unbiased=False))
            minimum_baseline = _to_float(baselines.min())
        else:
            baseline_mean = float(np.mean(baselines))
            baseline_std = float(np.std(baselines, ddof=0))
            minimum_baseline = float(np.min(baselines))
        baseline_cv = (
            baseline_std / baseline_mean
            if baseline_mean > minimum_predicted_baseline
            else math.nan
        )
        if minimum_baseline <= minimum_predicted_baseline or not math.isfinite(
            baseline_cv
        ):
            failure_reasons.append("degenerate_baseline")
        elif baseline_cv > max_baseline_cv:
            failure_reasons.append("baseline_cv_exceeds_threshold")

    if calibrated_rotation_right_from_left is None:
        if _is_torch(paired):
            calibrated_rotation_right_from_left = torch.eye(
                3, dtype=paired.dtype, device=paired.device
            )
        else:
            calibrated_rotation_right_from_left = np.eye(3, dtype=paired.dtype)
    calibrated_rotation = _as_floating(
        calibrated_rotation_right_from_left,
        "calibrated_rotation_right_from_left",
    )
    if tuple(calibrated_rotation.shape) != (3, 3):
        raise ValueError(
            "calibrated_rotation_right_from_left must have shape [3,3], got "
            f"{tuple(calibrated_rotation.shape)}"
        )
    if _is_torch(paired) != _is_torch(calibrated_rotation):
        raise TypeError("extrinsics and calibrated rotation must share a backend")

    predicted_relative_rotations = relative_stereo_rotations(
        paired, expected_pairs=expected_pairs
    )
    if not _rotation_matrices_are_valid(rotations) or not _rotation_matrices_are_valid(
        calibrated_rotation
    ):
        rotation_errors = _nan_vector_like(paired, pair_count)
        max_rotation_error = math.nan
        median_rotation_error = math.nan
        failure_reasons.append("invalid_rotation_matrix")
    else:
        rotation_errors = rotation_angle_error_deg(
            predicted_relative_rotations, calibrated_rotation
        )
        if not _all_finite(rotation_errors):
            max_rotation_error = math.nan
            median_rotation_error = math.nan
            failure_reasons.append("non_finite_rotation_error")
        elif _is_torch(rotation_errors):
            max_rotation_error = _to_float(rotation_errors.max())
            median_rotation_error = _to_float(rotation_errors.median())
        else:
            max_rotation_error = float(np.max(rotation_errors))
            median_rotation_error = float(np.median(rotation_errors))
        if math.isfinite(max_rotation_error) and (
            max_rotation_error > max_stereo_rotation_error_deg
        ):
            failure_reasons.append("stereo_rotation_error_exceeds_threshold")

    if reprojection_residual_px is not None and not math.isfinite(
        reprojection_residual_px
    ):
        failure_reasons.append("non_finite_reprojection_residual")
    elif (
        reprojection_residual_px is not None
        and max_reprojection_residual_px is not None
        and reprojection_residual_px > max_reprojection_residual_px
    ):
        failure_reasons.append("reprojection_residual_exceeds_threshold")

    return PoseQuality(
        baseline_coefficient_of_variation=baseline_cv,
        stereo_rotation_error_deg=max_rotation_error,
        stereo_rotation_error_median_deg=median_rotation_error,
        predicted_baselines_vggt_unit=baselines,
        stereo_rotation_errors_deg=rotation_errors,
        reprojection_residual_px=reprojection_residual_px,
        valid=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
    )


def estimate_baseline_metric_scale(
    extrinsics_camera_from_world: Any,
    calibrated_baseline_m: Real,
    *,
    calibrated_rotation_right_from_left: Any | None = None,
    expected_pairs: int | None = 5,
    max_baseline_cv: float = 0.10,
    max_stereo_rotation_error_deg: float = 5.0,
    reprojection_residual_px: float | None = None,
    max_reprojection_residual_px: float | None = None,
    minimum_predicted_baseline: float = 1e-8,
) -> BaselineScaleEstimate:
    """Estimate ``alpha`` from the median of predicted stereo baselines.

    Invalid/non-finite pose windows return ``alpha=NaN`` and ``valid=False``;
    they never produce a metric pose that downstream code could use by
    accident.
    """

    baseline_m = _finite_positive_scalar(calibrated_baseline_m, "calibrated_baseline_m")
    quality = assess_pose_quality(
        extrinsics_camera_from_world,
        calibrated_rotation_right_from_left=calibrated_rotation_right_from_left,
        expected_pairs=expected_pairs,
        max_baseline_cv=max_baseline_cv,
        max_stereo_rotation_error_deg=max_stereo_rotation_error_deg,
        reprojection_residual_px=reprojection_residual_px,
        max_reprojection_residual_px=max_reprojection_residual_px,
        minimum_predicted_baseline=minimum_predicted_baseline,
    )
    baselines = quality.predicted_baselines_vggt_unit
    if _all_finite(baselines) and int(baselines.shape[0]) > 0:
        if _is_torch(baselines):
            median_predicted = _to_float(baselines.median())
        else:
            median_predicted = float(np.median(baselines))
    else:
        median_predicted = math.nan

    scale_valid = quality.valid and math.isfinite(median_predicted) and (
        median_predicted > minimum_predicted_baseline
    )
    alpha = baseline_m / median_predicted if scale_valid else math.nan
    failure_reason = None if scale_valid else ";".join(quality.failure_reasons)
    if not scale_valid and not failure_reason:
        failure_reason = "invalid_median_predicted_baseline"
    return BaselineScaleEstimate(
        alpha_m_per_vggt_unit=alpha,
        calibrated_baseline_m=baseline_m,
        median_predicted_baseline_vggt_unit=median_predicted,
        quality=quality,
        valid=scale_valid,
        failure_reason=failure_reason,
    )


def scale_vggt_translations_and_depth(
    extrinsics_camera_from_world: Any,
    depth_vggt_unit: Any,
    alpha_m_per_vggt_unit: Real,
) -> tuple[Any, Any]:
    """Apply the same metric scale to pose translations and VGGT depth.

    Rotations and homogeneous bottom rows are left unchanged.  Dense invalid
    depth values (NaN/Inf) propagate rather than being replaced.  The inputs
    are never modified.
    """

    alpha = _finite_positive_scalar(alpha_m_per_vggt_unit, "alpha_m_per_vggt_unit")
    extrinsics = _as_floating(
        extrinsics_camera_from_world, "extrinsics_camera_from_world"
    )
    _extrinsic_parts(extrinsics)
    depth = _as_floating(depth_vggt_unit, "depth_vggt_unit")

    scaled_extrinsics = extrinsics.clone() if _is_torch(extrinsics) else extrinsics.copy()
    scaled_depth_m = depth.clone() if _is_torch(depth) else depth.copy()
    scaled_extrinsics[..., :3, 3] *= alpha
    scaled_depth_m *= alpha
    return scaled_extrinsics, scaled_depth_m


def metric_scale_vggt_geometry(
    extrinsics_camera_from_world: Any,
    depth_vggt_unit: Any,
    calibrated_baseline_m: Real,
    **quality_options: Any,
) -> MetricScaledVGGTGeometry:
    """Validate a pose window, then metric-scale translations and depth.

    ``quality_options`` are forwarded to :func:`estimate_baseline_metric_scale`.
    If validation fails, both scaled payloads are ``None`` so callers cannot
    accidentally use a rejected temporal pose.
    """

    scale = estimate_baseline_metric_scale(
        extrinsics_camera_from_world,
        calibrated_baseline_m,
        **quality_options,
    )
    if not scale.valid:
        return MetricScaledVGGTGeometry(None, None, scale)
    scaled_extrinsics, depth_m = scale_vggt_translations_and_depth(
        extrinsics_camera_from_world,
        depth_vggt_unit,
        scale.alpha_m_per_vggt_unit,
    )
    return MetricScaledVGGTGeometry(scaled_extrinsics, depth_m, scale)


# Explicit compatibility aliases used by early design notes.
camera_center_from_world_to_camera = camera_centers_from_extrinsics
estimate_metric_scale_from_stereo = estimate_baseline_metric_scale
apply_metric_scale = scale_vggt_translations_and_depth


__all__ = [
    "BaselineScaleEstimate",
    "MetricScaledVGGTGeometry",
    "PoseQuality",
    "apply_metric_scale",
    "assess_pose_quality",
    "camera_center_from_world_to_camera",
    "camera_centers_from_extrinsics",
    "estimate_baseline_metric_scale",
    "estimate_metric_scale_from_stereo",
    "metric_scale_vggt_geometry",
    "relative_stereo_rotations",
    "rotation_angle_error_deg",
    "scale_vggt_translations_and_depth",
    "stereo_baselines_from_extrinsics",
]
