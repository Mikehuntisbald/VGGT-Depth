"""Calibrated disparity-to-point-cloud, PLY export, and explicit metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from .disparity import MetricResult


@dataclass(frozen=True)
class PointCloudResult:
    """Organized camera-frame point cloud.

    ``points_camera_m`` has shape ``[B,H,W,3]`` and ``valid_mask`` has shape
    ``[B,H,W]``. Invalid disparities are represented by NaN XYZ values.
    """

    points_camera_m: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class PointCloudExportResult:
    """Receipt for a colored camera-frame PLY export."""

    path: Path
    point_count: int


def _canonical_disparity(disparity_hr_px: Tensor) -> Tensor:
    if not isinstance(disparity_hr_px, Tensor) or not disparity_hr_px.is_floating_point():
        raise TypeError("disparity_hr_px must be a floating-point torch.Tensor")
    if disparity_hr_px.ndim == 2:
        return disparity_hr_px.unsqueeze(0)
    if disparity_hr_px.ndim == 3:
        return disparity_hr_px
    if disparity_hr_px.ndim == 4 and disparity_hr_px.shape[1] == 1:
        return disparity_hr_px[:, 0]
    raise ValueError(
        "disparity_hr_px must have shape [H,W], [B,H,W], or [B,1,H,W]"
    )


def _canonical_intrinsics(
    K_hr_px: Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(K_hr_px, Tensor) or not K_hr_px.is_floating_point():
        raise TypeError("K_hr_px must be a floating-point torch.Tensor")
    if K_hr_px.shape == (3, 3):
        K = K_hr_px.unsqueeze(0).expand(batch_size, -1, -1)
    elif K_hr_px.shape == (batch_size, 3, 3):
        K = K_hr_px
    else:
        raise ValueError(f"K_hr_px must have shape [3,3] or [{batch_size},3,3]")
    K = K.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(K).all().item()):
        raise ValueError("K_hr_px must be finite")
    if not bool(((K[:, 0, 0] > 0) & (K[:, 1, 1] > 0)).all().item()):
        raise ValueError("K_hr_px focal lengths fx and fy must be positive")
    return K


def _canonical_baseline(
    baseline_m: Real | Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    baseline = torch.as_tensor(baseline_m, device=device, dtype=dtype)
    if baseline.ndim == 0:
        baseline = baseline.expand(batch_size)
    elif baseline.shape != (batch_size,):
        raise ValueError(f"baseline_m must be scalar or have shape [{batch_size}]")
    if not bool((torch.isfinite(baseline) & (baseline > 0)).all().item()):
        raise ValueError("baseline_m must be finite and positive")
    return baseline


def disparity_to_point_cloud(
    disparity_hr_px: Tensor,
    K_hr_px: Tensor,
    baseline_m: Real | Tensor,
) -> PointCloudResult:
    """Back-project rectified HR disparity into each left camera frame.

    Pixel coordinates use integer pixel centers ``u=0..W-1, v=0..H-1``.
    Depth is ``Z = fx * baseline_m / disparity_hr_px`` and XYZ are metric
    meters. Inputs may be ``[H,W]``, ``[B,H,W]`` or ``[B,1,H,W]``; output is
    always organized ``[B,H,W,3]``. Computation uses float64 only for float64
    input, otherwise float32 (including bf16/fp16 input).
    """

    disparity = _canonical_disparity(disparity_hr_px)
    batch_size, height, width = disparity.shape
    compute_dtype = (
        torch.float64 if disparity.dtype == torch.float64 else torch.float32
    )
    disparity = disparity.to(dtype=compute_dtype)
    K = _canonical_intrinsics(
        K_hr_px,
        batch_size,
        device=disparity.device,
        dtype=compute_dtype,
    )
    baseline = _canonical_baseline(
        baseline_m,
        batch_size,
        device=disparity.device,
        dtype=compute_dtype,
    )

    valid = torch.isfinite(disparity) & (disparity > 0)
    safe_disparity = torch.where(valid, disparity, torch.ones_like(disparity))
    fx = K[:, 0, 0, None, None]
    fy = K[:, 1, 1, None, None]
    cx = K[:, 0, 2, None, None]
    cy = K[:, 1, 2, None, None]
    depth_m = fx * baseline[:, None, None] / safe_disparity
    u = torch.arange(width, device=disparity.device, dtype=compute_dtype)[
        None, None, :
    ]
    v = torch.arange(height, device=disparity.device, dtype=compute_dtype)[
        None, :, None
    ]
    x_m = (u - cx) * depth_m / fx
    y_m = (v - cy) * depth_m / fy
    points_m = torch.stack((x_m, y_m, depth_m), dim=-1)
    points_m = torch.where(
        valid[..., None],
        points_m,
        torch.full_like(points_m, float("nan")),
    )
    return PointCloudResult(points_camera_m=points_m, valid_mask=valid)


def _canonical_single_rgb(
    color_rgb: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Return HxWx3 uint8 RGB and the pixels with finite RGB input."""

    if not isinstance(color_rgb, Tensor):
        raise TypeError("color_rgb must be a torch.Tensor")
    if color_rgb.ndim == 3 and color_rgb.shape == (height, width, 3):
        color = color_rgb
    elif color_rgb.ndim == 3 and color_rgb.shape == (3, height, width):
        color = color_rgb.permute(1, 2, 0)
    elif color_rgb.ndim == 4 and color_rgb.shape == (1, height, width, 3):
        color = color_rgb[0]
    elif color_rgb.ndim == 4 and color_rgb.shape == (1, 3, height, width):
        color = color_rgb[0].permute(1, 2, 0)
    else:
        raise ValueError(
            "color_rgb must have shape [H,W,3], [3,H,W], [1,H,W,3], "
            "or [1,3,H,W] matching disparity_hr_px"
        )
    if color.dtype == torch.uint8:
        return color, torch.ones((height, width), device=color.device, dtype=torch.bool)
    if not color.is_floating_point():
        raise TypeError("color_rgb must be uint8 or a floating-point torch.Tensor")
    finite = torch.isfinite(color).all(dim=-1)
    finite_values = color[torch.isfinite(color)]
    if finite_values.numel() and not bool(
        ((finite_values >= 0) & (finite_values <= 1)).all().item()
    ):
        raise ValueError("floating-point color_rgb values must be in [0, 1]")
    safe_color = torch.where(torch.isfinite(color), color, torch.zeros_like(color))
    return torch.round(safe_color * 255.0).to(dtype=torch.uint8), finite


def _canonical_single_optional_map(
    value: Tensor,
    *,
    name: str,
    height: int,
    width: int,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.shape == (height, width):
        return value
    if value.shape == (1, height, width):
        return value[0]
    if value.shape == (1, 1, height, width):
        return value[0, 0]
    raise ValueError(
        f"{name} must have shape [H,W], [1,H,W], or [1,1,H,W] matching disparity_hr_px"
    )


def _finite_bound(value: Real | None, *, name: str, positive: bool) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def export_colored_point_cloud_ply(
    disparity_hr_px: Tensor,
    color_rgb: Tensor,
    K_hr_px: Tensor,
    baseline_m: Real | Tensor,
    output_path: str | Path,
    *,
    confidence: Tensor | None = None,
    min_confidence: Real | None = None,
    min_depth_m: Real | None = None,
    max_depth_m: Real | None = None,
) -> PointCloudExportResult:
    """Export one calibrated HR-disparity frame as an ASCII RGB PLY.

    ``K_hr_px`` and ``baseline_m`` are required physical calibration inputs;
    this function never infers calibration.  ``color_rgb`` is uint8 RGB or
    normalized floating RGB in ``[0,1]``.  A pixel is written only when its
    disparity, back-projected XYZ, and (when supplied) confidence are finite,
    disparity is strictly positive, RGB is finite, and all requested confidence
    and depth bounds pass.  Coordinates are left-camera-frame meters.

    Only a single frame is accepted to ensure one PLY has one unambiguous
    camera frame.  No normals, correspondences, or geometric quality metrics
    are generated by this visualization/export utility.
    """

    disparity = _canonical_disparity(disparity_hr_px)
    if disparity.shape[0] != 1:
        raise ValueError("PLY export accepts exactly one disparity frame")
    _, height, width = disparity.shape
    min_confidence_value = _finite_bound(
        min_confidence, name="min_confidence", positive=False
    )
    min_depth_value = _finite_bound(min_depth_m, name="min_depth_m", positive=False)
    max_depth_value = _finite_bound(max_depth_m, name="max_depth_m", positive=True)
    if min_depth_value is not None and min_depth_value < 0:
        raise ValueError("min_depth_m must be finite and non-negative")
    if (
        min_depth_value is not None
        and max_depth_value is not None
        and min_depth_value > max_depth_value
    ):
        raise ValueError("min_depth_m must be less than or equal to max_depth_m")

    cloud = disparity_to_point_cloud(disparity, K_hr_px, baseline_m)
    points = cloud.points_camera_m[0]
    colors, color_valid = _canonical_single_rgb(
        color_rgb, height=height, width=width
    )
    colors = colors.to(device=points.device)
    valid = (
        cloud.valid_mask[0]
        & torch.isfinite(points).all(dim=-1)
        & color_valid.to(device=points.device)
    )
    depth_m = points[..., 2]
    if min_depth_value is not None:
        valid &= depth_m >= min_depth_value
    if max_depth_value is not None:
        valid &= depth_m <= max_depth_value
    if confidence is not None:
        confidence_map = _canonical_single_optional_map(
            confidence, name="confidence", height=height, width=width
        ).to(device=points.device)
        valid &= torch.isfinite(confidence_map)
        if min_confidence_value is not None:
            valid &= confidence_map >= min_confidence_value
    elif min_confidence_value is not None:
        raise ValueError("min_confidence requires confidence")

    selected_points = points[valid].detach().cpu().to(dtype=torch.float64)
    selected_colors = colors[valid].detach().cpu()
    destination = Path(output_path)
    if destination.suffix.lower() != ".ply":
        raise ValueError("output_path must end in .ply")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write("ply\nformat ascii 1.0\n")
            handle.write("comment camera_frame left; coordinates_m\n")
            handle.write(f"element vertex {selected_points.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
            handle.write("end_header\n")
            for point, color in zip(selected_points.tolist(), selected_colors.tolist()):
                handle.write(
                    f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return PointCloudExportResult(
        path=destination,
        point_count=int(selected_points.shape[0]),
    )


def point_to_plane_error(
    source_points_m: Tensor,
    target_points_m: Tensor,
    target_normals: Tensor,
    *,
    correspondence_mask: Tensor | None = None,
    normal_epsilon: float = 1e-12,
) -> MetricResult:
    """Mean absolute point-to-plane error for explicit matched points.

    All point tensors must have the same ``[...,3]`` shape. Element ``i`` in
    source is compared only with element ``i`` in target; this function never
    invents nearest-neighbor correspondences. Target normals are normalized
    internally. Non-finite points/normals and near-zero normals are excluded,
    as is any false ``correspondence_mask`` entry. The result is in meters.
    """

    values = {
        "source_points_m": source_points_m,
        "target_points_m": target_points_m,
        "target_normals": target_normals,
    }
    for name, value in values.items():
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if (
        source_points_m.shape != target_points_m.shape
        or source_points_m.shape != target_normals.shape
    ):
        raise ValueError(
            "source_points_m, target_points_m, and target_normals must have equal shape"
        )
    if source_points_m.ndim < 1 or source_points_m.shape[-1] != 3:
        raise ValueError("point tensors must have shape [...,3]")
    if normal_epsilon <= 0:
        raise ValueError("normal_epsilon must be positive")
    domain_shape = source_points_m.shape[:-1]
    if correspondence_mask is None:
        selected = torch.ones(
            domain_shape, device=source_points_m.device, dtype=torch.bool
        )
    else:
        if (
            not isinstance(correspondence_mask, Tensor)
            or correspondence_mask.shape != domain_shape
        ):
            raise ValueError(
                f"correspondence_mask must have shape {tuple(domain_shape)}"
            )
        selected = correspondence_mask.to(
            device=source_points_m.device, dtype=torch.bool
        )

    source = source_points_m
    target = target_points_m.to(device=source.device, dtype=source.dtype)
    normals = target_normals.to(device=source.device, dtype=source.dtype)
    normal_norm = torch.linalg.vector_norm(normals, dim=-1)
    valid = (
        selected
        & torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(normals).all(dim=-1)
        & torch.isfinite(normal_norm)
        & (normal_norm > normal_epsilon)
    )
    count = int(valid.sum().item())
    if count == 0:
        return MetricResult.invalid()
    unit_normals = normals / normal_norm.clamp_min(normal_epsilon)[..., None]
    residual_m = ((source - target) * unit_normals).sum(dim=-1).abs()
    selected_residual = residual_m[valid]
    numerator = float(selected_residual.to(dtype=torch.float64).sum().item())
    return MetricResult(numerator / count, numerator, count, True)


__all__ = [
    "PointCloudExportResult",
    "PointCloudResult",
    "disparity_to_point_cloud",
    "export_colored_point_cloud_ply",
    "point_to_plane_error",
]
