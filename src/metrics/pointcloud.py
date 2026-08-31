"""Calibrated disparity-to-point-cloud and matched point-to-plane metrics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

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


__all__ = ["PointCloudResult", "disparity_to_point_cloud", "point_to_plane_error"]
