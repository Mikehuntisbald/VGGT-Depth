"""Differentiable stereo and temporal reprojection for metric video losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReprojectionResult:
    image: Tensor
    valid_mask: Tensor
    occlusion_mask: Tensor
    projected_uv_px: Tensor
    projected_depth_m: Tensor | None = None
    sampled_source_depth_m: Tensor | None = None


def _assert_tensor_condition(condition: Tensor, message: str) -> None:
    """Validate one value invariant without synchronizing a valid CUDA path."""

    if condition.dtype != torch.bool or condition.numel() != 1:
        raise TypeError("validation condition must be a one-element bool Tensor")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


def _images(reference: Tensor, source: Tensor) -> tuple[int, int, int]:
    if reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("reference image must have shape [B,3,H,W]")
    if source.shape != reference.shape:
        raise ValueError("source image must match the reference image shape")
    if not reference.is_floating_point() or not source.is_floating_point():
        raise TypeError("images must be floating point")
    if reference.device != source.device:
        raise ValueError("images must share one device")
    return reference.shape[0], reference.shape[2], reference.shape[3]


def _scalar_image(value: Tensor, *, name: str, batch: int, height: int, width: int) -> None:
    if value.shape != (batch, 1, height, width) or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point [B,1,H,W]")


def _matrix(value: Tensor, *, name: str, batch: int, size: int, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.shape == (1, size, size) and batch != 1:
        value = value.expand(batch, -1, -1)
    if value.shape != (batch, size, size):
        raise ValueError(f"{name} must have shape [{size},{size}] or [B,{size},{size}]")
    if value.device != device:
        raise ValueError(f"{name} must be finite and share the image device")
    _assert_tensor_condition(
        torch.isfinite(value).all(),
        f"{name} must be finite and share the image device",
    )
    return value


def _grid_from_uv(uv: Tensor, height: int, width: int) -> Tensor:
    u, v = uv[:, 0], uv[:, 1]
    return torch.stack(
        (2.0 * (u + 0.5) / width - 1.0, 2.0 * (v + 0.5) / height - 1.0),
        dim=-1,
    )


def stereo_reproject_right_to_left(
    left_rgb: Tensor,
    right_rgb: Tensor,
    disparity_left_px: Tensor,
) -> ReprojectionResult:
    """Sample the right image at ``x_right=x_left-disparity_left``."""

    batch, height, width = _images(left_rgb, right_rgb)
    _scalar_image(
        disparity_left_px,
        name="disparity_left_px",
        batch=batch,
        height=height,
        width=width,
    )
    if disparity_left_px.device != left_rgb.device:
        raise ValueError("disparity and RGB must share one device")
    y, x = torch.meshgrid(
        torch.arange(height, device=left_rgb.device, dtype=torch.float32),
        torch.arange(width, device=left_rgb.device, dtype=torch.float32),
        indexing="ij",
    )
    u = x.unsqueeze(0) - disparity_left_px[:, 0].float()
    v = y.expand(batch, -1, -1)
    finite_positive = torch.isfinite(disparity_left_px[:, 0]) & (
        disparity_left_px[:, 0] > 0
    )
    valid = finite_positive & (u >= 0) & (u <= width - 1)
    safe_uv = torch.stack(
        (torch.where(valid, u, torch.zeros_like(u)), v), dim=1
    )
    warped = F.grid_sample(
        right_rgb,
        _grid_from_uv(safe_uv, height, width),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid_mask = valid.unsqueeze(1)
    return ReprojectionResult(
        image=torch.where(valid_mask, warped, torch.zeros_like(warped)),
        valid_mask=valid_mask,
        occlusion_mask=torch.zeros_like(valid_mask),
        projected_uv_px=safe_uv,
    )


def temporal_reproject_previous_to_current(
    current_rgb: Tensor,
    previous_rgb: Tensor,
    current_inverse_depth_m_inv: Tensor,
    intrinsics_current_3x3: Tensor,
    intrinsics_previous_3x3: Tensor,
    T_current_from_previous_m: Tensor,
    *,
    previous_inverse_depth_m_inv: Tensor | None = None,
    relative_depth_tolerance: float = 0.05,
    absolute_depth_tolerance_m: float = 0.05,
) -> ReprojectionResult:
    """Backward-warp the previous RGB using current metric inverse depth.

    Projection coordinates remain differentiable with respect to current
    inverse depth.  When previous inverse depth is supplied, a projected point
    behind the sampled previous surface is marked occluded.
    """

    batch, height, width = _images(current_rgb, previous_rgb)
    _scalar_image(
        current_inverse_depth_m_inv,
        name="current_inverse_depth_m_inv",
        batch=batch,
        height=height,
        width=width,
    )
    if current_inverse_depth_m_inv.device != current_rgb.device:
        raise ValueError("inverse depth and RGB must share one device")
    if relative_depth_tolerance < 0 or absolute_depth_tolerance_m < 0:
        raise ValueError("depth tolerances must be non-negative")
    current_K = _matrix(
        intrinsics_current_3x3,
        name="intrinsics_current_3x3",
        batch=batch,
        size=3,
        device=current_rgb.device,
    ).float()
    previous_K = _matrix(
        intrinsics_previous_3x3,
        name="intrinsics_previous_3x3",
        batch=batch,
        size=3,
        device=current_rgb.device,
    ).float()
    current_from_previous = _matrix(
        T_current_from_previous_m,
        name="T_current_from_previous_m",
        batch=batch,
        size=4,
        device=current_rgb.device,
    ).float()

    inverse = current_inverse_depth_m_inv.float()
    inverse_valid = torch.isfinite(inverse) & (inverse > 0)
    safe_inverse = torch.where(inverse_valid, inverse, torch.ones_like(inverse))
    depth_current = safe_inverse.reciprocal()
    y, x = torch.meshgrid(
        torch.arange(height, device=current_rgb.device, dtype=torch.float32),
        torch.arange(width, device=current_rgb.device, dtype=torch.float32),
        indexing="ij",
    )
    x = x.reshape(1, 1, height, width)
    y = y.reshape(1, 1, height, width)
    fx = current_K[:, 0, 0].reshape(batch, 1, 1, 1)
    fy = current_K[:, 1, 1].reshape(batch, 1, 1, 1)
    cx = current_K[:, 0, 2].reshape(batch, 1, 1, 1)
    cy = current_K[:, 1, 2].reshape(batch, 1, 1, 1)
    points_current = torch.cat(
        (
            (x - cx) * depth_current / fx,
            (y - cy) * depth_current / fy,
            depth_current,
            torch.ones_like(depth_current),
        ),
        dim=1,
    ).flatten(2)
    previous_from_current, inverse_info = torch.linalg.inv_ex(
        current_from_previous, check_errors=False
    )
    _assert_tensor_condition(
        (inverse_info == 0).all(), "T_current_from_previous_m must be invertible"
    )
    points_previous = (previous_from_current @ points_current)[:, :3]
    z_previous = points_previous[:, 2:3].reshape(batch, 1, height, width)
    safe_z = z_previous.clamp_min(1e-8)
    u_previous = (
        previous_K[:, 0, 0].reshape(batch, 1, 1) * points_previous[:, 0]
        / safe_z.flatten(2)[:, 0]
        + previous_K[:, 0, 2].reshape(batch, 1, 1)
    ).reshape(batch, height, width)
    v_previous = (
        previous_K[:, 1, 1].reshape(batch, 1, 1) * points_previous[:, 1]
        / safe_z.flatten(2)[:, 0]
        + previous_K[:, 1, 2].reshape(batch, 1, 1)
    ).reshape(batch, height, width)
    projected_uv = torch.stack((u_previous, v_previous), dim=1)
    coordinate_valid = (
        inverse_valid[:, 0]
        & torch.isfinite(u_previous)
        & torch.isfinite(v_previous)
        & torch.isfinite(z_previous[:, 0])
        & (z_previous[:, 0] > 0)
        & (u_previous >= 0)
        & (u_previous <= width - 1)
        & (v_previous >= 0)
        & (v_previous <= height - 1)
    )
    safe_uv = torch.where(
        coordinate_valid.unsqueeze(1), projected_uv, torch.zeros_like(projected_uv)
    )
    grid = _grid_from_uv(safe_uv, height, width)
    warped = F.grid_sample(
        previous_rgb,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    sampled_source_depth: Tensor | None = None
    occluded = torch.zeros(
        (batch, 1, height, width), dtype=torch.bool, device=current_rgb.device
    )
    source_support = torch.ones_like(occluded)
    if previous_inverse_depth_m_inv is not None:
        _scalar_image(
            previous_inverse_depth_m_inv,
            name="previous_inverse_depth_m_inv",
            batch=batch,
            height=height,
            width=width,
        )
        if previous_inverse_depth_m_inv.device != current_rgb.device:
            raise ValueError("previous inverse depth must share the RGB device")
        previous_valid = torch.isfinite(previous_inverse_depth_m_inv) & (
            previous_inverse_depth_m_inv > 0
        )
        previous_depth = torch.where(
            previous_valid,
            previous_inverse_depth_m_inv.float().clamp_min(1e-8).reciprocal(),
            torch.zeros_like(previous_inverse_depth_m_inv, dtype=torch.float32),
        )
        sampled_source_depth = F.grid_sample(
            previous_depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_support = F.grid_sample(
            previous_valid.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        source_support = sampled_support >= 1.0 - 1e-5
        tolerance = absolute_depth_tolerance_m + relative_depth_tolerance * sampled_source_depth
        occluded = source_support & (z_previous > sampled_source_depth + tolerance)

    valid = coordinate_valid.unsqueeze(1) & source_support & ~occluded
    return ReprojectionResult(
        image=torch.where(valid, warped, torch.zeros_like(warped)),
        valid_mask=valid,
        occlusion_mask=occluded,
        projected_uv_px=safe_uv,
        projected_depth_m=z_previous,
        sampled_source_depth_m=sampled_source_depth,
    )


__all__ = [
    "ReprojectionResult",
    "stereo_reproject_right_to_left",
    "temporal_reproject_previous_to_current",
]
