"""Forward-splat temporal reprojection with an explicit depth z-buffer.

The pose convention in this module is camera-from-world throughout::

    point_camera = E_camera_from_world @ point_world

Therefore a point expressed in the previous camera frame is transformed into
the current camera frame by::

    T_current_previous = E_current @ inverse(E_previous)

The implementation uses nearest-pixel forward splatting.  The continuous
projected coordinate is retained in :class:`WarpResult`, together with its
fractional offset from the selected target pixel, so a later temporal SR head
does not lose the sub-pixel phase.  When several source points select the same
target pixel, only the point with the smallest positive current-camera depth
is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor


Shape2D: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WarpResult:
    """Raster-aligned result of warping one previous frame into the current.

    All tensors are detached from their inputs.  Invalid target pixels contain
    zero in numeric fields and are identified by ``valid_mask``.

    Attributes:
        disparity_hr_px: Current-view rectified disparity in HR pixel units,
            shape ``[B,1,H,W]``.  It is recomputed as
            ``d_previous * Z_previous / Z_current``.
        depth_m: Z coordinate in the current camera, in metres, shape
            ``[B,1,H,W]``.
        confidence: Confidence carried by the winning source point, shape
            ``[B,1,H,W]``.
        valid_mask: Whether at least one valid source point landed in a target
            pixel, boolean shape ``[B,1,H,W]``.
        visibility_mask: Whether the stored point survived the z-buffer,
            boolean shape ``[B,1,H,W]``.  Because only winners are materialised
            in this target-grid representation, it equals ``valid_mask``.
        collision_mask: Whether two or more source points selected the target
            pixel before depth competition, boolean shape ``[B,1,H,W]``.
        projected_uv: Continuous current-image coordinate of the winning
            source point in HR pixels, shape ``[B,2,H,W]`` in ``(u,v)`` order.
        fractional_offset: ``projected_uv - (target_u,target_v)`` in HR pixels,
            shape ``[B,2,H,W]``.  With nearest splatting each component lies in
            ``[-0.5,0.5)`` apart from floating-point boundary effects.
        source_uv: Integer-valued source-grid coordinate of the winning point,
            stored as floating point ``[B,2,H,W]`` in ``(u,v)`` order.  This
            permits RGB photometric residuals to use the exact z-buffer winner.
    """

    disparity_hr_px: Tensor
    depth_m: Tensor
    confidence: Tensor
    valid_mask: Tensor
    visibility_mask: Tensor
    collision_mask: Tensor
    projected_uv: Tensor
    fractional_offset: Tensor
    source_uv: Tensor

    @property
    def disparity(self) -> Tensor:
        """Compatibility alias retaining the explicit HR-pixel field."""

        return self.disparity_hr_px

    @property
    def depth(self) -> Tensor:
        """Compatibility alias retaining the explicit metric-depth field."""

        return self.depth_m


def _image_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}"
        )
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    return value.detach()


def _batched_intrinsics(
    intrinsics_hr_3x3: Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if not isinstance(intrinsics_hr_3x3, Tensor):
        raise TypeError("intrinsics_hr_3x3 must be a torch.Tensor")
    intrinsics = intrinsics_hr_3x3.detach().to(device=device, dtype=dtype)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(
            "intrinsics_hr_3x3 must have shape [3,3] or [B,3,3], got "
            f"{tuple(intrinsics.shape)}"
        )
    if intrinsics.shape[0] == 1 and batch_size != 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1)
    elif intrinsics.shape[0] != batch_size:
        raise ValueError(
            "intrinsics_hr_3x3 batch does not match image batch: "
            f"{intrinsics.shape[0]} != {batch_size}"
        )
    if not bool(torch.isfinite(intrinsics).all().item()):
        raise ValueError("intrinsics_hr_3x3 must contain only finite values")
    if not bool(((intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0)).all()):
        raise ValueError("intrinsics focal lengths must be positive")
    expected_last_row = intrinsics.new_tensor((0.0, 0.0, 1.0))
    if not bool(
        torch.isclose(
            intrinsics[:, 2],
            expected_last_row.expand(batch_size, -1),
            atol=1e-6,
            rtol=0.0,
        ).all()
    ):
        raise ValueError("intrinsics_hr_3x3 must end with row [0,0,1]")
    return intrinsics


def _batched_homogeneous_extrinsics(
    extrinsics_camera_from_world: Tensor,
    *,
    name: str,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if not isinstance(extrinsics_camera_from_world, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    extrinsics = extrinsics_camera_from_world.detach().to(
        device=device, dtype=dtype
    )
    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)
    if extrinsics.ndim != 3 or tuple(extrinsics.shape[-2:]) not in {
        (3, 4),
        (4, 4),
    }:
        raise ValueError(
            f"{name} must have shape [3,4], [4,4], [B,3,4], or [B,4,4], "
            f"got {tuple(extrinsics.shape)}"
        )
    if extrinsics.shape[0] == 1 and batch_size != 1:
        extrinsics = extrinsics.expand(batch_size, -1, -1)
    elif extrinsics.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch does not match image batch: "
            f"{extrinsics.shape[0]} != {batch_size}"
        )
    if not bool(torch.isfinite(extrinsics).all().item()):
        raise ValueError(f"{name} must contain only finite values")

    if tuple(extrinsics.shape[-2:]) == (3, 4):
        homogeneous = torch.zeros(
            (batch_size, 4, 4), dtype=dtype, device=device
        )
        homogeneous[:, :3] = extrinsics
        homogeneous[:, 3, 3] = 1.0
    else:
        homogeneous = extrinsics.clone()
        expected_last_row = homogeneous.new_tensor((0.0, 0.0, 0.0, 1.0))
        if not bool(
            torch.isclose(
                homogeneous[:, 3],
                expected_last_row.expand(batch_size, -1),
                atol=1e-6,
                rtol=0.0,
            ).all()
        ):
            raise ValueError(f"{name} homogeneous last row must be [0,0,0,1]")
    return homogeneous


def _validate_rotation(rotation: Tensor, name: str) -> None:
    batch_size = rotation.shape[0]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    gram = rotation.transpose(-1, -2) @ rotation
    determinant = torch.linalg.det(rotation)
    if not bool(
        torch.isclose(
            gram,
            identity.expand(batch_size, -1, -1),
            atol=1e-4,
            rtol=1e-4,
        ).all()
    ) or not bool(
        torch.isclose(
            determinant,
            torch.ones_like(determinant),
            atol=1e-4,
            rtol=1e-4,
        ).all()
    ):
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")


def relative_camera_transform(
    extrinsics_previous_camera_from_world: Tensor,
    extrinsics_current_camera_from_world: Tensor,
) -> Tensor:
    """Return ``T_current_previous`` for camera-from-world extrinsics.

    Args:
        extrinsics_previous_camera_from_world: Previous camera-from-world pose,
            shape ``[3,4]``, ``[4,4]``, ``[B,3,4]``, or ``[B,4,4]``.
        extrinsics_current_camera_from_world: Current pose with the same
            accepted layouts and a compatible batch.

    Returns:
        Homogeneous current-from-previous transform ``[B,4,4]``.  Unbatched
        inputs return a one-element batch rather than silently changing rank.
    """

    if not isinstance(extrinsics_previous_camera_from_world, Tensor):
        raise TypeError(
            "extrinsics_previous_camera_from_world must be a torch.Tensor"
        )
    if not extrinsics_previous_camera_from_world.is_floating_point():
        raise TypeError(
            "extrinsics_previous_camera_from_world must be floating point"
        )
    previous_batch = (
        1
        if extrinsics_previous_camera_from_world.ndim == 2
        else int(extrinsics_previous_camera_from_world.shape[0])
    )
    if not isinstance(extrinsics_current_camera_from_world, Tensor):
        raise TypeError(
            "extrinsics_current_camera_from_world must be a torch.Tensor"
        )
    current_batch = (
        1
        if extrinsics_current_camera_from_world.ndim == 2
        else int(extrinsics_current_camera_from_world.shape[0])
    )
    batch_size = max(previous_batch, current_batch)
    dtype = torch.promote_types(
        extrinsics_previous_camera_from_world.dtype,
        extrinsics_current_camera_from_world.dtype,
    )
    device = extrinsics_previous_camera_from_world.device
    if extrinsics_current_camera_from_world.device != device:
        raise ValueError("previous and current extrinsics must share a device")
    previous = _batched_homogeneous_extrinsics(
        extrinsics_previous_camera_from_world,
        name="extrinsics_previous_camera_from_world",
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    current = _batched_homogeneous_extrinsics(
        extrinsics_current_camera_from_world,
        name="extrinsics_current_camera_from_world",
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    _validate_rotation(previous[:, :3, :3], "previous extrinsics")
    _validate_rotation(current[:, :3, :3], "current extrinsics")
    inverse_previous = torch.linalg.inv(previous)
    return current @ inverse_previous


def zbuffer_reproject(
    previous_disparity_hr_px: Tensor,
    previous_depth_m: Tensor,
    previous_confidence: Tensor,
    intrinsics_hr_3x3: Tensor,
    extrinsics_previous_camera_from_world: Tensor,
    extrinsics_current_camera_from_world: Tensor,
    *,
    minimum_depth_m: float = 1e-6,
) -> WarpResult:
    """Forward-project a previous disparity/depth map into the current view.

    Args:
        previous_disparity_hr_px: Previous rectified disparity in HR pixels,
            shape ``[B,1,H,W]``.
        previous_depth_m: Previous camera Z depth in metres, same shape.
        previous_confidence: Previous confidence, same shape.  Finite values
            are propagated without thresholding or clipping.
        intrinsics_hr_3x3: Calibrated HR pinhole intrinsics, shape ``[3,3]`` or
            ``[B,3,3]``.  The same calibration is used for both video frames.
        extrinsics_previous_camera_from_world: Previous camera-from-world pose,
            shape ``[3,4]``, ``[4,4]``, ``[B,3,4]``, or ``[B,4,4]``.
        extrinsics_current_camera_from_world: Current camera-from-world pose in
            one of the same layouts.
        minimum_depth_m: Strict lower bound for a projected point's current
            camera Z coordinate.

    Returns:
        :class:`WarpResult` on the current ``H x W`` target grid.

    Notes:
        The function is intentionally non-differentiable: inputs are detached,
        target pixels are selected by nearest-neighbour rounding, and z-buffer
        winners are selected with integer scatter reductions.
    """

    disparity_hr_px = _image_tensor(
        previous_disparity_hr_px, "previous_disparity_hr_px"
    )
    depth_m = _image_tensor(previous_depth_m, "previous_depth_m")
    confidence = _image_tensor(previous_confidence, "previous_confidence")
    if depth_m.shape != disparity_hr_px.shape:
        raise ValueError(
            "previous_depth_m shape must match previous_disparity_hr_px: "
            f"{tuple(depth_m.shape)} != {tuple(disparity_hr_px.shape)}"
        )
    if confidence.shape != disparity_hr_px.shape:
        raise ValueError(
            "previous_confidence shape must match previous_disparity_hr_px: "
            f"{tuple(confidence.shape)} != {tuple(disparity_hr_px.shape)}"
        )
    if depth_m.device != disparity_hr_px.device or confidence.device != disparity_hr_px.device:
        raise ValueError("all image tensors must share a device")
    if not torch.isfinite(torch.as_tensor(minimum_depth_m)) or minimum_depth_m <= 0:
        raise ValueError("minimum_depth_m must be finite and > 0")

    batch_size, _, height, width = disparity_hr_px.shape
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("image tensors must have positive B, H, and W dimensions")
    compute_dtype = torch.promote_types(disparity_hr_px.dtype, depth_m.dtype)
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    device = disparity_hr_px.device
    disparity = disparity_hr_px.to(dtype=compute_dtype)
    depth = depth_m.to(dtype=compute_dtype)
    confidence_compute = confidence.to(dtype=compute_dtype)
    intrinsics = _batched_intrinsics(
        intrinsics_hr_3x3,
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    previous_extrinsics = _batched_homogeneous_extrinsics(
        extrinsics_previous_camera_from_world,
        name="extrinsics_previous_camera_from_world",
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    current_extrinsics = _batched_homogeneous_extrinsics(
        extrinsics_current_camera_from_world,
        name="extrinsics_current_camera_from_world",
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    _validate_rotation(previous_extrinsics[:, :3, :3], "previous extrinsics")
    _validate_rotation(current_extrinsics[:, :3, :3], "current extrinsics")
    transform_current_previous = current_extrinsics @ torch.linalg.inv(
        previous_extrinsics
    )

    grid_v, grid_u = torch.meshgrid(
        torch.arange(height, dtype=compute_dtype, device=device),
        torch.arange(width, dtype=compute_dtype, device=device),
        indexing="ij",
    )
    grid_u = grid_u.reshape(1, -1).expand(batch_size, -1)
    grid_v = grid_v.reshape(1, -1).expand(batch_size, -1)
    depth_flat_m = depth[:, 0].reshape(batch_size, -1)
    disparity_flat_hr_px = disparity[:, 0].reshape(batch_size, -1)
    confidence_flat = confidence_compute[:, 0].reshape(batch_size, -1)

    fx_px = intrinsics[:, 0, 0:1]
    fy_px = intrinsics[:, 1, 1:2]
    cx_px = intrinsics[:, 0, 2:3]
    cy_px = intrinsics[:, 1, 2:3]
    point_previous = torch.stack(
        (
            (grid_u - cx_px) * depth_flat_m / fx_px,
            (grid_v - cy_px) * depth_flat_m / fy_px,
            depth_flat_m,
            torch.ones_like(depth_flat_m),
        ),
        dim=1,
    )
    point_current = transform_current_previous @ point_previous
    x_current = point_current[:, 0]
    y_current = point_current[:, 1]
    z_current_m = point_current[:, 2]
    projected_u = fx_px * x_current / z_current_m + cx_px
    projected_v = fy_px * y_current / z_current_m + cy_px

    source_valid = (
        torch.isfinite(disparity_flat_hr_px)
        & (disparity_flat_hr_px > 0)
        & torch.isfinite(depth_flat_m)
        & (depth_flat_m > 0)
        & torch.isfinite(confidence_flat)
        & torch.isfinite(projected_u)
        & torch.isfinite(projected_v)
        & torch.isfinite(z_current_m)
        & (z_current_m > minimum_depth_m)
    )
    target_u = torch.floor(projected_u + 0.5).to(dtype=torch.long)
    target_v = torch.floor(projected_v + 0.5).to(dtype=torch.long)
    source_valid &= (
        (target_u >= 0)
        & (target_u < width)
        & (target_v >= 0)
        & (target_v < height)
    )

    pixels_per_image = height * width
    source_linear = torch.arange(
        batch_size * pixels_per_image, dtype=torch.long, device=device
    ).reshape(batch_size, pixels_per_image)
    batch_offset = (
        torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1)
        * pixels_per_image
    )
    target_linear = batch_offset + target_v * width + target_u
    valid_source_linear = source_linear[source_valid]
    valid_target_linear = target_linear[source_valid]
    valid_depth_current_m = z_current_m[source_valid]

    output_size = batch_size * pixels_per_image
    zbuffer_m = torch.full(
        (output_size,), torch.inf, dtype=compute_dtype, device=device
    )
    collision_count = torch.zeros(
        (output_size,), dtype=torch.int64, device=device
    )
    if valid_target_linear.numel() > 0:
        zbuffer_m.scatter_reduce_(
            0,
            valid_target_linear,
            valid_depth_current_m,
            reduce="amin",
            include_self=True,
        )
        collision_count.scatter_add_(
            0,
            valid_target_linear,
            torch.ones_like(valid_target_linear, dtype=torch.int64),
        )

    # For exactly tied depths, retain the lowest source index deterministically.
    winning_depth = (
        valid_depth_current_m == zbuffer_m[valid_target_linear]
        if valid_target_linear.numel() > 0
        else torch.empty(0, dtype=torch.bool, device=device)
    )
    winner_source = torch.full(
        (output_size,), output_size, dtype=torch.long, device=device
    )
    if bool(winning_depth.any().item()):
        winner_source.scatter_reduce_(
            0,
            valid_target_linear[winning_depth],
            valid_source_linear[winning_depth],
            reduce="amin",
            include_self=True,
        )

    output_valid_flat = winner_source != output_size
    output_target_linear = torch.nonzero(
        output_valid_flat, as_tuple=False
    ).squeeze(1)
    output_source_linear = winner_source[output_valid_flat]

    output_disparity_hr_px = torch.zeros(
        output_size, dtype=compute_dtype, device=device
    )
    output_depth_m = torch.zeros(output_size, dtype=compute_dtype, device=device)
    output_confidence = torch.zeros(
        output_size, dtype=compute_dtype, device=device
    )
    output_projected_uv = torch.zeros(
        (output_size, 2), dtype=compute_dtype, device=device
    )
    output_fractional_offset = torch.zeros_like(output_projected_uv)
    output_source_uv = torch.zeros_like(output_projected_uv)

    if output_target_linear.numel() > 0:
        source_batch = torch.div(
            output_source_linear, pixels_per_image, rounding_mode="floor"
        )
        source_pixel = output_source_linear % pixels_per_image
        winning_previous_depth_m = depth_flat_m[source_batch, source_pixel]
        winning_previous_disparity_hr_px = disparity_flat_hr_px[
            source_batch, source_pixel
        ]
        winning_current_depth_m = z_current_m[source_batch, source_pixel]
        winning_projected_u = projected_u[source_batch, source_pixel]
        winning_projected_v = projected_v[source_batch, source_pixel]
        output_disparity_hr_px[output_target_linear] = (
            winning_previous_disparity_hr_px
            * winning_previous_depth_m
            / winning_current_depth_m
        )
        output_depth_m[output_target_linear] = winning_current_depth_m
        output_confidence[output_target_linear] = confidence_flat[
            source_batch, source_pixel
        ]
        output_projected_uv[output_target_linear, 0] = winning_projected_u
        output_projected_uv[output_target_linear, 1] = winning_projected_v
        output_target_u = (output_target_linear % pixels_per_image) % width
        output_target_v = torch.div(
            output_target_linear % pixels_per_image,
            width,
            rounding_mode="floor",
        )
        output_fractional_offset[output_target_linear, 0] = (
            winning_projected_u - output_target_u.to(dtype=compute_dtype)
        )
        output_fractional_offset[output_target_linear, 1] = (
            winning_projected_v - output_target_v.to(dtype=compute_dtype)
        )
        output_source_uv[output_target_linear, 0] = (
            source_pixel % width
        ).to(dtype=compute_dtype)
        output_source_uv[output_target_linear, 1] = torch.div(
            source_pixel, width, rounding_mode="floor"
        ).to(dtype=compute_dtype)

    output_shape = (batch_size, 1, height, width)
    vector_output_shape = (batch_size, height, width, 2)
    valid_mask = output_valid_flat.reshape(output_shape)
    return WarpResult(
        disparity_hr_px=output_disparity_hr_px.reshape(output_shape).to(
            dtype=previous_disparity_hr_px.dtype
        ),
        depth_m=output_depth_m.reshape(output_shape).to(dtype=previous_depth_m.dtype),
        confidence=output_confidence.reshape(output_shape).to(
            dtype=previous_confidence.dtype
        ),
        valid_mask=valid_mask,
        visibility_mask=valid_mask.clone(),
        collision_mask=(collision_count > 1).reshape(output_shape),
        projected_uv=output_projected_uv.reshape(vector_output_shape).permute(
            0, 3, 1, 2
        ),
        fractional_offset=output_fractional_offset.reshape(
            vector_output_shape
        ).permute(0, 3, 1, 2),
        source_uv=output_source_uv.reshape(vector_output_shape).permute(
            0, 3, 1, 2
        ),
    )


forward_splat_zbuffer = zbuffer_reproject
reproject_with_zbuffer = zbuffer_reproject


__all__ = [
    "WarpResult",
    "forward_splat_zbuffer",
    "relative_camera_transform",
    "reproject_with_zbuffer",
    "zbuffer_reproject",
]
