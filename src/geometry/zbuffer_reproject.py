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
import math
from numbers import Real
from typing import TypeAlias

import torch
from torch import Tensor


Shape2D: TypeAlias = tuple[int, int]


def _assert_tensor_condition(condition: Tensor, message: str) -> None:
    """Validate one value invariant without synchronizing a valid CUDA path."""

    if condition.dtype != torch.bool or condition.numel() != 1:
        raise TypeError("validation condition must be a one-element bool Tensor")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class WarpResult:
    """Raster-aligned result of warping one previous frame into the current.

    All tensors are detached from their inputs.  Invalid target pixels contain
    zero in numeric fields and are identified by ``valid_mask``.

    Attributes:
        disparity_hr_px: Current-view rectified disparity in HR pixel units,
            shape ``[B,1,H,W]``.  It is recomputed as
            ``d_previous * Z_previous / Z_current`` in legacy mode or exactly
            ``fx_target * B_target / Z_current`` under dual calibration.
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
    _assert_tensor_condition(
        torch.isfinite(intrinsics).all(),
        "intrinsics_hr_3x3 must contain only finite values",
    )
    _assert_tensor_condition(
        ((intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0)).all(),
        "intrinsics focal lengths must be positive",
    )
    _assert_tensor_condition(
        torch.isclose(
            intrinsics[:, 2, :2],
            torch.zeros_like(intrinsics[:, 2, :2]),
            atol=1e-6,
            rtol=0.0,
        ).all()
        & torch.isclose(
            intrinsics[:, 2, 2],
            torch.ones_like(intrinsics[:, 2, 2]),
            atol=1e-6,
            rtol=0.0,
        ).all(),
        "intrinsics_hr_3x3 must end with row [0,0,1]",
    )
    return intrinsics


def _batched_positive_scalar(
    value: Tensor,
    *,
    name: str,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Normalize a positive per-camera scalar to shape ``[B,1]``."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().to(device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1, 1)
    elif result.ndim == 1:
        result = result.reshape(-1, 1)
    elif result.ndim == 2 and result.shape[1] == 1:
        pass
    else:
        raise ValueError(
            f"{name} must have shape [], [B], or [B,1], got {tuple(result.shape)}"
        )
    if result.shape[0] == 1 and batch_size != 1:
        result = result.expand(batch_size, -1)
    elif result.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch does not match image batch: "
            f"{result.shape[0]} != {batch_size}"
        )
    _assert_tensor_condition(
        torch.isfinite(result).all() & (result > 0).all(),
        f"{name} must contain only finite positive values",
    )
    return result


def _metric_depth_from_explicit_stereo_disparity(
    disparity_hr_px: Tensor,
    supplied_depth_m: Tensor,
    *,
    intrinsics_source_hr_3x3: Tensor,
    baseline_source_m: Tensor,
    disparity_storage_dtype: torch.dtype,
    depth_storage_dtype: torch.dtype,
    name: str,
) -> Tensor:
    """Recompute source depth from explicit stereo calibration and fail closed.

    The explicit dual-calibration path must not merely carry an independently
    supplied depth alongside disparity.  ``fx_source * B_source / d_source``
    owns source metric depth; the supplied tensor is retained as a cache/API
    consistency witness and must agree at every positive-disparity pixel.
    """

    batch_size = disparity_hr_px.shape[0]
    numerator_m_px = (
        intrinsics_source_hr_3x3[:, 0, 0]
        * baseline_source_m.reshape(batch_size)
    ).reshape(batch_size, 1, 1, 1)
    disparity_valid = torch.isfinite(disparity_hr_px) & (disparity_hr_px > 0)
    recomputed_depth_m = torch.where(
        disparity_valid,
        numerator_m_px / disparity_hr_px.clamp_min(1e-12),
        torch.zeros_like(disparity_hr_px),
    )
    supplied_valid = torch.isfinite(supplied_depth_m) & (supplied_depth_m > 0)
    # Disparity and its cached depth witness may have been rounded
    # independently before this FP32 geometry path receives them.  For a
    # normal floating-point value the relative rounding error is bounded by
    # half an epsilon.  The product below bounds the combined error of the two
    # stored values; a small FP32 allowance covers the recomputation itself.
    disparity_roundoff = float(torch.finfo(disparity_storage_dtype).eps) / 2.0
    depth_roundoff = float(torch.finfo(depth_storage_dtype).eps) / 2.0
    quantization_rtol = (
        (1.0 + disparity_roundoff) * (1.0 + depth_roundoff) - 1.0
    ) + 8.0 * float(torch.finfo(disparity_hr_px.dtype).eps)
    consistency_rtol = max(2e-4, quantization_rtol)
    consistent = torch.isclose(
        supplied_depth_m,
        recomputed_depth_m,
        atol=1e-6,
        rtol=consistency_rtol,
    )
    mismatch = disparity_valid & (~supplied_valid | ~consistent)
    _assert_tensor_condition(
        ~mismatch.any(),
        f"{name} is inconsistent with source fx*baseline/disparity",
    )
    return recomputed_depth_m


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
    _assert_tensor_condition(
        torch.isfinite(extrinsics).all(), f"{name} must contain only finite values"
    )

    if tuple(extrinsics.shape[-2:]) == (3, 4):
        homogeneous = torch.zeros(
            (batch_size, 4, 4), dtype=dtype, device=device
        )
        homogeneous[:, :3] = extrinsics
        homogeneous[:, 3, 3] = 1.0
    else:
        homogeneous = extrinsics.clone()
        _assert_tensor_condition(
            torch.isclose(
                homogeneous[:, 3, :3],
                torch.zeros_like(homogeneous[:, 3, :3]),
                atol=1e-6,
                rtol=0.0,
            ).all()
            & torch.isclose(
                homogeneous[:, 3, 3],
                torch.ones_like(homogeneous[:, 3, 3]),
                atol=1e-6,
                rtol=0.0,
            ).all(),
            f"{name} homogeneous last row must be [0,0,0,1]",
        )
    return homogeneous


def _validate_rotation(rotation: Tensor, name: str) -> None:
    batch_size = rotation.shape[0]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    gram = rotation.transpose(-1, -2) @ rotation
    determinant = torch.linalg.det(rotation)
    _assert_tensor_condition(
        torch.isclose(
            gram,
            identity.expand(batch_size, -1, -1),
            atol=1e-4,
            rtol=1e-4,
        ).all()
        & torch.isclose(
            determinant,
            torch.ones_like(determinant),
            atol=1e-4,
            rtol=1e-4,
        ).all(),
        f"{name} rotation must be a proper orthonormal matrix",
    )


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
    inverse_previous, inverse_info = torch.linalg.inv_ex(
        previous, check_errors=False
    )
    _assert_tensor_condition(
        (inverse_info == 0).all(), "previous extrinsics must be invertible"
    )
    return current @ inverse_previous


def zbuffer_reproject(
    previous_disparity_hr_px: Tensor,
    previous_depth_m: Tensor,
    previous_confidence: Tensor,
    intrinsics_hr_3x3: Tensor,
    extrinsics_previous_camera_from_world: Tensor,
    extrinsics_current_camera_from_world: Tensor,
    *,
    intrinsics_current_hr_3x3: Tensor | None = None,
    baseline_previous_m: Tensor | None = None,
    baseline_current_m: Tensor | None = None,
    minimum_depth_m: float = 1e-6,
) -> WarpResult:
    """Forward-project a previous disparity/depth map into the current view.

    Args:
        previous_disparity_hr_px: Previous rectified disparity in HR pixels,
            shape ``[B,1,H,W]``.
        previous_depth_m: Previous camera Z depth in metres, same shape.
            With explicit source/target baselines this becomes a consistency
            witness: source depth is recomputed as ``fx_source*B_source/d``
            and any positive-disparity disagreement fails closed.
        previous_confidence: Previous confidence, same shape.  Finite values
            are propagated without thresholding or clipping.
        intrinsics_hr_3x3: Calibrated *source* HR pinhole intrinsics, shape
            ``[3,3]`` or ``[B,3,3]``.
        extrinsics_previous_camera_from_world: Previous camera-from-world pose,
            shape ``[3,4]``, ``[4,4]``, ``[B,3,4]``, or ``[B,4,4]``.
        extrinsics_current_camera_from_world: Current camera-from-world pose in
            one of the same layouts.
        intrinsics_current_hr_3x3: Optional calibrated target intrinsics. When
            omitted the source intrinsics are reused for exact legacy behavior.
        baseline_previous_m: Optional positive source stereo baseline ``[B]``.
        baseline_current_m: Optional positive target stereo baseline ``[B]``.
            The baselines must be supplied together. In that explicit mode,
            output disparity is recomputed in target HR-pixel units as
            ``fx_t * B_t / Z_t``.
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
    if (
        isinstance(minimum_depth_m, bool)
        or not isinstance(minimum_depth_m, Real)
        or not math.isfinite(float(minimum_depth_m))
        or minimum_depth_m <= 0
    ):
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
    intrinsics_previous = _batched_intrinsics(
        intrinsics_hr_3x3,
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    intrinsics_current = _batched_intrinsics(
        (
            intrinsics_hr_3x3
            if intrinsics_current_hr_3x3 is None
            else intrinsics_current_hr_3x3
        ),
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    if (baseline_previous_m is None) != (baseline_current_m is None):
        raise ValueError(
            "baseline_previous_m and baseline_current_m must be supplied together"
        )
    if intrinsics_current_hr_3x3 is not None and baseline_previous_m is None:
        raise ValueError(
            "explicit current intrinsics require source and target baselines"
        )
    target_disparity_numerator_m_px: Tensor | None = None
    if baseline_previous_m is not None and baseline_current_m is not None:
        source_baseline = _batched_positive_scalar(
            baseline_previous_m,
            name="baseline_previous_m",
            batch_size=batch_size,
            dtype=compute_dtype,
            device=device,
        )
        target_baseline = _batched_positive_scalar(
            baseline_current_m,
            name="baseline_current_m",
            batch_size=batch_size,
            dtype=compute_dtype,
            device=device,
        )
        depth = _metric_depth_from_explicit_stereo_disparity(
            disparity,
            depth,
            intrinsics_source_hr_3x3=intrinsics_previous,
            baseline_source_m=source_baseline,
            disparity_storage_dtype=disparity_hr_px.dtype,
            depth_storage_dtype=depth_m.dtype,
            name="previous_depth_m",
        )
        target_disparity_numerator_m_px = (
            intrinsics_current[:, 0, 0:1] * target_baseline
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
    inverse_previous, inverse_info = torch.linalg.inv_ex(
        previous_extrinsics, check_errors=False
    )
    _assert_tensor_condition(
        (inverse_info == 0).all(), "previous extrinsics must be invertible"
    )
    transform_current_previous = current_extrinsics @ inverse_previous

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

    fx_previous_px = intrinsics_previous[:, 0, 0:1]
    fy_previous_px = intrinsics_previous[:, 1, 1:2]
    cx_previous_px = intrinsics_previous[:, 0, 2:3]
    cy_previous_px = intrinsics_previous[:, 1, 2:3]
    fx_current_px = intrinsics_current[:, 0, 0:1]
    fy_current_px = intrinsics_current[:, 1, 1:2]
    cx_current_px = intrinsics_current[:, 0, 2:3]
    cy_current_px = intrinsics_current[:, 1, 2:3]
    point_previous = torch.stack(
        (
            (grid_u - cx_previous_px) * depth_flat_m / fx_previous_px,
            (grid_v - cy_previous_px) * depth_flat_m / fy_previous_px,
            depth_flat_m,
            torch.ones_like(depth_flat_m),
        ),
        dim=1,
    )
    point_current = transform_current_previous @ point_previous
    x_current = point_current[:, 0]
    y_current = point_current[:, 1]
    z_current_m = point_current[:, 2]
    projected_u = fx_current_px * x_current / z_current_m + cx_current_px
    projected_v = fy_current_px * y_current / z_current_m + cy_current_px

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
    output_size = batch_size * pixels_per_image
    source_linear = source_linear.reshape(output_size)
    source_valid = source_valid.reshape(output_size)
    target_linear = target_linear.reshape(output_size)
    depth_current_flat = z_current_m.reshape(output_size)
    safe_target_linear = torch.where(
        source_valid, target_linear, torch.zeros_like(target_linear)
    )
    safe_depth_current_m = torch.where(
        source_valid,
        depth_current_flat,
        torch.full_like(depth_current_flat, torch.inf),
    )
    zbuffer_m = torch.full(
        (output_size,), torch.inf, dtype=compute_dtype, device=device
    )
    collision_count = torch.zeros(
        (output_size,), dtype=torch.int64, device=device
    )
    zbuffer_m.scatter_reduce_(
        0,
        safe_target_linear,
        safe_depth_current_m,
        reduce="amin",
        include_self=True,
    )
    collision_count.scatter_add_(
        0,
        safe_target_linear,
        source_valid.to(dtype=torch.int64),
    )

    # For exactly tied depths, retain the lowest source index deterministically.
    winning_depth = source_valid & (
        depth_current_flat == zbuffer_m[safe_target_linear]
    )
    winner_source = torch.full(
        (output_size,), output_size, dtype=torch.long, device=device
    )
    winner_candidates = torch.where(
        winning_depth,
        source_linear,
        torch.full_like(source_linear, output_size),
    )
    winner_source.scatter_reduce_(
        0,
        safe_target_linear,
        winner_candidates,
        reduce="amin",
        include_self=True,
    )

    output_valid_flat = winner_source != output_size
    output_source_linear = torch.where(
        output_valid_flat, winner_source, torch.zeros_like(winner_source)
    )
    source_batch = torch.div(
        output_source_linear, pixels_per_image, rounding_mode="floor"
    )
    source_pixel = output_source_linear % pixels_per_image
    winning_previous_depth_m = depth_flat_m.reshape(output_size)[
        output_source_linear
    ]
    winning_previous_disparity_hr_px = disparity_flat_hr_px.reshape(output_size)[
        output_source_linear
    ]
    winning_current_depth_m = depth_current_flat[output_source_linear]
    winning_projected_u = projected_u.reshape(output_size)[output_source_linear]
    winning_projected_v = projected_v.reshape(output_size)[output_source_linear]
    winning_disparity_target_hr_px = (
        winning_previous_disparity_hr_px
        * winning_previous_depth_m
        / winning_current_depth_m.clamp_min(torch.finfo(compute_dtype).tiny)
    )
    if target_disparity_numerator_m_px is not None:
        winning_disparity_target_hr_px = (
            target_disparity_numerator_m_px[source_batch, 0]
            / winning_current_depth_m.clamp_min(torch.finfo(compute_dtype).tiny)
        )
    output_target_linear = torch.arange(output_size, dtype=torch.long, device=device)
    output_target_pixel = output_target_linear % pixels_per_image
    output_target_u = output_target_pixel % width
    output_target_v = torch.div(output_target_pixel, width, rounding_mode="floor")
    output_disparity_hr_px = torch.where(
        output_valid_flat,
        winning_disparity_target_hr_px,
        torch.zeros_like(winning_disparity_target_hr_px),
    )
    output_depth_m = torch.where(
        output_valid_flat,
        winning_current_depth_m,
        torch.zeros_like(winning_current_depth_m),
    )
    winning_confidence = confidence_flat.reshape(output_size)[output_source_linear]
    output_confidence = torch.where(
        output_valid_flat, winning_confidence, torch.zeros_like(winning_confidence)
    )
    winning_projected_uv = torch.stack(
        (winning_projected_u, winning_projected_v), dim=1
    )
    output_projected_uv = torch.where(
        output_valid_flat.unsqueeze(1),
        winning_projected_uv,
        torch.zeros_like(winning_projected_uv),
    )
    winning_fractional_offset = torch.stack(
        (
            winning_projected_u - output_target_u.to(dtype=compute_dtype),
            winning_projected_v - output_target_v.to(dtype=compute_dtype),
        ),
        dim=1,
    )
    output_fractional_offset = torch.where(
        output_valid_flat.unsqueeze(1),
        winning_fractional_offset,
        torch.zeros_like(winning_fractional_offset),
    )
    winning_source_uv = torch.stack(
        (
            (source_pixel % width).to(dtype=compute_dtype),
            torch.div(source_pixel, width, rounding_mode="floor").to(
                dtype=compute_dtype
            ),
        ),
        dim=1,
    )
    output_source_uv = torch.where(
        output_valid_flat.unsqueeze(1),
        winning_source_uv,
        torch.zeros_like(winning_source_uv),
    )

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
