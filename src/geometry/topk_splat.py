"""Opt-in top-K depth-aware forward splatting for causal history transport.

This module deliberately does not replace :mod:`geometry.zbuffer_reproject`.
The canonical MVP keeps its single z-buffer winner while experiments can use
the v2 interface here. Both implementations use camera-from-world poses and
forward splatting; neither uses ``grid_sample``.

The default projection distributes every continuous point over its four
bilinear target neighbours.  A named ``nearest`` compatibility footprint is
available only for strict regression against the canonical MVP.

The source grid may be HR (disparity and phase are HR-pixel quantities) or LR
(for example, a ConvGRU hidden-state grid).  In the latter case callers pass
the calibrated LR intrinsics while disparity remains in HR-pixel units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from .zbuffer_reproject import (
    WarpResult,
    _batched_homogeneous_extrinsics,
    _batched_intrinsics,
    _batched_positive_scalar,
    _metric_depth_from_explicit_stereo_disparity,
    _image_tensor,
    _validate_rotation,
)


@dataclass(frozen=True, slots=True)
class TopKSplatResult:
    """Top-K source candidates rasterised on the current target grid.

    The candidate axis is ordered by increasing current-camera depth.  Exactly
    tied depths use the flattened source index as a deterministic tie-break.
    Invalid candidate slots are zero-filled (``source_linear_index`` is -1)
    and must be selected with ``valid_mask``.

    Attributes:
        disparity_hr_px: Candidate disparity in HR pixels, ``[B,K,H,W]``.
        depth_m: Candidate current-camera Z in metres, ``[B,K,H,W]``.
        confidence: Propagated source confidence, ``[B,K,H,W]``.
        temporal_age_frames: Propagated non-negative age, ``[B,K,H,W]``.
        valid_mask: Geometrically valid retained candidates, ``[B,K,H,W]``.
        visibility_mask: Traditional z-buffer visibility.  Only valid rank 0
            is true, shape ``[B,K,H,W]``.
        collision_mask: Whether more than one source landed at this target,
            broadcast over retained candidates, ``[B,K,H,W]``.
        source_visibility_mask: Propagated source visibility, ``[B,K,H,W]``.
        source_collision_mask: Propagated source collision flag, ``[B,K,H,W]``.
        footprint_weight: Bilinear footprint coefficient in ``(0,1]`` for
            each retained candidate, ``[B,K,H,W]``.
        projected_uv_grid_px: Continuous coordinate in *target-grid pixels*,
            ``[B,K,2,H,W]`` in ``(u,v)`` order.
        fractional_offset_grid_px: Continuous coordinate minus its selected
            integer target, ``[B,K,2,H,W]``.
        source_uv_grid_px: Integer source-grid coordinate represented in the
            compute dtype, ``[B,K,2,H,W]``.
        source_sequence_index: Source-result index after multi-age merging,
            integer ``[B,K,H,W]``; -1 is invalid.  A direct splat uses zero.
        source_linear_index: Stable flattened pixel index, ``[B,K,H,W]``; -1
            is invalid.  It includes the batch offset.
        candidate_count: Number of valid sources landing at each target before
            top-K truncation, integer ``[B,1,H,W]``.
        z_aware_weights: Explicit normalised fusion weights, ``[B,K,H,W]``.
            They are proportional to bilinear footprint, confidence, source
            visibility,
            ``exp(-(z-z_nearest)/depth_temperature_m)``, age decay, and an
            optional propagated-collision penalty.  An invalid/all-zero set
            returns all-zero weights rather than a uniform fallback.
        aggregate_valid_mask: Weight denominator was finite and positive,
            boolean ``[B,1,H,W]``.
        weighted_disparity_hr_px: Weighted disparity, ``[B,1,H,W]``.
        weighted_depth_m: Weighted depth, ``[B,1,H,W]``.
        weighted_confidence: Weighted source confidence, ``[B,1,H,W]``.
        weighted_fractional_offset_grid_px: Weighted phase in grid pixels,
            ``[B,2,H,W]``.
        weighted_temporal_age_frames: Weighted age, ``[B,1,H,W]``.
        warped_hidden_feature: Optional per-candidate forward-splatted feature,
            ``[B,K,C,H,W]``.  Passing an LR ConvGRU state and LR intrinsics
            therefore warps hidden state to the current LR grid.
        weighted_hidden_feature: Optional weighted feature, ``[B,C,H,W]``.
        front_surface_mask: Optional v3.1 semantic mask ``[B,K,H,W]``.  It is
            true for candidates in the nearest, physically visible depth
            layer.  Legacy/v2 results leave this as ``None``.
        context_only_mask: Optional v3.1 semantic mask ``[B,K,H,W]``.  It is
            true for valid back-layer candidates which may provide context but
            must not be averaged into the metric history proposal.
        depth_layer_index: Optional v3.1 integer depth-layer identifier
            ``[B,K,H,W]``.  The front/same-surface layer is zero and invalid
            slots are -1.
        age2_depth_consistent_available_mask: Optional v3.1 pre-truncation
            audit mask ``[B,1,H,W]``.  It records where an age-2 candidate was
            available in the front depth layer, so age-2 survival is auditable
            rather than inferred from already-truncated candidates.
    """

    disparity_hr_px: Tensor
    depth_m: Tensor
    confidence: Tensor
    temporal_age_frames: Tensor
    valid_mask: Tensor
    visibility_mask: Tensor
    collision_mask: Tensor
    source_visibility_mask: Tensor
    source_collision_mask: Tensor
    footprint_weight: Tensor
    projected_uv_grid_px: Tensor
    fractional_offset_grid_px: Tensor
    source_uv_grid_px: Tensor
    source_sequence_index: Tensor
    source_linear_index: Tensor
    candidate_count: Tensor
    z_aware_weights: Tensor
    aggregate_valid_mask: Tensor
    weighted_disparity_hr_px: Tensor
    weighted_depth_m: Tensor
    weighted_confidence: Tensor
    weighted_fractional_offset_grid_px: Tensor
    weighted_temporal_age_frames: Tensor
    warped_hidden_feature: Tensor | None
    weighted_hidden_feature: Tensor | None
    front_surface_mask: Tensor | None = None
    context_only_mask: Tensor | None = None
    depth_layer_index: Tensor | None = None
    age2_depth_consistent_available_mask: Tensor | None = None

    @property
    def top_k(self) -> int:
        """Number of retained slots per target pixel."""

        return int(self.disparity_hr_px.shape[1])

    def as_single_winner(self) -> WarpResult:
        """Convert a ``K=1`` result to the canonical :class:`WarpResult`.

        This adapter is useful for ``splat_footprint="nearest"`` numerical
        regression tests and lets an
        opt-in caller stage the transport upgrade without changing downstream
        single-winner code.  It rejects ``K != 1`` rather than dropping data.
        """

        if self.top_k != 1:
            raise ValueError(f"as_single_winner requires K=1, got K={self.top_k}")
        return WarpResult(
            disparity_hr_px=self.disparity_hr_px,
            depth_m=self.depth_m,
            confidence=self.confidence,
            valid_mask=self.valid_mask,
            visibility_mask=self.visibility_mask,
            collision_mask=self.collision_mask,
            projected_uv=self.projected_uv_grid_px[:, 0],
            fractional_offset=self.fractional_offset_grid_px[:, 0],
            source_uv=self.source_uv_grid_px[:, 0],
        )


@dataclass(frozen=True, slots=True)
class TopKDiversityDiagnostics:
    """Scalar audit metrics for a v3.1 diverse candidate set.

    All values are detached scalar tensors so this helper can run on-device
    without introducing a synchronisation point. ``unique_age_fraction`` is
    the fraction of valid target pixels retaining more than one temporal age.
    ``age2_survival_rate`` uses the *pre-truncation, front-layer available*
    population as its denominator. Phase variance is circular variance on the
    unit sampling cell (averaged over u/v), and entropy is reported in nats.
    EPE fields are ``None`` unless a teacher/GT disparity is supplied.
    """

    valid_target_count: Tensor
    unique_age_fraction: Tensor
    age2_survival_rate: Tensor
    fractional_phase_variance: Tensor
    topk_weight_entropy: Tensor
    candidate_depth_spread_m: Tensor
    rank0_disparity_epe_hr_px: Tensor | None
    weighted_disparity_epe_hr_px: Tensor | None
    weighted_minus_rank0_epe_hr_px: Tensor | None


TOPK_DIVERSITY_V31_CONTRACT = "age_phase_diverse_front_surface_v3_1"


def _optional_mask(
    value: Tensor | None,
    *,
    name: str,
    reference: Tensor,
    default: bool,
) -> Tensor:
    if value is None:
        return torch.full_like(reference, default, dtype=torch.bool)
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must have shape {tuple(reference.shape)}, got {tuple(value.shape)}"
        )
    if value.device != reference.device:
        raise ValueError(f"{name} must share the image tensor device")
    return value.detach().to(dtype=torch.bool)


def _temporal_age_field(
    value: Tensor | Real,
    *,
    reference: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    batch_size = reference.shape[0]
    if isinstance(value, bool) or not isinstance(value, (Tensor, Real)):
        raise TypeError("temporal_age_frames must be a real scalar or tensor")
    if isinstance(value, Tensor):
        if not value.is_floating_point() and value.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("temporal_age_frames tensor must be real-valued")
        if value.device != reference.device:
            raise ValueError("temporal_age_frames must share the image tensor device")
        detached = value.detach().to(dtype=dtype)
        if detached.ndim == 0:
            result = detached.expand_as(reference)
        elif detached.shape == (batch_size,):
            result = detached.reshape(batch_size, 1, 1, 1).expand_as(reference)
        elif detached.shape == (batch_size, 1, 1, 1):
            result = detached.expand_as(reference)
        elif detached.shape == reference.shape:
            result = detached
        else:
            raise ValueError(
                "temporal_age_frames must be scalar, [B], [B,1,1,1], or "
                f"{tuple(reference.shape)}, got {tuple(detached.shape)}"
            )
    else:
        scalar_age = float(value)
        if not torch.isfinite(torch.tensor(scalar_age)) or scalar_age < 0:
            raise ValueError("scalar temporal_age_frames must be finite and non-negative")
        result = torch.full_like(reference, scalar_age, dtype=dtype)
    return result


def _positive_finite_scalar(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def topk_z_aware_splat(
    previous_disparity_hr_px: Tensor,
    previous_depth_m: Tensor,
    previous_confidence: Tensor,
    intrinsics_grid_3x3: Tensor,
    extrinsics_previous_camera_from_world: Tensor,
    extrinsics_current_camera_from_world: Tensor,
    *,
    intrinsics_current_grid_3x3: Tensor | None = None,
    intrinsics_previous_hr_3x3: Tensor | None = None,
    intrinsics_current_hr_3x3: Tensor | None = None,
    baseline_previous_m: Tensor | None = None,
    baseline_current_m: Tensor | None = None,
    top_k: int = 4,
    temporal_age_frames: Tensor | Real = 1.0,
    previous_hidden_feature: Tensor | None = None,
    source_valid_mask: Tensor | None = None,
    source_visibility_mask: Tensor | None = None,
    source_collision_mask: Tensor | None = None,
    splat_footprint: str = "bilinear",
    minimum_depth_m: float = 1e-6,
    depth_temperature_m: float = 0.25,
    age_temperature_frames: float = 3.0,
    source_collision_penalty: float = 0.5,
) -> TopKSplatResult:
    """Forward-splat disparity and optional hidden features with top-K depth.

    Args:
        previous_disparity_hr_px: Rectified disparity sampled on the source
            grid but expressed in HR pixels, ``[B,1,H,W]``.
        previous_depth_m: Previous-camera Z depth in metres, ``[B,1,H,W]``.
            In dual-calibration mode source ``fx*B/disparity`` owns depth and
            this tensor must agree at every positive-disparity pixel.
        previous_confidence: Source confidence, ``[B,1,H,W]``.  Negative
            confidence remains geometrically inspectable for K=1 compatibility
            but receives zero fusion weight.
        intrinsics_grid_3x3: Calibrated source intrinsics in *grid pixel*
            coordinates, ``[3,3]`` or ``[B,3,3]``. Use source LR intrinsics
            when transporting an LR hidden state.
        extrinsics_previous_camera_from_world: Previous camera-from-world pose.
        extrinsics_current_camera_from_world: Current camera-from-world pose.
        intrinsics_current_grid_3x3: Target intrinsics in target-grid pixels.
            Omit together with all other dual-calibration arguments to retain
            exact same-intrinsics legacy behavior.
        intrinsics_previous_hr_3x3: Source HR intrinsics defining the input
            disparity unit even when geometry is transported on an LR grid.
        intrinsics_current_hr_3x3: Target HR intrinsics defining output
            disparity units.
        baseline_previous_m: Positive source stereo baseline ``[B]``.
        baseline_current_m: Positive target stereo baseline ``[B]``. The four
            dual-calibration arguments are required together. Candidate output
            disparity is recomputed as ``fx_t * B_t / Z_t``.
        top_k: Positive number of candidates retained per target pixel.
        temporal_age_frames: Non-negative scalar, ``[B]``, ``[B,1,1,1]``, or
            per-source ``[B,1,H,W]`` age.
        previous_hidden_feature: Optional source feature ``[B,C,H,W]``.  It is
            gathered by the same forward-splat winners, never ``grid_sample``.
        source_valid_mask: Optional fail-closed source mask ``[B,1,H,W]``.
        source_visibility_mask: Optional propagated visibility mask.
        source_collision_mask: Optional propagated collision mask.
        splat_footprint: ``"bilinear"`` (default) distributes the source over
            four continuous-projection neighbours. ``"nearest"`` is the
            explicit legacy-regression path.
        minimum_depth_m: Strict positive current-camera Z threshold.
        depth_temperature_m: Positive exponential depth-decay temperature.
        age_temperature_frames: Positive exponential age-decay temperature.
        source_collision_penalty: Finite factor in ``[0,1]`` applied to a
            candidate carrying a previous collision flag.

    Returns:
        :class:`TopKSplatResult` on the current grid. Geometry, masks, and
        winner indices are detached. Hidden-feature gather and weighted
        aggregation retain gradients with respect to feature values (never
        with respect to projection/index selection). Non-finite disparity,
        depth, confidence, age, projection, or feature vectors are invalidated
        before target indexing.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if splat_footprint not in {"bilinear", "nearest"}:
        raise ValueError("splat_footprint must be 'bilinear' or 'nearest'")
    minimum_depth = _positive_finite_scalar(minimum_depth_m, "minimum_depth_m")
    depth_temperature = _positive_finite_scalar(
        depth_temperature_m, "depth_temperature_m"
    )
    age_temperature = _positive_finite_scalar(
        age_temperature_frames, "age_temperature_frames"
    )
    if (
        isinstance(source_collision_penalty, bool)
        or not isinstance(source_collision_penalty, Real)
        or not torch.isfinite(torch.tensor(float(source_collision_penalty)))
        or not 0.0 <= float(source_collision_penalty) <= 1.0
    ):
        raise ValueError("source_collision_penalty must be finite and in [0,1]")
    collision_penalty = float(source_collision_penalty)

    disparity_input = _image_tensor(
        previous_disparity_hr_px, "previous_disparity_hr_px"
    )
    depth_input = _image_tensor(previous_depth_m, "previous_depth_m")
    confidence_input = _image_tensor(previous_confidence, "previous_confidence")
    if depth_input.shape != disparity_input.shape:
        raise ValueError("previous_depth_m shape must match previous_disparity_hr_px")
    if confidence_input.shape != disparity_input.shape:
        raise ValueError(
            "previous_confidence shape must match previous_disparity_hr_px"
        )
    if (
        depth_input.device != disparity_input.device
        or confidence_input.device != disparity_input.device
    ):
        raise ValueError("all image tensors must share a device")

    batch_size, _, height, width = disparity_input.shape
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("image tensors must have positive B, H, and W dimensions")
    device = disparity_input.device
    compute_dtype = torch.promote_types(disparity_input.dtype, depth_input.dtype)
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    disparity = disparity_input.to(dtype=compute_dtype)
    depth = depth_input.to(dtype=compute_dtype)
    confidence = confidence_input.to(dtype=compute_dtype)
    age = _temporal_age_field(
        temporal_age_frames, reference=disparity_input, dtype=compute_dtype
    )
    explicit_valid = _optional_mask(
        source_valid_mask,
        name="source_valid_mask",
        reference=disparity_input,
        default=True,
    )
    source_visibility = _optional_mask(
        source_visibility_mask,
        name="source_visibility_mask",
        reference=disparity_input,
        default=True,
    )
    source_collision = _optional_mask(
        source_collision_mask,
        name="source_collision_mask",
        reference=disparity_input,
        default=False,
    )

    hidden_feature: Tensor | None = None
    hidden_channels = 0
    if previous_hidden_feature is not None:
        if not isinstance(previous_hidden_feature, Tensor):
            raise TypeError("previous_hidden_feature must be a torch.Tensor")
        if (
            previous_hidden_feature.ndim != 4
            or previous_hidden_feature.shape[0] != batch_size
            or previous_hidden_feature.shape[-2:] != (height, width)
            or previous_hidden_feature.shape[1] <= 0
        ):
            raise ValueError(
                "previous_hidden_feature must have shape [B,C,H,W] aligned "
                f"with the geometry grid, got {tuple(previous_hidden_feature.shape)}"
            )
        if (
            not previous_hidden_feature.is_floating_point()
            or previous_hidden_feature.is_complex()
        ):
            raise TypeError("previous_hidden_feature must be real floating point")
        if previous_hidden_feature.device != device:
            raise ValueError(
                "previous_hidden_feature must share the image tensor device"
            )
        # Geometry and winner selection are intentionally detached, but a
        # gathered recurrent feature remains differentiable with respect to
        # its values.  This permits truncated-BPTT callers to choose whether
        # to detach the hidden state before entering this transport.
        hidden_feature = previous_hidden_feature
        hidden_channels = int(hidden_feature.shape[1])

    intrinsics_previous_grid = _batched_intrinsics(
        intrinsics_grid_3x3,
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    dual_values = (
        intrinsics_current_grid_3x3,
        intrinsics_previous_hr_3x3,
        intrinsics_current_hr_3x3,
        baseline_previous_m,
        baseline_current_m,
    )
    if any(value is not None for value in dual_values) and any(
        value is None for value in dual_values
    ):
        raise ValueError(
            "dual calibration requires current-grid/source-HR/target-HR "
            "intrinsics and both baselines"
        )
    intrinsics_current_grid = _batched_intrinsics(
        (
            intrinsics_grid_3x3
            if intrinsics_current_grid_3x3 is None
            else intrinsics_current_grid_3x3
        ),
        batch_size=batch_size,
        dtype=compute_dtype,
        device=device,
    )
    target_disparity_numerator_m_px: Tensor | None = None
    if all(value is not None for value in dual_values):
        assert intrinsics_previous_hr_3x3 is not None
        assert intrinsics_current_hr_3x3 is not None
        assert baseline_previous_m is not None and baseline_current_m is not None
        disparity_intrinsics_previous = _batched_intrinsics(
            intrinsics_previous_hr_3x3,
            batch_size=batch_size,
            dtype=compute_dtype,
            device=device,
        )
        disparity_intrinsics_current = _batched_intrinsics(
            intrinsics_current_hr_3x3,
            batch_size=batch_size,
            dtype=compute_dtype,
            device=device,
        )
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
            intrinsics_source_hr_3x3=disparity_intrinsics_previous,
            baseline_source_m=source_baseline,
            disparity_storage_dtype=disparity_input.dtype,
            depth_storage_dtype=depth_input.dtype,
            name="previous_depth_m",
        )
        target_disparity_numerator_m_px = (
            disparity_intrinsics_current[:, 0, 0:1] * target_baseline
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
    pixels_per_image = height * width
    output_size = batch_size * pixels_per_image
    disparity_flat = disparity[:, 0].reshape(batch_size, -1)
    depth_flat = depth[:, 0].reshape(batch_size, -1)
    confidence_flat = confidence[:, 0].reshape(batch_size, -1)
    age_flat = age[:, 0].reshape(batch_size, -1)

    fx_previous = intrinsics_previous_grid[:, 0, 0:1]
    fy_previous = intrinsics_previous_grid[:, 1, 1:2]
    cx_previous = intrinsics_previous_grid[:, 0, 2:3]
    cy_previous = intrinsics_previous_grid[:, 1, 2:3]
    fx_current = intrinsics_current_grid[:, 0, 0:1]
    fy_current = intrinsics_current_grid[:, 1, 1:2]
    cx_current = intrinsics_current_grid[:, 0, 2:3]
    cy_current = intrinsics_current_grid[:, 1, 2:3]
    point_previous = torch.stack(
        (
            (grid_u - cx_previous) * depth_flat / fx_previous,
            (grid_v - cy_previous) * depth_flat / fy_previous,
            depth_flat,
            torch.ones_like(depth_flat),
        ),
        dim=1,
    )
    point_current = transform_current_previous @ point_previous
    x_current = point_current[:, 0]
    y_current = point_current[:, 1]
    z_current = point_current[:, 2]
    projected_u = fx_current * x_current / z_current + cx_current
    projected_v = fy_current * y_current / z_current + cy_current

    feature_finite = torch.ones(
        (batch_size, pixels_per_image), dtype=torch.bool, device=device
    )
    if hidden_feature is not None:
        feature_finite = torch.isfinite(hidden_feature).all(dim=1).reshape(
            batch_size, pixels_per_image
        )
    source_valid = (
        explicit_valid[:, 0].reshape(batch_size, -1)
        & torch.isfinite(disparity_flat)
        & (disparity_flat > 0)
        & torch.isfinite(depth_flat)
        & (depth_flat > 0)
        & torch.isfinite(confidence_flat)
        & torch.isfinite(age_flat)
        & (age_flat >= 0)
        & feature_finite
        & torch.isfinite(projected_u)
        & torch.isfinite(projected_v)
        & torch.isfinite(z_current)
        & (z_current > minimum_depth)
    )
    # Never convert NaN/Inf projected coordinates directly to integer indices.
    safe_projected_u = torch.where(
        torch.isfinite(projected_u), projected_u, torch.zeros_like(projected_u)
    )
    safe_projected_v = torch.where(
        torch.isfinite(projected_v), projected_v, torch.zeros_like(projected_v)
    )
    if splat_footprint == "bilinear":
        target_u_floor = torch.floor(safe_projected_u).to(dtype=torch.long)
        target_v_floor = torch.floor(safe_projected_v).to(dtype=torch.long)
        target_u = torch.stack(
            (
                target_u_floor,
                target_u_floor + 1,
                target_u_floor,
                target_u_floor + 1,
            ),
            dim=-1,
        )
        target_v = torch.stack(
            (
                target_v_floor,
                target_v_floor,
                target_v_floor + 1,
                target_v_floor + 1,
            ),
            dim=-1,
        )
        phase_u = safe_projected_u - target_u_floor.to(dtype=compute_dtype)
        phase_v = safe_projected_v - target_v_floor.to(dtype=compute_dtype)
        footprint = torch.stack(
            (
                (1.0 - phase_u) * (1.0 - phase_v),
                phase_u * (1.0 - phase_v),
                (1.0 - phase_u) * phase_v,
                phase_u * phase_v,
            ),
            dim=-1,
        )
    else:
        target_u = (
            torch.floor(safe_projected_u + 0.5)
            .to(dtype=torch.long)
            .unsqueeze(-1)
        )
        target_v = (
            torch.floor(safe_projected_v + 0.5)
            .to(dtype=torch.long)
            .unsqueeze(-1)
        )
        footprint = torch.ones_like(target_u, dtype=compute_dtype)
    contribution_valid = source_valid.unsqueeze(-1) & (
        (target_u >= 0)
        & (target_u < width)
        & (target_v >= 0)
        & (target_v < height)
        & torch.isfinite(footprint)
        & (footprint > 0)
    )

    source_linear = torch.arange(
        output_size, dtype=torch.long, device=device
    ).reshape(batch_size, pixels_per_image)
    batch_offset = (
        torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1)
        * pixels_per_image
    )
    target_linear = batch_offset.unsqueeze(-1) + target_v * width + target_u
    expanded_source_linear = source_linear.unsqueeze(-1).expand_as(target_linear)
    expanded_current_depth = z_current.unsqueeze(-1).expand_as(footprint)
    valid_source_linear = expanded_source_linear[contribution_valid]
    valid_target_linear = target_linear[contribution_valid]
    valid_current_depth = expanded_current_depth[contribution_valid]

    candidate_count_flat = torch.zeros(
        output_size, dtype=torch.int64, device=device
    )
    if valid_target_linear.numel() > 0:
        candidate_count_flat.scatter_add_(
            0,
            valid_target_linear,
            torch.ones_like(valid_target_linear, dtype=torch.int64),
        )

    # Boolean indexing above preserves flattened source order.  Stable depth
    # sorting therefore also provides the source-index tie-break.  A second
    # stable target sort groups pixels without perturbing their depth ordering.
    retained_source = torch.full(
        (output_size, top_k), -1, dtype=torch.long, device=device
    )
    if valid_target_linear.numel() > 0:
        depth_order = torch.argsort(valid_current_depth, stable=True)
        target_after_depth = valid_target_linear[depth_order]
        source_after_depth = valid_source_linear[depth_order]
        target_order = torch.argsort(target_after_depth, stable=True)
        grouped_target = target_after_depth[target_order]
        grouped_source = source_after_depth[target_order]
        positions = torch.arange(
            grouped_target.numel(), dtype=torch.long, device=device
        )
        new_group = torch.ones_like(grouped_target, dtype=torch.bool)
        new_group[1:] = grouped_target[1:] != grouped_target[:-1]
        group_start = torch.cummax(
            torch.where(new_group, positions, torch.zeros_like(positions)), dim=0
        ).values
        rank_in_target = positions - group_start
        retained = rank_in_target < top_k
        retained_source[
            grouped_target[retained], rank_in_target[retained]
        ] = grouped_source[retained]

    valid_slots_flat = retained_source >= 0
    safe_source = retained_source.clamp_min(0)
    source_batch = torch.div(safe_source, pixels_per_image, rounding_mode="floor")
    source_pixel = safe_source % pixels_per_image

    def gather_scalar(flattened_by_batch: Tensor) -> Tensor:
        gathered = flattened_by_batch[source_batch, source_pixel]
        return torch.where(valid_slots_flat, gathered, torch.zeros_like(gathered))

    gathered_previous_depth = gather_scalar(depth_flat)
    gathered_current_depth = gather_scalar(z_current)
    gathered_previous_disparity = gather_scalar(disparity_flat)
    gathered_disparity = torch.where(
        valid_slots_flat,
        gathered_previous_disparity
        * gathered_previous_depth
        / gathered_current_depth.clamp_min(minimum_depth),
        torch.zeros_like(gathered_previous_disparity),
    )
    if target_disparity_numerator_m_px is not None:
        gathered_disparity = torch.where(
            valid_slots_flat,
            target_disparity_numerator_m_px[source_batch, 0]
            / gathered_current_depth.clamp_min(minimum_depth),
            torch.zeros_like(gathered_disparity),
        )
    gathered_confidence = gather_scalar(confidence_flat)
    gathered_age = gather_scalar(age_flat)
    gathered_projected_u = gather_scalar(projected_u)
    gathered_projected_v = gather_scalar(projected_v)
    gathered_source_visibility = gather_scalar(
        source_visibility[:, 0].reshape(batch_size, -1).to(compute_dtype)
    ).to(dtype=torch.bool)
    gathered_source_collision = gather_scalar(
        source_collision[:, 0].reshape(batch_size, -1).to(compute_dtype)
    ).to(dtype=torch.bool)

    target_pixel = torch.arange(
        output_size, dtype=torch.long, device=device
    ).unsqueeze(1).expand(-1, top_k)
    target_pixel_in_image = target_pixel % pixels_per_image
    target_u_for_slot = target_pixel_in_image % width
    target_v_for_slot = torch.div(
        target_pixel_in_image, width, rounding_mode="floor"
    )
    fractional_u = torch.where(
        valid_slots_flat,
        gathered_projected_u - target_u_for_slot.to(compute_dtype),
        torch.zeros_like(gathered_projected_u),
    )
    fractional_v = torch.where(
        valid_slots_flat,
        gathered_projected_v - target_v_for_slot.to(compute_dtype),
        torch.zeros_like(gathered_projected_v),
    )
    source_u = torch.where(
        valid_slots_flat,
        (source_pixel % width).to(compute_dtype),
        torch.zeros_like(gathered_projected_u),
    )
    source_v = torch.where(
        valid_slots_flat,
        torch.div(source_pixel, width, rounding_mode="floor").to(compute_dtype),
        torch.zeros_like(gathered_projected_v),
    )
    if splat_footprint == "bilinear":
        gathered_footprint = torch.where(
            valid_slots_flat,
            (1.0 - fractional_u.abs()).clamp_min(0.0)
            * (1.0 - fractional_v.abs()).clamp_min(0.0),
            torch.zeros_like(fractional_u),
        )
    else:
        gathered_footprint = valid_slots_flat.to(dtype=compute_dtype)

    target_collision_flat = (
        candidate_count_flat.unsqueeze(1).expand(-1, top_k) > 1
    ) & valid_slots_flat
    rank = torch.arange(top_k, device=device).reshape(1, top_k)
    visibility_flat = valid_slots_flat & (rank == 0)

    nearest_depth = gathered_current_depth[:, :1]
    depth_delta = torch.where(
        valid_slots_flat,
        (gathered_current_depth - nearest_depth).clamp_min(0.0),
        torch.zeros_like(gathered_current_depth),
    )
    depth_factor = torch.exp(-depth_delta / depth_temperature)
    age_factor = torch.exp(-gathered_age / age_temperature)
    confidence_factor = gathered_confidence.clamp(0.0, 1.0)
    propagated_collision_factor = torch.where(
        gathered_source_collision,
        torch.full_like(gathered_confidence, collision_penalty),
        torch.ones_like(gathered_confidence),
    )
    unnormalised_weight = torch.where(
        valid_slots_flat & gathered_source_visibility,
        confidence_factor
        * gathered_footprint
        * depth_factor
        * age_factor
        * propagated_collision_factor,
        torch.zeros_like(gathered_confidence),
    )
    unnormalised_weight = torch.nan_to_num(
        unnormalised_weight, nan=0.0, posinf=0.0, neginf=0.0
    )
    weight_denominator = unnormalised_weight.sum(dim=1, keepdim=True)
    aggregate_valid_flat = torch.isfinite(weight_denominator) & (
        weight_denominator > 0
    )
    z_aware_weight = torch.where(
        aggregate_valid_flat,
        unnormalised_weight / weight_denominator.clamp_min(
            torch.finfo(compute_dtype).tiny
        ),
        torch.zeros_like(unnormalised_weight),
    )
    weighted_disparity = (z_aware_weight * gathered_disparity).sum(
        dim=1, keepdim=True
    )
    weighted_depth = (z_aware_weight * gathered_current_depth).sum(
        dim=1, keepdim=True
    )
    weighted_confidence = (z_aware_weight * gathered_confidence).sum(
        dim=1, keepdim=True
    )
    weighted_fractional = torch.stack(
        (
            (z_aware_weight * fractional_u).sum(dim=1),
            (z_aware_weight * fractional_v).sum(dim=1),
        ),
        dim=1,
    )
    weighted_age = (z_aware_weight * gathered_age).sum(dim=1, keepdim=True)

    warped_hidden: Tensor | None = None
    weighted_hidden: Tensor | None = None
    if hidden_feature is not None:
        feature_by_source = hidden_feature.permute(0, 2, 3, 1).reshape(
            output_size, hidden_channels
        )
        warped_hidden_flat = feature_by_source[safe_source]
        warped_hidden_flat = torch.where(
            valid_slots_flat.unsqueeze(-1),
            warped_hidden_flat,
            torch.zeros_like(warped_hidden_flat),
        )
        weighted_hidden_flat = (
            z_aware_weight.unsqueeze(-1)
            * warped_hidden_flat.to(dtype=compute_dtype)
        ).sum(dim=1)
        warped_hidden = warped_hidden_flat.reshape(
            batch_size, height, width, top_k, hidden_channels
        ).permute(0, 3, 4, 1, 2).to(dtype=hidden_feature.dtype)
        weighted_hidden = weighted_hidden_flat.reshape(
            batch_size, height, width, hidden_channels
        ).permute(0, 3, 1, 2).to(dtype=hidden_feature.dtype)

    vector_shape = (batch_size, height, width, top_k, 2)

    def scalar_image(value: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        result = value.reshape(batch_size, height, width, top_k).permute(
            0, 3, 1, 2
        )
        return result if dtype is None else result.to(dtype=dtype)

    projected_uv = torch.stack(
        (gathered_projected_u, gathered_projected_v), dim=-1
    )
    fractional = torch.stack((fractional_u, fractional_v), dim=-1)
    source_uv = torch.stack((source_u, source_v), dim=-1)

    return TopKSplatResult(
        disparity_hr_px=scalar_image(
            gathered_disparity, dtype=previous_disparity_hr_px.dtype
        ),
        depth_m=scalar_image(gathered_current_depth, dtype=previous_depth_m.dtype),
        confidence=scalar_image(
            gathered_confidence, dtype=previous_confidence.dtype
        ),
        temporal_age_frames=scalar_image(gathered_age),
        valid_mask=scalar_image(valid_slots_flat),
        visibility_mask=scalar_image(visibility_flat),
        collision_mask=scalar_image(target_collision_flat),
        source_visibility_mask=scalar_image(gathered_source_visibility),
        source_collision_mask=scalar_image(gathered_source_collision),
        footprint_weight=scalar_image(gathered_footprint),
        projected_uv_grid_px=projected_uv.reshape(vector_shape).permute(
            0, 3, 4, 1, 2
        ),
        fractional_offset_grid_px=fractional.reshape(vector_shape).permute(
            0, 3, 4, 1, 2
        ),
        source_uv_grid_px=source_uv.reshape(vector_shape).permute(0, 3, 4, 1, 2),
        source_sequence_index=scalar_image(
            torch.where(
                valid_slots_flat,
                torch.zeros_like(retained_source),
                torch.full_like(retained_source, -1),
            )
        ),
        source_linear_index=scalar_image(retained_source),
        candidate_count=candidate_count_flat.reshape(batch_size, 1, height, width),
        z_aware_weights=scalar_image(z_aware_weight),
        aggregate_valid_mask=aggregate_valid_flat.reshape(
            batch_size, 1, height, width
        ),
        weighted_disparity_hr_px=weighted_disparity.reshape(
            batch_size, 1, height, width
        ).to(dtype=previous_disparity_hr_px.dtype),
        weighted_depth_m=weighted_depth.reshape(
            batch_size, 1, height, width
        ).to(dtype=previous_depth_m.dtype),
        weighted_confidence=weighted_confidence.reshape(
            batch_size, 1, height, width
        ).to(dtype=previous_confidence.dtype),
        weighted_fractional_offset_grid_px=weighted_fractional.reshape(
            batch_size, 2, height, width
        ),
        weighted_temporal_age_frames=weighted_age.reshape(
            batch_size, 1, height, width
        ),
        warped_hidden_feature=warped_hidden,
        weighted_hidden_feature=weighted_hidden,
    )


def _check_merge_result_shapes(results: Sequence[TopKSplatResult], top_k: int) -> None:
    if not results:
        raise ValueError("results must contain at least one TopKSplatResult")
    first = results[0]
    if not isinstance(first, TopKSplatResult):
        raise TypeError("results must contain only TopKSplatResult values")
    batch, _, height, width = first.disparity_hr_px.shape
    feature_channels = (
        None
        if first.warped_hidden_feature is None
        else int(first.warped_hidden_feature.shape[2])
    )
    reference_device = first.disparity_hr_px.device
    for result_index, result in enumerate(results):
        if not isinstance(result, TopKSplatResult):
            raise TypeError("results must contain only TopKSplatResult values")
        if result.top_k < top_k:
            raise ValueError(
                f"result {result_index} retains K={result.top_k}, smaller than "
                f"requested merged K={top_k}"
            )
        if (
            result.disparity_hr_px.shape[0] != batch
            or result.disparity_hr_px.shape[-2:] != (height, width)
        ):
            raise ValueError("all results must share B,H,W")
        if result.disparity_hr_px.device != reference_device:
            raise ValueError("all results must share one device")
        result_channels = (
            None
            if result.warped_hidden_feature is None
            else int(result.warped_hidden_feature.shape[2])
        )
        if result_channels != feature_channels:
            raise ValueError(
                "all results must either omit hidden features or share one channel count"
            )


def _union_depth_layers(
    depth_m: Tensor,
    valid_mask: Tensor,
    *,
    absolute_gap_m: float,
    relative_gap: float,
) -> Tensor:
    """Assign deterministic front-to-back layer ids to a candidate union."""

    sortable = torch.where(valid_mask, depth_m, torch.full_like(depth_m, torch.inf))
    order = torch.argsort(sortable, dim=1, stable=True)
    sorted_depth = torch.gather(depth_m, 1, order)
    sorted_valid = torch.gather(valid_mask, 1, order)
    sorted_layer = torch.full_like(sorted_valid, -1, dtype=torch.long)
    first_valid = sorted_valid[:, :1]
    sorted_layer[:, :1] = torch.where(
        first_valid,
        torch.zeros_like(sorted_layer[:, :1]),
        torch.full_like(sorted_layer[:, :1], -1),
    )
    # Compare against the anchor depth of the current layer, not merely the
    # previous candidate.  This prevents single-link chaining (1.00, 1.04,
    # 1.08, ...) from incorrectly turning an arbitrarily thick slab into the
    # metric front surface.
    layer_anchor = sorted_depth[:, :1]
    for rank in range(1, depth_m.shape[1]):
        current = sorted_depth[:, rank : rank + 1]
        current_valid = sorted_valid[:, rank : rank + 1]
        threshold = torch.maximum(
            torch.full_like(layer_anchor, absolute_gap_m),
            layer_anchor.abs() * relative_gap,
        )
        starts_new_layer = (
            current_valid
            & first_valid
            & ((current - layer_anchor) > threshold)
        )
        sorted_layer[:, rank : rank + 1] = torch.where(
            current_valid,
            sorted_layer[:, rank - 1 : rank]
            + starts_new_layer.to(dtype=torch.long),
            torch.full_like(sorted_layer[:, rank : rank + 1], -1),
        )
        layer_anchor = torch.where(starts_new_layer, current, layer_anchor)
    sorted_layer = torch.where(
        sorted_valid, sorted_layer, torch.full_like(sorted_layer, -1)
    )
    layer = torch.full_like(sorted_layer, -1)
    layer.scatter_(1, order, sorted_layer)
    return layer


def _phase_diverse_union_order(
    *,
    union_valid: Tensor,
    union_depth_m: Tensor,
    union_fractional_offset_grid_px: Tensor,
    union_temporal_age_frames: Tensor,
    union_sequence_index: Tensor,
    sequence_count: int,
    depth_layer_index: Tensor,
    top_k: int,
    per_age_quota: int,
    phase_redundancy_sigma_grid_px: float,
    phase_redundancy_penalty: float,
    surface_absolute_gap_m: float,
    surface_relative_gap: float,
) -> tuple[Tensor, Tensor]:
    """Greedily retain depth-safe, age-balanced, phase-diverse candidates.

    The globally nearest candidate is selected first for every pixel.  Later
    slots obey a per-source-age quota.  If an age-2 candidate exists in depth
    layer zero, it is forced into the first available slot.  Within the
    nearest eligible depth layer, lower score favours both small depth delta
    and a phase which is not redundant with already selected samples.
    """

    batch, candidates, height, width = union_valid.shape
    device = union_valid.device
    sortable_depth = torch.where(
        union_valid,
        union_depth_m,
        torch.full_like(union_depth_m, torch.inf),
    )
    global_front_depth = sortable_depth.amin(dim=1, keepdim=True)
    age2_front_available = (
        union_valid
        & (depth_layer_index == 0)
        & torch.isclose(
            union_temporal_age_frames,
            torch.full_like(union_temporal_age_frames, 2.0),
            rtol=0.0,
            atol=1e-4,
        )
    ).any(dim=1, keepdim=True)

    canonical_phase = torch.remainder(union_fractional_offset_grid_px, 1.0)
    chosen = torch.zeros_like(union_valid)
    selected = torch.full(
        (batch, top_k, height, width), -1, dtype=torch.long, device=device
    )
    def quota_eligible() -> Tensor:
        result = torch.zeros_like(union_valid)
        for sequence_index in range(sequence_count):
            sequence_mask = union_sequence_index == sequence_index
            already_selected = (chosen & sequence_mask).sum(dim=1, keepdim=True)
            result |= sequence_mask & (already_selected < per_age_quota)
        return result

    def candidate_score(eligible: Tensor, selected_slots: int) -> Tensor:
        large_layer = torch.full_like(depth_layer_index, candidates + 1)
        eligible_layer = torch.where(eligible, depth_layer_index, large_layer)
        nearest_layer = eligible_layer.amin(dim=1, keepdim=True)
        same_nearest_layer = eligible & (depth_layer_index == nearest_layer)

        # Fractional offsets differing by an integer are the same sampling
        # phase.  Toroidal distance on [0,1)^2 therefore avoids rewarding the
        # two bilinear footprints of one projected source point as diversity.
        minimum_phase_distance_sq = torch.full_like(union_depth_m, 0.5)
        for slot in range(selected_slots):
            slot_index = selected[:, slot : slot + 1]
            slot_valid = slot_index >= 0
            safe_slot = slot_index.clamp_min(0)
            slot_phase = torch.gather(
                canonical_phase,
                1,
                safe_slot.unsqueeze(2).expand(-1, -1, 2, -1, -1),
            )
            delta = (canonical_phase - slot_phase).abs()
            delta = torch.minimum(delta, 1.0 - delta)
            distance_sq = delta.square().sum(dim=2)
            minimum_phase_distance_sq = torch.where(
                slot_valid,
                torch.minimum(minimum_phase_distance_sq, distance_sq),
                minimum_phase_distance_sq,
            )

        front_depth = torch.where(
            torch.isfinite(global_front_depth),
            global_front_depth,
            torch.zeros_like(global_front_depth),
        )
        surface_scale = torch.maximum(
            torch.full_like(front_depth, surface_absolute_gap_m),
            front_depth.abs() * surface_relative_gap,
        ).clamp_min(torch.finfo(union_depth_m.dtype).eps)
        relative_depth_score = (
            (union_depth_m - front_depth).clamp_min(0.0) / surface_scale
        )
        redundancy = torch.exp(
            -minimum_phase_distance_sq
            / (2.0 * phase_redundancy_sigma_grid_px**2)
        )
        score = relative_depth_score + phase_redundancy_penalty * redundancy
        return torch.where(
            same_nearest_layer,
            score,
            torch.full_like(score, torch.inf),
        )

    for slot in range(top_k):
        eligible = union_valid & ~chosen & quota_eligible()
        selected_age2 = (
            chosen
            & torch.isclose(
                union_temporal_age_frames,
                torch.full_like(union_temporal_age_frames, 2.0),
                rtol=0.0,
                atol=1e-4,
            )
        ).any(dim=1, keepdim=True)
        force_age2 = age2_front_available & ~selected_age2
        age2_eligible = eligible & (depth_layer_index == 0) & torch.isclose(
            union_temporal_age_frames,
            torch.full_like(union_temporal_age_frames, 2.0),
            rtol=0.0,
            atol=1e-4,
        )
        force_age2 &= age2_eligible.any(dim=1, keepdim=True)

        if slot == 0:
            # The physical front winner is immutable; diversity starts at the
            # second slot even when the front candidate is not age 2.
            score = sortable_depth
        else:
            normal_score = candidate_score(eligible, slot)
            age2_score = candidate_score(age2_eligible, slot)
            score = torch.where(force_age2, age2_score, normal_score)
        has_choice = torch.isfinite(score).any(dim=1, keepdim=True)
        chosen_index = torch.argmin(score, dim=1, keepdim=True)
        chosen_index = torch.where(
            has_choice, chosen_index, torch.full_like(chosen_index, -1)
        )
        selected[:, slot : slot + 1] = chosen_index
        safe_index = chosen_index.clamp_min(0)
        newly_chosen = torch.zeros_like(chosen)
        newly_chosen.scatter_(
            1,
            safe_index,
            has_choice,
        )
        chosen |= newly_chosen

    # Selection is diversity-aware, but the public candidate order remains
    # strictly front-to-back. Rank 0 is therefore the physical z-buffer winner.
    safe_selected = selected.clamp_min(0)
    selected_depth = torch.gather(union_depth_m, 1, safe_selected)
    selected_depth = torch.where(
        selected >= 0, selected_depth, torch.full_like(selected_depth, torch.inf)
    )
    reorder = torch.argsort(selected_depth, dim=1, stable=True)
    selected = torch.gather(selected, 1, reorder)
    return selected, age2_front_available


def merge_topk_splat_results(
    results: Sequence[TopKSplatResult],
    *,
    top_k: int,
    depth_temperature_m: float = 0.25,
    age_temperature_frames: float = 3.0,
    source_collision_penalty: float = 0.5,
    selection_contract: str = "global_depth_v2",
    per_age_quota: int = 2,
    surface_depth_gap_m: float = 0.05,
    surface_relative_depth_gap: float = 0.05,
    phase_redundancy_sigma_grid_px: float = 0.125,
    phase_redundancy_penalty: float = 0.25,
) -> TopKSplatResult:
    """Merge independently splatted history ages into one target-grid top-K.

    ``results`` is ordered from the preferred source to the less preferred
    source for exact depth ties (normally age 1, then age 2, ...).  Each input
    must have retained at least the requested output ``top_k``: the global top
    K of a union is then guaranteed to be present in the union of each source's
    local top K.  Candidate ordering is stable by
    ``(current depth, result input order, source pixel index)``.

    ``selection_contract=\"global_depth_v2\"`` is the exact historical path.
    The opt-in :data:`TOPK_DIVERSITY_V31_CONTRACT` path first keeps at most
    ``per_age_quota`` candidates from each input age, guarantees one
    front-layer age-2 candidate when available, and uses canonical fractional
    phase as a redundancy penalty within each depth layer.  It never displaces
    the physical global front winner. Back-layer candidates remain available
    as context but receive zero metric-proposal weight.

    Geometry/index tensors stay detached because each input splat detached
    them.  If candidate hidden features require gradients, the gather and
    weighted sum performed here preserve those gradients.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if selection_contract not in {"global_depth_v2", TOPK_DIVERSITY_V31_CONTRACT}:
        raise ValueError(
            "selection_contract must be global_depth_v2 or "
            f"{TOPK_DIVERSITY_V31_CONTRACT}"
        )
    depth_temperature = _positive_finite_scalar(
        depth_temperature_m, "depth_temperature_m"
    )
    age_temperature = _positive_finite_scalar(
        age_temperature_frames, "age_temperature_frames"
    )
    if (
        isinstance(source_collision_penalty, bool)
        or not isinstance(source_collision_penalty, Real)
        or not torch.isfinite(torch.tensor(float(source_collision_penalty)))
        or not 0.0 <= float(source_collision_penalty) <= 1.0
    ):
        raise ValueError("source_collision_penalty must be finite and in [0,1]")
    collision_penalty = float(source_collision_penalty)
    diversity_v31 = selection_contract == TOPK_DIVERSITY_V31_CONTRACT
    if diversity_v31:
        if len(results) > 1 and top_k < 2:
            raise ValueError("v3.1 multi-age merge requires top_k >= 2")
        if (
            isinstance(per_age_quota, bool)
            or not isinstance(per_age_quota, int)
            or per_age_quota <= 0
        ):
            raise ValueError("per_age_quota must be a positive integer")
        if per_age_quota > max(1, top_k // 2):
            raise ValueError("v3.1 per_age_quota must not exceed floor(top_k/2)")
        for name, value in (
            ("surface_depth_gap_m", surface_depth_gap_m),
            ("surface_relative_depth_gap", surface_relative_depth_gap),
            ("phase_redundancy_sigma_grid_px", phase_redundancy_sigma_grid_px),
        ):
            _positive_finite_scalar(value, name)
        if (
            isinstance(phase_redundancy_penalty, bool)
            or not isinstance(phase_redundancy_penalty, Real)
            or not torch.isfinite(torch.tensor(float(phase_redundancy_penalty)))
            or float(phase_redundancy_penalty) < 0
        ):
            raise ValueError("phase_redundancy_penalty must be finite and >= 0")
        _check_merge_result_shapes(results, per_age_quota)
    else:
        _check_merge_result_shapes(results, top_k)

    first = results[0]
    batch, _, height, width = first.disparity_hr_px.shape
    device = first.disparity_hr_px.device
    compute_dtype = first.z_aware_weights.dtype
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32

    # Concatenation order is result order, then already depth/source-stable
    # candidate rank. Stable depth argsort preserves that deterministic tie key.
    def cat(field: str) -> Tensor:
        return torch.cat([getattr(result, field) for result in results], dim=1)

    union_valid = cat("valid_mask")
    union_depth = cat("depth_m").to(dtype=compute_dtype)
    union_sequence_index = torch.cat(
        [
            torch.where(
                result.valid_mask,
                torch.full_like(result.source_linear_index, result_index),
                torch.full_like(result.source_linear_index, -1),
            )
            for result_index, result in enumerate(results)
        ],
        dim=1,
    )
    union_depth_layer: Tensor | None = None
    age2_front_available: Tensor | None = None
    if diversity_v31:
        union_depth_layer = _union_depth_layers(
            union_depth,
            union_valid,
            absolute_gap_m=float(surface_depth_gap_m),
            relative_gap=float(surface_relative_depth_gap),
        )
        union_order, age2_front_available = _phase_diverse_union_order(
            union_valid=union_valid,
            union_depth_m=union_depth,
            union_fractional_offset_grid_px=cat(
                "fractional_offset_grid_px"
            ).to(dtype=compute_dtype),
            union_temporal_age_frames=cat("temporal_age_frames").to(
                dtype=compute_dtype
            ),
            union_sequence_index=union_sequence_index,
            sequence_count=len(results),
            depth_layer_index=union_depth_layer,
            top_k=top_k,
            per_age_quota=per_age_quota,
            phase_redundancy_sigma_grid_px=float(
                phase_redundancy_sigma_grid_px
            ),
            phase_redundancy_penalty=float(phase_redundancy_penalty),
            surface_absolute_gap_m=float(surface_depth_gap_m),
            surface_relative_gap=float(surface_relative_depth_gap),
        )
        selected_slot_valid = union_order >= 0
        safe_union_order = union_order.clamp_min(0)
    else:
        sortable_depth = torch.where(
            union_valid,
            union_depth,
            torch.full_like(union_depth, torch.inf),
        )
        safe_union_order = torch.argsort(sortable_depth, dim=1, stable=True)[
            :, :top_k
        ]
        selected_slot_valid = torch.ones_like(safe_union_order, dtype=torch.bool)

    def gather_scalar(field: str) -> Tensor:
        union = cat(field)
        gathered = torch.gather(union, 1, safe_union_order)
        return torch.where(
            selected_slot_valid,
            gathered,
            torch.zeros_like(gathered),
        )

    def gather_vector(field: str) -> Tensor:
        union = cat(field)
        gathered = torch.gather(
            union,
            1,
            safe_union_order.unsqueeze(2).expand(
                -1, -1, union.shape[2], -1, -1
            ),
        )
        return torch.where(
            selected_slot_valid.unsqueeze(2),
            gathered,
            torch.zeros_like(gathered),
        )

    merged_valid = gather_scalar("valid_mask") & selected_slot_valid
    merged_disparity = gather_scalar("disparity_hr_px")
    merged_depth = gather_scalar("depth_m")
    merged_confidence = gather_scalar("confidence")
    merged_age = gather_scalar("temporal_age_frames")
    merged_source_visibility = gather_scalar("source_visibility_mask")
    merged_source_collision = gather_scalar("source_collision_mask")
    merged_footprint = gather_scalar("footprint_weight")
    merged_projected_uv = gather_vector("projected_uv_grid_px")
    merged_fractional = gather_vector("fractional_offset_grid_px")
    merged_source_uv = gather_vector("source_uv_grid_px")
    merged_source_linear = gather_scalar("source_linear_index")

    merged_sequence_index = torch.gather(
        union_sequence_index, 1, safe_union_order
    )
    merged_sequence_index = torch.where(
        merged_valid,
        merged_sequence_index,
        torch.full_like(merged_sequence_index, -1),
    )
    merged_source_linear = torch.where(
        merged_valid,
        merged_source_linear,
        torch.full_like(merged_source_linear, -1),
    )

    merged_depth_layer: Tensor | None = None
    merged_front_surface: Tensor | None = None
    merged_context_only: Tensor | None = None
    if union_depth_layer is not None:
        merged_depth_layer = torch.gather(
            union_depth_layer, 1, safe_union_order
        )
        merged_depth_layer = torch.where(
            merged_valid,
            merged_depth_layer,
            torch.full_like(merged_depth_layer, -1),
        )
        merged_front_surface = merged_valid & (merged_depth_layer == 0)
        merged_context_only = merged_valid & (merged_depth_layer > 0)

    total_candidate_count = torch.stack(
        [result.candidate_count for result in results], dim=0
    ).sum(dim=0)
    merged_collision = merged_valid & (total_candidate_count > 1)
    rank = torch.arange(top_k, device=device).reshape(1, top_k, 1, 1)
    merged_visibility = merged_valid & (rank == 0)

    nearest_depth = merged_depth[:, :1].to(dtype=compute_dtype)
    depth_delta = torch.where(
        merged_valid,
        (merged_depth.to(dtype=compute_dtype) - nearest_depth).clamp_min(0.0),
        torch.zeros_like(merged_depth, dtype=compute_dtype),
    )
    confidence_factor = merged_confidence.to(dtype=compute_dtype).clamp(0.0, 1.0)
    depth_factor = torch.exp(-depth_delta / depth_temperature)
    age_factor = torch.exp(
        -merged_age.to(dtype=compute_dtype) / age_temperature
    )
    collision_factor = torch.where(
        merged_source_collision,
        torch.full_like(confidence_factor, collision_penalty),
        torch.ones_like(confidence_factor),
    )
    metric_weight_mask = (
        merged_valid
        if merged_front_surface is None
        else merged_front_surface
    )
    unnormalised = torch.where(
        metric_weight_mask & merged_source_visibility,
        confidence_factor
        * merged_footprint.to(dtype=compute_dtype)
        * depth_factor
        * age_factor
        * collision_factor,
        torch.zeros_like(confidence_factor),
    )
    unnormalised = torch.nan_to_num(
        unnormalised, nan=0.0, posinf=0.0, neginf=0.0
    )
    denominator = unnormalised.sum(dim=1, keepdim=True)
    aggregate_valid = torch.isfinite(denominator) & (denominator > 0)
    weights = torch.where(
        aggregate_valid,
        unnormalised / denominator.clamp_min(torch.finfo(compute_dtype).tiny),
        torch.zeros_like(unnormalised),
    )
    weighted_disparity = (
        weights * merged_disparity.to(dtype=compute_dtype)
    ).sum(dim=1, keepdim=True).to(dtype=merged_disparity.dtype)
    weighted_depth = (weights * merged_depth.to(dtype=compute_dtype)).sum(
        dim=1, keepdim=True
    ).to(dtype=merged_depth.dtype)
    weighted_confidence = (
        weights * merged_confidence.to(dtype=compute_dtype)
    ).sum(dim=1, keepdim=True).to(dtype=merged_confidence.dtype)
    weighted_fractional = (
        weights.unsqueeze(2) * merged_fractional.to(dtype=compute_dtype)
    ).sum(dim=1)
    weighted_age = (
        weights * merged_age.to(dtype=compute_dtype)
    ).sum(dim=1, keepdim=True)

    merged_hidden: Tensor | None = None
    weighted_hidden: Tensor | None = None
    if first.warped_hidden_feature is not None:
        union_hidden = torch.cat(
            [
                result.warped_hidden_feature
                for result in results
                if result.warped_hidden_feature is not None
            ],
            dim=1,
        )
        merged_hidden = torch.gather(
            union_hidden,
            1,
            safe_union_order.unsqueeze(2).expand(
                -1, -1, union_hidden.shape[2], -1, -1
            ),
        )
        merged_hidden = torch.where(
            selected_slot_valid.unsqueeze(2),
            merged_hidden,
            torch.zeros_like(merged_hidden),
        )
        weighted_hidden = (
            weights.unsqueeze(2) * merged_hidden.to(dtype=compute_dtype)
        ).sum(dim=1).to(dtype=merged_hidden.dtype)

    return TopKSplatResult(
        disparity_hr_px=merged_disparity,
        depth_m=merged_depth,
        confidence=merged_confidence,
        temporal_age_frames=merged_age,
        valid_mask=merged_valid,
        visibility_mask=merged_visibility,
        collision_mask=merged_collision,
        source_visibility_mask=merged_source_visibility,
        source_collision_mask=merged_source_collision,
        footprint_weight=merged_footprint,
        projected_uv_grid_px=merged_projected_uv,
        fractional_offset_grid_px=merged_fractional,
        source_uv_grid_px=merged_source_uv,
        source_sequence_index=merged_sequence_index,
        source_linear_index=merged_source_linear,
        candidate_count=total_candidate_count,
        z_aware_weights=weights,
        aggregate_valid_mask=aggregate_valid,
        weighted_disparity_hr_px=weighted_disparity,
        weighted_depth_m=weighted_depth,
        weighted_confidence=weighted_confidence,
        weighted_fractional_offset_grid_px=weighted_fractional,
        weighted_temporal_age_frames=weighted_age,
        warped_hidden_feature=merged_hidden,
        weighted_hidden_feature=weighted_hidden,
        front_surface_mask=merged_front_surface,
        context_only_mask=merged_context_only,
        depth_layer_index=merged_depth_layer,
        age2_depth_consistent_available_mask=age2_front_available,
    )


def merge_topk_splat_results_v31(
    results: Sequence[TopKSplatResult],
    *,
    top_k: int = 4,
    per_age_quota: int = 2,
    depth_temperature_m: float = 0.25,
    age_temperature_frames: float = 3.0,
    source_collision_penalty: float = 0.5,
    surface_depth_gap_m: float = 0.05,
    surface_relative_depth_gap: float = 0.05,
    phase_redundancy_sigma_grid_px: float = 0.125,
    phase_redundancy_penalty: float = 0.25,
) -> TopKSplatResult:
    """Explicit v3.1 age/phase-diverse merge without altering v2 defaults."""

    return merge_topk_splat_results(
        results,
        top_k=top_k,
        depth_temperature_m=depth_temperature_m,
        age_temperature_frames=age_temperature_frames,
        source_collision_penalty=source_collision_penalty,
        selection_contract=TOPK_DIVERSITY_V31_CONTRACT,
        per_age_quota=per_age_quota,
        surface_depth_gap_m=surface_depth_gap_m,
        surface_relative_depth_gap=surface_relative_depth_gap,
        phase_redundancy_sigma_grid_px=phase_redundancy_sigma_grid_px,
        phase_redundancy_penalty=phase_redundancy_penalty,
    )


def topk_diversity_diagnostics(
    result: TopKSplatResult,
    *,
    reference_disparity_hr_px: Tensor | None = None,
    reference_valid_mask: Tensor | None = None,
) -> TopKDiversityDiagnostics:
    """Compute fail-closed v3.1 diversity and proposal diagnostics.

    Args:
        result: Output of :func:`merge_topk_splat_results_v31`.
        reference_disparity_hr_px: Optional teacher/GT disparity
            ``[B,1,H,W]`` in HR pixels.
        reference_valid_mask: Optional strict reference mask ``[B,1,H,W]``.

    Empty populations return exact zero rather than NaN.  Back-layer context
    contributes to phase/depth diversity statistics but never to the weighted
    metric proposal or its EPE.
    """

    if not isinstance(result, TopKSplatResult):
        raise TypeError("result must be a TopKSplatResult")
    if (
        result.front_surface_mask is None
        or result.context_only_mask is None
        or result.depth_layer_index is None
        or result.age2_depth_consistent_available_mask is None
    ):
        raise ValueError("diversity diagnostics require a v3.1 merged result")
    valid = result.valid_mask.detach().to(dtype=torch.bool)
    valid_target = valid.any(dim=1, keepdim=True)
    scalar_dtype = torch.float32
    device = valid.device

    def safe_mean(values: Tensor, mask: Tensor) -> Tensor:
        finite_mask = mask & torch.isfinite(values)
        numerator = torch.where(
            finite_mask, values, torch.zeros_like(values)
        ).sum(dtype=scalar_dtype)
        denominator = finite_mask.sum().to(dtype=scalar_dtype)
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            torch.zeros((), dtype=scalar_dtype, device=device),
        ).detach()

    unique_count = torch.zeros_like(valid_target, dtype=torch.long)
    for rank in range(result.top_k):
        current_valid = valid[:, rank : rank + 1]
        current_sequence = result.source_sequence_index[:, rank : rank + 1]
        seen = torch.zeros_like(current_valid)
        for previous_rank in range(rank):
            seen |= (
                valid[:, previous_rank : previous_rank + 1]
                & (result.source_sequence_index[:, previous_rank : previous_rank + 1]
                   == current_sequence)
            )
        unique_count += (current_valid & ~seen).to(dtype=torch.long)
    unique_age_fraction = safe_mean(
        (unique_count > 1).to(dtype=scalar_dtype), valid_target
    )

    age2_retained = (
        result.front_surface_mask
        & torch.isclose(
            result.temporal_age_frames,
            torch.full_like(result.temporal_age_frames, 2.0),
            rtol=0.0,
            atol=1e-4,
        )
    ).any(dim=1, keepdim=True)
    age2_available = result.age2_depth_consistent_available_mask.to(
        dtype=torch.bool
    )
    age2_survival_rate = safe_mean(
        age2_retained.to(dtype=scalar_dtype), age2_available
    )

    # Circular sampling-cell variance treats phases -0.75 and +0.25 as
    # identical, matching bilinear align-corners-false sampling geometry.
    phase = torch.remainder(
        result.fractional_offset_grid_px.detach().to(dtype=scalar_dtype), 1.0
    )
    uniform_weight = valid.to(dtype=scalar_dtype)
    count = uniform_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    angle = phase * (2.0 * torch.pi)
    mean_cos = (
        torch.cos(angle) * uniform_weight.unsqueeze(2)
    ).sum(dim=1) / count
    mean_sin = (
        torch.sin(angle) * uniform_weight.unsqueeze(2)
    ).sum(dim=1) / count
    circular_variance = 1.0 - torch.sqrt(
        (mean_cos.square() + mean_sin.square()).clamp(0.0, 1.0)
    )
    phase_variance = safe_mean(
        circular_variance,
        valid_target.expand(-1, 2, -1, -1),
    )

    weight = torch.where(
        valid,
        torch.nan_to_num(
            result.z_aware_weights.detach().to(dtype=scalar_dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0),
        torch.zeros_like(result.z_aware_weights, dtype=scalar_dtype),
    )
    entropy_map = -torch.where(
        weight > 0,
        weight * torch.log(weight.clamp_min(torch.finfo(scalar_dtype).tiny)),
        torch.zeros_like(weight),
    ).sum(dim=1, keepdim=True)
    weight_entropy = safe_mean(entropy_map, result.aggregate_valid_mask)

    depth = result.depth_m.detach().to(dtype=scalar_dtype)
    minimum_depth = torch.where(
        valid, depth, torch.full_like(depth, torch.inf)
    ).amin(dim=1, keepdim=True)
    maximum_depth = torch.where(
        valid, depth, torch.full_like(depth, -torch.inf)
    ).amax(dim=1, keepdim=True)
    depth_spread = torch.where(
        valid_target,
        (maximum_depth - minimum_depth).clamp_min(0.0),
        torch.zeros_like(minimum_depth),
    )
    mean_depth_spread = safe_mean(depth_spread, valid_target)

    rank0_epe: Tensor | None = None
    weighted_epe: Tensor | None = None
    epe_delta: Tensor | None = None
    if reference_disparity_hr_px is not None:
        expected_shape = (
            result.disparity_hr_px.shape[0],
            1,
            result.disparity_hr_px.shape[2],
            result.disparity_hr_px.shape[3],
        )
        if (
            not isinstance(reference_disparity_hr_px, Tensor)
            or reference_disparity_hr_px.shape != expected_shape
        ):
            raise ValueError(
                "reference_disparity_hr_px must have shape "
                f"{expected_shape}"
            )
        if reference_disparity_hr_px.device != device:
            raise ValueError("reference disparity must share the result device")
        if (
            not reference_disparity_hr_px.is_floating_point()
            or reference_disparity_hr_px.is_complex()
        ):
            raise TypeError("reference disparity must be real floating-point")
        reference = reference_disparity_hr_px.detach().to(dtype=scalar_dtype)
        strict_reference = torch.isfinite(reference) & (reference >= 0)
        if reference_valid_mask is not None:
            if (
                not isinstance(reference_valid_mask, Tensor)
                or reference_valid_mask.shape != expected_shape
                or reference_valid_mask.device != device
                or reference_valid_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "reference_valid_mask must be bool, share device, and have "
                    f"shape {expected_shape}"
                )
            strict_reference &= reference_valid_mask
        rank0_mask = strict_reference & valid[:, :1]
        weighted_mask = strict_reference & result.aggregate_valid_mask
        rank0_epe = safe_mean(
            (result.disparity_hr_px[:, :1].detach().to(dtype=scalar_dtype)
             - reference).abs(),
            rank0_mask,
        )
        weighted_epe = safe_mean(
            (result.weighted_disparity_hr_px.detach().to(dtype=scalar_dtype)
             - reference).abs(),
            weighted_mask,
        )
        epe_delta = (weighted_epe - rank0_epe).detach()
    elif reference_valid_mask is not None:
        raise ValueError("reference_valid_mask requires reference disparity")

    return TopKDiversityDiagnostics(
        valid_target_count=valid_target.sum().detach(),
        unique_age_fraction=unique_age_fraction,
        age2_survival_rate=age2_survival_rate,
        fractional_phase_variance=phase_variance,
        topk_weight_entropy=weight_entropy,
        candidate_depth_spread_m=mean_depth_spread,
        rank0_disparity_epe_hr_px=rank0_epe,
        weighted_disparity_epe_hr_px=weighted_epe,
        weighted_minus_rank0_epe_hr_px=epe_delta,
    )


# Explicit v2 name for config-driven callers.
topk_z_aware_splat_v2 = topk_z_aware_splat


__all__ = [
    "TOPK_DIVERSITY_V31_CONTRACT",
    "TopKDiversityDiagnostics",
    "TopKSplatResult",
    "merge_topk_splat_results",
    "merge_topk_splat_results_v31",
    "topk_diversity_diagnostics",
    "topk_z_aware_splat",
    "topk_z_aware_splat_v2",
]
