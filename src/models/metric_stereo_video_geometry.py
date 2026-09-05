"""Causal metric stereo-video geometry model with injectable backbones.

This module owns the trainable fusion, metric gauge, recurrent temporal
memory, and prediction heads.  It deliberately does not claim that the local
VGGT-Omega adapter exposes differentiable intermediate features: callers must
inject explicit :class:`VGGTCausalGeometryFeatures` produced by an upstream
wrapper.  The same contract applies to the stereo backbone.

Disparity inputs and outputs use full-resolution pixel units even when stored
on a lower-resolution grid.  Metric inverse depth is the single predicted
physical quantity; depth and left disparity are analytical views of it:

``depth = 1 / inverse_depth`` and
``disparity = fx_left * ||t_right_from_left|| * inverse_depth``.

Causality is structural.  :meth:`forward_step` consumes only one current
stereo pair plus a state from the immediately preceding step.  :meth:`forward`
is a Python scan over those steps.  VGGT features carry auditable context time
indices, and any index later than the current frame fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from geometry.causal_memory import (
    TemporalGeometryState,
    VisibilityAwareTemporalMemory,
)


@dataclass(frozen=True, slots=True)
class StereoBackboneFeatures:
    """Current-frame differentiable feature map from a stereo backbone."""

    feature_map: Tensor
    time_index: int


@dataclass(frozen=True, slots=True)
class VGGTCausalGeometryFeatures:
    """Current causal-prefix VGGT features with explicit scale provenance.

    ``inverse_depth_relative`` is positive but scale-ambiguous.  It is aligned
    to the calibrated stereo inverse depth inside the model. ``confidence``
    is a probability in ``[0,1]``; the upstream VGGT score ``1 + exp(logit)``
    must first be mapped explicitly, for example with ``(score - 1) / score``.
    The feature map and geometry tensors may use different spatial resolutions.
    ``context_time_indices`` must enumerate only frames at or before
    ``time_index``; a future index is rejected before any tensor is consumed.
    """

    feature_map: Tensor
    inverse_depth_relative: Tensor
    confidence: Tensor
    context_time_indices: tuple[int, ...]
    time_index: int


@dataclass(frozen=True, slots=True)
class MetricStereoFrameInput:
    """Inputs for exactly one causal stereo-video update.

    ``T_right_from_left_m`` and ``T_current_from_previous_m`` are homogeneous
    rigid transforms.  The temporal transform is optional only when no prior
    state is supplied.  ``lowres_disparity_left_px`` is expressed in current
    full-resolution left-image pixels, not low-resolution grid pixels.
    """

    left_rgb: Tensor
    right_rgb: Tensor
    intrinsics_left_3x3: Tensor
    T_right_from_left_m: Tensor
    T_current_from_previous_m: Tensor | None
    lowres_disparity_left_px: Tensor
    stereo_features: StereoBackboneFeatures
    vggt_features: VGGTCausalGeometryFeatures
    time_index: int
    lowres_disparity_valid_mask: Tensor | None = None
    lowres_disparity_confidence: Tensor | None = None


@dataclass(frozen=True, slots=True)
class MetricGaugeAlignment:
    """Scale-only alignment from VGGT-relative to metric inverse depth."""

    inverse_depth_m_inv: Tensor
    scale_m_inv_per_relative_unit: Tensor
    valid_mask: Tensor
    overlap_count: Tensor


@dataclass(frozen=True, slots=True)
class TemporalFusionDiagnostics:
    """Inspectable visibility and gate values for one update."""

    used_history: bool
    valid_mask: Tensor
    zbuffer_visible_mask: Tensor
    depth_consistent_mask: Tensor
    collision_mask: Tensor
    learned_gate: Tensor
    warped_inverse_depth_m_inv: Tensor
    warped_confidence: Tensor
    warped_inverse_depth_pre_consistency_m_inv: Tensor
    warped_confidence_pre_consistency: Tensor


@dataclass(frozen=True, slots=True)
class MetricStereoVideoGeometryOutput:
    """Unified current-frame metric geometry prediction.

    ``log_variance`` is the log variance of the dimensionless log-depth error
    ``log(Z_prediction / Z_target)`` used by the Laplace uncertainty loss.
    Consequently ``uncertainty = exp(log_variance)`` is also dimensionless;
    neither tensor represents variance in metres or disparity pixels.
    """

    inverse_depth_m_inv: Tensor
    depth_m: Tensor
    disparity_left_px: Tensor
    disparity_right_px: Tensor | None
    valid_logits: Tensor
    valid_probability: Tensor
    valid_mask: Tensor
    log_variance: Tensor
    uncertainty: Tensor
    confidence: Tensor
    state: TemporalGeometryState
    gauge: MetricGaugeAlignment
    temporal: TemporalFusionDiagnostics


@dataclass(frozen=True, slots=True)
class CausalMetricStereoClipOutput:
    """Ordered per-frame results and the final recurrent state."""

    frames: tuple[MetricStereoVideoGeometryOutput, ...]
    final_state: TemporalGeometryState


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _group_count(channels: int) -> int:
    return math.gcd(channels, 8)


def _assert_tensor_condition(condition: Tensor, message: str) -> None:
    """Validate one value invariant without synchronizing a valid CUDA path."""

    if condition.dtype != torch.bool or condition.numel() != 1:
        raise TypeError("validation condition must be a one-element bool Tensor")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


class _ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.silu(value + self.body(value))


def _batched_matrix(
    value: Tensor,
    *,
    name: str,
    batch: int,
    size: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or tuple(value.shape[-2:]) != (size, size):
        raise ValueError(f"{name} must have shape [{size},{size}] or [B,{size},{size}]")
    if value.shape[0] == 1 and batch != 1:
        value = value.expand(batch, -1, -1)
    elif value.shape[0] != batch:
        raise ValueError(f"{name} batch does not match RGB batch")
    if value.device != device:
        raise ValueError(f"{name} must share the RGB device")
    _assert_tensor_condition(
        torch.isfinite(value).all(), f"{name} must contain only finite values"
    )
    return value


def _validate_intrinsics(value: Tensor) -> None:
    _assert_tensor_condition(
        ((value[:, 0, 0] > 0) & (value[:, 1, 1] > 0)).all(),
        "intrinsics focal lengths must be positive",
    )
    _assert_tensor_condition(
        torch.isclose(
            value[:, 2, :2], torch.zeros_like(value[:, 2, :2]), atol=1e-6, rtol=0.0
        ).all()
        & torch.isclose(
            value[:, 2, 2], torch.ones_like(value[:, 2, 2]), atol=1e-6, rtol=0.0
        ).all(),
        "intrinsics must end with homogeneous row [0,0,1]",
    )


def _validate_rigid_transform(value: Tensor, name: str) -> None:
    _assert_tensor_condition(
        torch.isclose(
            value[:, 3, :3], torch.zeros_like(value[:, 3, :3]), atol=1e-5, rtol=0.0
        ).all()
        & torch.isclose(
            value[:, 3, 3], torch.ones_like(value[:, 3, 3]), atol=1e-5, rtol=0.0
        ).all(),
        f"{name} has a malformed homogeneous row",
    )
    with torch.autocast(device_type=value.device.type, enabled=False):
        rotation = value[:, :3, :3].float()
        identity = torch.eye(
            3, device=value.device, dtype=torch.float32
        ).expand(value.shape[0], -1, -1)
        rotation_valid = torch.isclose(
            rotation.transpose(1, 2) @ rotation,
            identity,
            atol=2e-3,
            rtol=2e-3,
        ).all()
        determinant = torch.linalg.det(rotation)
        determinant_valid = torch.isclose(
            determinant, torch.ones_like(determinant), atol=2e-3, rtol=2e-3
        )
    _assert_tensor_condition(
        rotation_valid, f"{name} rotation must be orthonormal"
    )
    _assert_tensor_condition(
        determinant_valid.all(), f"{name} rotation determinant must be +1"
    )


def _resize_scalar(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def align_vggt_inverse_depth_to_metric_stereo(
    inverse_depth_relative: Tensor,
    metric_inverse_depth_m_inv: Tensor,
    *,
    relative_confidence: Tensor,
    metric_confidence: Tensor,
    relative_valid_mask: Tensor,
    metric_valid_mask: Tensor,
    minimum_overlap: int = 4,
    epsilon: float = 1e-8,
) -> MetricGaugeAlignment:
    """Differentiably fit one positive scale per sample by weighted LS.

    This is the explicit metric-gauge coupling: VGGT supplies spatial geometry
    but cannot choose its own scale when calibrated stereo support exists.
    A sample with insufficient overlap returns an invalid alignment and its
    aligned geometry is excluded from metric ownership by the caller.
    """

    reference_shape = metric_inverse_depth_m_inv.shape
    if (
        len(reference_shape) != 4
        or reference_shape[1] != 1
        or inverse_depth_relative.shape != reference_shape
    ):
        raise ValueError("relative and metric inverse depth must share [B,1,H,W]")
    for name, value in (
        ("relative_confidence", relative_confidence),
        ("metric_confidence", metric_confidence),
    ):
        if value.shape != reference_shape or not value.is_floating_point():
            raise ValueError(f"{name} must match inverse-depth shape")
    for name, value in (
        ("relative_valid_mask", relative_valid_mask),
        ("metric_valid_mask", metric_valid_mask),
    ):
        if value.shape != reference_shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a matching bool tensor")
    if not isinstance(minimum_overlap, int) or isinstance(minimum_overlap, bool):
        raise TypeError("minimum_overlap must be an integer")
    if minimum_overlap <= 0:
        raise ValueError("minimum_overlap must be positive")

    finite = (
        torch.isfinite(inverse_depth_relative)
        & (inverse_depth_relative > 0)
        & torch.isfinite(metric_inverse_depth_m_inv)
        & (metric_inverse_depth_m_inv > 0)
        & torch.isfinite(relative_confidence)
        & torch.isfinite(metric_confidence)
    )
    overlap = relative_valid_mask & metric_valid_mask & finite
    weight = (
        torch.nan_to_num(relative_confidence, nan=0.0).clamp(0.0, 1.0)
        * torch.nan_to_num(metric_confidence, nan=0.0).clamp(0.0, 1.0)
        * overlap.to(dtype=metric_inverse_depth_m_inv.dtype)
    )
    relative = torch.where(
        overlap, inverse_depth_relative, torch.zeros_like(inverse_depth_relative)
    )
    metric = torch.where(
        overlap,
        metric_inverse_depth_m_inv,
        torch.zeros_like(metric_inverse_depth_m_inv),
    )
    reduce_dims = (1, 2, 3)
    numerator = (weight * relative * metric).sum(dim=reduce_dims, keepdim=True)
    denominator = (weight * relative.square()).sum(
        dim=reduce_dims, keepdim=True
    )
    overlap_count = overlap.sum(dim=reduce_dims)
    valid_sample = (overlap_count >= minimum_overlap) & (
        denominator.flatten(1).squeeze(1) > epsilon
    )
    scale = numerator / denominator.clamp_min(epsilon)
    scale = torch.where(
        valid_sample.reshape(-1, 1, 1, 1),
        scale.clamp(1e-6, 1e6),
        torch.ones_like(scale),
    )
    aligned = torch.where(
        torch.isfinite(inverse_depth_relative) & (inverse_depth_relative > 0),
        inverse_depth_relative * scale,
        torch.zeros_like(inverse_depth_relative),
    )
    return MetricGaugeAlignment(
        inverse_depth_m_inv=aligned,
        scale_m_inv_per_relative_unit=scale,
        valid_mask=valid_sample,
        overlap_count=overlap_count,
    )


class CausalMetricStereoVideoGeometry(nn.Module):
    """Joint stereo/VGGT/temporal fusion with one recurrent causal state."""

    geometry_channels = 9

    def __init__(
        self,
        *,
        stereo_feature_channels: int,
        vggt_feature_channels: int,
        hidden_channels: int = 96,
        residual_blocks: int = 4,
        minimum_gauge_overlap: int = 4,
        inverse_depth_residual_scale: float = 0.25,
        activation_checkpointing: bool = False,
        relative_depth_tolerance: float = 0.05,
        absolute_depth_tolerance_m: float = 0.05,
        enable_vggt_gauge: bool = True,
        enable_temporal_memory: bool = True,
        visibility_aware_gating: bool = True,
    ) -> None:
        super().__init__()
        stereo_feature_channels = _positive_integer(
            stereo_feature_channels, "stereo_feature_channels"
        )
        vggt_feature_channels = _positive_integer(
            vggt_feature_channels, "vggt_feature_channels"
        )
        hidden_channels = _positive_integer(hidden_channels, "hidden_channels")
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be at least 8")
        residual_blocks = _positive_integer(residual_blocks, "residual_blocks")
        minimum_gauge_overlap = _positive_integer(
            minimum_gauge_overlap, "minimum_gauge_overlap"
        )
        if inverse_depth_residual_scale <= 0:
            raise ValueError("inverse_depth_residual_scale must be positive")

        self.stereo_feature_channels = stereo_feature_channels
        self.vggt_feature_channels = vggt_feature_channels
        self.hidden_channels = hidden_channels
        self.minimum_gauge_overlap = minimum_gauge_overlap
        self.inverse_depth_residual_scale = float(inverse_depth_residual_scale)
        self.activation_checkpointing = bool(activation_checkpointing)
        for name, value in (
            ("enable_vggt_gauge", enable_vggt_gauge),
            ("enable_temporal_memory", enable_temporal_memory),
            ("visibility_aware_gating", visibility_aware_gating),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        self.enable_vggt_gauge = enable_vggt_gauge
        self.enable_temporal_memory = enable_temporal_memory
        self.visibility_aware_gating = visibility_aware_gating

        self.appearance_encoder = nn.Sequential(
            _ConvNormAct(6, hidden_channels // 2),
            _ConvNormAct(hidden_channels // 2, hidden_channels),
        )
        self.stereo_projection = nn.Sequential(
            nn.Conv2d(stereo_feature_channels, hidden_channels, 1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.vggt_projection = nn.Sequential(
            nn.Conv2d(vggt_feature_channels, hidden_channels, 1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.geometry_encoder = nn.Sequential(
            _ConvNormAct(self.geometry_channels, hidden_channels // 2),
            _ConvNormAct(hidden_channels // 2, hidden_channels),
        )
        self.current_fusion = _ConvNormAct(4 * hidden_channels, hidden_channels)
        self.temporal_gate = nn.Sequential(
            _ConvNormAct(2 * hidden_channels + 5, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )
        self.recurrent_fusion = _ConvNormAct(2 * hidden_channels, hidden_channels)
        self.residual_body = nn.ModuleList(
            _ResidualBlock(hidden_channels) for _ in range(residual_blocks)
        )
        self.lowres_head = nn.Conv2d(hidden_channels, 3, 3, padding=1)

        hr_channels = max(8, hidden_channels // 4)
        self.hr_rgb_encoder = nn.Sequential(
            _ConvNormAct(6, hr_channels),
            _ConvNormAct(hr_channels, hr_channels),
        )
        self.hr_hidden_projection = nn.Conv2d(hidden_channels, hr_channels, 1)
        self.hr_refinement = nn.Sequential(
            _ConvNormAct(2 * hr_channels, hr_channels),
            nn.Conv2d(hr_channels, 3, 3, padding=1),
        )

        self.memory_warp = VisibilityAwareTemporalMemory(
            relative_depth_tolerance=relative_depth_tolerance,
            absolute_depth_tolerance_m=absolute_depth_tolerance_m,
        )
        nn.init.normal_(self.lowres_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lowres_head.bias)
        final_hr = self.hr_refinement[-1]
        assert isinstance(final_hr, nn.Conv2d)
        nn.init.normal_(final_hr.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_hr.bias)

    def _run_residual_body(self, value: Tensor) -> Tensor:
        for block in self.residual_body:
            if self.activation_checkpointing and self.training and value.requires_grad:
                value = activation_checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return value

    @staticmethod
    def _validate_feature(
        value: Tensor,
        *,
        name: str,
        batch: int,
        channels: int,
        device: torch.device,
    ) -> None:
        if (
            not isinstance(value, Tensor)
            or value.ndim != 4
            or value.shape[0] != batch
            or value.shape[1] != channels
        ):
            raise ValueError(f"{name} must have shape [B,{channels},H,W]")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be real floating point")
        if value.device != device:
            raise ValueError(f"{name} must share the RGB device")
        _assert_tensor_condition(
            torch.isfinite(value).all(), f"{name} must contain only finite values"
        )

    def _validate_frame(
        self,
        frame: MetricStereoFrameInput,
        state: TemporalGeometryState | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(frame.time_index, int) or isinstance(frame.time_index, bool):
            raise TypeError("time_index must be an integer")
        if frame.time_index < 0:
            raise ValueError("time_index must be non-negative")
        left = frame.left_rgb
        right = frame.right_rgb
        if (
            not isinstance(left, Tensor)
            or left.ndim != 4
            or left.shape[1] != 3
            or not left.is_floating_point()
        ):
            raise ValueError("left_rgb must be floating point [B,3,H,W]")
        if right.shape != left.shape or not right.is_floating_point():
            raise ValueError("right_rgb must match left_rgb")
        if right.device != left.device:
            raise ValueError("left_rgb and right_rgb must share a device")
        _assert_tensor_condition(
            torch.isfinite(left).all() & torch.isfinite(right).all(),
            "RGB inputs must contain only finite values",
        )
        batch = left.shape[0]
        if min(left.shape[0], left.shape[-2], left.shape[-1]) <= 0:
            raise ValueError("RGB dimensions must be positive")

        intrinsics = _batched_matrix(
            frame.intrinsics_left_3x3,
            name="intrinsics_left_3x3",
            batch=batch,
            size=3,
            device=left.device,
        )
        _validate_intrinsics(intrinsics)
        stereo_transform = _batched_matrix(
            frame.T_right_from_left_m,
            name="T_right_from_left_m",
            batch=batch,
            size=4,
            device=left.device,
        )
        _validate_rigid_transform(stereo_transform, "T_right_from_left_m")
        with torch.autocast(device_type=stereo_transform.device.type, enabled=False):
            stereo_rotation = stereo_transform[:, :3, :3].float()
            identity_rotation = torch.eye(
                3, device=left.device, dtype=torch.float32
            ).expand_as(stereo_rotation)
            rectified_rotation = torch.isclose(
                stereo_rotation, identity_rotation, atol=2e-4, rtol=0.0
            ).all()
            translation = stereo_transform[:, :3, 3].float()
            horizontal_translation = (
                (translation[:, 0].abs() > 1e-8)
                & torch.isclose(
                    translation[:, 1:],
                    torch.zeros_like(translation[:, 1:]),
                    atol=1e-6,
                    rtol=0.0,
                ).all(dim=1)
            ).all()
        _assert_tensor_condition(
            rectified_rotation & horizontal_translation,
            "metric disparity requires identity stereo rotation and x-only baseline",
        )
        baseline = torch.linalg.vector_norm(stereo_transform[:, :3, 3].float(), dim=1)
        _assert_tensor_condition(
            (baseline > 1e-8).all(),
            "T_right_from_left_m must encode a non-zero baseline",
        )

        disparity = frame.lowres_disparity_left_px
        if (
            not isinstance(disparity, Tensor)
            or disparity.ndim != 4
            or disparity.shape[:2] != (batch, 1)
            or not disparity.is_floating_point()
        ):
            raise ValueError("lowres_disparity_left_px must be [B,1,h,w]")
        if disparity.device != left.device or min(disparity.shape[-2:]) <= 0:
            raise ValueError("lowres disparity must use the RGB device and positive size")

        stereo = frame.stereo_features
        if stereo.time_index != frame.time_index:
            raise ValueError("stereo feature time_index does not match the frame")
        self._validate_feature(
            stereo.feature_map,
            name="stereo feature_map",
            batch=batch,
            channels=self.stereo_feature_channels,
            device=left.device,
        )
        vggt = frame.vggt_features
        if vggt.time_index != frame.time_index:
            raise ValueError("VGGT feature time_index does not match the frame")
        if not isinstance(vggt.context_time_indices, tuple) or not vggt.context_time_indices:
            raise ValueError("VGGT context_time_indices cannot be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in vggt.context_time_indices
        ):
            raise TypeError("VGGT context_time_indices must contain integers")
        if tuple(sorted(set(vggt.context_time_indices))) != vggt.context_time_indices:
            raise ValueError("VGGT context_time_indices must be unique and increasing")
        if vggt.context_time_indices[-1] != frame.time_index:
            raise ValueError("VGGT context must end at the current frame")
        if any(index < 0 or index > frame.time_index for index in vggt.context_time_indices):
            raise ValueError("VGGT context contains a future or negative frame index")
        self._validate_feature(
            vggt.feature_map,
            name="VGGT feature_map",
            batch=batch,
            channels=self.vggt_feature_channels,
            device=left.device,
        )
        relative = vggt.inverse_depth_relative
        if (
            relative.ndim != 4
            or relative.shape[:2] != (batch, 1)
            or not relative.is_floating_point()
            or relative.device != left.device
        ):
            raise ValueError("VGGT inverse_depth_relative must be [B,1,h,w]")
        if vggt.confidence.shape != relative.shape or not vggt.confidence.is_floating_point():
            raise ValueError("VGGT confidence must match inverse_depth_relative")
        if vggt.confidence.device != left.device:
            raise ValueError("VGGT confidence must share the RGB device")
        _assert_tensor_condition(
            torch.isfinite(vggt.confidence).all()
            & ((vggt.confidence >= 0) & (vggt.confidence <= 1)).all(),
            "VGGT confidence must contain probabilities in [0,1]",
        )

        if state is None:
            if frame.T_current_from_previous_m is not None:
                temporal_transform = _batched_matrix(
                    frame.T_current_from_previous_m,
                    name="T_current_from_previous_m",
                    batch=batch,
                    size=4,
                    device=left.device,
                )
                _validate_rigid_transform(temporal_transform, "T_current_from_previous_m")
        else:
            if frame.time_index <= state.time_index:
                raise ValueError("causal state time must be earlier than the current frame")
            if frame.time_index != state.time_index + 1:
                raise ValueError("recurrent state must come from the immediately previous frame")
            if frame.T_current_from_previous_m is None:
                raise ValueError("temporal transform is required when state is supplied")
            temporal_transform = _batched_matrix(
                frame.T_current_from_previous_m,
                name="T_current_from_previous_m",
                batch=batch,
                size=4,
                device=left.device,
            )
            _validate_rigid_transform(temporal_transform, "T_current_from_previous_m")
            if state.feature.shape[0] != batch or state.feature.device != left.device:
                raise ValueError("causal state batch/device does not match the frame")
        return intrinsics, stereo_transform, baseline, disparity

    @staticmethod
    def _calibration_geometry(
        *,
        intrinsics: Tensor,
        baseline_m: Tensor,
        image_size_hw: tuple[int, int],
        feature_size_hw: tuple[int, int],
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = reference.shape[0]
        image_height, image_width = image_size_hw
        height, width = feature_size_hw
        grid_v, grid_u = torch.meshgrid(
            torch.arange(height, device=reference.device, dtype=torch.float32),
            torch.arange(width, device=reference.device, dtype=torch.float32),
            indexing="ij",
        )
        scale_x = image_width / width
        scale_y = image_height / height
        u_hr = (grid_u + 0.5) * scale_x - 0.5
        v_hr = (grid_v + 0.5) * scale_y - 0.5
        cx = intrinsics[:, 0, 2].reshape(batch, 1, 1, 1)
        cy = intrinsics[:, 1, 2].reshape(batch, 1, 1, 1)
        fx = intrinsics[:, 0, 0].reshape(batch, 1, 1, 1)
        fy = intrinsics[:, 1, 1].reshape(batch, 1, 1, 1)
        ray_x = (u_hr.reshape(1, 1, height, width) - cx) / fx
        ray_y = (v_hr.reshape(1, 1, height, width) - cy) / fy
        ray_x = ray_x.expand(batch, -1, -1, -1).to(dtype=reference.dtype)
        ray_y = ray_y.expand(batch, -1, -1, -1).to(dtype=reference.dtype)
        calibration = torch.cat((ray_x, ray_y), dim=1)
        log_baseline = torch.log(baseline_m.clamp_min(1e-8)).reshape(batch, 1, 1, 1)
        log_baseline = log_baseline.expand(-1, -1, height, width).to(reference.dtype)
        return calibration, log_baseline, ray_x

    def forward_step(
        self,
        frame: MetricStereoFrameInput,
        state: TemporalGeometryState | None = None,
    ) -> MetricStereoVideoGeometryOutput:
        """Predict one current frame from current inputs and past state only."""

        intrinsics, _, baseline, lowres_disparity = self._validate_frame(frame, state)
        left = frame.left_rgb
        right = frame.right_rgb
        batch, _, image_height, image_width = left.shape
        feature_size = tuple(lowres_disparity.shape[-2:])
        height, width = feature_size
        scalar_shape = (batch, 1, height, width)

        if frame.lowres_disparity_valid_mask is None:
            stereo_valid = torch.isfinite(lowres_disparity) & (lowres_disparity > 0)
        else:
            stereo_valid = frame.lowres_disparity_valid_mask
            if stereo_valid.shape != scalar_shape or stereo_valid.dtype != torch.bool:
                raise ValueError("lowres_disparity_valid_mask must be matching bool [B,1,h,w]")
            if stereo_valid.device != lowres_disparity.device:
                raise ValueError("lowres disparity validity must share its device")
            stereo_valid = (
                stereo_valid
                & torch.isfinite(lowres_disparity)
                & (lowres_disparity > 0)
            )
        safe_disparity = torch.where(
            stereo_valid, lowres_disparity, torch.zeros_like(lowres_disparity)
        )
        if frame.lowres_disparity_confidence is None:
            stereo_confidence = stereo_valid.to(dtype=lowres_disparity.dtype)
        else:
            stereo_confidence = frame.lowres_disparity_confidence
            if stereo_confidence.shape != scalar_shape or not stereo_confidence.is_floating_point():
                raise ValueError("lowres_disparity_confidence must match disparity")
            if stereo_confidence.device != lowres_disparity.device:
                raise ValueError("lowres disparity confidence must share its device")
            stereo_confidence = torch.nan_to_num(
                stereo_confidence, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(0.0, 1.0)
            stereo_confidence = stereo_confidence * stereo_valid.to(stereo_confidence.dtype)

        metric_numerator = (
            intrinsics[:, 0, 0].float() * baseline
        ).reshape(batch, 1, 1, 1)
        stereo_inverse = (
            safe_disparity.float() / metric_numerator.clamp_min(1e-8)
        ).to(dtype=lowres_disparity.dtype)

        relative_vggt = _resize_scalar(
            frame.vggt_features.inverse_depth_relative, feature_size
        )
        vggt_confidence = _resize_scalar(frame.vggt_features.confidence, feature_size)
        vggt_valid = (
            torch.isfinite(relative_vggt)
            & (relative_vggt > 0)
            & torch.isfinite(vggt_confidence)
            & (vggt_confidence > 0)
        )
        vggt_confidence = torch.nan_to_num(
            vggt_confidence, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if self.enable_vggt_gauge:
            gauge = align_vggt_inverse_depth_to_metric_stereo(
                relative_vggt,
                stereo_inverse,
                relative_confidence=vggt_confidence,
                metric_confidence=stereo_confidence,
                relative_valid_mask=vggt_valid,
                metric_valid_mask=stereo_valid,
                minimum_overlap=self.minimum_gauge_overlap,
            )
        else:
            # Keep the output schema stable while removing VGGT metric
            # ownership from a surgical ablation.
            zero = torch.zeros_like(stereo_inverse)
            gauge = MetricGaugeAlignment(
                inverse_depth_m_inv=zero,
                scale_m_inv_per_relative_unit=torch.zeros(
                    (batch, 1, 1, 1), device=zero.device, dtype=zero.dtype
                ),
                valid_mask=torch.zeros((batch,), device=zero.device, dtype=torch.bool),
                overlap_count=torch.zeros((batch,), device=zero.device, dtype=torch.long),
            )
        gauge_valid = gauge.valid_mask.reshape(batch, 1, 1, 1)
        metric_vggt_valid = vggt_valid & gauge_valid

        stereo_weight = stereo_confidence * stereo_valid.to(stereo_confidence.dtype)
        vggt_weight = vggt_confidence * metric_vggt_valid.to(vggt_confidence.dtype)
        base_weight = stereo_weight + vggt_weight
        base_valid = base_weight > 0
        base_inverse = torch.where(
            base_valid,
            (
                stereo_weight * stereo_inverse
                + vggt_weight * gauge.inverse_depth_m_inv
            )
            / base_weight.clamp_min(1e-8),
            torch.zeros_like(stereo_inverse),
        )
        base_confidence = torch.where(
            base_valid,
            (stereo_weight + vggt_weight).clamp(0.0, 1.0),
            torch.zeros_like(base_weight),
        )

        rgb_lr = F.interpolate(
            torch.cat((left, right), dim=1),
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        appearance_feature = self.appearance_encoder(rgb_lr)
        stereo_feature = self.stereo_projection(
            F.interpolate(
                frame.stereo_features.feature_map,
                size=feature_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        vggt_feature = self.vggt_projection(
            F.interpolate(
                frame.vggt_features.feature_map,
                size=feature_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        rays, log_baseline, _ = self._calibration_geometry(
            intrinsics=intrinsics.float(),
            baseline_m=baseline,
            image_size_hw=(image_height, image_width),
            feature_size_hw=feature_size,
            reference=stereo_inverse,
        )
        geometry_input = torch.cat(
            (
                stereo_inverse,
                stereo_confidence.to(stereo_inverse.dtype),
                stereo_valid.to(stereo_inverse.dtype),
                gauge.inverse_depth_m_inv.to(stereo_inverse.dtype),
                vggt_confidence.to(stereo_inverse.dtype),
                metric_vggt_valid.to(stereo_inverse.dtype),
                rays,
                log_baseline,
            ),
            dim=1,
        )
        if geometry_input.shape[1] != self.geometry_channels:
            raise RuntimeError("internal geometry channel contract changed")
        geometry_feature = self.geometry_encoder(geometry_input)
        current_feature = self.current_fusion(
            torch.cat(
                (appearance_feature, stereo_feature, vggt_feature, geometry_feature),
                dim=1,
            )
        )

        if state is None or not self.enable_temporal_memory:
            warped = None
            temporal_gate = torch.zeros_like(current_feature)
            recurrent_input = torch.cat(
                (current_feature, torch.zeros_like(current_feature)), dim=1
            )
        else:
            assert frame.T_current_from_previous_m is not None
            warped = self.memory_warp(
                state,
                intrinsics_current_hr_3x3=intrinsics.float(),
                baseline_current_m=baseline,
                image_size_current_hw=(image_height, image_width),
                T_current_from_previous_m=frame.T_current_from_previous_m,
                current_inverse_depth_m_inv=base_inverse,
                current_valid_mask=base_valid,
            )
            if self.visibility_aware_gating:
                warped_feature = warped.feature.to(dtype=current_feature.dtype)
                warped_inverse = warped.inverse_depth_m_inv
                warped_confidence = warped.confidence
                warped_valid = warped.valid_mask
            else:
                # The ablation keeps z-buffer winners but bypasses the
                # current-depth consistency and collision suppression that
                # are applied by the normal visibility-aware path.
                warped_feature = warped.feature_pre_consistency.to(
                    dtype=current_feature.dtype
                )
                warped_inverse = warped.inverse_depth_pre_consistency_m_inv
                warped_confidence = warped.confidence_pre_consistency
                warped_valid = warped.zbuffer_visible_mask
            log_depth_ratio = torch.where(
                warped_valid & base_valid,
                torch.log(
                    warped_inverse.to(base_inverse.dtype).clamp_min(1e-8)
                    / base_inverse.clamp_min(1e-8)
                ).clamp(-8.0, 8.0),
                torch.zeros_like(base_inverse),
            )
            temporal_metadata = torch.cat(
                (
                    warped_confidence.to(current_feature.dtype),
                    warped_valid.to(current_feature.dtype),
                    warped.depth_consistent_mask.to(current_feature.dtype),
                    warped.collision_mask.to(current_feature.dtype),
                    log_depth_ratio.to(current_feature.dtype),
                ),
                dim=1,
            )
            temporal_gate = torch.sigmoid(
                self.temporal_gate(
                    torch.cat(
                        (current_feature, warped_feature, temporal_metadata), dim=1
                    )
                )
            )
            if self.visibility_aware_gating:
                temporal_gate = temporal_gate * warped.valid_mask.to(
                    current_feature.dtype
                )
            else:
                # Retain z-buffer winners, but remove depth-consistency and
                # collision-confidence suppression from the learned fusion.
                temporal_gate = temporal_gate * warped.zbuffer_visible_mask.to(
                    current_feature.dtype
                )
            recurrent_input = torch.cat(
                (current_feature, temporal_gate * warped_feature), dim=1
            )
        hidden = self._run_residual_body(self.recurrent_fusion(recurrent_input))

        lowres_prediction = self.lowres_head(hidden)
        with torch.autocast(device_type=lowres_prediction.device.type, enabled=False):
            inverse_log_residual = self.inverse_depth_residual_scale * torch.tanh(
                lowres_prediction[:, 0:1].float()
            )
            completion_inverse = (
                F.softplus(lowres_prediction[:, 0:1].float() - 4.0) + 1e-6
            )
            inverse_depth_lr = torch.where(
                base_valid,
                base_inverse.float().clamp_min(1e-8)
                * torch.exp(inverse_log_residual),
                completion_inverse,
            ).clamp_min(1e-8)
        validity_prior = torch.where(
            base_valid,
            torch.full_like(base_inverse, 2.0),
            torch.full_like(base_inverse, -2.0),
        )
        valid_logits_lr = validity_prior + lowres_prediction[:, 1:2]
        log_variance_prior = -2.0 * torch.log(base_confidence.clamp_min(1e-3))
        log_variance_lr = (
            log_variance_prior + lowres_prediction[:, 2:3]
        ).clamp(-8.0, 8.0)

        hr_size = (image_height, image_width)
        inverse_depth_hr = F.interpolate(
            inverse_depth_lr, size=hr_size, mode="bilinear", align_corners=False
        )
        valid_logits_hr = F.interpolate(
            valid_logits_lr, size=hr_size, mode="bilinear", align_corners=False
        )
        log_variance_hr = F.interpolate(
            log_variance_lr, size=hr_size, mode="bilinear", align_corners=False
        )
        hr_rgb = self.hr_rgb_encoder(torch.cat((left, right), dim=1))
        hr_hidden = self.hr_hidden_projection(
            F.interpolate(hidden, size=hr_size, mode="bilinear", align_corners=False)
        )
        hr_delta = self.hr_refinement(torch.cat((hr_rgb, hr_hidden), dim=1))
        inverse_depth_hr = inverse_depth_hr * torch.exp(
            self.inverse_depth_residual_scale * torch.tanh(hr_delta[:, 0:1])
        )
        valid_logits_hr = valid_logits_hr + hr_delta[:, 1:2]
        log_variance_hr = (log_variance_hr + hr_delta[:, 2:3]).clamp(-8.0, 8.0)

        inverse_depth_hr = inverse_depth_hr.clamp_min(1e-8)
        depth_hr = inverse_depth_hr.reciprocal()
        disparity_left_hr = inverse_depth_hr * metric_numerator.to(
            dtype=inverse_depth_hr.dtype
        )
        valid_probability = torch.sigmoid(valid_logits_hr)
        valid_mask = valid_probability >= 0.5
        uncertainty = torch.exp(log_variance_hr)
        confidence = valid_probability * torch.exp(-0.5 * uncertainty).clamp(0.0, 1.0)

        state_valid = torch.sigmoid(valid_logits_lr) >= 0.5
        state_confidence = (
            torch.sigmoid(valid_logits_lr)
            * torch.exp(-0.5 * torch.exp(log_variance_lr)).clamp(0.0, 1.0)
        )
        next_state = TemporalGeometryState(
            feature=hidden,
            inverse_depth_m_inv=inverse_depth_lr,
            confidence=state_confidence,
            valid_mask=state_valid,
            intrinsics_hr_3x3=intrinsics,
            baseline_m=baseline,
            image_size_hw=(image_height, image_width),
            time_index=frame.time_index,
        )
        if warped is None:
            zero_mask = torch.zeros(scalar_shape, dtype=torch.bool, device=left.device)
            temporal = TemporalFusionDiagnostics(
                used_history=False,
                valid_mask=zero_mask,
                zbuffer_visible_mask=zero_mask,
                depth_consistent_mask=zero_mask,
                collision_mask=zero_mask,
                learned_gate=temporal_gate,
                warped_inverse_depth_m_inv=torch.zeros_like(base_inverse),
                warped_confidence=torch.zeros_like(base_confidence),
                warped_inverse_depth_pre_consistency_m_inv=torch.zeros_like(
                    base_inverse
                ),
                warped_confidence_pre_consistency=torch.zeros_like(base_confidence),
            )
        else:
            temporal = TemporalFusionDiagnostics(
                used_history=True,
                valid_mask=warped.valid_mask,
                zbuffer_visible_mask=warped.zbuffer_visible_mask,
                depth_consistent_mask=warped.depth_consistent_mask,
                collision_mask=warped.collision_mask,
                learned_gate=temporal_gate,
                warped_inverse_depth_m_inv=warped.inverse_depth_m_inv,
                warped_confidence=warped.confidence,
                warped_inverse_depth_pre_consistency_m_inv=(
                    warped.inverse_depth_pre_consistency_m_inv
                ),
                warped_confidence_pre_consistency=(
                    warped.confidence_pre_consistency
                ),
            )
        return MetricStereoVideoGeometryOutput(
            inverse_depth_m_inv=inverse_depth_hr,
            depth_m=depth_hr,
            disparity_left_px=disparity_left_hr,
            disparity_right_px=None,
            valid_logits=valid_logits_hr,
            valid_probability=valid_probability,
            valid_mask=valid_mask,
            log_variance=log_variance_hr,
            uncertainty=uncertainty,
            confidence=confidence,
            state=next_state,
            gauge=gauge,
            temporal=temporal,
        )

    def forward(
        self,
        frames: Sequence[MetricStereoFrameInput],
        initial_state: TemporalGeometryState | None = None,
    ) -> CausalMetricStereoClipOutput:
        """Scan an ordered causal clip without constructing bidirectional context."""

        if not isinstance(frames, Sequence) or len(frames) == 0:
            raise ValueError("frames must be a non-empty ordered sequence")
        state = initial_state
        outputs: list[MetricStereoVideoGeometryOutput] = []
        for frame in frames:
            output = self.forward_step(frame, state)
            outputs.append(output)
            state = output.state
        assert state is not None
        return CausalMetricStereoClipOutput(frames=tuple(outputs), final_state=state)


__all__ = [
    "CausalMetricStereoClipOutput",
    "CausalMetricStereoVideoGeometry",
    "MetricGaugeAlignment",
    "MetricStereoFrameInput",
    "MetricStereoVideoGeometryOutput",
    "StereoBackboneFeatures",
    "TemporalFusionDiagnostics",
    "VGGTCausalGeometryFeatures",
    "align_vggt_inverse_depth_to_metric_stereo",
]
