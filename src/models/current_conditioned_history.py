"""Current-frame-conditioned selection of top-K temporal observations.

The transport layer deliberately keeps several causal observations per target
pixel.  This module is the first point where those candidates are allowed to
look at the current RGB/FFS state.  It produces two different aggregations:

* a metric history proposal, restricted to the front/same-surface layer; and
* a context feature, allowed to use valid back-layer candidates as context.

Keeping these paths separate prevents an occluded background disparity from
being averaged into a non-physical intermediate surface while still exposing
useful appearance/motion evidence to the ConvGRU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CurrentConditionedHistoryOutput:
    """Selected top-K history on the LR model grid."""

    metric_disparity_hr_px: Tensor
    metric_confidence: Tensor
    metric_valid_mask: Tensor
    context_feature: Tensor
    context_valid_mask: Tensor
    metric_weights: Tensor
    context_weights: Tensor
    candidate_pose_descriptor: Tensor


def _masked_softmax(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Fail-closed softmax over K without an all-invalid uniform fallback."""

    if logits.shape != mask.shape:
        raise ValueError("masked-softmax logits and mask must have one shape")
    if mask.dtype != torch.bool:
        raise TypeError("masked-softmax mask must be bool")
    valid_any = mask.any(dim=1, keepdim=True)
    safe_logits = torch.where(
        mask,
        torch.nan_to_num(logits, nan=-80.0, posinf=80.0, neginf=-80.0),
        torch.full_like(logits, -80.0),
    )
    safe_logits = safe_logits - safe_logits.amax(dim=1, keepdim=True)
    numerator = torch.exp(safe_logits) * mask.to(dtype=logits.dtype)
    denominator = numerator.sum(dim=1, keepdim=True)
    weights = torch.where(
        valid_any,
        numerator / denominator.clamp_min(torch.finfo(logits.dtype).tiny),
        torch.zeros_like(numerator),
    )
    return weights, valid_any


def _rotation_6d(rotation: Tensor) -> Tensor:
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def _factorized_temporal_pose(
    transforms_current_from_history_m: Tensor,
    baseline_m: Tensor,
    temporal_pose_valid: Tensor,
) -> Tensor:
    """Return age-1/age-2 motion descriptors ``[B,2,10]`` in FP32."""

    if transforms_current_from_history_m.dtype != torch.float32:
        raise TypeError("T_current_from_history_m must have dtype torch.float32")
    if baseline_m.dtype != torch.float32:
        raise TypeError("baseline_m must have dtype torch.float32")
    if temporal_pose_valid.dtype != torch.bool:
        raise TypeError("temporal_pose_valid must have bool dtype")
    batch = transforms_current_from_history_m.shape[0]
    if transforms_current_from_history_m.shape != (batch, 2, 4, 4):
        raise ValueError("T_current_from_history_m must have shape [B,2,4,4]")
    if baseline_m.shape != (batch,):
        raise ValueError("baseline_m must have shape [B]")
    if temporal_pose_valid.shape != (batch, 2):
        raise ValueError("temporal_pose_valid must have shape [B,2]")
    if not bool(torch.isfinite(transforms_current_from_history_m).all()):
        raise ValueError("T_current_from_history_m must be finite")
    if not bool(torch.isfinite(baseline_m).all()) or not bool((baseline_m > 0).all()):
        raise ValueError("baseline_m must be finite and positive")

    rotation = transforms_current_from_history_m[..., :3, :3]
    translation = transforms_current_from_history_m[..., :3, 3]
    translation_norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    direction = translation / translation_norm.clamp_min(1e-8)
    direction = torch.where(
        translation_norm > 1e-8, direction, torch.zeros_like(direction)
    )
    magnitude_ratio = torch.log1p(
        translation_norm / baseline_m[:, None, None]
    )
    descriptor = torch.cat(
        (_rotation_6d(rotation), direction, magnitude_ratio), dim=-1
    )
    return torch.where(
        temporal_pose_valid.unsqueeze(-1), descriptor, torch.zeros_like(descriptor)
    )


class CurrentConditionedTopKAttention(nn.Module):
    """Select metric and contextual history using the current frame.

    The learned score is a residual on the explicit z-aware transport prior.
    Its final layer is zero-initialized, so the initial model follows the
    geometry prior within each permitted mask.  Candidate count is a runtime
    dimension and therefore does not alter checkpoint parameter shapes.
    """

    pose_descriptor_channels = 10
    scalar_metadata_channels = 10

    def __init__(
        self,
        *,
        rgb_channels: int = 96,
        geometry_channels: int = 64,
        candidate_channels: int = 32,
        query_channels: int = 32,
        maximum_disparity_hr_px: float = 2048.0,
        maximum_depth_m: float = 1000.0,
        maximum_age_frames: float = 32.0,
        maximum_depth_layers: int = 16,
    ) -> None:
        super().__init__()
        for name, value in (
            ("rgb_channels", rgb_channels),
            ("geometry_channels", geometry_channels),
            ("candidate_channels", candidate_channels),
            ("query_channels", query_channels),
            ("maximum_depth_layers", maximum_depth_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("maximum_disparity_hr_px", maximum_disparity_hr_px),
            ("maximum_depth_m", maximum_depth_m),
            ("maximum_age_frames", maximum_age_frames),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.rgb_channels = rgb_channels
        self.geometry_channels = geometry_channels
        self.candidate_channels = candidate_channels
        self.query_channels = query_channels
        self.maximum_disparity_hr_px = float(maximum_disparity_hr_px)
        self.maximum_depth_m = float(maximum_depth_m)
        self.maximum_age_frames = float(maximum_age_frames)
        self.maximum_depth_layers = int(maximum_depth_layers)

        self.query_encoder = nn.Sequential(
            nn.Conv2d(rgb_channels + geometry_channels, query_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(query_channels, query_channels, 1),
        )
        score_input_channels = (
            candidate_channels
            + query_channels
            + self.scalar_metadata_channels
            + self.pose_descriptor_channels
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(score_input_channels, query_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(query_channels, 2, 1),
        )
        final = self.score_head[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _check_scalar(
        name: str,
        value: Tensor,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
    ) -> None:
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point Tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}")

    def forward(
        self,
        *,
        rgb_feature_lr: Tensor,
        geometry_feature_lr: Tensor,
        candidate_feature: Tensor,
        disparity_hr_px: Tensor,
        depth_m: Tensor,
        confidence: Tensor,
        fractional_phase_grid_px: Tensor,
        temporal_age_frames: Tensor,
        z_aware_prior_weights: Tensor,
        pose_quality: Tensor,
        depth_layer_index: Tensor,
        valid_mask: Tensor,
        front_surface_mask: Tensor,
        context_only_mask: Tensor,
        current_ffs_disparity_hr_px: Tensor,
        current_ffs_confidence: Tensor,
        T_current_from_history_m: Tensor,
        baseline_m: Tensor,
        temporal_pose_valid: Tensor,
    ) -> CurrentConditionedHistoryOutput:
        """Attend candidates on the LR grid.

        Disparity is in target-frame HR pixels, depth is in metres, phase is in
        LR-grid pixels, and age is in frames. ``front_surface_mask`` is the
        only candidate set allowed to own the metric proposal. Valid candidates
        outside it are exposed solely through ``context_feature``.
        """

        if rgb_feature_lr.ndim != 4 or rgb_feature_lr.shape[1] != self.rgb_channels:
            raise ValueError("rgb_feature_lr has the wrong [B,C,H,W] contract")
        if not rgb_feature_lr.is_floating_point() or rgb_feature_lr.is_complex():
            raise TypeError("rgb_feature_lr must be a real floating-point Tensor")
        if not bool(torch.isfinite(rgb_feature_lr).all()):
            raise ValueError("rgb_feature_lr must contain only finite values")
        batch, _, height, width = rgb_feature_lr.shape
        if geometry_feature_lr.shape != (
            batch, self.geometry_channels, height, width
        ):
            raise ValueError("geometry_feature_lr has the wrong [B,C,H,W] contract")
        if (
            not geometry_feature_lr.is_floating_point()
            or geometry_feature_lr.is_complex()
        ):
            raise TypeError("geometry_feature_lr must be a real floating-point Tensor")
        if geometry_feature_lr.device != rgb_feature_lr.device:
            raise ValueError("geometry_feature_lr must share the current feature device")
        if not bool(torch.isfinite(geometry_feature_lr).all()):
            raise ValueError("geometry_feature_lr must contain only finite values")
        if candidate_feature.ndim != 5:
            raise ValueError("candidate_feature must have shape [B,K,C,H,W]")
        candidates = candidate_feature.shape[1]
        if candidate_feature.shape != (
            batch, candidates, self.candidate_channels, height, width
        ):
            raise ValueError("candidate_feature has the wrong [B,K,C,H,W] contract")
        if candidate_feature.device != rgb_feature_lr.device:
            raise ValueError("candidate_feature must share the current feature device")
        if not candidate_feature.is_floating_point() or candidate_feature.is_complex():
            raise TypeError("candidate_feature must be a real floating-point Tensor")
        if not bool(torch.isfinite(candidate_feature).all()):
            raise ValueError("candidate_feature must contain only finite values")
        scalar_shape = (batch, candidates, height, width)
        for name, value in (
            ("disparity_hr_px", disparity_hr_px),
            ("depth_m", depth_m),
            ("confidence", confidence),
            ("temporal_age_frames", temporal_age_frames),
            ("z_aware_prior_weights", z_aware_prior_weights),
            ("pose_quality", pose_quality),
        ):
            self._check_scalar(
                name, value, scalar_shape, device=rgb_feature_lr.device
            )
        if not isinstance(depth_layer_index, Tensor):
            raise TypeError("depth_layer_index must be a Tensor")
        if depth_layer_index.shape != scalar_shape:
            raise ValueError(
                f"depth_layer_index must have shape {scalar_shape}, got "
                f"{tuple(depth_layer_index.shape)}"
            )
        if depth_layer_index.device != rgb_feature_lr.device:
            raise ValueError("depth_layer_index must share the current feature device")
        if depth_layer_index.dtype == torch.bool or depth_layer_index.is_complex():
            raise TypeError("depth_layer_index must have a real numeric dtype")
        if fractional_phase_grid_px.shape != (
            batch, candidates, 2, height, width
        ):
            raise ValueError(
                "fractional_phase_grid_px must have shape [B,K,2,H,W]"
            )
        if not fractional_phase_grid_px.is_floating_point():
            raise TypeError("fractional_phase_grid_px must be floating point")
        if fractional_phase_grid_px.device != rgb_feature_lr.device:
            raise ValueError(
                "fractional_phase_grid_px must share the current feature device"
            )
        for name, mask in (
            ("valid_mask", valid_mask),
            ("front_surface_mask", front_surface_mask),
            ("context_only_mask", context_only_mask),
        ):
            if mask.dtype != torch.bool or mask.shape != scalar_shape:
                raise TypeError(f"{name} must be bool with shape {scalar_shape}")
            if mask.device != rgb_feature_lr.device:
                raise ValueError(f"{name} must share the current feature device")
        if bool((front_surface_mask & context_only_mask).any()):
            raise ValueError("front_surface_mask and context_only_mask must be disjoint")
        current_shape = (batch, 1, height, width)
        self._check_scalar(
            "current_ffs_disparity_hr_px",
            current_ffs_disparity_hr_px,
            current_shape,
            device=rgb_feature_lr.device,
        )
        self._check_scalar(
            "current_ffs_confidence",
            current_ffs_confidence,
            current_shape,
            device=rgb_feature_lr.device,
        )
        if T_current_from_history_m.device != rgb_feature_lr.device:
            raise ValueError("T_current_from_history_m must share the feature device")
        if baseline_m.device != rgb_feature_lr.device:
            raise ValueError("baseline_m must share the feature device")
        if temporal_pose_valid.device != rgb_feature_lr.device:
            raise ValueError("temporal_pose_valid must share the feature device")

        target_dtype = rgb_feature_lr.dtype
        valid = (
            valid_mask
            & torch.isfinite(disparity_hr_px)
            & (disparity_hr_px > 0)
            & torch.isfinite(depth_m)
            & (depth_m > 0)
            & torch.isfinite(confidence)
            & torch.isfinite(fractional_phase_grid_px).all(dim=2)
            & torch.isfinite(temporal_age_frames)
            & torch.isfinite(z_aware_prior_weights)
            & (z_aware_prior_weights > 0)
            & torch.isfinite(pose_quality)
            & torch.isfinite(depth_layer_index)
            & (depth_layer_index >= 0)
        )
        metric_mask = valid & front_surface_mask & ~context_only_mask
        context_mask = valid

        safe_disparity = torch.nan_to_num(
            disparity_hr_px, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, self.maximum_disparity_hr_px)
        safe_depth = torch.nan_to_num(
            depth_m, nan=0.0, posinf=self.maximum_depth_m, neginf=0.0
        ).clamp(0.0, self.maximum_depth_m)
        safe_confidence = torch.nan_to_num(
            confidence, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        safe_phase = torch.nan_to_num(
            fractional_phase_grid_px, nan=0.0, posinf=1.0, neginf=-1.0
        ).clamp(-1.0, 1.0)
        safe_age = torch.nan_to_num(
            temporal_age_frames, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, self.maximum_age_frames)
        safe_prior = torch.nan_to_num(
            z_aware_prior_weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        safe_quality = torch.nan_to_num(
            pose_quality, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        safe_layer = torch.nan_to_num(
            depth_layer_index, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, float(self.maximum_depth_layers))

        pose_by_age = _factorized_temporal_pose(
            T_current_from_history_m,
            baseline_m,
            temporal_pose_valid,
        )
        age_one = torch.isclose(safe_age.float(), torch.ones_like(safe_age.float()))
        age_two = torch.isclose(
            safe_age.float(), torch.full_like(safe_age.float(), 2.0)
        )
        candidate_pose = (
            pose_by_age[:, 0, None, :, None, None]
            * age_one[:, :, None].float()
            + pose_by_age[:, 1, None, :, None, None]
            * age_two[:, :, None].float()
        )
        candidate_pose = candidate_pose.to(dtype=target_dtype)
        candidate_pose = torch.where(
            valid.unsqueeze(2), candidate_pose, torch.zeros_like(candidate_pose)
        )

        query = self.query_encoder(
            torch.cat((rgb_feature_lr, geometry_feature_lr), dim=1)
        )
        query = query[:, None].expand(-1, candidates, -1, -1, -1)
        current_disparity = torch.nan_to_num(
            current_ffs_disparity_hr_px, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        current_confidence = torch.nan_to_num(
            current_ffs_confidence, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        relative_disparity = (
            (safe_disparity - current_disparity[:, 0, None])
            / current_disparity[:, 0, None].clamp_min(1.0)
        ).clamp(-4.0, 4.0)
        front_depth = torch.where(
            metric_mask, safe_depth, torch.full_like(safe_depth, self.maximum_depth_m)
        ).amin(dim=1, keepdim=True)
        relative_depth = (
            (safe_depth - front_depth) / front_depth.clamp_min(1e-3)
        ).clamp(0.0, 4.0)
        metadata = torch.cat(
            (
                (safe_disparity / self.maximum_disparity_hr_px).unsqueeze(2),
                safe_confidence.unsqueeze(2),
                safe_phase,
                (safe_age / self.maximum_age_frames).unsqueeze(2),
                torch.log(safe_prior.clamp_min(1e-8)).clamp(-18.0, 0.0).unsqueeze(2),
                relative_disparity.unsqueeze(2),
                relative_depth.unsqueeze(2),
                safe_quality.unsqueeze(2),
                (safe_layer / float(self.maximum_depth_layers)).unsqueeze(2),
            ),
            dim=2,
        ).to(dtype=target_dtype)
        # metadata contains disparity, confidence, phase2, age, prior,
        # relative disparity/depth, pose quality, and layer index = 10 channels.
        if metadata.shape[2] != 10:
            raise RuntimeError("current-conditioned metadata contract changed")
        score_input = torch.cat(
            (candidate_feature, query, metadata, candidate_pose), dim=2
        ).reshape(batch * candidates, -1, height, width)
        score_residual = self.score_head(score_input).reshape(
            batch, candidates, 2, height, width
        )
        log_prior = torch.log(safe_prior.clamp_min(1e-8)).to(dtype=target_dtype)
        metric_weights, metric_valid = _masked_softmax(
            log_prior + score_residual[:, :, 0], metric_mask
        )
        context_weights, context_valid = _masked_softmax(
            log_prior + score_residual[:, :, 1], context_mask
        )
        metric_disparity = (
            metric_weights * safe_disparity.to(dtype=target_dtype)
        ).sum(dim=1, keepdim=True)
        metric_confidence = (
            metric_weights * safe_confidence.to(dtype=target_dtype)
        ).sum(dim=1, keepdim=True)
        context_feature = (
            context_weights.unsqueeze(2) * candidate_feature
        ).sum(dim=1)
        metric_disparity = torch.where(
            metric_valid, metric_disparity, torch.zeros_like(metric_disparity)
        )
        metric_confidence = torch.where(
            metric_valid, metric_confidence, torch.zeros_like(metric_confidence)
        )
        context_feature = torch.where(
            context_valid, context_feature, torch.zeros_like(context_feature)
        )
        return CurrentConditionedHistoryOutput(
            metric_disparity_hr_px=metric_disparity,
            metric_confidence=metric_confidence,
            metric_valid_mask=metric_valid,
            context_feature=context_feature,
            context_valid_mask=context_valid,
            metric_weights=metric_weights,
            context_weights=context_weights,
            candidate_pose_descriptor=candidate_pose,
        )


__all__ = [
    "CurrentConditionedHistoryOutput",
    "CurrentConditionedTopKAttention",
]
