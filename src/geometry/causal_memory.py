"""Pose-aware, visibility-gated transport for causal video geometry memory.

The memory stored here is expressed in the immediately preceding left-camera
frame.  A caller must provide ``T_current_from_previous_m`` for every update;
the module never receives a future frame or a whole pose trajectory.

Geometry uses the canonical nearest-pixel z-buffer implementation from
``geometry.zbuffer_reproject``.  Winner selection is intentionally discrete,
but the winning feature values are gathered again from the original memory
tensor so gradients can cross frame boundaries during truncated BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .camera import resize_intrinsics_align_corners_false
from .zbuffer_reproject import zbuffer_reproject


@dataclass(frozen=True, slots=True)
class TemporalGeometryState:
    """Recurrent state owned by one causal left-camera time step.

    ``inverse_depth_m_inv`` and ``feature`` live on the same feature grid.
    Intrinsics remain in full-resolution image coordinates so a later frame
    can derive the correct feature-grid calibration without accumulating
    repeated resize error.
    """

    feature: Tensor
    inverse_depth_m_inv: Tensor
    confidence: Tensor
    valid_mask: Tensor
    intrinsics_hr_3x3: Tensor
    baseline_m: Tensor
    image_size_hw: tuple[int, int]
    time_index: int

    def detach(self) -> "TemporalGeometryState":
        """Return an explicitly truncated-BPTT copy of this state."""

        return TemporalGeometryState(
            feature=self.feature.detach(),
            inverse_depth_m_inv=self.inverse_depth_m_inv.detach(),
            confidence=self.confidence.detach(),
            valid_mask=self.valid_mask.detach(),
            intrinsics_hr_3x3=self.intrinsics_hr_3x3.detach(),
            baseline_m=self.baseline_m.detach(),
            image_size_hw=self.image_size_hw,
            time_index=self.time_index,
        )


@dataclass(frozen=True, slots=True)
class VisibilityAwareWarp:
    """Previous state reprojected onto the current feature grid."""

    feature: Tensor
    feature_pre_consistency: Tensor
    inverse_depth_m_inv: Tensor
    confidence: Tensor
    valid_mask: Tensor
    zbuffer_visible_mask: Tensor
    depth_consistent_mask: Tensor
    collision_mask: Tensor
    source_uv_grid_px: Tensor
    inverse_depth_pre_consistency_m_inv: Tensor
    confidence_pre_consistency: Tensor


def _feature_intrinsics(
    intrinsics_hr_3x3: Tensor,
    *,
    image_size_hw: tuple[int, int],
    feature_size_hw: tuple[int, int],
) -> Tensor:
    image_height, image_width = image_size_hw
    feature_height, feature_width = feature_size_hw
    if min(image_height, image_width, feature_height, feature_width) <= 0:
        raise ValueError("image and feature sizes must be positive")
    return resize_intrinsics_align_corners_false(
        intrinsics_hr_3x3,
        feature_width / image_width,
        feature_height / image_height,
    )


def _gather_winning_features(feature: Tensor, source_uv: Tensor) -> Tensor:
    """Gather z-buffer winners while preserving feature-value gradients."""

    batch, channels, height, width = feature.shape
    source_u = source_uv[:, 0].to(dtype=torch.long).clamp(0, width - 1)
    source_v = source_uv[:, 1].to(dtype=torch.long).clamp(0, height - 1)
    source_linear = (source_v * width + source_u).reshape(batch, 1, -1)
    source_linear = source_linear.expand(-1, channels, -1)
    return torch.gather(feature.flatten(2), 2, source_linear).reshape_as(feature)


class VisibilityAwareTemporalMemory(nn.Module):
    """Warp one recurrent state and reject depth-inconsistent history.

    The z-buffer resolves collisions among historical points.  A second test
    compares the winning historical surface to the current metric surface.
    This fail-closed check is what prevents a stale foreground/background
    surface from being treated as valid memory after disocclusion or motion.
    """

    def __init__(
        self,
        *,
        relative_depth_tolerance: float = 0.05,
        absolute_depth_tolerance_m: float = 0.05,
        collision_confidence_scale: float = 0.5,
    ) -> None:
        super().__init__()
        for name, value in (
            ("relative_depth_tolerance", relative_depth_tolerance),
            ("absolute_depth_tolerance_m", absolute_depth_tolerance_m),
            ("collision_confidence_scale", collision_confidence_scale),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real scalar")
        if relative_depth_tolerance < 0 or absolute_depth_tolerance_m < 0:
            raise ValueError("depth tolerances must be non-negative")
        if not 0.0 <= collision_confidence_scale <= 1.0:
            raise ValueError("collision_confidence_scale must be in [0,1]")
        self.relative_depth_tolerance = float(relative_depth_tolerance)
        self.absolute_depth_tolerance_m = float(absolute_depth_tolerance_m)
        self.collision_confidence_scale = float(collision_confidence_scale)

    def forward(
        self,
        state: TemporalGeometryState,
        *,
        intrinsics_current_hr_3x3: Tensor,
        baseline_current_m: Tensor,
        image_size_current_hw: tuple[int, int],
        T_current_from_previous_m: Tensor,
        current_inverse_depth_m_inv: Tensor,
        current_valid_mask: Tensor,
    ) -> VisibilityAwareWarp:
        feature = state.feature
        if feature.ndim != 4 or not feature.is_floating_point():
            raise ValueError("state.feature must be floating point [B,C,H,W]")
        batch, _, height, width = feature.shape
        scalar_shape = (batch, 1, height, width)
        for name, value in (
            ("state.inverse_depth_m_inv", state.inverse_depth_m_inv),
            ("state.confidence", state.confidence),
            ("current_inverse_depth_m_inv", current_inverse_depth_m_inv),
        ):
            if value.shape != scalar_shape or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point {scalar_shape}")
            if value.device != feature.device:
                raise ValueError(f"{name} must share the state feature device")
        if state.valid_mask.shape != scalar_shape or state.valid_mask.dtype != torch.bool:
            raise ValueError(f"state.valid_mask must be bool {scalar_shape}")
        if state.valid_mask.device != feature.device:
            raise ValueError("state.valid_mask must share the state feature device")
        if current_valid_mask.shape != scalar_shape or current_valid_mask.dtype != torch.bool:
            raise ValueError(f"current_valid_mask must be bool {scalar_shape}")
        if current_valid_mask.device != feature.device:
            raise ValueError("current_valid_mask must share the state feature device")
        if state.image_size_hw != image_size_current_hw:
            raise ValueError(
                "causal memory currently requires a fixed crop size across a clip"
            )

        # Projection math stays FP32 under BF16 autocast.  The returned winner
        # indices are then used to gather the original (possibly BF16) feature.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            inverse_previous = state.inverse_depth_m_inv.detach().float()
            valid_previous = (
                state.valid_mask.detach()
                & torch.isfinite(inverse_previous)
                & (inverse_previous > 0)
            )
            canonical_inverse_previous = torch.where(
                valid_previous,
                inverse_previous.clamp_min(1e-8),
                torch.zeros_like(inverse_previous),
            )
            depth_previous = torch.where(
                valid_previous,
                canonical_inverse_previous.clamp_min(1e-8).reciprocal(),
                torch.zeros_like(canonical_inverse_previous),
            )
            intrinsics_previous_feature = _feature_intrinsics(
                state.intrinsics_hr_3x3.detach().float(),
                image_size_hw=state.image_size_hw,
                feature_size_hw=(height, width),
            )
            intrinsics_current_feature = _feature_intrinsics(
                intrinsics_current_hr_3x3.detach().float(),
                image_size_hw=image_size_current_hw,
                feature_size_hw=(height, width),
            )
            baseline_previous = state.baseline_m.detach().float()
            baseline_current = baseline_current_m.detach().float()
            numerator_previous = (
                intrinsics_previous_feature[:, 0, 0]
                * baseline_previous
            ).reshape(batch, 1, 1, 1)
            disparity_previous_feature_px = (
                numerator_previous * canonical_inverse_previous
            )
            confidence_previous = torch.where(
                valid_previous,
                torch.nan_to_num(
                    state.confidence.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0),
                torch.zeros_like(canonical_inverse_previous),
            )
            identity = torch.eye(
                4, device=feature.device, dtype=torch.float32
            ).expand(batch, -1, -1)
            warp = zbuffer_reproject(
                disparity_previous_feature_px,
                depth_previous,
                confidence_previous,
                intrinsics_previous_feature,
                identity,
                T_current_from_previous_m.detach().float(),
                intrinsics_current_hr_3x3=intrinsics_current_feature,
                baseline_previous_m=baseline_previous,
                baseline_current_m=baseline_current,
            )
            warped_inverse = torch.where(
                warp.valid_mask,
                warp.depth_m.clamp_min(1e-8).reciprocal(),
                torch.zeros_like(warp.depth_m),
            )
            current_inverse = current_inverse_depth_m_inv.detach().float()
            finite_current = (
                current_valid_mask.detach()
                & torch.isfinite(current_inverse)
                & (current_inverse > 0)
            )
            current_depth = torch.where(
                finite_current,
                current_inverse.clamp_min(1e-8).reciprocal(),
                torch.zeros_like(current_inverse),
            )
            depth_tolerance = self.absolute_depth_tolerance_m + (
                self.relative_depth_tolerance * current_depth
            )
            depth_consistent = (
                warp.valid_mask
                & finite_current
                & ((warp.depth_m - current_depth).abs() <= depth_tolerance)
            )
            visible = warp.valid_mask & (~finite_current | depth_consistent)
            collision_scale = torch.where(
                warp.collision_mask,
                torch.full_like(warp.confidence, self.collision_confidence_scale),
                torch.ones_like(warp.confidence),
            )
            warped_confidence = torch.where(
                visible,
                warp.confidence.clamp(0.0, 1.0) * collision_scale,
                torch.zeros_like(warp.confidence),
            )

        winning_feature = _gather_winning_features(feature, warp.source_uv)
        pre_consistency_feature = torch.where(
            warp.valid_mask.to(device=feature.device),
            winning_feature,
            torch.zeros_like(winning_feature),
        )
        visible_feature = torch.where(
            visible.to(device=feature.device),
            winning_feature,
            torch.zeros_like(winning_feature),
        )
        return VisibilityAwareWarp(
            feature=visible_feature,
            feature_pre_consistency=pre_consistency_feature,
            inverse_depth_m_inv=torch.where(
                visible, warped_inverse, torch.zeros_like(warped_inverse)
            ).to(dtype=state.inverse_depth_m_inv.dtype),
            confidence=warped_confidence.to(dtype=state.confidence.dtype),
            valid_mask=visible,
            zbuffer_visible_mask=warp.visibility_mask,
            depth_consistent_mask=depth_consistent,
            collision_mask=warp.collision_mask,
            source_uv_grid_px=warp.source_uv,
            inverse_depth_pre_consistency_m_inv=warped_inverse.to(
                dtype=state.inverse_depth_m_inv.dtype
            ),
            confidence_pre_consistency=torch.where(
                warp.valid_mask,
                warp.confidence.clamp(0.0, 1.0),
                torch.zeros_like(warp.confidence),
            ).to(dtype=state.confidence.dtype),
        )


__all__ = [
    "TemporalGeometryState",
    "VisibilityAwareTemporalMemory",
    "VisibilityAwareWarp",
]
