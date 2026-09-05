"""End-to-end causal metric stereo-video geometry composition.

The system joins the trainable stereo and VGGT-Omega wrappers to
``CausalMetricStereoVideoGeometry`` while keeping the fragile contracts in one
place:

* FFS runs on x2-downsampled stereo images, so its disparity values are
  multiplied by two before entering the HR-pixel-unit fusion model.
* left/right consistency creates the stereo validity and confidence supplied
  to the metric gauge;
* VGGT runs exactly once on the complete past-to-current prefix and only its
  endpoint is injected; earlier fusion warm-up steps receive explicit invalid
  zero VGGT geometry and features;
* recurrent fusion is scanned oldest-to-current and returns only the endpoint
  prediction used by the training loss.

The current VGGT wrapper has no padding attention mask.  Consequently this
module rejects left-padded mixed-length batches instead of silently letting
padding contaminate the learned geometry.  A training DataLoader should bucket
clips by length.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from backbones.trainable_stereo import (
    TrainableFastFoundationStereo,
    TrainableStereoOutput,
    half_resolution_stereo_images,
)
from backbones.trainable_vggt_omega import (
    TrainableVGGTOmega,
    TrainableVGGTOmegaOutput,
)
from .metric_stereo_video_geometry import (
    CausalMetricStereoVideoGeometry,
    MetricGaugeAlignment,
    MetricStereoFrameInput,
    MetricStereoVideoGeometryOutput,
    StereoBackboneFeatures,
    VGGTCausalGeometryFeatures,
)


@dataclass(frozen=True, slots=True)
class LeftRightConsistencyResult:
    """Unit-preserving left/right check on one shared disparity grid."""

    sampled_right_px: Tensor
    error_px: Tensor
    valid_left_mask: Tensor
    confidence_left: Tensor


@dataclass(frozen=True, slots=True)
class StereoConsistencyDiagnostics:
    """Online FFS observations with both grid and physical unit provenance."""

    disparity_left_lr_px: Tensor
    disparity_right_lr_px: Tensor | None
    disparity_left_hr_px_lr_grid: Tensor
    disparity_right_hr_px_lr_grid: Tensor | None
    sampled_right_hr_px_lr_grid: Tensor
    left_right_error_lr_px: Tensor
    left_right_error_hr_px_lr_grid: Tensor
    valid_left_mask_lr: Tensor
    confidence_left_lr: Tensor


@dataclass(frozen=True, slots=True)
class MetricStereoVideoSystemOutput:
    """Endpoint prediction plus online-backbone training diagnostics."""

    endpoint: MetricStereoVideoGeometryOutput
    disparity_right_px: Tensor | None
    ffs_iteration_disparities_left_hr_px_lr_grid: tuple[Tensor, ...]
    stereo: StereoConsistencyDiagnostics
    vggt_inverse_depth_relative_endpoint: Tensor
    vggt_confidence_endpoint: Tensor
    gauge: MetricGaugeAlignment

    @property
    def inverse_depth_m_inv(self) -> Tensor:
        return self.endpoint.inverse_depth_m_inv

    @property
    def depth_m(self) -> Tensor:
        return self.endpoint.depth_m

    @property
    def disparity_left_px(self) -> Tensor:
        return self.endpoint.disparity_left_px

    @property
    def valid_logits(self) -> Tensor:
        return self.endpoint.valid_logits

    @property
    def valid_probability(self) -> Tensor:
        return self.endpoint.valid_probability

    @property
    def valid_mask(self) -> Tensor:
        return self.endpoint.valid_mask

    @property
    def log_variance(self) -> Tensor:
        return self.endpoint.log_variance

    @property
    def confidence(self) -> Tensor:
        return self.endpoint.confidence

    def as_loss_predictions(self, *, include_lowres_auxiliary: bool = True) -> dict[str, Any]:
        """Return the stable prediction mapping consumed by the joint loss."""

        predictions: dict[str, Any] = {
            "inverse_depth_m_inv": self.inverse_depth_m_inv,
            "disparity_left_px": self.disparity_left_px,
            "valid_logits": self.valid_logits,
            "log_variance": self.log_variance,
        }
        if include_lowres_auxiliary:
            predictions["inverse_depth_pyramid_m_inv"] = (
                self.endpoint.state.inverse_depth_m_inv,
            )
        if self.disparity_right_px is not None:
            predictions["disparity_right_px"] = self.disparity_right_px
        return predictions


def _debug_phase(phase: str, *, time_index: int | None = None) -> None:
    """Emit rank-local forward progress without any distributed operations."""

    if os.environ.get("METRIC_STEREO_DEBUG_PHASES") != "1":
        return
    fields = [f"rank={os.environ.get('RANK', 'unknown')}", f"phase={phase}"]
    if time_index is not None:
        fields.append(f"time_index={time_index}")
    print(f"[{' '.join(fields)}]", flush=True)


def _assert_tensor_condition(condition: Tensor, message: str) -> None:
    """Check a scalar invariant without synchronizing a valid CUDA batch."""

    if condition.dtype != torch.bool or condition.numel() != 1:
        raise TypeError("validation condition must be a one-element bool Tensor")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


def _require_tensor(
    batch: Mapping[str, Any],
    name: str,
    *,
    ndim: int,
    floating: bool | None = None,
) -> Tensor:
    if name not in batch:
        raise KeyError(f"joint stereo-video batch is missing {name!r}")
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] must be a Tensor")
    if value.ndim != ndim:
        raise ValueError(f"batch[{name!r}] must have rank {ndim}")
    if floating is True and not value.is_floating_point():
        raise TypeError(f"batch[{name!r}] must be floating point")
    if floating is False and value.dtype != torch.bool:
        raise TypeError(f"batch[{name!r}] must be bool")
    return value


def vggt_unbounded_confidence_to_probability(score: Tensor) -> Tensor:
    """Map upstream ``1 + exp(logit)`` scores back to sigmoid probabilities."""

    if not isinstance(score, Tensor) or not score.is_floating_point():
        raise TypeError("VGGT confidence score must be a floating-point Tensor")
    finite_valid = torch.isfinite(score) & (score >= 1.0)
    safe = torch.where(finite_valid, score, torch.ones_like(score))
    probability = (safe - 1.0) / safe.clamp_min(1.0)
    return torch.where(finite_valid, probability, torch.zeros_like(probability)).clamp(
        0.0, 1.0
    )


def left_right_stereo_consistency(
    disparity_left_px: Tensor,
    disparity_right_px: Tensor | None,
    *,
    maximum_error_px: float = 1.0,
    confidence_temperature_px: float = 0.5,
) -> LeftRightConsistencyResult:
    """Build left-view validity/confidence from a rectified stereo pair.

    Inputs may have arbitrary leading dimensions but must end in ``[1,H,W]``.
    Both values use the same pixel unit.  If right disparity is unavailable,
    finite positive left observations remain valid with unit confidence and
    all right/error diagnostic fields are zero.
    """

    for name, value in (
        ("maximum_error_px", maximum_error_px),
        ("confidence_temperature_px", confidence_temperature_px),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if (
        not isinstance(disparity_left_px, Tensor)
        or not disparity_left_px.is_floating_point()
        or disparity_left_px.ndim < 4
        or disparity_left_px.shape[-3] != 1
    ):
        raise ValueError("left disparity must end in [1,H,W] and be floating point")
    left_valid = torch.isfinite(disparity_left_px) & (disparity_left_px > 0)
    if disparity_right_px is None:
        zeros = torch.zeros_like(disparity_left_px)
        return LeftRightConsistencyResult(
            sampled_right_px=zeros,
            error_px=zeros,
            valid_left_mask=left_valid,
            confidence_left=left_valid.to(dtype=disparity_left_px.dtype),
        )
    if disparity_right_px.shape != disparity_left_px.shape:
        raise ValueError("right disparity must match left disparity")
    if not disparity_right_px.is_floating_point():
        raise TypeError("right disparity must be floating point")
    if disparity_right_px.device != disparity_left_px.device:
        raise ValueError("left/right disparity must share one device")

    leading = disparity_left_px.shape[:-3]
    height, width = disparity_left_px.shape[-2:]
    left = disparity_left_px.float().reshape(-1, 1, height, width)
    right = disparity_right_px.float().reshape(-1, 1, height, width)
    count = left.shape[0]
    grid_x = torch.arange(width, dtype=torch.float32, device=left.device).reshape(
        1, 1, width
    )
    grid_y = torch.arange(height, dtype=torch.float32, device=left.device).reshape(
        1, height, 1
    )
    target_x = grid_x - left[:, 0]
    coordinate_valid = (
        torch.isfinite(target_x)
        & torch.isfinite(left[:, 0])
        & (left[:, 0] > 0)
        & (target_x >= 0)
        & (target_x <= width - 1)
    )
    safe_x = torch.where(coordinate_valid, target_x, torch.zeros_like(target_x))
    sample_grid = torch.stack(
        (
            2.0 * (safe_x + 0.5) / width - 1.0,
            2.0
            * (grid_y.expand(count, height, width) + 0.5)
            / height
            - 1.0,
        ),
        dim=-1,
    )
    right_valid = torch.isfinite(right) & (right > 0)
    sampled_right = F.grid_sample(
        torch.where(right_valid, right, torch.zeros_like(right)),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled_support = F.grid_sample(
        right_valid.float(),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    correspondence_valid = (
        coordinate_valid.unsqueeze(1)
        & (sampled_support >= 1.0 - 1e-5)
        & torch.isfinite(sampled_right)
        & (sampled_right > 0)
    )
    error = torch.where(
        correspondence_valid,
        (left - sampled_right).abs(),
        torch.zeros_like(left),
    )
    consistent = correspondence_valid & (error <= float(maximum_error_px))
    confidence = torch.where(
        consistent,
        torch.exp(-error / float(confidence_temperature_px)),
        torch.zeros_like(error),
    )
    output_shape = (*leading, 1, height, width)
    return LeftRightConsistencyResult(
        sampled_right_px=sampled_right.reshape(output_shape).to(
            disparity_left_px.dtype
        ),
        error_px=error.reshape(output_shape).to(disparity_left_px.dtype),
        valid_left_mask=consistent.reshape(output_shape),
        confidence_left=confidence.reshape(output_shape).to(
            disparity_left_px.dtype
        ),
    )


class MetricStereoVideoSystem(nn.Module):
    """Compose online trainable backbones and causal metric fusion."""

    def __init__(
        self,
        stereo_backbone: TrainableFastFoundationStereo,
        vggt_backbone: TrainableVGGTOmega,
        geometry_model: CausalMetricStereoVideoGeometry,
        *,
        stereo_feature_level: int = 0,
        left_right_maximum_error_lr_px: float = 1.0,
        left_right_confidence_temperature_lr_px: float = 0.5,
        require_right_disparity: bool = True,
        enable_vggt_dense_features: bool = True,
        enable_vggt_geometry: bool = True,
        enable_vggt_features: bool | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(stereo_backbone, nn.Module):
            raise TypeError("stereo_backbone must be an nn.Module")
        if not isinstance(vggt_backbone, nn.Module):
            raise TypeError("vggt_backbone must be an nn.Module")
        if not isinstance(geometry_model, CausalMetricStereoVideoGeometry):
            raise TypeError("geometry_model must be CausalMetricStereoVideoGeometry")
        if (
            not isinstance(stereo_feature_level, int)
            or isinstance(stereo_feature_level, bool)
            or stereo_feature_level < 0
        ):
            raise ValueError("stereo_feature_level must be a non-negative integer")
        if not isinstance(require_right_disparity, bool):
            raise TypeError("require_right_disparity must be bool")
        for name, value in (
            ("enable_vggt_dense_features", enable_vggt_dense_features),
            ("enable_vggt_geometry", enable_vggt_geometry),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if enable_vggt_features is not None:
            if not isinstance(enable_vggt_features, bool):
                raise TypeError("enable_vggt_features must be bool or None")
            enable_vggt_dense_features = enable_vggt_features
            enable_vggt_geometry = enable_vggt_features
        self.stereo_backbone = stereo_backbone
        self.vggt_backbone = vggt_backbone
        self.geometry_model = geometry_model
        self.stereo_feature_level = stereo_feature_level
        self.left_right_maximum_error_lr_px = float(
            left_right_maximum_error_lr_px
        )
        self.left_right_confidence_temperature_lr_px = float(
            left_right_confidence_temperature_lr_px
        )
        self.require_right_disparity = bool(require_right_disparity)
        self.enable_vggt_dense_features = enable_vggt_dense_features
        self.enable_vggt_geometry = enable_vggt_geometry
        if self.require_right_disparity and getattr(
            stereo_backbone, "predict_right", True
        ) is False:
            raise ValueError(
                "joint LR consistency requires stereo predict_right=True"
            )
        # Validate scalar policy eagerly through the shared implementation.
        probe = torch.ones(1, 1, 1, 2)
        left_right_stereo_consistency(
            probe,
            probe,
            maximum_error_px=self.left_right_maximum_error_lr_px,
            confidence_temperature_px=self.left_right_confidence_temperature_lr_px,
        )

    @property
    def enable_vggt_features(self) -> bool:
        """Compatibility alias for the old coupled ablation switch."""

        return self.enable_vggt_dense_features and self.enable_vggt_geometry

    @enable_vggt_features.setter
    def enable_vggt_features(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enable_vggt_features must be bool")
        self.enable_vggt_dense_features = enabled
        self.enable_vggt_geometry = enabled

    @staticmethod
    def _validate_batch(batch: Mapping[str, Any]) -> tuple[Tensor, ...]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be the raw stereo-video collate mapping")
        rgb = _require_tensor(batch, "rgb", ndim=6, floating=True)
        if rgb.shape[2:4] != (2, 3):
            raise ValueError("batch['rgb'] must have shape [B,T,2,3,H,W]")
        batch_size, frames = rgb.shape[:2]
        if min(batch_size, frames, rgb.shape[-2], rgb.shape[-1]) <= 0:
            raise ValueError("RGB clip dimensions must be positive")
        if rgb.shape[-2] % 2 or rgb.shape[-1] % 2:
            raise ValueError("x2 stereo observation requires even HR height and width")
        _assert_tensor_condition(
            torch.isfinite(rgb).all(),
            "batch RGB contains NaN or infinity",
        )
        time_valid = _require_tensor(batch, "time_valid_mask", ndim=2, floating=False)
        if time_valid.shape != (batch_size, frames):
            raise ValueError("time_valid_mask must have shape [B,T]")
        if time_valid.device != rgb.device:
            raise ValueError("time_valid_mask must share the RGB device")
        _assert_tensor_condition(
            time_valid.all(),
            "VGGT has no padding mask; use length-bucketed batches without left padding",
        )

        target_time = _require_tensor(
            batch, "target_time_mask", ndim=2, floating=False
        )
        if target_time.shape != (batch_size, frames) or target_time.device != rgb.device:
            raise ValueError("target_time_mask must be bool [B,T] on the RGB device")
        expected_target = torch.zeros_like(target_time)
        expected_target[:, -1] = True
        _assert_tensor_condition(
            (target_time == expected_target).all(),
            "causal endpoint supervision must select only the final frame",
        )

        frame_ids = _require_tensor(batch, "frame_ids", ndim=2)
        timestamps = _require_tensor(batch, "timestamps", ndim=2, floating=True)
        manifest_indices = _require_tensor(batch, "manifest_indices", ndim=2)
        for name, value in (
            ("frame_ids", frame_ids),
            ("timestamps", timestamps),
            ("manifest_indices", manifest_indices),
        ):
            if value.shape != (batch_size, frames) or value.device != rgb.device:
                raise ValueError(f"{name} must have shape [B,T] on the RGB device")
        if frame_ids.is_floating_point() or frame_ids.dtype == torch.bool:
            raise TypeError("frame_ids must be an integer tensor")
        if manifest_indices.is_floating_point() or manifest_indices.dtype == torch.bool:
            raise TypeError("manifest_indices must be an integer tensor")
        if frames > 1:
            provenance_increases = (
                (frame_ids[:, 1:] > frame_ids[:, :-1]).all()
                & (timestamps[:, 1:] > timestamps[:, :-1]).all()
                & (manifest_indices[:, 1:] > manifest_indices[:, :-1]).all()
            )
            _assert_tensor_condition(
                provenance_increases,
                "causal provenance must be strictly increasing oldest-to-current",
            )

        intrinsics = _require_tensor(batch, "K", ndim=5, floating=True)
        if intrinsics.shape != (batch_size, frames, 2, 3, 3):
            raise ValueError("batch['K'] must have shape [B,T,2,3,3]")
        with torch.autocast(device_type=intrinsics.device.type, enabled=False):
            calibrated = intrinsics.float()
            matching_rectified_intrinsics = torch.isclose(
                calibrated[:, :, 0],
                calibrated[:, :, 1],
                atol=1e-4,
                rtol=1e-5,
            ).all()
        _assert_tensor_condition(
            matching_rectified_intrinsics,
            "d=fx*baseline*inverse_depth requires matching rectified left/right K",
        )
        _assert_tensor_condition(
            torch.isfinite(intrinsics).all()
            & ((intrinsics[..., 0, 0] > 0) & (intrinsics[..., 1, 1] > 0)).all(),
            "batch intrinsics must be finite with positive focal lengths",
        )
        homogeneous_K_row = intrinsics[..., 2, :]
        _assert_tensor_condition(
            (homogeneous_K_row[..., :2].abs() <= 1e-6).all()
            & ((homogeneous_K_row[..., 2] - 1.0).abs() <= 1e-6).all(),
            "batch intrinsics have malformed homogeneous rows",
        )
        stereo_transform = _require_tensor(
            batch, "T_right_from_left", ndim=4, floating=True
        )
        temporal_transform = _require_tensor(
            batch, "T_current_from_previous", ndim=4, floating=True
        )
        expected_transform_shape = (batch_size, frames, 4, 4)
        if stereo_transform.shape != expected_transform_shape:
            raise ValueError("T_right_from_left must have shape [B,T,4,4]")
        if temporal_transform.shape != expected_transform_shape:
            raise ValueError("T_current_from_previous must have shape [B,T,4,4]")
        for name, transform in (
            ("T_right_from_left", stereo_transform),
            ("T_current_from_previous", temporal_transform),
        ):
            homogeneous_transform_row = transform[..., 3, :]
            _assert_tensor_condition(
                torch.isfinite(transform).all()
                & (homogeneous_transform_row[..., :3].abs() <= 1e-5).all()
                & (
                    (homogeneous_transform_row[..., 3] - 1.0).abs() <= 1e-5
                ).all(),
                f"batch[{name!r}] contains an invalid transform",
            )
            with torch.autocast(device_type=transform.device.type, enabled=False):
                rotation = transform[..., :3, :3].float()
                identity = torch.eye(
                    3, device=rotation.device, dtype=torch.float32
                ).expand_as(rotation)
                rotation_valid = torch.isclose(
                    rotation.transpose(-1, -2) @ rotation,
                    identity,
                    atol=2e-3,
                    rtol=2e-3,
                ).all()
            _assert_tensor_condition(
                rotation_valid,
                f"batch[{name!r}] rotation is not orthonormal",
            )
        with torch.autocast(
            device_type=stereo_transform.device.type, enabled=False
        ):
            stereo_rotation = stereo_transform[..., :3, :3].float()
            identity_rotation = torch.eye(
                3, device=stereo_rotation.device, dtype=torch.float32
            ).expand_as(stereo_rotation)
            rectified_rotation = torch.isclose(
                stereo_rotation, identity_rotation, atol=2e-4, rtol=0.0
            ).all()
            stereo_translation = stereo_transform[..., :3, 3].float()
            horizontal_translation = (
                stereo_translation[..., 0].abs() > 1e-8
            ) & torch.isclose(
                stereo_translation[..., 1:],
                torch.zeros_like(stereo_translation[..., 1:]),
                atol=1e-6,
                rtol=0.0,
            ).all(dim=-1)
        _assert_tensor_condition(
            rectified_rotation & horizontal_translation.all(),
            "disparity output requires identity stereo rotation and x-only baseline",
        )

        temporal_valid_name = (
            "T_current_from_previous_valid"
            if "T_current_from_previous_valid" in batch
            else "temporal_transform_valid"
        )
        temporal_valid = _require_tensor(
            batch, temporal_valid_name, ndim=2, floating=False
        )
        if temporal_valid.shape != (batch_size, frames):
            raise ValueError("temporal transform validity must have shape [B,T]")
        temporal_pattern_valid = ~temporal_valid[:, 0].any()
        if frames > 1:
            temporal_pattern_valid = temporal_pattern_valid & temporal_valid[
                :, 1:
            ].all()
        _assert_tensor_condition(
            temporal_pattern_valid,
            "temporal transforms must be invalid at t=0 and valid thereafter",
        )
        device = rgb.device
        for name, value in (
            ("K", intrinsics),
            ("T_right_from_left", stereo_transform),
            ("T_current_from_previous", temporal_transform),
            ("time_valid_mask", time_valid),
            (temporal_valid_name, temporal_valid),
        ):
            if value.device != device:
                raise ValueError(f"batch[{name!r}] must share the RGB device")

        if "baseline_m" in batch:
            baseline = _require_tensor(batch, "baseline_m", ndim=2, floating=True)
            if baseline.shape != (batch_size, frames):
                raise ValueError("baseline_m must have shape [B,T]")
            if baseline.device != device:
                raise ValueError("baseline_m must share the RGB device")
            with torch.autocast(
                device_type=stereo_transform.device.type, enabled=False
            ):
                transform_baseline = torch.linalg.vector_norm(
                    stereo_transform[..., :3, 3].float(), dim=-1
                )
            tolerance = torch.maximum(
                torch.full_like(transform_baseline, 1e-6),
                1e-3 * baseline.float(),
            )
            _assert_tensor_condition(
                (
                    torch.isfinite(baseline)
                    & (baseline > 0)
                    & ((transform_baseline - baseline.float()).abs() <= tolerance)
                ).all(),
                "baseline_m disagrees with T_right_from_left",
            )
        return (
            rgb,
            intrinsics,
            stereo_transform,
            temporal_transform,
            time_valid,
            temporal_valid,
        )

    @staticmethod
    def _validate_stereo_output(
        output: TrainableStereoOutput,
        *,
        batch: int,
        frames: int,
        size_lr: tuple[int, int],
        feature_level: int,
    ) -> Tensor:
        if not isinstance(output, TrainableStereoOutput):
            raise TypeError("stereo backbone must return TrainableStereoOutput")
        expected_disparity = (batch, frames, 1, *size_lr)
        if output.disparity_left_lr_px.shape != expected_disparity:
            raise ValueError(
                "stereo left disparity has wrong shape: "
                f"{tuple(output.disparity_left_lr_px.shape)} != {expected_disparity}"
            )
        if output.disparity_right_lr_px is not None and (
            output.disparity_right_lr_px.shape != expected_disparity
        ):
            raise ValueError("stereo right disparity must match left disparity")
        if not output.iteration_disparities_left_lr_px:
            raise ValueError("stereo backbone returned no iterative disparities")
        if any(
            prediction.shape != expected_disparity
            for prediction in output.iteration_disparities_left_lr_px
        ):
            raise ValueError("every iterative stereo disparity must match final left")
        if feature_level >= len(output.left_features):
            raise ValueError("configured stereo feature level is unavailable")
        feature = output.left_features[feature_level]
        if feature.ndim != 5 or feature.shape[:2] != (batch, frames):
            raise ValueError("selected stereo feature must have shape [B,T,C,h,w]")
        return feature

    @staticmethod
    def _validate_vggt_output(
        output: TrainableVGGTOmegaOutput,
        *,
        batch: int,
    ) -> None:
        if not isinstance(output, TrainableVGGTOmegaOutput):
            raise TypeError("VGGT backbone must return TrainableVGGTOmegaOutput")
        for name, value, channels in (
            ("depth_current_arbitrary", output.depth_current_arbitrary, 1),
            ("confidence_current_unbounded", output.confidence_current_unbounded, 1),
            ("geometry_current", output.geometry_current, None),
        ):
            if (
                value.ndim != 4
                or value.shape[0] != batch
                or (channels is not None and value.shape[1] != channels)
                or not value.is_floating_point()
            ):
                raise ValueError(f"VGGT {name} has an invalid [B,C,H,W] contract")
        if output.confidence_current_unbounded.shape != (
            output.depth_current_arbitrary.shape
        ):
            raise ValueError("VGGT depth and confidence must share one shape")
        device = output.depth_current_arbitrary.device
        if output.confidence_current_unbounded.device != device or (
            output.geometry_current.device != device
        ):
            raise ValueError("all VGGT outputs must share one device")

    def forward(self, batch: Mapping[str, Any]) -> MetricStereoVideoSystemOutput:
        """Run both backbones once, causally warm state, and return the endpoint."""

        _debug_phase("system_forward_start")
        (
            rgb,
            intrinsics,
            stereo_transform,
            temporal_transform,
            _time_valid,
            _temporal_valid,
        ) = self._validate_batch(batch)
        _debug_phase("batch_validation_end")
        batch_size, frames, _, _, height_hr, width_hr = rgb.shape
        left_rgb = rgb[:, :, 0]
        right_rgb = rgb[:, :, 1]

        left_lr, right_lr = half_resolution_stereo_images(left_rgb, right_rgb)
        _debug_phase("stereo_resize_end")
        _debug_phase("stereo_start")
        stereo_output = self.stereo_backbone(left_lr, right_lr)
        _debug_phase("stereo_end")
        stereo_feature = self._validate_stereo_output(
            stereo_output,
            batch=batch_size,
            frames=frames,
            size_lr=(height_hr // 2, width_hr // 2),
            feature_level=self.stereo_feature_level,
        )
        if self.require_right_disparity and (
            stereo_output.disparity_right_lr_px is None
        ):
            raise RuntimeError(
                "stereo backbone omitted right disparity required for LR consistency"
            )
        # FFS values are pixels of its x2-downsampled input.  Spatial upsampling
        # alone does not change units, so the explicit x2 factor is mandatory.
        left_disparity_hr_units = 2.0 * stereo_output.disparity_left_lr_px
        right_disparity_hr_units = (
            None
            if stereo_output.disparity_right_lr_px is None
            else 2.0 * stereo_output.disparity_right_lr_px
        )
        lr_consistency = left_right_stereo_consistency(
            stereo_output.disparity_left_lr_px,
            stereo_output.disparity_right_lr_px,
            maximum_error_px=self.left_right_maximum_error_lr_px,
            confidence_temperature_px=self.left_right_confidence_temperature_lr_px,
        )
        stereo_consistency = StereoConsistencyDiagnostics(
            disparity_left_lr_px=stereo_output.disparity_left_lr_px,
            disparity_right_lr_px=stereo_output.disparity_right_lr_px,
            disparity_left_hr_px_lr_grid=left_disparity_hr_units,
            disparity_right_hr_px_lr_grid=right_disparity_hr_units,
            sampled_right_hr_px_lr_grid=2.0 * lr_consistency.sampled_right_px,
            left_right_error_lr_px=lr_consistency.error_px,
            left_right_error_hr_px_lr_grid=2.0 * lr_consistency.error_px,
            valid_left_mask_lr=lr_consistency.valid_left_mask,
            confidence_left_lr=lr_consistency.confidence_left,
        )
        ffs_iterations_hr_units = tuple(
            2.0 * prediction
            for prediction in stereo_output.iteration_disparities_left_lr_px
        )

        # Exactly one globally attentive call, on a prefix ending at the target.
        _debug_phase("vggt_start")
        vggt_output = self.vggt_backbone(left_rgb)
        _debug_phase("vggt_end")
        self._validate_vggt_output(vggt_output, batch=batch_size)
        depth_vggt = vggt_output.depth_current_arbitrary
        depth_valid = torch.isfinite(depth_vggt) & (depth_vggt > 0)
        inverse_depth_vggt = torch.where(
            depth_valid,
            depth_vggt.clamp_min(1e-8).reciprocal(),
            torch.zeros_like(depth_vggt),
        )
        vggt_confidence = vggt_unbounded_confidence_to_probability(
            vggt_output.confidence_current_unbounded
        ) * depth_valid.to(dtype=depth_vggt.dtype)

        invalid_vggt_feature = torch.zeros_like(vggt_output.geometry_current)
        invalid_vggt_inverse = torch.zeros_like(inverse_depth_vggt)
        invalid_vggt_confidence = torch.zeros_like(vggt_confidence)
        state = None
        endpoint: MetricStereoVideoGeometryOutput | None = None
        for time_index in range(frames):
            is_endpoint = time_index == frames - 1
            if is_endpoint and (
                self.enable_vggt_dense_features or self.enable_vggt_geometry
            ):
                vggt_feature = (
                    vggt_output.geometry_current
                    if self.enable_vggt_dense_features
                    else invalid_vggt_feature
                )
                vggt_inverse = (
                    inverse_depth_vggt
                    if self.enable_vggt_geometry
                    else invalid_vggt_inverse
                )
                vggt_probability = (
                    vggt_confidence
                    if self.enable_vggt_geometry
                    else invalid_vggt_confidence
                )
                context_indices = tuple(range(frames))
            else:
                # A zero feature with zero confidence is explicit missing data.
                # It cannot carry the endpoint's globally aggregated future state.
                vggt_feature = invalid_vggt_feature
                vggt_inverse = invalid_vggt_inverse
                vggt_probability = invalid_vggt_confidence
                context_indices = (time_index,)
            frame = MetricStereoFrameInput(
                left_rgb=left_rgb[:, time_index],
                right_rgb=right_rgb[:, time_index],
                intrinsics_left_3x3=intrinsics[:, time_index, 0],
                T_right_from_left_m=stereo_transform[:, time_index],
                T_current_from_previous_m=(
                    None
                    if time_index == 0
                    else temporal_transform[:, time_index]
                ),
                lowres_disparity_left_px=left_disparity_hr_units[:, time_index],
                lowres_disparity_valid_mask=(
                    stereo_consistency.valid_left_mask_lr[:, time_index]
                ),
                lowres_disparity_confidence=(
                    stereo_consistency.confidence_left_lr[:, time_index]
                ),
                stereo_features=StereoBackboneFeatures(
                    feature_map=stereo_feature[:, time_index],
                    time_index=time_index,
                ),
                vggt_features=VGGTCausalGeometryFeatures(
                    feature_map=vggt_feature,
                    inverse_depth_relative=vggt_inverse,
                    confidence=vggt_probability,
                    context_time_indices=context_indices,
                    time_index=time_index,
                ),
                time_index=time_index,
            )
            _debug_phase("geometry_forward_step_start", time_index=time_index)
            endpoint = self.geometry_model.forward_step(frame, state)
            _debug_phase("geometry_forward_step_end", time_index=time_index)
            state = endpoint.state
        assert endpoint is not None

        right_disparity_hr: Tensor | None = None
        if right_disparity_hr_units is not None:
            right_disparity_hr = F.interpolate(
                right_disparity_hr_units[:, -1],
                size=(height_hr, width_hr),
                mode="bilinear",
                align_corners=False,
            )
        return MetricStereoVideoSystemOutput(
            endpoint=endpoint,
            disparity_right_px=right_disparity_hr,
            ffs_iteration_disparities_left_hr_px_lr_grid=(
                ffs_iterations_hr_units
            ),
            stereo=stereo_consistency,
            vggt_inverse_depth_relative_endpoint=inverse_depth_vggt,
            vggt_confidence_endpoint=vggt_confidence,
            gauge=endpoint.gauge,
        )


__all__ = [
    "LeftRightConsistencyResult",
    "MetricStereoVideoSystem",
    "MetricStereoVideoSystemOutput",
    "StereoConsistencyDiagnostics",
    "left_right_stereo_consistency",
    "vggt_unbounded_confidence_to_probability",
]
