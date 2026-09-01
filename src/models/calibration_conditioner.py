"""Calibration conditioning for the opt-in FFS-\u03a9-TSR architecture v3.

The conditioner follows the useful part of MapAnything's input factorization:
camera intrinsics become dense local ray directions, while camera transforms
are split into rotation and translation direction before being broadcast over
the spatial grid.  It deliberately does *not* predict or apply metric scale.

All public extrinsics in this module are homogeneous camera-from-camera
transforms. In particular, ``T_right_rectified_from_left_rectified_m`` obeys

``X_right = R_right_from_left @ X_left + t_right_from_left``.

Temporal slots are ordered ``[age1, age2]`` and map the corresponding history
camera into the current camera.  Translation magnitudes are represented only
as the dimensionless ratio ``log1p(||t|| / baseline_m)``.  The stereo baseline
itself is validated against the static transform but is never sent to a
learned scale head (there is no such head in this architecture).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .rgb_encoder import ConvNormAct


CALIBRATION_FEATURE_CHANNELS = 64
TEMPORAL_POSE_AGES = 2


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_floating_tensor(
    value: Tensor | None,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}, got {value.device}")
    value_fp32 = value.detach()
    if not bool(torch.isfinite(value_fp32).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value_fp32


def _validate_homogeneous_rows(transforms: Tensor, *, name: str) -> None:
    expected = transforms.new_tensor((0.0, 0.0, 0.0, 1.0))
    rows = transforms[..., 3, :]
    if not bool(
        torch.isclose(
            rows,
            expected.expand_as(rows),
            atol=1e-6,
            rtol=0.0,
        ).all()
    ):
        raise ValueError(f"{name} must have homogeneous bottom row [0,0,0,1]")


def _validate_rotation_matrices(
    rotations: Tensor,
    *,
    name: str,
    valid_mask: Tensor | None = None,
) -> None:
    """Require proper SO(3) rotations, optionally only at valid slots."""

    flat = rotations.reshape(-1, 3, 3)
    if valid_mask is not None:
        valid = valid_mask.reshape(-1)
        if valid.shape[0] != flat.shape[0]:
            raise ValueError(f"{name} validity mask does not match rotations")
        flat = flat[valid]
    if flat.numel() == 0:
        return
    with torch.autocast(device_type=flat.device.type, enabled=False):
        flat_fp32 = flat.float()
        identity = torch.eye(3, dtype=torch.float32, device=flat.device)
        gram = flat_fp32.transpose(-1, -2) @ flat_fp32
        determinants = torch.linalg.det(flat_fp32)
        orthonormal = torch.isclose(
            gram,
            identity.expand_as(gram),
            atol=1e-4,
            rtol=1e-4,
        ).all()
        proper = torch.isclose(
            determinants,
            torch.ones_like(determinants),
            atol=1e-4,
            rtol=1e-4,
        ).all()
    if not bool(orthonormal and proper):
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")


def _rotation_6d(rotations: Tensor) -> Tensor:
    """Return first-column then second-column SO(3) representation."""

    return torch.cat((rotations[..., :, 0], rotations[..., :, 1]), dim=-1)


def dense_unit_rays_from_K_hr(
    K_left_hr_px: Tensor,
    *,
    height_lr: int,
    width_lr: int,
    spatial_scale: int = 2,
) -> Tensor:
    """Build FP32 left-camera unit rays on the LR model grid.

    Args:
        K_left_hr_px: Cropped HR pinhole intrinsics ``[B,3,3]`` in pixels.
        height_lr, width_lr: LR geometry-grid dimensions.
        spatial_scale: HR/LR spatial scale.  The repository convention scales
            the first two rows of ``K`` without a half-pixel offset.

    Returns:
        FP32 unit ray directions ``[B,3,H,W]`` on the input device.
    """

    if not isinstance(K_left_hr_px, Tensor) or not K_left_hr_px.is_floating_point():
        raise TypeError("K_left_hr_px must be a floating-point torch.Tensor")
    if K_left_hr_px.dtype != torch.float32:
        raise TypeError("K_left_hr_px must have dtype torch.float32")
    if K_left_hr_px.ndim != 3 or tuple(K_left_hr_px.shape[-2:]) != (3, 3):
        raise ValueError(
            "K_left_hr_px must have shape [B,3,3], got "
            f"{tuple(K_left_hr_px.shape)}"
        )
    for value, name in ((height_lr, "height_lr"), (width_lr, "width_lr")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(spatial_scale, bool)
        or not isinstance(spatial_scale, int)
        or spatial_scale <= 0
    ):
        raise ValueError("spatial_scale must be a positive integer")

    with torch.autocast(device_type=K_left_hr_px.device.type, enabled=False):
        intrinsics_hr = K_left_hr_px.detach()
        if not bool(torch.isfinite(intrinsics_hr).all()):
            raise ValueError("K_left_hr_px must contain only finite values")
        batch_size = intrinsics_hr.shape[0]
        if not bool(
            ((intrinsics_hr[:, 0, 0] > 0) & (intrinsics_hr[:, 1, 1] > 0)).all()
        ):
            raise ValueError("K_left_hr_px focal lengths must be positive")
        expected_last_row = intrinsics_hr.new_tensor((0.0, 0.0, 1.0))
        if not bool(
            torch.isclose(
                intrinsics_hr[:, 2],
                expected_last_row.expand(batch_size, -1),
                atol=1e-6,
                rtol=0.0,
            ).all()
        ):
            raise ValueError("K_left_hr_px must end with row [0,0,1]")

        intrinsics_lr = intrinsics_hr.clone()
        intrinsics_lr[:, 0, :] /= float(spatial_scale)
        intrinsics_lr[:, 1, :] /= float(spatial_scale)
        intrinsics_lr[:, 2] = intrinsics_hr[:, 2]

        v_grid, u_grid = torch.meshgrid(
            torch.arange(height_lr, dtype=torch.float32, device=intrinsics_hr.device),
            torch.arange(width_lr, dtype=torch.float32, device=intrinsics_hr.device),
            indexing="ij",
        )
        fx = intrinsics_lr[:, 0, 0].reshape(batch_size, 1, 1)
        fy = intrinsics_lr[:, 1, 1].reshape(batch_size, 1, 1)
        cx = intrinsics_lr[:, 0, 2].reshape(batch_size, 1, 1)
        cy = intrinsics_lr[:, 1, 2].reshape(batch_size, 1, 1)
        x = (u_grid.unsqueeze(0) - cx) / fx
        y = (v_grid.unsqueeze(0) - cy) / fy
        rays = torch.stack((x, y, torch.ones_like(x)), dim=1)
        return torch.nn.functional.normalize(rays, dim=1, eps=1e-8)


class CalibrationConditionerV3(nn.Module):
    """Encode rays and factorized stereo/temporal poses into a 64ch residual."""

    output_channels = CALIBRATION_FEATURE_CHANNELS

    def __init__(
        self,
        *,
        spatial_scale: int = 2,
        use_rays: bool = True,
        use_stereo_pose: bool = True,
        use_temporal_pose: bool = True,
    ) -> None:
        super().__init__()
        if (
            isinstance(spatial_scale, bool)
            or not isinstance(spatial_scale, int)
            or spatial_scale <= 0
        ):
            raise ValueError("spatial_scale must be a positive integer")
        self.spatial_scale = spatial_scale
        self.use_rays = _validate_bool(use_rays, "use_rays")
        self.use_stereo_pose = _validate_bool(use_stereo_pose, "use_stereo_pose")
        self.use_temporal_pose = _validate_bool(
            use_temporal_pose, "use_temporal_pose"
        )

        # All branches are instantiated irrespective of switches.  This keeps
        # ray/pose ablations parameter matched within the v3 lineage.
        self.ray_encoder = nn.Sequential(
            ConvNormAct(3, 32),
            ConvNormAct(32, self.output_channels),
        )
        self.stereo_pose_encoder = nn.Sequential(
            nn.Linear(9, self.output_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.output_channels, self.output_channels),
            nn.LayerNorm(self.output_channels),
        )
        # Age-1 and age-2 are encoded by independent, parameter-matched MLPs.
        # Masking is applied *after* each MLP, so an invalid age embedding is
        # exactly zero even after its encoder has learned affine biases.
        self.temporal_pose_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(10, self.output_channels),
                nn.SiLU(inplace=True),
                nn.Linear(self.output_channels, self.output_channels),
                nn.LayerNorm(self.output_channels),
            )
            for _ in range(TEMPORAL_POSE_AGES)
        )
        self.fusion_norm = nn.GroupNorm(8, self.output_channels)
        self.output_adapter = nn.Conv2d(
            self.output_channels,
            self.output_channels,
            kernel_size=1,
            bias=False,
        )
        # Enabling v3 starts as an exact no-op with respect to the legacy
        # geometry feature.  The adapter learns first, then opens gradients to
        # the modality encoders on subsequent updates.
        nn.init.zeros_(self.output_adapter.weight)

    def _validate_inputs(
        self,
        reference_feature_lr: Tensor,
        *,
        K_left_hr_px: Tensor | None,
        baseline_m: Tensor | None,
        T_right_rectified_from_left_rectified_m: Tensor | None,
        T_current_from_history_m: Tensor | None,
        temporal_pose_valid: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if (
            not isinstance(reference_feature_lr, Tensor)
            or not reference_feature_lr.is_floating_point()
        ):
            raise TypeError("reference_feature_lr must be a floating-point Tensor")
        if (
            reference_feature_lr.ndim != 4
            or reference_feature_lr.shape[1] != self.output_channels
        ):
            raise ValueError(
                "reference_feature_lr must have shape [B,64,H,W], got "
                f"{tuple(reference_feature_lr.shape)}"
            )
        batch_size = reference_feature_lr.shape[0]
        device = reference_feature_lr.device

        intrinsics: Tensor | None = None
        if self.use_rays:
            intrinsics = _require_floating_tensor(
                K_left_hr_px,
                name="K_left_hr_px",
                shape=(batch_size, 3, 3),
                device=device,
            )
        elif K_left_hr_px is not None:
            raise ValueError("K_left_hr_px was provided while use_rays is disabled")

        needs_static = self.use_stereo_pose or self.use_temporal_pose
        baseline: Tensor | None = None
        static_extrinsics: Tensor | None = None
        if needs_static:
            baseline = _require_floating_tensor(
                baseline_m,
                name="baseline_m",
                shape=(batch_size,),
                device=device,
            )
            if not bool((baseline > 0).all()):
                raise ValueError("baseline_m must be strictly positive")
            static_extrinsics = _require_floating_tensor(
                T_right_rectified_from_left_rectified_m,
                name="T_right_rectified_from_left_rectified_m",
                shape=(batch_size, 4, 4),
                device=device,
            )
            _validate_homogeneous_rows(
                static_extrinsics,
                name="T_right_rectified_from_left_rectified_m",
            )
            _validate_rotation_matrices(
                static_extrinsics[:, :3, :3],
                name="T_right_rectified_from_left_rectified_m",
            )
            static_norm = torch.linalg.vector_norm(
                static_extrinsics[:, :3, 3], dim=-1
            )
            if not bool((static_norm > 0).all()):
                raise ValueError(
                    "T_right_rectified_from_left_rectified_m translation must be non-zero"
                )
            if not bool(
                torch.isclose(
                    static_norm,
                    baseline,
                    atol=1e-6,
                    rtol=1e-5,
                ).all()
            ):
                raise ValueError(
                    "static stereo translation norm must match baseline_m"
                )
        else:
            if baseline_m is not None:
                raise ValueError(
                    "baseline_m was provided while stereo/temporal pose are disabled"
                )
            if T_right_rectified_from_left_rectified_m is not None:
                raise ValueError(
                    "T_right_rectified_from_left_rectified_m was provided while pose inputs are disabled"
                )

        temporal_extrinsics: Tensor | None = None
        temporal_mask: Tensor | None = None
        if self.use_temporal_pose:
            temporal_extrinsics = _require_floating_tensor(
                T_current_from_history_m,
                name="T_current_from_history_m",
                shape=(batch_size, TEMPORAL_POSE_AGES, 4, 4),
                device=device,
            )
            _validate_homogeneous_rows(
                temporal_extrinsics,
                name="T_current_from_history_m",
            )
            if not isinstance(temporal_pose_valid, Tensor):
                raise TypeError("temporal_pose_valid must be a torch.Tensor")
            if temporal_pose_valid.dtype != torch.bool:
                raise TypeError("temporal_pose_valid must have dtype torch.bool")
            if tuple(temporal_pose_valid.shape) != (
                batch_size,
                TEMPORAL_POSE_AGES,
            ):
                raise ValueError(
                    "temporal_pose_valid must have shape "
                    f"{(batch_size, TEMPORAL_POSE_AGES)}, got "
                    f"{tuple(temporal_pose_valid.shape)}"
                )
            if temporal_pose_valid.device != device:
                raise ValueError(
                    "temporal_pose_valid must share the model input device"
                )
            temporal_mask = temporal_pose_valid.detach()
            _validate_rotation_matrices(
                temporal_extrinsics[:, :, :3, :3],
                name="T_current_from_history_m",
                valid_mask=temporal_mask,
            )
        else:
            if T_current_from_history_m is not None:
                raise ValueError(
                    "temporal extrinsics were provided while use_temporal_pose is disabled"
                )
            if temporal_pose_valid is not None:
                raise ValueError(
                    "temporal pose mask was provided while use_temporal_pose is disabled"
                )

        return (
            intrinsics,
            baseline,
            static_extrinsics,
            temporal_extrinsics,
            temporal_mask,
        )

    def forward(
        self,
        reference_feature_lr: Tensor,
        *,
        K_left_hr_px: Tensor | None = None,
        baseline_m: Tensor | None = None,
        T_right_rectified_from_left_rectified_m: Tensor | None = None,
        T_current_from_history_m: Tensor | None = None,
        temporal_pose_valid: Tensor | None = None,
    ) -> Tensor:
        """Return a zero-initialized calibration residual ``[B,64,H,W]``."""

        with torch.autocast(
            device_type=reference_feature_lr.device.type, enabled=False
        ):
            (
                intrinsics,
                baseline,
                static_extrinsics,
                temporal_extrinsics,
                temporal_mask,
            ) = self._validate_inputs(
                reference_feature_lr,
                K_left_hr_px=K_left_hr_px,
                baseline_m=baseline_m,
                T_right_rectified_from_left_rectified_m=(
                    T_right_rectified_from_left_rectified_m
                ),
                T_current_from_history_m=(
                    T_current_from_history_m
                ),
                temporal_pose_valid=temporal_pose_valid,
            )
        batch_size, _, height_lr, width_lr = reference_feature_lr.shape
        target_dtype = reference_feature_lr.dtype
        fused = torch.zeros_like(reference_feature_lr)

        if self.use_rays:
            assert intrinsics is not None
            rays_fp32 = dense_unit_rays_from_K_hr(
                intrinsics,
                height_lr=height_lr,
                width_lr=width_lr,
                spatial_scale=self.spatial_scale,
            )
            ray_feature = self.ray_encoder(rays_fp32.to(dtype=target_dtype))
            fused = fused + ray_feature

        if self.use_stereo_pose:
            assert static_extrinsics is not None
            with torch.autocast(
                device_type=reference_feature_lr.device.type, enabled=False
            ):
                static_rotation = static_extrinsics[:, :3, :3]
                static_translation = static_extrinsics[:, :3, 3]
                static_norm = torch.linalg.vector_norm(
                    static_translation, dim=-1, keepdim=True
                )
                static_pose = torch.cat(
                    (
                        _rotation_6d(static_rotation),
                        static_translation / static_norm,
                    ),
                    dim=-1,
                )
            static_feature = self.stereo_pose_encoder(
                static_pose.to(dtype=target_dtype)
            )
            fused = fused + static_feature.reshape(
                batch_size, self.output_channels, 1, 1
            )

        if self.use_temporal_pose:
            assert baseline is not None
            assert temporal_extrinsics is not None
            assert temporal_mask is not None
            with torch.autocast(
                device_type=reference_feature_lr.device.type, enabled=False
            ):
                temporal_rotation = temporal_extrinsics[:, :, :3, :3]
                temporal_translation = temporal_extrinsics[:, :, :3, 3]
                temporal_norm = torch.linalg.vector_norm(
                    temporal_translation, dim=-1, keepdim=True
                )
                temporal_direction = (
                    temporal_translation / temporal_norm.clamp_min(1e-8)
                )
                temporal_scale_ratio = torch.log1p(
                    temporal_norm / baseline.reshape(batch_size, 1, 1)
                )
                temporal_pose = torch.cat(
                    (
                        _rotation_6d(temporal_rotation),
                        temporal_direction,
                        temporal_scale_ratio,
                    ),
                    dim=-1,
                )
            age_features: list[Tensor] = []
            for age_index, encoder in enumerate(self.temporal_pose_encoders):
                age_mask = temporal_mask[:, age_index : age_index + 1]
                pose_for_age = torch.where(
                    age_mask,
                    temporal_pose[:, age_index],
                    torch.zeros_like(temporal_pose[:, age_index]),
                )
                feature = encoder(
                    pose_for_age.to(dtype=target_dtype)
                )
                # Post-MLP masking is the exact-zero contract. It prevents
                # learned Linear/LayerNorm biases from leaking an invalid age.
                feature = feature * age_mask.to(dtype=feature.dtype)
                age_features.append(feature)
            temporal_feature = torch.stack(age_features, dim=1).sum(dim=1)
            fused = fused + temporal_feature.reshape(
                batch_size, self.output_channels, 1, 1
            )

        # Avoid running GroupNorm with a degenerate single-value group in tiny
        # synthetic tests; real LR grids are always much larger.  The condition
        # is based solely on shape and does not alter formal model behavior.
        values_per_group = (
            self.output_channels // self.fusion_norm.num_groups
        ) * height_lr * width_lr
        if values_per_group > 1:
            fused = self.fusion_norm(fused)
        return self.output_adapter(fused)


__all__ = [
    "CALIBRATION_FEATURE_CHANNELS",
    "TEMPORAL_POSE_AGES",
    "CalibrationConditionerV3",
    "dense_unit_rays_from_K_hr",
]
