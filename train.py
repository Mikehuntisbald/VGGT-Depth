#!/usr/bin/env python3
"""Stage-A spatial and Stage-B causal training from frozen-backbone caches.

The learning-rate schedule is linear warmup for ``warmup_steps`` optimizer
updates followed by cosine decay to zero at the final configured update.
Micro-batches are accumulated before each optimizer update; ``steps_spatial``
and temporal ``steps`` therefore count updates, not DataLoader iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Sampler


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import CacheIdentity, sha256_file  # noqa: E402
from data.collate import (  # noqa: E402
    collate_temporal_training_samples,
    collate_training_samples,
)
from data.manifest import load_manifest  # noqa: E402
from data.stereo_calibration import (  # noqa: E402
    RectifiedCalibrationIndex,
    load_rectified_calibration_sidecar,
)
from data.training_dataset import CachedFFSTrainingDataset  # noqa: E402
from data.temporal_training_dataset import (  # noqa: E402
    CachedTemporalTrainingDataset,
)
from losses import (  # noqa: E402
    LossBreakdown,
    LossWeights,
    combine_loss_terms,
    disparity_loss,
    ffs_gate_regularizer,
    gradient_loss,
    laplace_uncertainty_nll,
    lower_bound_penalty,
    measurement_consistency_loss,
    sample_hr_at_lr_centers,
    temporal_consistency_loss,
    temporal_residual_consistency_loss,
    validity_completion_loss,
)
from geometry.history_confidence import history_confidence  # noqa: E402
from geometry.camera import resize_intrinsics_align_corners_false  # noqa: E402
from geometry.calibration_context import (  # noqa: E402
    rectified_stereo_transform_4x4,
    temporal_conditioning_transforms,
)
from geometry.topk_splat import (  # noqa: E402
    TOPK_DIVERSITY_V31_CONTRACT,
    TopKSplatResult,
    merge_topk_splat_results,
    topk_z_aware_splat,
)
from geometry.zbuffer_reproject import WarpResult, zbuffer_reproject  # noqa: E402
from models.ffs_omega_tsr import (  # noqa: E402
    FFSOmegaTSR,
    ModelOutput,
    count_trainable_parameters,
)
from utils.checkpoint import (  # noqa: E402
    config_fingerprint,
    load_model_initialization_checkpoint,
    load_training_checkpoint,
    repository_git_hash,
    save_training_checkpoint,
)
from utils.seed import seed_data_worker, seed_everything  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "ffs_omega_tsr_x2",
    "seed": 42,
    "data": {
        "scale": 2,
        "hr_crop": [384, 768],
        "sequence_length": 1,
        "vggt_context_pairs": 5,
        "crop_origin_multiple": 2,
        "cache_dtype": "float16",
        "manifest_path": None,
        "observation_cache_root": None,
        "teacher_cache_root": None,
        "derived_geometry_cache_root": None,
        "calibration_sidecar_path": None,
        "derived_contract": "legacy_v1",
        "observation_cache_identity": None,
        "teacher_cache_identity": None,
        "derived_cache_lineage": None,
        "calibration_sidecar_lineage": None,
        "crop_mode": "random",
        # Temporal transport defaults to the frozen VGGT pose cache.  Spring
        # ablations can set this to ``gt`` without changing the VGGT cache
        # lineage; the temporal dataset exposes a separate GT pose tensor.
        "temporal_pose_source": "vggt",
    },
    "ffs": {
        "observation_iters": 4,
        "teacher_iters": 8,
        "volume_backend": "pytorch1",
        "return_aux": True,
        "right_left_check": True,
        "max_disp_lr": "auto",
    },
    "vggt": {
        "model": "1B-512",
        "input_mode": "balanced",
        "use_stereo_pairs": True,
        "causal": True,
        "cache_depth": True,
        "cache_depth_conf": True,
        "cache_extrinsics": True,
        "cache_registers": True,
        "use_registers_in_model": False,
    },
    "calibration_conditioning_v3": {
        "enabled": False,
        "protocol_version": "disabled",
        "use_rays": False,
        "use_stereo_pose": False,
        "use_temporal_pose": False,
    },
    "model": {
        "rgb_channels": [32, 64, 96],
        "geometry_channels": 64,
        "hidden_dim": 96,
        "gru_layers": 2,
        "residual_limit_hr_px": 8.0,
        "convex_scale": 2,
        "predict_uncertainty": True,
        "epipolar_refinement": False,
        "use_history": False,
        "use_vggt_pose": False,
        # Explicit ablation switch: VGGT depth prior can be removed while
        # retaining the same model/checkpoint graph and (optionally) pose.
        "use_vggt_depth": True,
        "use_registers_in_model": False,
    },
    "train": {
        "precision": "bf16",
        "micro_batch_size": 2,
        "grad_accumulation": 4,
        "effective_batch_size": 8,
        "optimizer": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_steps": 500,
        "gradient_clip": 1.0,
        "steps_spatial": 5000,
        "steps_temporal": 15000,
        "stage": "spatial",
        "steps": 15000,
        "init_from_stage": None,
        "initialization_checkpoint": None,
        "initialization_checkpoint_sha256": None,
        "history_detach": True,
        "photometric_temperature": 0.10,
        "disparity_temperature_hr_px": 2.0,
        "history_conflict_hr_px": 2.0,
        "temporal_photometric_threshold": 0.10,
        "temporal_geometry_threshold_hr_px": 2.0,
        "finite_diagnostic_interval": 0,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "compile_model": False,
        "checkpoint_interval": 500,
        "log_interval": 1,
        "output_dir": None,
    },
    "loss": {
        "disparity": 1.00,
        "measurement": 0.50,
        "gradient": 0.20,
        "temporal": 0.10,
        "epipolar": 0.05,
        "uncertainty_nll": 0.01,
        "gate_regularizer": 0.02,
    },
}


TEMPORAL_POSE_SOURCES = frozenset({"vggt", "gt"})


def temporal_pose_source_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> str:
    """Resolve the explicit temporal pose source for transport/conditioning.

    ``data.temporal_pose_source`` is canonical.  ``model.pose_source`` is
    accepted as a compatibility alias for experiment configs that group all
    pose switches under ``model``.  If both are present they must agree.
    Missing values retain the historical ``vggt`` behavior.
    """

    data_section = config.get("data")
    model_section = config.get("model")
    data_value = (
        data_section.get("temporal_pose_source")
        if isinstance(data_section, Mapping)
        else None
    )
    model_value = (
        model_section.get("pose_source")
        if isinstance(model_section, Mapping)
        else None
    )
    values = [value for value in (data_value, model_value) if value is not None]
    if len(values) > 1 and str(values[0]).lower() != str(values[1]).lower():
        raise ValueError(
            "data.temporal_pose_source and model.pose_source disagree: "
            f"{values[0]!r} vs {values[1]!r}"
        )
    source = str(values[0] if values else "vggt").strip().lower()
    aliases = {"ground_truth": "gt", "ground-truth": "gt", "manifest": "gt"}
    source = aliases.get(source, source)
    if source not in TEMPORAL_POSE_SOURCES:
        raise ValueError(
            "temporal pose source must be one of "
            f"{sorted(TEMPORAL_POSE_SOURCES)}, got {source!r}"
        )
    return source


def temporal_pose_inputs_from_batch(
    batch: Mapping[str, Any],
    config: Mapping[str, Any] | DictConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(poses, valid, quality)`` for the configured pose source.

    Both pose sources use the same ``[B,T,10,3,4]`` view contract, so callers
    can pass the returned tensors directly to the existing transport and
    calibration helpers.  GT poses are trusted only when the optional field is
    present; their quality score is one for valid entries.
    """

    source = "vggt" if config is None else temporal_pose_source_from_config(config)
    if source == "gt":
        poses = batch.get("gt_pose_sequence")
        if poses is None:
            poses = batch.get("gt_extrinsics_camera_from_world_sequence")
        if not isinstance(poses, Tensor):
            raise ValueError(
                "temporal_pose_source='gt' requires "
                "gt_extrinsics_camera_from_world_sequence"
            )
        valid = batch.get("gt_temporal_pose_valid_sequence")
        if valid is None:
            if poses.ndim != 5:
                raise ValueError(
                    "GT temporal poses must have shape [B,T,10,3,4]"
                )
            valid = torch.ones(
                poses.shape[:2], device=poses.device, dtype=torch.bool
            )
        quality = batch.get("gt_temporal_pose_quality_score_sequence")
        if quality is None:
            quality = torch.ones(
                valid.shape, device=poses.device, dtype=torch.float32
            )
    else:
        poses = batch.get("vggt_extrinsics_camera_from_world_metric_sequence")
        valid = batch.get("temporal_pose_valid_sequence")
        quality = batch.get("temporal_pose_quality_score_sequence")
        if not isinstance(poses, Tensor) or not isinstance(valid, Tensor):
            raise ValueError("VGGT temporal pose fields are missing from batch")
        if quality is None:
            quality = torch.where(
                valid,
                torch.ones(valid.shape, device=poses.device, dtype=torch.float32),
                torch.zeros(valid.shape, device=poses.device, dtype=torch.float32),
            )
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        raise ValueError("temporal pose validity must be a bool Tensor")
    if not isinstance(quality, Tensor) or not quality.is_floating_point():
        raise ValueError("temporal pose quality must be a floating Tensor")
    if poses.ndim != 5 or tuple(poses.shape[-3:]) != (10, 3, 4):
        raise ValueError(
            "temporal poses must have shape [B,T,10,3,4], got "
            f"{tuple(poses.shape)}"
        )
    if tuple(valid.shape) != tuple(poses.shape[:2]) or tuple(quality.shape) != tuple(
        poses.shape[:2]
    ):
        raise ValueError("temporal pose masks/quality must have shape [B,T]")
    if poses.device != valid.device or poses.device != quality.device:
        raise ValueError("temporal pose fields must share a device")
    if not bool(torch.isfinite(poses).all().item()):
        raise ValueError("temporal poses contain NaN or infinity")
    return poses, valid, quality


@dataclass(frozen=True, slots=True)
class PositivityAblation:
    """Explicit D-025-only controls, absent from all baseline configs.

    The default is intentionally not represented in ``DEFAULT_CONFIG``.  That
    keeps resolved baseline configs byte-for-byte identical for exact
    checkpoint resume; only the dedicated ablation YAML opts in.
    """

    enabled: bool = False
    sanitize_invalid_sources: bool = False
    lower_bound_hr_px: float | None = None
    lr_negative_penalty_weight: float = 0.0
    raw_negative_penalty_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class PhysicalOutputV2:
    """Opt-in explicit valid/completion and non-negative output contract."""

    enabled: bool = False
    valid_threshold: float = 0.5
    completion_threshold: float = 0.5
    # Compatibility default is intentionally ``False``: existing V2
    # checkpoints use ``>=`` at the hard decision boundary.  New corrected
    # runs can opt into strict ``>`` semantics to avoid a p==0.5 tie.
    strict_threshold: bool = False
    trusted_ffs_confidence_threshold: float = 0.8
    valid_bce_weight: float = 0.0
    completion_bce_weight: float = 0.0
    calibration_weight: float = 0.0


TEMPORAL_HISTORY_V2_PROTOCOL = "topk_z_aware_hidden_warp_v2"
TEMPORAL_RESIDUAL_V2_PROTOCOL = "teacher_gt_temporal_residual_v2"
CALIBRATION_CONDITIONING_V3_PROTOCOL = "dense_rays_factorized_pose_v3"
ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT = (
    "align_corners_false_half_pixel_v3_1"
)
MEASUREMENT_OWNERSHIP_V31_PROTOCOL = (
    "lr_center_projection_bounded_subpixel_v3_1"
)
TEMPORAL_CANDIDATE_FUSION_V31_PROTOCOL = (
    "current_conditioned_age_phase_diverse_v3_1"
)


@dataclass(frozen=True, slots=True)
class TemporalHistoryV2:
    """Opt-in multi-age top-K history and hidden-state warp contract."""

    enabled: bool = False
    top_k: int = 1
    memory_frames: int = 1
    splat_footprint: str = "nearest"
    depth_temperature_m: float = 0.25
    age_temperature_frames: float = 3.0
    source_collision_penalty: float = 0.5
    candidate_feature_channels: int = 32
    collision_depth_gap_m: float = 0.05
    collision_relative_depth_gap: float = 0.05


@dataclass(frozen=True, slots=True)
class TemporalResidualV2:
    """Opt-in teacher/GT temporal-residual supervision contract."""

    enabled: bool = False
    reference: str = "teacher"


@dataclass(frozen=True, slots=True)
class CalibrationConditioningV3:
    """Opt-in calibrated-ray and factorized-pose model input contract."""

    enabled: bool = False
    use_rays: bool = False
    use_stereo_pose: bool = False
    use_temporal_pose: bool = False
    align_corners_false_pixel_centers: bool = False


@dataclass(frozen=True, slots=True)
class MeasurementOwnershipV31:
    """Opt-in LR-observation-domain FFS ownership contract."""

    enabled: bool = False
    minimum_subpixel_residual_hr_px: float = 1.0
    maximum_subpixel_residual_hr_px: float = 8.0
    boundary_relative_scale: float = 0.10


@dataclass(frozen=True, slots=True)
class TemporalCandidateFusionV31:
    """Age/phase-diverse transport plus current-conditioned candidate fusion."""

    enabled: bool = False
    per_age_quota: int = 2
    surface_depth_gap_m: float = 0.05
    surface_relative_depth_gap: float = 0.05
    phase_redundancy_sigma_grid_px: float = 0.125
    phase_redundancy_penalty: float = 0.25


def measurement_ownership_v3_1_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> MeasurementOwnershipV31:
    section = config.get("measurement_ownership_v3_1")
    if section is None:
        return MeasurementOwnershipV31()
    if not isinstance(section, Mapping):
        raise ValueError("measurement_ownership_v3_1 must be a mapping")
    enabled = section.get("enabled")
    if enabled is False:
        return MeasurementOwnershipV31()
    if enabled is not True:
        raise ValueError("measurement_ownership_v3_1.enabled must be a bool")
    if section.get("protocol_version") != MEASUREMENT_OWNERSHIP_V31_PROTOCOL:
        raise ValueError("measurement_ownership_v3_1 protocol version mismatch")

    def finite(name: str, *, minimum: float, strict: bool) -> float:
        value = section.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (float(value) <= minimum if strict else float(value) < minimum)
        ):
            comparison = ">" if strict else ">="
            raise ValueError(
                f"measurement_ownership_v3_1.{name} must be finite and "
                f"{comparison} {minimum}"
            )
        return float(value)

    minimum = finite(
        "minimum_subpixel_residual_hr_px", minimum=0.0, strict=False
    )
    maximum = finite(
        "maximum_subpixel_residual_hr_px", minimum=0.0, strict=False
    )
    if maximum < minimum:
        raise ValueError(
            "measurement_ownership_v3_1 maximum residual must be >= minimum"
        )
    return MeasurementOwnershipV31(
        enabled=True,
        minimum_subpixel_residual_hr_px=minimum,
        maximum_subpixel_residual_hr_px=maximum,
        boundary_relative_scale=finite(
            "boundary_relative_scale", minimum=0.0, strict=True
        ),
    )


def temporal_candidate_fusion_v3_1_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> TemporalCandidateFusionV31:
    section = config.get("temporal_candidate_fusion_v3_1")
    if section is None:
        return TemporalCandidateFusionV31()
    if not isinstance(section, Mapping):
        raise ValueError("temporal_candidate_fusion_v3_1 must be a mapping")
    enabled = section.get("enabled")
    if enabled is False:
        return TemporalCandidateFusionV31()
    if enabled is not True:
        raise ValueError("temporal_candidate_fusion_v3_1.enabled must be a bool")
    if section.get("protocol_version") != TEMPORAL_CANDIDATE_FUSION_V31_PROTOCOL:
        raise ValueError("temporal_candidate_fusion_v3_1 protocol version mismatch")
    per_age_quota = section.get("per_age_quota")
    if (
        isinstance(per_age_quota, bool)
        or not isinstance(per_age_quota, int)
        or per_age_quota <= 0
    ):
        raise ValueError(
            "temporal_candidate_fusion_v3_1.per_age_quota must be positive"
        )

    def positive(name: str) -> float:
        value = section.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"temporal_candidate_fusion_v3_1.{name} must be finite and > 0"
            )
        return float(value)

    phase_penalty = section.get("phase_redundancy_penalty")
    if (
        isinstance(phase_penalty, bool)
        or not isinstance(phase_penalty, (int, float))
        or not math.isfinite(float(phase_penalty))
        or float(phase_penalty) < 0
    ):
        raise ValueError(
            "temporal_candidate_fusion_v3_1.phase_redundancy_penalty must be "
            "finite and >= 0"
        )
    return TemporalCandidateFusionV31(
        enabled=True,
        per_age_quota=per_age_quota,
        surface_depth_gap_m=positive("surface_depth_gap_m"),
        surface_relative_depth_gap=positive("surface_relative_depth_gap"),
        phase_redundancy_sigma_grid_px=positive(
            "phase_redundancy_sigma_grid_px"
        ),
        phase_redundancy_penalty=float(phase_penalty),
    )


def calibration_conditioning_v3_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> CalibrationConditioningV3:
    section = config.get("calibration_conditioning_v3")
    if section is None:
        return CalibrationConditioningV3()
    if not isinstance(section, Mapping):
        raise ValueError("calibration_conditioning_v3 must be a mapping")
    pixel_center_contract = section.get("pixel_center_contract")
    if (
        pixel_center_contract is not None
        and pixel_center_contract != ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT
    ):
        raise ValueError(
            "calibration_conditioning_v3.pixel_center_contract must be absent "
            f"for legacy behavior or {ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT!r}"
        )
    enabled = section.get("enabled")
    if enabled is False:
        return CalibrationConditioningV3()
    if enabled is not True:
        raise ValueError("calibration_conditioning_v3.enabled must be a bool")
    if section.get("protocol_version") != CALIBRATION_CONDITIONING_V3_PROTOCOL:
        raise ValueError("calibration_conditioning_v3 protocol version mismatch")
    switches: dict[str, bool] = {}
    for name in ("use_rays", "use_stereo_pose", "use_temporal_pose"):
        value = section.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"calibration_conditioning_v3.{name} must be a bool")
        switches[name] = value
    if pixel_center_contract is None:
        corrected_pixel_centers = False
    elif pixel_center_contract == ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT:
        corrected_pixel_centers = True
    else:  # pragma: no cover - guarded before the enabled branch.
        raise AssertionError("unreachable pixel-center contract")
    return CalibrationConditioningV3(
        enabled=True,
        align_corners_false_pixel_centers=corrected_pixel_centers,
        **switches,
    )


def temporal_history_v2_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> TemporalHistoryV2:
    """Parse top-K transport without changing resolved legacy configs."""

    section = config.get("temporal_history_v2")
    if section is None:
        return TemporalHistoryV2()
    if not isinstance(section, Mapping):
        raise ValueError("temporal_history_v2 must be a mapping")
    enabled = section.get("enabled")
    if enabled is False:
        return TemporalHistoryV2()
    if enabled is not True:
        raise ValueError("temporal_history_v2.enabled must be a bool")
    if section.get("protocol_version") != TEMPORAL_HISTORY_V2_PROTOCOL:
        raise ValueError("temporal_history_v2 protocol version mismatch")

    def positive_integer(name: str) -> int:
        value = section.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"temporal_history_v2.{name} must be a positive integer")
        return value

    def positive_float(name: str) -> float:
        value = section.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"temporal_history_v2.{name} must be finite and > 0")
        return float(value)

    top_k = positive_integer("top_k")
    memory_frames = positive_integer("memory_frames")
    candidate_channels = positive_integer("candidate_feature_channels")
    if memory_frames > 2:
        raise ValueError("causal T=3 supports at most two history memory frames")
    footprint = section.get("splat_footprint")
    if footprint not in {"bilinear", "nearest"}:
        raise ValueError(
            "temporal_history_v2.splat_footprint must be bilinear or nearest"
        )
    collision_penalty = section.get("source_collision_penalty")
    if (
        isinstance(collision_penalty, bool)
        or not isinstance(collision_penalty, (int, float))
        or not math.isfinite(float(collision_penalty))
        or not 0.0 <= float(collision_penalty) <= 1.0
    ):
        raise ValueError(
            "temporal_history_v2.source_collision_penalty must be in [0,1]"
        )
    return TemporalHistoryV2(
        enabled=True,
        top_k=top_k,
        memory_frames=memory_frames,
        splat_footprint=str(footprint),
        depth_temperature_m=positive_float("depth_temperature_m"),
        age_temperature_frames=positive_float("age_temperature_frames"),
        source_collision_penalty=float(collision_penalty),
        candidate_feature_channels=candidate_channels,
        collision_depth_gap_m=positive_float("collision_depth_gap_m"),
        collision_relative_depth_gap=positive_float(
            "collision_relative_depth_gap"
        ),
    )


def temporal_residual_v2_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> TemporalResidualV2:
    """Parse teacher/GT residual supervision while retaining legacy TEPE."""

    section = config.get("temporal_residual_v2")
    if section is None:
        return TemporalResidualV2()
    if not isinstance(section, Mapping):
        raise ValueError("temporal_residual_v2 must be a mapping")
    enabled = section.get("enabled")
    if enabled is False:
        return TemporalResidualV2()
    if enabled is not True:
        raise ValueError("temporal_residual_v2.enabled must be a bool")
    if section.get("protocol_version") != TEMPORAL_RESIDUAL_V2_PROTOCOL:
        raise ValueError("temporal_residual_v2 protocol version mismatch")
    reference = section.get("reference")
    if reference not in {"teacher", "gt"}:
        raise ValueError("temporal_residual_v2.reference must be teacher or gt")
    # The current cache dataset exposes the teacher path.  Reserving ``gt`` in
    # the schema makes the metric definition stable, but fail closed until a
    # real-GT sequence field is present end to end.
    if reference == "gt":
        raise ValueError("temporal_residual_v2 reference=gt is not yet cache-backed")
    return TemporalResidualV2(enabled=True, reference=str(reference))


def physical_output_v2_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> PhysicalOutputV2:
    """Parse V2 without adding keys to resolved canonical configurations."""

    section = config.get("physical_output_v2")
    if section is None:
        return PhysicalOutputV2()
    if not isinstance(section, Mapping):
        raise ValueError("physical_output_v2 must be a mapping")
    if section.get("enabled") is not True:
        if section.get("enabled") is False:
            return PhysicalOutputV2()
        raise ValueError("physical_output_v2.enabled must be a bool")
    if section.get("protocol_version") != "explicit_valid_completion_nonnegative_v2":
        raise ValueError("physical_output_v2 protocol version mismatch")
    if positivity_ablation_from_config(config).enabled:
        raise ValueError(
            "physical_output_v2 and legacy positivity_ablation are separate lineages"
        )

    def probability(name: str) -> float:
        value = section.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"physical_output_v2.{name} must be in (0,1)")
        return float(value)

    strict_threshold = section.get("strict_threshold", False)
    if not isinstance(strict_threshold, bool):
        raise ValueError("physical_output_v2.strict_threshold must be a bool")

    def positive_weight(name: str) -> float:
        value = section.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"physical_output_v2.{name} must be positive")
        return float(value)

    return PhysicalOutputV2(
        enabled=True,
        valid_threshold=probability("valid_threshold"),
        completion_threshold=probability("completion_threshold"),
        strict_threshold=strict_threshold,
        trusted_ffs_confidence_threshold=probability(
            "trusted_ffs_confidence_threshold"
        ),
        valid_bce_weight=positive_weight("valid_bce_weight"),
        completion_bce_weight=positive_weight("completion_bce_weight"),
        calibration_weight=positive_weight("calibration_weight"),
    )


def convex_initialization_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> str:
    """Return the opt-in convex-mask initialization mode.

    The key is deliberately optional and is not added to ``DEFAULT_CONFIG``;
    this preserves the resolved configuration/fingerprint of all legacy runs.
    ``convex_init`` is accepted as a short compatibility alias for local
    ablation files, while ``convex_initialization`` is canonical.
    """

    model = config.get("model")
    if model is None:
        return "uniform"
    value = model.get("convex_initialization")
    alias = model.get("convex_init")
    if value is not None and alias is not None and str(value).strip().lower() != str(alias).strip().lower():
        raise ValueError(
            "model.convex_initialization and model.convex_init disagree"
        )
    if value is None:
        value = alias
    if value is None:
        return "uniform"
    if not isinstance(value, str):
        raise ValueError(
            "model.convex_initialization must be 'uniform' or 'bilinear'"
        )
    normalized = value.strip().lower()
    if normalized not in {"uniform", "bilinear"}:
        raise ValueError(
            "model.convex_initialization must be 'uniform' or 'bilinear'"
        )
    return normalized


def positivity_ablation_from_config(
    config: Mapping[str, Any] | DictConfig,
) -> PositivityAblation:
    """Parse the opt-in D-025 physical positivity ablation strictly."""

    section = config.get("positivity_ablation")
    if section is None:
        return PositivityAblation()
    if not isinstance(section, Mapping):
        raise ValueError("positivity_ablation must be a mapping")

    def required(name: str) -> Any:
        if name not in section:
            raise ValueError(f"positivity_ablation.{name} is required")
        return section[name]

    enabled = required("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("positivity_ablation.enabled must be a bool")
    if not enabled:
        return PositivityAblation()

    sanitize_invalid_sources = required("sanitize_invalid_sources")
    if sanitize_invalid_sources is not True:
        raise ValueError(
            "D-025 positivity ablation requires sanitize_invalid_sources=true"
        )
    lower_bound_hr_px = float(required("lower_bound_hr_px"))
    if not math.isfinite(lower_bound_hr_px) or lower_bound_hr_px != 0.0:
        raise ValueError(
            "D-025 lower_bound_hr_px must be exactly 0.0; epsilon fills are forbidden"
        )
    lr_weight = float(required("lr_negative_penalty_weight"))
    raw_weight = float(required("raw_negative_penalty_weight"))
    if (
        not math.isfinite(lr_weight)
        or not math.isfinite(raw_weight)
        or lr_weight < 0
        or raw_weight < 0
        or (lr_weight == 0 and raw_weight == 0)
    ):
        raise ValueError("D-025 negative-penalty weights must be finite and at least one positive")
    return PositivityAblation(
        enabled=True,
        sanitize_invalid_sources=True,
        lower_bound_hr_px=lower_bound_hr_px,
        lr_negative_penalty_weight=lr_weight,
        raw_negative_penalty_weight=raw_weight,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage A (T=1) or Stage B (causal T=3) FFS-Omega-TSR."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--manifest", type=Path, help="explicit JSONL manifest")
    parser.add_argument(
        "--observation-cache-root",
        type=Path,
        help="directory containing observation run_receipt.json",
    )
    parser.add_argument(
        "--teacher-cache-root",
        type=Path,
        help="directory containing teacher run_receipt.json",
    )
    parser.add_argument(
        "--derived-cache-root",
        type=Path,
        help="Stage-B vggt-ffs-derived-geometry cache directory",
    )
    parser.add_argument(
        "--calibration-sidecar",
        type=Path,
        help="manifest-bound rectified stereo calibration JSONL sidecar",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Stage-A checkpoint used to initialize a new Stage-B run",
    )
    parser.add_argument("--output-dir", type=Path, help="training output directory")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume exactly")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit CUDA device"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build data/model and evaluate one batch loss without an optimizer step",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.steps_spatial=2",
    )
    return parser


def _load_yaml_with_defaults(path: Path, seen: set[Path] | None = None) -> DictConfig:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"config does not exist: {resolved_path}")
    seen = set() if seen is None else seen
    if resolved_path in seen:
        raise ValueError(f"cyclic defaults_from chain at {resolved_path}")
    seen.add(resolved_path)
    loaded = OmegaConf.load(resolved_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"config must resolve to a mapping: {resolved_path}")
    inherited_path = loaded.get("defaults_from")
    if inherited_path is None:
        return loaded
    del loaded["defaults_from"]
    candidate = Path(str(inherited_path)).expanduser()
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        candidate = project_candidate if project_candidate.exists() else resolved_path.parent / candidate
    inherited = _load_yaml_with_defaults(candidate, seen)
    return OmegaConf.merge(inherited, loaded)


def resolve_config(config_path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Merge checked defaults, YAML inheritance, and struct-checked dotlist values."""

    defaults = OmegaConf.create(DEFAULT_CONFIG)
    loaded = _load_yaml_with_defaults(Path(config_path))
    config = OmegaConf.merge(defaults, loaded)
    OmegaConf.set_struct(config, True)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def loss_weights_from_config(config: Mapping[str, Any] | DictConfig) -> LossWeights:
    section = config["loss"]
    weights = LossWeights(
        disparity=float(section["disparity"]),
        measurement=float(section["measurement"]),
        gradient=float(section["gradient"]),
        temporal=float(section["temporal"]),
        epipolar=float(section["epipolar"]),
        uncertainty_nll=float(section["uncertainty_nll"]),
        gate_regularizer=float(section["gate_regularizer"]),
    )
    if weights != LossWeights():
        raise ValueError(
            f"Stage A requires the declared loss weights {LossWeights()}, got {weights}"
        )
    return weights


def _validate_common_training_config(config: DictConfig, *, total_steps: int) -> None:
    temporal_pose_source_from_config(config)
    physical_v2 = physical_output_v2_from_config(config)
    temporal_history = temporal_history_v2_from_config(config)
    temporal_residual = temporal_residual_v2_from_config(config)
    calibration_v3 = calibration_conditioning_v3_from_config(config)
    measurement_v31 = measurement_ownership_v3_1_from_config(config)
    candidate_v31 = temporal_candidate_fusion_v3_1_from_config(config)
    if temporal_history.enabled != temporal_residual.enabled:
        raise ValueError(
            "temporal_history_v2 and temporal_residual_v2 must be enabled together"
        )
    if temporal_history.enabled:
        if temporal_history.top_k < 2:
            raise ValueError("temporal_history_v2 must retain at least K=2 candidates")
        if temporal_history.candidate_feature_channels != 32:
            raise ValueError("the V2 top-K candidate feature width is fixed to 32")
    derived_contract = str(config.data.derived_contract)
    if calibration_v3.enabled:
        if not physical_v2.enabled or not temporal_history.enabled:
            raise ValueError("calibration v3 must extend the complete architecture-v2 lineage")
        if derived_contract != "calibrated_stereo_v2":
            raise ValueError("calibration v3 requires data.derived_contract=calibrated_stereo_v2")
        sidecar = config.data.calibration_sidecar_path
        if sidecar is None or not str(sidecar).strip():
            raise ValueError("calibration v3 requires data.calibration_sidecar_path")
    else:
        if derived_contract != "legacy_v1":
            raise ValueError("legacy/v2 configs require data.derived_contract=legacy_v1")
    if measurement_v31.enabled:
        if not physical_v2.enabled or not calibration_v3.enabled:
            raise ValueError(
                "measurement ownership v3.1 requires physical output v2 and "
                "calibration v3"
            )
        if not calibration_v3.align_corners_false_pixel_centers:
            raise ValueError(
                "measurement ownership v3.1 requires the corrected half-pixel contract"
            )
    if candidate_v31.enabled:
        if not calibration_v3.enabled or not temporal_history.enabled:
            raise ValueError(
                "temporal candidate fusion v3.1 requires calibration v3 and "
                "temporal history v2"
            )
        if not calibration_v3.align_corners_false_pixel_centers:
            raise ValueError(
                "temporal candidate fusion v3.1 requires the corrected "
                "half-pixel contract"
            )
        if temporal_history.top_k < 4 or temporal_history.memory_frames != 2:
            raise ValueError(
                "temporal candidate fusion v3.1 requires top_k>=4 and two ages"
            )
        if candidate_v31.per_age_quota > temporal_history.top_k // 2:
            raise ValueError(
                "v3.1 per-age quota must not exceed floor(top_k/2)"
            )
    if int(config.data.scale) != 2 or int(config.model.convex_scale) != 2:
        raise ValueError("the first-round training pipeline is fixed to x2")
    if list(config.model.rgb_channels) != [32, 64, 96]:
        raise ValueError("model.rgb_channels must be [32,64,96]")
    if str(config.train.optimizer).lower() != "adamw":
        raise ValueError("optimizer must be AdamW")
    if str(config.train.precision).lower() != "bf16":
        raise ValueError("training precision must be bf16")
    if bool(config.train.compile_model):
        raise ValueError("torch.compile is disabled for the first-round baseline")
    micro_batch = _positive_int(config.train.micro_batch_size, "micro_batch_size")
    accumulation = _positive_int(config.train.grad_accumulation, "grad_accumulation")
    effective = _positive_int(config.train.effective_batch_size, "effective_batch_size")
    if micro_batch * accumulation != effective:
        raise ValueError(
            "effective_batch_size must equal micro_batch_size * grad_accumulation"
        )
    _positive_int(total_steps, "total training steps")
    _nonnegative_int(config.train.warmup_steps, "warmup_steps")
    _positive_int(config.train.checkpoint_interval, "checkpoint_interval")
    _positive_int(config.train.log_interval, "log_interval")
    _nonnegative_int(config.train.num_workers, "num_workers")
    if not math.isfinite(float(config.train.learning_rate)) or float(
        config.train.learning_rate
    ) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(config.train.weight_decay)) or float(
        config.train.weight_decay
    ) < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(float(config.train.gradient_clip)) or float(
        config.train.gradient_clip
    ) <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    crop = list(config.data.hr_crop)
    if len(crop) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in crop
    ):
        raise ValueError("data.hr_crop must be [height,width] positive integers")
    if any(value % 2 for value in crop):
        raise ValueError("data.hr_crop dimensions must be divisible by x2")
    if str(config.data.crop_mode) not in {"random", "fixed"}:
        raise ValueError("data.crop_mode must be random or fixed")
    _nonnegative_int(
        config.train.finite_diagnostic_interval, "finite_diagnostic_interval"
    )
    loss_weights_from_config(config)


def validate_stage_a_config(config: DictConfig) -> None:
    """Reject settings that violate the fixed Stage-A experiment contract."""

    if int(config.data.sequence_length) != 1:
        raise ValueError("Stage A is T=1: data.sequence_length must equal 1")
    if positivity_ablation_from_config(config).enabled:
        raise ValueError("D-025 positivity ablation is restricted to a separate Stage-B run")
    _validate_common_training_config(
        config, total_steps=int(config.train.steps_spatial)
    )


def validate_stage_b_config(config: DictConfig) -> None:
    """Reject non-causal or incomplete Stage-B temporal configurations."""

    if int(config.data.sequence_length) != 3:
        raise ValueError("Stage B is causal T=3: data.sequence_length must equal 3")
    if int(config.data.vggt_context_pairs) != 5:
        raise ValueError("Stage B requires five causal VGGT stereo pairs")
    if not bool(config.vggt.causal):
        raise ValueError("Stage B forbids non-causal VGGT context")
    if not bool(config.model.use_history):
        raise ValueError("Stage B requires model.use_history")
    pose_source = temporal_pose_source_from_config(config)
    if pose_source == "vggt" and not bool(config.model.use_vggt_pose):
        raise ValueError(
            "Stage B with temporal_pose_source='vggt' requires "
            "model.use_vggt_pose"
        )
    if bool(config.model.epipolar_refinement):
        raise ValueError("HR epipolar refinement belongs to Stage C")
    if str(config.train.init_from_stage).lower() != "spatial":
        raise ValueError("Stage B must initialize from the spatial stage")
    if not bool(config.train.history_detach):
        raise ValueError("Stage B MVP requires detached disparity history")
    temporal_history = temporal_history_v2_from_config(config)
    if temporal_history.enabled and temporal_history.memory_frames != 2:
        raise ValueError("causal T=3 V2 requires exactly two history memory frames")
    # Parsing here makes a malformed ablation config fail before cache/model
    # construction, while an absent section leaves the formal contract alone.
    positivity_ablation_from_config(config)
    for name in (
        "photometric_temperature",
        "disparity_temperature_hr_px",
        "history_conflict_hr_px",
        "temporal_photometric_threshold",
        "temporal_geometry_threshold_hr_px",
    ):
        value = float(config.train[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{name} must be finite and positive")
    _validate_common_training_config(config, total_steps=int(config.train.steps))


def training_stage(config: DictConfig) -> str:
    stage = str(config.train.stage).lower()
    if stage not in {"spatial", "temporal"}:
        raise ValueError(f"train.stage must be spatial or temporal, got {stage!r}")
    return stage


def validate_training_config(config: DictConfig) -> str:
    stage = training_stage(config)
    if stage == "spatial":
        validate_stage_a_config(config)
    else:
        validate_stage_b_config(config)
    return stage


def learning_rate_multiplier(
    update_index: int, *, total_steps: int, warmup_steps: int
) -> float:
    """Linear warmup then cosine decay for a zero-based optimizer update."""

    update_index = _nonnegative_int(update_index, "update_index")
    total_steps = _positive_int(total_steps, "total_steps")
    warmup_steps = _nonnegative_int(warmup_steps, "warmup_steps")
    if warmup_steps and update_index < warmup_steps:
        return float(update_index + 1) / float(warmup_steps)
    decay_updates = total_steps - warmup_steps
    if decay_updates <= 1:
        return 1.0
    progress = min(max(update_index - warmup_steps, 0), decay_updates - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress / (decay_updates - 1)))


def should_optimizer_step(micro_step: int, grad_accumulation: int) -> bool:
    """Return whether a one-based micro-step completes an accumulation group."""

    micro_step = _positive_int(micro_step, "micro_step")
    grad_accumulation = _positive_int(grad_accumulation, "grad_accumulation")
    return micro_step % grad_accumulation == 0


class DeterministicEpochSampler(Sampler[int]):
    """Stateless epoch-keyed random permutation for exact resume."""

    def __init__(self, dataset_size: int, *, seed: int) -> None:
        self.dataset_size = _positive_int(dataset_size, "dataset_size")
        self.seed = _nonnegative_int(seed, "sampler seed")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _nonnegative_int(epoch, "sampler epoch")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        # Use a large odd increment and stay inside Torch's signed seed range.
        epoch_seed = (self.seed + self.epoch * 0x1E35A7BD) % (2**63 - 1)
        generator.manual_seed(epoch_seed)
        return iter(torch.randperm(self.dataset_size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.dataset_size


def _identity_from_mapping(value: Mapping[str, Any], receipt_path: Path) -> CacheIdentity:
    expected_fields = {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"cache identity fields in {receipt_path} must be exactly "
            f"{sorted(expected_fields)}, got {sorted(value)}"
        )
    return CacheIdentity(
        component=str(value["component"]),
        upstream_commit=str(value["upstream_commit"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        torch_version=str(value["torch_version"]),
        cuda_version=(
            None if value["cuda_version"] is None else str(value["cuda_version"])
        ),
        config_sha256=str(value["config_sha256"]),
    )


def load_receipt_identity(
    cache_root: str | Path,
    *,
    expected_component: str,
    manifest_path: str | Path,
) -> CacheIdentity:
    """Load the canonical cache receipt and bind it to the active manifest."""

    root = Path(cache_root).expanduser().resolve()
    receipt_path = root / "run_receipt.json"
    if not root.is_dir():
        raise FileNotFoundError(f"cache root does not exist: {root}")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"canonical cache receipt does not exist: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cache receipt {receipt_path}: {exc}") from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != 1:
        raise ValueError(f"unsupported cache receipt schema: {receipt_path}")
    identity_mapping = receipt.get("identity")
    if not isinstance(identity_mapping, Mapping):
        raise ValueError(f"cache receipt identity is missing: {receipt_path}")
    identity = _identity_from_mapping(identity_mapping, receipt_path)
    if identity.component != expected_component:
        raise ValueError(
            f"cache receipt component mismatch: expected {expected_component!r}, "
            f"got {identity.component!r} in {receipt_path}"
        )
    manifest = Path(manifest_path).expanduser().resolve()
    actual_manifest_sha256 = sha256_file(manifest)
    if receipt.get("manifest_sha256") != actual_manifest_sha256:
        raise ValueError(
            f"cache receipt manifest SHA-256 mismatch in {receipt_path}: expected "
            f"{actual_manifest_sha256}, got {receipt.get('manifest_sha256')!r}"
        )
    selected = receipt.get("selected_records")
    written = receipt.get("written_records")
    reused = receipt.get("reused_records")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (selected, written, reused)):
        raise ValueError(f"cache receipt record counts are malformed: {receipt_path}")
    if selected != written + reused:
        raise ValueError(f"cache receipt record counts are incomplete: {receipt_path}")
    manifest_records = load_manifest(manifest)
    if selected != len(manifest_records):
        raise ValueError(
            f"cache receipt covers {selected} records but manifest has "
            f"{len(manifest_records)}: {receipt_path}"
        )
    return identity


def _require_explicit_path(config: DictConfig, name: str) -> Path:
    value = OmegaConf.select(config, name)
    if value is None or not str(value).strip():
        raise ValueError(
            f"{name} is required; provide it in YAML, as a dotlist override, "
            "or with the corresponding CLI flag"
        )
    return Path(str(value)).expanduser().resolve()


def _resolved_container(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):  # pragma: no cover - defensive
        raise TypeError("resolved config is not a mapping")
    return value


def _calibration_index_from_config(
    config: DictConfig, *, manifest_path: Path
) -> RectifiedCalibrationIndex | None:
    contract = calibration_conditioning_v3_from_config(config)
    if not contract.enabled:
        return None
    sidecar_path = _require_explicit_path(config, "data.calibration_sidecar_path")
    return load_rectified_calibration_sidecar(
        sidecar_path, expected_manifest_path=manifest_path
    )


def build_dataset_and_identities(
    config: DictConfig,
) -> tuple[CachedFFSTrainingDataset, CacheIdentity, CacheIdentity]:
    manifest_path = _require_explicit_path(config, "data.manifest_path")
    observation_root = _require_explicit_path(config, "data.observation_cache_root")
    teacher_root = _require_explicit_path(config, "data.teacher_cache_root")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    observation_identity = load_receipt_identity(
        observation_root,
        expected_component="ffs-observation",
        manifest_path=manifest_path,
    )
    teacher_identity = load_receipt_identity(
        teacher_root,
        expected_component="ffs-teacher",
        manifest_path=manifest_path,
    )
    calibration_index = _calibration_index_from_config(
        config, manifest_path=manifest_path
    )
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    dataset = CachedFFSTrainingDataset(
        manifest_path=manifest_path,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        rectified_calibration_index=calibration_index,
        crop_size_hr_hw=(crop_height, crop_width),
        crop_mode=str(config.data.crop_mode),
        spatial_scale=int(config.data.scale),
        seed=int(config.seed),
    )
    if not dataset:
        raise ValueError("training dataset is empty")
    return dataset, observation_identity, teacher_identity


def build_temporal_dataset_and_identities(
    config: DictConfig,
) -> tuple[CachedTemporalTrainingDataset, CacheIdentity, CacheIdentity]:
    manifest_path = _require_explicit_path(config, "data.manifest_path")
    observation_root = _require_explicit_path(config, "data.observation_cache_root")
    teacher_root = _require_explicit_path(config, "data.teacher_cache_root")
    derived_root = _require_explicit_path(config, "data.derived_geometry_cache_root")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    observation_identity = load_receipt_identity(
        observation_root,
        expected_component="ffs-observation",
        manifest_path=manifest_path,
    )
    teacher_identity = load_receipt_identity(
        teacher_root,
        expected_component="ffs-teacher",
        manifest_path=manifest_path,
    )
    calibration_index = _calibration_index_from_config(
        config, manifest_path=manifest_path
    )
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    dataset = CachedTemporalTrainingDataset(
        manifest_path=manifest_path,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        derived_cache_root=derived_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        rectified_calibration_index=calibration_index,
        derived_contract=str(config.data.derived_contract),
        crop_size_hr_hw=(crop_height, crop_width),
        crop_mode=str(config.data.crop_mode),
        spatial_scale=int(config.data.scale),
        student_sequence_length=int(config.data.sequence_length),
        vggt_context_pairs=int(config.data.vggt_context_pairs),
        seed=int(config.seed),
    )
    if not dataset:
        raise ValueError("temporal training dataset is empty")
    return dataset, observation_identity, teacher_identity


def build_model(config: DictConfig) -> FFSOmegaTSR:
    positivity_ablation = positivity_ablation_from_config(config)
    physical_v2 = physical_output_v2_from_config(config)
    convex_initialization = convex_initialization_from_config(config)
    calibration_v3 = calibration_conditioning_v3_from_config(config)
    temporal_history_v2 = temporal_history_v2_from_config(config)
    measurement_v31 = measurement_ownership_v3_1_from_config(config)
    candidate_v31 = temporal_candidate_fusion_v3_1_from_config(config)
    model = FFSOmegaTSR(
        rgb_channels=tuple(int(value) for value in config.model.rgb_channels),
        geometry_channels=int(config.model.geometry_channels),
        hidden_channels=int(config.model.hidden_dim),
        gru_layers=int(config.model.gru_layers),
        scale=int(config.model.convex_scale),
        residual_limit_hr_px=float(config.model.residual_limit_hr_px),
        sanitize_invalid_source_disparities=(
            positivity_ablation.sanitize_invalid_sources
        ),
        positivity_floor_hr_px=positivity_ablation.lower_bound_hr_px,
        physical_output_v2=physical_v2.enabled,
        physical_valid_threshold=physical_v2.valid_threshold,
        completion_threshold=physical_v2.completion_threshold,
        strict_threshold=physical_v2.strict_threshold,
        trusted_ffs_confidence_threshold=(
            physical_v2.trusted_ffs_confidence_threshold
        ),
        convex_initialization=convex_initialization,
        temporal_history_top_k=(
            temporal_history_v2.top_k if temporal_history_v2.enabled else None
        ),
        temporal_history_feature_channels=(
            temporal_history_v2.candidate_feature_channels
        ),
        calibration_conditioning_v3=calibration_v3.enabled,
        use_rays=calibration_v3.use_rays,
        use_stereo_pose=calibration_v3.use_stereo_pose,
        use_temporal_pose=calibration_v3.use_temporal_pose,
        align_corners_false_pixel_centers=(
            calibration_v3.align_corners_false_pixel_centers
        ),
        measurement_ownership_v3_1=measurement_v31.enabled,
        measurement_minimum_subpixel_residual_hr_px=(
            measurement_v31.minimum_subpixel_residual_hr_px
        ),
        measurement_maximum_subpixel_residual_hr_px=(
            measurement_v31.maximum_subpixel_residual_hr_px
        ),
        measurement_boundary_relative_scale=(
            measurement_v31.boundary_relative_scale
        ),
        current_conditioned_history_v3_1=candidate_v31.enabled,
    )
    parameter_count = count_trainable_parameters(model)
    if parameter_count <= 0 or parameter_count >= 12_000_000:
        raise ValueError(
            f"trainable parameter count must be in (0,12M), got {parameter_count}"
        )
    return model


def compute_stage_a_loss(
    output: ModelOutput,
    batch: Mapping[str, Any],
    *,
    scale: int = 2,
    weights: LossWeights = LossWeights(),
    positivity_ablation: PositivityAblation = PositivityAblation(),
    physical_output_v2: PhysicalOutputV2 = PhysicalOutputV2(),
) -> LossBreakdown:
    """Compute the exact Stage-A objective from teacher and observation masks."""

    target = batch.get("teacher_disparity_hr_px")
    target_confidence = batch.get("teacher_confidence")
    target_trusted = batch.get("teacher_trusted_mask")
    if not all(isinstance(value, Tensor) for value in (target, target_confidence, target_trusted)):
        raise ValueError("Stage A requires teacher disparity/confidence/trusted tensors")
    observation_lr_px = batch.get("observation_disparity_lr_px")
    observation_confidence = batch.get("observation_confidence")
    observation_trusted = batch.get("observation_trusted_mask")
    if not all(
        isinstance(value, Tensor)
        for value in (
            observation_lr_px,
            observation_confidence,
            observation_trusted,
        )
    ):
        raise ValueError("Stage A requires observation disparity/confidence/trusted tensors")

    disparity = disparity_loss(
        output.disparity_hr_px,
        target,
        valid_mask=target_trusted,
        weights=target_confidence,
    )
    gradient = gradient_loss(
        output.disparity_hr_px, target, valid_mask=target_trusted
    )
    uncertainty_nll = laplace_uncertainty_nll(
        output.disparity_hr_px,
        target,
        output.log_variance,
        valid_mask=target_trusted,
    )
    measurement = measurement_consistency_loss(
        output.disparity_hr_px,
        observation_lr_px,
        observation_trusted,
        scale=scale,
        confidence_ffs_lr=observation_confidence,
    )
    gate_regularizer = ffs_gate_regularizer(
        output.source_weights,
        observation_confidence,
        observation_trusted,
    )
    # Stage A has neither temporal transport nor the later HR epipolar head.
    # Tie explicit zero terms to the graph so they remain differentiable.
    differentiable_zero = output.disparity_hr_px.sum() * 0.0
    baseline = combine_loss_terms(
        disparity=disparity,
        measurement=measurement,
        gradient=gradient,
        temporal=differentiable_zero,
        epipolar=differentiable_zero,
        uncertainty_nll=uncertainty_nll,
        gate_regularizer=gate_regularizer,
        weights=weights,
    )
    if positivity_ablation.enabled:
        baseline = _with_positivity_penalty(
            baseline, output, positivity_ablation
        )
    if physical_output_v2.enabled:
        baseline = _with_physical_output_v2_loss(
            baseline, output, batch, physical_output_v2
        )
    return baseline


def _with_physical_output_v2_loss(
    baseline: LossBreakdown,
    output: ModelOutput,
    batch: Mapping[str, Any],
    contract: PhysicalOutputV2,
) -> LossBreakdown:
    """Attach explicit validity/completion supervision to a V2 trajectory."""

    fields = (
        output.valid_logits,
        output.completion_logits,
        output.valid_probability,
        output.completion_probability,
    )
    if not all(isinstance(value, Tensor) for value in fields):
        raise ValueError("physical_output_v2 requires all validity output fields")
    teacher_valid = batch.get("teacher_valid_mask")
    teacher_confidence = batch.get("teacher_confidence")
    observation_valid = batch.get("valid_ffs")
    if not all(
        isinstance(value, Tensor)
        for value in (teacher_valid, teacher_confidence, observation_valid)
    ):
        raise ValueError(
            "physical_output_v2 requires teacher validity/confidence and FFS validity"
        )
    assert output.valid_logits is not None
    observation_valid_hr = functional.interpolate(
        observation_valid.to(dtype=torch.float32),
        size=output.valid_logits.shape[-2:],
        mode="nearest",
    ).to(dtype=torch.bool)
    source_support_hr = functional.interpolate(
        output.source_valid_mask.any(dim=1, keepdim=True).to(dtype=torch.float32),
        size=output.valid_logits.shape[-2:],
        mode="nearest",
    ).to(dtype=torch.bool)
    terms = validity_completion_loss(
        valid_logits=output.valid_logits,
        completion_logits=output.completion_logits,  # type: ignore[arg-type]
        valid_probability=output.valid_probability,  # type: ignore[arg-type]
        completion_probability=output.completion_probability,  # type: ignore[arg-type]
        teacher_valid_mask=teacher_valid,
        teacher_confidence=teacher_confidence,
        observation_valid_mask_hr=observation_valid_hr,
        source_support_mask_hr=source_support_hr,
    )
    additional = (
        contract.valid_bce_weight * terms.valid_bce
        + contract.completion_bce_weight * terms.completion_bce
        + contract.calibration_weight * terms.calibration
    )
    return replace(
        baseline,
        total=baseline.total + additional,
        valid_bce=terms.valid_bce,
        completion_bce=terms.completion_bce,
        validity_calibration=terms.calibration,
    )


def _with_positivity_penalty(
    baseline: LossBreakdown,
    output: ModelOutput,
    positivity_ablation: PositivityAblation,
) -> LossBreakdown:
    """Attach the separately weighted D-025 loss without touching baseline math."""

    if not positivity_ablation.enabled:
        return baseline
    pre_lr = output.disparity_pre_lower_bound_hr_px_lr_grid
    pre_raw = output.disparity_pre_lower_bound_raw_hr_px
    if pre_lr is None or pre_raw is None:
        raise ValueError("positivity ablation requires pre-lower-bound model taps")
    assert positivity_ablation.lower_bound_hr_px is not None
    penalty = (
        positivity_ablation.lr_negative_penalty_weight
        * lower_bound_penalty(pre_lr, lower_bound_hr_px=positivity_ablation.lower_bound_hr_px)
        + positivity_ablation.raw_negative_penalty_weight
        * lower_bound_penalty(pre_raw, lower_bound_hr_px=positivity_ablation.lower_bound_hr_px)
    )
    return replace(
        baseline,
        total=baseline.total + penalty,
        positivity_penalty=penalty,
    )


@dataclass(frozen=True, slots=True)
class TemporalTransport:
    """Detached HR warp for loss plus LR winner samples for model fusion."""

    disparity_history_hr_px: Tensor
    confidence_history: Tensor
    visibility_mask: Tensor
    valid_history: Tensor
    collision_mask: Tensor
    photometric_residual: Tensor
    fractional_offset_px: Tensor
    static_mask: Tensor
    geometry_consistent_mask: Tensor
    disparity_history_loss_hr_px: Tensor
    confidence_history_hr: Tensor
    visibility_mask_hr: Tensor
    valid_history_hr: Tensor
    collision_mask_hr: Tensor
    photometric_residual_hr: Tensor
    static_mask_hr: Tensor
    geometry_consistent_mask_hr: Tensor
    topk_disparity_history_hr_px: Tensor | None = None
    topk_confidence_history: Tensor | None = None
    topk_fractional_offset_px: Tensor | None = None
    topk_temporal_age_frames: Tensor | None = None
    topk_z_aware_weights: Tensor | None = None
    topk_metric_prior_weights: Tensor | None = None
    topk_valid_mask: Tensor | None = None
    topk_depth_m: Tensor | None = None
    topk_pose_quality: Tensor | None = None
    topk_depth_layer_index: Tensor | None = None
    topk_front_surface_mask: Tensor | None = None
    topk_context_only_mask: Tensor | None = None
    topk_age2_depth_consistent_available_mask: Tensor | None = None
    topk_warped_hidden_feature: Tensor | None = None
    warped_hidden_state: tuple[Tensor, ...] | None = None


@dataclass(frozen=True, slots=True)
class TemporalMemoryEntry:
    """One causal model state retained for direct warp into a future frame."""

    output: ModelOutput
    rgb_hr: Tensor
    time_index: int
    intrinsics_hr: Tensor
    baseline_m: Tensor


@dataclass(frozen=True, slots=True)
class ReferenceTemporalWarp:
    """Teacher/GT age-1 warp used only by temporal-residual supervision."""

    disparity_hr_px: Tensor
    prediction_disparity_hr_px: Tensor | None
    valid_mask_hr: Tensor
    visibility_mask_hr: Tensor
    collision_mask_hr: Tensor


def _identity_camera_from_world(
    batch_size: int, *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    value = torch.zeros(batch_size, 3, 4, dtype=dtype, device=device)
    value[:, :3, :3] = torch.eye(3, dtype=dtype, device=device)
    return value


def _rgb_photometric_residual_from_winners(
    previous_rgb_hr: Tensor,
    current_rgb_hr: Tensor,
    warp: WarpResult,
    *,
    valid_mask: Tensor,
) -> Tensor:
    """Compare HR RGB to the exact previous HR z-buffer winner."""

    height_hr, width_hr = warp.disparity_hr_px.shape[-2:]
    expected_hr = (height_hr, width_hr)
    if previous_rgb_hr.shape != current_rgb_hr.shape or tuple(
        previous_rgb_hr.shape[-2:]
    ) != expected_hr:
        raise ValueError("previous/current RGB must share the x2-aligned HR shape")
    previous_hr = previous_rgb_hr.detach()
    current_hr = current_rgb_hr.detach()
    source_u = warp.source_uv[:, 0].round().long().clamp(0, width_hr - 1)
    source_v = warp.source_uv[:, 1].round().long().clamp(0, height_hr - 1)
    source_linear = (source_v * width_hr + source_u).reshape(
        previous_hr.shape[0], 1, -1
    )
    warped_previous = torch.gather(
        previous_hr.reshape(previous_hr.shape[0], 3, -1),
        2,
        source_linear.expand(-1, 3, -1),
    ).reshape_as(current_hr)
    residual = (warped_previous - current_hr).abs().mean(dim=1, keepdim=True)
    return torch.where(
        valid_mask,
        torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(residual),
    )


def _sample_hr_winner_grid_to_lr(
    value: Tensor,
    *,
    scale: int,
    align_corners_false_pixel_centers: bool = False,
) -> Tensor:
    """Sample an HR transport field on the selected LR pixel-centre contract.

    The default is the immutable v1/v2 integer selection ``(s*v,s*u)``.
    Architecture v3 passes ``align_corners_false_pixel_centers=True`` so
    floating fields use the same point operator as the LR observation and
    boolean validity requires the complete bilinear point-sample support.
    Disparity values remain in HR-pixel units.
    """

    if value.ndim != 4 or value.shape[-2] % scale or value.shape[-1] % scale:
        raise ValueError("HR transport tensor must be [B,C,sH,sW]")
    if not isinstance(align_corners_false_pixel_centers, bool):
        raise TypeError("align_corners_false_pixel_centers must be a bool")
    if align_corners_false_pixel_centers:
        if value.dtype == torch.bool:
            support = sample_hr_at_lr_centers(
                value.to(dtype=torch.float32), scale=scale
            )
            # A point-sampled disparity/depth is valid only when every pixel
            # participating in that exact bilinear centre sample is valid.
            # ``nearest-exact`` would arbitrarily choose one of the four x2
            # support pixels and could validate a foreground/background mix.
            return support >= 1.0 - 1e-6
        if not value.is_floating_point():
            raise TypeError("corrected HR-to-LR sampling requires float or bool")
        return sample_hr_at_lr_centers(value, scale=scale).contiguous()
    return value[..., ::scale, ::scale].contiguous()


def build_temporal_transport(
    *,
    previous_output: ModelOutput,
    previous_rgb_hr: Tensor,
    current_rgb_hr: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    intrinsics_current_hr: Tensor,
    baseline_current_m: Tensor,
    temporal_extrinsics_camera_from_world: Tensor,
    temporal_pose_valid: Tensor,
    scale: int = 2,
    photometric_temperature: float = 0.10,
    disparity_temperature_hr_px: float = 2.0,
    reject_conflict_hr_px: float = 2.0,
    photometric_threshold: float = 0.10,
    geometry_threshold_hr_px: float = 2.0,
) -> TemporalTransport:
    """Detach, z-buffer, and gate one predicted frame into its successor."""

    if current_ffs_disparity_hr_px.ndim != 4:
        raise ValueError("current FFS disparity must have shape [B,1,H,W]")
    batch_size = current_ffs_disparity_hr_px.shape[0]
    scalar_shape = current_ffs_disparity_hr_px.shape
    if current_ffs_confidence.shape != scalar_shape:
        raise ValueError("current FFS confidence shape mismatch")
    if intrinsics_current_hr.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics_current_hr must have shape [B,3,3]")
    if baseline_current_m.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("baseline_current_m must have shape [B] or [B,1]")
    if temporal_extrinsics_camera_from_world.shape != (batch_size, 10, 3, 4):
        raise ValueError("temporal extrinsics must have shape [B,10,3,4]")
    pose_valid = temporal_pose_valid.reshape(-1).to(dtype=torch.bool)
    if pose_valid.shape != (batch_size,):
        raise ValueError("temporal_pose_valid must contain one value per batch item")

    previous_disparity_hr = previous_output.disparity_hr_px.detach()
    previous_confidence_hr = torch.exp(
        -0.5 * previous_output.log_variance.detach()
    ).clamp(0.0, 1.0)
    expected_hr_shape = (
        batch_size,
        1,
        scalar_shape[-2] * scale,
        scalar_shape[-1] * scale,
    )
    if previous_disparity_hr.shape != expected_hr_shape:
        raise ValueError(
            f"previous HR disparity must have shape {expected_hr_shape}, got "
            f"{tuple(previous_disparity_hr.shape)}"
        )
    fx_hr = intrinsics_current_hr[:, 0, 0].reshape(batch_size, 1, 1, 1)
    baseline_m = baseline_current_m.reshape(batch_size, 1, 1, 1)
    valid_previous = (
        torch.isfinite(previous_disparity_hr) & (previous_disparity_hr > 0)
    )
    previous_depth_m = torch.where(
        valid_previous,
        fx_hr * baseline_m / previous_disparity_hr.clamp_min(1e-6),
        torch.zeros_like(previous_disparity_hr),
    )

    previous_pose = temporal_extrinsics_camera_from_world[:, 6].detach()
    current_pose = temporal_extrinsics_camera_from_world[:, 8].detach()
    identity = _identity_camera_from_world(
        batch_size, dtype=previous_pose.dtype, device=previous_pose.device
    )
    pose_selector = pose_valid.reshape(batch_size, 1, 1)
    previous_pose = torch.where(pose_selector, previous_pose, identity)
    current_pose = torch.where(pose_selector, current_pose, identity)
    # Geometry and scatter-reduce are a deliberate FP32 island inside BF16
    # model autocast.  In particular, rotation orthonormality checks must not
    # inherit BF16 matrix multiplication from the enclosing training context.
    with torch.autocast(
        device_type=current_ffs_disparity_hr_px.device.type, enabled=False
    ):
        warp = zbuffer_reproject(
            previous_disparity_hr.float(),
            previous_depth_m.float(),
            previous_confidence_hr.float(),
            intrinsics_current_hr.float(),
            previous_pose.float(),
            current_pose.float(),
        )
    pose_mask = pose_valid.reshape(batch_size, 1, 1, 1)
    visibility = warp.visibility_mask & pose_mask
    collision = warp.collision_mask & pose_mask
    warped_disparity = torch.where(
        visibility, warp.disparity_hr_px, torch.zeros_like(warp.disparity_hr_px)
    )
    warped_confidence = torch.where(
        visibility, warp.confidence, torch.zeros_like(warp.confidence)
    )
    photometric_residual = _rgb_photometric_residual_from_winners(
        previous_rgb_hr,
        current_rgb_hr,
        warp,
        valid_mask=visibility,
    )
    current_ffs_disparity_hr = functional.interpolate(
        current_ffs_disparity_hr_px,
        size=expected_hr_shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    current_ffs_confidence_hr = functional.interpolate(
        current_ffs_confidence,
        size=expected_hr_shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    confidence_result = history_confidence(
        warped_confidence,
        visibility,
        collision,
        photometric_residual,
        warped_disparity,
        current_ffs_disparity_hr,
        current_ffs_confidence_hr,
        photometric_temperature=photometric_temperature,
        disparity_temperature_hr_px=disparity_temperature_hr_px,
        reject_conflict_hr_px=reject_conflict_hr_px,
    )
    geometry_consistent = (
        confidence_result.valid_mask
        & torch.isfinite(confidence_result.disparity_error_hr_px)
        & (confidence_result.disparity_error_hr_px <= geometry_threshold_hr_px)
    )
    static_mask = (
        geometry_consistent
        & torch.isfinite(photometric_residual)
        & (photometric_residual <= photometric_threshold)
    )
    effective_valid = confidence_result.valid_mask & pose_mask
    disparity_history_hr = torch.where(
        effective_valid, warped_disparity, torch.zeros_like(warped_disparity)
    ).detach()
    confidence_history_hr = torch.where(
        effective_valid,
        confidence_result.confidence,
        torch.zeros_like(confidence_result.confidence),
    ).detach()
    visibility_hr = (visibility & effective_valid).detach()
    fractional_offset_hr = torch.where(
        effective_valid.expand(-1, 2, -1, -1),
        warp.fractional_offset,
        torch.zeros_like(warp.fractional_offset),
    ).detach()
    return TemporalTransport(
        disparity_history_hr_px=_sample_hr_winner_grid_to_lr(
            disparity_history_hr, scale=scale
        ),
        confidence_history=_sample_hr_winner_grid_to_lr(
            confidence_history_hr, scale=scale
        ),
        visibility_mask=_sample_hr_winner_grid_to_lr(
            visibility_hr, scale=scale
        ),
        valid_history=_sample_hr_winner_grid_to_lr(
            effective_valid.detach(), scale=scale
        ),
        collision_mask=_sample_hr_winner_grid_to_lr(
            collision.detach(), scale=scale
        ),
        photometric_residual=_sample_hr_winner_grid_to_lr(
            photometric_residual.detach(), scale=scale
        ),
        fractional_offset_px=_sample_hr_winner_grid_to_lr(
            fractional_offset_hr, scale=scale
        ),
        static_mask=_sample_hr_winner_grid_to_lr(
            static_mask.detach(), scale=scale
        ),
        geometry_consistent_mask=_sample_hr_winner_grid_to_lr(
            geometry_consistent.detach(), scale=scale
        ),
        disparity_history_loss_hr_px=disparity_history_hr,
        confidence_history_hr=confidence_history_hr,
        visibility_mask_hr=visibility_hr,
        valid_history_hr=effective_valid.detach(),
        collision_mask_hr=collision.detach(),
        photometric_residual_hr=photometric_residual.detach(),
        static_mask_hr=static_mask.detach(),
        geometry_consistent_mask_hr=geometry_consistent.detach(),
    )


def _lr_intrinsics_from_hr(
    intrinsics_hr: Tensor,
    *,
    scale: int,
    align_corners_false_pixel_centers: bool = False,
) -> Tensor:
    """Return LR-grid intrinsics without changing disparity units.

    ``align_corners_false_pixel_centers=False`` is the immutable v1/v2
    ``K_lr=K_hr/scale`` convention.  Architecture v3 passes ``True`` because
    its LR observations are produced by ``F.interpolate(...,
    align_corners=False)``; those pixel centres require
    ``c_lr=(c_hr+0.5)/scale-0.5``.
    """

    if intrinsics_hr.ndim != 3 or intrinsics_hr.shape[-2:] != (3, 3):
        raise ValueError("intrinsics_hr must have shape [B,3,3]")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if not isinstance(align_corners_false_pixel_centers, bool):
        raise TypeError("align_corners_false_pixel_centers must be a bool")
    if align_corners_false_pixel_centers:
        return resize_intrinsics_align_corners_false(
            intrinsics_hr,
            scale_x=1.0 / float(scale),
            scale_y=1.0 / float(scale),
        )
    intrinsics_lr = intrinsics_hr.clone()
    intrinsics_lr[:, 0, :] /= float(scale)
    intrinsics_lr[:, 1, :] /= float(scale)
    intrinsics_lr[:, 2] = intrinsics_hr[:, 2]
    return intrinsics_lr


def validate_v2_temporal_calibration(
    intrinsics_hr_sequence: Tensor,
    baseline_m_sequence: Tensor,
) -> None:
    """Validate explicit per-time source/target calibration for causal V2."""

    if (
        intrinsics_hr_sequence.ndim != 4
        or intrinsics_hr_sequence.shape[1:] != (3, 3, 3)
    ):
        raise ValueError("K_hr_sequence must have shape [B,3,3,3]")
    if baseline_m_sequence.shape != (intrinsics_hr_sequence.shape[0], 3):
        raise ValueError("baseline_m_sequence must have shape [B,3]")
    if not bool(torch.isfinite(intrinsics_hr_sequence).all().item()):
        raise ValueError("K_hr_sequence must contain only finite values")
    if not bool(
        (
            (intrinsics_hr_sequence[:, :, 0, 0] > 0)
            & (intrinsics_hr_sequence[:, :, 1, 1] > 0)
        ).all().item()
    ):
        raise ValueError("K_hr_sequence focal lengths must be positive")
    expected_last_row = intrinsics_hr_sequence.new_tensor((0.0, 0.0, 1.0))
    if not bool(
        torch.isclose(
            intrinsics_hr_sequence[:, :, 2],
            expected_last_row.reshape(1, 1, 3).expand_as(
                intrinsics_hr_sequence[:, :, 2]
            ),
            atol=1e-6,
            rtol=0.0,
        ).all().item()
    ):
        raise ValueError("every K_hr_sequence matrix must end with [0,0,1]")
    if not bool(torch.isfinite(baseline_m_sequence).all().item()) or not bool(
        (baseline_m_sequence > 0).all().item()
    ):
        raise ValueError("baseline_m_sequence must contain finite positive values")


def _metric_depth_from_hr_disparity(
    disparity_hr_px: Tensor,
    *,
    intrinsics_hr: Tensor,
    baseline_m: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert HR-pixel disparity to metric depth and an explicit valid mask."""

    batch_size = disparity_hr_px.shape[0]
    if disparity_hr_px.ndim != 4 or disparity_hr_px.shape[1] != 1:
        raise ValueError("disparity_hr_px must have shape [B,1,H,W]")
    if intrinsics_hr.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics_hr must have shape [B,3,3]")
    if baseline_m.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("baseline_m must have shape [B] or [B,1]")
    fx_hr = intrinsics_hr[:, 0, 0].reshape(batch_size, 1, 1, 1)
    baseline = baseline_m.reshape(batch_size, 1, 1, 1)
    valid = torch.isfinite(disparity_hr_px) & (disparity_hr_px > 0)
    depth_m = torch.where(
        valid,
        fx_hr * baseline / disparity_hr_px.clamp_min(1e-12),
        torch.zeros_like(disparity_hr_px),
    )
    return depth_m, valid


def _vggt_pose_pair_for_age(
    temporal_extrinsics_camera_from_world: Tensor,
    temporal_pose_valid: Tensor,
    *,
    age_frames: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select same-window left-camera poses for a causal history age."""

    if (
        temporal_extrinsics_camera_from_world.ndim != 4
        or temporal_extrinsics_camera_from_world.shape[1:] != (10, 3, 4)
    ):
        raise ValueError("temporal extrinsics must have shape [B,10,3,4]")
    if isinstance(age_frames, bool) or not isinstance(age_frames, int) or not 1 <= age_frames <= 4:
        raise ValueError("age_frames must be an integer in [1,4]")
    batch_size = temporal_extrinsics_camera_from_world.shape[0]
    pose_valid = temporal_pose_valid.reshape(-1).to(dtype=torch.bool)
    if pose_valid.shape != (batch_size,):
        raise ValueError("temporal_pose_valid must contain one value per batch item")
    previous_index = 8 - 2 * age_frames
    previous_pose = temporal_extrinsics_camera_from_world[:, previous_index].detach()
    current_pose = temporal_extrinsics_camera_from_world[:, 8].detach()
    identity = _identity_camera_from_world(
        batch_size, dtype=previous_pose.dtype, device=previous_pose.device
    )
    selector = pose_valid.reshape(batch_size, 1, 1)
    return (
        torch.where(selector, previous_pose, identity),
        torch.where(selector, current_pose, identity),
        pose_valid,
    )


def _topk_splat_for_memory(
    *,
    disparity_hr_px: Tensor,
    confidence: Tensor,
    hidden_feature: Tensor | None,
    source_valid_mask: Tensor,
    intrinsics_previous_grid: Tensor,
    intrinsics_current_grid: Tensor,
    intrinsics_previous_hr: Tensor,
    intrinsics_current_hr: Tensor,
    baseline_previous_m: Tensor,
    baseline_current_m: Tensor,
    previous_pose: Tensor,
    current_pose: Tensor,
    pose_valid: Tensor,
    age_frames: int,
    contract: TemporalHistoryV2,
) -> TopKSplatResult:
    with torch.autocast(device_type=disparity_hr_px.device.type, enabled=False):
        depth_m, positive = _metric_depth_from_hr_disparity(
            disparity_hr_px.float(),
            intrinsics_hr=intrinsics_previous_hr.float(),
            baseline_m=baseline_previous_m.float(),
        )
        source_valid = (
            source_valid_mask.to(dtype=torch.bool)
            & positive
            & pose_valid.reshape(-1, 1, 1, 1)
        )
        return topk_z_aware_splat(
            disparity_hr_px.float(),
            depth_m.float(),
            confidence.float(),
            intrinsics_previous_grid.float(),
            previous_pose.float(),
            current_pose.float(),
            intrinsics_current_grid_3x3=intrinsics_current_grid.float(),
            intrinsics_previous_hr_3x3=intrinsics_previous_hr.float(),
            intrinsics_current_hr_3x3=intrinsics_current_hr.float(),
            baseline_previous_m=baseline_previous_m.float(),
            baseline_current_m=baseline_current_m.float(),
            top_k=contract.top_k,
            temporal_age_frames=age_frames,
            previous_hidden_feature=hidden_feature,
            source_valid_mask=source_valid,
            splat_footprint=contract.splat_footprint,
            depth_temperature_m=contract.depth_temperature_m,
            age_temperature_frames=contract.age_temperature_frames,
            source_collision_penalty=contract.source_collision_penalty,
        )


def _merge_topk_results(
    results: Sequence[TopKSplatResult],
    contract: TemporalHistoryV2,
    candidate_contract: TemporalCandidateFusionV31 = TemporalCandidateFusionV31(),
) -> TopKSplatResult:
    if not results:
        raise ValueError("top-K temporal memory cannot be empty")
    if len(results) == 1 and not candidate_contract.enabled:
        return results[0]
    return merge_topk_splat_results(
        results,
        top_k=contract.top_k,
        depth_temperature_m=contract.depth_temperature_m,
        age_temperature_frames=contract.age_temperature_frames,
        source_collision_penalty=contract.source_collision_penalty,
        selection_contract=(
            TOPK_DIVERSITY_V31_CONTRACT
            if candidate_contract.enabled
            else "global_depth_v2"
        ),
        per_age_quota=candidate_contract.per_age_quota,
        surface_depth_gap_m=candidate_contract.surface_depth_gap_m,
        surface_relative_depth_gap=candidate_contract.surface_relative_depth_gap,
        phase_redundancy_sigma_grid_px=(
            candidate_contract.phase_redundancy_sigma_grid_px
        ),
        phase_redundancy_penalty=candidate_contract.phase_redundancy_penalty,
    )


def _topk_photometric_residual(
    result: TopKSplatResult,
    current_rgb_hr: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    warped_rgb = result.weighted_hidden_feature
    if warped_rgb is None or warped_rgb.shape != current_rgb_hr.shape:
        raise ValueError("HR top-K transport must carry a warped RGB feature")
    residual = (warped_rgb.detach().float() - current_rgb_hr.detach().float()).abs().mean(
        dim=1, keepdim=True
    )
    return torch.where(
        valid_mask,
        torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(residual),
    )


def _topk_depth_layer_collision(
    result: TopKSplatResult,
    contract: TemporalHistoryV2,
) -> Tensor:
    """Detect a competing depth layer, not ordinary bilinear overlap.

    Four-neighbour splatting and multi-age same-surface memory naturally put
    multiple candidates at a target pixel.  Those are not occlusion
    collisions.  A strict collision exists only when a retained candidate
    behind the nearest layer differs by both the configured absolute/relative
    depth tolerance.
    """

    batch, candidates, height, width = result.depth_m.shape
    if candidates < 2:
        return torch.zeros(
            (batch, 1, height, width),
            dtype=torch.bool,
            device=result.depth_m.device,
        )
    nearest_depth = result.depth_m[:, :1]
    threshold = torch.maximum(
        torch.full_like(nearest_depth, contract.collision_depth_gap_m),
        nearest_depth.abs() * contract.collision_relative_depth_gap,
    )
    competing = (
        result.valid_mask[:, 1:]
        & torch.isfinite(result.depth_m[:, 1:])
        & torch.isfinite(nearest_depth)
        & ((result.depth_m[:, 1:] - nearest_depth) > threshold)
    )
    return competing.any(dim=1, keepdim=True)


def _topk_context_prior_weights(
    result: TopKSplatResult, contract: TemporalHistoryV2
) -> Tensor:
    """Normalize a finite prior over every valid front/back candidate.

    ``TopKSplatResult.z_aware_weights`` is intentionally metric-only in v3.1:
    back layers receive zero so they cannot create a mixed disparity.  The
    current-conditioned context branch still needs a weak, depth-aware prior
    over those candidates.  This helper mirrors the explicit confidence,
    footprint, age, depth and collision factors without the front-layer mask.
    """

    compute_dtype = result.depth_m.dtype
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    nearest = result.depth_m[:, :1].to(dtype=compute_dtype)
    depth_delta = (
        result.depth_m.to(dtype=compute_dtype) - nearest
    ).clamp_min(0.0)
    collision_factor = torch.where(
        result.source_collision_mask,
        torch.full_like(
            result.confidence.to(dtype=compute_dtype),
            contract.source_collision_penalty,
        ),
        torch.ones_like(result.confidence, dtype=compute_dtype),
    )
    unnormalized = torch.where(
        result.valid_mask,
        result.confidence.to(dtype=compute_dtype).clamp(0.0, 1.0)
        * result.footprint_weight.to(dtype=compute_dtype).clamp_min(0.0)
        * torch.exp(-depth_delta / contract.depth_temperature_m).clamp_min(1e-6)
        * torch.exp(
            -result.temporal_age_frames.to(dtype=compute_dtype)
            / contract.age_temperature_frames
        ).clamp_min(1e-6)
        * collision_factor,
        torch.zeros_like(result.confidence, dtype=compute_dtype),
    )
    unnormalized = torch.nan_to_num(
        unnormalized, nan=0.0, posinf=0.0, neginf=0.0
    )
    denominator = unnormalized.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0,
        unnormalized / denominator.clamp_min(torch.finfo(compute_dtype).tiny),
        torch.zeros_like(unnormalized),
    ).to(dtype=result.z_aware_weights.dtype)


def _topk_quality_masks(
    result: TopKSplatResult,
    *,
    contract: TemporalHistoryV2,
    current_rgb_hr: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    pose_valid: Tensor,
    photometric_temperature: float,
    disparity_temperature_hr_px: float,
    reject_conflict_hr_px: float,
    photometric_threshold: float,
    geometry_threshold_hr_px: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gate one top-K aggregate and return confidence plus strict masks."""

    pose_mask = pose_valid.reshape(-1, 1, 1, 1)
    visibility = result.aggregate_valid_mask & pose_mask
    collision = _topk_depth_layer_collision(result, contract) & pose_mask
    warped_disparity = torch.where(
        visibility,
        result.weighted_disparity_hr_px,
        torch.zeros_like(result.weighted_disparity_hr_px),
    )
    warped_confidence = torch.where(
        visibility,
        result.weighted_confidence,
        torch.zeros_like(result.weighted_confidence),
    )
    photometric = _topk_photometric_residual(
        result, current_rgb_hr, visibility
    )
    current_ffs_disparity_hr = functional.interpolate(
        current_ffs_disparity_hr_px,
        size=warped_disparity.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    current_ffs_confidence_hr = functional.interpolate(
        current_ffs_confidence,
        size=warped_disparity.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    confidence_result = history_confidence(
        warped_confidence,
        visibility,
        # Top-K is the explicit collision-resolution mechanism. Keep the
        # collision tensor for strict losses/diagnostics, but do not discard
        # the z-aware aggregate merely because more than one candidate landed.
        torch.zeros_like(collision),
        photometric,
        warped_disparity,
        current_ffs_disparity_hr,
        current_ffs_confidence_hr,
        photometric_temperature=photometric_temperature,
        disparity_temperature_hr_px=disparity_temperature_hr_px,
        reject_conflict_hr_px=reject_conflict_hr_px,
    )
    effective_valid = confidence_result.valid_mask & pose_mask
    geometry_consistent = (
        effective_valid
        & torch.isfinite(confidence_result.disparity_error_hr_px)
        & (confidence_result.disparity_error_hr_px <= geometry_threshold_hr_px)
    )
    static_mask = (
        geometry_consistent
        & torch.isfinite(photometric)
        & (photometric <= photometric_threshold)
    )
    effective_confidence = torch.where(
        effective_valid,
        confidence_result.confidence,
        torch.zeros_like(confidence_result.confidence),
    )
    return (
        effective_valid,
        effective_confidence,
        visibility & effective_valid,
        collision,
        photometric,
        static_mask,
        geometry_consistent,
    )


def build_topk_temporal_transport(
    *,
    memory: Sequence[TemporalMemoryEntry],
    current_time_index: int,
    current_rgb_hr: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    intrinsics_current_hr: Tensor,
    baseline_current_m: Tensor,
    temporal_extrinsics_camera_from_world: Tensor,
    temporal_pose_valid: Tensor,
    contract: TemporalHistoryV2,
    temporal_pose_quality_score: Tensor | None = None,
    candidate_contract: TemporalCandidateFusionV31 = TemporalCandidateFusionV31(),
    scale: int = 2,
    align_corners_false_pixel_centers: bool = False,
    photometric_temperature: float = 0.10,
    disparity_temperature_hr_px: float = 2.0,
    reject_conflict_hr_px: float = 2.0,
    photometric_threshold: float = 0.10,
    geometry_threshold_hr_px: float = 2.0,
) -> TemporalTransport:
    """Warp all available causal memories and their ConvGRU state to now.

    Disparity/RGB are splatted on the HR grid before LR centre selection, so
    thin geometry is never averaged away. ConvGRU states are independently
    splatted on the calibrated LR grid and supplied as the current recurrent
    prior. Geometry/index selection is detached; hidden feature values retain
    gradients for the three-step unroll.
    """

    if not contract.enabled:
        raise ValueError("build_topk_temporal_transport requires an enabled contract")
    if candidate_contract.enabled and not align_corners_false_pixel_centers:
        raise ValueError(
            "v3.1 candidate fusion requires align_corners_false pixel centres"
        )
    if not memory:
        raise ValueError("top-K temporal transport requires at least one memory")
    selected = list(memory)[-contract.memory_frames :]
    selected.sort(key=lambda item: current_time_index - item.time_index)
    ages = [current_time_index - item.time_index for item in selected]
    if any(age < 1 or age > contract.memory_frames for age in ages):
        raise ValueError("memory entries must be distinct causal predecessors")
    if len(set(ages)) != len(ages):
        raise ValueError("temporal memory contains duplicate frame ages")

    batch_size = current_ffs_disparity_hr_px.shape[0]
    if candidate_contract.enabled:
        if (
            not isinstance(temporal_pose_quality_score, Tensor)
            or not temporal_pose_quality_score.is_floating_point()
            or temporal_pose_quality_score.shape not in {
                (batch_size,),
                (batch_size, 1),
            }
            or temporal_pose_quality_score.device
            != current_ffs_disparity_hr_px.device
        ):
            raise ValueError(
                "v3.1 temporal_pose_quality_score must be floating [B] on "
                "the transport device"
            )
        quality_score = temporal_pose_quality_score.reshape(batch_size)
        pose_gate = temporal_pose_valid.reshape(batch_size).to(dtype=torch.bool)
        if (
            not bool(torch.isfinite(quality_score).all())
            or bool(((quality_score < 0) | (quality_score > 1)).any())
            or bool((pose_gate & (quality_score <= 0)).any())
            or bool(((~pose_gate) & (quality_score != 0)).any())
        ):
            raise ValueError(
                "temporal pose quality must be in (0,1] for valid poses and "
                "exact zero for rejected poses"
            )
    else:
        quality_score = None
    if intrinsics_current_hr.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics_current_hr must have shape [B,3,3]")
    if baseline_current_m.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("baseline_current_m must have shape [B] or [B,1]")
    intrinsics_current_lr = _lr_intrinsics_from_hr(
        intrinsics_current_hr,
        scale=scale,
        align_corners_false_pixel_centers=(
            align_corners_false_pixel_centers
        ),
    )
    hr_results: list[TopKSplatResult] = []
    lr_hidden_results: list[TopKSplatResult] = []
    lr_candidate_results: list[TopKSplatResult] = []
    hidden_widths: tuple[int, ...] | None = None
    age_one_hr: TopKSplatResult | None = None

    for entry, age in zip(selected, ages, strict=True):
        if entry.intrinsics_hr.shape != (batch_size, 3, 3):
            raise ValueError(
                "temporal memory intrinsics_hr must have shape [B,3,3]"
            )
        if entry.baseline_m.shape not in {(batch_size,), (batch_size, 1)}:
            raise ValueError(
                "temporal memory baseline_m must have shape [B] or [B,1]"
            )
        intrinsics_previous_lr = _lr_intrinsics_from_hr(
            entry.intrinsics_hr,
            scale=scale,
            align_corners_false_pixel_centers=(
                align_corners_false_pixel_centers
            ),
        )
        previous_pose, current_pose, pose_valid = _vggt_pose_pair_for_age(
            temporal_extrinsics_camera_from_world,
            temporal_pose_valid,
            age_frames=age,
        )
        previous_disparity_hr = entry.output.disparity_hr_px.detach()
        previous_confidence_hr = torch.exp(
            -0.5 * entry.output.log_variance.detach()
        ).clamp(0.0, 1.0)
        previous_valid_hr = (
            torch.isfinite(previous_disparity_hr) & (previous_disparity_hr > 0)
        )
        if entry.output.output_valid_mask is not None:
            previous_valid_hr &= entry.output.output_valid_mask.detach()
        if entry.output.valid_probability is not None:
            previous_confidence_hr = (
                previous_confidence_hr
                * entry.output.valid_probability.detach().to(previous_confidence_hr)
            )
        hr_result = _topk_splat_for_memory(
            disparity_hr_px=previous_disparity_hr,
            confidence=previous_confidence_hr,
            hidden_feature=entry.rgb_hr.detach(),
            source_valid_mask=previous_valid_hr,
            intrinsics_previous_grid=entry.intrinsics_hr,
            intrinsics_current_grid=intrinsics_current_hr,
            intrinsics_previous_hr=entry.intrinsics_hr,
            intrinsics_current_hr=intrinsics_current_hr,
            baseline_previous_m=entry.baseline_m,
            baseline_current_m=baseline_current_m,
            previous_pose=previous_pose,
            current_pose=current_pose,
            pose_valid=pose_valid,
            age_frames=age,
            contract=contract,
        )
        hr_results.append(hr_result)
        if age == 1:
            age_one_hr = hr_result

        # The corrected v3 path constructs model candidates directly on the
        # align_corners=False LR grid. This avoids assigning an HR winner at
        # ``s*u`` to an LR centre which physically lies at
        # ``(u+0.5)*s-0.5``. Legacy v2 keeps its exact historical behaviour.
        # Age-1 alone owns the recurrent initial state.  V3.1 additionally
        # carries each age's final-layer hidden feature as a per-candidate
        # attention value; that feature never bypasses the learned candidate
        # selector into the recurrent state.
        if age == 1 or align_corners_false_pixel_centers:
            if not entry.output.hidden_state:
                raise ValueError(
                    "top-K hidden warp requires a non-empty ConvGRU state"
                )
            if age == 1:
                hidden_widths = tuple(
                    int(state.shape[1]) for state in entry.output.hidden_state
                )
                recurrent_hidden_feature: Tensor | None = torch.cat(
                    tuple(entry.output.hidden_state), dim=1
                )
            else:
                recurrent_hidden_feature = None
            previous_disparity_lr = _sample_hr_winner_grid_to_lr(
                previous_disparity_hr,
                scale=scale,
                align_corners_false_pixel_centers=(
                    align_corners_false_pixel_centers
                ),
            )
            previous_confidence_lr = _sample_hr_winner_grid_to_lr(
                previous_confidence_hr,
                scale=scale,
                align_corners_false_pixel_centers=(
                    align_corners_false_pixel_centers
                ),
            )
            previous_valid_lr = _sample_hr_winner_grid_to_lr(
                previous_valid_hr,
                scale=scale,
                align_corners_false_pixel_centers=(
                    align_corners_false_pixel_centers
                ),
            )
            lr_result = _topk_splat_for_memory(
                disparity_hr_px=previous_disparity_lr,
                confidence=previous_confidence_lr,
                hidden_feature=recurrent_hidden_feature,
                source_valid_mask=previous_valid_lr,
                intrinsics_previous_grid=intrinsics_previous_lr,
                intrinsics_current_grid=intrinsics_current_lr,
                intrinsics_previous_hr=entry.intrinsics_hr,
                intrinsics_current_hr=intrinsics_current_hr,
                baseline_previous_m=entry.baseline_m,
                baseline_current_m=baseline_current_m,
                previous_pose=previous_pose,
                current_pose=current_pose,
                pose_valid=pose_valid,
                age_frames=age,
                contract=contract,
            )
            if age == 1:
                lr_hidden_results.append(lr_result)
            if align_corners_false_pixel_centers:
                candidate_hidden = (
                    entry.output.history_value_feature
                    if candidate_contract.enabled
                    else None
                )
                if candidate_contract.enabled and candidate_hidden is None:
                    raise ValueError(
                        "v3.1 memory entry lacks compressed history value feature"
                    )
                lr_candidate_results.append(
                    _topk_splat_for_memory(
                        disparity_hr_px=previous_disparity_lr,
                        confidence=previous_confidence_lr,
                        hidden_feature=candidate_hidden,
                        source_valid_mask=previous_valid_lr,
                        intrinsics_previous_grid=intrinsics_previous_lr,
                        intrinsics_current_grid=intrinsics_current_lr,
                        intrinsics_previous_hr=entry.intrinsics_hr,
                        intrinsics_current_hr=intrinsics_current_hr,
                        baseline_previous_m=entry.baseline_m,
                        baseline_current_m=baseline_current_m,
                        previous_pose=previous_pose,
                        current_pose=current_pose,
                        pose_valid=pose_valid,
                        age_frames=age,
                        contract=contract,
                    )
                )

    if age_one_hr is None or hidden_widths is None:
        raise ValueError("top-K transport requires the immediate age-1 memory")
    merged_hr = _merge_topk_results(hr_results, contract, candidate_contract)
    merged_lr_hidden = _merge_topk_results(lr_hidden_results, contract)
    merged_lr_candidates = (
        _merge_topk_results(lr_candidate_results, contract, candidate_contract)
        if align_corners_false_pixel_centers
        else None
    )
    _, _, pose_valid = _vggt_pose_pair_for_age(
        temporal_extrinsics_camera_from_world,
        temporal_pose_valid,
        age_frames=1,
    )
    (
        effective_valid_hr,
        confidence_history_hr,
        visibility_hr,
        collision_hr,
        photometric_hr,
        static_hr,
        geometry_consistent_hr,
    ) = _topk_quality_masks(
        merged_hr,
        contract=contract,
        current_rgb_hr=current_rgb_hr,
        current_ffs_disparity_hr_px=current_ffs_disparity_hr_px,
        current_ffs_confidence=current_ffs_confidence,
        pose_valid=pose_valid,
        photometric_temperature=photometric_temperature,
        disparity_temperature_hr_px=disparity_temperature_hr_px,
        reject_conflict_hr_px=reject_conflict_hr_px,
        photometric_threshold=photometric_threshold,
        geometry_threshold_hr_px=geometry_threshold_hr_px,
    )
    disparity_history_hr = torch.where(
        effective_valid_hr,
        merged_hr.weighted_disparity_hr_px,
        torch.zeros_like(merged_hr.weighted_disparity_hr_px),
    ).detach()
    fractional_hr = torch.where(
        effective_valid_hr.expand(-1, 2, -1, -1),
        merged_hr.weighted_fractional_offset_grid_px,
        torch.zeros_like(merged_hr.weighted_fractional_offset_grid_px),
    ).detach()

    # The loss/TEPE transition remains strictly age 1 even though the model
    # receives an age-1/age-2 top-K aggregate.
    (
        loss_valid_hr,
        loss_confidence_hr,
        loss_visibility_hr,
        loss_collision_hr,
        loss_photometric_hr,
        loss_static_hr,
        loss_geometry_consistent_hr,
    ) = _topk_quality_masks(
        age_one_hr,
        contract=contract,
        current_rgb_hr=current_rgb_hr,
        current_ffs_disparity_hr_px=current_ffs_disparity_hr_px,
        current_ffs_confidence=current_ffs_confidence,
        pose_valid=pose_valid,
        photometric_temperature=photometric_temperature,
        disparity_temperature_hr_px=disparity_temperature_hr_px,
        reject_conflict_hr_px=reject_conflict_hr_px,
        photometric_threshold=photometric_threshold,
        geometry_threshold_hr_px=geometry_threshold_hr_px,
    )
    loss_disparity_hr = torch.where(
        loss_valid_hr,
        age_one_hr.weighted_disparity_hr_px,
        torch.zeros_like(age_one_hr.weighted_disparity_hr_px),
    ).detach()

    def sample_hr_to_lr(value: Tensor) -> Tensor:
        return _sample_hr_winner_grid_to_lr(
            value,
            scale=scale,
            align_corners_false_pixel_centers=(
                align_corners_false_pixel_centers
            ),
        )

    effective_valid_lr = sample_hr_to_lr(
        effective_valid_hr
    )
    merged_hidden = merged_lr_hidden.weighted_hidden_feature
    if merged_hidden is None:
        raise RuntimeError("LR top-K transport did not return a hidden feature")
    hidden_valid_lr = (
        merged_lr_hidden.aggregate_valid_mask
        & pose_valid.reshape(-1, 1, 1, 1)
        & effective_valid_lr
    )
    merged_hidden = torch.where(
        hidden_valid_lr,
        merged_hidden,
        torch.zeros_like(merged_hidden),
    )
    hidden_state: list[Tensor] = []
    offset = 0
    for width in hidden_widths:
        hidden_state.append(merged_hidden[:, offset : offset + width])
        offset += width
    if offset != merged_hidden.shape[1]:
        raise RuntimeError("warped hidden-state channel partition is inconsistent")

    candidate_valid_hr = merged_hr.valid_mask & effective_valid_hr
    candidate_weights_hr = torch.where(
        candidate_valid_hr,
        merged_hr.z_aware_weights,
        torch.zeros_like(merged_hr.z_aware_weights),
    )
    topk_depth_lr: Tensor | None = None
    topk_pose_quality_lr: Tensor | None = None
    topk_depth_layer_lr: Tensor | None = None
    topk_front_surface_lr: Tensor | None = None
    topk_context_only_lr: Tensor | None = None
    topk_age2_available_lr: Tensor | None = None
    topk_warped_hidden_lr: Tensor | None = None
    topk_metric_prior_lr: Tensor | None = None
    if align_corners_false_pixel_centers:
        if merged_lr_candidates is None:
            raise RuntimeError("v3 LR top-K candidates were not constructed")
        candidate_valid_lr = (
            merged_lr_candidates.valid_mask
            & effective_valid_lr
            & pose_valid.reshape(-1, 1, 1, 1)
        )
        metric_candidate_weights_lr = torch.where(
            candidate_valid_lr,
            merged_lr_candidates.z_aware_weights,
            torch.zeros_like(merged_lr_candidates.z_aware_weights),
        )
        topk_metric_prior_lr = metric_candidate_weights_lr
        candidate_weights_lr = (
            torch.where(
                candidate_valid_lr,
                _topk_context_prior_weights(
                    merged_lr_candidates, contract
                ),
                torch.zeros_like(merged_lr_candidates.z_aware_weights),
            )
            if candidate_contract.enabled
            else metric_candidate_weights_lr
        )
        disparity_history_lr = torch.where(
            effective_valid_lr,
            merged_lr_candidates.weighted_disparity_hr_px,
            torch.zeros_like(merged_lr_candidates.weighted_disparity_hr_px),
        )
        topk_disparity_lr = torch.where(
            candidate_valid_lr,
            merged_lr_candidates.disparity_hr_px,
            torch.zeros_like(merged_lr_candidates.disparity_hr_px),
        )
        topk_confidence_lr = torch.where(
            candidate_valid_lr,
            merged_lr_candidates.confidence,
            torch.zeros_like(merged_lr_candidates.confidence),
        )
        topk_fractional_lr = torch.where(
            candidate_valid_lr.unsqueeze(2),
            merged_lr_candidates.fractional_offset_grid_px,
            torch.zeros_like(merged_lr_candidates.fractional_offset_grid_px),
        )
        topk_age_lr = torch.where(
            candidate_valid_lr,
            merged_lr_candidates.temporal_age_frames,
            torch.zeros_like(merged_lr_candidates.temporal_age_frames),
        )
        if candidate_contract.enabled:
            for name, value in (
                ("front_surface_mask", merged_lr_candidates.front_surface_mask),
                ("context_only_mask", merged_lr_candidates.context_only_mask),
                ("depth_layer_index", merged_lr_candidates.depth_layer_index),
                (
                    "age2_depth_consistent_available_mask",
                    merged_lr_candidates.age2_depth_consistent_available_mask,
                ),
                ("warped_hidden_feature", merged_lr_candidates.warped_hidden_feature),
            ):
                if value is None:
                    raise RuntimeError(
                        f"v3.1 candidate merge did not populate {name}"
                    )
            assert merged_lr_candidates.front_surface_mask is not None
            assert merged_lr_candidates.context_only_mask is not None
            assert merged_lr_candidates.depth_layer_index is not None
            assert (
                merged_lr_candidates.age2_depth_consistent_available_mask
                is not None
            )
            assert merged_lr_candidates.warped_hidden_feature is not None
            topk_depth_lr = torch.where(
                candidate_valid_lr,
                merged_lr_candidates.depth_m,
                torch.zeros_like(merged_lr_candidates.depth_m),
            )
            assert quality_score is not None
            topk_pose_quality_lr = (
                quality_score.to(dtype=topk_depth_lr.dtype)
                .reshape(batch_size, 1, 1, 1)
                .expand_as(topk_depth_lr)
                * candidate_valid_lr.to(dtype=topk_depth_lr.dtype)
            )
            topk_depth_layer_lr = torch.where(
                candidate_valid_lr,
                merged_lr_candidates.depth_layer_index,
                torch.full_like(merged_lr_candidates.depth_layer_index, -1),
            )
            topk_front_surface_lr = (
                candidate_valid_lr
                & merged_lr_candidates.front_surface_mask
            )
            topk_context_only_lr = (
                candidate_valid_lr
                & merged_lr_candidates.context_only_mask
            )
            topk_age2_available_lr = (
                merged_lr_candidates.age2_depth_consistent_available_mask
                & effective_valid_lr
            )
            topk_warped_hidden_lr = torch.where(
                candidate_valid_lr.unsqueeze(2),
                merged_lr_candidates.warped_hidden_feature,
                torch.zeros_like(merged_lr_candidates.warped_hidden_feature),
            )
        # Aggregate from the explicit [B,K,2,H,W] candidate layout. The
        # generic splat result's historical flat reshape is retained for v2,
        # while corrected v3 phase must stay attached to its target pixel.
        fractional_offset_lr = (
            metric_candidate_weights_lr.unsqueeze(2) * topk_fractional_lr
        ).sum(dim=1)
        fractional_offset_lr = torch.where(
            effective_valid_lr.expand(-1, 2, -1, -1),
            fractional_offset_lr,
            torch.zeros_like(fractional_offset_lr),
        )
    else:
        candidate_valid_lr = sample_hr_to_lr(candidate_valid_hr)
        candidate_weights_lr = sample_hr_to_lr(candidate_weights_hr)
        topk_metric_prior_lr = candidate_weights_lr
        disparity_history_lr = sample_hr_to_lr(disparity_history_hr)
        fractional_offset_lr = sample_hr_to_lr(fractional_hr)
        topk_disparity_lr = sample_hr_to_lr(
            torch.where(
                candidate_valid_hr,
                merged_hr.disparity_hr_px,
                torch.zeros_like(merged_hr.disparity_hr_px),
            )
        )
        topk_confidence_lr = sample_hr_to_lr(
            torch.where(
                candidate_valid_hr,
                merged_hr.confidence,
                torch.zeros_like(merged_hr.confidence),
            )
        )
        topk_fractional_lr = (
            torch.where(
                candidate_valid_hr.unsqueeze(2),
                merged_hr.fractional_offset_grid_px,
                torch.zeros_like(merged_hr.fractional_offset_grid_px),
            )[..., ::scale, ::scale]
            .contiguous()
        )
        topk_age_lr = sample_hr_to_lr(
            torch.where(
                candidate_valid_hr,
                merged_hr.temporal_age_frames,
                torch.zeros_like(merged_hr.temporal_age_frames),
            )
        )
    return TemporalTransport(
        disparity_history_hr_px=disparity_history_lr.detach(),
        confidence_history=sample_hr_to_lr(confidence_history_hr.detach()),
        visibility_mask=sample_hr_to_lr(visibility_hr.detach()),
        valid_history=effective_valid_lr.detach(),
        collision_mask=sample_hr_to_lr(collision_hr.detach()),
        photometric_residual=sample_hr_to_lr(photometric_hr.detach()),
        fractional_offset_px=fractional_offset_lr.detach(),
        static_mask=sample_hr_to_lr(static_hr.detach()),
        geometry_consistent_mask=sample_hr_to_lr(
            geometry_consistent_hr.detach()
        ),
        disparity_history_loss_hr_px=loss_disparity_hr,
        confidence_history_hr=loss_confidence_hr.detach(),
        visibility_mask_hr=loss_visibility_hr.detach(),
        valid_history_hr=loss_valid_hr.detach(),
        collision_mask_hr=loss_collision_hr.detach(),
        photometric_residual_hr=loss_photometric_hr.detach(),
        static_mask_hr=loss_static_hr.detach(),
        geometry_consistent_mask_hr=loss_geometry_consistent_hr.detach(),
        topk_disparity_history_hr_px=topk_disparity_lr.detach(),
        topk_confidence_history=topk_confidence_lr.detach(),
        topk_fractional_offset_px=topk_fractional_lr.detach(),
        topk_temporal_age_frames=topk_age_lr.detach(),
        topk_z_aware_weights=candidate_weights_lr.detach(),
        topk_metric_prior_weights=(
            None
            if topk_metric_prior_lr is None
            else topk_metric_prior_lr.detach()
        ),
        topk_valid_mask=candidate_valid_lr.detach(),
        topk_depth_m=(None if topk_depth_lr is None else topk_depth_lr.detach()),
        topk_pose_quality=(
            None if topk_pose_quality_lr is None else topk_pose_quality_lr.detach()
        ),
        topk_depth_layer_index=(
            None if topk_depth_layer_lr is None else topk_depth_layer_lr.detach()
        ),
        topk_front_surface_mask=(
            None if topk_front_surface_lr is None else topk_front_surface_lr.detach()
        ),
        topk_context_only_mask=(
            None if topk_context_only_lr is None else topk_context_only_lr.detach()
        ),
        topk_age2_depth_consistent_available_mask=(
            None if topk_age2_available_lr is None else topk_age2_available_lr.detach()
        ),
        # Hidden values remain differentiable through the causal three-step
        # unroll; projection indices and every geometric field above detach.
        topk_warped_hidden_feature=topk_warped_hidden_lr,
        warped_hidden_state=tuple(hidden_state),
    )


def build_reference_temporal_warp(
    *,
    previous_reference_disparity_hr_px: Tensor,
    previous_reference_confidence: Tensor,
    previous_reference_valid_mask: Tensor,
    previous_prediction_disparity_hr_px: Tensor | None = None,
    intrinsics_previous_hr: Tensor,
    baseline_previous_m: Tensor,
    intrinsics_current_hr: Tensor,
    baseline_current_m: Tensor,
    temporal_extrinsics_camera_from_world: Tensor,
    temporal_pose_valid: Tensor,
    contract: TemporalHistoryV2,
) -> ReferenceTemporalWarp:
    """Warp teacher and prediction with shared source correspondences.

    Source disparity is unprojected with ``intrinsics_previous_hr`` and
    ``baseline_previous_m``. Projection and returned HR-pixel disparity use
    the explicit current calibration.
    """

    previous_pose, current_pose, pose_valid = _vggt_pose_pair_for_age(
        temporal_extrinsics_camera_from_world,
        temporal_pose_valid,
        age_frames=1,
    )
    result = _topk_splat_for_memory(
        disparity_hr_px=previous_reference_disparity_hr_px.detach(),
        confidence=previous_reference_confidence.detach(),
        hidden_feature=None,
        source_valid_mask=previous_reference_valid_mask.detach(),
        intrinsics_previous_grid=intrinsics_previous_hr,
        intrinsics_current_grid=intrinsics_current_hr,
        intrinsics_previous_hr=intrinsics_previous_hr,
        intrinsics_current_hr=intrinsics_current_hr,
        baseline_previous_m=baseline_previous_m,
        baseline_current_m=baseline_current_m,
        previous_pose=previous_pose,
        current_pose=current_pose,
        pose_valid=pose_valid,
        age_frames=1,
        contract=contract,
    )
    pose_mask = pose_valid.reshape(-1, 1, 1, 1)
    valid = result.aggregate_valid_mask & pose_mask
    collision = _topk_depth_layer_collision(result, contract) & pose_mask
    strict_valid = valid & ~collision
    disparity = torch.where(
        strict_valid,
        result.weighted_disparity_hr_px,
        torch.zeros_like(result.weighted_disparity_hr_px),
    )
    prediction_disparity: Tensor | None = None
    if previous_prediction_disparity_hr_px is not None:
        if previous_prediction_disparity_hr_px.shape != (
            previous_reference_disparity_hr_px.shape
        ):
            raise ValueError(
                "previous prediction/reference disparity shapes must match"
            )
        source_index = result.source_linear_index.clamp_min(0)
        prediction_flat = previous_prediction_disparity_hr_px.detach().reshape(-1)
        reference_flat = previous_reference_disparity_hr_px.detach().reshape(-1)
        prediction_candidate = prediction_flat[source_index]
        previous_reference_candidate = reference_flat[source_index]
        ratio = torch.where(
            result.valid_mask & torch.isfinite(previous_reference_candidate)
            & (previous_reference_candidate > 0),
            result.disparity_hr_px
            / previous_reference_candidate.clamp_min(1e-6),
            torch.zeros_like(result.disparity_hr_px),
        )
        transported_prediction_candidates = torch.where(
            result.valid_mask,
            prediction_candidate * ratio,
            torch.zeros_like(prediction_candidate),
        )
        prediction_disparity = (
            result.z_aware_weights * transported_prediction_candidates
        ).sum(dim=1, keepdim=True)
        prediction_disparity = torch.where(
            strict_valid,
            prediction_disparity,
            torch.zeros_like(prediction_disparity),
        ).detach()
    return ReferenceTemporalWarp(
        disparity_hr_px=disparity.detach(),
        prediction_disparity_hr_px=prediction_disparity,
        valid_mask_hr=strict_valid.detach(),
        visibility_mask_hr=strict_valid.detach(),
        collision_mask_hr=collision.detach(),
    )


def compute_stage_b_step_loss(
    output: ModelOutput,
    batch: Mapping[str, Any],
    *,
    transport: TemporalTransport | None,
    reference_transport: ReferenceTemporalWarp | None = None,
    scale: int = 2,
    weights: LossWeights = LossWeights(),
    max_photometric_residual: float = 0.10,
    positivity_ablation: PositivityAblation = PositivityAblation(),
    physical_output_v2: PhysicalOutputV2 = PhysicalOutputV2(),
    temporal_residual_v2: TemporalResidualV2 = TemporalResidualV2(),
) -> LossBreakdown:
    """Compute spatial supervision plus visibility-gated temporal consistency."""

    spatial = compute_stage_a_loss(
        output,
        batch,
        scale=scale,
        weights=weights,
        positivity_ablation=positivity_ablation,
        physical_output_v2=physical_output_v2,
    )
    if transport is None:
        temporal = output.disparity_hr_px.sum() * 0.0
    elif temporal_residual_v2.enabled:
        if reference_transport is None:
            raise ValueError(
                "temporal_residual_v2 requires a teacher/GT reference warp"
            )
        if reference_transport.prediction_disparity_hr_px is None:
            raise ValueError(
                "temporal_residual_v2 reference warp lacks teacher-correspondence prediction"
            )
        target = batch.get("teacher_disparity_hr_px")
        target_valid = batch.get("teacher_valid_mask")
        target_trusted = batch.get("teacher_trusted_mask")
        if not all(
            isinstance(value, Tensor)
            for value in (target, target_valid, target_trusted)
        ):
            raise ValueError(
                "temporal_residual_v2 requires teacher disparity/valid/trusted tensors"
            )
        current_reference_valid = (
            target_valid.to(dtype=torch.bool)
            & target_trusted.to(dtype=torch.bool)
            & torch.isfinite(target)
            & (target > 0)
        )
        temporal = temporal_residual_consistency_loss(
            output.disparity_hr_px,
            reference_transport.prediction_disparity_hr_px,
            target,
            reference_transport.disparity_hr_px,
            static_mask=transport.static_mask_hr,
            visibility_mask=transport.visibility_mask_hr,
            collision_mask=transport.collision_mask_hr,
            photometric_residual=transport.photometric_residual_hr,
            max_photometric_residual=max_photometric_residual,
            geometry_consistent_mask=transport.geometry_consistent_mask_hr,
            current_reference_valid_mask=current_reference_valid,
            warped_previous_reference_valid_mask=(
                reference_transport.valid_mask_hr
            ),
            history_confidence=transport.confidence_history_hr,
        )
    else:
        temporal = temporal_consistency_loss(
            output.disparity_hr_px,
            transport.disparity_history_loss_hr_px,
            static_mask=transport.static_mask_hr,
            visibility_mask=transport.visibility_mask_hr,
            collision_mask=transport.collision_mask_hr,
            photometric_residual=transport.photometric_residual_hr,
            max_photometric_residual=max_photometric_residual,
            geometry_consistent_mask=transport.geometry_consistent_mask_hr,
            history_confidence=transport.confidence_history_hr,
        )
    baseline = combine_loss_terms(
        disparity=spatial.disparity,
        measurement=spatial.measurement,
        gradient=spatial.gradient,
        temporal=temporal,
        epipolar=spatial.epipolar,
        uncertainty_nll=spatial.uncertainty_nll,
        gate_regularizer=spatial.gate_regularizer,
        weights=weights,
    )
    # ``spatial`` already owns the weighted ablation term.  Recombine the
    # standard Stage-B terms first, then add it once after temporal loss.
    extra_total = baseline.total
    if positivity_ablation.enabled:
        if spatial.positivity_penalty is None:
            raise RuntimeError("enabled positivity ablation produced no penalty")
        extra_total = extra_total + spatial.positivity_penalty
    if physical_output_v2.enabled:
        if any(
            value is None
            for value in (
                spatial.valid_bce,
                spatial.completion_bce,
                spatial.validity_calibration,
            )
        ):
            raise RuntimeError("enabled physical V2 produced no validity losses")
        extra_total = extra_total + (
            physical_output_v2.valid_bce_weight * spatial.valid_bce
            + physical_output_v2.completion_bce_weight * spatial.completion_bce
            + physical_output_v2.calibration_weight
            * spatial.validity_calibration
        )
    return replace(
        baseline,
        total=extra_total,
        positivity_penalty=spatial.positivity_penalty,
        valid_bce=spatial.valid_bce,
        completion_bce=spatial.completion_bce,
        validity_calibration=spatial.validity_calibration,
    )


def average_loss_breakdowns(
    values: Sequence[LossBreakdown],
) -> LossBreakdown:
    if not values:
        raise ValueError("cannot average an empty loss sequence")

    def mean(name: str) -> Tensor:
        return torch.stack([getattr(value, name) for value in values]).mean()

    def optional_mean(name: str) -> Tensor | None:
        optional_values = [getattr(value, name) for value in values]
        if any(value is None for value in optional_values):
            if not all(value is None for value in optional_values):
                raise ValueError(f"cannot mix enabled/disabled optional loss {name}")
            return None
        return torch.stack(
            [value for value in optional_values if value is not None]
        ).mean()
    return LossBreakdown(
        total=mean("total"),
        disparity=mean("disparity"),
        measurement=mean("measurement"),
        gradient=mean("gradient"),
        temporal=mean("temporal"),
        epipolar=mean("epipolar"),
        uncertainty_nll=mean("uncertainty_nll"),
        gate_regularizer=mean("gate_regularizer"),
        positivity_penalty=optional_mean("positivity_penalty"),
        valid_bce=optional_mean("valid_bce"),
        completion_bce=optional_mean("completion_bce"),
        validity_calibration=optional_mean("validity_calibration"),
    )


def _move_training_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _temporal_step_batch(batch: Mapping[str, Any], time_index: int) -> dict[str, Any]:
    """Select one ``[B,T,...]`` slice using the spatial loss field names."""

    mapping = {
        "rgb_hr": "rgb_hr_sequence",
        "disparity_ffs_hr_px": "disparity_ffs_hr_px_sequence",
        "confidence_ffs": "confidence_ffs_sequence",
        "valid_ffs": "valid_ffs_sequence",
        "observation_disparity_lr_px": "observation_disparity_lr_px_sequence",
        "observation_confidence": "observation_confidence_sequence",
        "observation_trusted_mask": "observation_trusted_mask_sequence",
        "teacher_disparity_hr_px": "teacher_disparity_hr_px_sequence",
        "teacher_confidence": "teacher_confidence_sequence",
        "teacher_valid_mask": "teacher_valid_mask_sequence",
        "teacher_trusted_mask": "teacher_trusted_mask_sequence",
    }
    result: dict[str, Any] = {}
    for output_name, sequence_name in mapping.items():
        value = batch.get(sequence_name)
        result[output_name] = None if value is None else value[:, time_index]
    return result


def _reset_hidden_where_pose_invalid(
    hidden_state: Sequence[Tensor] | None, pose_valid: Tensor
) -> tuple[Tensor, ...] | None:
    if hidden_state is None:
        return None
    selector = pose_valid.reshape(-1, 1, 1, 1).to(dtype=torch.bool)
    return tuple(torch.where(selector, state, torch.zeros_like(state)) for state in hidden_state)


def calibration_model_kwargs_spatial(
    batch: Mapping[str, Any], contract: CalibrationConditioningV3
) -> dict[str, Tensor]:
    """Build fail-closed v3 inputs for one spatial batch."""

    if not contract.enabled:
        return {}
    result: dict[str, Tensor] = {}
    if contract.use_rays:
        intrinsics = batch.get("K_hr")
        if not isinstance(intrinsics, Tensor):
            raise ValueError("calibration v3 spatial batch lacks K_hr")
        result["K_left_hr_px"] = intrinsics.float()
    if contract.use_stereo_pose or contract.use_temporal_pose:
        baseline = batch.get("baseline_m")
        transform = batch.get("T_right_rectified_from_left_rectified_m")
        if not isinstance(baseline, Tensor) or not isinstance(transform, Tensor):
            raise ValueError("calibration v3 spatial batch lacks baseline/stereo E")
        result["baseline_m"] = baseline.float().reshape(-1)
        result["T_right_rectified_from_left_rectified_m"] = (
            rectified_stereo_transform_4x4(transform)
        )
    if contract.use_temporal_pose:
        reference = batch.get("rgb_hr")
        if not isinstance(reference, Tensor):
            raise ValueError("calibration v3 spatial batch lacks rgb_hr")
        batch_size = reference.shape[0]
        identity = torch.eye(
            4, device=reference.device, dtype=torch.float32
        ).reshape(1, 1, 4, 4).expand(batch_size, 2, -1, -1).clone()
        result["T_current_from_history_m"] = identity
        result["temporal_pose_valid"] = torch.zeros(
            batch_size, 2, device=reference.device, dtype=torch.bool
        )
    return result


def calibration_model_kwargs_temporal(
    batch: Mapping[str, Any],
    *,
    time_index: int,
    contract: CalibrationConditioningV3,
    config: Mapping[str, Any] | DictConfig | None = None,
) -> dict[str, Tensor]:
    """Build current/causal-past-only v3 inputs for one temporal step."""

    if not contract.enabled:
        return {}
    spatial_view: dict[str, Any] = {
        "rgb_hr": batch["rgb_hr_sequence"][:, time_index],
        "K_hr": batch["K_hr_sequence"][:, time_index],
        "baseline_m": batch["baseline_m_sequence"][:, time_index],
        "T_right_rectified_from_left_rectified_m": (
            None
            if batch.get("T_right_rectified_from_left_rectified_m_sequence") is None
            else batch["T_right_rectified_from_left_rectified_m_sequence"][:, time_index]
        ),
    }
    result = calibration_model_kwargs_spatial(spatial_view, contract)
    if contract.use_temporal_pose:
        pose_sequence, pose_valid_sequence, _pose_quality_sequence = (
            temporal_pose_inputs_from_batch(batch, config)
        )
        transforms, valid = temporal_conditioning_transforms(
            pose_sequence[:, time_index],
            pose_valid_sequence[:, time_index],
            student_time_index=time_index,
        )
        result["T_current_from_history_m"] = transforms
        result["temporal_pose_valid"] = valid
    return result


def _forward_temporal_loss(
    model: FFSOmegaTSR,
    batch: Mapping[str, Any],
    *,
    config: DictConfig,
    weights: LossWeights,
    positivity_ablation: PositivityAblation = PositivityAblation(),
    diagnostic: bool = False,
) -> LossBreakdown:
    """Unroll exactly three causal steps and average their supervised losses."""

    physical_v2 = physical_output_v2_from_config(config)
    temporal_history_v2 = temporal_history_v2_from_config(config)
    temporal_residual_v2 = temporal_residual_v2_from_config(config)
    calibration_v3 = calibration_conditioning_v3_from_config(config)
    candidate_v31 = temporal_candidate_fusion_v3_1_from_config(config)
    # Resolve the pose source once and project it onto the historical field
    # names consumed by the existing transport code.  GT poses are supplied by
    # the Spring manifest and remain separate from the VGGT cache lineage until
    # this explicit, local aliasing boundary.
    temporal_poses, temporal_pose_valid, temporal_pose_quality = (
        temporal_pose_inputs_from_batch(batch, config)
    )
    batch = dict(batch)
    batch["vggt_extrinsics_camera_from_world_metric_sequence"] = temporal_poses
    batch["temporal_pose_valid_sequence"] = temporal_pose_valid
    batch["temporal_pose_quality_score_sequence"] = temporal_pose_quality
    temporal_pose_sequence = temporal_poses
    temporal_pose_valid_sequence = temporal_pose_valid
    temporal_pose_quality_sequence = temporal_pose_quality
    if temporal_history_v2.enabled and not calibration_v3.enabled:
        validate_v2_temporal_calibration(
            batch["K_hr_sequence"], batch["baseline_m_sequence"]
        )
    rgb_sequence = batch["rgb_hr_sequence"]
    if rgb_sequence.ndim != 5 or rgb_sequence.shape[1] != 3:
        raise ValueError("temporal RGB batch must have shape [B,3,3,H,W]")
    hidden_state: Sequence[Tensor] | None = None
    previous_output: ModelOutput | None = None
    previous_rgb_hr: Tensor | None = None
    memory: list[TemporalMemoryEntry] = []
    losses: list[LossBreakdown] = []
    for time_index in range(3):
        step = _temporal_step_batch(batch, time_index)
        pose_valid = temporal_pose_valid_sequence[:, time_index]
        transport: TemporalTransport | None = None
        reference_transport: ReferenceTemporalWarp | None = None
        if time_index > 0:
            if temporal_history_v2.enabled:
                transport = build_topk_temporal_transport(
                    memory=memory,
                    current_time_index=time_index,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
                    baseline_current_m=batch["baseline_m_sequence"][:, time_index],
                    temporal_extrinsics_camera_from_world=temporal_pose_sequence[
                        :, time_index
                    ],
                    temporal_pose_valid=pose_valid,
                    contract=temporal_history_v2,
                    temporal_pose_quality_score=(
                        temporal_pose_quality_sequence[:, time_index]
                    ),
                    candidate_contract=candidate_v31,
                    scale=int(config.data.scale),
                    align_corners_false_pixel_centers=(
                        calibration_v3.align_corners_false_pixel_centers
                    ),
                    photometric_temperature=float(config.train.photometric_temperature),
                    disparity_temperature_hr_px=float(
                        config.train.disparity_temperature_hr_px
                    ),
                    reject_conflict_hr_px=float(config.train.history_conflict_hr_px),
                    photometric_threshold=float(
                        config.train.temporal_photometric_threshold
                    ),
                    geometry_threshold_hr_px=float(
                        config.train.temporal_geometry_threshold_hr_px
                    ),
                )
                hidden_state = transport.warped_hidden_state
                previous_teacher = batch.get("teacher_disparity_hr_px_sequence")
                previous_teacher_confidence = batch.get(
                    "teacher_confidence_sequence"
                )
                previous_teacher_valid = batch.get("teacher_valid_mask_sequence")
                previous_teacher_trusted = batch.get(
                    "teacher_trusted_mask_sequence"
                )
                if not all(
                    isinstance(value, Tensor)
                    for value in (
                        previous_teacher,
                        previous_teacher_confidence,
                        previous_teacher_valid,
                        previous_teacher_trusted,
                    )
                ):
                    raise ValueError(
                        "temporal V2 requires teacher sequence disparity/confidence/masks"
                    )
                reference_transport = build_reference_temporal_warp(
                    previous_reference_disparity_hr_px=previous_teacher[
                        :, time_index - 1
                    ],
                    previous_reference_confidence=previous_teacher_confidence[
                        :, time_index - 1
                    ],
                    previous_reference_valid_mask=(
                        previous_teacher_valid[:, time_index - 1]
                        & previous_teacher_trusted[:, time_index - 1]
                    ),
                    previous_prediction_disparity_hr_px=memory[-1].output.disparity_hr_px,
                    intrinsics_previous_hr=memory[-1].intrinsics_hr,
                    baseline_previous_m=memory[-1].baseline_m,
                    intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
                    baseline_current_m=batch["baseline_m_sequence"][:, time_index],
                    temporal_extrinsics_camera_from_world=temporal_pose_sequence[
                        :, time_index
                    ],
                    temporal_pose_valid=pose_valid,
                    contract=temporal_history_v2,
                )
            else:
                assert previous_output is not None and previous_rgb_hr is not None
                hidden_state = _reset_hidden_where_pose_invalid(hidden_state, pose_valid)
                transport = build_temporal_transport(
                    previous_output=previous_output,
                    previous_rgb_hr=previous_rgb_hr,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
                    baseline_current_m=batch["baseline_m_sequence"][:, time_index],
                    temporal_extrinsics_camera_from_world=temporal_pose_sequence[
                        :, time_index
                    ],
                    temporal_pose_valid=pose_valid,
                    scale=int(config.data.scale),
                    photometric_temperature=float(config.train.photometric_temperature),
                    disparity_temperature_hr_px=float(
                        config.train.disparity_temperature_hr_px
                    ),
                    reject_conflict_hr_px=float(config.train.history_conflict_hr_px),
                    photometric_threshold=float(
                        config.train.temporal_photometric_threshold
                    ),
                    geometry_threshold_hr_px=float(
                        config.train.temporal_geometry_threshold_hr_px
                    ),
                )
        static_prior = batch["static_prior_valid_sequence"][:, time_index]
        use_vggt_depth = bool(config.model.get("use_vggt_depth", True))
        if use_vggt_depth:
            valid_vggt = batch["valid_vggt_sequence"][:, time_index] & static_prior.reshape(
                -1, 1, 1, 1
            )
            vggt_disparity = batch["disparity_vggt_hr_px_sequence"][:, time_index]
            vggt_confidence = batch["confidence_vggt_sequence"][:, time_index]
        else:
            # Preserve tensor shapes and the model graph while making the prior
            # genuinely absent.  This is the S2/S3 control needed to isolate the
            # effect of adding VGGT depth in S4.
            vggt_disparity = torch.zeros_like(
                batch["disparity_vggt_hr_px_sequence"][:, time_index]
            )
            vggt_confidence = torch.zeros_like(
                batch["confidence_vggt_sequence"][:, time_index]
            )
            valid_vggt = torch.zeros_like(
                batch["valid_vggt_sequence"][:, time_index], dtype=torch.bool
            )
        model_kwargs: dict[str, Any] = {
            "disparity_vggt_hr_px": vggt_disparity,
            "confidence_vggt": vggt_confidence,
            "valid_vggt": valid_vggt,
            "valid_ffs": step["valid_ffs"],
            "hidden_state": hidden_state,
        }
        model_kwargs.update(
            calibration_model_kwargs_temporal(
                batch,
                time_index=time_index,
                contract=calibration_v3,
                config=config,
            )
        )
        if transport is not None:
            model_kwargs.update(
                {
                    "disparity_history_hr_px": transport.disparity_history_hr_px,
                    "confidence_history": transport.confidence_history,
                    "history_visibility": transport.visibility_mask.to(
                        dtype=step["rgb_hr"].dtype
                    ),
                    "photometric_residual": transport.photometric_residual,
                    "fractional_offset_px": transport.fractional_offset_px,
                    "valid_history": transport.valid_history,
                }
            )
            if temporal_history_v2.enabled:
                topk_values = (
                    transport.topk_disparity_history_hr_px,
                    transport.topk_confidence_history,
                    transport.topk_fractional_offset_px,
                    transport.topk_temporal_age_frames,
                    transport.topk_z_aware_weights,
                    transport.topk_valid_mask,
                )
                if any(value is None for value in topk_values):
                    raise RuntimeError("V2 transport did not populate top-K model inputs")
                model_kwargs.update(
                    {
                        "history_topk_disparity_hr_px": topk_values[0],
                        "history_topk_confidence": topk_values[1],
                        "history_topk_fractional_offset_px": topk_values[2],
                        "history_topk_age_frames": topk_values[3],
                        "history_topk_weights": topk_values[4],
                        "history_topk_valid_mask": topk_values[5],
                    }
                )
                if candidate_v31.enabled:
                    v31_values = (
                        transport.topk_depth_m,
                        transport.topk_pose_quality,
                        transport.topk_depth_layer_index,
                        transport.topk_front_surface_mask,
                        transport.topk_context_only_mask,
                        transport.topk_warped_hidden_feature,
                    )
                    if any(value is None for value in v31_values):
                        raise RuntimeError(
                            "v3.1 transport did not populate candidate metadata"
                        )
                    model_kwargs.update(
                        {
                            "history_topk_depth_m": v31_values[0],
                            "history_topk_pose_quality": v31_values[1],
                            "history_topk_depth_layer_index": v31_values[2],
                            "history_topk_front_surface_mask": v31_values[3],
                            "history_topk_context_only_mask": v31_values[4],
                            "history_topk_warped_hidden_feature": v31_values[5],
                        }
                    )
        output = model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            **model_kwargs,
        )
        if diagnostic:
            finite = torch.stack(
                [
                    torch.isfinite(output.disparity_hr_px).all(),
                    torch.isfinite(output.log_variance).all(),
                    torch.isfinite(output.source_weights).all(),
                ]
            )
            if not bool(finite.all().item()):
                raise FloatingPointError(
                    f"non-finite temporal model output at time index {time_index}"
                )
        losses.append(
            compute_stage_b_step_loss(
                output,
                step,
                transport=transport,
                reference_transport=reference_transport,
                scale=int(config.data.scale),
                weights=weights,
                max_photometric_residual=float(
                    config.train.temporal_photometric_threshold
                ),
                positivity_ablation=positivity_ablation,
                physical_output_v2=physical_v2,
                temporal_residual_v2=temporal_residual_v2,
            )
        )
        if temporal_history_v2.enabled:
            memory.append(
                TemporalMemoryEntry(
                    output=output,
                    rgb_hr=step["rgb_hr"],
                    time_index=time_index,
                    intrinsics_hr=batch["K_hr_sequence"][:, time_index],
                    baseline_m=batch["baseline_m_sequence"][:, time_index],
                )
            )
            memory = memory[-temporal_history_v2.memory_frames :]
            hidden_state = None
        else:
            hidden_state = output.hidden_state
        previous_output = output
        previous_rgb_hr = step["rgb_hr"]
    return average_loss_breakdowns(losses)


def _forward_loss(
    model: FFSOmegaTSR,
    batch: Mapping[str, Any],
    *,
    scale: int,
    weights: LossWeights,
    positivity_ablation: PositivityAblation = PositivityAblation(),
    physical_output_v2: PhysicalOutputV2 = PhysicalOutputV2(),
    calibration_v3: CalibrationConditioningV3 = CalibrationConditioningV3(),
    diagnostic: bool = False,
) -> LossBreakdown:
    output = model(
        batch["rgb_hr"],
        batch["disparity_ffs_hr_px"],
        batch["confidence_ffs"],
        valid_ffs=batch["valid_ffs"],
        **calibration_model_kwargs_spatial(batch, calibration_v3),
    )
    if diagnostic:
        finite = torch.stack(
            [
                torch.isfinite(output.disparity_hr_px).all(),
                torch.isfinite(output.log_variance).all(),
                torch.isfinite(output.source_weights).all(),
            ]
        )
        if not bool(finite.all().item()):
            raise FloatingPointError("Stage-A model output is non-finite")
    breakdown = compute_stage_a_loss(
        output,
        batch,
        scale=scale,
        weights=weights,
        positivity_ablation=positivity_ablation,
        physical_output_v2=physical_output_v2,
    )
    if diagnostic and not bool(torch.isfinite(breakdown.total.detach()).item()):
        raise FloatingPointError("Stage-A total loss is non-finite")
    return breakdown


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return device


def _infinite_batches(
    loader: DataLoader[dict[str, Any]],
    dataset: Any,
    sampler: DeterministicEpochSampler,
    *,
    start_micro_step: int = 0,
) -> Iterator[dict[str, Any]]:
    start_micro_step = _nonnegative_int(start_micro_step, "start_micro_step")
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("DataLoader has no complete micro-batches")
    epoch, skip_batches = divmod(start_micro_step, batches_per_epoch)
    while True:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        yielded = False
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            yielded = True
            yield batch
        if not yielded:
            # A resume exactly at an epoch boundary legitimately skips none;
            # any other empty pass indicates a malformed offset/loader.
            if skip_batches:
                raise RuntimeError("resume batch offset exhausted the DataLoader")
            raise RuntimeError("DataLoader yielded no training batches")
        skip_batches = 0
        epoch += 1


def _append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def build_run_summary(
    *,
    stage: str,
    completed_steps: int,
    run_steps: int,
    elapsed_seconds: float,
    device: torch.device,
    git_hash: str,
    resolved_config: Mapping[str, Any],
    final_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Build the machine-readable receipt for one completed training run."""

    if stage not in {"spatial", "temporal"}:
        raise ValueError("summary stage must be spatial or temporal")
    completed_steps = _positive_int(completed_steps, "completed_steps")
    run_steps = _positive_int(run_steps, "run_steps")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    checkpoint_path = Path(final_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"final checkpoint is missing: {checkpoint_path}")
    canonical_config = config_fingerprint(resolved_config)
    is_cuda = device.type == "cuda"
    return {
        "stage": stage,
        "status": "TRAINING_COMPLETE",
        "steps": completed_steps,
        "run_steps": run_steps,
        "elapsed_seconds": float(elapsed_seconds),
        "steps_per_second": float(run_steps / elapsed_seconds),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if is_cuda else None,
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "git_hash": str(git_hash),
        "config_fingerprint": hashlib.sha256(
            canonical_config.encode("utf-8")
        ).hexdigest(),
        "final_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if is_cuda else None
        ),
    }


def write_run_summary_atomic(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Atomically write a strict-JSON training receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(summary),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _populate_temporal_initialization_config(
    config: DictConfig, args: argparse.Namespace
) -> Path | None:
    """Bind a new Stage-B run to Stage A, or recover that binding on resume."""

    if args.init_from is not None:
        path = args.init_from.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"initialization checkpoint is missing: {path}")
        OmegaConf.update(config, "train.initialization_checkpoint", str(path), merge=False)
        OmegaConf.update(
            config,
            "train.initialization_checkpoint_sha256",
            sha256_file(path),
            merge=False,
        )
        return path
    if args.resume is None:
        raise ValueError("a new Stage-B run requires --init-from STAGE_A_CHECKPOINT")
    resume_path = args.resume.expanduser().resolve()
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    saved_config = payload.get("config") if isinstance(payload, Mapping) else None
    saved_train = saved_config.get("train") if isinstance(saved_config, Mapping) else None
    if not isinstance(saved_train, Mapping):
        raise ValueError("resume checkpoint has no temporal training config")
    initialization_path = saved_train.get("initialization_checkpoint")
    initialization_sha256 = saved_train.get("initialization_checkpoint_sha256")
    if not initialization_path or not initialization_sha256:
        raise ValueError("resume checkpoint has no Stage-A initialization lineage")
    OmegaConf.update(
        config,
        "train.initialization_checkpoint",
        str(initialization_path),
        merge=False,
    )
    OmegaConf.update(
        config,
        "train.initialization_checkpoint_sha256",
        str(initialization_sha256),
        merge=False,
    )
    return None


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args.config, args.overrides)
    cli_values = {
        "data.manifest_path": args.manifest,
        "data.observation_cache_root": args.observation_cache_root,
        "data.teacher_cache_root": args.teacher_cache_root,
        "data.derived_geometry_cache_root": args.derived_cache_root,
        "data.calibration_sidecar_path": getattr(args, "calibration_sidecar", None),
        "train.output_dir": args.output_dir,
    }
    for key, value in cli_values.items():
        if value is not None:
            OmegaConf.update(config, key, str(value.expanduser().resolve()), merge=False)
    stage = training_stage(config)
    initialization_path: Path | None = None
    if stage == "temporal":
        initialization_path = _populate_temporal_initialization_config(config, args)
    elif args.init_from is not None or args.derived_cache_root is not None:
        raise ValueError("--init-from and --derived-cache-root are Stage-B-only")
    validate_training_config(config)
    seed_everything(int(config.seed), deterministic=True)

    if stage == "spatial":
        dataset, observation_identity, teacher_identity = build_dataset_and_identities(
            config
        )
        collate_function = collate_training_samples
        total_steps = int(config.train.steps_spatial)
        derived_cache_lineage: Mapping[str, Any] | None = None
    else:
        dataset, observation_identity, teacher_identity = (
            build_temporal_dataset_and_identities(config)
        )
        collate_function = collate_temporal_training_samples
        total_steps = int(config.train.steps)
        derived_cache_lineage = dataset.cache_lineage_summary
    calibration_index = dataset.spatial_dataset.rectified_calibration_index if isinstance(
        dataset, CachedTemporalTrainingDataset
    ) else dataset.rectified_calibration_index
    calibration_sidecar_lineage = (
        None
        if calibration_index is None
        else {
            "component": "rectified-stereo-calibration",
            "contract_version": "stored_rectified_virtual_cameras_v1",
            "sidecar_path": str(calibration_index.sidecar_path),
            "sidecar_sha256": calibration_index.sidecar_sha256,
            "receipt_path": str(calibration_index.receipt_path),
            "receipt_sha256": calibration_index.receipt_sha256,
            "source_manifest_sha256": calibration_index.source_manifest_sha256,
            "pixel_audit_sha256": calibration_index.pixel_audit_sha256,
            "spring_native": bool(calibration_index.spring_native),
        }
    )
    OmegaConf.update(
        config,
        "data.observation_cache_identity",
        observation_identity.to_dict(),
        merge=False,
    )
    OmegaConf.update(
        config,
        "data.teacher_cache_identity",
        teacher_identity.to_dict(),
        merge=False,
    )
    OmegaConf.update(
        config,
        "data.derived_cache_lineage",
        derived_cache_lineage,
        merge=False,
    )
    OmegaConf.update(
        config,
        "data.calibration_sidecar_lineage",
        calibration_sidecar_lineage,
        merge=False,
    )
    workers = int(config.train.num_workers)
    persistent_workers = bool(config.train.persistent_workers) and workers > 0
    sampler = DeterministicEpochSampler(len(dataset), seed=int(config.seed))
    loader_generator = torch.Generator().manual_seed(int(config.seed))
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset,
        batch_size=int(config.train.micro_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(config.train.pin_memory),
        persistent_workers=persistent_workers,
        collate_fn=collate_function,
        worker_init_fn=seed_data_worker,
        generator=loader_generator,
        drop_last=True,
    )
    device = _resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is unavailable on {torch.cuda.get_device_name(device)}")
    model = build_model(config).to(device)
    parameter_count = count_trainable_parameters(model)
    calibration_v3 = calibration_conditioning_v3_from_config(config)
    initialization_lineage: Mapping[str, Any] | None = None
    if stage == "temporal" and args.resume is None:
        assert initialization_path is not None
        required_stage_a_v3 = (
            {
                "enabled": True,
                "protocol_version": CALIBRATION_CONDITIONING_V3_PROTOCOL,
                **(
                    {
                        "pixel_center_contract": (
                            ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT
                        )
                    }
                    if calibration_v3.align_corners_false_pixel_centers
                    else {}
                ),
                "use_rays": True,
                "use_stereo_pose": True,
                "use_temporal_pose": False,
            }
            if calibration_v3.enabled
            else None
        )
        required_v31_sections = (
            {
                "measurement_ownership_v3_1": dict(
                    config.measurement_ownership_v3_1
                ),
                "temporal_candidate_fusion_v3_1": dict(
                    config.temporal_candidate_fusion_v3_1
                ),
            }
            if calibration_v3.align_corners_false_pixel_centers
            else None
        )
        initialization_lineage = load_model_initialization_checkpoint(
            initialization_path,
            model=model,
            expected_parameter_count=parameter_count,
            required_sequence_length=1,
            required_seed=int(config.seed) if calibration_v3.enabled else None,
            required_calibration_conditioning_v3=required_stage_a_v3,
            required_config_sections=required_v31_sections,
        )
        if (
            initialization_lineage["checkpoint_sha256"]
            != str(config.train.initialization_checkpoint_sha256)
        ):
            raise ValueError("Stage-A checkpoint changed while the run was being built")
    weights = loss_weights_from_config(config)
    positivity_ablation = positivity_ablation_from_config(config)
    physical_v2 = physical_output_v2_from_config(config)
    warmup_steps = int(config.train.warmup_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update_index: learning_rate_multiplier(
            update_index, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    resolved_config = _resolved_container(config)
    if args.dry_run:
        batch_iterator = _infinite_batches(loader, dataset, sampler)
        model.eval()
        batch = _move_training_batch(next(batch_iterator), device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=str(config.train.precision).lower() == "bf16",
        ):
            if stage == "spatial":
                breakdown = _forward_loss(
                    model,
                    batch,
                    scale=int(config.data.scale),
                    weights=weights,
                    positivity_ablation=positivity_ablation,
                    physical_output_v2=physical_v2,
                    calibration_v3=calibration_v3,
                    diagnostic=True,
                )
            else:
                breakdown = _forward_temporal_loss(
                    model,
                    batch,
                    config=config,
                    weights=weights,
                    positivity_ablation=positivity_ablation,
                    diagnostic=True,
                )
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "stage": stage,
                    "device": str(device),
                    "parameter_count": parameter_count,
                    "dataset_windows": len(dataset),
                    "observation_identity": observation_identity.to_dict(),
                    "teacher_identity": teacher_identity.to_dict(),
                    "derived_cache_lineage": derived_cache_lineage,
                    "calibration_sidecar_lineage": calibration_sidecar_lineage,
                    "initialization_lineage": initialization_lineage,
                    "loss": breakdown.detached_scalars(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_value = config.train.output_dir
    output_dir = (
        PROJECT_ROOT / "outputs" / str(config.experiment)
        if output_value is None or not str(output_value).strip()
        else Path(str(output_value)).expanduser().resolve()
    )
    latest_path = output_dir / "latest.pt"
    final_path = output_dir / "final.pt"
    log_path = output_dir / "train.jsonl"
    summary_path = output_dir / "run_summary.json"
    if args.resume is None and (latest_path.exists() or final_path.exists()):
        raise FileExistsError(
            f"output already contains a checkpoint; pass --resume or choose another directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_step = 0
    if args.resume is not None:
        completed_step = load_training_checkpoint(
            args.resume.expanduser().resolve(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=resolved_config,
            expected_parameter_count=parameter_count,
            scaler=None,
            restore_rng=True,
        )
        if completed_step >= total_steps:
            raise ValueError(
                f"checkpoint already completed step {completed_step} of {total_steps}"
            )

    starting_step = completed_step
    accumulation = int(config.train.grad_accumulation)
    batch_iterator = _infinite_batches(
        loader,
        dataset,
        sampler,
        start_micro_step=completed_step * accumulation,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    repository_hash = repository_git_hash(PROJECT_ROOT)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log_mode = "a" if args.resume is not None else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        while completed_step < total_steps:
            summed_terms: dict[str, Tensor] = {}
            for accumulation_index in range(accumulation):
                batch = _move_training_batch(next(batch_iterator), device)
                diagnostic_interval = int(config.train.finite_diagnostic_interval)
                diagnostic = diagnostic_interval > 0 and (
                    completed_step + 1
                ) % diagnostic_interval == 0 and accumulation_index == 0
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=str(config.train.precision).lower() == "bf16",
                ):
                    if stage == "spatial":
                        breakdown = _forward_loss(
                            model,
                            batch,
                            scale=int(config.data.scale),
                            weights=weights,
                            positivity_ablation=positivity_ablation,
                            physical_output_v2=physical_v2,
                            calibration_v3=calibration_v3,
                            diagnostic=diagnostic,
                        )
                    else:
                        breakdown = _forward_temporal_loss(
                            model,
                            batch,
                            config=config,
                            weights=weights,
                            positivity_ablation=positivity_ablation,
                            diagnostic=diagnostic,
                        )
                    scaled_loss = breakdown.total / float(accumulation)
                scaled_loss.backward()
                term_names = [
                    "total",
                    "disparity",
                    "measurement",
                    "gradient",
                    "temporal",
                    "epipolar",
                    "uncertainty_nll",
                    "gate_regularizer",
                ]
                if breakdown.positivity_penalty is not None:
                    term_names.append("positivity_penalty")
                for optional_name in (
                    "valid_bce",
                    "completion_bce",
                    "validity_calibration",
                ):
                    if getattr(breakdown, optional_name) is not None:
                        term_names.append(optional_name)
                for name in term_names:
                    value = getattr(breakdown, name).detach()
                    summed_terms[name] = summed_terms.get(name, value.new_zeros(())) + value
                if not should_optimizer_step(accumulation_index + 1, accumulation):
                    continue

            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(config.train.gradient_clip), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step += 1

            if completed_step % int(config.train.log_interval) == 0:
                term_names = sorted(summed_terms)
                logged_values = (
                    torch.cat(
                        (
                            torch.stack(
                                [summed_terms[name] for name in term_names]
                            ).div(float(accumulation)),
                            gradient_norm.detach().reshape(1),
                        )
                    )
                    .cpu()
                    .tolist()
                )
                term_values = logged_values[:-1]
                record = {
                    "step": completed_step,
                    "stage": stage,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": float(logged_values[-1]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "loss": dict(zip(term_names, term_values, strict=True)),
                }
                _append_jsonl(log_handle, record)
                print(json.dumps(record, sort_keys=True), flush=True)

            if completed_step % int(config.train.checkpoint_interval) == 0:
                save_training_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_step=completed_step,
                    config=resolved_config,
                    git_hash=repository_hash,
                    parameter_count=parameter_count,
                    scaler=None,
                )

    save_training_checkpoint(
        latest_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=completed_step,
        config=resolved_config,
        git_hash=repository_hash,
        parameter_count=parameter_count,
        scaler=None,
    )
    save_training_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=completed_step,
        config=resolved_config,
        git_hash=repository_hash,
        parameter_count=parameter_count,
        scaler=None,
    )
    elapsed_seconds = time.perf_counter() - started
    summary = build_run_summary(
        stage=stage,
        completed_steps=completed_step,
        run_steps=completed_step - starting_step,
        elapsed_seconds=elapsed_seconds,
        device=device,
        git_hash=repository_hash,
        resolved_config=resolved_config,
        final_checkpoint_path=final_path,
    )
    write_run_summary_atomic(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE",
                "stage": stage,
                "step": completed_step,
                "parameter_count": parameter_count,
                "final_checkpoint": str(final_path),
                "run_summary": str(summary_path),
                "elapsed_seconds": summary["elapsed_seconds"],
                "steps_per_second": summary["steps_per_second"],
                "peak_cuda_allocated_bytes": summary[
                    "peak_cuda_allocated_bytes"
                ],
                "peak_cuda_reserved_bytes": summary[
                    "peak_cuda_reserved_bytes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
