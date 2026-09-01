#!/usr/bin/env python3
"""Evaluate Stage-A T=1 and Stage-B causal T=3 disparity reconstruction.

Targets are explicitly trusted HR FFS teacher pseudo-GT. Stage B scores only
the endpoint of strict three-frame causal windows and reports temporal/VGGT
ablations without claiming epipolar refinement, paper GT, or paper accuracy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import sha256_file  # noqa: E402
from data.collate import (  # noqa: E402
    collate_temporal_training_samples,
    collate_training_samples,
)
from data.manifest import load_manifest  # noqa: E402
from data.training_dataset import (  # noqa: E402
    CachedFFSTrainingDataset,
    build_causal_windows,
)
from data.temporal_training_dataset import CachedTemporalTrainingDataset  # noqa: E402
from evaluation import (  # noqa: E402
    MethodMetricAccumulator,
    POINT_TO_PLANE_NOT_AVAILABLE,
    PSEUDO_GT_LABEL,
    aggregate_metric_change,
    comparison_from_aggregates,
    compute_sample_metrics,
    hr_temporal_metric,
    hr_temporal_residual_metric,
    hr_temporal_safe_mask,
    load_model_for_evaluation,
    physical_disparity_clamp_min_zero,
    upsample_ffs_inputs_to_hr,
    validate_checkpoint_lineage,
    validate_spatial_checkpoint_binding,
    validate_temporal_batch_causality,
)
from models.ffs_omega_tsr import ModelOutput, count_trainable_parameters  # noqa: E402
from metrics.disparity import MetricResult  # noqa: E402
from metrics.pointcloud import export_colored_point_cloud_ply  # noqa: E402
from metrics.temporal import (  # noqa: E402
    temporal_disparity_error,
    temporal_residual_error,
)
from train import (  # noqa: E402
    DEFAULT_CONFIG,
    ReferenceTemporalWarp,
    TemporalMemoryEntry,
    TemporalTransport,
    _reset_hidden_where_pose_invalid,
    _temporal_step_batch,
    build_model,
    build_reference_temporal_warp,
    build_topk_temporal_transport,
    build_temporal_transport,
    load_receipt_identity,
    physical_output_v2_from_config,
    temporal_history_v2_from_config,
    temporal_residual_v2_from_config,
    validate_v2_temporal_calibration,
)
from utils.checkpoint import repository_git_hash  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from utils.visualization import (  # noqa: E402
    grayscale_to_rgb_uint8,
    save_rgb_uint8,
    scalar_to_rgb_uint8,
)


EVALUATION_DEFAULTS: dict[str, Any] = {
    "eval": {
        "output_dir": None,
        "crop_mode": "fixed",
        "fixed_crop_origin_hr_xy": None,
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "precision": "bf16",
        "limit": None,
        "start": 0,
        "visualization_samples": 4,
        # Opt-in only: temporal flicker video is a non-metric visualization.
        # Keep it disabled so ordinary evaluation has identical behavior and
        # does not require optional imageio/FFmpeg dependencies.
        "temporal_flicker_video": False,
        "temporal_flicker_video_fps": 5,
        "temporal_flicker_disparity_range_hr_px": [0.0, 384.0],
        "temporal_flicker_error_range_hr_px": [0.0, 20.0],
        "temporal_flicker_uncertainty_range": [0.0, 10.0],
        # Default-off worst-case bundle extraction.  CPU tensor retention is
        # bounded globally across every criterion while evaluation is running.
        "failure_samples_per_criterion": 0,
        "failure_samples_cpu_limit_bytes": 536_870_912,
        "low_confidence_threshold": 0.8,
        "boundary_gradient_threshold_px": 1.0,
        "boundary_radius_px": 1,
    }
}

FORMAL_STAGE_A_TRAINING_STEPS = 5_000
FORMAL_STAGE_B_TRAINING_STEPS = 15_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bilinear/T1 or causal T3/VGGT methods against trusted HR "
            "FFS teacher pseudo-GT."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Stage-A T1 or Stage-B T3 training checkpoint, matching the config",
    )
    parser.add_argument(
        "--spatial-checkpoint",
        type=Path,
        help=(
            "T1 checkpoint for Stage-B comparison; defaults to the exact Stage-A "
            "initialization path recorded by the temporal checkpoint"
        ),
    )
    parser.add_argument("--manifest", type=Path, help="explicit validation JSONL manifest")
    parser.add_argument(
        "--observation-cache-root",
        type=Path,
        help="validation FFS observation cache root",
    )
    parser.add_argument(
        "--teacher-cache-root",
        type=Path,
        help="validation HR FFS teacher cache root",
    )
    parser.add_argument(
        "--derived-cache-root",
        type=Path,
        help="validation vggt-ffs-derived-geometry cache root for T=3",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="directory for metrics.json, metrics.csv, and visualizations",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, help="evaluation batch size")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count")
    parser.add_argument("--limit", type=int, help="evaluate the first N records")
    parser.add_argument(
        "--start", type=int, help="zero-based deterministic record/window start"
    )
    parser.add_argument(
        "--visualization-samples",
        type=int,
        help="number of leading samples to visualize",
    )
    parser.add_argument(
        "--temporal-flicker-video",
        action="store_true",
        default=None,
        help=(
            "opt in to non-metric causal T3 temporal flicker MP4 visualizations "
            "(uses imageio/FFmpeg or a direct ffmpeg executable)"
        ),
    )
    parser.add_argument(
        "--temporal-flicker-video-fps",
        type=int,
        help="frames per second for opt-in temporal flicker MP4s",
    )
    parser.add_argument(
        "--failure-samples-per-criterion",
        type=int,
        help=(
            "opt in to deterministic T3 failure bundles per criterion; "
            "0 keeps the evaluator's normal visualization path unchanged"
        ),
    )
    parser.add_argument(
        "--crop-mode",
        choices=("fixed", "full"),
        help="fixed center/origin crop, or full-resolution evaluation",
    )
    parser.add_argument(
        "--crop-origin",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="scale-aligned HR origin for fixed crops",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config and construct the model without loading data/checkpoint",
    )
    parser.add_argument(
        "--allow-non-holdout-smoke",
        action="store_true",
        help=(
            "allow a limited T3 pipeline smoke whose checkpoint trained on the "
            "same validation artifacts; report is forcibly marked non-formal"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. eval.limit=8 data.hr_crop=[384,768]",
    )
    return parser


def _load_yaml_with_inheritance(path: Path, seen: set[Path] | None = None) -> DictConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"config does not exist: {resolved}")
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic defaults_from chain at {resolved}")
    seen.add(resolved)
    loaded = OmegaConf.load(resolved)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"config must resolve to a mapping: {resolved}")
    inherited_name = loaded.get("defaults_from")
    if inherited_name is None:
        return loaded
    del loaded["defaults_from"]
    inherited_path = Path(str(inherited_name)).expanduser()
    if not inherited_path.is_absolute():
        project_candidate = PROJECT_ROOT / inherited_path
        inherited_path = (
            project_candidate
            if project_candidate.exists()
            else resolved.parent / inherited_path
        )
    return OmegaConf.merge(
        _load_yaml_with_inheritance(inherited_path, seen), loaded
    )


def resolve_evaluation_config(
    config_path: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> DictConfig:
    """Resolve training/model defaults plus struct-checked evaluation options."""

    config = OmegaConf.merge(
        OmegaConf.create(DEFAULT_CONFIG),
        OmegaConf.create(EVALUATION_DEFAULTS),
        _load_yaml_with_inheritance(Path(config_path)),
    )
    OmegaConf.set_struct(config, True)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def _update_cli_values(config: DictConfig, args: argparse.Namespace) -> None:
    values = {
        "data.manifest_path": args.manifest,
        "data.observation_cache_root": args.observation_cache_root,
        "data.teacher_cache_root": args.teacher_cache_root,
        "data.derived_geometry_cache_root": args.derived_cache_root,
        "eval.output_dir": args.output_dir,
        "eval.batch_size": args.batch_size,
        "eval.num_workers": args.num_workers,
        "eval.limit": args.limit,
        "eval.start": args.start,
        "eval.visualization_samples": args.visualization_samples,
        "eval.temporal_flicker_video": args.temporal_flicker_video,
        "eval.temporal_flicker_video_fps": args.temporal_flicker_video_fps,
        "eval.failure_samples_per_criterion": args.failure_samples_per_criterion,
        "eval.crop_mode": args.crop_mode,
        "eval.fixed_crop_origin_hr_xy": args.crop_origin,
    }
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value.expanduser().resolve())
        elif isinstance(value, tuple):
            value = list(value)
        OmegaConf.update(config, key, value, merge=False)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fixed_display_range(value: Any, name: str) -> tuple[float, float]:
    """Validate an explicit, fixed scalar visualization range."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be [minimum, maximum]")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be [minimum, maximum]") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must be [minimum, maximum]")
    minimum, maximum = (float(item) for item in values)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"{name} must be finite with maximum > minimum")
    return minimum, maximum


def validate_evaluation_config(config: DictConfig) -> str:
    """Validate x2 deterministic evaluation and return ``spatial``/``temporal``."""

    history_v2 = temporal_history_v2_from_config(config)
    residual_v2 = temporal_residual_v2_from_config(config)
    if history_v2.enabled != residual_v2.enabled:
        raise ValueError(
            "temporal_history_v2 and temporal_residual_v2 must be enabled together"
        )
    sequence_length = int(config.data.sequence_length)
    if sequence_length == 1:
        stage = "spatial"
    elif sequence_length == 3:
        stage = "temporal"
        if int(config.data.vggt_context_pairs) != 5:
            raise ValueError("Stage-B evaluation requires five VGGT context pairs")
        if not bool(config.vggt.causal):
            raise ValueError("Stage-B evaluation forbids future VGGT frames")
        if not bool(config.model.use_history) or not bool(
            config.model.use_vggt_pose
        ):
            raise ValueError("Stage-B evaluation requires history and VGGT pose")
        if bool(config.model.epipolar_refinement):
            raise ValueError("Stage-B evaluation must not claim Stage-C epipolar output")
        if history_v2.enabled and history_v2.top_k < 2:
            raise ValueError("temporal_history_v2 evaluation requires top_k >= 2")
    else:
        raise ValueError("evaluation supports only T=1 spatial or causal T=3")
    if stage != "temporal" and (history_v2.enabled or residual_v2.enabled):
        raise ValueError("temporal v2 evaluation requires causal T=3")
    if int(config.data.scale) != 2 or int(config.model.convex_scale) != 2:
        raise ValueError("first-round evaluation is fixed to x2")
    if list(config.model.rgb_channels) != [32, 64, 96]:
        raise ValueError("model.rgb_channels must be [32,64,96]")
    if str(config.eval.crop_mode) not in {"fixed", "full"}:
        raise ValueError("eval.crop_mode must be fixed or full")
    crop = list(config.data.hr_crop)
    if len(crop) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in crop
    ):
        raise ValueError("data.hr_crop must be [height,width] positive integers")
    if any(value % 2 for value in crop):
        raise ValueError("data.hr_crop dimensions must be divisible by x2")
    origin = config.eval.fixed_crop_origin_hr_xy
    if origin is not None:
        values = list(origin)
        if len(values) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("eval.fixed_crop_origin_hr_xy must be [x,y] non-negative")
        if any(value % 2 for value in values):
            raise ValueError("fixed evaluation crop origin must be x2-aligned")
        if str(config.eval.crop_mode) != "fixed":
            raise ValueError("a fixed crop origin requires eval.crop_mode=fixed")
    _positive_int(config.eval.batch_size, "eval.batch_size")
    _nonnegative_int(config.eval.num_workers, "eval.num_workers")
    _nonnegative_int(config.eval.visualization_samples, "eval.visualization_samples")
    if not isinstance(config.eval.temporal_flicker_video, bool):
        raise ValueError("eval.temporal_flicker_video must be boolean")
    if config.eval.temporal_flicker_video and stage != "temporal":
        raise ValueError("eval.temporal_flicker_video requires causal T=3 evaluation")
    _positive_int(
        config.eval.temporal_flicker_video_fps,
        "eval.temporal_flicker_video_fps",
    )
    failure_samples = _nonnegative_int(
        config.eval.failure_samples_per_criterion,
        "eval.failure_samples_per_criterion",
    )
    _positive_int(
        config.eval.failure_samples_cpu_limit_bytes,
        "eval.failure_samples_cpu_limit_bytes",
    )
    if failure_samples and stage != "temporal":
        raise ValueError("failure sample bundles require causal T=3 evaluation")
    _fixed_display_range(
        config.eval.temporal_flicker_disparity_range_hr_px,
        "eval.temporal_flicker_disparity_range_hr_px",
    )
    _fixed_display_range(
        config.eval.temporal_flicker_error_range_hr_px,
        "eval.temporal_flicker_error_range_hr_px",
    )
    _fixed_display_range(
        config.eval.temporal_flicker_uncertainty_range,
        "eval.temporal_flicker_uncertainty_range",
    )
    if config.eval.limit is not None:
        _positive_int(config.eval.limit, "eval.limit")
    _nonnegative_int(config.eval.start, "eval.start")
    threshold = float(config.eval.low_confidence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("eval.low_confidence_threshold must be in [0,1]")
    if float(config.eval.boundary_gradient_threshold_px) < 0.0:
        raise ValueError("boundary gradient threshold must be non-negative")
    _nonnegative_int(config.eval.boundary_radius_px, "eval.boundary_radius_px")
    if str(config.eval.precision).lower() not in {"bf16", "fp32"}:
        raise ValueError("eval.precision must be bf16 or fp32")
    return stage


def _required_path(config: DictConfig, key: str, *, directory: bool) -> Path:
    value = OmegaConf.select(config, key)
    if value is None or not str(value).strip():
        raise ValueError(
            f"{key} is required and must identify the validation artifact explicitly"
        )
    path = Path(str(value)).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{key} {kind} does not exist: {path}")
    return path


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return device


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


@dataclass(frozen=True, slots=True)
class TemporalEndpointPredictions:
    """Endpoint outputs from causal T=3 VGGT interventions."""

    vggt_on: ModelOutput
    source_mask_off: ModelOutput
    no_vggt: ModelOutput
    shared_transport: TemporalTransport
    source_mask_off_transport: TemporalTransport
    no_vggt_transport: TemporalTransport
    reference_transport: ReferenceTemporalWarp | None
    source_mask_off_reference_transport: ReferenceTemporalWarp | None
    no_vggt_reference_transport: ReferenceTemporalWarp | None


@dataclass(frozen=True, slots=True)
class SpatialEndpointPrediction:
    """T1 endpoint plus its independently measured temporal transport."""

    output: ModelOutput
    transport: TemporalTransport
    reference_transport: ReferenceTemporalWarp | None


# These names deliberately describe operations rather than implementation
# modules.  They form the stable machine-readable sign-health contract for the
# primary causal T3+VGGT endpoint.
T3_VGGT_SIGN_HEALTH_STAGES: tuple[tuple[str, str, str], ...] = (
    (
        "source_mix_lr",
        "disparity_source_mix_hr_px_lr_grid",
        "LR",
    ),
    (
        "post_lr_residual",
        "disparity_post_lr_residual_hr_px_lr_grid",
        "LR",
    ),
    ("post_convex", "disparity_post_convex_hr_px", "HR"),
    ("post_hr_residual_raw", "disparity_raw_hr_px", "HR"),
    ("post_anchor_final", "disparity_hr_px", "HR"),
)

T3_VGGT_SIGN_HEALTH_STRATA: tuple[str, ...] = (
    "all_pixels",
    "bilinear_disparity_lt_0_hr_px",
    "bilinear_disparity_ge_0_lt_0_25_hr_px",
    "bilinear_disparity_ge_0_25_hr_px",
    "bilinear_disparity_nonfinite",
    "ffs_valid",
    "ffs_invalid",
    "history_valid",
    "history_invalid",
    "pose_valid",
    "pose_invalid",
)


class T3VGGTSignHealthAccumulator:
    """Accumulate native-grid sign diagnostics without changing predictions.

    Every stage is counted over its own complete native grid.  No teacher,
    pseudo-GT, confidence, or trusted mask is applied.  The masks supplied to
    :meth:`update` are the exact primary ``T3_VGGT`` endpoint masks used by the
    evaluator.  This class is intentionally evaluation-only and never alters a
    tensor consumed by a metric or a later model step.
    """

    _COUNT_NAMES = (
        "diagnostic_domain_count",
        "finite_count",
        "negative_count",
        "nonfinite_count",
    )

    def __init__(self) -> None:
        self.records_accumulated = 0
        self._counts = {
            stage_name: {
                stratum: {count_name: 0 for count_name in self._COUNT_NAMES}
                for stratum in T3_VGGT_SIGN_HEALTH_STRATA
            }
            for stage_name, _, _ in T3_VGGT_SIGN_HEALTH_STAGES
        }

    @staticmethod
    def _check_mask(name: str, mask: Tensor, reference: Tensor) -> Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape != reference.shape:
            raise ValueError(
                f"{name} must have shape {tuple(reference.shape)}, got {tuple(mask.shape)}"
            )
        return mask.to(device=reference.device, dtype=torch.bool)

    @staticmethod
    def _resize_mask(mask: Tensor, size_hw: tuple[int, int]) -> Tensor:
        if tuple(mask.shape[-2:]) == size_hw:
            return mask.to(dtype=torch.bool)
        return functional.interpolate(
            mask.to(dtype=torch.float32), size=size_hw, mode="nearest"
        ).to(dtype=torch.bool)

    @staticmethod
    def _stage_tensors(output: ModelOutput) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for stage_name, field_name, _ in T3_VGGT_SIGN_HEALTH_STAGES:
            value = getattr(output, field_name)
            if not isinstance(value, Tensor):
                raise RuntimeError(
                    "T3_VGGT sign-health requires a real FFSOmegaTSR output; "
                    f"diagnostic field {field_name!r} is unavailable"
                )
            if value.ndim != 4 or value.shape[1] != 1 or not value.is_floating_point():
                raise ValueError(
                    f"{field_name} must be floating [B,1,H,W], got {tuple(value.shape)}"
                )
            tensors[stage_name] = value.float()
        return tensors

    def update(
        self,
        output: ModelOutput,
        *,
        bilinear_disparity_hr_px: Tensor,
        ffs_valid_lr: Tensor,
        ffs_valid_hr: Tensor,
        history_valid_lr: Tensor,
        history_valid_hr: Tensor,
        pose_valid: Tensor,
    ) -> None:
        """Add one endpoint batch to every operation-stage/stratum counter."""

        stages = self._stage_tensors(output)
        final = stages["post_anchor_final"]
        if (
            bilinear_disparity_hr_px.ndim != 4
            or bilinear_disparity_hr_px.shape[1] != 1
            or bilinear_disparity_hr_px.shape[0] != final.shape[0]
        ):
            raise ValueError("bilinear_disparity_hr_px must be [B,1,H_hr,W_hr]")
        bilinear_disparity_hr_px = bilinear_disparity_hr_px.float()
        ffs_valid_lr = self._check_mask(
            "ffs_valid_lr", ffs_valid_lr, stages["source_mix_lr"]
        )
        ffs_valid_hr = self._check_mask("ffs_valid_hr", ffs_valid_hr, final)
        history_valid_lr = self._check_mask(
            "history_valid_lr", history_valid_lr, stages["source_mix_lr"]
        )
        history_valid_hr = self._check_mask(
            "history_valid_hr", history_valid_hr, final
        )
        if pose_valid.ndim != 1 or pose_valid.shape[0] != final.shape[0]:
            raise ValueError(
                f"pose_valid must have shape [{final.shape[0]}], got {tuple(pose_valid.shape)}"
            )
        pose_valid = pose_valid.to(device=final.device, dtype=torch.bool)

        for stage_name, _, grid in T3_VGGT_SIGN_HEALTH_STAGES:
            value = stages[stage_name]
            size_hw = tuple(int(item) for item in value.shape[-2:])
            if value.shape[0] != final.shape[0]:
                raise ValueError("all sign-health stages must have the same batch size")
            bilinear_native = functional.interpolate(
                bilinear_disparity_hr_px,
                size=size_hw,
                mode="bilinear",
                align_corners=False,
            )
            if grid == "LR":
                ffs_valid_native = self._resize_mask(ffs_valid_lr, size_hw)
                history_valid_native = self._resize_mask(history_valid_lr, size_hw)
            else:
                ffs_valid_native = self._resize_mask(ffs_valid_hr, size_hw)
                history_valid_native = self._resize_mask(history_valid_hr, size_hw)
            pose_valid_native = pose_valid.reshape(-1, 1, 1, 1).expand_as(value)
            bilinear_finite = torch.isfinite(bilinear_native)
            strata = {
                "all_pixels": torch.ones_like(value, dtype=torch.bool),
                "bilinear_disparity_lt_0_hr_px": (
                    bilinear_finite & (bilinear_native < 0.0)
                ),
                "bilinear_disparity_ge_0_lt_0_25_hr_px": (
                    bilinear_finite
                    & (bilinear_native >= 0.0)
                    & (bilinear_native < 0.25)
                ),
                "bilinear_disparity_ge_0_25_hr_px": (
                    bilinear_finite & (bilinear_native >= 0.25)
                ),
                "bilinear_disparity_nonfinite": ~bilinear_finite,
                "ffs_valid": ffs_valid_native,
                "ffs_invalid": ~ffs_valid_native,
                "history_valid": history_valid_native,
                "history_invalid": ~history_valid_native,
                "pose_valid": pose_valid_native,
                "pose_invalid": ~pose_valid_native,
            }
            finite = torch.isfinite(value)
            negative = finite & (value < 0.0)
            domain_stack = torch.stack(
                [strata[name] for name in T3_VGGT_SIGN_HEALTH_STRATA], dim=0
            )
            reduction_dims = tuple(range(1, domain_stack.ndim))
            batch_counts = torch.stack(
                (
                    domain_stack.sum(dim=reduction_dims),
                    (domain_stack & finite.unsqueeze(0)).sum(dim=reduction_dims),
                    (domain_stack & negative.unsqueeze(0)).sum(dim=reduction_dims),
                    (domain_stack & ~finite.unsqueeze(0)).sum(dim=reduction_dims),
                ),
                dim=1,
            ).detach().cpu().tolist()
            for stratum, values in zip(
                T3_VGGT_SIGN_HEALTH_STRATA, batch_counts, strict=True
            ):
                destination = self._counts[stage_name][stratum]
                for count_name, count in zip(self._COUNT_NAMES, values, strict=True):
                    destination[count_name] += int(count)
        self.records_accumulated += int(final.shape[0])

    @staticmethod
    def _finalize_counts(counts: dict[str, int]) -> dict[str, int | float | None]:
        domain_count = counts["diagnostic_domain_count"]
        finite_count = counts["finite_count"]
        negative_count = counts["negative_count"]
        nonfinite_count = counts["nonfinite_count"]
        if finite_count + nonfinite_count != domain_count:
            raise RuntimeError("sign-health finite/non-finite counts do not partition domain")
        if negative_count > finite_count:
            raise RuntimeError("sign-health negative count exceeds finite count")
        return {
            **counts,
            "negative_rate_over_diagnostic_domain": (
                None if domain_count == 0 else negative_count / domain_count
            ),
            "negative_rate_over_finite": (
                None if finite_count == 0 else negative_count / finite_count
            ),
            "nonfinite_rate_over_diagnostic_domain": (
                None if domain_count == 0 else nonfinite_count / domain_count
            ),
        }

    def finalize(self) -> dict[str, Any]:
        """Return an exhaustive JSON-safe diagnostic receipt."""

        stages: dict[str, Any] = {}
        for stage_name, field_name, grid in T3_VGGT_SIGN_HEALTH_STAGES:
            strata = {
                stratum: self._finalize_counts(self._counts[stage_name][stratum])
                for stratum in T3_VGGT_SIGN_HEALTH_STRATA
            }
            stages[stage_name] = {
                "model_output_field": field_name,
                "native_grid": grid,
                "disparity_unit": "HR_PIXEL",
                "diagnostic_domain": (
                    "all primary T3_VGGT endpoint pixels on this operation's "
                    "native grid; no pseudo-GT, teacher, confidence, or trusted mask"
                ),
                "diagnostic_domain_count": strata["all_pixels"][
                    "diagnostic_domain_count"
                ],
                "strata": strata,
            }
        return {
            "schema_version": 1,
            "method": "T3_VGGT",
            "causal_endpoint_time_index": 2,
            "records_accumulated": self.records_accumulated,
            "negative_definition": "isfinite(disparity) AND disparity < 0",
            "nonfinite_definition": "NOT isfinite(disparity)",
            "rate_unit": "fraction",
            "bilinear_strata_reference": (
                "the exact sanitized HR bilinear FFS metric baseline used by this "
                "evaluation, bilinearly resized to each operation's native grid"
            ),
            "mask_strata_reference": {
                "ffs_valid": "endpoint observation valid mask at native LR/HR grid",
                "history_valid": (
                    "primary T3_VGGT endpoint z-buffer valid-history mask at native "
                    "LR/HR grid"
                ),
                "pose_valid": (
                    "endpoint VGGT temporal-pose valid flag broadcast over native grid"
                ),
            },
            "partition_contract": {
                "bilinear_disparity": [
                    "bilinear_disparity_lt_0_hr_px",
                    "bilinear_disparity_ge_0_lt_0_25_hr_px",
                    "bilinear_disparity_ge_0_25_hr_px",
                    "bilinear_disparity_nonfinite",
                ],
                "ffs_validity": ["ffs_valid", "ffs_invalid"],
                "history_validity": ["history_valid", "history_invalid"],
                "pose_validity": ["pose_valid", "pose_invalid"],
            },
            "stages": stages,
        }


def _build_eval_transport(
    *,
    previous_output: ModelOutput,
    previous_rgb_hr: Tensor,
    current_rgb_hr: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    batch: dict[str, Any],
    time_index: int,
    config: DictConfig,
) -> TemporalTransport:
    """Call the exact Stage-B training transport with evaluation config values."""

    return build_temporal_transport(
        previous_output=previous_output,
        previous_rgb_hr=previous_rgb_hr,
        current_rgb_hr=current_rgb_hr,
        current_ffs_disparity_hr_px=current_ffs_disparity_hr_px,
        current_ffs_confidence=current_ffs_confidence,
        intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
        baseline_current_m=batch["baseline_m_sequence"][:, time_index],
        temporal_extrinsics_camera_from_world=batch[
            "vggt_extrinsics_camera_from_world_metric_sequence"
        ][:, time_index],
        temporal_pose_valid=batch["temporal_pose_valid_sequence"][:, time_index],
        scale=int(config.data.scale),
        photometric_temperature=float(config.train.photometric_temperature),
        disparity_temperature_hr_px=float(
            config.train.disparity_temperature_hr_px
        ),
        reject_conflict_hr_px=float(config.train.history_conflict_hr_px),
        photometric_threshold=float(config.train.temporal_photometric_threshold),
        geometry_threshold_hr_px=float(
            config.train.temporal_geometry_threshold_hr_px
        ),
    )


def _build_eval_topk_transport(
    *,
    memory: list[TemporalMemoryEntry],
    current_rgb_hr: Tensor,
    current_ffs_disparity_hr_px: Tensor,
    current_ffs_confidence: Tensor,
    batch: dict[str, Any],
    time_index: int,
    config: DictConfig,
) -> TemporalTransport:
    """Build the exact opt-in V2 transport for one evaluation branch."""

    contract = temporal_history_v2_from_config(config)
    if not contract.enabled:
        raise ValueError("top-K evaluation transport requires temporal_history_v2")
    return build_topk_temporal_transport(
        memory=memory,
        current_time_index=time_index,
        current_rgb_hr=current_rgb_hr,
        current_ffs_disparity_hr_px=current_ffs_disparity_hr_px,
        current_ffs_confidence=current_ffs_confidence,
        intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
        baseline_current_m=batch["baseline_m_sequence"][:, time_index],
        temporal_extrinsics_camera_from_world=batch[
            "vggt_extrinsics_camera_from_world_metric_sequence"
        ][:, time_index],
        temporal_pose_valid=batch["temporal_pose_valid_sequence"][:, time_index],
        contract=contract,
        scale=int(config.data.scale),
        photometric_temperature=float(config.train.photometric_temperature),
        disparity_temperature_hr_px=float(
            config.train.disparity_temperature_hr_px
        ),
        reject_conflict_hr_px=float(config.train.history_conflict_hr_px),
        photometric_threshold=float(config.train.temporal_photometric_threshold),
        geometry_threshold_hr_px=float(
            config.train.temporal_geometry_threshold_hr_px
        ),
    )


def _build_eval_reference_transport(
    *,
    batch: dict[str, Any],
    time_index: int,
    config: DictConfig,
    previous_prediction_disparity_hr_px: Tensor,
) -> ReferenceTemporalWarp:
    """Warp the trusted previous teacher into the current camera for V2 TEPE."""

    contract = temporal_history_v2_from_config(config)
    residual_contract = temporal_residual_v2_from_config(config)
    if not contract.enabled or not residual_contract.enabled:
        raise ValueError("reference warp requires both temporal V2 contracts")
    if time_index <= 0:
        raise ValueError("reference warp requires an immediate previous frame")
    disparity = batch.get("teacher_disparity_hr_px_sequence")
    confidence = batch.get("teacher_confidence_sequence")
    valid = batch.get("teacher_valid_mask_sequence")
    trusted = batch.get("teacher_trusted_mask_sequence")
    if not all(
        isinstance(value, Tensor)
        for value in (disparity, confidence, valid, trusted)
    ):
        raise ValueError(
            "teacher temporal-residual evaluation requires disparity, confidence, "
            "valid, and trusted sequences"
        )
    return build_reference_temporal_warp(
        previous_reference_disparity_hr_px=disparity[:, time_index - 1],
        previous_reference_confidence=confidence[:, time_index - 1],
        previous_reference_valid_mask=(
            valid[:, time_index - 1] & trusted[:, time_index - 1]
        ),
        previous_prediction_disparity_hr_px=(
            previous_prediction_disparity_hr_px
        ),
        intrinsics_current_hr=batch["K_hr_sequence"][:, time_index],
        baseline_current_m=batch["baseline_m_sequence"][:, time_index],
        temporal_extrinsics_camera_from_world=batch[
            "vggt_extrinsics_camera_from_world_metric_sequence"
        ][:, time_index],
        temporal_pose_valid=batch["temporal_pose_valid_sequence"][:, time_index],
        contract=contract,
    )


def _history_model_kwargs(
    transport: TemporalTransport | None,
    *,
    rgb_dtype: torch.dtype,
    temporal_v2: bool = False,
) -> dict[str, Tensor]:
    if transport is None:
        return {}
    result = {
        "disparity_history_hr_px": transport.disparity_history_hr_px,
        "confidence_history": transport.confidence_history,
        "history_visibility": transport.visibility_mask.to(dtype=rgb_dtype),
        "photometric_residual": transport.photometric_residual,
        "fractional_offset_px": transport.fractional_offset_px,
        "valid_history": transport.valid_history,
    }
    if not temporal_v2:
        return result
    topk_values = {
        "history_topk_disparity_hr_px": transport.topk_disparity_history_hr_px,
        "history_topk_confidence": transport.topk_confidence_history,
        "history_topk_fractional_offset_px": (
            transport.topk_fractional_offset_px
        ),
        "history_topk_age_frames": transport.topk_temporal_age_frames,
        "history_topk_weights": transport.topk_z_aware_weights,
        "history_topk_valid_mask": transport.topk_valid_mask,
    }
    missing = sorted(name for name, value in topk_values.items() if value is None)
    if missing:
        raise RuntimeError(f"V2 transport is missing model inputs: {missing}")
    result.update(
        {name: value for name, value in topk_values.items() if value is not None}
    )
    return result


def _temporal_residual_metric_for_transport(
    prediction_hr_px: Tensor,
    transport: TemporalTransport,
    *,
    current_reference_hr_px: Tensor,
    current_reference_valid_hr: Tensor,
    reference_transport: ReferenceTemporalWarp,
    paired_domain_mask_hr: Tensor | None = None,
) -> MetricResult:
    """Evaluate one branch with the V2 teacher/GT residual definition."""

    warped_previous_prediction_hr_px = (
        reference_transport.prediction_disparity_hr_px
    )
    if warped_previous_prediction_hr_px is None:
        raise ValueError(
            "teacher-correspondence reference warp has no carried prediction"
        )
    return hr_temporal_residual_metric(
        prediction_hr_px,
        warped_previous_prediction_hr_px,
        current_reference_hr_px,
        reference_transport.disparity_hr_px,
        visibility_mask_hr=transport.visibility_mask_hr,
        static_mask_hr=transport.static_mask_hr,
        collision_mask_hr=transport.collision_mask_hr,
        geometry_consistent_mask_hr=transport.geometry_consistent_mask_hr,
        valid_prediction_history_hr=transport.valid_history_hr,
        current_reference_valid_mask_hr=current_reference_valid_hr,
        warped_previous_reference_valid_mask_hr=(
            reference_transport.valid_mask_hr
        ),
        paired_domain_mask_hr=paired_domain_mask_hr,
    )


def _binary_ratio_result(numerator: int, denominator: int) -> MetricResult:
    if denominator <= 0:
        return MetricResult.invalid()
    return MetricResult(
        value=float(numerator) / denominator,
        numerator=float(numerator),
        count=denominator,
        valid=True,
    )


def _strict_brier_result(
    probability: Tensor, target: Tensor, domain: Tensor
) -> MetricResult:
    count = int(domain.sum().item())
    if count == 0:
        return MetricResult.invalid()
    selected = (probability.float() - target.float()).square()[domain]
    if not bool(torch.isfinite(selected).all().item()):
        return MetricResult.invalid(count=count)
    numerator = float(selected.to(dtype=torch.float64).sum().item())
    return MetricResult(numerator / count, numerator, count, True)


def _explicit_validity_completion_metrics(
    output: ModelOutput | None,
    *,
    target_disparity_hr_px: Tensor,
    teacher_valid_mask_hr: Tensor,
    ffs_valid_mask_hr: Tensor,
) -> dict[str, MetricResult]:
    """Score opt-in physical-validity heads with aggregation-safe counts."""

    names = (
        "explicit_valid_precision",
        "explicit_valid_recall",
        "explicit_valid_f1",
        "explicit_valid_brier",
        "ffs_hole_completion_precision",
        "ffs_hole_completion_recall",
        "ffs_hole_completion_f1",
        "ffs_hole_completion_brier",
    )
    if output is None or any(
        value is None
        for value in (
            output.valid_probability,
            output.completion_probability,
            output.output_valid_mask,
            output.completion_mask,
        )
    ):
        return {name: MetricResult.invalid() for name in names}

    reference_shape = target_disparity_hr_px.shape
    fields = {
        "teacher_valid_mask_hr": teacher_valid_mask_hr,
        "ffs_valid_mask_hr": ffs_valid_mask_hr,
        "valid_probability": output.valid_probability,
        "completion_probability": output.completion_probability,
        "output_valid_mask": output.output_valid_mask,
        "completion_mask": output.completion_mask,
    }
    for name, value in fields.items():
        if not isinstance(value, Tensor) or value.shape != reference_shape:
            raise ValueError(f"{name} must have shape {tuple(reference_shape)}")

    teacher_positive = (
        teacher_valid_mask_hr.to(dtype=torch.bool)
        & torch.isfinite(target_disparity_hr_px)
        & (target_disparity_hr_px > 0)
    )
    predicted_valid = output.output_valid_mask.to(dtype=torch.bool)
    true_positive = int((predicted_valid & teacher_positive).sum().item())
    predicted_count = int(predicted_valid.sum().item())
    teacher_count = int(teacher_positive.sum().item())
    valid_f1_denominator = predicted_count + teacher_count

    ffs_hole = ~ffs_valid_mask_hr.to(dtype=torch.bool)
    predicted_completion = output.completion_mask.to(dtype=torch.bool) & ffs_hole
    target_completion = teacher_positive & ffs_hole
    completion_true_positive = int(
        (predicted_completion & target_completion).sum().item()
    )
    predicted_completion_count = int(predicted_completion.sum().item())
    target_completion_count = int(target_completion.sum().item())
    completion_f1_denominator = (
        predicted_completion_count + target_completion_count
    )
    full_domain = torch.ones_like(teacher_positive)
    return {
        "explicit_valid_precision": _binary_ratio_result(
            true_positive, predicted_count
        ),
        "explicit_valid_recall": _binary_ratio_result(true_positive, teacher_count),
        "explicit_valid_f1": _binary_ratio_result(
            2 * true_positive, valid_f1_denominator
        ),
        "explicit_valid_brier": _strict_brier_result(
            output.valid_probability, teacher_positive, full_domain
        ),
        "ffs_hole_completion_precision": _binary_ratio_result(
            completion_true_positive, predicted_completion_count
        ),
        "ffs_hole_completion_recall": _binary_ratio_result(
            completion_true_positive, target_completion_count
        ),
        "ffs_hole_completion_f1": _binary_ratio_result(
            2 * completion_true_positive, completion_f1_denominator
        ),
        "ffs_hole_completion_brier": _strict_brier_result(
            output.completion_probability,
            target_completion,
            ffs_hole,
        ),
    }


def _run_spatial_endpoint(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    config: DictConfig,
) -> SpatialEndpointPrediction:
    """Run T1 independently at each time and return the final transition."""

    previous_output: ModelOutput | None = None
    previous_rgb_hr: Tensor | None = None
    final_transport: TemporalTransport | None = None
    final_reference_transport: ReferenceTemporalWarp | None = None
    memory: list[TemporalMemoryEntry] = []
    output: ModelOutput | None = None
    temporal_v2 = temporal_history_v2_from_config(config).enabled
    for time_index in range(3):
        step = _temporal_step_batch(batch, time_index)
        output = model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            valid_ffs=step["valid_ffs"],
            hidden_state=None,
        )
        if time_index > 0:
            assert previous_output is not None and previous_rgb_hr is not None
            if temporal_v2:
                final_transport = _build_eval_topk_transport(
                    memory=memory,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
                final_reference_transport = _build_eval_reference_transport(
                    batch=batch,
                    time_index=time_index,
                    config=config,
                    previous_prediction_disparity_hr_px=(
                        previous_output.disparity_hr_px
                    ),
                )
            else:
                final_transport = _build_eval_transport(
                    previous_output=previous_output,
                    previous_rgb_hr=previous_rgb_hr,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
        if temporal_v2:
            memory.append(
                TemporalMemoryEntry(
                    output=output,
                    rgb_hr=step["rgb_hr"],
                    time_index=time_index,
                )
            )
            memory = memory[-temporal_history_v2_from_config(config).memory_frames :]
        previous_output = output
        previous_rgb_hr = step["rgb_hr"]
    assert output is not None and final_transport is not None
    if temporal_v2 and final_reference_transport is None:
        raise RuntimeError("V2 spatial endpoint did not produce a reference warp")
    return SpatialEndpointPrediction(
        output=output,
        transport=final_transport,
        reference_transport=final_reference_transport,
    )


def _run_temporal_endpoint_ablation(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    config: DictConfig,
) -> TemporalEndpointPredictions:
    """Unroll T=3 and isolate the VGGT source-mask intervention.

    Legacy evaluation preserves the historical shared-history source-mask
    intervention exactly.  Under temporal V2, every branch propagates its own
    causal top-K memory and warped hidden state; the source-mask branch still
    changes only ``valid_vggt`` at each model call.  The no-VGGT branch also
    zeros the VGGT geometry channels.
    """

    hidden_on: tuple[Tensor, ...] | None = None
    hidden_mask_off: tuple[Tensor, ...] | None = None
    hidden_no_vggt: tuple[Tensor, ...] | None = None
    previous_on: ModelOutput | None = None
    previous_mask_off: ModelOutput | None = None
    previous_no_vggt: ModelOutput | None = None
    previous_rgb_hr: Tensor | None = None
    transport: TemporalTransport | None = None
    mask_off_transport: TemporalTransport | None = None
    no_vggt_transport: TemporalTransport | None = None
    reference_transport: ReferenceTemporalWarp | None = None
    mask_off_reference_transport: ReferenceTemporalWarp | None = None
    no_vggt_reference_transport: ReferenceTemporalWarp | None = None
    memory_on: list[TemporalMemoryEntry] = []
    memory_mask_off: list[TemporalMemoryEntry] = []
    memory_no_vggt: list[TemporalMemoryEntry] = []
    output_on: ModelOutput | None = None
    output_mask_off: ModelOutput | None = None
    output_no_vggt: ModelOutput | None = None
    history_contract = temporal_history_v2_from_config(config)
    residual_contract = temporal_residual_v2_from_config(config)
    temporal_v2 = history_contract.enabled and residual_contract.enabled
    for time_index in range(3):
        step = _temporal_step_batch(batch, time_index)
        pose_valid = batch["temporal_pose_valid_sequence"][:, time_index]
        if time_index > 0:
            assert previous_on is not None and previous_rgb_hr is not None
            assert previous_mask_off is not None and previous_no_vggt is not None
            if temporal_v2:
                transport = _build_eval_topk_transport(
                    memory=memory_on,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
                mask_off_transport = _build_eval_topk_transport(
                    memory=memory_mask_off,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
                no_vggt_transport = _build_eval_topk_transport(
                    memory=memory_no_vggt,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
                hidden_on = transport.warped_hidden_state
                hidden_mask_off = mask_off_transport.warped_hidden_state
                hidden_no_vggt = no_vggt_transport.warped_hidden_state
                reference_transport = _build_eval_reference_transport(
                    batch=batch,
                    time_index=time_index,
                    config=config,
                    previous_prediction_disparity_hr_px=(
                        previous_on.disparity_hr_px
                    ),
                )
                mask_off_reference_transport = _build_eval_reference_transport(
                    batch=batch,
                    time_index=time_index,
                    config=config,
                    previous_prediction_disparity_hr_px=(
                        previous_mask_off.disparity_hr_px
                    ),
                )
                no_vggt_reference_transport = _build_eval_reference_transport(
                    batch=batch,
                    time_index=time_index,
                    config=config,
                    previous_prediction_disparity_hr_px=(
                        previous_no_vggt.disparity_hr_px
                    ),
                )
            else:
                hidden_on = _reset_hidden_where_pose_invalid(hidden_on, pose_valid)
                hidden_mask_off = _reset_hidden_where_pose_invalid(
                    hidden_mask_off, pose_valid
                )
                hidden_no_vggt = _reset_hidden_where_pose_invalid(
                    hidden_no_vggt, pose_valid
                )
                transport = _build_eval_transport(
                    previous_output=previous_on,
                    previous_rgb_hr=previous_rgb_hr,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
                # Historical source-mask intervention shares the primary
                # branch transport exactly.
                mask_off_transport = transport
                no_vggt_transport = _build_eval_transport(
                    previous_output=previous_no_vggt,
                    previous_rgb_hr=previous_rgb_hr,
                    current_rgb_hr=step["rgb_hr"],
                    current_ffs_disparity_hr_px=step["disparity_ffs_hr_px"],
                    current_ffs_confidence=step["confidence_ffs"],
                    batch=batch,
                    time_index=time_index,
                    config=config,
                )
        static_prior = batch["static_prior_valid_sequence"][:, time_index]
        valid_vggt = batch["valid_vggt_sequence"][:, time_index] & static_prior.reshape(
            -1, 1, 1, 1
        )
        history_kwargs = _history_model_kwargs(
            transport,
            rgb_dtype=step["rgb_hr"].dtype,
            temporal_v2=temporal_v2,
        )
        geometry_kwargs: dict[str, Any] = {
            "disparity_vggt_hr_px": batch["disparity_vggt_hr_px_sequence"][
                :, time_index
            ],
            "confidence_vggt": batch["confidence_vggt_sequence"][:, time_index],
            "valid_ffs": step["valid_ffs"],
        }
        output_on = model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            valid_vggt=valid_vggt,
            hidden_state=hidden_on,
            **geometry_kwargs,
            **history_kwargs,
        )
        mask_off_history_kwargs = _history_model_kwargs(
            mask_off_transport,
            rgb_dtype=step["rgb_hr"].dtype,
            temporal_v2=temporal_v2,
        )
        output_mask_off = model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            valid_vggt=torch.zeros_like(valid_vggt),
            hidden_state=hidden_mask_off,
            **geometry_kwargs,
            **mask_off_history_kwargs,
        )
        no_vggt_history_kwargs = _history_model_kwargs(
            no_vggt_transport,
            rgb_dtype=step["rgb_hr"].dtype,
            temporal_v2=temporal_v2,
        )
        zero_vggt = torch.zeros_like(
            batch["disparity_vggt_hr_px_sequence"][:, time_index]
        )
        output_no_vggt = model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            disparity_vggt_hr_px=zero_vggt,
            confidence_vggt=torch.zeros_like(zero_vggt),
            valid_vggt=torch.zeros_like(valid_vggt),
            valid_ffs=step["valid_ffs"],
            hidden_state=hidden_no_vggt,
            **no_vggt_history_kwargs,
        )
        hidden_on = output_on.hidden_state
        hidden_mask_off = output_mask_off.hidden_state
        hidden_no_vggt = output_no_vggt.hidden_state
        if temporal_v2:
            entries = (
                (memory_on, output_on),
                (memory_mask_off, output_mask_off),
                (memory_no_vggt, output_no_vggt),
            )
            for branch_memory, branch_output in entries:
                branch_memory.append(
                    TemporalMemoryEntry(
                        output=branch_output,
                        rgb_hr=step["rgb_hr"],
                        time_index=time_index,
                    )
                )
                del branch_memory[: -history_contract.memory_frames]
        previous_on = output_on
        previous_mask_off = output_mask_off
        previous_no_vggt = output_no_vggt
        previous_rgb_hr = step["rgb_hr"]
    assert (
        output_on is not None
        and output_mask_off is not None
        and output_no_vggt is not None
        and transport is not None
        and mask_off_transport is not None
        and no_vggt_transport is not None
    )
    if temporal_v2 and any(
        value is None
        for value in (
            reference_transport,
            mask_off_reference_transport,
            no_vggt_reference_transport,
        )
    ):
        raise RuntimeError("V2 temporal endpoint did not produce branch reference warps")
    return TemporalEndpointPredictions(
        vggt_on=output_on,
        source_mask_off=output_mask_off,
        no_vggt=output_no_vggt,
        shared_transport=transport,
        source_mask_off_transport=mask_off_transport,
        no_vggt_transport=no_vggt_transport,
        reference_transport=reference_transport,
        source_mask_off_reference_transport=mask_off_reference_transport,
        no_vggt_reference_transport=no_vggt_reference_transport,
    )


def _rgb_chw_to_uint8(rgb: Tensor) -> np.ndarray:
    array = (
        rgb.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return np.rint(array * 255.0).astype(np.uint8)


class _TemporalFlickerVideoUnavailable(RuntimeError):
    """Raised only for an optional imageio/FFmpeg encoding failure."""


DIRECT_FFMPEG_EXECUTABLE = Path("/usr/bin/ffmpeg")


class _DirectFFmpegRgb24Writer:
    """Stream fixed-size CPU uint8 RGB24 frames to one explicit FFmpeg process."""

    def __init__(
        self,
        executable: Path,
        path: Path,
        *,
        fps: int,
        frame_size_hw: tuple[int, int],
    ) -> None:
        height, width = frame_size_hw
        if height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError("direct FFmpeg frames must have positive even H/W")
        self.path = path
        self.frame_size_hw = frame_size_hw
        self._closed = False
        command = [
            str(executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise _TemporalFlickerVideoUnavailable(
                f"could not launch direct FFmpeg {executable}: {exc}"
            ) from exc

    def append_data(self, frame: np.ndarray) -> None:
        if self._closed:
            raise _TemporalFlickerVideoUnavailable("direct FFmpeg writer is closed")
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape != (*self.frame_size_hw, 3)
        ):
            raise ValueError(
                "direct FFmpeg frame must be fixed-size HxWx3 uint8 RGB24"
            )
        if self._process.stdin is None:
            raise _TemporalFlickerVideoUnavailable("direct FFmpeg stdin is unavailable")
        try:
            self._process.stdin.write(frame.tobytes(order="C"))
        except (BrokenPipeError, OSError) as exc:
            raise _TemporalFlickerVideoUnavailable(
                "direct FFmpeg rejected RGB24 frame: "
                f"{self._stderr_text() or type(exc).__name__}"
            ) from exc

    def _stderr_text(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            value = self._process.stderr.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return value.strip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        return_code = self._process.wait()
        stderr = self._stderr_text()
        if return_code != 0:
            raise _TemporalFlickerVideoUnavailable(
                "direct FFmpeg failed with return code "
                f"{return_code}: {stderr or 'no stderr'}"
            )
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise _TemporalFlickerVideoUnavailable(
                "direct FFmpeg exited successfully but wrote no MP4"
            )


def _open_imageio_temporal_flicker_writer(path: Path, *, fps: int) -> Any:
    """Open the preferred imageio FFmpeg writer, propagating ImportError only."""

    import imageio.v2 as imageio
    try:
        return imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
    except Exception as exc:  # pragma: no cover - depends on local encoder state
        raise _TemporalFlickerVideoUnavailable(
            f"imageio/FFmpeg could not open MP4 writer: {type(exc).__name__}: {exc}"
        ) from exc


def _open_direct_ffmpeg_temporal_flicker_writer(
    path: Path,
    *,
    fps: int,
    frame_size_hw: tuple[int, int],
) -> _DirectFFmpegRgb24Writer:
    """Probe the explicit system FFmpeg and open a raw RGB24 streaming pipe."""

    executable = DIRECT_FFMPEG_EXECUTABLE
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise _TemporalFlickerVideoUnavailable(
            "imageio is unavailable and direct FFmpeg executable is not usable: "
            f"{executable}"
        )
    return _DirectFFmpegRgb24Writer(
        executable,
        path,
        fps=fps,
        frame_size_hw=frame_size_hw,
    )


def _open_temporal_flicker_video_writer(
    path: Path,
    *,
    fps: int,
    frame_size_hw: tuple[int, int],
) -> tuple[Any, str]:
    """Prefer imageio; fall back to explicit direct FFmpeg without Python installs."""

    try:
        return _open_imageio_temporal_flicker_writer(path, fps=fps), "imageio_ffmpeg"
    except ImportError:
        return (
            _open_direct_ffmpeg_temporal_flicker_writer(
                path,
                fps=fps,
                frame_size_hw=frame_size_hw,
            ),
            "direct_ffmpeg_rgb24",
        )


def _temporal_flicker_tile(
    image_rgb_uint8: np.ndarray,
    *,
    label: str,
    colorbar_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return one labelled CPU uint8 panel with an optional fixed color bar."""

    if (
        image_rgb_uint8.dtype != np.uint8
        or image_rgb_uint8.ndim != 3
        or image_rgb_uint8.shape[-1] != 3
    ):
        raise ValueError("temporal flicker tile must be HxWx3 uint8")
    height, width = image_rgb_uint8.shape[:2]
    label_height = 24
    canvas = Image.new("RGB", (width, height + label_height), color=(0, 0, 0))
    canvas.paste(Image.fromarray(image_rgb_uint8, mode="RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), label, fill=(255, 255, 255))
    if colorbar_range is not None:
        minimum, maximum = colorbar_range
        values = np.linspace(minimum, maximum, num=128, dtype=np.float32)[None, :]
        colorbar = scalar_to_rgb_uint8(
            values,
            minimum=minimum,
            maximum=maximum,
        )
        colorbar_width = min(128, max(1, width // 3))
        colorbar_image = Image.fromarray(colorbar, mode="RGB").resize(
            (colorbar_width, 8)
        )
        canvas.paste(colorbar_image, (width - colorbar_width - 4, 8))
    return np.asarray(canvas, dtype=np.uint8)


def build_temporal_flicker_panel(
    *,
    rgb_hr: Tensor,
    bilinear_disparity_hr_px: Tensor,
    t3_disparity_hr_px: Tensor,
    t3_vggt_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    target_trusted_mask: Tensor,
    uncertainty_variance: Tensor,
    disparity_range_hr_px: tuple[float, float],
    error_range_hr_px: tuple[float, float],
    uncertainty_range: tuple[float, float],
) -> np.ndarray:
    """Build one non-metric six-panel temporal frame on CPU as uint8.

    Inputs may reside on CUDA, but each is detached and converted immediately;
    no tensor or float frame is retained by the video collector.  Scalar
    color ranges are supplied explicitly, making every panel comparable over
    time and across videos.
    """

    error = (t3_vggt_disparity_hr_px - target_disparity_hr_px).abs()
    panels = (
        _temporal_flicker_tile(_rgb_chw_to_uint8(rgb_hr), label="RGB"),
        _temporal_flicker_tile(
            scalar_to_rgb_uint8(
                bilinear_disparity_hr_px,
                minimum=disparity_range_hr_px[0],
                maximum=disparity_range_hr_px[1],
            ),
            label=(
                "Bilinear FFS "
                f"[{disparity_range_hr_px[0]:g},{disparity_range_hr_px[1]:g}] px"
            ),
            colorbar_range=disparity_range_hr_px,
        ),
        _temporal_flicker_tile(
            scalar_to_rgb_uint8(
                t3_disparity_hr_px,
                minimum=disparity_range_hr_px[0],
                maximum=disparity_range_hr_px[1],
            ),
            label=(
                "T3 (no VGGT prior) "
                f"[{disparity_range_hr_px[0]:g},{disparity_range_hr_px[1]:g}] px"
            ),
            colorbar_range=disparity_range_hr_px,
        ),
        _temporal_flicker_tile(
            scalar_to_rgb_uint8(
                t3_vggt_disparity_hr_px,
                minimum=disparity_range_hr_px[0],
                maximum=disparity_range_hr_px[1],
            ),
            label=(
                "T3 + VGGT "
                f"[{disparity_range_hr_px[0]:g},{disparity_range_hr_px[1]:g}] px"
            ),
            colorbar_range=disparity_range_hr_px,
        ),
        _temporal_flicker_tile(
            scalar_to_rgb_uint8(
                error,
                valid_mask=target_trusted_mask,
                minimum=error_range_hr_px[0],
                maximum=error_range_hr_px[1],
            ),
            label=(
                "|T3 + VGGT - pseudo-GT| "
                f"[{error_range_hr_px[0]:g},{error_range_hr_px[1]:g}] px"
            ),
            colorbar_range=error_range_hr_px,
        ),
        _temporal_flicker_tile(
            scalar_to_rgb_uint8(
                uncertainty_variance,
                minimum=uncertainty_range[0],
                maximum=uncertainty_range[1],
            ),
            label=(
                "Uncertainty variance "
                f"[{uncertainty_range[0]:g},{uncertainty_range[1]:g}]"
            ),
            colorbar_range=uncertainty_range,
        ),
    )
    height, width, _ = panels[0].shape
    if any(panel.shape != (height, width, 3) for panel in panels):
        raise ValueError("temporal flicker panels must have matching dimensions")
    frame = np.concatenate(
        (np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)),
        axis=0,
    )
    # yuv420p encoders require even frame dimensions. Padding is visual-only
    # and occurs only after all metric/model tensors have been released.
    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if pad_height or pad_width:
        frame = np.pad(
            frame,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
        )
    return frame


@dataclass(slots=True)
class _TemporalFlickerSequenceState:
    writer: Any
    backend: str
    temporary_path: Path
    output_path: Path
    frame_size_hw: tuple[int, int]
    last_frame_id: int
    last_timestamp: float
    frame_count: int = 0
    published: bool = False


class TemporalFlickerVideoCollector:
    """Streaming, opt-in MP4 writer for causal T3 visualization only.

    The collector stores only writer handles and small scalar bookkeeping. Each
    frame is a transient CPU uint8 array passed immediately to FFmpeg; it never
    retains model outputs, GPU tensors, or a sequence of float frames.
    """

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool,
        fps: int,
        disparity_range_hr_px: tuple[float, float],
        error_range_hr_px: tuple[float, float],
        uncertainty_range: tuple[float, float],
    ) -> None:
        self.root = root
        self.enabled = enabled
        self.fps = fps
        self.disparity_range_hr_px = disparity_range_hr_px
        self.error_range_hr_px = error_range_hr_px
        self.uncertainty_range = uncertainty_range
        self._states: dict[str, _TemporalFlickerSequenceState] = {}
        self._not_available_reason: str | None = None

    @staticmethod
    def _file_stem(sequence_id: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_id).strip("._")
        if not stem:
            raise ValueError("temporal flicker sequence_id has no safe filename stem")
        return stem

    def _abort(self, reason: str) -> None:
        if self._not_available_reason is not None:
            return
        self._not_available_reason = reason
        for state in self._states.values():
            try:
                state.writer.close()
            except Exception:
                pass
            state.temporary_path.unlink(missing_ok=True)
            if state.published:
                state.output_path.unlink(missing_ok=True)
        self._states.clear()

    def abort_for_interruption(self, exception: BaseException) -> None:
        """Close pipes and delete temporary artifacts before an evaluation abort."""

        self._abort(
            "evaluation interrupted before temporal flicker MP4 finalization: "
            f"{type(exception).__name__}: {exception}"
        )

    def append(
        self,
        *,
        sequence_id: str,
        frame_id: int,
        timestamp: float,
        rgb_hr: Tensor,
        bilinear_disparity_hr_px: Tensor,
        t3_disparity_hr_px: Tensor,
        t3_vggt_disparity_hr_px: Tensor,
        target_disparity_hr_px: Tensor,
        target_trusted_mask: Tensor,
        uncertainty_variance: Tensor,
    ) -> None:
        """Encode one immediate CPU uint8 panel; enforce sequence time order."""

        if not self.enabled or self._not_available_reason is not None:
            return
        state = self._states.get(sequence_id)
        if state is not None and (
            frame_id <= state.last_frame_id or timestamp <= state.last_timestamp
        ):
            raise ValueError(
                "temporal flicker input must be strictly increasing per sequence: "
                f"{sequence_id!r} frame={frame_id} timestamp={timestamp}"
            )
        try:
            frame = build_temporal_flicker_panel(
                rgb_hr=rgb_hr,
                bilinear_disparity_hr_px=bilinear_disparity_hr_px,
                t3_disparity_hr_px=t3_disparity_hr_px,
                t3_vggt_disparity_hr_px=t3_vggt_disparity_hr_px,
                target_disparity_hr_px=target_disparity_hr_px,
                target_trusted_mask=target_trusted_mask,
                uncertainty_variance=uncertainty_variance,
                disparity_range_hr_px=self.disparity_range_hr_px,
                error_range_hr_px=self.error_range_hr_px,
                uncertainty_range=self.uncertainty_range,
            )
            if state is None:
                self.root.mkdir(parents=True, exist_ok=True)
                output_path = self.root / f"{self._file_stem(sequence_id)}.mp4"
                temporary_path = self.root / (
                    f".{self._file_stem(sequence_id)}.incomplete.mp4"
                )
                writer, backend = _open_temporal_flicker_video_writer(
                    temporary_path,
                    fps=self.fps,
                    frame_size_hw=tuple(int(value) for value in frame.shape[:2]),
                )
                state = _TemporalFlickerSequenceState(
                    writer=writer,
                    backend=backend,
                    temporary_path=temporary_path,
                    output_path=output_path,
                    frame_size_hw=tuple(int(value) for value in frame.shape[:2]),
                    last_frame_id=frame_id,
                    last_timestamp=timestamp,
                )
                self._states[sequence_id] = state
            elif tuple(int(value) for value in frame.shape[:2]) != state.frame_size_hw:
                raise ValueError(
                    "temporal flicker frame size changed within sequence: "
                    f"{state.frame_size_hw} -> {tuple(frame.shape[:2])}"
                )
            state.writer.append_data(frame)
            state.last_frame_id = frame_id
            state.last_timestamp = timestamp
            state.frame_count += 1
        except _TemporalFlickerVideoUnavailable as exc:
            self._abort(str(exc))
        except Exception as exc:
            self._abort(f"MP4 encoding failed: {type(exc).__name__}: {exc}")

    def finalize(self) -> dict[str, Any]:
        """Close/atomically publish MP4s and return explicit non-metric status."""

        common = {
            "enabled": self.enabled,
            "metric_participation": "NONE",
            "disparity_range_hr_px": list(self.disparity_range_hr_px),
            "error_range_hr_px": list(self.error_range_hr_px),
            "uncertainty_variance_range": list(self.uncertainty_range),
            "fps": self.fps,
        }
        if not self.enabled:
            return common | {"status": "DISABLED", "videos": []}
        if self._not_available_reason is not None:
            return common | {
                "status": "NOT_AVAILABLE",
                "reason": self._not_available_reason,
                "videos": [],
            }
        videos: list[dict[str, Any]] = []
        try:
            for sequence_id, state in self._states.items():
                state.writer.close()
                os.replace(state.temporary_path, state.output_path)
                state.published = True
                videos.append(
                    {
                        "sequence_id": sequence_id,
                        "path": str(state.output_path),
                        "frame_count": state.frame_count,
                        "backend": state.backend,
                    }
                )
        except Exception as exc:
            self._abort(f"MP4 finalization failed: {type(exc).__name__}: {exc}")
            return common | {
                "status": "NOT_AVAILABLE",
                "reason": self._not_available_reason,
                "videos": [],
            }
        return common | {"status": "COMPLETE", "videos": videos}


@contextmanager
def _cleanup_temporal_flicker_on_abort(
    collector: TemporalFlickerVideoCollector | None,
) -> Any:
    """Ensure an interrupted/failed evaluation never leaves encoder processes/files."""

    try:
        yield
    except BaseException as exc:
        if collector is not None:
            collector.abort_for_interruption(exc)
        raise


def _save_visualization(
    root: Path,
    *,
    sample_name: str,
    rgb_hr: Tensor,
    K_hr_px: Tensor,
    baseline_m: Tensor,
    baseline_hr_px: Tensor,
    output_hr_px: Tensor,
    target_hr_px: Tensor,
    target_trusted_mask: Tensor,
    source_weights_lr: Tensor,
    uncertainty_hr: Tensor,
    vggt_disparity_hr_px: Tensor | None = None,
    vggt_valid_mask_hr: Tensor | None = None,
    history_disparity_hr_px: Tensor | None = None,
    history_valid_mask_hr: Tensor | None = None,
    vggt_off_output_hr_px: Tensor | None = None,
    no_vggt_output_hr_px: Tensor | None = None,
    prediction_filename: str = "t1_disparity_hr_px.png",
) -> None:
    sample_root = root / sample_name
    target_mask = target_trusted_mask.to(dtype=torch.bool)
    absolute_error = (output_hr_px - target_hr_px).abs()
    final_negative_mask = torch.isfinite(output_hr_px) & (output_hr_px < 0.0)
    save_rgb_uint8(sample_root / "rgb.png", _rgb_chw_to_uint8(rgb_hr))
    export_colored_point_cloud_ply(
        output_hr_px,
        rgb_hr,
        K_hr_px,
        baseline_m,
        sample_root / "point_cloud_camera_frame.ply",
    )
    for filename, value, mask in (
        ("bilinear_ffs_hr_px.png", baseline_hr_px, None),
        (prediction_filename, output_hr_px, None),
        ("teacher_pseudo_gt_hr_px.png", target_hr_px, target_mask),
        ("absolute_error_hr_px.png", absolute_error, target_mask),
        ("uncertainty_variance.png", uncertainty_hr, None),
    ):
        save_rgb_uint8(
            sample_root / filename,
            scalar_to_rgb_uint8(value, valid_mask=mask),
        )
    save_rgb_uint8(
        sample_root / "final_negative_mask.png",
        grayscale_to_rgb_uint8(
            final_negative_mask.to(dtype=torch.float32), minimum=0.0, maximum=1.0
        ),
    )
    source_names = ("ffs", "vggt", "history")
    for source_index, source_name in enumerate(source_names):
        source_hr = functional.interpolate(
            source_weights_lr[source_index : source_index + 1].unsqueeze(0),
            size=output_hr_px.shape[-2:],
            mode="nearest",
        )[0]
        save_rgb_uint8(
            sample_root / f"source_weight_{source_name}.png",
            grayscale_to_rgb_uint8(source_hr, minimum=0.0, maximum=1.0),
        )
    if vggt_disparity_hr_px is not None:
        save_rgb_uint8(
            sample_root / "vggt_aligned_disparity_hr_px.png",
            scalar_to_rgb_uint8(
                vggt_disparity_hr_px, valid_mask=vggt_valid_mask_hr
            ),
        )
    if history_disparity_hr_px is not None:
        save_rgb_uint8(
            sample_root / "zbuffer_history_disparity_hr_px.png",
            scalar_to_rgb_uint8(
                history_disparity_hr_px, valid_mask=history_valid_mask_hr
            ),
        )
    if vggt_off_output_hr_px is not None:
        save_rgb_uint8(
            sample_root / "t3_vggt_source_off_hr_px.png",
            scalar_to_rgb_uint8(vggt_off_output_hr_px),
        )
    if no_vggt_output_hr_px is not None:
        save_rgb_uint8(
            sample_root / "t3_no_vggt_prior_hr_px.png",
            scalar_to_rgb_uint8(no_vggt_output_hr_px),
        )


FAILURE_SAMPLE_CRITERIA: dict[str, str] = {
    "raw_negative_rate": (
        "T3_VGGT raw-output finite-negative rate over every endpoint HR pixel"
    ),
    "low_confidence_epe_px": (
        "T3_VGGT EPE on trusted teacher pseudo-GT pixels where FFS confidence "
        "is below eval.low_confidence_threshold"
    ),
    "boundary_epe_px": (
        "T3_VGGT EPE on trusted teacher pseudo-GT disparity-boundary pixels"
    ),
    "strict_temporal_error_px": (
        "T3_VGGT EPE against z-buffer history on the strict visible/static/"
        "non-collision/geometry-consistent/valid-history domain"
    ),
}

TEMPORAL_RESIDUAL_V2_FAILURE_SAMPLE_CRITERIA: dict[str, str] = {
    **{
        name: definition
        for name, definition in FAILURE_SAMPLE_CRITERIA.items()
        if name != "strict_temporal_error_px"
    },
    "strict_temporal_residual_error_px": (
        "T3_VGGT teacher temporal-residual error on the common trusted-teacher/"
        "z-buffer-visible/static/non-collision/geometry-consistent domain"
    ),
}


@dataclass(slots=True)
class _FailureSampleCandidate:
    criterion: str
    metric: MetricResult
    sequence_id: str
    frame_id: int
    manifest_index: int
    timestamp: float
    tensors_cpu: dict[str, Tensor]
    tensor_bytes: int

    @property
    def sort_key(self) -> tuple[float, str, int, int]:
        # Ascending tuple order implements descending score, then the required
        # stable sequence/frame/manifest-index tie break.
        return (-float(self.metric.value), self.sequence_id, self.frame_id, self.manifest_index)


def _metric_result_dict(metric: MetricResult) -> dict[str, float | int | bool]:
    if not metric.valid or not math.isfinite(metric.value) or not math.isfinite(metric.numerator):
        raise ValueError("failure bundles require a finite valid per-sample metric")
    return {
        "value": float(metric.value),
        "numerator": float(metric.numerator),
        "count": int(metric.count),
        "valid": True,
    }


def _safe_failure_sample_stem(
    *, rank: int, sequence_id: str, frame_id: int, manifest_index: int
) -> str:
    sequence_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sequence_id).strip("._")
    if not sequence_stem:
        raise ValueError("failure sample sequence_id has no safe filename stem")
    return f"{rank:04d}_{sequence_stem}_{frame_id}_{manifest_index}"


class FailureSampleCollector:
    """Keep only current deterministic top-k T3 failure payloads on CPU."""

    def __init__(
        self,
        *,
        samples_per_criterion: int,
        cpu_limit_bytes: int,
        criteria: Mapping[str, str] | None = None,
    ) -> None:
        if samples_per_criterion <= 0:
            raise ValueError("samples_per_criterion must be positive")
        if cpu_limit_bytes <= 0:
            raise ValueError("cpu_limit_bytes must be positive")
        self.samples_per_criterion = samples_per_criterion
        self.cpu_limit_bytes = cpu_limit_bytes
        self.cpu_bytes_retained = 0
        self.criteria = dict(
            FAILURE_SAMPLE_CRITERIA if criteria is None else criteria
        )
        if not self.criteria or any(
            not isinstance(name, str) or not isinstance(definition, str)
            for name, definition in self.criteria.items()
        ):
            raise ValueError("failure criteria must be a non-empty string mapping")
        self._candidates: dict[str, list[_FailureSampleCandidate]] = {
            criterion: [] for criterion in self.criteria
        }

    @staticmethod
    def _payload_bytes(payload: Mapping[str, Tensor]) -> int:
        total = 0
        for name, tensor in payload.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"failure payload {name!r} must be a tensor")
            total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _detach_payload_to_cpu(payload: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: tensor.detach().to(device="cpu", copy=True).contiguous()
            for name, tensor in payload.items()
        }

    def consider(
        self,
        criterion: str,
        metric: MetricResult,
        *,
        sequence_id: str,
        frame_id: int,
        manifest_index: int,
        timestamp: float,
        payload_factory: Callable[[], Mapping[str, Tensor]],
    ) -> None:
        """Retain this sample only when it enters the current top-k list."""

        if criterion not in self._candidates:
            raise ValueError(f"unknown failure criterion {criterion!r}")
        if (
            not metric.valid
            or metric.count <= 0
            or not math.isfinite(metric.value)
            or not math.isfinite(metric.numerator)
        ):
            return
        provisional_key = (-float(metric.value), sequence_id, frame_id, manifest_index)
        candidates = self._candidates[criterion]
        if len(candidates) >= self.samples_per_criterion and provisional_key >= candidates[-1].sort_key:
            return

        payload = payload_factory()
        payload_bytes = self._payload_bytes(payload)
        evicted: _FailureSampleCandidate | None = None
        if len(candidates) >= self.samples_per_criterion:
            evicted = candidates.pop()
            self.cpu_bytes_retained -= evicted.tensor_bytes
        if self.cpu_bytes_retained + payload_bytes > self.cpu_limit_bytes:
            if evicted is not None:
                candidates.append(evicted)
                candidates.sort(key=lambda candidate: candidate.sort_key)
                self.cpu_bytes_retained += evicted.tensor_bytes
            raise MemoryError(
                "failure sample CPU tensor limit exceeded: requested "
                f"{payload_bytes} bytes with {self.cpu_bytes_retained} retained, "
                f"limit={self.cpu_limit_bytes}"
            )
        candidate = _FailureSampleCandidate(
            criterion=criterion,
            metric=metric,
            sequence_id=sequence_id,
            frame_id=frame_id,
            manifest_index=manifest_index,
            timestamp=timestamp,
            tensors_cpu=self._detach_payload_to_cpu(payload),
            tensor_bytes=payload_bytes,
        )
        candidates.append(candidate)
        candidates.sort(key=lambda item: item.sort_key)
        self.cpu_bytes_retained += payload_bytes

    def write(
        self,
        root: Path,
        *,
        checkpoint: Mapping[str, Any],
        evaluator: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write selection JSON and reuse the normal visualization bundle writer."""

        checkpoint_receipt = {
            key: checkpoint[key]
            for key in ("path", "checkpoint_sha256", "step", "git_hash")
        }
        evaluator_receipt = {
            key: evaluator[key]
            for key in ("git_hash", "eval_py_sha256", "evaluation_module_sha256")
        }
        report: dict[str, Any] = {
            "status": "COMPLETE",
            "samples_per_criterion": self.samples_per_criterion,
            "cpu_limit_bytes": self.cpu_limit_bytes,
            "cpu_bytes_retained": self.cpu_bytes_retained,
            "criteria": {},
        }
        for criterion, definition in self.criteria.items():
            criterion_root = root / criterion
            selected: list[dict[str, Any]] = []
            for rank, candidate in enumerate(self._candidates[criterion], start=1):
                bundle_name = _safe_failure_sample_stem(
                    rank=rank,
                    sequence_id=candidate.sequence_id,
                    frame_id=candidate.frame_id,
                    manifest_index=candidate.manifest_index,
                )
                tensors = candidate.tensors_cpu
                _save_visualization(
                    criterion_root,
                    sample_name=bundle_name,
                    rgb_hr=tensors["rgb_hr"],
                    K_hr_px=tensors["K_hr_px"],
                    baseline_m=tensors["baseline_m"],
                    baseline_hr_px=tensors["baseline_hr_px"],
                    output_hr_px=tensors["output_hr_px"],
                    target_hr_px=tensors["target_hr_px"],
                    target_trusted_mask=tensors["target_trusted_mask"],
                    source_weights_lr=tensors["source_weights_lr"],
                    uncertainty_hr=tensors["uncertainty_hr"],
                    vggt_disparity_hr_px=tensors["vggt_disparity_hr_px"],
                    vggt_valid_mask_hr=tensors["vggt_valid_mask_hr"],
                    history_disparity_hr_px=tensors["history_disparity_hr_px"],
                    history_valid_mask_hr=tensors["history_valid_mask_hr"],
                    vggt_off_output_hr_px=tensors["vggt_off_output_hr_px"],
                    no_vggt_output_hr_px=tensors["no_vggt_output_hr_px"],
                    prediction_filename="t3_vggt_disparity_hr_px.png",
                )
                selected.append(
                    {
                        "rank": rank,
                        "sequence_id": candidate.sequence_id,
                        "frame_id": candidate.frame_id,
                        "manifest_index": candidate.manifest_index,
                        "timestamp": candidate.timestamp,
                        "metric": _metric_result_dict(candidate.metric),
                        "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
                        "evaluator_eval_py_sha256": evaluator_receipt[
                            "eval_py_sha256"
                        ],
                        "evaluator_evaluation_module_sha256": evaluator_receipt[
                            "evaluation_module_sha256"
                        ],
                        "tensor_bytes": candidate.tensor_bytes,
                        "bundle_directory": str((criterion_root / bundle_name).resolve()),
                    }
                )
            selection = {
                "schema_version": 1,
                "status": "COMPLETE",
                "criterion": criterion,
                "definition": definition,
                "target": {
                    "type": PSEUDO_GT_LABEL,
                    "paper_gt": False,
                    "paper_accuracy": False,
                },
                "selection_order": (
                    "descending per-sample metric value; ties ascending sequence_id, "
                    "frame_id, manifest_index"
                ),
                "requested_samples": self.samples_per_criterion,
                "selected_samples": len(selected),
                "checkpoint": checkpoint_receipt,
                "evaluator": evaluator_receipt,
                "selected": selected,
            }
            criterion_root.mkdir(parents=True, exist_ok=True)
            (criterion_root / "selection.json").write_text(
                json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            report["criteria"][criterion] = {
                "selection_json": str((criterion_root / "selection.json").resolve()),
                "selected_samples": len(selected),
            }
        return report


def _t3_failure_sample_metrics(
    *,
    prediction_hr_px: Tensor,
    target_hr_px: Tensor,
    target_trusted_mask: Tensor,
    ffs_confidence_hr: Tensor,
    ffs_valid_mask_hr: Tensor,
    ffs_trusted_mask_hr: Tensor,
    history_hr_px: Tensor,
    strict_temporal_safe_mask: Tensor,
    warped_reference_history_hr_px: Tensor | None = None,
    warped_reference_valid_mask: Tensor | None = None,
    low_confidence_threshold: float,
    boundary_gradient_threshold_px: float,
    boundary_radius_px: int,
) -> dict[str, MetricResult]:
    """Compute the four D-025 failure scores for exactly one T3 endpoint."""

    spatial = compute_sample_metrics(
        prediction_hr_px,
        target_hr_px,
        target_trusted_mask=target_trusted_mask,
        ffs_confidence_hr=ffs_confidence_hr,
        ffs_valid_mask_hr=ffs_valid_mask_hr,
        ffs_trusted_mask_hr=ffs_trusted_mask_hr,
        low_confidence_threshold=low_confidence_threshold,
        boundary_gradient_threshold_px=boundary_gradient_threshold_px,
        boundary_radius_px=boundary_radius_px,
    )
    temporal_name = "strict_temporal_error_px"
    temporal_metric = temporal_disparity_error(
        prediction_hr_px,
        history_hr_px,
        safe_mask=strict_temporal_safe_mask,
    )
    if warped_reference_history_hr_px is not None:
        if warped_reference_valid_mask is None:
            raise ValueError(
                "warped_reference_valid_mask is required with a reference warp"
            )
        temporal_name = "strict_temporal_residual_error_px"
        temporal_metric = temporal_residual_error(
            prediction_hr_px,
            history_hr_px,
            target_hr_px,
            warped_reference_history_hr_px,
            safe_mask=strict_temporal_safe_mask,
            current_reference_valid_mask=target_trusted_mask,
            warped_previous_reference_valid_mask=warped_reference_valid_mask,
        )
    return {
        "raw_negative_rate": spatial["output_negative_rate"],
        "low_confidence_epe_px": spatial["low_confidence_epe_px"],
        "boundary_epe_px": spatial["boundary_epe_px"],
        temporal_name: temporal_metric,
    }


def _t3_failure_payload(
    *,
    batch: Mapping[str, Any],
    item_index: int,
    output_size_hw: tuple[int, int],
    endpoint_rgb: Tensor,
    endpoint_K_hr: Tensor,
    endpoint_baseline_m: Tensor,
    baseline_hr_px: Tensor,
    target_hr_px: Tensor,
    target_trusted_mask: Tensor,
    temporal_predictions: TemporalEndpointPredictions,
) -> dict[str, Tensor]:
    """Return only one selected sample's visualization tensors on their device."""

    vggt_lr = batch["disparity_vggt_hr_px_sequence"][item_index : item_index + 1, 2]
    vggt_valid_lr = batch["valid_vggt_sequence"][item_index : item_index + 1, 2]
    vggt_hr = functional.interpolate(
        vggt_lr,
        size=output_size_hw,
        mode="bilinear",
        align_corners=False,
    )[0]
    vggt_valid_hr = functional.interpolate(
        vggt_valid_lr.float(), size=output_size_hw, mode="nearest"
    )[0].bool()
    return {
        "rgb_hr": endpoint_rgb[item_index],
        "K_hr_px": endpoint_K_hr[item_index],
        "baseline_m": endpoint_baseline_m[item_index],
        "baseline_hr_px": baseline_hr_px[item_index].float(),
        "output_hr_px": temporal_predictions.vggt_on.disparity_hr_px[item_index].float(),
        "target_hr_px": target_hr_px[item_index].float(),
        "target_trusted_mask": target_trusted_mask[item_index],
        "source_weights_lr": temporal_predictions.vggt_on.source_weights[item_index].float(),
        "uncertainty_hr": temporal_predictions.vggt_on.uncertainty[item_index].float(),
        "vggt_disparity_hr_px": vggt_hr.float(),
        "vggt_valid_mask_hr": vggt_valid_hr,
        "history_disparity_hr_px": (
            temporal_predictions.shared_transport.disparity_history_loss_hr_px[item_index]
        ).float(),
        "history_valid_mask_hr": temporal_predictions.shared_transport.valid_history_hr[
            item_index
        ],
        "vggt_off_output_hr_px": temporal_predictions.source_mask_off.disparity_hr_px[
            item_index
        ].float(),
        "no_vggt_output_hr_px": temporal_predictions.no_vggt.disparity_hr_px[
            item_index
        ].float(),
    }


def _temporal_metric_contract(*, temporal_metric_v2: bool) -> dict[str, Any]:
    """Return the machine-readable definition of the primary temporal metric."""

    if temporal_metric_v2:
        return {
            "protocol_version": "teacher_gt_temporal_residual_v2",
            "primary": True,
            "reference": "trusted_hr_ffs_teacher_pseudo_gt",
            "paper_gt": False,
            "unit": "HR_pixel_disparity",
            "formula": (
                "mean(abs((d_hat_t-W(d_hat_t-1))-(d_star_t-W(d_star_t-1))))"
            ),
            "correspondence_contract": (
                "W(d_hat_t-1) and W(d_star_t-1) use the exact same teacher/GT "
                "top-K source correspondences and depth-ratio transport"
            ),
            "native_field": "temporal_residual_error_native_px",
            "paired_field": "temporal_residual_error_paired_px",
            "native_domain": (
                "method visible AND static AND non-collision AND geometry-consistent "
                "AND valid prediction history AND current trusted reference AND "
                "valid non-collision z-buffer reference history"
            ),
            "paired_domain": (
                "intersection of T1 and T3 native safety/reference domains"
            ),
            "constant_bias_semantics": (
                "a time-constant prediction bias cancels; prediction-only flicker "
                "is penalized"
            ),
            "legacy_fields_emitted": False,
        }
    return {
        "protocol_version": "current_vs_warped_history_legacy_v1",
        "primary": False,
        "reference": None,
        "paper_gt": False,
        "unit": "HR_pixel_disparity",
        "formula": "mean(abs(d_hat_t-W(d_hat_t-1)))",
        "native_field": "temporal_disparity_error_native_px",
        "paired_field": "temporal_disparity_error_paired_px",
        "warning": (
            "legacy diagnostic penalizes real temporal disparity change and is not "
            "teacher/GT temporal-residual TEPE"
        ),
        "legacy_fields_emitted": True,
    }


def _write_csv(
    path: Path,
    methods: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
) -> None:
    metric_names = sorted(
        {
            name
            for method in methods.values()
            for name, value in method.items()
            if isinstance(value, dict) and "value" in value
        }
    )
    fields = ["method", "target_type", "point_to_plane"]
    for name in metric_names:
        fields.extend((name, f"{name}_valid", f"{name}_count", f"{name}_numerator"))
    fields.extend(
        (
            "trusted_region_degradation_percent",
            "invalid_region_completeness_change_percent",
            "temporal_error_change_vs_t1_percent",
            "temporal_residual_error_change_vs_t1_percent",
            "vggt_prior_effect_epe_change_percent",
            "vggt_source_mask_ablation_change_percent",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name, method in methods.items():
            row: dict[str, Any] = {
                "method": method_name,
                "target_type": PSEUDO_GT_LABEL,
                "point_to_plane": "NOT_AVAILABLE",
            }
            for name in metric_names:
                metric = method.get(name)
                if not isinstance(metric, dict):
                    continue
                row[name] = metric["value"]
                row[f"{name}_valid"] = metric["valid"]
                row[f"{name}_count"] = metric["count"]
                row[f"{name}_numerator"] = metric["numerator"]
            if method_name == "bilinear":
                trusted = methods["bilinear"]["trusted_region_epe_px"]
                row["trusted_region_degradation_percent"] = (
                    0.0 if trusted["valid"] else None
                )
                row["invalid_region_completeness_change_percent"] = (
                    0.0
                    if methods["bilinear"]["invalid_region_completeness"]["valid"]
                    else None
                )
            else:
                # The flat fallback keeps Stage-A reports written by the
                # original evaluator compatible while Stage-B uses explicit
                # per-method comparison blocks.
                versus_bilinear = comparisons.get(
                    f"{method_name}_vs_bilinear", comparisons
                )
                row["trusted_region_degradation_percent"] = versus_bilinear[
                    "trusted_region_degradation"
                ]["relative_change_percent"]
                row["invalid_region_completeness_change_percent"] = versus_bilinear[
                    "invalid_region_completeness_change"
                ]["relative_change_percent"]
            temporal_comparison = comparisons.get(
                f"{method_name}_vs_T1_temporal"
            )
            if isinstance(temporal_comparison, dict):
                comparison_field = str(temporal_comparison.get("metric", ""))
                column = (
                    "temporal_residual_error_change_vs_t1_percent"
                    if comparison_field == "temporal_residual_error_paired_px"
                    else "temporal_error_change_vs_t1_percent"
                )
                row[column] = temporal_comparison["relative_change_percent"]
            vggt_comparison = comparisons.get(
                f"{method_name}_vs_T3_vggt_source_mask"
            )
            if isinstance(vggt_comparison, dict):
                row["vggt_source_mask_ablation_change_percent"] = vggt_comparison[
                    "relative_change_percent"
                ]
            if method_name == "T3_VGGT":
                row["vggt_prior_effect_epe_change_percent"] = comparisons[
                    "T3_VGGT_vs_T3_prior_effect"
                ]["relative_change_percent"]
            writer.writerow(row)


def _resolved_dict(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config is not a mapping")
    return value


def _validate_formal_temporal_coverage(
    dataset: CachedTemporalTrainingDataset,
) -> dict[str, Any]:
    """Require the complete formal derived endpoint set, never a subset cache."""

    receipt_path = dataset.derived_cache_root / "run_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read formal derived receipt {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("formal derived receipt is not a mapping")
    selection = receipt.get("selection")
    counts = receipt.get("counts")
    inputs = receipt.get("inputs")
    if not all(isinstance(value, dict) for value in (selection, counts, inputs)):
        raise ValueError("formal derived receipt coverage fields are missing")
    if selection.get("start_window") != 0 or selection.get("limit") is not None:
        raise ValueError("evaluation refuses a subset-derived cache receipt")
    candidates = build_causal_windows(
        dataset.records,
        student_sequence_length=3,
        vggt_context_pairs=5,
    )
    expected_endpoint_indices = {window.endpoint_index for window in candidates}
    actual_endpoint_indices = set(dataset.derived_entries)
    if actual_endpoint_indices != expected_endpoint_indices:
        missing = sorted(expected_endpoint_indices - actual_endpoint_indices)
        extra = sorted(actual_endpoint_indices - expected_endpoint_indices)
        raise ValueError(
            "derived endpoint coverage is incomplete or non-canonical: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    derived_count = len(expected_endpoint_indices)
    if (
        selection.get("selected_windows") != derived_count
        or counts.get("selected") != derived_count
        or inputs.get("vggt_available_windows") != derived_count
    ):
        raise ValueError("derived receipt does not cover every formal VGGT endpoint")
    raw_manifest_path_value = inputs.get("vggt_cache_manifest")
    raw_manifest_sha256 = inputs.get("vggt_cache_manifest_sha256")
    if not isinstance(raw_manifest_path_value, str) or not isinstance(
        raw_manifest_sha256, str
    ):
        raise ValueError("derived receipt is not bound to the raw VGGT manifest")
    raw_manifest_path = Path(raw_manifest_path_value).expanduser().resolve()
    if not raw_manifest_path.is_file() or sha256_file(
        raw_manifest_path
    ) != raw_manifest_sha256:
        raise ValueError("derived/raw VGGT cache-manifest SHA-256 mismatch")
    expected_evaluable = [
        window
        for window in candidates
        if all(index in expected_endpoint_indices for index in window.student_indices)
    ]
    expected_windows = [
        (window.endpoint_index, window.student_indices) for window in expected_evaluable
    ]
    actual_windows = [
        (window.endpoint_index, window.student_indices) for window in dataset.windows
    ]
    if actual_windows != expected_windows:
        raise ValueError("temporal dataset window set is not the complete causal set")
    return {
        "manifest_records": len(dataset.records),
        "derived_endpoint_records": derived_count,
        "evaluable_t3_windows": len(expected_evaluable),
        "derived_run_receipt_path": str(receipt_path.resolve()),
        "derived_run_receipt_sha256": sha256_file(receipt_path),
        "derived_cache_manifest_path": str(
            (dataset.derived_cache_root / "cache_manifest.jsonl").resolve()
        ),
        "derived_cache_manifest_sha256": sha256_file(
            dataset.derived_cache_root / "cache_manifest.jsonl"
        ),
        "raw_vggt_cache_manifest_path": str(raw_manifest_path),
        "raw_vggt_cache_manifest_sha256": raw_manifest_sha256,
    }


def checkpoint_training_completion(
    checkpoint_metadata: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Classify execution completion separately from canonical finality.

    A checkpoint that completed a shortened configured run is still not the
    canonical Stage-A/Stage-B checkpoint. Likewise, a 7,500-step snapshot from
    a 15,000-step Stage-B run is an intermediate artifact even when evaluated
    over the complete held-out corpus.
    """

    if stage not in {"spatial", "temporal"}:
        raise ValueError("stage must be spatial or temporal")
    actual_step = checkpoint_metadata.get("step")
    config = checkpoint_metadata.get("training_config")
    train_config = config.get("train") if isinstance(config, dict) else None
    if (
        isinstance(actual_step, bool)
        or not isinstance(actual_step, int)
        or actual_step < 0
        or not isinstance(train_config, dict)
    ):
        raise ValueError("checkpoint training completion metadata is malformed")

    configured_field = "steps_spatial" if stage == "spatial" else "steps"
    configured_steps = train_config.get(configured_field)
    declared_schedule_field = (
        "steps_spatial" if stage == "spatial" else "steps_temporal"
    )
    declared_schedule_steps = train_config.get(declared_schedule_field)
    for name, value in (
        (configured_field, configured_steps),
        (declared_schedule_field, declared_schedule_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint train.{name} is malformed")

    canonical_steps = (
        FORMAL_STAGE_A_TRAINING_STEPS
        if stage == "spatial"
        else FORMAL_STAGE_B_TRAINING_STEPS
    )
    execution_complete = actual_step == configured_steps
    canonical_schedule = (
        configured_steps == canonical_steps
        and declared_schedule_steps == canonical_steps
    )
    final_training_checkpoint = execution_complete and canonical_schedule
    return {
        "stage": stage,
        "actual_step": actual_step,
        "configured_steps_field": f"train.{configured_field}",
        "configured_steps": configured_steps,
        "declared_schedule_field": f"train.{declared_schedule_field}",
        "declared_schedule_steps": declared_schedule_steps,
        "canonical_steps": canonical_steps,
        "execution_complete": execution_complete,
        "canonical_schedule": canonical_schedule,
        "final_training_checkpoint": final_training_checkpoint,
    }


def evaluation_eligibility_status(
    *,
    stage: str,
    full_selection: bool,
    allow_non_holdout_smoke: bool,
    formal_holdout: bool | None,
    checkpoint_completion: dict[str, Any],
    spatial_checkpoint_completion: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate corpus coverage from final-checkpoint acceptance eligibility."""

    if stage not in {"spatial", "temporal"}:
        raise ValueError("stage must be spatial or temporal")
    if checkpoint_completion.get("stage") != stage:
        raise ValueError("primary checkpoint completion stage mismatch")
    if stage == "temporal" and (
        not isinstance(spatial_checkpoint_completion, dict)
        or spatial_checkpoint_completion.get("stage") != "spatial"
    ):
        raise ValueError("temporal evaluation requires Stage-A completion metadata")
    coverage_eligible = bool(
        full_selection
        and not allow_non_holdout_smoke
        and (stage == "spatial" or formal_holdout is True)
    )
    final_training_checkpoint = bool(
        checkpoint_completion.get("final_training_checkpoint") is True
        and (
            stage == "spatial"
            or spatial_checkpoint_completion.get("final_training_checkpoint")
            is True
        )
    )
    final_acceptance_eligible = bool(
        coverage_eligible and final_training_checkpoint
    )
    if allow_non_holdout_smoke:
        status = "NON_HOLDOUT_SMOKE_COMPLETE"
    elif final_acceptance_eligible:
        status = "FINAL_CHECKPOINT_EVALUATION_COMPLETE"
    elif coverage_eligible:
        status = "INTERMEDIATE_CHECKPOINT_EVALUATION_COMPLETE"
    else:
        status = "LIMITED_EVALUATION_COMPLETE"
    return {
        "coverage_eligible": coverage_eligible,
        "final_training_checkpoint": final_training_checkpoint,
        "final_acceptance_eligible": final_acceptance_eligible,
        "status": status,
    }


def _materialize_checkpoint_cache_identities(
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Recover legacy smoke identities from their exact bound receipts."""

    config = checkpoint_metadata.get("training_config")
    data = config.get("data") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        return checkpoint_metadata
    missing = [
        name
        for name in ("observation_cache_identity", "teacher_cache_identity")
        if not isinstance(data.get(name), dict)
    ]
    if not missing:
        return checkpoint_metadata
    manifest_value = data.get("manifest_path")
    observation_value = data.get("observation_cache_root")
    teacher_value = data.get("teacher_cache_root")
    if not all(
        isinstance(value, str)
        for value in (manifest_value, observation_value, teacher_value)
    ):
        return checkpoint_metadata
    manifest_path = Path(manifest_value).expanduser().resolve()
    recovered = {
        "observation_cache_identity": load_receipt_identity(
            observation_value,
            expected_component="ffs-observation",
            manifest_path=manifest_path,
        ).to_dict(),
        "teacher_cache_identity": load_receipt_identity(
            teacher_value,
            expected_component="ffs-teacher",
            manifest_path=manifest_path,
        ).to_dict(),
    }
    metadata = copy.deepcopy(checkpoint_metadata)
    metadata["training_config"]["data"].update(recovered)
    metadata["cache_identity_recovery"] = {
        "status": "RECOVERED_FROM_EXACT_TRAINING_RECEIPTS",
        "fields": missing,
        "manifest_path": str(manifest_path),
    }
    return metadata


def _validated_raw_vggt_receipt(
    raw_vggt_root: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    receipt_path = raw_vggt_root / "run_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read raw VGGT receipt {receipt_path}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("raw VGGT receipt schema is incompatible")
    identity = receipt.get("identity")
    config = receipt.get("config")
    if not isinstance(identity, dict) or not isinstance(config, dict):
        raise ValueError("raw VGGT receipt identity/config is missing")
    required_identity = {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
    if set(identity) != required_identity or identity.get("component") != "vggt-omega":
        raise ValueError("raw VGGT cache identity is malformed")
    expected_view_order = [
        label
        for time_label in ("t-4", "t-3", "t-2", "t-1", "t")
        for label in (f"L[{time_label}]", f"R[{time_label}]")
    ]
    if (
        config.get("causal") is not True
        or config.get("context_pairs") != 5
        or config.get("current_left_view_index") != 8
        or config.get("view_order") != expected_view_order
    ):
        raise ValueError("raw VGGT cache is not the strict causal five-pair layout")
    selected = receipt.get("selected_windows")
    available = receipt.get("available_windows")
    written = receipt.get("written_records")
    reused = receipt.get("reused_records")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (selected, available, written, reused)
    ):
        raise ValueError("raw VGGT receipt counts are malformed")
    if selected != available or written + reused != selected:
        raise ValueError("raw VGGT cache is incomplete")
    if receipt.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("raw VGGT receipt is bound to a different manifest")
    return {
        "root": str(raw_vggt_root.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "identity": dict(identity),
        "config": dict(config),
        "selected_windows": selected,
        "available_windows": available,
        "manifest_sha256": expected_manifest_sha256,
    }


def _audit_temporal_holdout_and_raw_lineage(
    *,
    checkpoint_metadata: dict[str, Any],
    dataset: CachedTemporalTrainingDataset,
    evaluation_manifest_path: Path,
    observation_identity: dict[str, Any],
    allow_non_holdout_smoke: bool = False,
) -> dict[str, Any]:
    """Verify video isolation and raw VGGT identity through derived records."""

    training_config = checkpoint_metadata.get("training_config")
    if not isinstance(training_config, dict) or not isinstance(
        training_config.get("data"), dict
    ):
        raise ValueError("temporal checkpoint data lineage is missing")
    training_data = training_config["data"]
    training_manifest_value = training_data.get("manifest_path")
    if not isinstance(training_manifest_value, str):
        raise ValueError("temporal checkpoint has no training manifest path")
    training_manifest_path = Path(training_manifest_value).expanduser().resolve()
    if not training_manifest_path.is_file():
        raise FileNotFoundError(
            f"training manifest provenance is unavailable: {training_manifest_path}"
        )
    same_manifest = training_manifest_path == evaluation_manifest_path.resolve()
    if same_manifest and not allow_non_holdout_smoke:
        raise ValueError("evaluation manifest is the temporal training manifest")
    training_sequences = {
        record.sequence_id for record in load_manifest(training_manifest_path)
    }
    evaluation_sequences = {record.sequence_id for record in dataset.records}
    overlap = sorted(training_sequences & evaluation_sequences)
    if overlap and not allow_non_holdout_smoke:
        raise ValueError(f"training/evaluation sequences overlap: {overlap}")
    for field_name, current_root in (
        ("observation_cache_root", dataset.observation_cache_root),
        ("teacher_cache_root", dataset.spatial_dataset.teacher_cache_root),
    ):
        saved_root = training_data.get(field_name)
        if not isinstance(saved_root, str):
            raise ValueError(f"checkpoint training {field_name} is missing")
        if (
            Path(saved_root).expanduser().resolve() == current_root
            and not allow_non_holdout_smoke
        ):
            raise ValueError(f"evaluation reuses the training {field_name}")

    saved_derived = training_data.get("derived_cache_lineage")
    if not isinstance(saved_derived, dict) or not isinstance(
        saved_derived.get("derived_cache_root"), str
    ):
        raise ValueError("checkpoint training derived-cache root is missing")
    training_derived_root = Path(
        saved_derived["derived_cache_root"]
    ).expanduser().resolve()
    training_raw_root = training_derived_root.parent / "vggt"
    evaluation_raw_root = dataset.derived_cache_root.parent / "vggt"
    training_raw = _validated_raw_vggt_receipt(
        training_raw_root,
        expected_manifest_sha256=sha256_file(training_manifest_path),
    )
    evaluation_raw = _validated_raw_vggt_receipt(
        evaluation_raw_root,
        expected_manifest_sha256=sha256_file(evaluation_manifest_path),
    )
    if training_raw["identity"] != evaluation_raw["identity"]:
        raise ValueError("training/evaluation raw VGGT identities differ")
    if training_raw["config"] != evaluation_raw["config"]:
        raise ValueError("training/evaluation raw VGGT inference configs differ")

    # mmap avoids paging dense tensor storage while auditing every formal
    # derived record's raw-lineage metadata.
    for entry in dataset.derived_entries.values():
        if sha256_file(entry.cache_path) != entry.cache_sha256:
            raise ValueError(
                f"derived cache SHA-256 differs from its manifest: {entry.cache_path}"
            )
        payload = torch.load(
            entry.cache_path, map_location="cpu", weights_only=True, mmap=True
        )
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        source = metadata.get("source") if isinstance(metadata, dict) else None
        linkage = source.get("linkage") if isinstance(source, dict) else None
        if not isinstance(linkage, dict):
            raise ValueError(f"derived raw linkage missing: {entry.cache_path}")
        if linkage.get("vggt_raw_identity") != evaluation_raw["identity"]:
            raise ValueError(
                f"derived record uses a different raw VGGT identity: {entry.cache_path}"
            )
        if linkage.get("ffs_raw_identity") != observation_identity:
            raise ValueError(
                f"derived record uses a different raw FFS identity: {entry.cache_path}"
            )
    return {
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "evaluation_manifest_path": str(evaluation_manifest_path.resolve()),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "training_sequences": sorted(training_sequences),
        "evaluation_sequences": sorted(evaluation_sequences),
        "sequence_overlap": overlap,
        "same_manifest": same_manifest,
        "formal_holdout": not same_manifest and not overlap,
        "non_holdout_smoke_override": allow_non_holdout_smoke,
        "training_raw_vggt": training_raw,
        "evaluation_raw_vggt": evaluation_raw,
        "audited_evaluation_derived_records": len(dataset.derived_entries),
    }


def run(args: argparse.Namespace) -> int:
    config = resolve_evaluation_config(args.config, args.overrides)
    _update_cli_values(config, args)
    stage = validate_evaluation_config(config)
    temporal_metric_v2 = temporal_residual_v2_from_config(config).enabled
    physical_metric_v2 = physical_output_v2_from_config(config).enabled
    stage_label = "T1_SPATIAL_ONLY" if stage == "spatial" else "T3_CAUSAL_STAGE_B"
    seed_everything(int(config.seed), deterministic=True)
    model = build_model(config)
    parameter_count = count_trainable_parameters(model)
    if parameter_count <= 0 or parameter_count >= 12_000_000:
        raise ValueError(
            f"trainable parameter count must be in (0,12M), got {parameter_count}"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "stage": stage_label,
                    "target_type": PSEUDO_GT_LABEL,
                    "parameter_count": parameter_count,
                    "crop_mode": str(config.eval.crop_mode),
                    "causal_endpoint_only": stage == "temporal",
                    "future_frames_allowed": False,
                    "resolved_config": _resolved_dict(config),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry-run is used")
    if args.allow_non_holdout_smoke:
        if stage != "temporal":
            raise ValueError("--allow-non-holdout-smoke is only valid for Stage B")
        if config.eval.limit is None or int(config.eval.limit) > 4:
            raise ValueError(
                "non-holdout smoke requires an explicit eval.limit in [1,4]"
            )
    manifest_path = _required_path(config, "data.manifest_path", directory=False)
    observation_root = _required_path(
        config, "data.observation_cache_root", directory=True
    )
    teacher_root = _required_path(config, "data.teacher_cache_root", directory=True)
    output_value = config.eval.output_dir
    output_dir = (
        PROJECT_ROOT / "outputs" / str(config.experiment) / "evaluation"
        if output_value is None or not str(output_value).strip()
        else Path(str(output_value)).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    full_resolution = str(config.eval.crop_mode) == "full"
    crop_size = None if full_resolution else tuple(int(v) for v in config.data.hr_crop)
    origin_value = config.eval.fixed_crop_origin_hr_xy
    fixed_origin = (
        None if origin_value is None else tuple(int(value) for value in origin_value)
    )
    derived_lineage: dict[str, Any] | None = None
    formal_coverage: dict[str, int] | None = None
    if stage == "spatial":
        dataset: CachedFFSTrainingDataset | CachedTemporalTrainingDataset = (
            CachedFFSTrainingDataset(
                manifest_path=manifest_path,
                observation_cache_root=observation_root,
                teacher_cache_root=teacher_root,
                observation_identity=observation_identity,
                teacher_identity=teacher_identity,
                crop_size_hr_hw=crop_size,
                crop_mode="fixed",
                fixed_crop_origin_hr_xy=fixed_origin,
                spatial_scale=int(config.data.scale),
                seed=int(config.seed),
            )
        )
        collate_function = collate_training_samples
    else:
        derived_root = _required_path(
            config, "data.derived_geometry_cache_root", directory=True
        )
        dataset = CachedTemporalTrainingDataset(
            manifest_path=manifest_path,
            observation_cache_root=observation_root,
            teacher_cache_root=teacher_root,
            derived_cache_root=derived_root,
            observation_identity=observation_identity,
            teacher_identity=teacher_identity,
            crop_size_hr_hw=crop_size,
            crop_mode="fixed",
            fixed_crop_origin_hr_xy=fixed_origin,
            spatial_scale=int(config.data.scale),
            student_sequence_length=3,
            vggt_context_pairs=5,
            seed=int(config.seed),
        )
        formal_coverage = _validate_formal_temporal_coverage(dataset)
        derived_lineage = dict(dataset.cache_lineage_summary)
        collate_function = collate_temporal_training_samples
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    start_index = int(config.eval.start)
    if start_index >= len(dataset):
        raise ValueError(
            f"eval.start={start_index} is outside dataset length {len(dataset)}"
        )
    requested_limit = config.eval.limit
    available_count = len(dataset) - start_index
    sample_count = (
        available_count
        if requested_limit is None
        else min(int(requested_limit), available_count)
    )
    if sample_count <= 0:
        raise ValueError("evaluation selects no validation records")
    selected_dataset = Subset(
        dataset, range(start_index, start_index + sample_count)
    )
    device = _resolve_device(args.device)
    pin_memory = bool(config.eval.pin_memory) and device.type == "cuda"
    loader = DataLoader(
        selected_dataset,
        batch_size=int(config.eval.batch_size),
        shuffle=False,
        num_workers=int(config.eval.num_workers),
        persistent_workers=int(config.eval.num_workers) > 0,
        pin_memory=pin_memory,
        collate_fn=collate_function,
    )

    checkpoint_metadata = load_model_for_evaluation(
        args.checkpoint,
        model,
        expected_parameter_count=parameter_count,
        require_full_training_state=True,
    )
    checkpoint_metadata = _materialize_checkpoint_cache_identities(
        checkpoint_metadata
    )
    checkpoint_lineage = validate_checkpoint_lineage(
        checkpoint_metadata,
        required_stage=stage,
        observation_cache_identity=observation_identity.to_dict(),
        teacher_cache_identity=teacher_identity.to_dict(),
        derived_cache_lineage=derived_lineage,
        evaluation_config=_resolved_dict(config) if stage == "temporal" else None,
    )
    holdout_raw_lineage: dict[str, Any] | None = None
    if stage == "temporal":
        assert isinstance(dataset, CachedTemporalTrainingDataset)
        holdout_raw_lineage = _audit_temporal_holdout_and_raw_lineage(
            checkpoint_metadata=checkpoint_metadata,
            dataset=dataset,
            evaluation_manifest_path=manifest_path,
            observation_identity=observation_identity.to_dict(),
            allow_non_holdout_smoke=args.allow_non_holdout_smoke,
        )
    spatial_model: torch.nn.Module | None = None
    spatial_checkpoint_metadata: dict[str, Any] | None = None
    spatial_checkpoint_lineage: dict[str, Any] | None = None
    if stage == "temporal":
        recorded_spatial_path = Path(
            str(checkpoint_lineage["stage_a_initialization_path"])
        ).expanduser()
        spatial_checkpoint_path = (
            args.spatial_checkpoint.expanduser().resolve()
            if args.spatial_checkpoint is not None
            else recorded_spatial_path.resolve()
        )
        if not spatial_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Stage-A checkpoint from temporal lineage is unavailable; pass "
                f"--spatial-checkpoint with the exact bound artifact: "
                f"{spatial_checkpoint_path}"
            )
        spatial_model = build_model(config)
        spatial_checkpoint_metadata = load_model_for_evaluation(
            spatial_checkpoint_path,
            spatial_model,
            expected_parameter_count=parameter_count,
            require_full_training_state=True,
        )
        spatial_checkpoint_metadata = _materialize_checkpoint_cache_identities(
            spatial_checkpoint_metadata
        )
        validate_spatial_checkpoint_binding(
            spatial_checkpoint_metadata, checkpoint_lineage
        )
        spatial_checkpoint_lineage = validate_checkpoint_lineage(
            spatial_checkpoint_metadata,
            required_stage="spatial",
            observation_cache_identity=observation_identity.to_dict(),
            teacher_cache_identity=teacher_identity.to_dict(),
        )
        spatial_model.to(device=device).eval()
    model.to(device=device).eval()
    base_method_names = (
        ("bilinear", "T1")
        if stage == "spatial"
        else (
            "bilinear",
            "T1",
            "T3",
            "T3_VGGT",
            "T3_VGGT_mask_off",
        )
    )
    method_names = tuple(
        method_name
        for base_name in base_method_names
        for method_name in (base_name, f"{base_name}_clamp0")
    )
    accumulators = {
        method_name: MethodMetricAccumulator() for method_name in method_names
    }
    visualization_limit = min(int(config.eval.visualization_samples), sample_count)
    visualized = 0
    visualization_records: list[dict[str, Any]] = []
    temporal_flicker_collector: TemporalFlickerVideoCollector | None = None
    if config.eval.temporal_flicker_video:
        temporal_flicker_collector = TemporalFlickerVideoCollector(
            output_dir / "temporal_flicker_videos",
            enabled=True,
            fps=int(config.eval.temporal_flicker_video_fps),
            disparity_range_hr_px=_fixed_display_range(
                config.eval.temporal_flicker_disparity_range_hr_px,
                "eval.temporal_flicker_disparity_range_hr_px",
            ),
            error_range_hr_px=_fixed_display_range(
                config.eval.temporal_flicker_error_range_hr_px,
                "eval.temporal_flicker_error_range_hr_px",
            ),
            uncertainty_range=_fixed_display_range(
                config.eval.temporal_flicker_uncertainty_range,
                "eval.temporal_flicker_uncertainty_range",
            ),
        )
    failure_sample_collector: FailureSampleCollector | None = None
    if int(config.eval.failure_samples_per_criterion) > 0:
        failure_sample_collector = FailureSampleCollector(
            samples_per_criterion=int(config.eval.failure_samples_per_criterion),
            cpu_limit_bytes=int(config.eval.failure_samples_cpu_limit_bytes),
            criteria=(
                TEMPORAL_RESIDUAL_V2_FAILURE_SAMPLE_CRITERIA
                if temporal_metric_v2
                else FAILURE_SAMPLE_CRITERIA
            ),
        )
    endpoint_pose_valid_count = 0
    endpoint_static_prior_valid_count = 0
    t3_vggt_sign_health = (
        T3VGGTSignHealthAccumulator() if stage == "temporal" else None
    )
    started = time.perf_counter()

    use_cuda_bf16 = (
        device.type == "cuda" and str(config.eval.precision).lower() == "bf16"
    )
    with _cleanup_temporal_flicker_on_abort(temporal_flicker_collector), torch.inference_mode():
        for batch in loader:
            if stage == "temporal":
                validate_temporal_batch_causality(batch)
                if temporal_metric_v2:
                    validate_v2_temporal_calibration(
                        batch["K_hr_sequence"], batch["baseline_m_sequence"]
                    )
            batch = _move_batch(batch, device)
            if stage == "spatial":
                target = batch["teacher_disparity_hr_px"]
                target_valid = batch["teacher_valid_mask"]
                target_trusted = batch["teacher_trusted_mask"]
                endpoint_rgb = batch["rgb_hr"]
                endpoint_ffs_disparity = batch["observation_disparity_hr_px"]
                endpoint_ffs_confidence = batch["observation_confidence"]
                endpoint_ffs_valid = batch["observation_valid_mask"]
                endpoint_ffs_trusted = batch["observation_trusted_mask"]
                endpoint_K_hr = batch["K_hr"]
                endpoint_baseline_m = batch["baseline_m"]
            else:
                target = batch["teacher_disparity_hr_px_sequence"][:, 2]
                target_valid = batch["teacher_valid_mask_sequence"][:, 2]
                target_trusted = batch["teacher_trusted_mask_sequence"][:, 2]
                endpoint_rgb = batch["rgb_hr_sequence"][:, 2]
                endpoint_ffs_disparity = batch[
                    "observation_disparity_hr_px_sequence"
                ][:, 2]
                endpoint_ffs_confidence = batch[
                    "observation_confidence_sequence"
                ][:, 2]
                endpoint_ffs_valid = batch["observation_valid_mask_sequence"][:, 2]
                endpoint_ffs_trusted = batch[
                    "observation_trusted_mask_sequence"
                ][:, 2]
                endpoint_K_hr = batch["K_hr_sequence"][:, 2]
                endpoint_baseline_m = batch["baseline_m_sequence"][:, 2]
                endpoint_pose_valid_count += int(
                    batch["temporal_pose_valid_sequence"][:, 2].sum().item()
                )
                endpoint_static_prior_valid_count += int(
                    batch["static_prior_valid_sequence"][:, 2].sum().item()
                )
            if not all(
                isinstance(value, Tensor)
                for value in (target, target_valid, target_trusted)
            ):
                raise ValueError(
                    "evaluation requires teacher disparity, valid, and trusted masks"
                )
            output_size = tuple(int(value) for value in target.shape[-2:])
            baseline, confidence_hr, valid_hr, trusted_hr = upsample_ffs_inputs_to_hr(
                endpoint_ffs_disparity,
                endpoint_ffs_confidence,
                endpoint_ffs_valid,
                endpoint_ffs_trusted,
                output_size_hw=output_size,
            )
            # Construct a fresh context manager per batch; this remains safe
            # for context-manager implementations that are not re-entrant.
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda_bf16
                else nullcontext()
            )
            with autocast_context:
                if stage == "spatial":
                    model_output = model(
                        endpoint_rgb,
                        endpoint_ffs_disparity,
                        endpoint_ffs_confidence,
                        valid_ffs=endpoint_ffs_valid,
                    )
                    raw_predictions = {
                        "bilinear": baseline.float(),
                        "T1": model_output.disparity_hr_px.float(),
                    }
                    model_outputs_by_method: dict[str, ModelOutput] = {
                        "T1": model_output
                    }
                    temporal_results: dict[str, MetricResult] = {}
                    temporal_visualization: TemporalEndpointPredictions | None = None
                else:
                    assert spatial_model is not None
                    spatial_endpoint = _run_spatial_endpoint(
                        spatial_model, batch, config=config
                    )
                    temporal_visualization = _run_temporal_endpoint_ablation(
                        model, batch, config=config
                    )
                    raw_predictions = {
                        "bilinear": baseline.float(),
                        "T1": spatial_endpoint.output.disparity_hr_px.float(),
                        "T3": temporal_visualization.no_vggt.disparity_hr_px.float(),
                        "T3_VGGT": (
                            temporal_visualization.vggt_on.disparity_hr_px.float()
                        ),
                        "T3_VGGT_mask_off": (
                            temporal_visualization.source_mask_off.disparity_hr_px.float()
                        ),
                    }
                    model_outputs_by_method = {
                        "T1": spatial_endpoint.output,
                        "T3": temporal_visualization.no_vggt,
                        "T3_VGGT": temporal_visualization.vggt_on,
                        "T3_VGGT_mask_off": (
                            temporal_visualization.source_mask_off
                        ),
                    }
                    t1_transport = spatial_endpoint.transport
                    t3_transport = temporal_visualization.shared_transport
                    mask_off_transport = (
                        temporal_visualization.source_mask_off_transport
                    )
                    no_vggt_transport = temporal_visualization.no_vggt_transport
                    t1_safe = hr_temporal_safe_mask(
                        raw_predictions["T1"],
                        visibility_mask_hr=t1_transport.visibility_mask_hr,
                        static_mask_hr=t1_transport.static_mask_hr,
                        collision_mask_hr=t1_transport.collision_mask_hr,
                        geometry_consistent_mask_hr=(
                            t1_transport.geometry_consistent_mask_hr
                        ),
                        valid_history_hr=t1_transport.valid_history_hr,
                    )
                    t3_safe = hr_temporal_safe_mask(
                        raw_predictions["T3_VGGT"],
                        visibility_mask_hr=t3_transport.visibility_mask_hr,
                        static_mask_hr=t3_transport.static_mask_hr,
                        collision_mask_hr=t3_transport.collision_mask_hr,
                        geometry_consistent_mask_hr=(
                            t3_transport.geometry_consistent_mask_hr
                        ),
                        valid_history_hr=t3_transport.valid_history_hr,
                    )
                    no_vggt_safe = hr_temporal_safe_mask(
                        raw_predictions["T3"],
                        visibility_mask_hr=no_vggt_transport.visibility_mask_hr,
                        static_mask_hr=no_vggt_transport.static_mask_hr,
                        collision_mask_hr=no_vggt_transport.collision_mask_hr,
                        geometry_consistent_mask_hr=(
                            no_vggt_transport.geometry_consistent_mask_hr
                        ),
                        valid_history_hr=no_vggt_transport.valid_history_hr,
                    )
                    mask_off_safe = hr_temporal_safe_mask(
                        raw_predictions["T3_VGGT_mask_off"],
                        visibility_mask_hr=mask_off_transport.visibility_mask_hr,
                        static_mask_hr=mask_off_transport.static_mask_hr,
                        collision_mask_hr=mask_off_transport.collision_mask_hr,
                        geometry_consistent_mask_hr=(
                            mask_off_transport.geometry_consistent_mask_hr
                        ),
                        valid_history_hr=mask_off_transport.valid_history_hr,
                    )
                    paired_t1_no_vggt = t1_safe & no_vggt_safe
                    if temporal_metric_v2:
                        t1_reference = spatial_endpoint.reference_transport
                        t3_reference = temporal_visualization.reference_transport
                        mask_off_reference = (
                            temporal_visualization.source_mask_off_reference_transport
                        )
                        no_vggt_reference = (
                            temporal_visualization.no_vggt_reference_transport
                        )
                        if any(
                            value is None
                            for value in (
                                t1_reference,
                                t3_reference,
                                mask_off_reference,
                                no_vggt_reference,
                            )
                        ):
                            raise RuntimeError(
                                "V2 evaluation did not return teacher reference warps"
                            )
                        assert (
                            t1_reference is not None
                            and t3_reference is not None
                            and mask_off_reference is not None
                            and no_vggt_reference is not None
                        )
                        current_reference_valid = (
                            target_valid.to(dtype=torch.bool)
                            & target_trusted.to(dtype=torch.bool)
                            & torch.isfinite(target)
                            & (target > 0)
                        )
                        paired_reference_domain = (
                            paired_t1_no_vggt
                            & t1_reference.valid_mask_hr
                            & no_vggt_reference.valid_mask_hr
                        )
                        temporal_results = {
                            "bilinear": MetricResult.invalid(),
                            "T1": _temporal_residual_metric_for_transport(
                                raw_predictions["T1"],
                                t1_transport,
                                current_reference_hr_px=target.float(),
                                current_reference_valid_hr=current_reference_valid,
                                reference_transport=t1_reference,
                            ),
                            "T3": _temporal_residual_metric_for_transport(
                                raw_predictions["T3"],
                                no_vggt_transport,
                                current_reference_hr_px=target.float(),
                                current_reference_valid_hr=current_reference_valid,
                                reference_transport=no_vggt_reference,
                            ),
                            "T3_VGGT": _temporal_residual_metric_for_transport(
                                raw_predictions["T3_VGGT"],
                                t3_transport,
                                current_reference_hr_px=target.float(),
                                current_reference_valid_hr=current_reference_valid,
                                reference_transport=t3_reference,
                            ),
                            "T3_VGGT_mask_off": (
                                _temporal_residual_metric_for_transport(
                                    raw_predictions["T3_VGGT_mask_off"],
                                    mask_off_transport,
                                    current_reference_hr_px=target.float(),
                                    current_reference_valid_hr=(
                                        current_reference_valid
                                    ),
                                    reference_transport=mask_off_reference,
                                )
                            ),
                        }
                        paired_temporal_results = {
                            "T1": _temporal_residual_metric_for_transport(
                                raw_predictions["T1"],
                                t1_transport,
                                current_reference_hr_px=target.float(),
                                current_reference_valid_hr=current_reference_valid,
                                reference_transport=t1_reference,
                                paired_domain_mask_hr=paired_reference_domain,
                            ),
                            "T3": _temporal_residual_metric_for_transport(
                                raw_predictions["T3"],
                                no_vggt_transport,
                                current_reference_hr_px=target.float(),
                                current_reference_valid_hr=current_reference_valid,
                                reference_transport=no_vggt_reference,
                                paired_domain_mask_hr=paired_reference_domain,
                            ),
                            "T3_VGGT": MetricResult.invalid(),
                            "T3_VGGT_mask_off": MetricResult.invalid(),
                        }
                    else:
                        temporal_results = {
                            "bilinear": MetricResult.invalid(),
                            "T1": hr_temporal_metric(
                                raw_predictions["T1"],
                                t1_transport.disparity_history_loss_hr_px,
                                visibility_mask_hr=t1_transport.visibility_mask_hr,
                                static_mask_hr=t1_transport.static_mask_hr,
                                collision_mask_hr=t1_transport.collision_mask_hr,
                                geometry_consistent_mask_hr=(
                                    t1_transport.geometry_consistent_mask_hr
                                ),
                                valid_history_hr=t1_transport.valid_history_hr,
                            ),
                            "T3": hr_temporal_metric(
                                raw_predictions["T3"],
                                no_vggt_transport.disparity_history_loss_hr_px,
                                visibility_mask_hr=(
                                    no_vggt_transport.visibility_mask_hr
                                ),
                                static_mask_hr=no_vggt_transport.static_mask_hr,
                                collision_mask_hr=(
                                    no_vggt_transport.collision_mask_hr
                                ),
                                geometry_consistent_mask_hr=(
                                    no_vggt_transport.geometry_consistent_mask_hr
                                ),
                                valid_history_hr=no_vggt_transport.valid_history_hr,
                            ),
                            "T3_VGGT": hr_temporal_metric(
                                raw_predictions["T3_VGGT"],
                                t3_transport.disparity_history_loss_hr_px,
                                visibility_mask_hr=t3_transport.visibility_mask_hr,
                                static_mask_hr=t3_transport.static_mask_hr,
                                collision_mask_hr=t3_transport.collision_mask_hr,
                                geometry_consistent_mask_hr=(
                                    t3_transport.geometry_consistent_mask_hr
                                ),
                                valid_history_hr=t3_transport.valid_history_hr,
                            ),
                            "T3_VGGT_mask_off": temporal_disparity_error(
                                raw_predictions["T3_VGGT_mask_off"],
                                mask_off_transport.disparity_history_loss_hr_px,
                                safe_mask=mask_off_safe,
                            ),
                        }
                        paired_temporal_results = {
                            "T1": temporal_disparity_error(
                                raw_predictions["T1"],
                                t1_transport.disparity_history_loss_hr_px,
                                safe_mask=paired_t1_no_vggt,
                            ),
                            "T3": temporal_disparity_error(
                                raw_predictions["T3"],
                                no_vggt_transport.disparity_history_loss_hr_px,
                                safe_mask=paired_t1_no_vggt,
                            ),
                            "T3_VGGT": MetricResult.invalid(),
                            "T3_VGGT_mask_off": MetricResult.invalid(),
                        }
            if stage == "temporal":
                assert temporal_visualization is not None
                assert t3_vggt_sign_health is not None
                t3_vggt_sign_health.update(
                    temporal_visualization.vggt_on,
                    bilinear_disparity_hr_px=baseline,
                    ffs_valid_lr=endpoint_ffs_valid,
                    ffs_valid_hr=valid_hr,
                    history_valid_lr=(
                        temporal_visualization.shared_transport.valid_history
                    ),
                    history_valid_hr=(
                        temporal_visualization.shared_transport.valid_history_hr
                    ),
                    pose_valid=batch["temporal_pose_valid_sequence"][:, 2],
                )
            predictions: dict[str, Tensor] = {}
            for method_name, prediction in raw_predictions.items():
                predictions[method_name] = prediction
                # Physical output policy: negative disparities become zero.
                # Never use epsilon here; zero remains invalid/unfilled and
                # cannot fabricate hole completeness.
                predictions[f"{method_name}_clamp0"] = (
                    physical_disparity_clamp_min_zero(prediction)
                )
            for method_name, prediction in predictions.items():
                sample_metrics = compute_sample_metrics(
                    prediction,
                    target.float(),
                    target_trusted_mask=target_trusted,
                    ffs_confidence_hr=confidence_hr.float(),
                    ffs_valid_mask_hr=valid_hr,
                    ffs_trusted_mask_hr=trusted_hr,
                    low_confidence_threshold=float(config.eval.low_confidence_threshold),
                    boundary_gradient_threshold_px=float(
                        config.eval.boundary_gradient_threshold_px
                    ),
                    boundary_radius_px=int(config.eval.boundary_radius_px),
                )
                if physical_metric_v2:
                    explicit_output = (
                        None
                        if method_name.endswith("_clamp0")
                        else model_outputs_by_method.get(method_name)
                    )
                    sample_metrics.update(
                        _explicit_validity_completion_metrics(
                            explicit_output,
                            target_disparity_hr_px=target.float(),
                            teacher_valid_mask_hr=target_valid,
                            ffs_valid_mask_hr=valid_hr,
                        )
                    )
                if stage == "temporal":
                    base_method_name = method_name.removesuffix("_clamp0")
                    is_postprocessed = method_name.endswith("_clamp0")
                    native_temporal_field = (
                        "temporal_residual_error_native_px"
                        if temporal_metric_v2
                        else "temporal_disparity_error_native_px"
                    )
                    paired_temporal_field = (
                        "temporal_residual_error_paired_px"
                        if temporal_metric_v2
                        else "temporal_disparity_error_paired_px"
                    )
                    sample_metrics[native_temporal_field] = (
                        MetricResult.invalid()
                        if is_postprocessed
                        else temporal_results[base_method_name]
                    )
                    sample_metrics[paired_temporal_field] = (
                        MetricResult.invalid()
                        if is_postprocessed or base_method_name == "bilinear"
                        else paired_temporal_results[base_method_name]
                    )
                accumulators[method_name].update(sample_metrics)

            if failure_sample_collector is not None:
                assert stage == "temporal"
                assert temporal_visualization is not None
                for item_index in range(target.shape[0]):
                    failure_metrics = _t3_failure_sample_metrics(
                        prediction_hr_px=raw_predictions["T3_VGGT"][
                            item_index : item_index + 1
                        ],
                        target_hr_px=target[item_index : item_index + 1].float(),
                        target_trusted_mask=target_trusted[item_index : item_index + 1],
                        ffs_confidence_hr=confidence_hr[item_index : item_index + 1].float(),
                        ffs_valid_mask_hr=valid_hr[item_index : item_index + 1],
                        ffs_trusted_mask_hr=trusted_hr[item_index : item_index + 1],
                        history_hr_px=(
                            (
                                temporal_visualization.reference_transport.prediction_disparity_hr_px
                                if temporal_metric_v2
                                else temporal_visualization.shared_transport.disparity_history_loss_hr_px
                            )[item_index : item_index + 1]
                        ),
                        strict_temporal_safe_mask=t3_safe[item_index : item_index + 1],
                        warped_reference_history_hr_px=(
                            None
                            if not temporal_metric_v2
                            else temporal_visualization.reference_transport.disparity_hr_px[
                                item_index : item_index + 1
                            ]
                        ),
                        warped_reference_valid_mask=(
                            None
                            if not temporal_metric_v2
                            else temporal_visualization.reference_transport.valid_mask_hr[
                                item_index : item_index + 1
                            ]
                        ),
                        low_confidence_threshold=float(config.eval.low_confidence_threshold),
                        boundary_gradient_threshold_px=float(
                            config.eval.boundary_gradient_threshold_px
                        ),
                        boundary_radius_px=int(config.eval.boundary_radius_px),
                    )
                    sequence_id = str(batch["sequence_id"][item_index])
                    frame_id = int(batch["frame_ids"][item_index, 2].item())
                    manifest_index = int(
                        batch["manifest_indices"][item_index, 2].item()
                    )
                    timestamp = float(batch["timestamps"][item_index, 2].item())
                    for criterion, metric in failure_metrics.items():
                        failure_sample_collector.consider(
                            criterion,
                            metric,
                            sequence_id=sequence_id,
                            frame_id=frame_id,
                            manifest_index=manifest_index,
                            timestamp=timestamp,
                            payload_factory=(
                                lambda item_index=item_index: _t3_failure_payload(
                                    batch=batch,
                                    item_index=item_index,
                                    output_size_hw=output_size,
                                    endpoint_rgb=endpoint_rgb,
                                    endpoint_K_hr=endpoint_K_hr,
                                    endpoint_baseline_m=endpoint_baseline_m,
                                    baseline_hr_px=baseline,
                                    target_hr_px=target,
                                    target_trusted_mask=target_trusted,
                                    temporal_predictions=temporal_visualization,
                                )
                            ),
                        )

            if stage == "temporal" and temporal_flicker_collector is not None:
                assert temporal_visualization is not None
                for item_index in range(target.shape[0]):
                    temporal_flicker_collector.append(
                        sequence_id=str(batch["sequence_id"][item_index]),
                        frame_id=int(batch["frame_ids"][item_index, 2].item()),
                        timestamp=float(batch["timestamps"][item_index, 2].item()),
                        rgb_hr=endpoint_rgb[item_index],
                        bilinear_disparity_hr_px=baseline[item_index],
                        t3_disparity_hr_px=raw_predictions["T3"][item_index],
                        t3_vggt_disparity_hr_px=(
                            raw_predictions["T3_VGGT"][item_index]
                        ),
                        target_disparity_hr_px=target[item_index],
                        target_trusted_mask=target_trusted[item_index],
                        uncertainty_variance=(
                            temporal_visualization.vggt_on.uncertainty[item_index]
                        ),
                    )

            batch_size = target.shape[0]
            for item_index in range(batch_size):
                if visualized >= visualization_limit:
                    break
                visualization_class = "spatial"
                endpoint_pose_valid = None
                endpoint_static_prior_valid = None
                history_valid_pixels = None
                if stage == "temporal":
                    assert temporal_visualization is not None
                    endpoint_pose_valid = bool(
                        batch["temporal_pose_valid_sequence"][item_index, 2]
                        .detach()
                        .cpu()
                        .item()
                    )
                    endpoint_static_prior_valid = bool(
                        batch["static_prior_valid_sequence"][item_index, 2]
                        .detach()
                        .cpu()
                        .item()
                    )
                    history_valid_pixels = int(
                        temporal_visualization.shared_transport.valid_history_hr[
                            item_index
                        ]
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )
                    # Alternate positive-source and fail-closed examples. This
                    # prevents the first few rejected windows from producing
                    # an apparently empty VGGT/history visualization set.
                    want_valid_sources = visualized % 2 == 0
                    has_valid_sources = (
                        endpoint_pose_valid
                        and endpoint_static_prior_valid
                        and history_valid_pixels > 0
                    )
                    if want_valid_sources and not has_valid_sources:
                        continue
                    if not want_valid_sources and endpoint_pose_valid:
                        continue
                    visualization_class = (
                        "valid_geometry_and_history"
                        if want_valid_sources
                        else "pose_rejected_fail_closed"
                    )
                sequence_id = str(batch["sequence_id"][item_index]).replace("/", "_")
                frame_id = int(
                    batch["frame_id"][item_index].item()
                    if stage == "spatial"
                    else batch["frame_ids"][item_index, 2].item()
                )
                visualization_output = (
                    model_output if stage == "spatial" else temporal_visualization.vggt_on
                )
                if stage == "temporal":
                    assert temporal_visualization is not None
                    vggt_lr = batch["disparity_vggt_hr_px_sequence"][item_index : item_index + 1, 2]
                    vggt_valid_lr = batch["valid_vggt_sequence"][item_index : item_index + 1, 2]
                    vggt_hr = functional.interpolate(
                        vggt_lr,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    vggt_valid_hr = functional.interpolate(
                        vggt_valid_lr.float(), size=output_size, mode="nearest"
                    )[0].bool()
                    history_hr = (
                        temporal_visualization.shared_transport
                        .disparity_history_loss_hr_px[item_index]
                    )
                    history_valid_hr = temporal_visualization.shared_transport.valid_history_hr[
                        item_index
                    ]
                else:
                    vggt_hr = None
                    vggt_valid_hr = None
                    history_hr = None
                    history_valid_hr = None
                _save_visualization(
                    output_dir / "visualizations",
                    sample_name=f"{visualized:04d}_{sequence_id}_{frame_id}",
                    rgb_hr=endpoint_rgb[item_index],
                    K_hr_px=endpoint_K_hr[item_index],
                    baseline_m=endpoint_baseline_m[item_index],
                    baseline_hr_px=baseline[item_index],
                    output_hr_px=visualization_output.disparity_hr_px[
                        item_index
                    ].float(),
                    target_hr_px=target[item_index].float(),
                    target_trusted_mask=target_trusted[item_index],
                    source_weights_lr=visualization_output.source_weights[
                        item_index
                    ].float(),
                    uncertainty_hr=visualization_output.uncertainty[item_index].float(),
                    vggt_disparity_hr_px=vggt_hr,
                    vggt_valid_mask_hr=vggt_valid_hr,
                    history_disparity_hr_px=history_hr,
                    history_valid_mask_hr=history_valid_hr,
                    vggt_off_output_hr_px=(
                        None
                        if stage == "spatial"
                        else temporal_visualization.source_mask_off.disparity_hr_px[
                            item_index
                        ].float()
                    ),
                    no_vggt_output_hr_px=(
                        None
                        if stage == "spatial"
                        else temporal_visualization.no_vggt.disparity_hr_px[
                            item_index
                        ].float()
                    ),
                    prediction_filename=(
                        "t1_disparity_hr_px.png"
                        if stage == "spatial"
                        else "t3_vggt_disparity_hr_px.png"
                    ),
                )
                visualization_records.append(
                    {
                        "sample_index": visualized,
                        "sequence_id": sequence_id,
                        "frame_id": frame_id,
                        "selection_class": visualization_class,
                        "endpoint_pose_valid": endpoint_pose_valid,
                        "endpoint_static_prior_valid": endpoint_static_prior_valid,
                        "history_valid_pixels": history_valid_pixels,
                    }
                )
                visualized += 1

    temporal_flicker_report = (
        temporal_flicker_collector.finalize()
        if temporal_flicker_collector is not None
        else None
    )
    if temporal_flicker_report is not None:
        print(
            json.dumps(
                {
                    "temporal_flicker_video": temporal_flicker_report["status"],
                    "reason": temporal_flicker_report.get("reason"),
                    "videos": len(temporal_flicker_report["videos"]),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    evaluator_provenance = {
        "git_hash": repository_git_hash(PROJECT_ROOT),
        "eval_py_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_module_sha256": sha256_file(
            (SRC_ROOT / "evaluation.py").resolve()
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    failure_sample_report = (
        None
        if failure_sample_collector is None
        else failure_sample_collector.write(
            output_dir / "failures",
            checkpoint=checkpoint_metadata,
            evaluator=evaluator_provenance,
        )
    )
    elapsed_seconds = time.perf_counter() - started
    full_selection = start_index == 0 and sample_count == len(dataset)
    checkpoint_completion = checkpoint_training_completion(
        checkpoint_metadata,
        stage=stage,
    )
    spatial_checkpoint_completion = (
        None
        if spatial_checkpoint_metadata is None
        else checkpoint_training_completion(
            spatial_checkpoint_metadata,
            stage="spatial",
        )
    )
    eligibility = evaluation_eligibility_status(
        stage=stage,
        full_selection=full_selection,
        allow_non_holdout_smoke=args.allow_non_holdout_smoke,
        formal_holdout=(
            None
            if holdout_raw_lineage is None
            else holdout_raw_lineage.get("formal_holdout") is True
        ),
        checkpoint_completion=checkpoint_completion,
        spatial_checkpoint_completion=spatial_checkpoint_completion,
    )
    coverage_eligible = bool(eligibility["coverage_eligible"])
    final_training_checkpoint = bool(
        eligibility["final_training_checkpoint"]
    )
    final_acceptance_eligible = bool(
        eligibility["final_acceptance_eligible"]
    )
    evaluation_status = str(eligibility["status"])
    aggregate_methods = {
        method: accumulator.finalize()
        for method, accumulator in accumulators.items()
    }
    finalized = {
        method: {name: result.to_dict() for name, result in metrics.items()}
        for method, metrics in aggregate_methods.items()
    }
    for method in finalized.values():
        method["point_to_plane_error_m"] = dict(POINT_TO_PLANE_NOT_AVAILABLE)
    for method_name, method in finalized.items():
        method["output_variant"] = (
            {
                "type": "PHYSICAL_CLAMP_MIN_ZERO",
                "source_method": method_name.removesuffix("_clamp0"),
                "epsilon_fill": False,
            }
            if method_name.endswith("_clamp0")
            else {"type": "RAW_MODEL_OUTPUT", "source_method": method_name}
        )
    comparisons: dict[str, Any] = {}
    for method_name in method_names:
        if method_name == "bilinear":
            continue
        reference_name = (
            "bilinear_clamp0"
            if method_name.endswith("_clamp0")
            and method_name != "bilinear_clamp0"
            else "bilinear"
        )
        comparison = comparison_from_aggregates(
            aggregate_methods[reference_name], aggregate_methods[method_name]
        )
        comparison["reference_method"] = reference_name
        comparisons[f"{method_name}_vs_bilinear"] = comparison
    if stage == "temporal":
        paired_temporal_metric_name = (
            "temporal_residual_error_paired_px"
            if temporal_metric_v2
            else "temporal_disparity_error_paired_px"
        )
        for method_name in ("T3",):
            comparisons[f"{method_name}_vs_T1_temporal"] = (
                aggregate_metric_change(
                    aggregate_methods["T1"],
                    aggregate_methods[method_name],
                    paired_temporal_metric_name,
                )
            )
        comparisons["T3_VGGT_vs_T3_prior_effect"] = (
            aggregate_metric_change(
                aggregate_methods["T3"],
                aggregate_methods["T3_VGGT"],
                "epe_px",
            )
        )
        comparisons["T3_VGGT_mask_off_vs_T3_vggt_source_mask"] = (
            aggregate_metric_change(
                aggregate_methods["T3_VGGT"],
                aggregate_methods["T3_VGGT_mask_off"],
                "epe_px",
            )
        )
    report = {
        "schema_version": 1,
        "stage": stage_label,
        "status": evaluation_status,
        "evaluator": {
            **evaluator_provenance,
        },
        "target": {
            "type": PSEUDO_GT_LABEL,
            "paper_accuracy": False,
            "warning": (
                "Metrics use trusted output from the same FFS family as pseudo-GT; "
                "they are engineering validation only."
            ),
        },
        "temporal_metric_contract": (
            None
            if stage != "temporal"
            else _temporal_metric_contract(
                temporal_metric_v2=temporal_metric_v2
            )
        ),
        "explicit_validity_completion_contract": (
            None
            if not physical_metric_v2
            else {
                "protocol_version": "explicit_valid_completion_nonnegative_v2",
                "reference": "HR FFS teacher pseudo-validity",
                "paper_gt": False,
                "valid_metrics": ["precision", "recall", "f1", "brier"],
                "completion_domain": "current FFS invalid/hole pixels",
                "completion_metrics": ["precision", "recall", "f1", "brier"],
                "aggregation": (
                    "global confusion counts and global squared-error numerator/"
                    "pixel count"
                ),
                "methods_without_explicit_heads": "invalid metric receipt",
                "postprocessed_clamp0_rows": "invalid metric receipt",
            }
        ),
        "claims": {
            "paper_accuracy": False,
            "paper_gt": False,
            "epipolar_refinement": False,
            "temporal_future_frames": False,
            "teacher_temporal_residual_metric": (
                stage == "temporal" and temporal_metric_v2
            ),
            "formal_holdout": (
                None
                if stage == "spatial"
                else bool(
                    holdout_raw_lineage
                    and holdout_raw_lineage.get("formal_holdout")
                )
            ),
            "coverage_eligible": coverage_eligible,
            "coverage_eligible_definition": (
                "Complete selected validation corpus and, for Stage B, "
                "video-disjoint formal holdout lineage. This does not imply "
                "that the training checkpoint is final."
            ),
            "final_training_checkpoint": final_training_checkpoint,
            "final_acceptance_eligible": final_acceptance_eligible,
            "final_acceptance_eligible_definition": (
                "coverage_eligible AND canonical final training schedules "
                "completed (Stage A 5000; Stage B 15000 with final Stage A). "
                "Eligibility does not assert that metric thresholds passed."
            ),
            # Backward-compatible key, narrowed to the only safe meaning.
            "acceptance_eligible": final_acceptance_eligible,
            "acceptance_eligible_definition": (
                "Backward-compatible alias of final_acceptance_eligible; use "
                "coverage_eligible for intermediate full-holdout evaluations."
            ),
            "full_validation_selection": full_selection,
        },
        "postprocess_contract": {
            "raw_rows": list(base_method_names),
            "physical_variant_suffix": "_clamp0",
            "operation": "torch.clamp_min(disparity_hr_px, 0.0)",
            "epsilon_fill": False,
            "zero_semantics": "invalid and not complete",
            "temporal_metrics_for_endpoint_only_variant": "NOT_AVAILABLE",
            "acceptance_note": (
                "Raw and physical variants are reported separately; the "
                "physical variant never overwrites raw model output."
            ),
        },
        "methods": finalized,
        "comparisons": comparisons,
        "diagnostics": {
            "t3_vggt_sign_health": (
                None
                if t3_vggt_sign_health is None
                else t3_vggt_sign_health.finalize()
            )
        },
        "point_to_plane": dict(POINT_TO_PLANE_NOT_AVAILABLE),
        "records_evaluated": sample_count,
        "selection_start": start_index,
        "visualizations_written": visualized,
        "visualization_selection": visualization_records,
        **(
            {}
            if temporal_flicker_report is None
            else {"temporal_flicker_video": temporal_flicker_report}
        ),
        **(
            {}
            if failure_sample_report is None
            else {"failure_sample_bundles": failure_sample_report}
        ),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "crop_mode": str(config.eval.crop_mode),
        "hr_crop": None if full_resolution else list(crop_size or ()),
        "parameter_count": parameter_count,
        "checkpoint": checkpoint_metadata,
        "checkpoint_training_completion": checkpoint_completion,
        "checkpoint_lineage": checkpoint_lineage,
        "spatial_checkpoint": spatial_checkpoint_metadata,
        "spatial_checkpoint_training_completion": (
            spatial_checkpoint_completion
        ),
        "spatial_checkpoint_lineage": spatial_checkpoint_lineage,
        "manifest_path": str(manifest_path),
        "cache_identities": {
            "observation": asdict(observation_identity),
            "teacher": asdict(teacher_identity),
        },
        "derived_cache_lineage": derived_lineage,
        "formal_temporal_coverage": formal_coverage,
        "holdout_and_raw_lineage": holdout_raw_lineage,
        "causal_contract": (
            None
            if stage == "spatial"
            else {
                "student_frames": 3,
                "scored_time_index": 2,
                "endpoint_only": True,
                "shared_fixed_crop": True,
                "each_student_time_has_endpoint_derived_geometry": True,
                "future_frames": False,
                "vggt_context_pairs": 5,
                "temporal_metric_domain": (
                    (
                        "HR z-buffer visible AND static AND non-collision AND "
                        "geometry-consistent AND valid prediction-history AND "
                        "current/warped trusted teacher"
                    )
                    if temporal_metric_v2
                    else (
                        "HR z-buffer visible AND static AND non-collision AND "
                        "geometry-consistent AND valid-history"
                    )
                ),
                "t3_vs_t1_temporal_comparison_domain": (
                    "intersection of T1 and T3 strict HR safe masks"
                ),
                "endpoint_pose_valid_count": endpoint_pose_valid_count,
                "endpoint_pose_rejected_count": (
                    sample_count - endpoint_pose_valid_count
                ),
                "endpoint_static_prior_valid_count": (
                    endpoint_static_prior_valid_count
                ),
                "acceptance_t3_method": (
                    "T3: causal recurrent/history model with zero static VGGT "
                    "prior channels and independently propagated history"
                ),
                "vggt_prior_method": "T3_VGGT",
                "vggt_source_mask_diagnostic": (
                    "same history and pose as T3_VGGT; only valid_vggt is "
                    "false; raw VGGT geometry channels remain visible to the encoder"
                ),
                "no_vggt_ablation": (
                    "The canonical T3 row zeros VGGT disparity/confidence/mask "
                    "and independently unrolls hidden/history using unchanged poses"
                ),
            }
        ),
        "resolved_config": _resolved_dict(config),
    }
    metrics_json = output_dir / "metrics.json"
    metrics_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", finalized, comparisons)
    print(
        json.dumps(
            {
                "status": evaluation_status,
                "stage": stage_label,
                "target_type": PSEUDO_GT_LABEL,
                "coverage_eligible": coverage_eligible,
                "final_training_checkpoint": final_training_checkpoint,
                "final_acceptance_eligible": final_acceptance_eligible,
                (
                    "records_evaluated"
                    if stage == "spatial"
                    else "windows_evaluated"
                ): sample_count,
                "parameter_count": parameter_count,
                "metrics_json": str(metrics_json),
                "metrics_csv": str(output_dir / "metrics.csv"),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
