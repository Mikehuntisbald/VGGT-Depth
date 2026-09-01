#!/usr/bin/env python3
"""Formal held-out evaluator for Stage-C HR epipolar refinement.

All accuracy targets are trusted HR FFS teacher pseudo-GT. Results are
engineering evidence only: ``paper_ground_truth=false`` and no point-cloud
metric is invented without target normals/correspondences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import sha256_file  # noqa: E402
from data.epipolar_training_dataset import (  # noqa: E402
    EpipolarTrainingDataset,
    collate_epipolar_training_samples,
)
from data.manifest import load_manifest  # noqa: E402
from data.temporal_training_dataset import CachedTemporalTrainingDataset  # noqa: E402
from eval import (  # noqa: E402
    _audit_temporal_holdout_and_raw_lineage,
    _validate_formal_temporal_coverage,
)
from evaluation import (  # noqa: E402
    MethodMetricAccumulator,
    POINT_TO_PLANE_NOT_AVAILABLE,
    PSEUDO_GT_LABEL,
    aggregate_metric_results,
    aggregate_metric_change,
    comparison_from_aggregates,
    compute_sample_metrics,
    load_model_for_evaluation,
    physical_disparity_clamp_min_zero,
    upsample_ffs_inputs_to_hr,
    validate_checkpoint_lineage,
    validate_temporal_batch_causality,
)
from metrics.disparity import MetricResult  # noqa: E402
from models.epipolar_refiner import HREpipolarRefiner  # noqa: E402
from models.epipolar_stage import FrozenTemporalEpipolarStage  # noqa: E402
from train import (  # noqa: E402
    build_model,
    learning_rate_multiplier,
    load_receipt_identity,
)
from train_epipolar import (  # noqa: E402
    EPIPOLAR_GEOMETRY_CONTRACT,
    PSEUDO_GT_SUPERVISION,
    STAGE_C_RUNTIME_GIT_SCOPES,
    _validate_base_data_lineage,
    _validated_rectification_audit,
    predict_frozen_stage_b_endpoint,
    resolve_epipolar_config,
    validate_epipolar_config,
)
from utils.checkpoint import CheckpointMismatchError, repository_git_hash  # noqa: E402
from utils.seed import (  # noqa: E402
    STRICT_CUBLAS_WORKSPACE_CONFIG,
    deterministic_runtime_state,
    seed_everything,
    strict_determinism_enabled,
)
from utils.visualization import (  # noqa: E402
    grayscale_to_rgb_uint8,
    save_rgb_uint8,
    scalar_to_rgb_uint8,
)


STAGE_C_COMPONENT = "ffs-omega-tsr-epipolar-stage-c"
STAGE_C_MODEL_COMPONENT = "hr_epipolar_refiner"
EXPECTED_REFINER_PARAMETERS = 69_905
FORMAL_VALIDATION_RECORDS = 244
FORMAL_DERIVED_ENDPOINTS = 240
FORMAL_EVALUABLE_T3_WINDOWS = 238
FORMAL_STAGE_C_STEPS = 5_000
FORMAL_STAGE_B_STEPS = 15_000
FORMAL_STAGE_C_HR_CROP = (384, 768)
FORMAL_VALIDATION_MANIFEST_SHA256 = (
    "014bd75de8ffbf74530c64eac76394a30bfc62d65b2da02397de2fb5c984760c"
)
FORMAL_RECTIFICATION_AUDIT_SHA256 = (
    "3eb3e8853e4723b9e0703aeaffd36b9ef482b311ff5b9cff5a79e28e60e84429"
)
STAGE_C_TRAINING_RUNTIME_FIELDS = {
    "device",
    "device_type",
    "device_name",
    "device_capability",
    "torch_version",
    "cuda_version",
    "cuda_available",
    "bf16_supported",
    "autocast_enabled",
    "autocast_dtype",
    "deterministic_algorithms_enabled",
    "deterministic_algorithms_warn_only",
    "cublas_workspace_config",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "strict_determinism_eligible",
    "formal_cuda_bf16_eligible",
}


@dataclass(frozen=True, slots=True)
class FiniteStatistics:
    count: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FiniteStatisticsAccumulator:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: Tensor, mask: Tensor | None = None) -> None:
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("statistics value must be a floating-point tensor")
        selected = torch.isfinite(value)
        if mask is not None:
            if mask.shape != value.shape:
                raise ValueError("statistics mask shape must match value")
            selected &= mask.to(dtype=torch.bool)
        count = int(selected.sum().item())
        if count == 0:
            return
        finite = value[selected].to(dtype=torch.float64)
        self.count += count
        self.total += float(finite.sum().item())
        self.minimum = min(self.minimum, float(finite.min().item()))
        self.maximum = max(self.maximum, float(finite.max().item()))

    def finalize(self) -> FiniteStatistics:
        if self.count == 0:
            return FiniteStatistics(0, None, None, None, False)
        return FiniteStatistics(
            count=self.count,
            mean=self.total / self.count,
            minimum=self.minimum,
            maximum=self.maximum,
            valid=True,
        )


def _metric_mean(value: Tensor, mask: Tensor) -> MetricResult:
    selected = mask.to(dtype=torch.bool)
    count = int(selected.sum().item())
    if count == 0:
        return MetricResult.invalid()
    values = value[selected]
    if not bool(torch.isfinite(values).all().item()):
        return MetricResult.invalid(count=count)
    numerator = float(values.to(dtype=torch.float64).sum().item())
    return MetricResult(numerator / count, numerator, count, True)


def _metric_rate(event: Tensor, mask: Tensor) -> MetricResult:
    selected = mask.to(dtype=torch.bool)
    count = int(selected.sum().item())
    if count == 0:
        return MetricResult.invalid()
    numerator = float(event[selected].to(dtype=torch.float64).sum().item())
    return MetricResult(numerator / count, numerator, count, True)


def _strict_paired_outcome_rate(
    event: Tensor,
    target_domain: Tensor,
    finite_pair: Tensor,
) -> MetricResult:
    count = int(target_domain.to(dtype=torch.bool).sum().item())
    if count == 0:
        return MetricResult.invalid()
    if not bool(finite_pair[target_domain.to(dtype=torch.bool)].all().item()):
        return MetricResult.invalid(count=count)
    return _metric_rate(event, target_domain)


def paired_refinement_metrics(
    base_disparity_hr_px: Tensor,
    refined_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    target_trusted_mask: Tensor,
) -> dict[str, MetricResult]:
    """Direct base/refiner changes on one identical trusted HR domain."""

    shape = base_disparity_hr_px.shape
    for name, value in (
        ("refined_disparity_hr_px", refined_disparity_hr_px),
        ("target_disparity_hr_px", target_disparity_hr_px),
        ("target_trusted_mask", target_trusted_mask),
    ):
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")
    target_domain = (
        target_trusted_mask.to(dtype=torch.bool)
        & torch.isfinite(target_disparity_hr_px)
        & (target_disparity_hr_px > 0)
    )
    finite_pair = torch.isfinite(base_disparity_hr_px) & torch.isfinite(
        refined_disparity_hr_px
    )
    base_error = (base_disparity_hr_px - target_disparity_hr_px).abs()
    refined_error = (refined_disparity_hr_px - target_disparity_hr_px).abs()
    improvement = base_error - refined_error
    return {
        "paired_epe_improvement_hr_px": _metric_mean(improvement, target_domain),
        "paired_refined_better_rate": _strict_paired_outcome_rate(
            improvement > 0, target_domain, finite_pair
        ),
        "paired_refined_worse_rate": _strict_paired_outcome_rate(
            improvement < 0, target_domain, finite_pair
        ),
        "paired_unchanged_rate": _strict_paired_outcome_rate(
            improvement == 0, target_domain, finite_pair
        ),
        "paired_finite_coverage_rate": _metric_rate(finite_pair, target_domain),
        "paired_nonfinite_rate": _metric_rate(~finite_pair, target_domain),
    }


def require_formal_stage_c_coverage(
    coverage: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, int]:
    """Require the canonical video-disjoint validation corpus, not a mini split."""

    expected = {
        "manifest_records": FORMAL_VALIDATION_RECORDS,
        "derived_endpoint_records": FORMAL_DERIVED_ENDPOINTS,
        "evaluable_t3_windows": FORMAL_EVALUABLE_T3_WINDOWS,
    }
    actual: dict[str, int] = {}
    for name, expected_value in expected.items():
        value = coverage.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"formal Stage-C coverage field is malformed: {name}")
        actual[name] = value
        if value != expected_value:
            raise ValueError(
                "formal Stage-C held-out coverage must be exactly "
                f"244/240/238; {name}={value}, expected {expected_value}"
            )
    if manifest_sha256 != FORMAL_VALIDATION_MANIFEST_SHA256:
        raise ValueError(
            "validation manifest has canonical counts but is not the bound "
            "video-disjoint Stage-C manifest"
        )
    return actual


def checkpoint_completion_status(
    stage_c_metadata: Mapping[str, Any],
    base_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Make intermediate Stage-B/Stage-C artifacts acceptance-ineligible."""

    stage_c_config = stage_c_metadata.get("config")
    base_config = base_metadata.get("training_config")
    stage_c_train = (
        stage_c_config.get("train")
        if isinstance(stage_c_config, Mapping)
        else None
    )
    base_train = (
        base_config.get("train") if isinstance(base_config, Mapping) else None
    )
    if not isinstance(stage_c_train, Mapping) or not isinstance(
        base_train, Mapping
    ):
        raise CheckpointMismatchError("checkpoint completion configs are missing")
    expected_stage_c = _positive_int(
        stage_c_train.get("steps_epipolar"), "Stage-C configured steps"
    )
    expected_base = _positive_int(
        base_train.get("steps"), "Stage-B configured execution steps"
    )
    declared_base_schedule = _positive_int(
        base_train.get("steps_temporal"), "Stage-B declared temporal steps"
    )
    actual_stage_c = _nonnegative_int(
        stage_c_metadata.get("step"), "Stage-C checkpoint step"
    )
    actual_base = _nonnegative_int(
        base_metadata.get("step"), "Stage-B checkpoint step"
    )
    stage_c_execution_complete = actual_stage_c == expected_stage_c
    stage_c_canonical_schedule = expected_stage_c == FORMAL_STAGE_C_STEPS
    stage_c_complete = stage_c_execution_complete and stage_c_canonical_schedule
    base_execution_complete = actual_base == expected_base
    base_canonical_schedule = (
        expected_base == FORMAL_STAGE_B_STEPS
        and declared_base_schedule == FORMAL_STAGE_B_STEPS
    )
    base_complete = base_execution_complete and base_canonical_schedule
    return {
        "stage_c": {
            "actual_step": actual_stage_c,
            "configured_steps": expected_stage_c,
            "declared_epipolar_schedule_steps": FORMAL_STAGE_C_STEPS,
            "execution_complete": stage_c_execution_complete,
            "canonical_schedule": stage_c_canonical_schedule,
            "complete": stage_c_complete,
        },
        "stage_b_base": {
            "actual_step": actual_base,
            "configured_steps": expected_base,
            "declared_temporal_schedule_steps": declared_base_schedule,
            "execution_complete": base_execution_complete,
            "canonical_schedule": base_canonical_schedule,
            "complete": base_complete,
        },
        "all_complete": stage_c_complete and base_complete,
    }


def validate_formal_crop_contract(
    stage_c_metadata: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    *,
    limited_smoke: bool,
) -> dict[str, Any]:
    """Bind formal evaluation to the trained canonical 384x768 HR crop."""

    stage_config = stage_c_metadata.get("config")
    stage_data = stage_config.get("data") if isinstance(stage_config, Mapping) else None
    evaluation_data = evaluation_config.get("data")
    if not isinstance(stage_data, Mapping) or not isinstance(
        evaluation_data, Mapping
    ):
        raise CheckpointMismatchError("Stage-C/evaluation crop configs are missing")

    def crop_tuple(value: object, name: str) -> tuple[int, int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            )
        ):
            raise CheckpointMismatchError(f"{name} is malformed")
        return int(value[0]), int(value[1])

    trained_crop = crop_tuple(stage_data.get("hr_crop"), "Stage-C training crop")
    evaluation_crop = crop_tuple(
        evaluation_data.get("hr_crop"), "Stage-C evaluation crop"
    )
    exact_training_crop = evaluation_crop == trained_crop
    canonical_crop = evaluation_crop == FORMAL_STAGE_C_HR_CROP
    trained_on_canonical_crop = trained_crop == FORMAL_STAGE_C_HR_CROP
    training_crop_mode = stage_data.get("crop_mode")
    evaluation_crop_mode = evaluation_data.get("crop_mode")
    canonical_modes = (
        training_crop_mode == "random" and evaluation_crop_mode == "fixed"
    )
    eligible = (
        exact_training_crop
        and canonical_crop
        and trained_on_canonical_crop
        and canonical_modes
    )
    if not eligible and not limited_smoke:
        raise CheckpointMismatchError(
            "formal Stage-C evaluation requires checkpoint and evaluation HR crop "
            "[384,768], random training crops, and fixed evaluation crops; "
            "deviations require --limit and are smoke-only"
        )
    return {
        "trained_hr_crop": list(trained_crop),
        "evaluation_hr_crop": list(evaluation_crop),
        "canonical_hr_crop": list(FORMAL_STAGE_C_HR_CROP),
        "exact_training_crop": exact_training_crop,
        "canonical_crop": canonical_crop,
        "training_crop_mode": training_crop_mode,
        "evaluation_crop_mode": evaluation_crop_mode,
        "canonical_modes": canonical_modes,
        "eligible": eligible,
    }


def validate_formal_execution_contract(
    stage_c_metadata: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    *,
    device: torch.device,
    limited_smoke: bool,
) -> dict[str, Any]:
    """Require the declared BF16/CUDA/AdamW formal numerical path."""

    stage_config = stage_c_metadata.get("config")
    stage_train = (
        stage_config.get("train") if isinstance(stage_config, Mapping) else None
    )
    evaluation_train = evaluation_config.get("train")
    if not isinstance(stage_train, Mapping) or not isinstance(
        evaluation_train, Mapping
    ):
        raise CheckpointMismatchError("Stage-C numerical configs are missing")
    saved_precision = str(stage_train.get("precision", "")).lower()
    evaluation_precision = str(evaluation_train.get("precision", "")).lower()
    saved_optimizer = str(stage_train.get("optimizer", "")).lower()
    expected_training_values = {
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_steps": 500,
        "correction_regularizer_weight": 0.01,
    }
    canonical_training_values = all(
        stage_train.get(name) == expected
        for name, expected in expected_training_values.items()
    )
    batch_schedule = (
        stage_train.get("micro_batch_size"),
        stage_train.get("grad_accumulation"),
    )
    canonical_batch_schedule = batch_schedule in {(2, 4), (1, 8)}
    cuda_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    recorded_runtime = stage_c_metadata.get("training_runtime_receipt")
    if not isinstance(recorded_runtime, Mapping):
        raise CheckpointMismatchError(
            "Stage-C validated training-runtime receipt is missing"
        )
    recorded_training_eligible = recorded_runtime.get("eligible") is True
    recorded_values = recorded_runtime.get("recorded")
    if not isinstance(recorded_values, Mapping):
        raise CheckpointMismatchError(
            "Stage-C recorded training-runtime values are missing"
        )
    evaluation_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    evaluation_device_capability = (
        list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None
    )
    evaluation_cuda_version = torch.version.cuda
    evaluation_cuda_version_tuple: tuple[int, int] | None = None
    if isinstance(evaluation_cuda_version, str):
        components = evaluation_cuda_version.split(".")
        try:
            if len(components) >= 2:
                evaluation_cuda_version_tuple = (
                    int(components[0]),
                    int(components[1]),
                )
        except ValueError:
            evaluation_cuda_version_tuple = None
    evaluation_cuda_12_8_or_newer = bool(
        evaluation_cuda_version_tuple is not None
        and evaluation_cuda_version_tuple >= (12, 8)
    )
    evaluation_blackwell = bool(
        isinstance(evaluation_device_capability, list)
        and tuple(evaluation_device_capability) >= (12, 0)
    )
    evaluation_rtx_5090 = bool(
        isinstance(evaluation_device_name, str)
        and "5090" in evaluation_device_name.lower()
    )
    evaluation_versions_match_training = bool(
        str(torch.__version__) == recorded_values.get("torch_version")
        and evaluation_cuda_version == recorded_values.get("cuda_version")
        and evaluation_device_name == recorded_values.get("device_name")
        and evaluation_device_capability
        == recorded_values.get("device_capability")
    )
    evaluation_determinism = deterministic_runtime_state()
    evaluation_strict_determinism = strict_determinism_enabled()
    evaluation_runtime_eligible = bool(
        cuda_bf16
        and evaluation_cuda_12_8_or_newer
        and evaluation_blackwell
        and evaluation_rtx_5090
        and evaluation_versions_match_training
        and evaluation_strict_determinism
    )
    eligible = (
        saved_precision == "bf16"
        and evaluation_precision == "bf16"
        and saved_optimizer == "adamw"
        and canonical_training_values
        and canonical_batch_schedule
        and recorded_training_eligible
        and evaluation_runtime_eligible
    )
    if not eligible and not limited_smoke:
        raise CheckpointMismatchError(
            "formal Stage-C evaluation requires recorded RTX 5090 CUDA 12.8+ "
            "BF16 training, saved/evaluation BF16, AdamW, and a CUDA device "
            "with BF16 support; use --limit for smoke evaluation"
        )
    return {
        "saved_precision": saved_precision,
        "evaluation_precision": evaluation_precision,
        "saved_optimizer": saved_optimizer,
        "expected_training_values": expected_training_values,
        "canonical_training_values": canonical_training_values,
        "batch_schedule": list(batch_schedule),
        "allowed_batch_schedules": [[2, 4], [1, 8]],
        "canonical_batch_schedule": canonical_batch_schedule,
        "recorded_training_runtime": dict(recorded_runtime),
        "recorded_training_eligible": recorded_training_eligible,
        "device": str(device),
        "cuda_bf16_supported": cuda_bf16,
        "autocast_dtype": "torch.bfloat16" if cuda_bf16 else None,
        "evaluation_runtime": {
            "device": str(device),
            "device_name": evaluation_device_name,
            "device_capability": evaluation_device_capability,
            "torch_version": str(torch.__version__),
            "cuda_version": evaluation_cuda_version,
            "cuda_bf16_supported": cuda_bf16,
            "autocast_dtype": "torch.bfloat16" if cuda_bf16 else None,
            "cuda_12_8_or_newer": evaluation_cuda_12_8_or_newer,
            "blackwell_capability": evaluation_blackwell,
            "rtx_5090": evaluation_rtx_5090,
            "versions_and_device_match_training": (
                evaluation_versions_match_training
            ),
            **evaluation_determinism,
            "strict_determinism_eligible": evaluation_strict_determinism,
            "eligible": evaluation_runtime_eligible,
        },
        "eligible": eligible,
    }


def validate_stage_c_training_runtime(value: object) -> dict[str, Any]:
    """Validate the producer-recorded device/autocast receipt.

    A CPU one-step checkpoint remains loadable for limited integration smoke,
    but only an actual RTX 5090 CUDA 12.8+ BF16 run is formal-training eligible.
    The eligibility result is derived from typed fields rather than trusting
    the producer's summary boolean alone.
    """

    if not isinstance(value, Mapping) or set(value) != STAGE_C_TRAINING_RUNTIME_FIELDS:
        raise CheckpointMismatchError(
            "Stage-C training_runtime fields are missing or malformed"
        )
    device_value = value.get("device")
    device_type = value.get("device_type")
    torch_version = value.get("torch_version")
    cuda_version = value.get("cuda_version")
    if not isinstance(device_value, str) or not device_value:
        raise CheckpointMismatchError("Stage-C training runtime device is malformed")
    try:
        parsed_device = torch.device(device_value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointMismatchError(
            "Stage-C training runtime device is malformed"
        ) from exc
    if device_type not in {"cpu", "cuda"} or parsed_device.type != device_type:
        raise CheckpointMismatchError(
            "Stage-C training runtime device/type are inconsistent"
        )
    if not isinstance(torch_version, str) or not torch_version:
        raise CheckpointMismatchError(
            "Stage-C training runtime torch version is malformed"
        )
    if cuda_version is not None and (
        not isinstance(cuda_version, str) or not cuda_version
    ):
        raise CheckpointMismatchError(
            "Stage-C training runtime CUDA version is malformed"
        )
    boolean_fields = (
        "cuda_available",
        "bf16_supported",
        "autocast_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "strict_determinism_eligible",
        "formal_cuda_bf16_eligible",
    )
    if any(type(value.get(name)) is not bool for name in boolean_fields):
        raise CheckpointMismatchError(
            "Stage-C training runtime boolean fields are malformed"
        )
    device_name = value.get("device_name")
    capability = value.get("device_capability")
    autocast_dtype = value.get("autocast_dtype")
    cuda_device = device_type == "cuda"
    if cuda_device:
        if not isinstance(device_name, str) or not device_name.strip():
            raise CheckpointMismatchError(
                "Stage-C CUDA training device name is malformed"
            )
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
        ):
            raise CheckpointMismatchError(
                "Stage-C CUDA training capability is malformed"
            )
        if not isinstance(cuda_version, str) or not cuda_version:
            raise CheckpointMismatchError(
                "Stage-C CUDA training build version is malformed"
            )
        if not (
            value["cuda_available"]
            and value["bf16_supported"]
            and value["autocast_enabled"]
        ):
            raise CheckpointMismatchError(
                "Stage-C CUDA checkpoint was not produced on the required BF16 path"
            )
    elif device_name is not None or capability is not None:
        raise CheckpointMismatchError(
            "Stage-C CPU training runtime contains CUDA device metadata"
        )
    elif value["bf16_supported"] or value["autocast_enabled"]:
        raise CheckpointMismatchError(
            "Stage-C CPU training runtime claims CUDA BF16 execution"
        )
    if value["autocast_enabled"]:
        if autocast_dtype != "torch.bfloat16":
            raise CheckpointMismatchError(
                "Stage-C training autocast dtype is not torch.bfloat16"
            )
    elif autocast_dtype is not None:
        raise CheckpointMismatchError(
            "Stage-C disabled autocast has a non-null dtype"
        )
    cublas_workspace_config = value.get("cublas_workspace_config")
    deterministic_eligible = bool(
        value["deterministic_algorithms_enabled"]
        and not value["deterministic_algorithms_warn_only"]
        and cublas_workspace_config == STRICT_CUBLAS_WORKSPACE_CONFIG
        and value["cudnn_deterministic"]
        and not value["cudnn_benchmark"]
    )
    if value["strict_determinism_eligible"] != deterministic_eligible:
        raise CheckpointMismatchError(
            "Stage-C strict determinism eligibility flag is inconsistent"
        )
    if not deterministic_eligible:
        raise CheckpointMismatchError(
            "Stage-C checkpoint is legacy/ineligible: strict deterministic "
            "algorithms, warn_only=false, CUBLAS_WORKSPACE_CONFIG=:4096:8, "
            "and deterministic cuDNN settings are required"
        )
    producer_eligible = bool(
        cuda_device
        and value["cuda_available"]
        and value["bf16_supported"]
        and value["autocast_enabled"]
        and autocast_dtype == "torch.bfloat16"
        and deterministic_eligible
    )
    if value["formal_cuda_bf16_eligible"] != producer_eligible:
        raise CheckpointMismatchError(
            "Stage-C training runtime eligibility flag is inconsistent"
        )

    cuda_version_tuple: tuple[int, int] | None = None
    if isinstance(cuda_version, str):
        components = cuda_version.split(".")
        try:
            if len(components) >= 2:
                cuda_version_tuple = (int(components[0]), int(components[1]))
        except ValueError:
            cuda_version_tuple = None
    cuda_12_8_or_newer = (
        cuda_version_tuple is not None and cuda_version_tuple >= (12, 8)
    )
    blackwell_capability = bool(
        isinstance(capability, list) and tuple(capability) >= (12, 0)
    )
    rtx_5090 = isinstance(device_name, str) and "5090" in device_name.lower()
    eligible = bool(
        producer_eligible
        and cuda_12_8_or_newer
        and blackwell_capability
        and rtx_5090
    )
    return {
        "recorded": dict(value),
        "producer_cuda_bf16_eligible": producer_eligible,
        "strict_determinism_eligible": deterministic_eligible,
        "cuda_12_8_or_newer": cuda_12_8_or_newer,
        "blackwell_capability": blackwell_capability,
        "rtx_5090": rtx_5090,
        "eligible": eligible,
    }


def validate_recorded_stage_c_training_lineage(
    stage_c_metadata: Mapping[str, Any],
    *,
    recomputed_base_lineage: Mapping[str, Any],
    recomputed_raw_lineage: Mapping[str, Any],
) -> None:
    """Require every stored training-side receipt/hash to remain reproducible."""

    if stage_c_metadata.get("base_lineage") != recomputed_base_lineage:
        raise CheckpointMismatchError(
            "Stage-C recorded base lineage differs from current training artifacts"
        )
    if stage_c_metadata.get("raw_lineage") != recomputed_raw_lineage:
        raise CheckpointMismatchError(
            "Stage-C recorded raw/derived lineage differs from training artifacts"
        )


def validate_rectification_audit_binding(
    stage_c_metadata: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    validation_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind both train and held-out validation to the canonical pixel audit."""

    config = stage_c_metadata.get("config")
    data = config.get("data") if isinstance(config, Mapping) else None
    recorded = stage_c_metadata.get("rectification_audit")
    if not isinstance(data, Mapping) or not isinstance(recorded, Mapping):
        raise CheckpointMismatchError("Stage-C rectification audit lineage is missing")
    train_manifest_value = data.get("manifest_path")
    if not isinstance(train_manifest_value, str):
        raise CheckpointMismatchError("Stage-C training manifest path is missing")
    train_manifest = Path(train_manifest_value).expanduser().resolve()
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    current = _validated_rectification_audit(
        receipt_path,
        expected_train_manifest_sha256=sha256_file(train_manifest),
    )
    if current.get("sha256") != FORMAL_RECTIFICATION_AUDIT_SHA256:
        raise CheckpointMismatchError(
            "rectification audit is not the canonical first-round receipt"
        )
    recorded_content = {name: value for name, value in recorded.items() if name != "path"}
    current_content = {name: value for name, value in current.items() if name != "path"}
    if recorded_content != current_content:
        raise CheckpointMismatchError(
            "current rectification audit differs from the Stage-C checkpoint"
        )
    configured = data.get("epipolar_rectification_audit")
    configured_path = data.get("epipolar_rectification_audit_path")
    if configured != recorded or not isinstance(configured_path, str) or (
        Path(configured_path).expanduser().resolve()
        != Path(str(recorded["path"])).expanduser().resolve()
    ):
        raise CheckpointMismatchError(
            "Stage-C config is not bound to the exact rectification audit"
        )
    manifests = current.get("manifest_sha256")
    counts = current.get("counts")
    evidence = current.get("pixel_evidence")
    if (
        not isinstance(manifests, Mapping)
        or manifests.get("validation") != validation_manifest_sha256
        or manifests.get("train") != sha256_file(train_manifest)
        or current.get("contract_version")
        != EPIPOLAR_GEOMETRY_CONTRACT["version"]
        or current.get("status") != "PASS"
        or not isinstance(counts, Mapping)
        or counts.get("sampled_frames") != 96
        or counts.get("covered_frames") != 96
        or counts.get("ratio_matches") != 98_095
        or counts.get("ransac_inliers") != 71_436
        or not isinstance(evidence, Mapping)
    ):
        raise CheckpointMismatchError(
            "rectification audit manifest/count/evidence contract differs"
        )
    return {
        **current,
        "checkpoint_recorded_path": str(recorded["path"]),
        "current_verified_path": str(current["path"]),
    }


def recompute_stage_c_training_lineage(
    stage_c_metadata: Mapping[str, Any],
    *,
    base_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open the exact Stage-C training receipts and verify their hashes."""

    config = stage_c_metadata.get("config")
    data = config.get("data") if isinstance(config, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(data, Mapping):
        raise CheckpointMismatchError("Stage-C training data lineage is malformed")

    def required_path(name: str, *, directory: bool) -> Path:
        value = data.get(name)
        if not isinstance(value, str):
            raise CheckpointMismatchError(f"Stage-C training {name} is missing")
        path = Path(value).expanduser().resolve()
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            raise FileNotFoundError(
                f"Stage-C training provenance is unavailable: {path}"
            )
        return path

    manifest = required_path("manifest_path", directory=False)
    observation_root = required_path("observation_cache_root", directory=True)
    teacher_root = required_path("teacher_cache_root", directory=True)
    derived_root = required_path("derived_geometry_cache_root", directory=True)
    observation_identity = load_receipt_identity(
        observation_root,
        expected_component="ffs-observation",
        manifest_path=manifest,
    )
    teacher_identity = load_receipt_identity(
        teacher_root,
        expected_component="ffs-teacher",
        manifest_path=manifest,
    )
    crop = data.get("hr_crop")
    if (
        not isinstance(crop, (list, tuple))
        or len(crop) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in crop)
    ):
        raise CheckpointMismatchError("Stage-C training HR crop is malformed")
    temporal = CachedTemporalTrainingDataset(
        manifest,
        observation_root,
        teacher_root,
        derived_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        crop_size_hr_hw=(int(crop[0]), int(crop[1])),
        crop_mode="fixed",
        spatial_scale=2,
        student_sequence_length=3,
        vggt_context_pairs=5,
        seed=int(config.get("seed", 42)),
    )
    epipolar_dataset = EpipolarTrainingDataset(temporal)
    recomputed_raw = _validate_base_data_lineage(base_metadata, epipolar_dataset)
    recomputed_base = validate_checkpoint_lineage(
        base_metadata,
        required_stage="temporal",
        observation_cache_identity=observation_identity.to_dict(),
        teacher_cache_identity=teacher_identity.to_dict(),
        derived_cache_lineage=temporal.cache_lineage_summary,
        evaluation_config=config,
    )
    validate_recorded_stage_c_training_lineage(
        stage_c_metadata,
        recomputed_base_lineage=recomputed_base,
        recomputed_raw_lineage=recomputed_raw,
    )
    try:
        observation_rows = [
            json.loads(line)
            for line in (observation_root / "cache_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointMismatchError(
            f"cannot audit Stage-C training right sources: {exc}"
        ) from exc
    source_rows: dict[int, tuple[int, str, int, str]] = {}
    endpoint_indices = {
        int(window.student_indices[-1]) for window in temporal.windows
    }
    for manifest_index in sorted(endpoint_indices):
        row = observation_rows[manifest_index]
        source = row.get("source") if isinstance(row, Mapping) else None
        record = source.get("manifest_record") if isinstance(source, Mapping) else None
        expected_sha256 = source.get("right_sha256") if isinstance(source, Mapping) else None
        right_path_value = record.get("right_path") if isinstance(record, Mapping) else None
        if not isinstance(right_path_value, str) or not isinstance(
            expected_sha256, str
        ):
            raise CheckpointMismatchError(
                "Stage-C training endpoint right-source lineage is malformed"
            )
        right_path = Path(right_path_value).expanduser()
        if not right_path.is_absolute():
            right_path = manifest.parent / right_path
        right_path = right_path.resolve()
        if not right_path.is_file() or sha256_file(right_path) != expected_sha256:
            raise CheckpointMismatchError(
                f"Stage-C training right source changed: {right_path}"
            )
        temporal_record = temporal.records[manifest_index]
        source_rows[manifest_index] = (
            manifest_index,
            temporal_record.sequence_id,
            int(temporal_record.frame_id),
            expected_sha256,
        )
    encoded_sources = json.dumps(
        [source_rows[index] for index in sorted(source_rows)],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    audited_source_digest = {
        "algorithm": (
            "sha256(canonical_json([manifest_index,sequence_id,frame_id,"
            "right_sha256]))"
        ),
        "records": len(source_rows),
        "sha256": hashlib.sha256(encoded_sources).hexdigest(),
    }
    if recomputed_raw.get("endpoint_right_source_digest") != audited_source_digest:
        raise CheckpointMismatchError(
            "current Stage-C training right images differ from checkpoint lineage"
        )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "observation_identity": observation_identity.to_dict(),
        "teacher_identity": teacher_identity.to_dict(),
        "base_lineage": recomputed_base,
        "raw_lineage": recomputed_raw,
        "audited_endpoint_right_source_digest": audited_source_digest,
        "derived_endpoint_records": len(temporal.derived_entries),
        "evaluable_t3_windows": len(temporal),
    }


def validate_epipolar_batch_causality(
    batch: Mapping[str, Any],
) -> list[tuple[int, str, int, str]]:
    """Repeat causal/crop checks and bind endpoint right RGB to its source SHA.

    Returns canonical ``(manifest_index, sequence_id, frame_id, right_sha256)``
    rows for the held-out provenance digest.
    """

    validate_temporal_batch_causality(batch)
    rgb_sequence = batch.get("rgb_hr_sequence")
    rgb_right = batch.get("rgb_right_hr")
    crop_metadata = batch.get("epipolar_crop_hr_px")
    identity_metadata = batch.get("identity_metadata")
    right_paths = batch.get("right_path")
    right_sha256s = batch.get("right_sha256")
    sequence_ids = batch.get("sequence_id")
    frame_ids = batch.get("frame_ids")
    manifest_indices = batch.get("manifest_indices")
    intrinsics_left_sequence = batch.get("K_hr_sequence")
    intrinsics_right_batch = batch.get("K_right_hr")
    intrinsics_sources = batch.get("right_intrinsics_source")
    right_row_scale = batch.get("epipolar_right_row_scale")
    right_row_offset = batch.get("epipolar_right_row_offset_hr_px")
    right_row_mapping_sources = batch.get("epipolar_right_row_mapping_source")
    if not isinstance(rgb_sequence, Tensor) or not isinstance(rgb_right, Tensor):
        raise ValueError("epipolar batch left/right RGB tensors are missing")
    if rgb_sequence.ndim != 5 or rgb_right.shape != rgb_sequence[:, -1].shape:
        raise ValueError("endpoint right RGB does not match the fixed left HR crop")
    batch_size = int(rgb_sequence.shape[0])
    if not isinstance(frame_ids, Tensor) or frame_ids.shape != (batch_size, 3):
        raise ValueError("epipolar batch frame_ids must have shape [B,3]")
    if not isinstance(manifest_indices, Tensor) or manifest_indices.shape != (
        batch_size,
        3,
    ):
        raise ValueError("epipolar batch manifest_indices must have shape [B,3]")
    if not isinstance(intrinsics_left_sequence, Tensor) or (
        intrinsics_left_sequence.shape != (batch_size, 3, 3, 3)
    ):
        raise ValueError("epipolar batch K_hr_sequence must have shape [B,3,3,3]")
    if not isinstance(intrinsics_right_batch, Tensor) or (
        intrinsics_right_batch.shape != (batch_size, 3, 3)
    ):
        raise ValueError("epipolar batch K_right_hr must have shape [B,3,3]")
    if not isinstance(right_row_scale, Tensor) or right_row_scale.shape != (
        batch_size,
    ):
        raise ValueError("epipolar batch right-row scale must have shape [B]")
    if not isinstance(right_row_offset, Tensor) or right_row_offset.shape != (
        batch_size,
    ):
        raise ValueError("epipolar batch right-row offset must have shape [B]")
    expected_runtime_scale = float(
        EPIPOLAR_GEOMETRY_CONTRACT["runtime_right_row_scale"]
    )
    expected_runtime_offset = float(
        EPIPOLAR_GEOMETRY_CONTRACT["runtime_right_row_offset_hr_px"]
    )
    if not torch.allclose(
        right_row_scale,
        torch.full_like(right_row_scale, expected_runtime_scale),
        rtol=0.0,
        atol=1e-7,
    ) or not torch.allclose(
        right_row_offset,
        torch.full_like(right_row_offset, expected_runtime_offset),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("epipolar batch row mapping differs from audited contract")
    if (
        not isinstance(crop_metadata, list)
        or not isinstance(identity_metadata, list)
        or not isinstance(right_paths, list)
        or not isinstance(right_sha256s, list)
        or not isinstance(sequence_ids, list)
        or not isinstance(intrinsics_sources, list)
        or not isinstance(right_row_mapping_sources, list)
        or len(crop_metadata) != len(identity_metadata)
        or len(right_paths) != len(identity_metadata)
        or len(right_sha256s) != len(identity_metadata)
        or len(sequence_ids) != len(identity_metadata)
        or len(intrinsics_sources) != len(identity_metadata)
        or len(right_row_mapping_sources) != len(identity_metadata)
    ):
        raise ValueError("epipolar crop lineage does not match the batch")
    for batch_index, (
        right_crop,
        temporal_identity,
        right_path_value,
        right_sha256,
        sequence_id,
        intrinsics_source,
        row_mapping_source,
    ) in enumerate(
        zip(
            crop_metadata,
            identity_metadata,
            right_paths,
            right_sha256s,
            sequence_ids,
            intrinsics_sources,
            right_row_mapping_sources,
        )
    ):
        if (
            not isinstance(right_sha256, str)
            or len(right_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in right_sha256
            )
        ):
            raise ValueError("endpoint right source SHA-256 is malformed")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("endpoint sequence identity is malformed")
        if intrinsics_source != "manifest.K_right":
            raise ValueError(
                "formal Stage-C requires right_intrinsics_source=manifest.K_right"
            )
        if row_mapping_source != EPIPOLAR_GEOMETRY_CONTRACT["version"]:
            raise ValueError("epipolar row-mapping source differs from contract")
        temporal_crop = (
            temporal_identity.get("crop_hr_px")
            if isinstance(temporal_identity, Mapping)
            else None
        )
        if right_crop != temporal_crop:
            raise ValueError("endpoint right RGB does not use the temporal HR crop")
        per_time = (
            temporal_identity.get("per_time_ffs")
            if isinstance(temporal_identity, Mapping)
            else None
        )
        endpoint = per_time[-1] if isinstance(per_time, list) and per_time else None
        record = endpoint.get("manifest_record") if isinstance(endpoint, Mapping) else None
        source_sha256 = (
            endpoint.get("source_sha256") if isinstance(endpoint, Mapping) else None
        )
        expected_right_sha256 = (
            source_sha256.get("right")
            if isinstance(source_sha256, Mapping)
            else None
        )
        manifest_path_value = (
            temporal_identity.get("manifest_path")
            if isinstance(temporal_identity, Mapping)
            else None
        )
        if not isinstance(record, Mapping) or not isinstance(
            manifest_path_value, str
        ):
            raise ValueError("endpoint stereo source-size lineage is missing")
        try:
            record_intrinsics_left = np.asarray(record["K"], dtype=np.float64)
            record_intrinsics_right = np.asarray(
                record["K_right"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("endpoint calibrated stereo intrinsics are missing") from exc
        if record_intrinsics_left.shape != (3, 3) or (
            record_intrinsics_right.shape != (3, 3)
        ):
            raise ValueError("endpoint stereo intrinsics must be 3x3")
        crop_x = int(right_crop["x"])
        crop_y = int(right_crop["y"])
        expected_left = record_intrinsics_left.copy()
        expected_right = record_intrinsics_right.copy()
        expected_left[0, 2] -= crop_x
        expected_left[1, 2] -= crop_y
        expected_right[0, 2] -= crop_x
        expected_right[1, 2] -= crop_y
        actual_left = intrinsics_left_sequence[batch_index, -1].detach().cpu()
        actual_right = intrinsics_right_batch[batch_index].detach().cpu()
        if not torch.allclose(
            actual_left,
            torch.as_tensor(expected_left, dtype=actual_left.dtype),
            rtol=1e-6,
            atol=1e-4,
        ) or not torch.allclose(
            actual_right,
            torch.as_tensor(expected_right, dtype=actual_right.dtype),
            rtol=1e-6,
            atol=1e-4,
        ):
            raise ValueError("cropped left/right intrinsics differ from the manifest")
        if not math.isclose(
            float(expected_left[0, 0]),
            float(expected_right[0, 0]),
            rel_tol=1e-6,
            abs_tol=1e-4,
        ) or not math.isclose(
            float(expected_left[0, 2]),
            float(expected_right[0, 2]),
            rel_tol=1e-6,
            abs_tol=1e-4,
        ):
            raise ValueError(
                "formal Stage-C horizontal disparity convention requires equal "
                "left/right fx and cx"
            )
        expected_size = record.get("image_size_wh")
        if (
            not isinstance(expected_size, list)
            or len(expected_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in expected_size
            )
        ):
            raise ValueError("endpoint manifest image_size_wh is malformed")
        manifest_directory = Path(manifest_path_value).expanduser().resolve().parent

        def source_path(value: object, name: str) -> Path:
            if not isinstance(value, str):
                raise ValueError(f"endpoint manifest {name} is missing")
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = manifest_directory / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        left_path = source_path(record.get("left_path"), "left_path")
        recorded_right_path = source_path(record.get("right_path"), "right_path")
        loaded_right_path = Path(str(right_path_value)).expanduser().resolve()
        if recorded_right_path != loaded_right_path:
            raise ValueError("loaded right image differs from endpoint manifest")
        actual_right_sha256 = sha256_file(recorded_right_path)
        if not right_sha256 == expected_right_sha256 == actual_right_sha256:
            raise ValueError(
                "loaded right image SHA-256 differs from endpoint FFS lineage"
            )
        expected_wh = tuple(expected_size)
        with Image.open(left_path) as left_image, Image.open(
            recorded_right_path
        ) as right_image:
            if left_image.size != expected_wh or right_image.size != expected_wh:
                raise ValueError(
                    "endpoint left/right original dimensions differ from the "
                    "manifest rectified HR coordinate frame"
                )
    return [
        (
            int(manifest_indices[index, -1].item()),
            str(sequence_ids[index]),
            int(frame_ids[index, -1].item()),
            str(right_sha256s[index]),
        )
        for index in range(batch_size)
    ]


def audit_validation_raw_payload_hashes(
    dataset: CachedTemporalTrainingDataset,
) -> dict[str, Any]:
    """Hash every raw VGGT/FFS payload referenced by formal derived records."""

    expected_roots = {
        "vggt": (dataset.derived_cache_root.parent / "vggt").resolve(),
        "ffs": dataset.observation_cache_root.resolve(),
    }
    digest_rows: list[tuple[int, str, str, str]] = []
    for endpoint_index, entry in sorted(dataset.derived_entries.items()):
        payload = torch.load(
            entry.cache_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        source = metadata.get("source") if isinstance(metadata, Mapping) else None
        if not isinstance(source, Mapping):
            raise CheckpointMismatchError(
                f"derived raw-payload source is missing: {entry.cache_path}"
            )
        for role in ("vggt", "ffs"):
            path_value = source.get(f"{role}_cache_path")
            expected_sha256 = source.get(f"{role}_cache_sha256")
            if not isinstance(path_value, str) or (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha256
                )
            ):
                raise CheckpointMismatchError(
                    f"derived {role} raw-payload lineage is malformed"
                )
            raw_path = Path(path_value).expanduser().resolve()
            try:
                raw_path.relative_to(expected_roots[role])
            except ValueError as exc:
                raise CheckpointMismatchError(
                    f"derived {role} payload escapes its formal cache root"
                ) from exc
            if not raw_path.is_file() or sha256_file(raw_path) != expected_sha256:
                raise CheckpointMismatchError(
                    f"derived {role} raw payload SHA-256 mismatch: {raw_path}"
                )
            digest_rows.append(
                (int(endpoint_index), role, str(raw_path), expected_sha256)
            )
    encoded = json.dumps(
        digest_rows,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "derived_records": len(dataset.derived_entries),
        "vggt_payloads_hashed": sum(row[1] == "vggt" for row in digest_rows),
        "ffs_payloads_hashed": sum(row[1] == "ffs" for row in digest_rows),
        "canonical_reference_digest_sha256": hashlib.sha256(encoded).hexdigest(),
        "all_payload_sha256_match": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-C epipolar refinement on held-out causal T3 data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="relocated Stage-B base; SHA/step must match Stage-C lineage",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-cache-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--derived-cache-root", type=Path, required=True)
    parser.add_argument("--rectification-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualization-samples", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        help="limited held-out smoke; output is always acceptance-ineligible",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _resolved_dict(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return device


def _refiner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise CheckpointMismatchError("Stage-C checkpoint model config is missing")
    return {
        "feature_channels": int(model["epipolar_feature_channels"]),
        "correlation_groups": int(model["epipolar_correlation_groups"]),
        "candidate_offsets_hr_px": tuple(
            float(value) for value in model["epipolar_offsets_hr_px"]
        ),
        "correction_limit_hr_px": float(
            model["epipolar_correction_limit_hr_px"]
        ),
        "confidence_temperature": float(
            model["epipolar_confidence_temperature"]
        ),
        "head_channels": int(model["epipolar_head_channels"]),
    }


def validate_stage_c_training_state(
    payload: Mapping[str, Any],
    *,
    expected_parameters: list[Tensor],
) -> dict[str, Any]:
    """Cross-check optimizer/scheduler progress against the declared step."""

    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise CheckpointMismatchError(
            "formal Stage-C checkpoint must contain at least one optimizer step"
        )
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    rng_states = payload.get("rng_states")
    if not isinstance(optimizer, Mapping) or not isinstance(scheduler, Mapping):
        raise CheckpointMismatchError("Stage-C optimizer/scheduler state is malformed")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list) or not groups:
        raise CheckpointMismatchError("Stage-C AdamW state is empty or malformed")
    parameter_ids: list[object] = []
    for group in groups:
        params = group.get("params") if isinstance(group, Mapping) else None
        if not isinstance(params, list) or not params:
            raise CheckpointMismatchError("Stage-C optimizer param group is malformed")
        parameter_ids.extend(params)
    expected_parameter_tensors = len(expected_parameters)
    if len(parameter_ids) != expected_parameter_tensors or (
        len(set(parameter_ids)) != expected_parameter_tensors
    ):
        raise CheckpointMismatchError(
            "Stage-C optimizer does not cover every refiner parameter tensor"
        )
    if set(state) != set(parameter_ids):
        raise CheckpointMismatchError(
            "Stage-C optimizer state does not match its parameter groups"
        )
    for parameter_id, parameter in zip(
        parameter_ids, expected_parameters, strict=True
    ):
        item = state[parameter_id]
        if not isinstance(item, Mapping):
            raise CheckpointMismatchError("Stage-C AdamW parameter state is malformed")
        optimizer_step = item.get("step")
        if isinstance(optimizer_step, Tensor):
            if optimizer_step.numel() != 1:
                raise CheckpointMismatchError("Stage-C AdamW step is not scalar")
            optimizer_step = float(optimizer_step.item())
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, (int, float))
            or not math.isfinite(float(optimizer_step))
            or float(optimizer_step) != float(step)
        ):
            raise CheckpointMismatchError(
                "Stage-C AdamW parameter step differs from checkpoint step"
            )
        for name in ("exp_avg", "exp_avg_sq"):
            value = item.get(name)
            if not isinstance(value, Tensor) or not bool(
                torch.isfinite(value).all().item()
            ):
                raise CheckpointMismatchError(
                    f"Stage-C AdamW {name} is missing or non-finite"
                )
            if value.shape != parameter.shape:
                raise CheckpointMismatchError(
                    f"Stage-C AdamW {name} shape differs from refiner parameter"
                )
            if not value.is_floating_point() or value.dtype != parameter.dtype:
                raise CheckpointMismatchError(
                    f"Stage-C AdamW {name} dtype differs from refiner parameter"
                )
    if scheduler.get("last_epoch") != step or scheduler.get("_step_count") != step + 1:
        raise CheckpointMismatchError(
            "Stage-C scheduler progress differs from checkpoint step"
        )
    last_lrs = scheduler.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(groups) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in last_lrs
    ):
        raise CheckpointMismatchError("Stage-C scheduler learning rates are malformed")
    config = payload.get("config")
    train = config.get("train") if isinstance(config, Mapping) else None
    if not isinstance(train, Mapping) or len(groups) != 1:
        raise CheckpointMismatchError("Stage-C learning-rate schedule is malformed")
    try:
        base_lr = float(train["learning_rate"])
        weight_decay = float(train["weight_decay"])
        total_steps = int(train["steps_epipolar"])
        warmup_steps = int(train["warmup_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointMismatchError(
            "Stage-C learning-rate config is malformed"
        ) from exc
    expected_lr = base_lr * learning_rate_multiplier(
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    if not math.isclose(
        float(last_lrs[0]), expected_lr, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise CheckpointMismatchError(
            "Stage-C scheduler learning rate differs from warmup/cosine schedule"
        )
    group = groups[0]
    expected_group_values = {
        "initial_lr": base_lr,
        "lr": expected_lr,
        "weight_decay": weight_decay,
        "eps": 1e-8,
    }
    if not isinstance(group, Mapping) or any(
        not isinstance(group.get(name), (int, float))
        or isinstance(group.get(name), bool)
        or not math.isclose(
            float(group[name]), expected, rel_tol=1e-12, abs_tol=1e-15
        )
        for name, expected in expected_group_values.items()
    ) or group.get("betas") != (0.9, 0.999) or any(
        group.get(name) is not False for name in ("amsgrad", "maximize")
    ):
        raise CheckpointMismatchError(
            "Stage-C AdamW hyperparameters differ from the saved config"
        )
    base_lrs = scheduler.get("base_lrs")
    if not isinstance(base_lrs, list) or len(base_lrs) != 1 or not math.isclose(
        float(base_lrs[0]), base_lr, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise CheckpointMismatchError("Stage-C scheduler base learning rate differs")
    expected_rng_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(rng_states, Mapping) or set(rng_states) != expected_rng_keys:
        raise CheckpointMismatchError("Stage-C RNG state schema is malformed")
    python_rng = rng_states["python"]
    numpy_rng = rng_states["numpy"]
    torch_cpu_rng = rng_states["torch_cpu"]
    torch_cuda_rng = rng_states["torch_cuda"]
    valid_python = (
        isinstance(python_rng, tuple)
        and len(python_rng) == 3
        and isinstance(python_rng[0], int)
        and isinstance(python_rng[1], tuple)
    )
    valid_numpy = (
        isinstance(numpy_rng, tuple)
        and len(numpy_rng) == 5
        and isinstance(numpy_rng[0], str)
        and isinstance(numpy_rng[1], np.ndarray)
        and numpy_rng[1].dtype == np.uint32
        and numpy_rng[1].ndim == 1
        and isinstance(numpy_rng[2], int)
    )
    valid_torch_cpu = (
        isinstance(torch_cpu_rng, Tensor)
        and torch_cpu_rng.dtype == torch.uint8
        and torch_cpu_rng.ndim == 1
        and torch_cpu_rng.numel() > 0
    )
    valid_torch_cuda = isinstance(torch_cuda_rng, list) and all(
        isinstance(value, Tensor)
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() > 0
        for value in torch_cuda_rng
    )
    if not (valid_python and valid_numpy and valid_torch_cpu and valid_torch_cuda):
        raise CheckpointMismatchError("Stage-C RNG values are not restorable states")
    return {
        "checkpoint_step": step,
        "optimizer_parameter_tensors": expected_parameter_tensors,
        "optimizer_steps_consistent": True,
        "scheduler_progress_consistent": True,
        "scheduler_learning_rate_consistent": True,
        "rng_schema_consistent": True,
    }


def load_stage_c_checkpoint(
    checkpoint_path: str | Path,
    *,
    evaluation_config: Mapping[str, Any],
) -> tuple[HREpipolarRefiner, dict[str, Any]]:
    """Load a strict full-state Stage-C checkpoint; reject legacy refiner keys."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-C checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("Stage-C checkpoint is not a mapping")
    required = {
        "schema_version",
        "component",
        "model_component",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "config",
        "git_hash",
        "rng_states",
        "base_checkpoint",
        "base_lineage",
        "raw_lineage",
        "geometry_contract",
        "rectification_audit",
        "runtime_source_bundle",
        "training_runtime",
        "supervision",
        "parameter_count",
        "trainable_refiner_parameter_count",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointMismatchError(
            f"Stage-C checkpoint required fields are missing: {missing}"
        )
    if payload["schema_version"] != 1:
        raise CheckpointMismatchError("Stage-C checkpoint schema mismatch")
    if payload["component"] != STAGE_C_COMPONENT or (
        payload["model_component"] != STAGE_C_MODEL_COMPONENT
    ):
        raise CheckpointMismatchError("Stage-C checkpoint component mismatch")
    if not all(
        isinstance(payload[name], Mapping)
        for name in ("model", "optimizer", "scheduler", "scaler", "rng_states")
    ):
        raise CheckpointMismatchError("Stage-C training/model states are malformed")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointMismatchError("Stage-C checkpoint step is malformed")
    git_hash = payload["git_hash"]
    if (
        not isinstance(git_hash, str)
        or len(git_hash) != 40
        or any(character not in "0123456789abcdef" for character in git_hash)
    ):
        raise CheckpointMismatchError("Stage-C checkpoint git hash is malformed")
    if payload["parameter_count"] != EXPECTED_REFINER_PARAMETERS or payload[
        "trainable_refiner_parameter_count"
    ] != EXPECTED_REFINER_PARAMETERS:
        raise CheckpointMismatchError(
            "Stage-C parameter count mismatch: expected "
            f"{EXPECTED_REFINER_PARAMETERS}, got parameter_count="
            f"{payload['parameter_count']!r}, trainable_refiner_parameter_count="
            f"{payload['trainable_refiner_parameter_count']!r}"
        )
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise CheckpointMismatchError("Stage-C checkpoint config is malformed")
    try:
        saved_config = OmegaConf.create(dict(config))
        validate_epipolar_config(saved_config)
    except Exception as exc:
        raise CheckpointMismatchError(
            f"saved Stage-C causal/geometry config is invalid: {exc}"
        ) from exc
    train = config.get("train")
    model_config = config.get("model")
    if (
        not isinstance(train, Mapping)
        or str(train.get("stage")).lower() != "epipolar"
        or not isinstance(model_config, Mapping)
        or model_config.get("epipolar_refinement") is not True
    ):
        raise CheckpointMismatchError("checkpoint is not a Stage-C epipolar run")
    if payload["supervision"] != PSEUDO_GT_SUPERVISION:
        raise CheckpointMismatchError("Stage-C pseudo-GT supervision contract differs")
    if payload["geometry_contract"] != EPIPOLAR_GEOMETRY_CONTRACT:
        raise CheckpointMismatchError("Stage-C epipolar geometry contract differs")
    rectification_audit = payload["rectification_audit"]
    configured_audit = (
        config.get("data", {}).get("epipolar_rectification_audit")
        if isinstance(config.get("data"), Mapping)
        else None
    )
    if not isinstance(rectification_audit, Mapping) or (
        configured_audit != rectification_audit
    ):
        raise CheckpointMismatchError(
            "Stage-C rectification audit/config binding is missing or inconsistent"
        )
    saved_refiner_config = _refiner_config(config)
    current_refiner_config = _refiner_config(evaluation_config)
    if saved_refiner_config["candidate_offsets_hr_px"] != (
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ):
        raise CheckpointMismatchError(
            "formal Stage-C checkpoint must use offsets [-2,-1,0,1,2]"
        )
    if saved_refiner_config != current_refiner_config:
        raise CheckpointMismatchError(
            "evaluation refiner architecture/search differs from checkpoint"
        )
    refiner = HREpipolarRefiner(**saved_refiner_config)
    if refiner.trainable_parameter_count != EXPECTED_REFINER_PARAMETERS:
        raise CheckpointMismatchError(
            "reconstructed HREpipolarRefiner does not have 69,905 parameters"
        )
    try:
        refiner.load_state_dict(payload["model"], strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointMismatchError(
            f"Stage-C model state is incompatible: {exc}"
        ) from exc
    validate_finite_module_state(refiner, label="Stage-C refiner")
    training_state_receipt = validate_stage_c_training_state(
        payload,
        expected_parameters=list(refiner.parameters()),
    )
    training_runtime_receipt = validate_stage_c_training_runtime(
        payload["training_runtime"]
    )
    base = payload["base_checkpoint"]
    if not isinstance(base, Mapping):
        raise CheckpointMismatchError("Stage-C frozen-base lineage is missing")
    if set(base) != {"path", "sha256", "step"}:
        raise CheckpointMismatchError(
            "Stage-C frozen-base reference fields must be path/sha256/step"
        )
    base_path = base["path"]
    base_sha256 = base["sha256"]
    base_step = base["step"]
    if not isinstance(base_path, str) or not base_path:
        raise CheckpointMismatchError("Stage-C frozen-base path is malformed")
    if (
        not isinstance(base_sha256, str)
        or len(base_sha256) != 64
        or any(character not in "0123456789abcdef" for character in base_sha256)
    ):
        raise CheckpointMismatchError("Stage-C frozen-base SHA-256 is malformed")
    if isinstance(base_step, bool) or not isinstance(base_step, int) or base_step < 0:
        raise CheckpointMismatchError("Stage-C frozen-base step is malformed")
    if not isinstance(payload["base_lineage"], Mapping) or not isinstance(
        payload["raw_lineage"], Mapping
    ):
        raise CheckpointMismatchError("Stage-C frozen-base/cache lineage is malformed")
    metadata = {
        "path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "step": step,
        "git_hash": git_hash,
        "parameter_count": EXPECTED_REFINER_PARAMETERS,
        "config": dict(config),
        "base_checkpoint": dict(base),
        "base_lineage": payload["base_lineage"],
        "raw_lineage": payload["raw_lineage"],
        "geometry_contract": payload["geometry_contract"],
        "rectification_audit": dict(rectification_audit),
        "training_state_receipt": training_state_receipt,
        "runtime_source_bundle": payload["runtime_source_bundle"],
        "training_runtime": dict(payload["training_runtime"]),
        "training_runtime_receipt": training_runtime_receipt,
        "supervision": payload["supervision"],
    }
    return refiner, metadata


def validate_finite_module_state(module: torch.nn.Module, *, label: str) -> None:
    """Reject a loaded model with any non-finite parameter or buffer."""

    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be torch.nn.Module")
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise CheckpointMismatchError(f"{label} has non-finite state: {name}")


def validate_runtime_source_bundle(
    checkpoint_bundle: Mapping[str, Any],
    *,
    checkpoint_git_hash: str,
    project_root: Path = PROJECT_ROOT,
    expected_scopes: tuple[str, ...] = STAGE_C_RUNTIME_GIT_SCOPES,
    expected_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Require every prediction/data dependency to match the checkpoint commit."""

    root = project_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise CheckpointMismatchError("runtime source root is not a Git worktree")
    required = {
        "schema_version",
        "git_head",
        "relevant_paths_clean",
        "git_scopes",
        "files",
        "bundle_sha256",
    }
    if set(checkpoint_bundle) != required or (
        checkpoint_bundle.get("schema_version") != 1
        or checkpoint_bundle.get("git_head") != checkpoint_git_hash
        or checkpoint_bundle.get("relevant_paths_clean") is not True
    ):
        raise CheckpointMismatchError("Stage-C runtime source bundle is malformed")
    records = checkpoint_bundle.get("files")
    scopes = checkpoint_bundle.get("git_scopes")
    if not isinstance(records, list) or not records or not isinstance(scopes, list):
        raise CheckpointMismatchError("Stage-C runtime source records are missing")
    if scopes != list(expected_scopes):
        raise CheckpointMismatchError("Stage-C runtime Git scopes are malformed")
    if expected_paths is None:
        expected_paths = (
            "train_epipolar.py",
            "train.py",
            "eval.py",
            "configs/epipolar_x2.yaml",
            "configs/temporal_x2.yaml",
            "configs/mvp_x2.yaml",
            "pyproject.toml",
            *(str(path.relative_to(root)) for path in sorted((root / "src").rglob("*.py"))),
        )
    encoded = json.dumps(
        {"git_head": checkpoint_git_hash, "files": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    bundle_sha256 = hashlib.sha256(encoded).hexdigest()
    if checkpoint_bundle.get("bundle_sha256") != bundle_sha256:
        raise CheckpointMismatchError("Stage-C runtime bundle SHA-256 is malformed")
    relative_paths = [
        record.get("path") if isinstance(record, Mapping) else None
        for record in records
    ]
    if any(not isinstance(path, str) or not path for path in relative_paths) or (
        len(set(relative_paths)) != len(relative_paths)
    ):
        raise CheckpointMismatchError("Stage-C runtime source paths are malformed")
    if relative_paths != list(expected_paths):
        raise CheckpointMismatchError(
            "Stage-C runtime source bundle is truncated or non-canonical"
        )
    files: dict[str, dict[str, str]] = {}
    for record, relative in zip(records, relative_paths, strict=True):
        assert isinstance(relative, str) and isinstance(record, Mapping)
        recorded_sha256 = record.get("sha256")
        if (
            not isinstance(recorded_sha256, str)
            or len(recorded_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in recorded_sha256
            )
        ):
            raise CheckpointMismatchError(
                f"Stage-C runtime source SHA-256 is malformed: {relative}"
            )
        current_path = root / relative
        if not current_path.is_file():
            raise CheckpointMismatchError(f"runtime source is missing: {relative}")
        result = subprocess.run(
            ["git", "show", f"{checkpoint_git_hash}:{relative}"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise CheckpointMismatchError(
                f"checkpoint Git tree has no runtime source: {relative}"
            )
        committed_sha256 = hashlib.sha256(result.stdout).hexdigest()
        current_sha256 = sha256_file(current_path)
        if current_sha256 != committed_sha256 or current_sha256 != recorded_sha256:
            raise CheckpointMismatchError(
                f"runtime source differs from Stage-C checkpoint commit: {relative}"
            )
        files[relative] = {
            "current_sha256": current_sha256,
            "checkpoint_commit_sha256": committed_sha256,
        }
    return {
        "checkpoint_git_hash": checkpoint_git_hash,
        "checkpoint_bundle_sha256": checkpoint_bundle["bundle_sha256"],
        "files": files,
        "all_byte_identical": True,
    }


def _validate_stage_c_and_base_lineage(
    *,
    stage_c_metadata: Mapping[str, Any],
    base_metadata: Mapping[str, Any],
    recomputed_base_lineage: Mapping[str, Any],
    validation_dataset: CachedTemporalTrainingDataset,
    holdout_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind Stage C to the exact base and a video-disjoint validation split."""

    recorded_base = stage_c_metadata.get("base_checkpoint")
    if not isinstance(recorded_base, Mapping):
        raise CheckpointMismatchError("Stage-C base checkpoint lineage is missing")
    for name, actual in (
        ("sha256", base_metadata.get("checkpoint_sha256")),
        ("step", base_metadata.get("step")),
    ):
        if recorded_base.get(name) != actual:
            raise CheckpointMismatchError(
                f"Stage-C frozen base {name} differs from loaded Stage-B checkpoint"
            )
    if stage_c_metadata.get("base_lineage") != recomputed_base_lineage:
        raise CheckpointMismatchError(
            "Stage-C recorded base lineage differs from recomputed validation lineage"
        )
    stage_c_config = stage_c_metadata.get("config")
    stage_c_data = (
        stage_c_config.get("data") if isinstance(stage_c_config, Mapping) else None
    )
    base_config = base_metadata.get("training_config")
    base_data = base_config.get("data") if isinstance(base_config, Mapping) else None
    if not isinstance(stage_c_data, Mapping) or not isinstance(base_data, Mapping):
        raise CheckpointMismatchError("Stage-C/Stage-B data configs are malformed")
    for name in (
        "manifest_path",
        "observation_cache_root",
        "teacher_cache_root",
        "derived_geometry_cache_root",
    ):
        stage_value = stage_c_data.get(name)
        base_value = base_data.get(name)
        if not isinstance(stage_value, str) or not isinstance(base_value, str) or (
            Path(stage_value).expanduser().resolve()
            != Path(base_value).expanduser().resolve()
        ):
            raise CheckpointMismatchError(
                f"Stage-C training {name} differs from frozen Stage-B lineage"
            )
    training_manifest = Path(str(stage_c_data["manifest_path"])).expanduser().resolve()
    if not training_manifest.is_file():
        raise FileNotFoundError(
            f"Stage-C training manifest provenance is unavailable: {training_manifest}"
        )
    training_sequences = {record.sequence_id for record in load_manifest(training_manifest)}
    validation_sequences = {record.sequence_id for record in validation_dataset.records}
    overlap = sorted(training_sequences & validation_sequences)
    if overlap:
        raise CheckpointMismatchError(
            f"Stage-C training and validation videos overlap: {overlap}"
        )
    if holdout_lineage.get("sequence_overlap"):
        raise CheckpointMismatchError("Stage-B training and validation videos overlap")
    recorded_raw = stage_c_metadata.get("raw_lineage")
    evaluation_raw = holdout_lineage.get("evaluation_raw_vggt")
    if not isinstance(recorded_raw, Mapping) or not isinstance(
        evaluation_raw, Mapping
    ):
        raise CheckpointMismatchError("Stage-C/raw validation lineage is missing")
    if recorded_raw.get("raw_vggt_identity") != evaluation_raw.get("identity"):
        raise CheckpointMismatchError(
            "Stage-C training and validation raw VGGT identities differ"
        )
    return {
        "stage_c_training_manifest": str(training_manifest),
        "stage_c_training_manifest_sha256": sha256_file(training_manifest),
        "stage_c_training_sequences": sorted(training_sequences),
        "validation_sequences": sorted(validation_sequences),
        "sequence_overlap": [],
        "base_checkpoint_sha256": base_metadata["checkpoint_sha256"],
        "base_checkpoint_step": base_metadata["step"],
        "raw_vggt_identity": evaluation_raw["identity"],
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def _rgb_uint8(rgb: Tensor) -> np.ndarray:
    return np.rint(
        rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)


def _save_visualization(
    root: Path,
    *,
    sample_name: str,
    rgb_left_hr: Tensor,
    rgb_right_hr: Tensor,
    base_disparity_hr_px: Tensor,
    refined_disparity_hr_px: Tensor,
    correction_hr_px: Tensor,
    confidence: Tensor,
    target_disparity_hr_px: Tensor,
    target_trusted_mask: Tensor,
    candidate_valid_mask: Tensor,
    correction_limit_hr_px: float,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    directory = root / sample_name
    mask = target_trusted_mask.to(dtype=torch.bool)
    trusted_target_values = target_disparity_hr_px[
        mask & torch.isfinite(target_disparity_hr_px)
    ]
    disparity_minimum = (
        float(trusted_target_values.min().item())
        if trusted_target_values.numel()
        else None
    )
    disparity_maximum = (
        float(trusted_target_values.max().item())
        if trusted_target_values.numel()
        else None
    )
    base_error = (base_disparity_hr_px - target_disparity_hr_px).abs()
    refined_error = (refined_disparity_hr_px - target_disparity_hr_px).abs()
    finite_error_values = torch.cat(
        (
            base_error[mask & torch.isfinite(base_error)],
            refined_error[mask & torch.isfinite(refined_error)],
        )
    )
    error_maximum = (
        float(finite_error_values.max().item())
        if finite_error_values.numel()
        else None
    )
    save_rgb_uint8(directory / "rgb_left.png", _rgb_uint8(rgb_left_hr))
    save_rgb_uint8(directory / "rgb_right.png", _rgb_uint8(rgb_right_hr))
    maps = (
        (
            "base_disparity_hr_px.png",
            base_disparity_hr_px,
            None,
            disparity_minimum,
            disparity_maximum,
        ),
        (
            "refined_disparity_hr_px.png",
            refined_disparity_hr_px,
            None,
            disparity_minimum,
            disparity_maximum,
        ),
        (
            "target_disparity_hr_px.png",
            target_disparity_hr_px,
            mask,
            disparity_minimum,
            disparity_maximum,
        ),
        (
            "correction_hr_px.png",
            correction_hr_px,
            None,
            -correction_limit_hr_px,
            correction_limit_hr_px,
        ),
        (
            "base_absolute_error_hr_px.png",
            base_error,
            mask,
            0.0,
            error_maximum,
        ),
        (
            "refined_absolute_error_hr_px.png",
            refined_error,
            mask,
            0.0,
            error_maximum,
        ),
    )
    for filename, value, valid, minimum, maximum in maps:
        save_rgb_uint8(
            directory / filename,
            scalar_to_rgb_uint8(
                value,
                valid_mask=valid,
                minimum=minimum,
                maximum=maximum,
            ),
        )
    save_rgb_uint8(
        directory / "epipolar_confidence.png",
        grayscale_to_rgb_uint8(confidence, minimum=0.0, maximum=1.0),
    )
    save_rgb_uint8(
        directory / "target_trusted_mask.png",
        grayscale_to_rgb_uint8(
            target_trusted_mask.to(dtype=torch.float32), minimum=0.0, maximum=1.0
        ),
    )
    save_rgb_uint8(
        directory / "candidate_valid_mask.png",
        grayscale_to_rgb_uint8(
            candidate_valid_mask.to(dtype=torch.float32), minimum=0.0, maximum=1.0
        ),
    )
    (directory / "visualization_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_name": sample_name,
                "units": "HR pixels",
                "shared_disparity_display_range_hr_px": [
                    disparity_minimum,
                    disparity_maximum,
                ],
                "shared_absolute_error_display_range_hr_px": [
                    0.0,
                    error_maximum,
                ],
                "correction_display_range_hr_px": [
                    -correction_limit_hr_px,
                    correction_limit_hr_px,
                ],
                "target_trusted_pixels": int(mask.sum().item()),
                "candidate_valid_pixels": int(
                    candidate_valid_mask.to(dtype=torch.bool).sum().item()
                ),
                "provenance": dict(provenance or {}),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, methods: Mapping[str, Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            name
            for method in methods.values()
            for name, value in method.items()
            if isinstance(value, Mapping) and "value" in value
        }
    )
    fields = ["method", "target_type", "paper_ground_truth", "point_to_plane"]
    for name in metric_names:
        fields.extend((name, f"{name}_valid", f"{name}_count", f"{name}_numerator"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name, method in methods.items():
            row: dict[str, Any] = {
                "method": method_name,
                "target_type": PSEUDO_GT_LABEL,
                "paper_ground_truth": False,
                "point_to_plane": "NOT_AVAILABLE",
            }
            for name in metric_names:
                metric = method.get(name)
                if not isinstance(metric, Mapping):
                    continue
                row[name] = metric["value"]
                row[f"{name}_valid"] = metric["valid"]
                row[f"{name}_count"] = metric["count"]
                row[f"{name}_numerator"] = metric["numerator"]
            writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    _positive_int(args.batch_size, "batch_size")
    _nonnegative_int(args.num_workers, "num_workers")
    _nonnegative_int(args.visualization_samples, "visualization_samples")
    if args.limit is not None:
        _positive_int(args.limit, "limit")
    config = resolve_epipolar_config(args.config, args.overrides)
    for name, value in (
        ("data.manifest_path", args.manifest),
        ("data.observation_cache_root", args.observation_cache_root),
        ("data.teacher_cache_root", args.teacher_cache_root),
        ("data.derived_geometry_cache_root", args.derived_cache_root),
        ("data.epipolar_rectification_audit_path", args.rectification_audit),
    ):
        OmegaConf.update(config, name, str(value.expanduser().resolve()), merge=False)
    OmegaConf.update(config, "data.crop_mode", "fixed", merge=False)
    validate_epipolar_config(config)
    seed_everything(int(config.seed), deterministic=True)
    manifest = args.manifest.expanduser().resolve()
    observation_root = args.observation_cache_root.expanduser().resolve()
    teacher_root = args.teacher_cache_root.expanduser().resolve()
    derived_root = args.derived_cache_root.expanduser().resolve()
    rectification_audit_path = args.rectification_audit.expanduser().resolve()
    for path, directory in (
        (manifest, False),
        (observation_root, True),
        (teacher_root, True),
        (derived_root, True),
        (rectification_audit_path, False),
    ):
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(path)
    observation_identity = load_receipt_identity(
        observation_root,
        expected_component="ffs-observation",
        manifest_path=manifest,
    )
    teacher_identity = load_receipt_identity(
        teacher_root,
        expected_component="ffs-teacher",
        manifest_path=manifest,
    )
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    temporal_dataset = CachedTemporalTrainingDataset(
        manifest,
        observation_root,
        teacher_root,
        derived_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        crop_size_hr_hw=(crop_height, crop_width),
        crop_mode="fixed",
        spatial_scale=2,
        student_sequence_length=3,
        vggt_context_pairs=5,
        seed=int(config.seed),
    )
    formal_coverage = _validate_formal_temporal_coverage(temporal_dataset)
    canonical_coverage = require_formal_stage_c_coverage(
        formal_coverage,
        manifest_sha256=sha256_file(manifest),
    )
    dataset = EpipolarTrainingDataset(temporal_dataset)
    selected_count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    selected = Subset(dataset, range(selected_count))

    evaluation_config = _resolved_dict(config)
    refiner, stage_c_metadata = load_stage_c_checkpoint(
        args.checkpoint,
        evaluation_config=evaluation_config,
    )
    runtime_source_bundle = validate_runtime_source_bundle(
        stage_c_metadata["runtime_source_bundle"],
        checkpoint_git_hash=str(stage_c_metadata["git_hash"]),
    )
    crop_contract = validate_formal_crop_contract(
        stage_c_metadata,
        evaluation_config,
        limited_smoke=args.limit is not None,
    )
    rectification_audit = validate_rectification_audit_binding(
        stage_c_metadata,
        receipt_path=rectification_audit_path,
        validation_manifest_sha256=sha256_file(manifest),
    )
    recorded_base = stage_c_metadata["base_checkpoint"]
    recorded_base_path = recorded_base.get("path")
    base_path = (
        args.base_checkpoint.expanduser().resolve()
        if args.base_checkpoint is not None
        else Path(str(recorded_base_path)).expanduser().resolve()
    )
    if not base_path.is_file():
        raise FileNotFoundError(
            "recorded Stage-B base is unavailable; pass --base-checkpoint with "
            f"the exact SHA-bound artifact: {base_path}"
        )
    base_model = build_model(config)
    base_metadata = load_model_for_evaluation(
        base_path,
        base_model,
        expected_parameter_count=base_model.trainable_parameter_count,
        require_full_training_state=True,
    )
    validate_finite_module_state(base_model, label="Stage-B frozen base")
    if base_metadata["checkpoint_sha256"] != recorded_base.get("sha256") or (
        base_metadata["step"] != recorded_base.get("step")
    ):
        raise CheckpointMismatchError(
            "loaded Stage-B base SHA/step differs from Stage-C checkpoint lineage"
        )
    training_lineage_receipt = recompute_stage_c_training_lineage(
        stage_c_metadata,
        base_metadata=base_metadata,
    )
    completion = checkpoint_completion_status(stage_c_metadata, base_metadata)
    recomputed_base_lineage = validate_checkpoint_lineage(
        base_metadata,
        required_stage="temporal",
        observation_cache_identity=observation_identity.to_dict(),
        teacher_cache_identity=teacher_identity.to_dict(),
        derived_cache_lineage=temporal_dataset.cache_lineage_summary,
        evaluation_config=evaluation_config,
    )
    holdout_lineage = _audit_temporal_holdout_and_raw_lineage(
        checkpoint_metadata=base_metadata,
        dataset=temporal_dataset,
        evaluation_manifest_path=manifest,
        observation_identity=observation_identity.to_dict(),
    )
    raw_payload_audit = audit_validation_raw_payload_hashes(temporal_dataset)
    stage_lineage = _validate_stage_c_and_base_lineage(
        stage_c_metadata=stage_c_metadata,
        base_metadata=base_metadata,
        recomputed_base_lineage=recomputed_base_lineage,
        validation_dataset=temporal_dataset,
        holdout_lineage=holdout_lineage,
    )
    device = _resolve_device(args.device)
    execution_contract = validate_formal_execution_contract(
        stage_c_metadata,
        evaluation_config,
        device=device,
        limited_smoke=args.limit is not None,
    )
    stage = FrozenTemporalEpipolarStage(
        base_model,
        refiner,
        lambda module, batch: predict_frozen_stage_b_endpoint(
            module, batch, config=config
        ),
    ).to(device).eval()
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_epipolar_training_samples,
    )
    method_names = (
        "T3_VGGT_base",
        "T3_VGGT_base_clamp0",
        "T3_VGGT_epipolar",
        "T3_VGGT_epipolar_clamp0",
    )
    accumulators = {
        name: MethodMetricAccumulator() for name in method_names
    }
    paired_accumulator = MethodMetricAccumulator()
    correction_signed = FiniteStatisticsAccumulator()
    correction_absolute = FiniteStatisticsAccumulator()
    confidence_stats = FiniteStatisticsAccumulator()
    right_row_scale_stats = FiniteStatisticsAccumulator()
    right_row_offset_stats = FiniteStatisticsAccumulator()
    metadata_row_scale_stats = FiniteStatisticsAccumulator()
    metadata_row_offset_stats = FiniteStatisticsAccumulator()
    validation_right_source_rows: dict[int, tuple[int, str, int, str]] = {}
    correction_nonzero: list[MetricResult] = []
    correction_saturated: list[MetricResult] = []
    candidate_coverage: list[MetricResult] = []
    output_dir = args.output.expanduser().resolve()
    if (output_dir / "metrics.json").exists() or (
        output_dir / "metrics.csv"
    ).exists():
        raise FileExistsError(
            f"evaluation output already contains metrics: {output_dir}"
        )
    visualization_limit = min(args.visualization_samples, selected_count)
    visualized = 0
    started = time.perf_counter()
    use_bf16 = device.type == "cuda" and str(config.train.precision).lower() == "bf16"
    with torch.inference_mode():
        for batch in loader:
            right_source_rows = validate_epipolar_batch_causality(batch)
            for row in right_source_rows:
                manifest_index = row[0]
                previous = validation_right_source_rows.get(manifest_index)
                if previous is not None:
                    raise RuntimeError(
                        "held-out Stage-C endpoint appears more than once: "
                        f"{manifest_index}"
                    )
                validation_right_source_rows[manifest_index] = row
            batch = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = stage(batch)
            runtime_row_scale = batch["epipolar_right_row_scale"].float()
            runtime_row_offset = batch[
                "epipolar_right_row_offset_hr_px"
            ].float()
            if not torch.allclose(
                output.refinement.right_row_scale.float(),
                runtime_row_scale,
                rtol=0.0,
                atol=1e-7,
            ) or not torch.allclose(
                output.refinement.right_row_offset_hr_px.float(),
                runtime_row_offset,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "runtime epipolar row mapping differs from audited contract"
                )
            right_row_scale_stats.update(runtime_row_scale)
            right_row_offset_stats.update(runtime_row_offset)
            endpoint_left_intrinsics = batch["K_hr_sequence"][:, -1].float()
            endpoint_right_intrinsics = batch["K_right_hr"].float()
            metadata_row_scale = (
                endpoint_right_intrinsics[:, 1, 1]
                / endpoint_left_intrinsics[:, 1, 1]
            )
            metadata_row_offset = (
                endpoint_right_intrinsics[:, 1, 2]
                - metadata_row_scale * endpoint_left_intrinsics[:, 1, 2]
            )
            metadata_row_scale_stats.update(metadata_row_scale)
            metadata_row_offset_stats.update(metadata_row_offset)
            base = output.base_disparity_hr_px.float()
            refined = output.refined_disparity_hr_px.float()
            target_sequence = batch.get("teacher_disparity_hr_px_sequence")
            trusted_sequence = batch.get("teacher_trusted_mask_sequence")
            if not isinstance(target_sequence, Tensor) or not isinstance(
                trusted_sequence, Tensor
            ):
                raise ValueError(
                    "held-out Stage-C evaluation requires teacher pseudo-GT/trusted cache"
                )
            target = target_sequence[:, -1].float()
            trusted_target = trusted_sequence[:, -1]
            output_size = tuple(int(value) for value in target.shape[-2:])
            _, confidence_ffs_hr, valid_ffs_hr, trusted_ffs_hr = (
                upsample_ffs_inputs_to_hr(
                    batch["observation_disparity_hr_px_sequence"][:, -1],
                    batch["observation_confidence_sequence"][:, -1],
                    batch["observation_valid_mask_sequence"][:, -1],
                    batch["observation_trusted_mask_sequence"][:, -1],
                    output_size_hw=output_size,
                )
            )
            predictions = {
                "T3_VGGT_base": base,
                "T3_VGGT_base_clamp0": physical_disparity_clamp_min_zero(base),
                "T3_VGGT_epipolar": refined,
                "T3_VGGT_epipolar_clamp0": (
                    physical_disparity_clamp_min_zero(refined)
                ),
            }
            for method_name, prediction in predictions.items():
                accumulators[method_name].update(
                    compute_sample_metrics(
                        prediction,
                        target,
                        target_trusted_mask=trusted_target,
                        ffs_confidence_hr=confidence_ffs_hr,
                        ffs_valid_mask_hr=valid_ffs_hr,
                        ffs_trusted_mask_hr=trusted_ffs_hr,
                    )
                )
            paired_accumulator.update(
                paired_refinement_metrics(base, refined, target, trusted_target)
            )
            candidate_valid = output.refinement.candidate_valid_mask.any(
                dim=1, keepdim=True
            )
            correction = output.correction_hr_px.float()
            correction_signed.update(correction, candidate_valid)
            correction_absolute.update(correction.abs(), candidate_valid)
            confidence_stats.update(output.confidence.float(), candidate_valid)
            correction_nonzero.append(
                _metric_rate(correction.abs() > 1e-6, candidate_valid)
            )
            limit = float(config.model.epipolar_correction_limit_hr_px)
            correction_saturated.append(
                _metric_rate(correction.abs() >= 0.99 * limit, candidate_valid)
            )
            candidate_coverage.append(
                _metric_rate(
                    candidate_valid,
                    torch.ones_like(candidate_valid, dtype=torch.bool),
                )
            )
            for item in range(target.shape[0]):
                if visualized >= visualization_limit:
                    break
                sequence = str(batch["sequence_id"][item]).replace("/", "_")
                frame = int(batch["frame_ids"][item, -1].item())
                _save_visualization(
                    output_dir / "visualizations",
                    sample_name=f"{visualized:04d}_{sequence}_{frame}",
                    rgb_left_hr=batch["rgb_hr_sequence"][item, -1],
                    rgb_right_hr=batch["rgb_right_hr"][item],
                    base_disparity_hr_px=base[item],
                    refined_disparity_hr_px=refined[item],
                    correction_hr_px=correction[item],
                    confidence=output.confidence[item].float(),
                    target_disparity_hr_px=target[item],
                    target_trusted_mask=trusted_target[item],
                    candidate_valid_mask=candidate_valid[item],
                    correction_limit_hr_px=limit,
                    provenance={
                        "sequence_id": batch["sequence_id"][item],
                        "frame_id": frame,
                        "manifest_index": int(
                            batch["manifest_indices"][item, -1].item()
                        ),
                        "right_path": batch["right_path"][item],
                        "right_sha256": batch["right_sha256"][item],
                        "crop_hr_px": batch["epipolar_crop_hr_px"][item],
                    },
                )
                visualized += 1
    elapsed = time.perf_counter() - started
    if len(validation_right_source_rows) != selected_count:
        raise RuntimeError(
            "held-out right-source audit coverage differs from evaluated windows"
        )
    encoded_right_sources = json.dumps(
        [
            validation_right_source_rows[index]
            for index in sorted(validation_right_source_rows)
        ],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    validation_right_source_audit = {
        "algorithm": (
            "sha256(canonical_json([manifest_index,sequence_id,frame_id,"
            "right_sha256]))"
        ),
        "records": len(validation_right_source_rows),
        "sha256": hashlib.sha256(encoded_right_sources).hexdigest(),
        "all_source_sha256_match": True,
    }
    aggregate_methods = {
        name: accumulator.finalize() for name, accumulator in accumulators.items()
    }
    methods = {
        name: {metric: result.to_dict() for metric, result in values.items()}
        for name, values in aggregate_methods.items()
    }
    for name, method in methods.items():
        method["output_variant"] = {
            "type": (
                "PHYSICAL_CLAMP_MIN_ZERO"
                if name.endswith("_clamp0")
                else "RAW_MODEL_OUTPUT"
            ),
            "epsilon_fill": False,
        }
        method["point_to_plane_error_m"] = dict(POINT_TO_PLANE_NOT_AVAILABLE)
    paired = {
        name: result.to_dict()
        for name, result in paired_accumulator.finalize().items()
    }
    comparisons = {
        "raw_refined_vs_base": comparison_from_aggregates(
            aggregate_methods["T3_VGGT_base"],
            aggregate_methods["T3_VGGT_epipolar"],
        ),
        "clamp0_refined_vs_base": comparison_from_aggregates(
            aggregate_methods["T3_VGGT_base_clamp0"],
            aggregate_methods["T3_VGGT_epipolar_clamp0"],
        ),
        "raw_epe_change": aggregate_metric_change(
            aggregate_methods["T3_VGGT_base"],
            aggregate_methods["T3_VGGT_epipolar"],
            "epe_px",
        ),
        "paired_pixel_changes": paired,
        "raw_all_metric_changes": {
            metric: aggregate_metric_change(
                aggregate_methods["T3_VGGT_base"],
                aggregate_methods["T3_VGGT_epipolar"],
                metric,
            )
            for metric in aggregate_methods["T3_VGGT_base"]
        },
        "clamp0_all_metric_changes": {
            metric: aggregate_metric_change(
                aggregate_methods["T3_VGGT_base_clamp0"],
                aggregate_methods["T3_VGGT_epipolar_clamp0"],
                metric,
            )
            for metric in aggregate_methods["T3_VGGT_base_clamp0"]
        },
    }
    refinement_statistics = {
        "correction_signed_hr_px": correction_signed.finalize().to_dict(),
        "correction_absolute_hr_px": correction_absolute.finalize().to_dict(),
        "confidence": confidence_stats.finalize().to_dict(),
        "correction_nonzero_rate": aggregate_metric_results(
            correction_nonzero
        ).to_dict(),
        "correction_saturated_rate": aggregate_metric_results(
            correction_saturated
        ).to_dict(),
        "candidate_coverage_rate": aggregate_metric_results(
            candidate_coverage
        ).to_dict(),
    }
    runtime_geometry_statistics = {
        "contract": dict(EPIPOLAR_GEOMETRY_CONTRACT),
        "right_intrinsics_source": "manifest.K_right",
        "right_row_scale": right_row_scale_stats.finalize().to_dict(),
        "right_row_offset_hr_px": right_row_offset_stats.finalize().to_dict(),
        "diagnostic_K_right_implied_row_scale": (
            metadata_row_scale_stats.finalize().to_dict()
        ),
        "diagnostic_K_right_implied_row_offset_hr_px": (
            metadata_row_offset_stats.finalize().to_dict()
        ),
        "metadata_runtime_mismatch_is_expected": True,
    }
    full_selection = args.limit is None and selected_count == len(dataset)
    acceptance_eligible = (
        full_selection
        and bool(completion["all_complete"])
        and bool(crop_contract["eligible"])
        and bool(execution_contract["eligible"])
    )
    if acceptance_eligible:
        status = "EVALUATION_COMPLETE"
    elif not full_selection:
        status = "LIMITED_SMOKE_ONLY"
    else:
        status = "INTERMEDIATE_CHECKPOINT_EVALUATION"
    report = {
        "schema_version": 1,
        "stage": "STAGE_C_EPIPOLAR_HELD_OUT",
        "status": status,
        "claims": {
            "acceptance_eligible": acceptance_eligible,
            "paper_ground_truth": False,
            "paper_accuracy": False,
            "pseudo_gt_engineering_only": True,
            "future_frames": False,
            "point_to_plane": "NOT_AVAILABLE",
            "performance_acceptance_claimed": False,
            "primary_claim_method": "T3_VGGT_epipolar",
            "primary_claim_variant": "RAW_MODEL_OUTPUT",
            "primary_comparison": "raw_refined_vs_base",
            "clamp0_acceptance_owner": False,
            "primary_raw_output_health": {
                name: methods["T3_VGGT_epipolar"][name]
                for name in (
                    "output_invalid_rate",
                    "output_negative_rate",
                    "output_nan_rate",
                    "output_infinite_rate",
                    "output_zero_rate",
                )
            },
        },
        "target": {
            "type": PSEUDO_GT_LABEL,
            "warning": (
                "Trusted HR FFS teacher pseudo-GT is engineering supervision, "
                "not independent paper ground truth."
            ),
        },
        "postprocess_contract": {
            "operation": (
                "torch.where(isfinite(disparity_hr_px) & "
                "(disparity_hr_px < 0), 0, disparity_hr_px)"
            ),
            "epsilon_fill": False,
            "zero_remains_invalid": True,
            "nan_and_positive_negative_infinity_preserved": True,
            "completeness_is_not_fabricated": True,
            "raw_rows_are_retained": True,
            "role": "DECLARED_PHYSICAL_POSTPROCESS_DIAGNOSTIC",
        },
        "methods": methods,
        "comparisons": comparisons,
        "refinement_statistics": refinement_statistics,
        "runtime_geometry_statistics": runtime_geometry_statistics,
        "windows_evaluated": selected_count,
        "full_evaluable_windows": len(dataset),
        "formal_coverage": formal_coverage,
        "canonical_coverage": canonical_coverage,
        "fixed_hr_crop": [crop_height, crop_width],
        "crop_contract": crop_contract,
        "visualizations_written": visualized,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "checkpoint_completion": completion,
        "execution_contract": execution_contract,
        "stage_c_checkpoint": stage_c_metadata,
        "stage_b_base_checkpoint": {
            "path": base_metadata["path"],
            "sha256": base_metadata["checkpoint_sha256"],
            "step": base_metadata["step"],
        },
        "lineage": {
            "recomputed_stage_c_training": training_lineage_receipt,
            "rectification_audit": rectification_audit,
            "stage_c_and_base": stage_lineage,
            "held_out_validation": holdout_lineage,
            "validation_endpoint_right_sources": validation_right_source_audit,
            "validation_raw_payload_audit": raw_payload_audit,
            "observation_identity": observation_identity.to_dict(),
            "teacher_identity": teacher_identity.to_dict(),
        },
        "source_hashes": {
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "repository_git_hash": repository_git_hash(PROJECT_ROOT),
            "validation_manifest_sha256": sha256_file(manifest),
            "stage_c_checkpoint_sha256": stage_c_metadata["checkpoint_sha256"],
            "stage_b_checkpoint_sha256": base_metadata["checkpoint_sha256"],
            "runtime_source_bundle": runtime_source_bundle,
        },
        "resolved_config": evaluation_config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    metrics_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_csv, methods)
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceptance_eligible": acceptance_eligible,
                "paper_ground_truth": False,
                "windows_evaluated": selected_count,
                "metrics_json": str(metrics_json),
                "metrics_csv": str(metrics_csv),
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
