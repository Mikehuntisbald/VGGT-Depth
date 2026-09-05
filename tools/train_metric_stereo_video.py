#!/usr/bin/env python3
"""Train the causal Metric Stereo Video Geometry model with BF16 and FSDP.

The first formal configuration supervises only each causal prefix endpoint.
VGGT is bidirectional inside that past-to-current prefix, but never receives a
future frame.  The explicit temporal memory remains a forward causal scan.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


# Checkpoints are deliberately versioned independently of the model/loss
# schemas.  Version 1 was the initial rank-file format (without a manifest
# digest or resume metadata); version 2 adds the integrity/contract fields
# needed for a safe same-world-size restart.  The loader below keeps v1
# readable so an interrupted first run can still be resumed.
CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_KIND = "metric_stereo_video_fsdp"
_STEP_DIRECTORY_RE = re.compile(r"^step_(?P<step>[0-9]+)$")
_RANK_FILE_RE = re.compile(r"^rank_(?P<rank>[0-9]{4})\.pt$")

# These knobs affect logging/housekeeping only.  They are intentionally
# excluded from the resume contract so a run may be resumed into another
# output directory or with a different validation/checkpoint cadence.  The
# target step remains part of the contract because it is also the cosine LR
# horizon.  Every other resolved config value remains part of the contract
# (including data geometry, model dimensions, losses, optimizer
# hyperparameters, precision and gradient accumulation).
_RESUME_RUNTIME_TRAIN_KEYS = frozenset(
    {
        "output_dir",
        "log_interval",
        "validation_interval",
        "validation_batches",
        "checkpoint_interval",
        "keep_last_checkpoints",
    }
)


class CheckpointResumeError(RuntimeError):
    """Raised when a joint-training checkpoint is unsafe to resume."""


@dataclass(frozen=True, slots=True)
class ResumeState:
    """State restored from one same-world-size checkpoint directory."""

    step: int
    epoch: int
    batch_in_epoch: int


def _canonical_json(value: Any) -> str:
    """Serialize a resolved config deterministically and reject non-JSON data."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("checkpoint config must contain finite JSON values") from exc


def _resume_contract_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return config fields that alter model/data/optimization semantics.

    Output and observation cadence are operational controls rather than
    training-state identity, so they may change across a restart.  The target
    step is retained because changing it changes the cosine learning-rate
    schedule.  The returned object is a deep JSON round-trip, so callers cannot
    mutate the caller's nested mappings accidentally.
    """

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    try:
        contract = json.loads(_canonical_json(dict(config)))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical_json guards this
        raise TypeError("checkpoint config could not be canonicalized") from exc
    if not isinstance(contract, dict):  # pragma: no cover - defensive
        raise TypeError("checkpoint config must serialize to a mapping")
    train = contract.get("train")
    if isinstance(train, dict):
        for key in _RESUME_RUNTIME_TRAIN_KEYS:
            train.pop(key, None)
    return contract


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """SHA-256 fingerprint of the strict resume training contract."""

    return hashlib.sha256(
        _canonical_json(_resume_contract_config(config)).encode("utf-8")
    ).hexdigest()


def _input_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash manifests whose bytes define the resumed sample domain."""

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("checkpoint config requires a data mapping")
    return {
        name: _sha256(_project_path(data[name], f"data.{name}"))
        for name in ("train_manifest", "validation_manifest")
    }


def _runtime_source_fingerprints() -> dict[str, str]:
    """Bind checkpoints to the exact Python implementation that produced them."""

    paths = [
        PROJECT_ROOT / "tools" / "train_metric_stereo_video.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
    }


def _checkpoint_cursor(
    step: int,
    *,
    accumulation: int,
    batches_per_epoch: int,
) -> dict[str, int]:
    """Map completed optimizer updates to the next data-loader cursor."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("gradient accumulation must be a positive integer")
    if isinstance(batches_per_epoch, bool) or not isinstance(batches_per_epoch, int) or batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be a positive integer")
    consumed = step * accumulation
    return {
        "epoch": consumed // batches_per_epoch,
        "batch_in_epoch": consumed % batches_per_epoch,
        "accumulation": accumulation,
        "batches_per_epoch": batches_per_epoch,
    }


def _validate_cursor(value: Any, *, step: int) -> tuple[int, int]:
    """Validate serialized cursor metadata and return ``(epoch, offset)``."""

    if value is None:
        return 0, 0
    if not isinstance(value, Mapping):
        raise CheckpointResumeError("checkpoint data_cursor is not a mapping")
    epoch = value.get("epoch")
    offset = value.get("batch_in_epoch")
    for name, item in (("epoch", epoch), ("batch_in_epoch", offset)):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CheckpointResumeError(f"checkpoint data_cursor {name} is invalid")
    # A cursor with no explicit loader contract is still useful for old or
    # hand-authored checkpoints.  If supplied, the invariant below catches a
    # stale cursor before silently skipping/repeating data.
    batches = value.get("batches_per_epoch")
    accumulation = value.get("accumulation")
    if batches is not None:
        if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
            raise CheckpointResumeError("checkpoint data_cursor batches_per_epoch is invalid")
        if offset >= batches:
            raise CheckpointResumeError("checkpoint data_cursor batch_in_epoch exceeds epoch length")
        if accumulation is not None:
            if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
                raise CheckpointResumeError("checkpoint data_cursor accumulation is invalid")
            expected = epoch * batches + offset
            if expected != step * accumulation:
                raise CheckpointResumeError("checkpoint data_cursor disagrees with step")
    return int(epoch), int(offset)

from backbones.trainable_stereo import (  # noqa: E402
    TrainableFastFoundationStereo,
    load_fast_foundation_stereo,
)
from backbones.trainable_vggt_omega import (  # noqa: E402
    TrainableVGGTOmega,
    load_vggt_omega,
)
from data.raw_stereo_video_dataset import (  # noqa: E402
    RawStereoVideoClipDataset,
    collate_raw_stereo_video_samples,
)
from geometry.metric_reprojection import (  # noqa: E402
    stereo_reproject_right_to_left,
    temporal_reproject_previous_to_current,
)
from geometry.zbuffer_reproject import zbuffer_reproject  # noqa: E402
from losses.metric_stereo_video import (  # noqa: E402
    MetricStereoVideoLoss,
    MetricStereoVideoLossBreakdown,
    MetricStereoVideoLossWeights,
    normalized_log_depth_loss,
)
from models.metric_stereo_video_geometry import (  # noqa: E402
    CausalMetricStereoVideoGeometry,
    align_vggt_inverse_depth_to_metric_stereo,
)
from models.metric_stereo_video_system import (  # noqa: E402
    MetricStereoVideoSystem,
    MetricStereoVideoSystemOutput,
    left_right_stereo_consistency,
)


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True, slots=True)
class TrainingLoss:
    total: Tensor
    breakdown: MetricStereoVideoLossBreakdown
    aligned_vggt_depth_auxiliary: Tensor
    active_temporal_fraction: Tensor
    gauge_valid_fraction: Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/metric_stereo_video/best_t8_512.yaml",
    )
    parser.add_argument("--steps", type=int, help="Override optimizer steps")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--resume", type=Path, help="Same-world-size rank checkpoint directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one real forward/backward optimizer step without writing a checkpoint",
    )
    parser.add_argument(
        "--no-fsdp",
        action="store_true",
        help="Single-process diagnostic only; formal multi-GPU runs require FSDP",
    )
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"config does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    return payload


def _project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("joint training requires CUDA")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} is not a visible CUDA device")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", device_id=torch.device("cuda", local_rank)
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("distributed environment disagrees with process group")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_config(config: Mapping[str, Any], context: DistributedContext) -> None:
    for section in ("data", "stereo", "vggt", "fusion", "loss", "train"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"config requires mapping section {section!r}")
    crop = config["data"].get("crop_size_hw")
    if not isinstance(crop, list) or len(crop) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in crop
    ):
        raise ValueError("data.crop_size_hw must be [positive height, positive width]")
    if crop[0] % 64 or crop[1] % 64:
        raise ValueError("T8 joint crop dimensions must be divisible by 64")
    if int(config["data"].get("clip_length", 0)) < 2:
        raise ValueError("data.clip_length must be at least two")
    if str(config["train"].get("precision", "")).lower() != "bf16":
        raise ValueError("the formal joint recipe requires BF16")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is unsupported on {torch.cuda.get_device_name(context.device)}")
    if bool(config["train"].get("fsdp", False)) and context.world_size == 1:
        # A one-process dry run deliberately exercises the unwrapped model.
        pass
    if context.world_size > 1 and not bool(config["train"].get("fsdp", False)):
        raise ValueError("multi-process joint training requires train.fsdp=true")


def _dataset(config: Mapping[str, Any], *, training: bool) -> RawStereoVideoClipDataset:
    data = config["data"]
    manifest_key = "train_manifest" if training else "validation_manifest"
    return RawStereoVideoClipDataset(
        _project_path(data[manifest_key], f"data.{manifest_key}"),
        clip_length=int(data["clip_length"]),
        crop_size_hw=tuple(int(value) for value in data["crop_size_hw"]),
        crop_mode=str(data["train_crop_mode"] if training else data["validation_crop_mode"]),
        crop_alignment=int(data["crop_alignment"]),
        target_mode="last",
        seed=int(config["seed"]) + (0 if training else 10_000),
    )


def _loader(
    dataset: RawStereoVideoClipDataset,
    config: Mapping[str, Any],
    context: DistributedContext,
    *,
    training: bool,
) -> tuple[DataLoader[dict[str, Any]], DistributedSampler[Any] | None]:
    # Use DistributedSampler even for one rank.  Besides keeping the ordering
    # identical between diagnostic and FSDP runs, this makes an epoch cursor
    # restartable: ``set_epoch`` deterministically regenerates the same
    # permutation before the saved batch offset is skipped.
    sampler: DistributedSampler[Any] = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=training,
        seed=int(config["seed"]) + (0 if training else 10_000),
        drop_last=training,
    )
    workers = int(config["data"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["micro_batch_size_per_rank"]),
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["data"]["pin_memory"]),
        persistent_workers=workers > 0,
        drop_last=training,
        collate_fn=collate_raw_stereo_video_samples,
    )
    if len(loader) == 0:
        raise ValueError("DataLoader has no batches")
    return loader, sampler


def build_model(config: Mapping[str, Any]) -> MetricStereoVideoSystem:
    stereo_config = config["stereo"]
    vggt_config = config["vggt"]
    fusion_config = config["fusion"]
    stereo_model = load_fast_foundation_stereo(
        _project_path(stereo_config["checkpoint"], "stereo.checkpoint"),
        _project_path(stereo_config["repo"], "stereo.repo"),
        iterations=int(stereo_config["iterations"]),
        max_disp=int(stereo_config["max_disp_lr_px"]),
        amp_dtype=torch.bfloat16,
    )
    stereo = TrainableFastFoundationStereo(
        stereo_model,
        iterations=int(stereo_config["iterations"]),
        max_disp=int(stereo_config["max_disp_lr_px"]),
        predict_right=bool(stereo_config["predict_right"]),
    )
    vggt_model = load_vggt_omega(
        _project_path(vggt_config["checkpoint"], "vggt.checkpoint"),
        _project_path(vggt_config["repo"], "vggt.repo"),
        enable_camera=False,
    )
    vggt = TrainableVGGTOmega(
        vggt_model, geometry_channels=int(vggt_config["geometry_channels"])
    )
    geometry = CausalMetricStereoVideoGeometry(
        stereo_feature_channels=int(stereo_config["feature_channels"]),
        vggt_feature_channels=int(vggt_config["geometry_channels"]),
        hidden_channels=int(fusion_config["hidden_channels"]),
        residual_blocks=int(fusion_config["residual_blocks"]),
        minimum_gauge_overlap=int(fusion_config["minimum_gauge_overlap"]),
        inverse_depth_residual_scale=float(
            fusion_config["inverse_depth_residual_scale"]
        ),
        activation_checkpointing=bool(config["train"]["activation_checkpointing"]),
        relative_depth_tolerance=float(fusion_config["relative_depth_tolerance"]),
        absolute_depth_tolerance_m=float(
            fusion_config["absolute_depth_tolerance_m"]
        ),
    )
    return MetricStereoVideoSystem(
        stereo,
        vggt,
        geometry,
        stereo_feature_level=int(stereo_config["feature_level"]),
        left_right_maximum_error_lr_px=float(
            stereo_config["left_right_maximum_error_lr_px"]
        ),
        left_right_confidence_temperature_lr_px=float(
            stereo_config["left_right_confidence_temperature_lr_px"]
        ),
    )


def _apply_vggt_activation_checkpointing(model: nn.Module) -> int:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    targets = [module for module in model.modules() if type(module).__name__ == "SelfAttentionBlock"]
    if not targets:
        raise RuntimeError("no VGGT SelfAttentionBlock modules found for checkpointing")
    target_ids = {id(module) for module in targets}
    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: id(module) in target_ids,
    )
    return len(targets)


def _wrap_distributed(
    model: MetricStereoVideoSystem,
    config: Mapping[str, Any],
    context: DistributedContext,
    *,
    no_fsdp: bool,
) -> nn.Module:
    checkpointing = bool(config["train"]["activation_checkpointing"])
    if checkpointing:
        _apply_vggt_activation_checkpointing(model)
    if context.world_size == 1 or no_fsdp:
        return model.to(context.device)

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    auto_wrap = partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: isinstance(module, CheckpointWrapper),
    )
    # FFS batch-norm running statistics are frozen for causal invariance, but
    # its affine parameters may use BF16. Ignoring batch norm here would make
    # FSDP override auto_wrap and create one extra wrapper per normalization.
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        # The root argument is a structured batch containing FP32 K/T and
        # FP64 timestamps. Recursively casting it to BF16 destroys the metric
        # gauge and may collapse adjacent provenance timestamps.
        cast_root_forward_inputs=False,
        _module_classes_to_ignore=(
            type(model.vggt_backbone.model.dense_head),
        ),
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap if checkpointing else None,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=context.device,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False,
    )


def _unwrapped(model: nn.Module) -> MetricStereoVideoSystem:
    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, MetricStereoVideoSystem):
        raise TypeError(f"unexpected distributed model wrapper: {type(candidate)!r}")
    return candidate


def _optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    root = _unwrapped(model)
    train = config["train"]
    groups_with_names = (
        (
            "fusion",
            list(root.geometry_model.parameters())
            + list(root.vggt_backbone.geometry_projection.parameters()),
            float(train["learning_rate_fusion"]),
        ),
        (
            "stereo",
            list(root.stereo_backbone.parameters()),
            float(train["learning_rate_stereo"]),
        ),
        (
            "vggt_dense",
            list(root.vggt_backbone.model.dense_head.parameters()),
            float(train["learning_rate_vggt_dense"]),
        ),
        (
            "vggt_aggregator",
            list(root.vggt_backbone.model.aggregator.parameters()),
            float(train["learning_rate_vggt_aggregator"]),
        ),
    )
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    for name, parameters, learning_rate in groups_with_names:
        unique = [parameter for parameter in parameters if parameter.requires_grad and id(parameter) not in seen]
        seen.update(id(parameter) for parameter in unique)
        if not unique:
            raise RuntimeError(f"optimizer group {name!r} is empty")
        groups.append(
            {
                "params": unique,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "name": name,
            }
        )
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError(
            f"optimizer parameter partition mismatch: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    return torch.optim.AdamW(
        groups,
        weight_decay=float(train["weight_decay"]),
        fused=True,
    )


def _lr_multiplier(step: int, total_steps: int, warmup: int, minimum_ratio: float) -> float:
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    warmup: int,
    minimum_ratio: float,
) -> float:
    multiplier = _lr_multiplier(step, total_steps, warmup, minimum_ratio)
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * multiplier
    return multiplier


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _resize_bool(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.shape[-2:] == size:
        return value.to(dtype=torch.bool)
    return F.interpolate(value.float(), size=size, mode="nearest-exact").bool()


def _masked_resize_scalar(
    value: Tensor,
    valid: Tensor,
    size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Resize sparse scalar geometry without blending invalid zero sentinels."""

    if value.shape != valid.shape or value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("masked scalar resize expects matching [B,1,H,W] tensors")
    if valid.dtype != torch.bool:
        raise TypeError("masked scalar resize validity must be bool")
    finite_valid = valid & torch.isfinite(value)
    safe = torch.where(finite_valid, value.float(), torch.zeros_like(value).float())
    if value.shape[-2:] == size:
        return safe, finite_valid
    numerator = F.interpolate(
        safe * finite_valid.float(), size=size, mode="bilinear", align_corners=False
    )
    support = F.interpolate(
        finite_valid.float(), size=size, mode="bilinear", align_corners=False
    )
    resized_valid = support > 1e-6
    resized = torch.where(
        resized_valid,
        numerator / support.clamp_min(1e-6),
        torch.zeros_like(numerator),
    )
    return resized, resized_valid


def _target_temporal_warp(batch: Mapping[str, Any]) -> Any:
    previous_disparity = batch["previous_disparity_gt_left_px"].float()
    previous_valid = batch["previous_valid_gt_left"].bool()
    previous_disparity = torch.where(
        previous_valid & torch.isfinite(previous_disparity) & (previous_disparity > 0),
        previous_disparity,
        torch.zeros_like(previous_disparity),
    )
    intrinsics_previous = batch["K"][:, -2, 0].float()
    intrinsics_current = batch["K"][:, -1, 0].float()
    baseline_previous = batch["baseline_m"][:, -2].float()
    baseline_current = batch["baseline_m"][:, -1].float()
    numerator_previous = (
        intrinsics_previous[:, 0, 0] * baseline_previous
    ).reshape(-1, 1, 1, 1)
    previous_depth = torch.where(
        previous_valid,
        numerator_previous / previous_disparity.clamp_min(1e-8),
        torch.zeros_like(previous_disparity),
    )
    identity = torch.eye(
        4, device=previous_disparity.device, dtype=torch.float32
    ).expand(previous_disparity.shape[0], -1, -1)
    return zbuffer_reproject(
        previous_disparity,
        previous_depth,
        previous_valid.float(),
        intrinsics_previous,
        identity,
        batch["T_current_from_previous"][:, -1].float(),
        intrinsics_current_hr_3x3=intrinsics_current,
        baseline_previous_m=baseline_previous,
        baseline_current_m=baseline_current,
    )


def build_training_loss(
    output: MetricStereoVideoSystemOutput,
    batch: Mapping[str, Any],
    loss_module: MetricStereoVideoLoss,
    *,
    aligned_vggt_weight: float,
) -> TrainingLoss:
    """Assemble every valid first-run objective from one endpoint prediction."""

    left_rgb = batch["rgb"][:, -1, 0]
    right_rgb = batch["rgb"][:, -1, 1]
    target_disparity = batch["disparity_gt_left_px"][:, -1]
    target_disparity_right = batch["disparity_gt_right_px"][:, -1]
    target_valid = batch["valid_gt_left"][:, -1]
    target_valid_right = batch["valid_gt_right"][:, -1]
    intrinsics_current = batch["K"][:, -1, 0]
    intrinsics_previous = batch["K"][:, -2, 0]
    fx_px = intrinsics_current[:, 0, 0]
    baseline_current = batch["baseline_m"][:, -1]
    metric_factor = (fx_px * baseline_current).reshape(-1, 1, 1, 1)
    target_inverse = torch.where(
        target_valid & torch.isfinite(target_disparity) & (target_disparity > 0),
        target_disparity.float() / metric_factor,
        torch.zeros_like(target_disparity, dtype=torch.float32),
    )

    predictions = output.as_loss_predictions(include_lowres_auxiliary=False)
    iterative_inverse = tuple(
        prediction[:, -1].float() / metric_factor
        for prediction in output.ffs_iteration_disparities_left_hr_px_lr_grid
    )
    predictions["inverse_depth_pyramid_m_inv"] = (
        *iterative_inverse,
        output.endpoint.state.inverse_depth_m_inv,
    )

    gauge_size = output.gauge.inverse_depth_m_inv.shape[-2:]
    relative_vggt = F.interpolate(
        output.vggt_inverse_depth_relative_endpoint.float(),
        size=gauge_size,
        mode="bilinear",
        align_corners=False,
    )
    relative_confidence = F.interpolate(
        output.vggt_confidence_endpoint.float(),
        size=gauge_size,
        mode="bilinear",
        align_corners=False,
    )
    target_inverse_gauge, target_valid_gauge = _masked_resize_scalar(
        target_inverse, target_valid, gauge_size
    )
    target_gauge = align_vggt_inverse_depth_to_metric_stereo(
        relative_vggt,
        target_inverse_gauge,
        relative_confidence=relative_confidence,
        metric_confidence=target_valid_gauge.float(),
        relative_valid_mask=(
            torch.isfinite(relative_vggt)
            & (relative_vggt > 0)
            & (relative_confidence > 0)
        ),
        metric_valid_mask=target_valid_gauge,
        minimum_overlap=64,
    )
    predictions["log_scale"] = output.gauge.scale_m_inv_per_relative_unit.clamp_min(1e-8).log()

    stereo_reprojection = stereo_reproject_right_to_left(
        left_rgb, right_rgb, output.disparity_left_px
    )
    target_stereo_consistency = left_right_stereo_consistency(
        target_disparity.float(),
        target_disparity_right.float(),
        maximum_error_px=1.0,
        confidence_temperature_px=0.5,
    )
    previous_disparity = batch["previous_disparity_gt_left_px"].float()
    previous_valid = batch["previous_valid_gt_left"].bool()
    factor_previous = (
        intrinsics_previous[:, 0, 0] * batch["baseline_m"][:, -2]
    ).reshape(-1, 1, 1, 1)
    previous_inverse = torch.where(
        previous_valid & torch.isfinite(previous_disparity) & (previous_disparity > 0),
        previous_disparity / factor_previous,
        torch.zeros_like(previous_disparity),
    )
    temporal_reprojection = temporal_reproject_previous_to_current(
        left_rgb,
        batch["rgb"][:, -2, 0],
        output.inverse_depth_m_inv,
        intrinsics_current,
        intrinsics_previous,
        batch["T_current_from_previous"][:, -1],
        previous_inverse_depth_m_inv=None,
    )
    with torch.no_grad():
        target_temporal_reprojection = temporal_reproject_previous_to_current(
            left_rgb,
            batch["rgb"][:, -2, 0],
            target_inverse,
            intrinsics_current,
            intrinsics_previous,
            batch["T_current_from_previous"][:, -1],
            previous_inverse_depth_m_inv=previous_inverse,
        )
    target_warp = _target_temporal_warp(batch)
    warped_target_inverse = torch.where(
        target_warp.valid_mask,
        target_warp.depth_m.clamp_min(1e-8).reciprocal(),
        torch.zeros_like(target_warp.depth_m),
    )
    warped_prediction_inverse, zbuffer_temporal_valid = _masked_resize_scalar(
        output.endpoint.temporal.warped_inverse_depth_pre_consistency_m_inv.float(),
        output.endpoint.temporal.zbuffer_visible_mask,
        target_disparity.shape[-2:],
    )
    previous_available = batch["previous_disparity_gt_available"].reshape(-1, 1, 1, 1)
    dynamic_available = batch["dynamic_mask_available"].reshape(-1, 1, 1, 1)
    temporal_sample_available = previous_available & dynamic_available
    dynamic_mask = batch["dynamic_mask_current"].bool() | ~dynamic_available
    temporal_visibility = (
        target_warp.valid_mask
        & zbuffer_temporal_valid
        & temporal_sample_available
    )
    temporal_occlusion = (
        target_warp.collision_mask | target_temporal_reprojection.occlusion_mask
    )

    targets = {
        "disparity_left_px": target_disparity,
        "disparity_right_px": target_disparity_right,
        "valid": target_valid,
        "valid_right": target_valid_right,
        "left_rgb": left_rgb,
        "log_scale": target_gauge.scale_m_inv_per_relative_unit.detach().clamp_min(1e-8).log(),
    }
    geometry = {
        "fx_px": fx_px,
        "baseline_m": baseline_current,
        "temporal_warped_inverse_depth_m_inv": warped_prediction_inverse,
        "temporal_warped_target_inverse_depth_m_inv": warped_target_inverse,
        "stereo_reprojected_left_rgb": stereo_reprojection.image,
        "temporal_reprojected_left_rgb": temporal_reprojection.image,
    }
    masks = {
        "temporal_visibility": temporal_visibility,
        "temporal_dynamic": dynamic_mask,
        "temporal_occlusion": temporal_occlusion,
        "temporal_valid": target_valid & target_warp.valid_mask,
        "temporal_geometry_consistent": target_warp.valid_mask,
        "temporal_confidence": F.interpolate(
            output.endpoint.temporal.warped_confidence_pre_consistency.float(),
            size=target_disparity.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ),
        "stereo_reprojection_valid": (
            stereo_reprojection.valid_mask
            & target_valid
            & target_stereo_consistency.valid_left_mask
        ),
        "stereo_occlusion": target_valid & ~target_stereo_consistency.valid_left_mask,
        "stereo_reprojection_confidence": output.confidence.detach(),
        "temporal_reprojection_valid": (
            target_temporal_reprojection.valid_mask
            & target_valid
            & temporal_sample_available
        ),
        "temporal_reprojection_confidence": output.confidence.detach(),
        "left_right_valid": target_valid,
        "left_right_confidence": output.confidence.detach(),
        "validity_supervision": torch.ones_like(target_valid),
        "scale_valid": (
            output.gauge.valid_mask.reshape(-1, 1, 1, 1)
            & target_gauge.valid_mask.reshape(-1, 1, 1, 1)
        ),
    }
    breakdown = loss_module(predictions, targets, geometry, masks)

    aligned_vggt_grid_valid = (
        torch.isfinite(relative_vggt)
        & (relative_vggt > 0)
        & output.gauge.valid_mask.reshape(-1, 1, 1, 1)
    )
    aligned_vggt_hr, aligned_vggt_valid_hr = _masked_resize_scalar(
        output.gauge.inverse_depth_m_inv.float(),
        aligned_vggt_grid_valid,
        target_disparity.shape[-2:],
    )
    aligned_valid = target_valid & aligned_vggt_valid_hr
    aligned_vggt_auxiliary = normalized_log_depth_loss(
        aligned_vggt_hr,
        target_inverse,
        valid_mask=aligned_valid,
        confidence=output.confidence.detach(),
    )
    total = breakdown.total + float(aligned_vggt_weight) * aligned_vggt_auxiliary
    return TrainingLoss(
        total=total,
        breakdown=breakdown,
        aligned_vggt_depth_auxiliary=aligned_vggt_auxiliary,
        active_temporal_fraction=temporal_visibility.float().mean(),
        gauge_valid_fraction=output.gauge.valid_mask.float().mean(),
    )


def _loss_module(config: Mapping[str, Any]) -> MetricStereoVideoLoss:
    loss = config["loss"]
    return MetricStereoVideoLoss(
        MetricStereoVideoLossWeights(
            disparity=float(loss["disparity"]),
            depth=float(loss["depth"]),
            temporal=float(loss["temporal"]),
            reprojection=float(loss["reprojection"]),
            left_right_consistency=float(loss["left_right_consistency"]),
            pose_scale=float(loss["pose_scale"]),
            uncertainty=float(loss["uncertainty"]),
            validity=float(loss["validity"]),
        )
    )


def _infinite_batches(
    loader: DataLoader[dict[str, Any]],
    dataset: RawStereoVideoClipDataset,
    sampler: DistributedSampler[Any] | None,
    *,
    start_epoch: int = 0,
    start_batch: int = 0,
) -> Iterator[dict[str, Any]]:
    if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 0:
        raise ValueError("start_epoch must be a non-negative integer")
    if isinstance(start_batch, bool) or not isinstance(start_batch, int) or start_batch < 0:
        raise ValueError("start_batch must be a non-negative integer")
    epoch = start_epoch
    skip = start_batch
    while True:
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if skip and batch_index < skip:
                continue
            yield batch
        skip = 0
        epoch += 1


def _reduce_scalars(values: Mapping[str, Tensor], context: DistributedContext) -> dict[str, float]:
    names = tuple(values)
    stacked = torch.stack([values[name].detach().float().reshape(()) for name in names])
    if context.world_size > 1:
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        stacked /= context.world_size
    return {name: float(stacked[index].cpu()) for index, name in enumerate(names)}


def _all_ranks_finite(value: Tensor, context: DistributedContext) -> bool:
    """Return one agreed finite decision before entering a backward collective."""

    flag = torch.isfinite(value.detach()).all().to(
        device=context.device, dtype=torch.int32
    )
    if context.world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _debug_phase(context: DistributedContext, step: int, phase: str) -> None:
    if os.environ.get("METRIC_STEREO_DEBUG_PHASES") == "1":
        print(f"[rank={context.rank} step={step} phase={phase}]", flush=True)


def _capture_rank_rng_state(context: DistributedContext) -> dict[str, Any]:
    """Capture RNG streams needed to continue one rank's data/model path."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if context.device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(context.device)
    else:
        state["torch_cuda"] = None
    return state


def _restore_rank_rng_state(state: Any, context: DistributedContext) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointResumeError("checkpoint rng_state is not a mapping")
    required = ("python", "numpy", "torch_cpu", "torch_cuda")
    missing = [name for name in required if name not in state]
    if missing:
        raise CheckpointResumeError(f"checkpoint rng_state is missing {missing}")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_state = state["torch_cuda"]
        if cuda_state is not None:
            if context.device.type != "cuda" or not torch.cuda.is_available():
                raise CheckpointResumeError(
                    "checkpoint has CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state(cuda_state, context.device)
    except CheckpointResumeError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointResumeError(f"checkpoint rng_state is invalid: {exc}") from exc


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    context: DistributedContext,
    *,
    maximum_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(7, device=context.device, dtype=torch.float64)
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = _move_batch(cpu_batch, context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        target = batch["disparity_gt_left_px"][:, -1].float()
        valid = batch["valid_gt_left"][:, -1] & torch.isfinite(target) & (target > 0)
        prediction = output.disparity_left_px.float()
        error = (prediction - target).abs()
        baseline_lr = output.stereo.disparity_left_hr_px_lr_grid[:, -1]
        baseline = F.interpolate(
            baseline_lr.float(), target.shape[-2:], mode="bilinear", align_corners=False
        )
        baseline_error = (baseline - target).abs()
        count = valid.sum()
        totals[0] += torch.where(valid, error, torch.zeros_like(error)).sum().double()
        totals[1] += (valid & (error > 1.0)).sum().double()
        totals[2] += (valid & (error > 3.0)).sum().double()
        totals[3] += torch.where(valid, baseline_error, torch.zeros_like(error)).sum().double()
        totals[4] += count.double()
        totals[5] += output.valid_mask.float().sum().double()
        totals[6] += output.valid_mask.numel()
    if context.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    count = totals[4].clamp_min(1.0)
    return {
        "epe_px": float((totals[0] / count).cpu()),
        "bad1": float((totals[1] / count).cpu()),
        "bad3": float((totals[2] / count).cpu()),
        "stereo_prior_epe_px": float((totals[3] / count).cpu()),
        "valid_prediction_fraction": float((totals[5] / totals[6].clamp_min(1.0)).cpu()),
        "valid_pixels": int(totals[4].cpu()),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON manifest after all rank files are durable."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_dir_from_resume(path: Path) -> Path:
    """Resolve a user supplied checkpoint directory without guessing a step."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise CheckpointResumeError(f"resume checkpoint directory does not exist: {resolved}")
    if not (resolved / "manifest.json").is_file():
        raise CheckpointResumeError(
            f"resume path must contain manifest.json: {resolved}"
        )
    return resolved


def _validate_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointResumeError(f"checkpoint {name} must be a non-negative integer")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointResumeError(f"checkpoint {name} must be a positive integer")
    return int(value)


def _read_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    context: DistributedContext,
    expected_config: Mapping[str, Any],
) -> tuple[dict[str, Any], int, list[str], str]:
    """Read and validate manifest-level invariants (no tensor materialization)."""

    try:
        payload = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointResumeError(f"cannot read checkpoint manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointResumeError("checkpoint manifest is not a JSON object")
    schema = payload.get("schema_version")
    if schema not in {1, CHECKPOINT_SCHEMA_VERSION}:
        raise CheckpointResumeError(
            f"checkpoint manifest schema mismatch: expected 1 or {CHECKPOINT_SCHEMA_VERSION}, got {schema!r}"
        )
    if schema == CHECKPOINT_SCHEMA_VERSION and payload.get("kind") != _CHECKPOINT_KIND:
        raise CheckpointResumeError("checkpoint manifest kind mismatch")
    if schema == CHECKPOINT_SCHEMA_VERSION and payload.get("complete") is not True:
        raise CheckpointResumeError("checkpoint manifest is not complete")
    step = _validate_nonnegative_int(payload.get("step"), "step")
    match = _STEP_DIRECTORY_RE.fullmatch(checkpoint_dir.name)
    if match is None or int(match.group("step")) != step:
        raise CheckpointResumeError(
            "checkpoint directory name and manifest step disagree"
        )
    world_size = _validate_positive_int(payload.get("world_size"), "world_size")
    if world_size != context.world_size:
        raise CheckpointResumeError(
            f"checkpoint world size {world_size} does not match current world size {context.world_size}"
        )
    rank_files = payload.get("rank_files")
    if not isinstance(rank_files, list) or len(rank_files) != world_size:
        raise CheckpointResumeError("checkpoint manifest rank_files has wrong length")
    expected_names = [f"rank_{rank:04d}.pt" for rank in range(world_size)]
    if rank_files != expected_names:
        raise CheckpointResumeError(
            f"checkpoint manifest rank_files mismatch: expected {expected_names!r}, got {rank_files!r}"
        )
    for name in rank_files:
        if not isinstance(name, str) or _RANK_FILE_RE.fullmatch(name) is None:
            raise CheckpointResumeError(f"invalid checkpoint rank file name: {name!r}")
        if Path(name).name != name:
            raise CheckpointResumeError("checkpoint rank file escapes its directory")

    expected_fingerprint = _config_fingerprint(expected_config)
    manifest_fingerprint = payload.get("config_fingerprint")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        if not isinstance(manifest_fingerprint, str) or len(manifest_fingerprint) != 64:
            raise CheckpointResumeError("checkpoint manifest config_fingerprint is missing or malformed")
        if manifest_fingerprint != expected_fingerprint:
            raise CheckpointResumeError("resolved training config differs from checkpoint contract")
        digest_map = payload.get("rank_file_sha256")
        if not isinstance(digest_map, Mapping) or set(digest_map) != set(expected_names):
            raise CheckpointResumeError(
                "checkpoint manifest rank_file_sha256 must cover every rank file"
            )
        if payload.get("input_sha256") != _input_fingerprints(expected_config):
            raise CheckpointResumeError("training manifest contents differ from checkpoint")
        if payload.get("runtime_source_sha256") != _runtime_source_fingerprints():
            raise CheckpointResumeError("runtime source contents differ from checkpoint")
    elif manifest_fingerprint is not None and manifest_fingerprint != expected_fingerprint:
        raise CheckpointResumeError("resolved training config differs from checkpoint contract")
    return payload, step, expected_names, expected_fingerprint


def _validate_checkpoint_rank_file(
    checkpoint_dir: Path,
    filename: str,
    *,
    rank: int,
    world_size: int,
    step: int,
    expected_fingerprint: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one rank payload before any FSDP state-dict collective."""

    path = checkpoint_dir / filename
    if not path.is_file():
        raise CheckpointResumeError(f"checkpoint rank file is missing: {path}")
    digest_map = manifest.get("rank_file_sha256")
    if digest_map is not None:
        if not isinstance(digest_map, Mapping):
            raise CheckpointResumeError("checkpoint rank_file_sha256 is malformed")
        expected_digest = digest_map.get(filename)
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise CheckpointResumeError(f"checkpoint digest missing for {filename}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise CheckpointResumeError(f"checkpoint rank file SHA-256 mismatch: {filename}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch deserialization errors vary by version
        raise CheckpointResumeError(f"cannot load checkpoint rank file {filename}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointResumeError(f"checkpoint rank file {filename} is not a mapping")
    schema = payload.get("schema_version")
    if schema not in {1, CHECKPOINT_SCHEMA_VERSION}:
        raise CheckpointResumeError(f"checkpoint rank schema mismatch in {filename}")
    if schema == CHECKPOINT_SCHEMA_VERSION and payload.get("kind") != _CHECKPOINT_KIND:
        raise CheckpointResumeError(f"checkpoint rank kind mismatch in {filename}")
    if schema != manifest.get("schema_version"):
        raise CheckpointResumeError(
            f"checkpoint manifest/rank schema mismatch in {filename}"
        )
    if _validate_nonnegative_int(payload.get("step"), "step") != step:
        raise CheckpointResumeError(f"checkpoint rank step mismatch in {filename}")
    if _validate_positive_int(payload.get("world_size"), "world_size") != world_size:
        raise CheckpointResumeError(f"checkpoint rank world size mismatch in {filename}")
    encoded_rank = payload.get("rank")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        if isinstance(encoded_rank, bool) or not isinstance(encoded_rank, int) or encoded_rank != rank:
            raise CheckpointResumeError(f"checkpoint rank identity mismatch in {filename}")
        if payload.get("config_fingerprint") != expected_fingerprint:
            raise CheckpointResumeError(f"checkpoint rank config mismatch in {filename}")
        for key in ("model", "optimizer", "config", "rng_state", "data_cursor"):
            if key not in payload:
                raise CheckpointResumeError(f"checkpoint rank {filename} is missing {key}")
    else:
        # v1 had no explicit rank/fingerprint/RNG fields.  Its embedded config
        # is still checked against the current semantic contract below.
        if encoded_rank is not None and encoded_rank != rank:
            raise CheckpointResumeError(f"checkpoint rank identity mismatch in {filename}")
        if _config_fingerprint(payload.get("config", {})) != expected_fingerprint:
            raise CheckpointResumeError(f"checkpoint rank config mismatch in {filename}")
        for key in ("model", "optimizer"):
            if key not in payload:
                raise CheckpointResumeError(f"checkpoint rank {filename} is missing {key}")
    if not isinstance(payload.get("model"), Mapping) or not isinstance(payload.get("optimizer"), Mapping):
        raise CheckpointResumeError(f"checkpoint rank {filename} has malformed state dictionaries")
    return dict(payload)


def _distributed_error_consensus(error: str | None, context: DistributedContext) -> str | None:
    """Exchange small validation errors so one rank cannot deadlock the others."""

    if context.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return error
    errors: list[str | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(errors, error)
    messages = [message for message in errors if message]
    return messages[0] if messages else None


def _load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    resume: Path,
    config: Mapping[str, Any],
    context: DistributedContext,
    accumulation: int,
    batches_per_epoch: int,
) -> ResumeState:
    """Restore model, optimizer and cursor from a same-world-size checkpoint."""

    checkpoint_dir: Path | None = None
    payload: dict[str, Any] | None = None
    state: ResumeState | None = None
    local_error: str | None = None
    try:
        checkpoint_dir = _checkpoint_dir_from_resume(resume)
        manifest, step, rank_files, fingerprint = _read_checkpoint_manifest(
            checkpoint_dir, context=context, expected_config=config
        )
        payload = _validate_checkpoint_rank_file(
            checkpoint_dir,
            rank_files[context.rank],
            rank=context.rank,
            world_size=context.world_size,
            step=step,
            expected_fingerprint=fingerprint,
            manifest=manifest,
        )
        cursor = payload.get("data_cursor")
        if cursor is not None:
            epoch, batch_in_epoch = _validate_cursor(cursor, step=step)
            recorded_batches = cursor.get("batches_per_epoch") if isinstance(cursor, Mapping) else None
            recorded_accumulation = cursor.get("accumulation") if isinstance(cursor, Mapping) else None
            if recorded_batches is None or recorded_batches != batches_per_epoch:
                raise CheckpointResumeError(
                    "checkpoint data loader length differs from current loader"
                )
            if recorded_accumulation is None or recorded_accumulation != accumulation:
                raise CheckpointResumeError(
                    "checkpoint gradient accumulation differs from current run"
                )
            expected_cursor = _checkpoint_cursor(
                step, accumulation=accumulation, batches_per_epoch=batches_per_epoch
            )
            if (epoch, batch_in_epoch) != (
                expected_cursor["epoch"], expected_cursor["batch_in_epoch"]
            ):
                raise CheckpointResumeError("checkpoint data_cursor disagrees with step")
        else:
            cursor = _checkpoint_cursor(
                step, accumulation=accumulation, batches_per_epoch=batches_per_epoch
            )
            epoch, batch_in_epoch = int(cursor["epoch"]), int(cursor["batch_in_epoch"])
        state = ResumeState(step=step, epoch=epoch, batch_in_epoch=batch_in_epoch)
    except (CheckpointResumeError, OSError, ValueError, TypeError, RuntimeError) as exc:
        local_error = str(exc)
    consensus_error = _distributed_error_consensus(local_error, context)
    if consensus_error is not None:
        raise CheckpointResumeError(consensus_error)
    if payload is None or state is None or checkpoint_dir is None:  # pragma: no cover - defensive
        raise CheckpointResumeError("checkpoint validation produced no state")

    try:
        if context.world_size > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardedOptimStateDictConfig,
                ShardedStateDictConfig,
                StateDictType,
            )

            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
                ShardedOptimStateDictConfig(offload_to_cpu=True),
            ):
                model.load_state_dict(payload["model"], strict=True)
                optimizer_state = FSDP.optim_state_dict_to_load(
                    model, optimizer, dict(payload["optimizer"])
                )
                optimizer.load_state_dict(optimizer_state)
        else:
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
    except (KeyError, RuntimeError, ValueError, TypeError) as exc:
        raise CheckpointResumeError(f"checkpoint state is incompatible: {exc}") from exc

    if int(payload.get("schema_version", 1)) >= CHECKPOINT_SCHEMA_VERSION:
        _restore_rank_rng_state(payload.get("rng_state"), context)
    return state


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    directories: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and _STEP_DIRECTORY_RE.fullmatch(child.name):
            directories.append(child)
    return sorted(directories, key=lambda path: int(path.name.split("_")[1]))


def _prune_checkpoints(output_dir: Path, keep_last: int) -> list[Path]:
    """Delete only complete, valid-looking old checkpoint directories."""

    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
        raise ValueError("train.keep_last_checkpoints must be a positive integer")
    directories = _checkpoint_directories(output_dir)
    complete: list[Path] = []
    for directory in directories:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(manifest, Mapping) and manifest.get("complete") is True:
            complete.append(directory)
    victims = complete[:-keep_last]
    for directory in victims:
        # The directory name has already been strictly matched above; this
        # prevents an accidental broad recursive deletion if the helper is
        # called with a malformed output tree.
        if _STEP_DIRECTORY_RE.fullmatch(directory.name) is None:
            continue
        shutil.rmtree(directory)
    return victims


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    output_dir: Path,
    config: Mapping[str, Any],
    context: DistributedContext,
    accumulation: int = 1,
    batches_per_epoch: int | None = None,
) -> Path:
    """Save one atomic same-world-size rank bundle and retain recent bundles."""

    step = _validate_nonnegative_int(step, "step")
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if context.world_size > 1:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
    cursor = None
    if batches_per_epoch is not None:
        cursor = _checkpoint_cursor(
            step,
            accumulation=accumulation,
            batches_per_epoch=batches_per_epoch,
        )
    resolved_config = json.loads(_canonical_json(dict(config)))
    fingerprint = _config_fingerprint(config)
    rank_payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": _CHECKPOINT_KIND,
        "step": step,
        "world_size": context.world_size,
        "rank": context.rank,
        "model": model_state,
        "optimizer": optimizer_state,
        "config": resolved_config,
        "config_fingerprint": fingerprint,
        "input_sha256": _input_fingerprints(config),
        "runtime_source_sha256": _runtime_source_fingerprints(),
        "rng_state": _capture_rank_rng_state(context),
        "data_cursor": cursor,
    }
    rank_path = checkpoint_dir / f"rank_{context.rank:04d}.pt"
    temporary = rank_path.with_name(f".{rank_path.name}.{os.getpid()}.tmp")
    torch.save(rank_payload, temporary)
    os.replace(temporary, rank_path)
    if context.world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if context.primary:
        rank_files = [f"rank_{rank:04d}.pt" for rank in range(context.world_size)]
        missing = [name for name in rank_files if not (checkpoint_dir / name).is_file()]
        if missing:
            raise CheckpointResumeError(
                f"cannot publish checkpoint manifest; rank files missing: {missing}"
            )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": _CHECKPOINT_KIND,
            "complete": True,
            "step": step,
            "world_size": context.world_size,
            "rank_files": rank_files,
            "rank_file_sha256": {
                name: _sha256(checkpoint_dir / name) for name in rank_files
            },
            "config_fingerprint": fingerprint,
            "input_sha256": _input_fingerprints(config),
            "runtime_source_sha256": _runtime_source_fingerprints(),
            "created_unix_s": time.time(),
        }
        _atomic_write_json(checkpoint_dir / "manifest.json", manifest)
        _prune_checkpoints(
            output_dir,
            int(config.get("train", {}).get("keep_last_checkpoints", 1)),
        )
    if context.world_size > 1 and dist.is_available() and dist.is_initialized():
        # Ensure all ranks observe a complete manifest before the next update.
        dist.barrier()
    return checkpoint_dir


def _write_receipt(
    output_dir: Path,
    config: Mapping[str, Any],
    context: DistributedContext,
    *,
    checkpointed_blocks: int,
) -> None:
    if not context.primary:
        return
    data = config["data"]
    stereo = config["stereo"]
    vggt = config["vggt"]
    paths = {
        "train_manifest": _project_path(data["train_manifest"], "data.train_manifest"),
        "validation_manifest": _project_path(data["validation_manifest"], "data.validation_manifest"),
        "stereo_checkpoint": _project_path(stereo["checkpoint"], "stereo.checkpoint"),
        "vggt_checkpoint": _project_path(vggt["checkpoint"], "vggt.checkpoint"),
    }
    receipt = {
        "schema_version": 1,
        "experiment": config.get("experiment"),
        "created_unix_s": time.time(),
        "world_size": context.world_size,
        "gpu": torch.cuda.get_device_name(context.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported(),
        "checkpointed_attention_blocks": checkpointed_blocks,
        "causal_contract": "VGGT receives only past-to-current prefix; output is endpoint; memory is forward causal",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "runtime_source_sha256": _runtime_source_fingerprints(),
        "config": dict(config),
    }
    (output_dir / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    config = _read_config(args.config)
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        config["train"]["steps"] = args.steps
    if args.output_dir is not None:
        config["train"]["output_dir"] = str(args.output_dir)
    context = _distributed_context()
    _validate_config(config, context)
    seed = int(config["seed"])
    # Identical initialization is required because every rank loads independently.
    _seed_everything(seed)

    output_dir = _project_path(config["train"]["output_dir"], "train.output_dir")
    if context.primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if context.world_size > 1:
        dist.barrier()

    train_dataset = _dataset(config, training=True)
    validation_dataset = _dataset(config, training=False)
    train_loader, train_sampler = _loader(
        train_dataset, config, context, training=True
    )
    validation_loader, validation_sampler = _loader(
        validation_dataset, config, context, training=False
    )
    if validation_sampler is not None:
        validation_sampler.set_epoch(0)

    raw_model = build_model(config)
    checkpointed_blocks = sum(
        type(module).__name__ == "SelfAttentionBlock" for module in raw_model.modules()
    ) if bool(config["train"]["activation_checkpointing"]) else 0
    model = _wrap_distributed(raw_model, config, context, no_fsdp=args.no_fsdp)
    optimizer = _optimizer(model, config)
    loss_module = _loss_module(config).to(context.device)
    accumulation = int(config["train"]["gradient_accumulation"])
    if accumulation <= 0:
        raise ValueError("train.gradient_accumulation must be positive")
    if args.dry_run and args.resume is not None:
        raise ValueError("--dry-run cannot be combined with --resume")

    # A resume is deliberately performed only after the complete model has
    # been wrapped in the same FSDP topology and the optimizer has been built.
    # This makes sharded state-dict conversion identical to the save path and
    # lets every rank validate its own file before entering a collective.
    resume_state: ResumeState | None = None
    if args.resume is not None:
        resume_state = _load_checkpoint(
            model,
            optimizer,
            resume=args.resume,
            config=config,
            context=context,
            accumulation=accumulation,
            batches_per_epoch=len(train_loader),
        )
        if resume_state.step >= int(config["train"]["steps"]):
            raise ValueError(
                "checkpoint already reached the configured train.steps schedule horizon"
            )

    _write_receipt(
        output_dir,
        config,
        context,
        checkpointed_blocks=checkpointed_blocks,
    )

    configured_steps = int(config["train"]["steps"])
    start_step = 0 if resume_state is None else resume_state.step
    total_steps = 1 if args.dry_run else configured_steps - start_step
    batches = _infinite_batches(
        train_loader,
        train_dataset,
        train_sampler,
        start_epoch=0 if resume_state is None else resume_state.epoch,
        start_batch=0 if resume_state is None else resume_state.batch_in_epoch,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    log_path = output_dir / ("dry_run.jsonl" if args.dry_run else "train.jsonl")
    log_handle = log_path.open("a", encoding="utf-8") if context.primary else nullcontext()
    started = time.perf_counter()
    try:
        with log_handle as handle:
            for step in range(total_steps):
                step_started = time.perf_counter()
                global_step = start_step + step
                multiplier = _set_learning_rates(
                    optimizer,
                    step=global_step,
                    total_steps=configured_steps,
                    warmup=int(config["train"]["warmup_steps"]),
                    minimum_ratio=float(config["train"]["minimum_learning_rate_ratio"]),
                )
                last_loss: TrainingLoss | None = None
                for micro_step in range(accumulation):
                    _debug_phase(context, global_step + 1, "batch_begin")
                    batch = _move_batch(next(batches), context.device)
                    _debug_phase(context, global_step + 1, "batch_ready")
                    synchronize = micro_step == accumulation - 1
                    sync_context = (
                        nullcontext()
                        if synchronize or not hasattr(model, "no_sync")
                        else model.no_sync()
                    )
                    with sync_context:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(batch)
                        _debug_phase(context, global_step + 1, "forward_done")
                        last_loss = build_training_loss(
                            output,
                            batch,
                            loss_module,
                            aligned_vggt_weight=float(
                                config["loss"]["aligned_vggt_depth_auxiliary"]
                            ),
                        )
                        scaled_loss = last_loss.total / accumulation
                    _debug_phase(context, global_step + 1, "loss_done")
                    if not _all_ranks_finite(scaled_loss, context):
                        raise FloatingPointError(f"non-finite loss at step {global_step + 1}")
                    _debug_phase(context, global_step + 1, "loss_finite")
                    scaled_loss.backward()
                    _debug_phase(context, global_step + 1, "backward_done")
                assert last_loss is not None
                if hasattr(model, "clip_grad_norm_"):
                    gradient_norm = model.clip_grad_norm_(
                        float(config["train"]["gradient_clip_norm"])
                    )
                else:
                    gradient_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["train"]["gradient_clip_norm"])
                    )
                _debug_phase(context, global_step + 1, "grad_norm_done")
                if not _all_ranks_finite(gradient_norm, context):
                    raise FloatingPointError(f"non-finite gradient norm at step {global_step + 1}")
                optimizer.step()
                _debug_phase(context, global_step + 1, "optimizer_done")
                optimizer.zero_grad(set_to_none=True)
                if context.world_size > 1:
                    dist.barrier()
                _debug_phase(context, global_step + 1, "barrier_done")
                elapsed = time.perf_counter() - step_started
                terms = {
                    name: getattr(last_loss.breakdown, name)
                    for name in (
                        "total",
                        "disparity",
                        "depth",
                        "temporal",
                        "reprojection",
                        "left_right_consistency",
                        "pose_scale",
                        "uncertainty",
                        "validity",
                    )
                }
                terms.update(
                    {
                        "aligned_vggt_depth_auxiliary": last_loss.aligned_vggt_depth_auxiliary,
                        "joint_total": last_loss.total,
                        "gradient_norm": gradient_norm,
                        "active_temporal_fraction": last_loss.active_temporal_fraction,
                        "gauge_valid_fraction": last_loss.gauge_valid_fraction,
                    }
                )
                reduced = _reduce_scalars(terms, context)
                _debug_phase(context, global_step + 1, "reduce_done")
                completed = global_step + 1
                if context.primary and completed % int(config["train"]["log_interval"]) == 0:
                    record = {
                        "step": completed,
                        "elapsed_s": elapsed,
                        "steps_per_second": 1.0 / max(elapsed, 1e-9),
                        "lr_multiplier": multiplier,
                        "learning_rates": {
                            str(group["name"]): float(group["lr"])
                            for group in optimizer.param_groups
                        },
                        "peak_memory_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
                        "loss": reduced,
                        "active_terms": list(last_loss.breakdown.active_terms),
                    }
                    assert handle is not None
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)

                validation_interval = int(config["train"]["validation_interval"])
                if not args.dry_run and validation_interval > 0 and completed % validation_interval == 0:
                    metrics = _validate(
                        model,
                        validation_loader,
                        context,
                        maximum_batches=int(config["train"]["validation_batches"]),
                    )
                    if context.primary:
                        validation_record = {"step": completed, "validation": metrics}
                        with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as validation_handle:
                            validation_handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
                        print(json.dumps(validation_record, sort_keys=True), flush=True)

                checkpoint_interval = int(config["train"]["checkpoint_interval"])
                if not args.dry_run and checkpoint_interval > 0 and completed % checkpoint_interval == 0:
                    _save_checkpoint(
                        model,
                        optimizer,
                        step=completed,
                        output_dir=output_dir,
                        config=config,
                        context=context,
                        accumulation=accumulation,
                        batches_per_epoch=len(train_loader),
                    )
    finally:
        if context.world_size > 1:
            dist.destroy_process_group()
    if context.primary:
        summary = {
            "status": "DRY_RUN_PASS" if args.dry_run else "COMPLETE",
            "steps": total_steps,
            "elapsed_s": time.perf_counter() - started,
            "peak_memory_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
        }
        (output_dir / ("dry_run_summary.json" if args.dry_run else "run_summary.json")).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
