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
)
from geometry.history_confidence import history_confidence  # noqa: E402
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
        "observation_cache_identity": None,
        "teacher_cache_identity": None,
        "derived_cache_lineage": None,
        "crop_mode": "random",
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
    if not bool(config.model.use_history) or not bool(config.model.use_vggt_pose):
        raise ValueError("Stage B requires model.use_history and model.use_vggt_pose")
    if bool(config.model.epipolar_refinement):
        raise ValueError("HR epipolar refinement belongs to Stage C")
    if str(config.train.init_from_stage).lower() != "spatial":
        raise ValueError("Stage B must initialize from the spatial stage")
    if not bool(config.train.history_detach):
        raise ValueError("Stage B MVP requires detached disparity history")
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
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    dataset = CachedFFSTrainingDataset(
        manifest_path=manifest_path,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
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
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    dataset = CachedTemporalTrainingDataset(
        manifest_path=manifest_path,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        derived_cache_root=derived_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
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
    if not positivity_ablation.enabled:
        return baseline
    return _with_positivity_penalty(baseline, output, positivity_ablation)


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


def _sample_hr_winner_grid_to_lr(value: Tensor, *, scale: int) -> Tensor:
    """Select HR coordinates ``(s*v,s*u)`` matching ``K_lr=K_hr/s``.

    This is nearest winner selection, never area averaging.  It preserves
    disparity in HR-pixel units and fractional offsets in HR-pixel units.
    """

    if value.ndim != 4 or value.shape[-2] % scale or value.shape[-1] % scale:
        raise ValueError("HR transport tensor must be [B,C,sH,sW]")
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


def compute_stage_b_step_loss(
    output: ModelOutput,
    batch: Mapping[str, Any],
    *,
    transport: TemporalTransport | None,
    scale: int = 2,
    weights: LossWeights = LossWeights(),
    max_photometric_residual: float = 0.10,
    positivity_ablation: PositivityAblation = PositivityAblation(),
) -> LossBreakdown:
    """Compute spatial supervision plus visibility-gated temporal consistency."""

    spatial = compute_stage_a_loss(
        output,
        batch,
        scale=scale,
        weights=weights,
        positivity_ablation=positivity_ablation,
    )
    if transport is None:
        temporal = output.disparity_hr_px.sum() * 0.0
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
    if not positivity_ablation.enabled:
        return baseline
    if spatial.positivity_penalty is None:
        raise RuntimeError("enabled positivity ablation produced no penalty")
    return replace(
        baseline,
        total=baseline.total + spatial.positivity_penalty,
        positivity_penalty=spatial.positivity_penalty,
    )


def average_loss_breakdowns(
    values: Sequence[LossBreakdown],
) -> LossBreakdown:
    if not values:
        raise ValueError("cannot average an empty loss sequence")

    def mean(name: str) -> Tensor:
        return torch.stack([getattr(value, name) for value in values]).mean()

    positivity_values = [value.positivity_penalty for value in values]
    if any(value is None for value in positivity_values):
        if not all(value is None for value in positivity_values):
            raise ValueError("cannot mix baseline and positivity-ablation loss breakdowns")
        positivity_penalty = None
    else:
        positivity_penalty = torch.stack(
            [value for value in positivity_values if value is not None]
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
        positivity_penalty=positivity_penalty,
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

    rgb_sequence = batch["rgb_hr_sequence"]
    if rgb_sequence.ndim != 5 or rgb_sequence.shape[1] != 3:
        raise ValueError("temporal RGB batch must have shape [B,3,3,H,W]")
    hidden_state: Sequence[Tensor] | None = None
    previous_output: ModelOutput | None = None
    previous_rgb_hr: Tensor | None = None
    losses: list[LossBreakdown] = []
    for time_index in range(3):
        step = _temporal_step_batch(batch, time_index)
        pose_valid = batch["temporal_pose_valid_sequence"][:, time_index]
        transport: TemporalTransport | None = None
        if time_index > 0:
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
                temporal_extrinsics_camera_from_world=batch[
                    "vggt_extrinsics_camera_from_world_metric_sequence"
                ][:, time_index],
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
        valid_vggt = batch["valid_vggt_sequence"][:, time_index] & static_prior.reshape(
            -1, 1, 1, 1
        )
        model_kwargs: dict[str, Any] = {
            "disparity_vggt_hr_px": batch["disparity_vggt_hr_px_sequence"][:, time_index],
            "confidence_vggt": batch["confidence_vggt_sequence"][:, time_index],
            "valid_vggt": valid_vggt,
            "valid_ffs": step["valid_ffs"],
            "hidden_state": hidden_state,
        }
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
                scale=int(config.data.scale),
                weights=weights,
                max_photometric_residual=float(
                    config.train.temporal_photometric_threshold
                ),
                positivity_ablation=positivity_ablation,
            )
        )
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
    diagnostic: bool = False,
) -> LossBreakdown:
    output = model(
        batch["rgb_hr"],
        batch["disparity_ffs_hr_px"],
        batch["confidence_ffs"],
        valid_ffs=batch["valid_ffs"],
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
    initialization_lineage: Mapping[str, Any] | None = None
    if stage == "temporal" and args.resume is None:
        assert initialization_path is not None
        initialization_lineage = load_model_initialization_checkpoint(
            initialization_path,
            model=model,
            expected_parameter_count=parameter_count,
            required_sequence_length=1,
        )
        if (
            initialization_lineage["checkpoint_sha256"]
            != str(config.train.initialization_checkpoint_sha256)
        ):
            raise ValueError("Stage-A checkpoint changed while the run was being built")
    weights = loss_weights_from_config(config)
    positivity_ablation = positivity_ablation_from_config(config)
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
