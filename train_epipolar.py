#!/usr/bin/env python3
"""Train only the Stage-C HR epipolar refiner over a frozen Stage-B base."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import CacheIdentity, sha256_file  # noqa: E402
from data.epipolar_training_dataset import (  # noqa: E402
    EpipolarTrainingDataset,
    collate_epipolar_training_samples,
)
from data.temporal_training_dataset import CachedTemporalTrainingDataset  # noqa: E402
from evaluation import (  # noqa: E402
    load_model_for_evaluation,
    validate_checkpoint_lineage,
)
from eval import (  # noqa: E402
    _validate_formal_temporal_coverage,
    _validated_raw_vggt_receipt,
)
from geometry.epipolar import EPIPOLAR_GEOMETRY_CONTRACT  # noqa: E402
from models.epipolar_refiner import HREpipolarRefiner  # noqa: E402
from models.epipolar_stage import (  # noqa: E402
    EpipolarStageLoss,
    FrozenTemporalEpipolarStage,
    compute_epipolar_stage_loss,
)
from models.ffs_omega_tsr import FFSOmegaTSR, ModelOutput  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_CONFIG,
    DeterministicEpochSampler,
    _load_yaml_with_defaults,
    _reset_hidden_where_pose_invalid,
    _temporal_step_batch,
    build_model,
    build_temporal_transport,
    learning_rate_multiplier,
    load_receipt_identity,
)
from utils.checkpoint import (  # noqa: E402
    CheckpointMismatchError,
    atomic_torch_save,
    capture_rng_state,
    config_fingerprint,
    repository_git_hash,
    restore_rng_state,
)
from utils.seed import seed_data_worker, seed_everything  # noqa: E402


STAGE_C_DEFAULTS: dict[str, Any] = {
    "data": {
        "epipolar_rectification_audit_path": None,
        "epipolar_rectification_audit": None,
    },
    "model": {
        "epipolar_offsets_hr_px": [-2, -1, 0, 1, 2],
        "epipolar_feature_channels": 32,
        "epipolar_correlation_groups": 8,
        "epipolar_head_channels": 48,
        "epipolar_correction_limit_hr_px": 2.0,
        "epipolar_confidence_temperature": 1.0,
        "epipolar_vertical_geometry": EPIPOLAR_GEOMETRY_CONTRACT["version"],
    },
    "train": {
        "steps_epipolar": 5000,
        "correction_regularizer_weight": 0.01,
        "epipolar_output_dir": None,
    },
}

PSEUDO_GT_SUPERVISION = {
    "type": "trusted_hr_ffs_teacher_pseudo_gt",
    "purpose": "engineering supervision for Stage-C integration",
    "paper_ground_truth": False,
    "paper_accuracy_claim": False,
}

STAGE_C_RUNTIME_GIT_SCOPES = (
    "train_epipolar.py",
    "train.py",
    "eval.py",
    "configs/epipolar_x2.yaml",
    "configs/temporal_x2.yaml",
    "configs/mvp_x2.yaml",
    "pyproject.toml",
    "src",
)

STAGE_C_CHECKPOINT_SCHEMA_VERSION = 1
STAGE_C_COMPONENT = "ffs-omega-tsr-epipolar-stage-c"
STAGE_C_MODEL_COMPONENT = "hr_epipolar_refiner"
FORMAL_STAGE_B_STEPS = 15_000
FORMAL_STAGE_C_STEPS = 5_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage-C HR epipolar refinement with a frozen T3 base."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Stage-B checkpoint (required for a new run)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume exactly from a Stage-C latest checkpoint",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--observation-cache-root", type=Path)
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--derived-cache-root", type=Path)
    parser.add_argument("--rectification-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-steps", type=int, help="bounded optimizer steps")
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="allow only a dry-run or one-step CPU integration smoke",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one deterministic forward/loss without optimizer mutation",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser


def resolve_epipolar_config(
    config_path: str | Path, overrides: Sequence[str] = ()
) -> DictConfig:
    """Resolve inherited project YAML plus strict Stage-C-only defaults."""

    config = OmegaConf.merge(
        OmegaConf.create(copy.deepcopy(DEFAULT_CONFIG)),
        OmegaConf.create(STAGE_C_DEFAULTS),
        _load_yaml_with_defaults(Path(config_path)),
    )
    OmegaConf.set_struct(config, True)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_epipolar_config(config: DictConfig) -> None:
    """Reject any Stage-C setting that changes the frozen T3 base contract."""

    if str(config.train.stage).lower() != "epipolar":
        raise ValueError("Stage-C config requires train.stage=epipolar")
    if str(config.train.init_from_stage).lower() != "temporal":
        raise ValueError("Stage C must initialize from a temporal checkpoint")
    if int(config.data.sequence_length) != 3:
        raise ValueError("Stage C requires causal sequence_length=3")
    if int(config.data.vggt_context_pairs) != 5 or not bool(config.vggt.causal):
        raise ValueError("Stage C requires causal five-pair VGGT geometry")
    if int(config.data.scale) != 2 or int(config.model.convex_scale) != 2:
        raise ValueError("first-round Stage C is fixed to x2")
    if not bool(config.model.use_history) or not bool(config.model.use_vggt_pose):
        raise ValueError("Stage C requires the trained history/pose base path")
    if not bool(config.model.epipolar_refinement):
        raise ValueError("Stage-C config must enable epipolar_refinement")
    if (
        str(config.model.epipolar_vertical_geometry)
        != EPIPOLAR_GEOMETRY_CONTRACT["version"]
    ):
        raise ValueError("Stage-C vertical epipolar geometry contract mismatch")
    if bool(config.train.compile_model):
        raise ValueError("torch.compile remains disabled for initial Stage C")
    if str(config.train.precision).lower() != "bf16":
        raise ValueError("formal Stage C requires train.precision=bf16")
    if str(config.train.optimizer).lower() != "adamw":
        raise ValueError("Stage-C optimizer must be AdamW")
    offsets = [float(value) for value in config.model.epipolar_offsets_hr_px]
    if (
        not offsets
        or any(not math.isfinite(value) or abs(value) > 2.0 for value in offsets)
        or any(right <= left for left, right in zip(offsets, offsets[1:]))
        or 0.0 not in offsets
        or min(offsets) != -2.0
        or max(offsets) != 2.0
    ):
        raise ValueError(
            "model.epipolar_offsets_hr_px must be strictly increasing, include "
            "zero, and span exactly [-2,+2]"
        )
    correction_limit = float(config.model.epipolar_correction_limit_hr_px)
    if correction_limit != 2.0:
        raise ValueError("Stage-C correction limit must be exactly +/-2 HR pixels")
    _positive_int(config.train.steps_epipolar, "train.steps_epipolar")
    _nonnegative_int(config.train.warmup_steps, "train.warmup_steps")
    _positive_int(config.train.checkpoint_interval, "train.checkpoint_interval")
    _positive_int(config.train.log_interval, "train.log_interval")
    _nonnegative_int(config.train.num_workers, "train.num_workers")
    micro_batch = _positive_int(
        config.train.micro_batch_size, "train.micro_batch_size"
    )
    accumulation = _positive_int(
        config.train.grad_accumulation, "train.grad_accumulation"
    )
    effective_batch = _positive_int(
        config.train.effective_batch_size, "train.effective_batch_size"
    )
    if micro_batch * accumulation != effective_batch:
        raise ValueError(
            "effective batch size must equal micro_batch_size * grad_accumulation"
        )
    regularizer_weight = float(config.train.correction_regularizer_weight)
    if not math.isfinite(regularizer_weight) or regularizer_weight < 0:
        raise ValueError("correction regularizer weight must be non-negative")
    for name in ("learning_rate", "gradient_clip"):
        value = float(config.train[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{name} must be finite and positive")
    weight_decay = float(config.train.weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("train.weight_decay must be finite and non-negative")
    crop = list(config.data.hr_crop)
    if len(crop) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in crop
    ):
        raise ValueError("data.hr_crop must contain two positive integers")
    if any(value % int(config.data.scale) for value in crop):
        raise ValueError("data.hr_crop must be divisible by the spatial scale")
    if str(config.data.crop_mode) not in {"random", "fixed"}:
        raise ValueError("data.crop_mode must be random or fixed")


def _required_path(config: DictConfig, dotted_name: str, *, directory: bool) -> Path:
    value = OmegaConf.select(config, dotted_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{dotted_name} is required")
    path = Path(str(value)).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise FileNotFoundError(f"{dotted_name} does not exist: {path}")
    return path


def _resolved_dict(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def _runtime_source_bundle() -> dict[str, Any]:
    """Hash the committed Stage-C runtime and reject scoped dirty state."""

    command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *STAGE_C_RUNTIME_GIT_SCOPES,
    ]
    try:
        status = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot audit Stage-C runtime Git state: {exc}") from exc
    dirty = status.stdout.strip()
    if dirty:
        raise RuntimeError(
            "Stage-C runtime source paths must be committed and clean:\n" + dirty
        )
    git_head = repository_git_hash(PROJECT_ROOT)
    if git_head == "unknown":
        raise RuntimeError("Stage-C runtime source requires a Git commit identity")
    files = [
        PROJECT_ROOT / "train_epipolar.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "eval.py",
        PROJECT_ROOT / "configs" / "epipolar_x2.yaml",
        PROJECT_ROOT / "configs" / "temporal_x2.yaml",
        PROJECT_ROOT / "configs" / "mvp_x2.yaml",
        PROJECT_ROOT / "pyproject.toml",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]
    file_records = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    encoded = json.dumps(
        {"git_head": git_head, "files": file_records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "git_head": git_head,
        "relevant_paths_clean": True,
        "git_scopes": list(STAGE_C_RUNTIME_GIT_SCOPES),
        "files": file_records,
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validated_rectification_audit(
    path: str | Path,
    *,
    expected_train_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and compact the required same-row pixel audit receipt."""

    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"epipolar rectification audit receipt is missing: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse epipolar rectification audit: {exc}") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("epipolar rectification audit must be a JSON object")
    if receipt.get("schema_version") != 1 or receipt.get("component") != (
        "pixel-level-epipolar-rectification-audit"
    ):
        raise ValueError("epipolar rectification audit schema/component mismatch")
    if receipt.get("status") != "PASS" or receipt.get("published_contract") != (
        EPIPOLAR_GEOMETRY_CONTRACT["version"]
    ):
        raise ValueError("epipolar rectification audit did not publish the required contract")
    manifests = receipt.get("manifests")
    if not isinstance(manifests, Mapping) or not isinstance(
        manifests.get("train"), Mapping
    ) or not isinstance(manifests.get("validation"), Mapping):
        raise ValueError("epipolar rectification audit manifest binding is missing")
    if manifests["train"].get("sha256") != expected_train_manifest_sha256:
        raise ValueError("epipolar rectification audit train manifest SHA mismatch")
    if manifests.get("train_validation_sequence_disjoint") is not True:
        raise ValueError("epipolar rectification audit lacks train/validation isolation")
    threshold_checks = receipt.get("threshold_checks")
    if not isinstance(threshold_checks, list) or not threshold_checks or any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in threshold_checks
    ):
        raise ValueError("epipolar rectification audit threshold checks did not all pass")
    global_result = receipt.get("global")
    metadata_vs_pixels = receipt.get("metadata_vs_pixels")
    for name, value in (
        ("algorithm", receipt.get("algorithm")),
        ("thresholds", receipt.get("thresholds")),
        ("sampling", receipt.get("sampling")),
        ("global", global_result),
        ("metadata_vs_pixels", metadata_vs_pixels),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"epipolar rectification audit {name} is missing")
    if metadata_vs_pixels.get("conclusion") != (
        "INCONSISTENT_WITH_AUDITED_PIXEL_COORDINATES"
    ):
        raise ValueError("epipolar metadata/pixel-coordinate diagnosis is missing")
    counts = global_result.get("counts")
    dy = global_result.get("dy_right_minus_left_px")
    absolute_dy = dy.get("absolute") if isinstance(dy, Mapping) else None
    signed_dy = dy.get("signed") if isinstance(dy, Mapping) else None
    if not all(isinstance(value, Mapping) for value in (counts, absolute_dy, signed_dy)):
        raise ValueError("epipolar rectification audit aggregate evidence is malformed")
    sampled_frames = global_result.get("sampled_frames")
    covered_frames = global_result.get("covered_frames")
    ratio_matches = counts.get("ratio_matches")
    ransac_inliers = counts.get("ransac_inliers")
    for name, value in (
        ("sampled_frames", sampled_frames),
        ("covered_frames", covered_frames),
        ("ratio_matches", ratio_matches),
        ("ransac_inliers", ransac_inliers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"epipolar rectification audit {name} is invalid")
    if covered_frames != sampled_frames:
        raise ValueError("epipolar rectification audit does not cover every sampled frame")
    pixel_values = {
        "coverage_fraction": global_result.get("coverage_fraction"),
        "median_right_y_minus_left_y_px": signed_dy.get("p50"),
        "p95_abs_right_y_minus_left_y_px": absolute_dy.get("p95"),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in pixel_values.values()
    ):
        raise ValueError("epipolar rectification audit pixel evidence is non-finite")
    sample_identity = receipt["sampling"].get("sample_identity_sha256")
    required_hashes = (
        manifests["train"].get("sha256"),
        manifests["validation"].get("sha256"),
        sample_identity,
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in required_hashes
    ):
        raise ValueError("epipolar rectification audit SHA-256 binding is malformed")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "schema_version": 1,
        "component": receipt["component"],
        "status": receipt["status"],
        "contract_version": receipt["published_contract"],
        "manifest_sha256": {
            "train": manifests["train"]["sha256"],
            "validation": manifests["validation"]["sha256"],
        },
        "algorithm": dict(receipt["algorithm"]),
        "thresholds": dict(receipt["thresholds"]),
        "counts": {
            "sampled_frames": sampled_frames,
            "covered_frames": covered_frames,
            "ratio_matches": ratio_matches,
            "ransac_inliers": ransac_inliers,
        },
        "pixel_evidence": pixel_values,
        "metadata_vs_pixels": dict(metadata_vs_pixels),
        "sample_identity_sha256": sample_identity,
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return device


def _training_runtime(device: torch.device, *, use_bf16: bool) -> dict[str, Any]:
    cuda_device = device.type == "cuda"
    if cuda_device:
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        concrete_device = torch.device("cuda", device_index)
        with torch.cuda.device(concrete_device):
            bf16_supported = bool(
                torch.cuda.is_bf16_supported(including_emulation=False)
            )
    else:
        concrete_device = device
        bf16_supported = False
    autocast_enabled = bool(use_bf16)
    return {
        "device": str(concrete_device),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(concrete_device) if cuda_device else None
        ),
        "device_capability": (
            list(torch.cuda.get_device_capability(concrete_device))
            if cuda_device
            else None
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bf16_supported,
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": "torch.bfloat16" if autocast_enabled else None,
        "formal_cuda_bf16_eligible": bool(
            cuda_device and bf16_supported and autocast_enabled
        ),
    }


def _validate_execution_mode(
    device: torch.device,
    *,
    allow_cpu_smoke: bool,
    dry_run: bool,
    run_steps: int | None,
    training_runtime: Mapping[str, Any],
) -> None:
    """Fail closed unless this is native CUDA bf16 or a bounded CPU smoke."""

    if device.type == "cpu":
        if not allow_cpu_smoke:
            raise RuntimeError(
                "CPU Stage-C execution requires --allow-cpu-smoke and is never "
                "formal-training eligible"
            )
        if not dry_run and run_steps != 1:
            raise RuntimeError("--allow-cpu-smoke is limited to dry-run or one step")
        return
    if device.type != "cuda":
        raise RuntimeError(f"unsupported Stage-C execution device: {device}")
    if not bool(training_runtime.get("formal_cuda_bf16_eligible", False)):
        raise RuntimeError(
            f"formal Stage C requires native CUDA bf16 on {training_runtime.get('device_name')}"
        )


def _resume_config_paths(path: str | Path) -> dict[str, str]:
    """Read only the immutable path bindings needed to reconstruct a resume."""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("component") != STAGE_C_COMPONENT:
        raise CheckpointMismatchError("resume file is not a Stage-C checkpoint")
    config = payload.get("config")
    data = config.get("data") if isinstance(config, Mapping) else None
    train = config.get("train") if isinstance(config, Mapping) else None
    base = payload.get("base_checkpoint")
    if not all(isinstance(value, Mapping) for value in (data, train, base)):
        raise CheckpointMismatchError("resume checkpoint path bindings are malformed")
    bindings: dict[str, object] = {
        "data.manifest_path": data.get("manifest_path"),
        "data.observation_cache_root": data.get("observation_cache_root"),
        "data.teacher_cache_root": data.get("teacher_cache_root"),
        "data.derived_geometry_cache_root": data.get(
            "derived_geometry_cache_root"
        ),
        "data.epipolar_rectification_audit_path": data.get(
            "epipolar_rectification_audit_path"
        ),
        "train.initialization_checkpoint": base.get("path"),
    }
    if any(not isinstance(value, str) or not value for value in bindings.values()):
        raise CheckpointMismatchError(
            "resume checkpoint lacks an immutable training path binding"
        )
    output_value = train.get("epipolar_output_dir")
    if output_value is not None:
        if not isinstance(output_value, str) or not output_value:
            raise CheckpointMismatchError(
                "resume checkpoint output directory binding is malformed"
            )
        bindings["train.epipolar_output_dir"] = output_value
    return {name: str(value) for name, value in bindings.items()}


def _validate_base_completion(
    base_metadata: Mapping[str, Any],
    *,
    expected_steps: int,
    required: bool,
) -> dict[str, Any]:
    """Require a completed canonical Stage-B base for an unbounded CUDA run."""

    expected_steps = _positive_int(expected_steps, "expected Stage-B steps")
    training_config = base_metadata.get("training_config")
    train = (
        training_config.get("train")
        if isinstance(training_config, Mapping)
        else None
    )
    actual_step = base_metadata.get("step")
    if isinstance(actual_step, bool) or not isinstance(actual_step, int):
        raise ValueError("Stage-B checkpoint step is malformed")
    configured_steps = train.get("steps") if isinstance(train, Mapping) else None
    declared_steps = (
        train.get("steps_temporal") if isinstance(train, Mapping) else None
    )
    complete = (
        actual_step == expected_steps
        and configured_steps == expected_steps
        and declared_steps == expected_steps
        and expected_steps == FORMAL_STAGE_B_STEPS
    )
    receipt = {
        "actual_step": actual_step,
        "configured_steps": configured_steps,
        "declared_temporal_steps": declared_steps,
        "required_steps": expected_steps,
        "canonical_required_steps": FORMAL_STAGE_B_STEPS,
        "complete": complete,
        "required_for_this_run": bool(required),
    }
    if required and not complete:
        raise ValueError(
            "formal Stage C requires a completed canonical Stage-B base: "
            f"{receipt}"
        )
    return receipt


def _data_cursor(
    *,
    completed_steps: int,
    accumulation: int,
    batches_per_epoch: int,
) -> dict[str, Any]:
    completed_steps = _nonnegative_int(completed_steps, "completed_steps")
    accumulation = _positive_int(accumulation, "accumulation")
    batches_per_epoch = _positive_int(batches_per_epoch, "batches_per_epoch")
    completed_micro_steps = completed_steps * accumulation
    epoch, batch_offset = divmod(completed_micro_steps, batches_per_epoch)
    return {
        "completed_micro_steps": completed_micro_steps,
        "batches_per_epoch": batches_per_epoch,
        "epoch": epoch,
        "batch_offset_in_epoch": batch_offset,
        "grad_accumulation": accumulation,
        "drop_last": True,
    }


def _validate_finite_training_state(
    refiner: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    for name, value in list(refiner.named_parameters()) + list(
        refiner.named_buffers()
    ):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite Stage-C refiner state: {name}")
    for parameter, state in optimizer.state.items():
        del parameter
        for name, value in state.items():
            if isinstance(value, Tensor) and value.is_floating_point() and not bool(
                torch.isfinite(value).all().item()
            ):
                raise FloatingPointError(
                    f"non-finite Stage-C optimizer state: {name}"
                )


def _stage_c_checkpoint_payload(
    *,
    stage: FrozenTemporalEpipolarStage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_steps: int,
    config: DictConfig,
    git_hash: str,
    runtime_source_bundle: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
    base_checkpoint: Mapping[str, Any],
    base_lineage: Mapping[str, Any],
    raw_lineage: Mapping[str, Any],
    base_completion: Mapping[str, Any],
    rectification_audit: Mapping[str, Any],
    latest_loss: Mapping[str, Any],
    elapsed_seconds: float,
    batches_per_epoch: int,
) -> dict[str, Any]:
    """Build one full-state, optimizer-boundary Stage-C checkpoint."""

    completed_steps = _nonnegative_int(completed_steps, "completed_steps")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    parameter_count = stage.trainable_parameter_count
    if parameter_count <= 0:
        raise ValueError("Stage-C refiner parameter count must be positive")
    _validate_finite_training_state(stage.refiner, optimizer)
    configured_steps = int(config.train.steps_epipolar)
    execution_complete = completed_steps == configured_steps
    canonical_schedule = configured_steps == FORMAL_STAGE_C_STEPS
    completion = {
        "actual_step": completed_steps,
        "configured_steps": configured_steps,
        "execution_complete": execution_complete,
        "canonical_schedule": canonical_schedule,
        "base_complete": bool(base_completion.get("complete", False)),
        "cuda_bf16_eligible": bool(
            training_runtime.get("formal_cuda_bf16_eligible", False)
        ),
    }
    completion["formal_training_complete"] = bool(
        execution_complete
        and canonical_schedule
        and completion["base_complete"]
        and completion["cuda_bf16_eligible"]
    )
    return {
        "schema_version": STAGE_C_CHECKPOINT_SCHEMA_VERSION,
        "component": STAGE_C_COMPONENT,
        "model": stage.refiner.state_dict(),
        "model_component": STAGE_C_MODEL_COMPONENT,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": {},
        "step": completed_steps,
        "config": _resolved_dict(config),
        "git_hash": git_hash,
        "rng_states": capture_rng_state(),
        "data_cursor": _data_cursor(
            completed_steps=completed_steps,
            accumulation=int(config.train.grad_accumulation),
            batches_per_epoch=batches_per_epoch,
        ),
        "base_checkpoint": dict(base_checkpoint),
        "base_lineage": dict(base_lineage),
        "raw_lineage": dict(raw_lineage),
        "base_completion": dict(base_completion),
        "geometry_contract": EPIPOLAR_GEOMETRY_CONTRACT,
        "rectification_audit": dict(rectification_audit),
        "runtime_source_bundle": dict(runtime_source_bundle),
        "training_runtime": dict(training_runtime),
        "supervision": PSEUDO_GT_SUPERVISION,
        "parameter_count": parameter_count,
        "trainable_refiner_parameter_count": parameter_count,
        "loss": dict(latest_loss),
        "elapsed_seconds": float(elapsed_seconds),
        "completion": completion,
    }


def _load_stage_c_training_checkpoint(
    path: str | Path,
    *,
    stage: FrozenTemporalEpipolarStage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_config: Mapping[str, Any],
    expected_git_hash: str,
    expected_runtime_source_bundle: Mapping[str, Any],
    expected_training_runtime: Mapping[str, Any],
    expected_base_checkpoint: Mapping[str, Any],
    batches_per_epoch: int,
) -> tuple[int, float, dict[str, Any]]:
    """Validate and restore an exact optimizer-boundary Stage-C checkpoint."""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("Stage-C resume payload is not a mapping")
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
        "data_cursor",
        "base_checkpoint",
        "runtime_source_bundle",
        "training_runtime",
        "parameter_count",
        "trainable_refiner_parameter_count",
        "elapsed_seconds",
        "loss",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointMismatchError(
            f"Stage-C resume fields are missing: {missing}"
        )
    if payload["schema_version"] != STAGE_C_CHECKPOINT_SCHEMA_VERSION or (
        payload["component"] != STAGE_C_COMPONENT
        or payload["model_component"] != STAGE_C_MODEL_COMPONENT
    ):
        raise CheckpointMismatchError("Stage-C resume component/schema mismatch")
    parameter_count = stage.trainable_parameter_count
    if payload["parameter_count"] != parameter_count or payload[
        "trainable_refiner_parameter_count"
    ] != parameter_count:
        raise CheckpointMismatchError("Stage-C resume parameter count mismatch")
    saved_config = payload["config"]
    if not isinstance(saved_config, Mapping) or config_fingerprint(
        saved_config
    ) != config_fingerprint(expected_config):
        raise CheckpointMismatchError(
            "resolved Stage-C config differs from the resume checkpoint"
        )
    if payload["git_hash"] != expected_git_hash or payload[
        "runtime_source_bundle"
    ] != dict(expected_runtime_source_bundle):
        raise CheckpointMismatchError("Stage-C runtime source differs on resume")
    if payload["training_runtime"] != dict(expected_training_runtime):
        raise CheckpointMismatchError("Stage-C CUDA/BF16 runtime differs on resume")
    saved_base = payload["base_checkpoint"]
    if not isinstance(saved_base, Mapping) or dict(saved_base) != dict(
        expected_base_checkpoint
    ):
        raise CheckpointMismatchError("Stage-C frozen base differs on resume")
    if payload["scaler"] != {}:
        raise CheckpointMismatchError("Stage-C BF16 resume scaler must be empty")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointMismatchError("Stage-C resume step is malformed")
    expected_cursor = _data_cursor(
        completed_steps=step,
        accumulation=int(expected_config["train"]["grad_accumulation"]),
        batches_per_epoch=batches_per_epoch,
    )
    if payload["data_cursor"] != expected_cursor:
        raise CheckpointMismatchError("Stage-C resume data cursor is inconsistent")
    try:
        stage.refiner.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointMismatchError(
            f"Stage-C resume training state is incompatible: {exc}"
        ) from exc
    if scheduler.last_epoch != step or getattr(scheduler, "_step_count", None) != (
        step + 1
    ):
        raise CheckpointMismatchError("Stage-C scheduler step differs on resume")
    trainable_parameters = list(stage.refiner.parameters())
    if len(optimizer.state) != len(trainable_parameters) or any(
        parameter not in optimizer.state for parameter in trainable_parameters
    ):
        raise CheckpointMismatchError(
            "Stage-C optimizer state does not cover the complete refiner"
        )
    for state in optimizer.state.values():
        optimizer_step = state.get("step")
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
                "Stage-C optimizer progress differs on resume"
            )
    elapsed_seconds = payload["elapsed_seconds"]
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0
    ):
        raise CheckpointMismatchError("Stage-C elapsed time is malformed")
    loss = payload["loss"]
    if not isinstance(loss, Mapping):
        raise CheckpointMismatchError("Stage-C resume loss summary is malformed")
    rng_states = payload["rng_states"]
    if not isinstance(rng_states, Mapping):
        raise CheckpointMismatchError("Stage-C resume RNG state is malformed")
    _validate_finite_training_state(stage.refiner, optimizer)
    restore_rng_state(rng_states)
    return step, float(elapsed_seconds), dict(loss)


def _write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _reconcile_training_log(
    path: str | Path, *, completed_step: int, log_interval: int
) -> int:
    """Validate a resume log and atomically discard rows ahead of checkpoint."""

    destination = Path(path)
    completed_step = _nonnegative_int(completed_step, "completed_step")
    log_interval = _positive_int(log_interval, "log_interval")
    if not destination.is_file():
        if completed_step == 0:
            return 0
        raise CheckpointMismatchError(
            "Stage-C resume checkpoint has no matching train.jsonl"
        )
    try:
        raw_text = destination.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointMismatchError(f"cannot read Stage-C train.jsonl: {exc}") from exc
    lines = raw_text.splitlines()
    terminal_newline = not raw_text or raw_text.endswith("\n")
    nonempty = [line for line in lines if line.strip()]
    rows: list[Mapping[str, Any]] = []
    torn_tail = False
    for line_index, line in enumerate(nonempty):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_index == len(nonempty) - 1:
                torn_tail = True
                break
            raise CheckpointMismatchError(
                f"cannot parse Stage-C train.jsonl line {line_index + 1}: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise CheckpointMismatchError("Stage-C train.jsonl row is not an object")
        rows.append(row)
    expected_steps = list(
        range(log_interval, log_interval * (len(rows) + 1), log_interval)
    )
    actual_steps = [row.get("step") for row in rows]
    if actual_steps != expected_steps:
        raise CheckpointMismatchError(
            "Stage-C train.jsonl steps are not a strict monotonic interval sequence"
        )
    retained = [row for row in rows if int(row["step"]) <= completed_step]
    expected_retained = completed_step // log_interval
    if len(retained) != expected_retained:
        raise CheckpointMismatchError(
            "Stage-C train.jsonl does not cover the resume checkpoint"
        )
    # A crash can occur after a valid JSON object is written but before its
    # delimiter.  Rewriting that otherwise-valid row is necessary: opening the
    # log in append mode would concatenate the next object onto the same line.
    if torn_tail or not terminal_newline or len(retained) != len(rows):
        _write_jsonl_atomic(destination, retained)
    return len(retained)


def _write_jsonl_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def predict_frozen_stage_b_endpoint(
    base_model: nn.Module,
    batch: Mapping[str, Any],
    *,
    config: DictConfig,
) -> Tensor:
    """Return the exact Stage-B VGGT-on endpoint disparity ``[B,1,H,W]``.

    The unroll matches training: three causal calls, hidden reset on rejected
    pose, current endpoint-derived static prior, and detached z-buffer history.
    Callers should invoke this through :class:`FrozenTemporalEpipolarStage`,
    which supplies the unconditional no-grad boundary.
    """

    if not isinstance(base_model, FFSOmegaTSR):
        raise TypeError("base_model must be FFSOmegaTSR")
    rgb_sequence = batch.get("rgb_hr_sequence")
    if not isinstance(rgb_sequence, Tensor) or rgb_sequence.ndim != 5 or (
        rgb_sequence.shape[1:3] != (3, 3)
    ):
        raise ValueError("temporal RGB must have shape [B,3,3,H,W]")
    hidden_state: Sequence[Tensor] | None = None
    previous_output: ModelOutput | None = None
    previous_rgb_hr: Tensor | None = None
    output: ModelOutput | None = None
    for time_index in range(3):
        step = _temporal_step_batch(batch, time_index)
        pose_valid = batch["temporal_pose_valid_sequence"][:, time_index]
        transport = None
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
        kwargs: dict[str, Any] = {
            "disparity_vggt_hr_px": batch["disparity_vggt_hr_px_sequence"][
                :, time_index
            ],
            "confidence_vggt": batch["confidence_vggt_sequence"][:, time_index],
            "valid_vggt": valid_vggt,
            "valid_ffs": step["valid_ffs"],
            "hidden_state": hidden_state,
        }
        if transport is not None:
            kwargs.update(
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
        output = base_model(
            step["rgb_hr"],
            step["disparity_ffs_hr_px"],
            step["confidence_ffs"],
            **kwargs,
        )
        hidden_state = output.hidden_state
        previous_output = output
        previous_rgb_hr = step["rgb_hr"]
    assert output is not None
    return output.disparity_hr_px


def _build_temporal_dataset(
    config: DictConfig,
) -> tuple[EpipolarTrainingDataset, CacheIdentity, CacheIdentity]:
    manifest = _required_path(config, "data.manifest_path", directory=False)
    observation = _required_path(
        config, "data.observation_cache_root", directory=True
    )
    teacher = _required_path(config, "data.teacher_cache_root", directory=True)
    derived = _required_path(
        config, "data.derived_geometry_cache_root", directory=True
    )
    observation_identity = load_receipt_identity(
        observation,
        expected_component="ffs-observation",
        manifest_path=manifest,
    )
    teacher_identity = load_receipt_identity(
        teacher,
        expected_component="ffs-teacher",
        manifest_path=manifest,
    )
    crop_height, crop_width = (int(value) for value in config.data.hr_crop)
    temporal = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        crop_size_hr_hw=(crop_height, crop_width),
        crop_mode=str(config.data.crop_mode),
        spatial_scale=2,
        student_sequence_length=3,
        vggt_context_pairs=5,
        seed=int(config.seed),
    )
    _validate_formal_temporal_coverage(temporal)
    return EpipolarTrainingDataset(temporal), observation_identity, teacher_identity


def _validate_base_data_lineage(
    checkpoint_metadata: Mapping[str, Any],
    dataset: EpipolarTrainingDataset,
) -> dict[str, Any]:
    config = checkpoint_metadata.get("training_config")
    data = config.get("data") if isinstance(config, Mapping) else None
    if not isinstance(data, Mapping):
        raise ValueError("Stage-B checkpoint data config is missing")
    temporal = dataset.temporal_dataset
    expected_paths = {
        "manifest_path": temporal.manifest_path,
        "observation_cache_root": temporal.observation_cache_root,
        "teacher_cache_root": temporal.spatial_dataset.teacher_cache_root,
        "derived_geometry_cache_root": temporal.derived_cache_root,
    }
    for name, expected in expected_paths.items():
        value = data.get(name)
        if not isinstance(value, str) or Path(value).expanduser().resolve() != expected:
            raise ValueError(f"Stage-C {name} differs from Stage-B training lineage")
    manifest_sha256 = sha256_file(temporal.manifest_path)
    raw_root = temporal.derived_cache_root.parent / "vggt"
    current_raw = _validated_raw_vggt_receipt(
        raw_root, expected_manifest_sha256=manifest_sha256
    )
    saved_derived = data.get("derived_cache_lineage")
    if not isinstance(saved_derived, Mapping) or not isinstance(
        saved_derived.get("derived_cache_root"), str
    ):
        raise ValueError("Stage-B derived-cache root lineage is missing")
    saved_raw_root = (
        Path(str(saved_derived["derived_cache_root"])).expanduser().resolve().parent
        / "vggt"
    )
    saved_raw = _validated_raw_vggt_receipt(
        saved_raw_root, expected_manifest_sha256=manifest_sha256
    )
    if current_raw["identity"] != saved_raw["identity"] or (
        current_raw["config"] != saved_raw["config"]
    ):
        raise ValueError("Stage-C raw VGGT identity/config differs from Stage B")
    endpoint_indices = {
        int(window.student_indices[-1]) for window in temporal.windows
    }
    ffs_artifacts, right_source_digest = _ffs_artifact_lineage(
        temporal,
        endpoint_indices=endpoint_indices,
    )
    return {
        "manifest_sha256": manifest_sha256,
        "raw_vggt_identity": current_raw["identity"],
        "raw_vggt_receipt_sha256": current_raw["receipt_sha256"],
        "derived_cache_lineage": dict(temporal.cache_lineage_summary),
        "ffs_cache_artifacts": ffs_artifacts,
        "endpoint_right_source_digest": right_source_digest,
    }


def _ffs_artifact_lineage(
    temporal: CachedTemporalTrainingDataset,
    *,
    endpoint_indices: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind FFS receipts/manifests and endpoint right-image identities."""

    artifacts: dict[str, Any] = {}
    observation_rows: list[dict[str, Any]] | None = None
    roots = {
        "observation": temporal.observation_cache_root,
        "teacher": temporal.spatial_dataset.teacher_cache_root,
    }
    for role, root in roots.items():
        receipt_path = root / "run_receipt.json"
        manifest_path = root / "cache_manifest.jsonl"
        if not receipt_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Stage-C {role} receipt/manifest is missing")
        try:
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse Stage-C {role} cache manifest: {exc}") from exc
        if len(rows) != len(temporal.records):
            raise ValueError(f"Stage-C {role} cache manifest coverage is incomplete")
        for index, (row, record) in enumerate(
            zip(rows, temporal.records, strict=True)
        ):
            if not isinstance(row, Mapping) or row.get("selection_index") != index:
                raise ValueError(f"Stage-C {role} cache selection order mismatch")
            source = row.get("source")
            if not isinstance(source, Mapping) or (
                source.get("manifest_record") != record.to_dict()
            ):
                raise ValueError(f"Stage-C {role} cache source record mismatch")
        artifacts[role] = {
            "root": str(root),
            "run_receipt_sha256": sha256_file(receipt_path),
            "cache_manifest_sha256": sha256_file(manifest_path),
            "records": len(rows),
        }
        if role == "observation":
            observation_rows = rows

    assert observation_rows is not None
    digest_rows: list[tuple[int, str, int, str]] = []
    for index in sorted(endpoint_indices):
        row = observation_rows[index]
        source = row["source"]
        right_sha256 = source.get("right_sha256")
        if not isinstance(right_sha256, str) or len(right_sha256) != 64:
            raise ValueError("Stage-C observation right source SHA-256 is malformed")
        try:
            int(right_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "Stage-C observation right source SHA-256 is malformed"
            ) from exc
        record = temporal.records[index]
        digest_rows.append(
            (index, record.sequence_id, record.frame_id, right_sha256)
        )
    encoded = json.dumps(
        digest_rows,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return artifacts, {
        "algorithm": "sha256(canonical_json([manifest_index,sequence_id,frame_id,right_sha256]))",
        "records": len(digest_rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _infinite_batches(
    loader: DataLoader[dict[str, Any]],
    dataset: EpipolarTrainingDataset,
    sampler: DeterministicEpochSampler,
    *,
    start_micro_step: int = 0,
) -> Iterator[dict[str, Any]]:
    start_micro_step = _nonnegative_int(start_micro_step, "start_micro_step")
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("Stage-C DataLoader has no complete micro-batches")
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
            if skip_batches:
                raise RuntimeError("Stage-C resume batch offset exhausted DataLoader")
            raise RuntimeError("Stage-C DataLoader yielded no batches")
        skip_batches = 0
        epoch += 1


def _stage_loss(
    stage: FrozenTemporalEpipolarStage,
    batch: Mapping[str, Any],
    config: DictConfig,
) -> EpipolarStageLoss:
    output = stage(batch)
    target_sequence = batch.get("teacher_disparity_hr_px_sequence")
    trusted_sequence = batch.get("teacher_trusted_mask_sequence")
    confidence_sequence = batch.get("teacher_confidence_sequence")
    if not all(
        isinstance(value, Tensor)
        for value in (target_sequence, trusted_sequence, confidence_sequence)
    ):
        raise ValueError("Stage C requires teacher disparity/confidence/trusted cache")
    return compute_epipolar_stage_loss(
        output,
        target_sequence[:, -1],
        trusted_sequence[:, -1],
        target_confidence=confidence_sequence[:, -1],
        correction_regularizer_weight=float(
            config.train.correction_regularizer_weight
        ),
    )


def run_one_epipolar_optimizer_step(
    stage: FrozenTemporalEpipolarStage,
    batch: Mapping[str, Any],
    config: DictConfig,
    optimizer: torch.optim.Optimizer,
) -> EpipolarStageLoss:
    """Run one deterministic optimizer step over refiner parameters only."""

    trainable = [
        parameter for parameter in stage.refiner.parameters() if parameter.requires_grad
    ]
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != {id(parameter) for parameter in trainable}:
        raise ValueError("optimizer must own exactly the trainable refiner parameters")
    if any(parameter.requires_grad for parameter in stage.base_model.parameters()):
        raise RuntimeError("Stage-B base must remain frozen")
    stage.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _stage_loss(stage, batch, config)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(
        trainable,
        float(config.train.gradient_clip),
        error_if_nonfinite=True,
    )
    optimizer.step()
    if any(parameter.grad is not None for parameter in stage.base_model.parameters()):
        raise RuntimeError("Stage-B base unexpectedly received gradients")
    return loss


def run(args: argparse.Namespace) -> int:
    if args.resume is not None and args.dry_run:
        raise ValueError("--resume and --dry-run cannot be combined")
    config = resolve_epipolar_config(args.config, args.overrides)
    resume_bindings = (
        _resume_config_paths(args.resume) if args.resume is not None else {}
    )
    if args.resume is None and args.init_from is None:
        raise ValueError("a new Stage-C run requires --init-from STAGE_B_CHECKPOINT")
    cli_paths = {
        "data.manifest_path": args.manifest,
        "data.observation_cache_root": args.observation_cache_root,
        "data.teacher_cache_root": args.teacher_cache_root,
        "data.derived_geometry_cache_root": args.derived_cache_root,
        "data.epipolar_rectification_audit_path": args.rectification_audit,
        "train.epipolar_output_dir": args.output,
        "train.initialization_checkpoint": args.init_from,
    }
    for name, value in cli_paths.items():
        selected: str | Path | None = value
        if selected is None and name in resume_bindings:
            selected = resume_bindings[name]
        if selected is not None:
            OmegaConf.update(
                config,
                name,
                str(Path(selected).expanduser().resolve()),
                merge=False,
            )
    runtime_source_bundle = _runtime_source_bundle()
    validate_epipolar_config(config)
    seed_everything(int(config.seed), deterministic=True)
    dataset, observation_identity, teacher_identity = _build_temporal_dataset(config)
    rectification_audit_path = _required_path(
        config, "data.epipolar_rectification_audit_path", directory=False
    )
    rectification_audit = _validated_rectification_audit(
        rectification_audit_path,
        expected_train_manifest_sha256=sha256_file(
            dataset.temporal_dataset.manifest_path
        ),
    )
    OmegaConf.update(
        config,
        "data.epipolar_rectification_audit",
        rectification_audit,
        merge=False,
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
        dict(dataset.cache_lineage_summary),
        merge=False,
    )
    device = _resolve_device(args.device)
    use_bf16 = device.type == "cuda"
    training_runtime = _training_runtime(device, use_bf16=use_bf16)
    _validate_execution_mode(
        device,
        allow_cpu_smoke=bool(args.allow_cpu_smoke),
        dry_run=bool(args.dry_run),
        run_steps=args.run_steps,
        training_runtime=training_runtime,
    )
    base_checkpoint_path = _required_path(
        config, "train.initialization_checkpoint", directory=False
    )
    base_model = build_model(config)
    base_parameter_count = base_model.trainable_parameter_count
    base_metadata = load_model_for_evaluation(
        base_checkpoint_path,
        base_model,
        expected_parameter_count=base_parameter_count,
        require_full_training_state=True,
    )
    base_lineage = validate_checkpoint_lineage(
        base_metadata,
        required_stage="temporal",
        observation_cache_identity=observation_identity.to_dict(),
        teacher_cache_identity=teacher_identity.to_dict(),
        derived_cache_lineage=dataset.cache_lineage_summary,
        evaluation_config=_resolved_dict(config),
    )
    raw_lineage = _validate_base_data_lineage(base_metadata, dataset)
    formal_training_requested = bool(
        not args.dry_run and args.run_steps is None and device.type == "cuda"
    )
    base_completion = _validate_base_completion(
        base_metadata,
        expected_steps=int(config.train.steps_temporal),
        required=formal_training_requested,
    )
    offsets = tuple(float(value) for value in config.model.epipolar_offsets_hr_px)
    refiner = HREpipolarRefiner(
        feature_channels=int(config.model.epipolar_feature_channels),
        correlation_groups=int(config.model.epipolar_correlation_groups),
        candidate_offsets_hr_px=offsets,
        correction_limit_hr_px=float(
            config.model.epipolar_correction_limit_hr_px
        ),
        confidence_temperature=float(
            config.model.epipolar_confidence_temperature
        ),
        head_channels=int(config.model.epipolar_head_channels),
    )
    stage = FrozenTemporalEpipolarStage(
        base_model,
        refiner,
        lambda module, batch: predict_frozen_stage_b_endpoint(
            module, batch, config=config
        ),
    ).to(device)
    if stage.trainable_parameter_count != refiner.trainable_parameter_count:
        raise RuntimeError("Stage C has trainable parameters outside the refiner")
    if any(parameter.requires_grad for parameter in stage.base_model.parameters()):
        raise RuntimeError("Stage-B base is not fully frozen")
    workers = int(config.train.num_workers)
    sampler = DeterministicEpochSampler(len(dataset), seed=int(config.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(config.train.micro_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=bool(config.train.persistent_workers) and workers > 0,
        pin_memory=bool(config.train.pin_memory) and device.type == "cuda",
        collate_fn=collate_epipolar_training_samples,
        worker_init_fn=seed_data_worker,
        generator=torch.Generator().manual_seed(int(config.seed)),
        drop_last=True,
    )
    if args.dry_run:
        batches = _infinite_batches(loader, dataset, sampler)
        first_batch = _move_batch(next(batches), device)
        stage.eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            dry_loss = _stage_loss(stage, first_batch, config)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "stage": "epipolar",
                    "device": str(device),
                    "dataset_windows": len(dataset),
                    "base_parameter_count": base_parameter_count,
                    "trainable_refiner_parameter_count": stage.trainable_parameter_count,
                    "base_checkpoint": base_metadata,
                    "base_lineage": base_lineage,
                    "raw_lineage": raw_lineage,
                    "base_completion": base_completion,
                    "geometry_contract": EPIPOLAR_GEOMETRY_CONTRACT,
                    "rectification_audit": rectification_audit,
                    "runtime_source_bundle": runtime_source_bundle,
                    "training_runtime": training_runtime,
                    "supervision": PSEUDO_GT_SUPERVISION,
                    "loss": dry_loss.detached_scalars(),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    output_value = config.train.epipolar_output_dir
    output_dir = (
        PROJECT_ROOT / "outputs" / str(config.experiment)
        if output_value is None
        else Path(str(output_value)).expanduser().resolve()
    )
    latest_path = output_dir / "latest.pt"
    final_path = output_dir / "final.pt"
    log_path = output_dir / "train.jsonl"
    summary_path = output_dir / "run_summary.json"
    tracked_outputs = (latest_path, final_path, log_path, summary_path)
    if args.resume is None:
        existing = [str(path) for path in tracked_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Stage-C output already contains run artifacts: " + ", ".join(existing)
            )
    elif final_path.exists():
        raise FileExistsError(
            f"completed Stage-C output cannot be resumed: {final_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        stage.refiner.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    configured_steps = int(config.train.steps_epipolar)
    warmup_steps = int(config.train.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update_index: learning_rate_multiplier(
            update_index,
            total_steps=configured_steps,
            warmup_steps=warmup_steps,
        ),
    )
    accumulation = int(config.train.grad_accumulation)
    resolved_config = _resolved_dict(config)
    repository_hash = str(runtime_source_bundle["git_head"])
    base_checkpoint = {
        "path": base_metadata["path"],
        "sha256": base_metadata["checkpoint_sha256"],
        "step": base_metadata["step"],
    }
    completed_steps = 0
    previous_elapsed_seconds = 0.0
    latest_loss_summary: dict[str, Any] = {}
    if args.resume is not None:
        completed_steps, previous_elapsed_seconds, latest_loss_summary = (
            _load_stage_c_training_checkpoint(
                args.resume,
                stage=stage,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_config=resolved_config,
                expected_git_hash=repository_hash,
                expected_runtime_source_bundle=runtime_source_bundle,
                expected_training_runtime=training_runtime,
                expected_base_checkpoint=base_checkpoint,
                batches_per_epoch=len(loader),
            )
        )
        _reconcile_training_log(
            log_path,
            completed_step=completed_steps,
            log_interval=int(config.train.log_interval),
        )

    if completed_steps >= configured_steps:
        raise ValueError(
            f"Stage-C checkpoint already completed {completed_steps}/{configured_steps} steps"
        )
    starting_step = completed_steps
    if args.run_steps is None:
        target_step = configured_steps
    else:
        requested_segment_steps = _positive_int(args.run_steps, "run_steps")
        target_step = completed_steps + requested_segment_steps
        if target_step > configured_steps:
            raise ValueError(
                "bounded Stage-C run exceeds the configured schedule: "
                f"target={target_step}, configured={configured_steps}"
            )

    batches = _infinite_batches(
        loader,
        dataset,
        sampler,
        start_micro_step=completed_steps * accumulation,
    )
    stage.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log_mode = "a" if args.resume is not None else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        while completed_steps < target_step:
            summed_total: Tensor | None = None
            summed_disparity: Tensor | None = None
            summed_regularizer: Tensor | None = None
            summed_valid_pixels = 0
            for _ in range(accumulation):
                batch = _move_batch(next(batches), device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    loss = _stage_loss(stage, batch, config)
                    scaled = loss.total / float(accumulation)
                scaled.backward()
                summed_total = (
                    loss.total.detach()
                    if summed_total is None
                    else summed_total + loss.total.detach()
                )
                summed_disparity = (
                    loss.disparity.detach()
                    if summed_disparity is None
                    else summed_disparity + loss.disparity.detach()
                )
                summed_regularizer = (
                    loss.correction_regularizer.detach()
                    if summed_regularizer is None
                    else summed_regularizer
                    + loss.correction_regularizer.detach()
                )
                summed_valid_pixels += loss.valid_pixel_count
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                stage.refiner.parameters(),
                float(config.train.gradient_clip),
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            assert summed_total is not None
            assert summed_disparity is not None
            assert summed_regularizer is not None
            values = (
                torch.stack(
                    (
                        summed_total / float(accumulation),
                        summed_disparity / float(accumulation),
                        summed_regularizer / float(accumulation),
                        gradient_norm.detach(),
                    )
                )
                .float()
                .cpu()
                .tolist()
            )
            latest_loss_summary = {
                "total": float(values[0]),
                "disparity": float(values[1]),
                "correction_regularizer": float(values[2]),
                "valid_pixel_count": summed_valid_pixels,
            }
            if completed_steps % int(config.train.log_interval) == 0:
                record = {
                    "step": completed_steps,
                    "stage": "epipolar",
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": float(values[3]),
                    "elapsed_seconds": previous_elapsed_seconds
                    + time.perf_counter()
                    - started,
                    "loss": latest_loss_summary,
                }
                _append_jsonl(log_handle, record)
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

            if completed_steps % int(config.train.checkpoint_interval) == 0:
                # A durable checkpoint must never get ahead of its audit log.
                log_handle.flush()
                os.fsync(log_handle.fileno())
                payload = _stage_c_checkpoint_payload(
                    stage=stage,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_steps=completed_steps,
                    config=config,
                    git_hash=repository_hash,
                    runtime_source_bundle=runtime_source_bundle,
                    training_runtime=training_runtime,
                    base_checkpoint=base_checkpoint,
                    base_lineage=base_lineage,
                    raw_lineage=raw_lineage,
                    base_completion=base_completion,
                    rectification_audit=rectification_audit,
                    latest_loss=latest_loss_summary,
                    elapsed_seconds=previous_elapsed_seconds
                    + time.perf_counter()
                    - started,
                    batches_per_epoch=len(loader),
                )
                atomic_torch_save(payload, latest_path)
        log_handle.flush()
        os.fsync(log_handle.fileno())

    segment_elapsed_seconds = time.perf_counter() - started
    elapsed_seconds = previous_elapsed_seconds + segment_elapsed_seconds
    end_source_bundle = _runtime_source_bundle()
    if end_source_bundle != runtime_source_bundle or (
        repository_git_hash(PROJECT_ROOT) != repository_hash
    ):
        raise RuntimeError(
            "Stage-C runtime source changed during training; refusing completion checkpoint"
        )
    payload = _stage_c_checkpoint_payload(
        stage=stage,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_steps=completed_steps,
        config=config,
        git_hash=repository_hash,
        runtime_source_bundle=runtime_source_bundle,
        training_runtime=training_runtime,
        base_checkpoint=base_checkpoint,
        base_lineage=base_lineage,
        raw_lineage=raw_lineage,
        base_completion=base_completion,
        rectification_audit=rectification_audit,
        latest_loss=latest_loss_summary,
        elapsed_seconds=elapsed_seconds,
        batches_per_epoch=len(loader),
    )
    atomic_torch_save(payload, latest_path)
    execution_complete = completed_steps == configured_steps
    if execution_complete:
        atomic_torch_save(payload, final_path)
        config_sha256 = hashlib.sha256(
            config_fingerprint(resolved_config).encode("utf-8")
        ).hexdigest()
        summary = {
            "schema_version": 1,
            "component": "ffs-omega-tsr-epipolar-training-run",
            "status": "TRAINING_COMPLETE",
            "stage": "epipolar",
            "steps": completed_steps,
            "configured_steps": configured_steps,
            "run_steps": completed_steps - starting_step,
            "elapsed_seconds": elapsed_seconds,
            "segment_elapsed_seconds": segment_elapsed_seconds,
            "segment_steps_per_second": (
                (completed_steps - starting_step) / segment_elapsed_seconds
            ),
            "git_hash": repository_hash,
            "config_sha256": config_sha256,
            "training_runtime": training_runtime,
            "base_checkpoint": base_checkpoint,
            "base_completion": base_completion,
            "runtime_source_bundle_sha256": runtime_source_bundle["bundle_sha256"],
            "formal_training_complete": payload["completion"][
                "formal_training_complete"
            ],
            "final_checkpoint": {
                "path": str(final_path),
                "sha256": sha256_file(final_path),
            },
            "latest_checkpoint": {
                "path": str(latest_path),
                "sha256": sha256_file(latest_path),
            },
            "training_log": {
                "path": str(log_path),
                "sha256": sha256_file(log_path),
            },
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        }
        _write_json_atomic(summary_path, summary)
    status = "TRAINING_COMPLETE" if execution_complete else "BOUNDED_RUN_COMPLETE"
    print(
        json.dumps(
            {
                "status": status,
                "stage": "epipolar",
                "completed_steps": completed_steps,
                "configured_steps": configured_steps,
                "trainable_refiner_parameter_count": stage.trainable_parameter_count,
                "base_parameters_frozen": True,
                "supervision": PSEUDO_GT_SUPERVISION,
                "checkpoint": str(final_path if execution_complete else latest_path),
                "checkpoint_sha256": sha256_file(
                    final_path if execution_complete else latest_path
                ),
                "run_summary": str(summary_path) if execution_complete else None,
                "formal_training_complete": payload["completion"][
                    "formal_training_complete"
                ],
                "loss": latest_loss_summary,
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
