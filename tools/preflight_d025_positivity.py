#!/usr/bin/env python3
"""CPU-only readiness check for the separate D-025 full Stage-B rerun.

The preflight intentionally never calls the trainer, starts workers, allocates
CUDA tensors, opens an output directory, or performs an optimizer step.  It
does bind the requested config to the actual formal training caches, reads one
causal sample, and strictly loads the requested final Stage-A model state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import sha256_file  # noqa: E402
from train import (  # noqa: E402
    build_model,
    build_temporal_dataset_and_identities,
    positivity_ablation_from_config,
    resolve_config,
    validate_training_config,
)
from utils.checkpoint import (  # noqa: E402
    config_fingerprint,
    load_model_initialization_checkpoint,
    repository_git_hash,
)


PREFLIGHT_SCHEMA_VERSION = 1
PROTOCOL_NAME = "full_stage_b_rerun_from_final_stage_a"
REQUIRED_UPDATES = 15_000
REQUIRED_STAGE_A_UPDATES = 5_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only formal-input preflight for D-025 positivity."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-cache-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--derived-cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_full_rerun_protocol(config: DictConfig) -> dict[str, Any]:
    """Reject ambiguous Stage-B warm-start or shortened-run configurations."""

    if str(config.train.stage).lower() != "temporal":
        raise ValueError("D-025 preflight requires the Stage-B temporal trainer")
    protocol = _require_mapping(config.get("ablation_protocol"), "ablation_protocol")
    if protocol.get("name") != PROTOCOL_NAME:
        raise ValueError(f"ablation_protocol.name must be {PROTOCOL_NAME!r}")
    if protocol.get("required_updates") != REQUIRED_UPDATES:
        raise ValueError(f"ablation_protocol.required_updates must be {REQUIRED_UPDATES}")
    if protocol.get("stage_b_warm_start") != "forbidden":
        raise ValueError("D-025 protocol must explicitly forbid a Stage-B warm start")
    if int(config.train.steps) != REQUIRED_UPDATES:
        raise ValueError(f"D-025 train.steps must be {REQUIRED_UPDATES}")
    if str(config.train.init_from_stage).lower() != "spatial":
        raise ValueError("D-025 must initialize from the Stage-A spatial stage")
    ablation = positivity_ablation_from_config(config)
    if not ablation.enabled:
        raise ValueError("D-025 preflight requires positivity_ablation.enabled=true")
    return {
        "name": PROTOCOL_NAME,
        "required_updates": REQUIRED_UPDATES,
        "stage_b_warm_start": "forbidden",
        "trainer_stage": "temporal",
    }


def _set_runtime_paths(config: DictConfig, args: argparse.Namespace) -> None:
    for key, value in (
        ("data.manifest_path", args.manifest),
        ("data.observation_cache_root", args.observation_cache_root),
        ("data.teacher_cache_root", args.teacher_cache_root),
        ("data.derived_geometry_cache_root", args.derived_cache_root),
    ):
        OmegaConf.update(config, key, str(value.expanduser().resolve()), merge=False)


def _load_final_stage_a_summary(
    path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Stage-A summary: {path}") from exc
    if not isinstance(summary, Mapping):
        raise ValueError("Stage-A summary must be a JSON object")
    if summary.get("status") != "TRAINING_COMPLETE" or summary.get("stage") != "spatial":
        raise ValueError("Stage-A summary is not a completed spatial run")
    if (
        summary.get("steps") != REQUIRED_STAGE_A_UPDATES
        or summary.get("run_steps") != REQUIRED_STAGE_A_UPDATES
    ):
        raise ValueError("Stage-A summary is not the required completed 5,000-step run")
    final_checkpoint = _require_mapping(summary.get("final_checkpoint"), "final_checkpoint")
    if Path(str(final_checkpoint.get("path", ""))).resolve() != checkpoint.resolve():
        raise ValueError("Stage-A summary final checkpoint path differs from --stage-a-checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    if final_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("Stage-A summary final checkpoint SHA-256 differs from checkpoint")
    return {
        "summary_path": str(path.resolve()),
        "summary_sha256": sha256_file(path),
        "status": "TRAINING_COMPLETE",
        "stage": "spatial",
        "steps": REQUIRED_STAGE_A_UPDATES,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _checkpoint_payload_contract(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-A checkpoint is not a mapping")
    config = _require_mapping(payload.get("config"), "Stage-A checkpoint config")
    data = _require_mapping(config.get("data"), "Stage-A checkpoint data config")
    train = _require_mapping(config.get("train"), "Stage-A checkpoint train config")
    if data.get("sequence_length") != 1:
        raise ValueError("Stage-A checkpoint must have data.sequence_length=1")
    if (
        payload.get("step") != train.get("steps_spatial")
        or payload.get("step") != REQUIRED_STAGE_A_UPDATES
    ):
        raise ValueError("Stage-A checkpoint is not its completed 5,000-step final state")
    return {
        "path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "completed_step": int(payload["step"]),
        "source_sequence_length": 1,
        "parameter_count": int(payload["parameter_count"]),
        "git_hash": str(payload.get("git_hash", "unknown")),
    }


def _sample_shapes(sample: Any) -> dict[str, list[int] | None]:
    fields = (
        "rgb_hr_sequence",
        "disparity_ffs_hr_px_sequence",
        "confidence_ffs_sequence",
        "valid_ffs_sequence",
        "teacher_disparity_hr_px_sequence",
        "vggt_disparity_hr_px_sequence",
        "vggt_confidence_sequence",
        "vggt_valid_mask_sequence",
        "temporal_pose_valid_sequence",
    )
    result: dict[str, list[int] | None] = {}
    for name in fields:
        value = getattr(sample, name)
        result[name] = None if value is None else list(value.shape)
    return result


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run all read-only checks and return a strict-JSON receipt payload."""

    if torch.cuda.is_initialized():
        raise RuntimeError("D-025 preflight refuses a process with initialized CUDA")
    config = resolve_config(args.config)
    _set_runtime_paths(config, args)
    validate_training_config(config)
    protocol = validate_full_rerun_protocol(config)

    stage_a_checkpoint = args.stage_a_checkpoint.expanduser().resolve()
    stage_a_summary = args.stage_a_summary.expanduser().resolve()
    if not stage_a_checkpoint.is_file():
        raise FileNotFoundError(f"Stage-A checkpoint does not exist: {stage_a_checkpoint}")
    if not stage_a_summary.is_file():
        raise FileNotFoundError(f"Stage-A summary does not exist: {stage_a_summary}")
    stage_a_receipt = _load_final_stage_a_summary(stage_a_summary, stage_a_checkpoint)
    checkpoint_contract = _checkpoint_payload_contract(stage_a_checkpoint)

    dataset, observation_identity, teacher_identity = build_temporal_dataset_and_identities(
        config
    )
    if len(dataset) <= 0:
        raise ValueError("formal temporal dataset is empty")
    dataset.set_epoch(0)
    sample = dataset[0]

    model = build_model(config).cpu()
    initialization_lineage = load_model_initialization_checkpoint(
        stage_a_checkpoint,
        model=model,
        expected_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        required_sequence_length=1,
    )
    if initialization_lineage["checkpoint_sha256"] != checkpoint_contract["checkpoint_sha256"]:
        raise RuntimeError("strict Stage-A load returned a mismatched checkpoint hash")

    resolved_fingerprint = config_fingerprint(OmegaConf.to_container(config, resolve=True))
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PREFLIGHT_PASS",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "device": "cpu",
            "cuda_initialized": False,
            "training_started": False,
            "optimizer_steps": 0,
            "dataloader_workers_started": 0,
            "data_sample_loaded": True,
        },
        "protocol": protocol,
        "config": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": sha256_file(args.config.expanduser().resolve()),
            "resolved_fingerprint_sha256": hashlib.sha256(
                resolved_fingerprint.encode("utf-8")
            ).hexdigest(),
            "experiment": str(config.experiment),
            "train_steps": int(config.train.steps),
        },
        "stage_a_final": {**stage_a_receipt, **checkpoint_contract},
        "initialization_lineage": initialization_lineage,
        "formal_train_inputs": {
            "manifest_path": str(args.manifest.expanduser().resolve()),
            "manifest_sha256": sha256_file(args.manifest.expanduser().resolve()),
            "temporal_windows": len(dataset),
            "observation_identity": observation_identity.to_dict(),
            "teacher_identity": teacher_identity.to_dict(),
            "derived_cache_lineage": dataset.cache_lineage_summary,
            "sample": {
                "sequence_id": sample.sequence_id,
                "frame_ids": list(sample.frame_ids),
                "manifest_indices": list(sample.manifest_indices),
                "shapes": _sample_shapes(sample),
            },
        },
        "preflight_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "repository_git_hash": repository_git_hash(PROJECT_ROOT),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_preflight(args)
    receipt_path = args.receipt.expanduser().resolve()
    _write_json_atomic(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
