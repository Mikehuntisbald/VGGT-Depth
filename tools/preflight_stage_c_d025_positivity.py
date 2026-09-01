#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the D-025 Stage-C positivity arm."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import numpy as np
from numpy.core.multiarray import _reconstruct as numpy_reconstruct
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.cache_dataset import sha256_file  # noqa: E402
from train_epipolar import (  # noqa: E402
    resolve_epipolar_config,
    stage_c_positivity_ablation_from_config,
    validate_d025_stage_b_prerequisites,
    validate_epipolar_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only prerequisite check for Stage-C positivity from D-025."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--d025-base-checkpoint", type=Path, required=True)
    parser.add_argument("--d025-training-audit", type=Path, required=True)
    parser.add_argument("--d025-evaluation-audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def load_base_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D-025 base checkpoint is missing: {checkpoint}")
    payload_bytes = checkpoint.read_bytes()
    safe_globals = [
        numpy_reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    try:
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(
                io.BytesIO(payload_bytes),
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:  # noqa: BLE001 - normalize unsafe/corrupt inputs
        raise ValueError(f"cannot safe-load D-025 base checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("D-025 base checkpoint schema is malformed")
    for name in ("step", "config", "git_hash", "parameter_count", "model"):
        if name not in payload:
            raise ValueError(f"D-025 base checkpoint field is missing: {name}")
    if not isinstance(payload["config"], Mapping) or not isinstance(
        payload["model"], Mapping
    ):
        raise ValueError("D-025 base checkpoint config/model is malformed")
    return {
        "path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "step": payload["step"],
        "parameter_count": payload["parameter_count"],
        "git_hash": payload["git_hash"],
        "training_config": payload["config"],
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_epipolar_config(args.config)
    for name, value in (
        (
            "stage_c_positivity_ablation.d025_training_audit_path",
            args.d025_training_audit,
        ),
        (
            "stage_c_positivity_ablation.d025_evaluation_audit_path",
            args.d025_evaluation_audit,
        ),
        ("train.initialization_checkpoint", args.d025_base_checkpoint),
    ):
        OmegaConf.update(
            config,
            name,
            str(Path(value).expanduser().resolve()),
            merge=False,
        )
    validate_epipolar_config(config)
    ablation = stage_c_positivity_ablation_from_config(config)
    if not ablation.enabled:
        raise ValueError("preflight requires the Stage-C D-025 positivity config")
    base_metadata = load_base_metadata(args.d025_base_checkpoint)
    prerequisite = validate_d025_stage_b_prerequisites(
        base_metadata=base_metadata,
        stage_c_config=config,
        training_audit_path=args.d025_training_audit,
        evaluation_audit_path=args.d025_evaluation_audit,
    )
    receipt = {
        "schema_version": 1,
        "component": "d025-stage-c-positivity-preflight",
        "status": "PREFLIGHT_PASS",
        "read_only": True,
        "gpu_used": False,
        "canonical_stage_c_replacement": False,
        "config": OmegaConf.to_container(config, resolve=True),
        "prerequisite": prerequisite,
    }
    return receipt


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = run_preflight(args)
    except (FileNotFoundError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_BLOCKED",
                    "read_only": True,
                    "gpu_used": False,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    if args.receipt is not None:
        _write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
