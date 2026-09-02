#!/usr/bin/env python3
"""Build versioned offline Fast-FoundationStereo cache records from JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backbones.ffs_adapter import FFSAdapter, FFSOutput
from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import load_manifest


EXPECTED_CHECKPOINT_LABEL = {"observation": "20-30-48", "teacher": "23-36-37"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--role", choices=["observation", "teacher"], required=True)
    parser.add_argument("--scale", type=int)
    parser.add_argument(
        "--allow-full-resolution-observation",
        action="store_true",
        help=(
            "Allow the observation checkpoint to run at scale=1 for an explicit "
            "full-resolution baseline. This creates a separate cache identity; "
            "the normal observation contract remains scale=2."
        ),
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--max-disp",
        type=int,
        help="Defaults to 192 on the LR observation grid and 416 on the HR teacher grid",
    )
    parser.add_argument("--volume-backend", choices=["pytorch1", "triton"], default="pytorch1")
    parser.add_argument("--right-left-check", action="store_true")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--missing-normalize", choices=["error", "true", "false"], default="error")
    parser.add_argument(
        "--allow-provisional-role",
        action="store_true",
        help="Required when checkpoint-label is not the agreed role identity",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda",
        help="inference device (cuda, cuda:N, or cpu); CPU is intended for bounded screening",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "Fast-FoundationStereo",
    )
    return parser.parse_args()


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    return [record.to_dict() for record in load_manifest(path)]


def _safe_component(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"cannot create a safe path component from {value!r}")
    return normalized


def _rgb_0_255(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _downsample_integer(rgb_hr: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return rgb_hr
    height, width = rgb_hr.shape[-2:]
    if height % scale or width % scale:
        raise ValueError(
            f"HR image shape {(height, width)} is not divisible by integer scale {scale}"
        )
    return F.interpolate(
        rgb_hr,
        size=(height // scale, width // scale),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _load_model(
    *,
    checkpoint: Path,
    repo: Path,
    iterations: int,
    max_disp: int,
    missing_normalize: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(repo.resolve()))
    import core.foundation_stereo  # noqa: F401

    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    compatibility: dict[str, Any] = {"normalize_injected": False}
    normalize = model.args.get("normalize")
    if normalize is None:
        if missing_normalize == "error":
            raise ValueError(
                "checkpoint is missing model.args.normalize; choose an explicit "
                "--missing-normalize true|false policy"
            )
        normalize = missing_normalize == "true"
        model.args.normalize = normalize
        compatibility = {
            "normalize_injected": True,
            "normalize": normalize,
            "reason": "serialized checkpoint omitted field; explicit CLI policy",
        }
    else:
        compatibility["normalize"] = bool(normalize)
    model.args.valid_iters = iterations
    model.args.max_disp = max_disp
    if device.type != "cuda":
        # The upstream forward wraps CUDA autocast explicitly; disabling its
        # mixed-precision branch keeps the same model usable for CPU screening.
        model.args.mixed_precision = False
    model.requires_grad_(False).eval().to(device)
    return model, compatibility


def _cache_tensors(role: str, output: FFSOutput, cache_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if role == "observation":
        prefix = "observation"
        tensors = {
            f"{prefix}_disparity_lr_px": output.disparity_lr_px.to(cache_dtype),
            f"{prefix}_disparity_hr_px": output.disparity_hr_px.to(cache_dtype),
            f"{prefix}_confidence": output.confidence.to(cache_dtype),
            f"{prefix}_entropy": output.entropy.to(cache_dtype),
            f"{prefix}_last_update_magnitude_lr_px": output.last_update_magnitude_input_px.to(
                cache_dtype
            ),
            f"{prefix}_valid_mask": output.valid_mask,
        }
        if output.left_right_error_lr_px is not None:
            tensors[f"{prefix}_left_right_error_lr_px"] = output.left_right_error_lr_px.to(
                cache_dtype
            )
            trusted = (
                output.valid_mask
                & (output.confidence > 0.8)
                & (output.left_right_error_lr_px < 1.0)
            )
        else:
            trusted = output.valid_mask & (output.confidence > 0.8)
        tensors[f"{prefix}_trusted_mask"] = trusted
        return tensors

    prefix = "teacher"
    tensors = {
        f"{prefix}_disparity_hr_px": output.disparity_hr_px.to(cache_dtype),
        f"{prefix}_confidence": output.confidence.to(cache_dtype),
        f"{prefix}_entropy": output.entropy.to(cache_dtype),
        f"{prefix}_last_update_magnitude_hr_px": output.last_update_magnitude_input_px.to(
            cache_dtype
        ),
        f"{prefix}_valid_mask": output.valid_mask,
    }
    if output.left_right_error_lr_px is not None:
        # Teacher input is the HR grid, so input-pixel disparity is HR-pixel disparity.
        tensors[f"{prefix}_left_right_error_hr_px"] = output.left_right_error_lr_px.to(cache_dtype)
        trusted = (
            output.valid_mask
            & (output.confidence > 0.8)
            & (output.left_right_error_lr_px < 1.0)
        )
    else:
        trusted = output.valid_mask & (output.confidence > 0.8)
    tensors[f"{prefix}_trusted_mask"] = trusted
    return tensors


def main() -> int:
    args = parse_args()
    scale = args.scale if args.scale is not None else (2 if args.role == "observation" else 1)
    iterations = args.iterations if args.iterations is not None else (4 if args.role == "observation" else 8)
    max_disp = args.max_disp if args.max_disp is not None else (192 if args.role == "observation" else 416)
    expected_scale = 2 if args.role == "observation" else 1
    full_resolution_observation = bool(
        args.role == "observation"
        and args.allow_full_resolution_observation
        and scale == 1
    )
    if scale != expected_scale and not full_resolution_observation:
        raise ValueError(
            "MVP observation scale must be 2 and teacher scale must be 1; "
            "pass --allow-full-resolution-observation only for an explicit "
            "scale=1 observation baseline"
        )
    if max_disp <= 0 or max_disp % 16:
        raise ValueError("max_disp must be positive and divisible by 16")
    expected_label = EXPECTED_CHECKPOINT_LABEL[args.role]
    provisional = args.checkpoint_label != expected_label
    if provisional and not args.allow_provisional_role:
        raise ValueError(
            f"{args.role} requires checkpoint label {expected_label!r}; got "
            f"{args.checkpoint_label!r}. Pass --allow-provisional-role only for a labeled probe."
        )
    if args.start_index < 0 or args.limit is not None and args.limit <= 0:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if not args.checkpoint.is_file() or not args.repo.is_dir():
        raise FileNotFoundError("checkpoint and FFS repository must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")

    records = _manifest_records(args.manifest)
    selected = records[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("record selection is empty")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    upstream_commit = _git_head(args.repo)
    config = {
        "role": args.role,
        "scale": scale,
        "resolution_mode": (
            "full_resolution_observation"
            if full_resolution_observation
            else "mvp"
        ),
        "iterations": iterations,
        "max_disp": max_disp,
        "volume_backend": args.volume_backend,
        "right_left_check": args.right_left_check,
        "cache_dtype": args.cache_dtype,
        "checkpoint_label": args.checkpoint_label,
        "expected_checkpoint_label": expected_label,
        "provisional_checkpoint_role": provisional,
        "missing_normalize": args.missing_normalize,
    }
    identity = CacheIdentity(
        component=f"ffs-{args.role}",
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(config),
    )
    output_root = args.output / args.role
    canonical_receipt_path = output_root / "run_receipt.json"
    existing_canonical_receipt: dict[str, Any] | None = None
    if canonical_receipt_path.is_file():
        existing_canonical_receipt = json.loads(
            canonical_receipt_path.read_text(encoding="utf-8")
        )
        if existing_canonical_receipt.get("identity") != identity.to_dict():
            raise CacheMismatchError(
                "cache root canonical identity differs from this run; choose a new output root"
            )
        current_manifest_hash = sha256_file(args.manifest)
        if existing_canonical_receipt.get("manifest_sha256") != current_manifest_hash:
            raise CacheMismatchError(
                "cache root canonical manifest differs from this run; choose a new output root"
            )
    model, compatibility = _load_model(
        checkpoint=args.checkpoint,
        repo=args.repo,
        iterations=iterations,
        max_disp=max_disp,
        missing_normalize=args.missing_normalize,
        device=device,
    )
    adapter = FFSAdapter(
        model,
        spatial_scale=float(scale),
        iterations=iterations,
        volume_backend=args.volume_backend,
    )
    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    index_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for selection_index, record in enumerate(selected, start=args.start_index):
            left_path = Path(record["left_path"])
            right_path = Path(record["right_path"])
            if not left_path.is_file() or not right_path.is_file():
                raise FileNotFoundError(f"missing stereo input: {left_path} or {right_path}")
            left_sha256 = sha256_file(left_path)
            right_sha256 = sha256_file(right_path)
            relative_path = (
                Path(_safe_component(record["sequence_id"]))
                / f"{_safe_component(record['frame_id'])}.pt"
            )
            cache_path = output_root / relative_path
            if cache_path.exists() and not args.overwrite:
                payload = load_cache_record(cache_path, expected_identity=identity)
                cached_source = payload["metadata"].get("source", {})
                source_differences = {
                    "left_sha256": {
                        "expected": left_sha256,
                        "actual": cached_source.get("left_sha256"),
                    },
                    "right_sha256": {
                        "expected": right_sha256,
                        "actual": cached_source.get("right_sha256"),
                    },
                    "manifest_record": {
                        "expected": record,
                        "actual": cached_source.get("manifest_record"),
                    },
                }
                source_differences = {
                    key: value
                    for key, value in source_differences.items()
                    if value["expected"] != value["actual"]
                }
                if source_differences:
                    raise CacheMismatchError(
                        "cache source mismatch: "
                        + json.dumps(source_differences, sort_keys=True, separators=(",", ":"))
                    )
                index_rows.append(
                    {
                        "selection_index": selection_index,
                        "sequence_id": record["sequence_id"],
                        "frame_id": record["frame_id"],
                        "cache_path": str(cache_path.resolve()),
                        "status": "reused_identity_match",
                        "source": payload["metadata"].get("source"),
                    }
                )
                continue

            left_hr = _rgb_0_255(left_path)
            right_hr = _rgb_0_255(right_path)
            if left_hr.shape != right_hr.shape:
                raise ValueError(f"stereo shape mismatch: {left_hr.shape} vs {right_hr.shape}")
            left_input = _downsample_integer(left_hr, scale).to(device)
            right_input = _downsample_integer(right_hr, scale).to(device)
            output = adapter(left_input, right_input, right_left_check=args.right_left_check)
            tensors = _cache_tensors(args.role, output, cache_dtype)
            metadata = {
                "source": {
                    "manifest_path": str(args.manifest.resolve()),
                    "manifest_record": record,
                    "left_sha256": left_sha256,
                    "right_sha256": right_sha256,
                    "hr_shape_bchw": list(left_hr.shape),
                    "ffs_input_shape_bchw": list(left_input.shape),
                },
                "checkpoint": {
                    "path": str(args.checkpoint.resolve()),
                    "label": args.checkpoint_label,
                    "expected_role_label": expected_label,
                    "provisional_role": provisional,
                    "size_bytes": args.checkpoint.stat().st_size,
                    "sha256": checkpoint_sha256,
                },
                "config": config,
                "adapter": dict(output.metadata),
                "checkpoint_compatibility": compatibility,
                "units": {
                    name: (
                        "mask/dimensionless"
                        if name.endswith("mask")
                        else "dimensionless"
                        if "confidence" in name or "entropy" in name
                        else "LR pixels"
                        if "_lr_px" in name
                        else "HR pixels"
                    )
                    for name in tensors
                },
            }
            save_cache_record(
                cache_path,
                tensors=tensors,
                metadata=metadata,
                identity=identity,
            )
            index_rows.append(
                {
                    "selection_index": selection_index,
                    "sequence_id": record["sequence_id"],
                    "frame_id": record["frame_id"],
                    "cache_path": str(cache_path.resolve()),
                    "status": "written",
                    "source": metadata["source"],
                }
            )
            print(f"[{len(index_rows)}/{len(selected)}] {cache_path}")
    finally:
        adapter.close()

    elapsed_seconds = time.perf_counter() - started
    run_receipt = {
            "schema_version": 1,
            "identity": identity.to_dict(),
            "config": config,
            "checkpoint_compatibility": compatibility,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "selected_records": len(selected),
            "written_records": sum(row["status"] == "written" for row in index_rows),
            "reused_records": sum(row["status"].startswith("reused") for row in index_rows),
            "elapsed_seconds": elapsed_seconds,
        }
    selection_end = args.start_index + len(selected) - 1
    selection_tag = f"records_{args.start_index:06d}_{selection_end:06d}"
    _atomic_jsonl(output_root / "runs" / f"{selection_tag}.jsonl", index_rows)
    _atomic_json(output_root / "runs" / f"{selection_tag}.json", run_receipt)
    existing_selected = (
        int(existing_canonical_receipt.get("selected_records", 0))
        if existing_canonical_receipt is not None
        else 0
    )
    if len(selected) >= existing_selected:
        _atomic_jsonl(output_root / "cache_manifest.jsonl", index_rows)
        _atomic_json(canonical_receipt_path, run_receipt)
    print(f"cache role={args.role} records={len(index_rows)} elapsed={elapsed_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
