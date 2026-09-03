#!/usr/bin/env python3
"""Cache Fast-FoundationStereo outputs for a Spring manifest.

This is the Spring-specific counterpart of :mod:`tools.cache_ffs`.  It uses
the same frozen serialized Fast-FoundationStereo checkpoints and adapter, but
resolves Spring manifest paths relative to the manifest file.  Observation
caches may be produced on either the half-resolution F1 grid or the
full-resolution F0 grid.  Those variants use distinct cache components and
output directories so they cannot be reused across arms accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backbones.ffs_adapter import FFSAdapter  # noqa: E402
from data.cache_dataset import (  # noqa: E402
    CacheIdentity,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import load_manifest  # noqa: E402

try:  # Importable both as ``tools.cache_spring_ffs`` and as a script.
    from tools.cache_ffs import _load_model  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from cache_ffs import _load_model  # type: ignore[no-redef]  # noqa: E402


EXPECTED_CHECKPOINT_LABEL = {"observation": "20-30-48", "teacher": "23-36-37"}


@dataclass(frozen=True, slots=True)
class CacheVariant:
    """Role-bound FFS grid and cache namespace."""

    role: str
    scale: int
    resolution_mode: str
    component: str
    output_directory: str
    default_iterations: int
    default_max_disp: int
    max_disp_policy: str

    @property
    def default_hr_equivalent_max_disp(self) -> int:
        return self.scale * self.default_max_disp


def _cache_variant(role: str, scale: int | None = None) -> CacheVariant:
    """Resolve a cache variant while preserving the legacy half-grid default."""

    if role == "observation":
        resolved_scale = 2 if scale is None else int(scale)
        if resolved_scale == 2:
            return CacheVariant(
                role=role,
                scale=2,
                resolution_mode="half",
                component="ffs-observation",
                output_directory="observation",
                default_iterations=4,
                default_max_disp=192,
                max_disp_policy="matched_physical_search_range_384_hr_px",
            )
        if resolved_scale == 1:
            return CacheVariant(
                role=role,
                scale=1,
                resolution_mode="full",
                component="ffs-observation-full-resolution",
                output_directory="observation_full_resolution",
                default_iterations=4,
                # Match F1's physical search domain: 192 LR px * scale 2.
                default_max_disp=384,
                max_disp_policy="matched_physical_search_range_384_hr_px",
            )
        raise ValueError("Spring FFS observation scale must be 1 (F0) or 2 (F1)")
    if role == "teacher":
        resolved_scale = 1 if scale is None else int(scale)
        if resolved_scale != 1:
            raise ValueError("Spring FFS teacher scale must be 1")
        return CacheVariant(
            role=role,
            scale=1,
            resolution_mode="full",
            component="ffs-teacher",
            output_directory="teacher",
            default_iterations=8,
            default_max_disp=416,
            max_disp_policy="high_accuracy_teacher_416_hr_px",
        )
    raise ValueError(f"unknown Spring FFS cache role: {role!r}")


def _safe(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError(f"invalid path component: {value!r}")
    return text


def _git_head(repo: Path) -> str:
    """Return the exact upstream checkout revision used by this cache run."""

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Fast-FoundationStereo repository must be a readable Git checkout; "
            f"cannot resolve HEAD under {repo}: {exc}"
        ) from exc
    dirty = status.stdout.strip()
    if dirty:
        raise RuntimeError(
            "Fast-FoundationStereo repository must be clean before caching; "
            f"dirty paths under {repo}:\n{dirty}"
        )
    revision = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError(
            f"Fast-FoundationStereo repository returned an invalid Git revision: {revision!r}"
        )
    return revision.lower()


def _resolve_manifest_path(path_text: str, manifest_directory: Path) -> Path:
    """Resolve a manifest path relative to its JSONL file, per schema contract."""

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def _rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _resize(rgb: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return rgb
    h, w = rgb.shape[-2:]
    if h % scale or w % scale:
        raise ValueError(f"image shape {(h, w)} is not divisible by scale {scale}")
    return F.interpolate(
        rgb,
        size=(h // scale, w // scale),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _cache_tensors(
    role: str, output: Any, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    if role == "observation":
        tensors: dict[str, torch.Tensor] = {
            "observation_disparity_lr_px": output.disparity_lr_px.to(dtype),
            "observation_disparity_hr_px": output.disparity_hr_px.to(dtype),
            "observation_confidence": output.confidence.to(dtype),
            "observation_entropy": output.entropy.to(dtype),
            "observation_last_update_magnitude_lr_px": output.last_update_magnitude_input_px.to(
                dtype
            ),
            "observation_valid_mask": output.valid_mask,
        }
        if output.left_right_error_lr_px is not None:
            tensors["observation_left_right_error_lr_px"] = (
                output.left_right_error_lr_px.to(dtype)
            )
            trusted = (
                output.valid_mask
                & (output.confidence > 0.8)
                & (output.left_right_error_lr_px < 1.0)
            )
        else:
            trusted = output.valid_mask & (output.confidence > 0.8)
        tensors["observation_trusted_mask"] = trusted
        return tensors
    tensors = {
        "teacher_disparity_hr_px": output.disparity_hr_px.to(dtype),
        "teacher_confidence": output.confidence.to(dtype),
        "teacher_entropy": output.entropy.to(dtype),
        "teacher_last_update_magnitude_hr_px": output.last_update_magnitude_input_px.to(
            dtype
        ),
        "teacher_valid_mask": output.valid_mask,
    }
    if output.left_right_error_lr_px is not None:
        tensors["teacher_left_right_error_hr_px"] = output.left_right_error_lr_px.to(
            dtype
        )
        trusted = (
            output.valid_mask
            & (output.confidence > 0.8)
            & (output.left_right_error_lr_px < 1.0)
        )
    else:
        trusted = output.valid_mask & (output.confidence > 0.8)
    tensors["teacher_trusted_mask"] = trusted
    return tensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-label")
    parser.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "Fast-FoundationStereo",
    )
    parser.add_argument("--role", choices=["observation", "teacher"], required=True)
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2),
        help=(
            "observation input scale: 1 creates the isolated full-resolution "
            "F0 cache, 2 creates the half-resolution F1/training cache (default)"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-disp", type=int)
    parser.add_argument(
        "--volume-backend", choices=("pytorch1", "triton"), default="pytorch1"
    )
    parser.add_argument(
        "--missing-normalize",
        choices=("error", "true", "false"),
        default="error",
    )
    parser.add_argument("--allow-provisional-role", action="store_true")
    parser.add_argument("--right-left-check", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cache-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid start/limit")
    variant = _cache_variant(args.role, args.scale)
    scale = variant.scale
    iterations = (
        args.iterations if args.iterations is not None else variant.default_iterations
    )
    max_disp = args.max_disp if args.max_disp is not None else variant.default_max_disp
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if max_disp <= 0 or max_disp % 16:
        raise ValueError("max-disp must be positive and divisible by 16")
    manifest_path = args.manifest.expanduser().resolve()
    default_checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "ffs"
        / EXPECTED_CHECKPOINT_LABEL[args.role]
        / "model_best_bp2_serialize.pth"
    )
    checkpoint_path = (args.checkpoint or default_checkpoint).expanduser().resolve()
    repo_path = args.repo.expanduser().resolve()
    if not checkpoint_path.is_file() or not repo_path.is_dir():
        raise FileNotFoundError(
            "checkpoint and Fast-FoundationStereo repository must exist"
        )
    checkpoint_label = args.checkpoint_label or checkpoint_path.parent.name
    expected_label = EXPECTED_CHECKPOINT_LABEL[args.role]
    provisional = checkpoint_label != expected_label
    if provisional and not args.allow_provisional_role:
        raise ValueError(
            f"{args.role} requires checkpoint label {expected_label!r}; got "
            f"{checkpoint_label!r}. Pass --allow-provisional-role only for a labeled probe."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    records = load_manifest(manifest_path)
    manifest_directory = manifest_path.parent
    selected = records[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("manifest selection is empty")
    checkpoint_sha = sha256_file(checkpoint_path)
    upstream_commit = _git_head(repo_path)
    config = {
        "dataset": "Spring",
        "role": args.role,
        "scale": scale,
        "resolution_mode": variant.resolution_mode,
        "iterations": iterations,
        "max_disp": max_disp,
        "max_disp_input_grid_px": max_disp,
        "max_disp_hr_equivalent_px": max_disp * scale,
        "max_disp_policy": variant.max_disp_policy,
        "volume_backend": args.volume_backend,
        "right_left_check": bool(args.right_left_check),
        "cache_dtype": args.cache_dtype,
        "checkpoint_label": checkpoint_label,
        "expected_checkpoint_label": expected_label,
        "provisional_checkpoint_role": provisional,
        "missing_normalize": args.missing_normalize,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "adapter": "FFSAdapter",
        "upstream_repo": str(repo_path),
        "upstream_commit": upstream_commit,
    }
    identity = CacheIdentity(
        component=variant.component,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(config),
    )
    root = args.output.expanduser().resolve() / variant.output_directory
    receipt_path = root / "run_receipt.json"
    if receipt_path.is_file() and not args.overwrite:
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("identity") != identity.to_dict() or old.get(
            "manifest_sha256"
        ) != sha256_file(manifest_path):
            raise RuntimeError(
                f"existing cache receipt identity differs: {receipt_path}"
            )
    model, compatibility = _load_model(
        checkpoint=checkpoint_path,
        repo=repo_path,
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
    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for selection_index, record in enumerate(selected, start=args.start_index):
            left = _resolve_manifest_path(record.left_path, manifest_directory)
            right = _resolve_manifest_path(record.right_path, manifest_directory)
            if not left.is_file() or not right.is_file():
                raise FileNotFoundError(f"missing Spring images: {left}, {right}")
            path = root / _safe(record.sequence_id) / f"{_safe(record.frame_id)}.pt"
            left_hash, right_hash = sha256_file(left), sha256_file(right)
            if path.is_file() and not args.overwrite:
                payload = load_cache_record(path, expected_identity=identity)
                source = payload["metadata"].get("source", {})
                if (
                    source.get("left_sha256") != left_hash
                    or source.get("right_sha256") != right_hash
                ):
                    raise RuntimeError(f"cache source mismatch: {path}")
                rows.append(
                    {
                        "selection_index": selection_index,
                        "sequence_id": record.sequence_id,
                        "frame_id": record.frame_id,
                        "cache_path": str(path),
                        "cache_sha256": sha256_file(path),
                        "status": "reused",
                        # Stage-C binds the cache inventory to the exact
                        # source record and endpoint right-image digest. Keep
                        # this row-level provenance even when the tensor file
                        # itself is reused.
                        "source": payload["metadata"].get("source"),
                    }
                )
                continue
            left_hr = _rgb(left)
            right_hr = _rgb(right)
            if left_hr.shape != right_hr.shape:
                raise ValueError(
                    f"stereo shape mismatch: {left_hr.shape} vs {right_hr.shape}"
                )
            left_input = _resize(left_hr, scale).to(device=device)
            right_input = _resize(right_hr, scale).to(device=device)
            output = adapter(
                left_input, right_input, right_left_check=bool(args.right_left_check)
            )
            tensors = _cache_tensors(args.role, output, dtype)
            metadata = {
                "source": {
                    "manifest_path": str(manifest_path),
                    "manifest_record": record.to_dict(),
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                    "hr_shape_bchw": list(left_hr.shape),
                    "ffs_input_shape_bchw": list(left_input.shape),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha,
                    "label": checkpoint_label,
                    "expected_role_label": expected_label,
                    "provisional_role": provisional,
                    "size_bytes": checkpoint_path.stat().st_size,
                },
                "config": config,
                "adapter": dict(output.metadata),
                "checkpoint_compatibility": compatibility,
            }
            save_cache_record(
                path, tensors=tensors, metadata=metadata, identity=identity
            )
            rows.append(
                {
                    "selection_index": selection_index,
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "cache_path": str(path),
                    "cache_sha256": sha256_file(path),
                    "status": "written",
                    "source": metadata["source"],
                }
            )
            print(f"[{len(rows)}/{len(selected)}] {path}", flush=True)
    finally:
        adapter.close()
    manifest_sha = sha256_file(manifest_path)
    cache_manifest_path = root / "cache_manifest.jsonl"
    _atomic_jsonl(cache_manifest_path, rows)
    receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "config": config,
        "checkpoint_compatibility": compatibility,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "cache_manifest": str(cache_manifest_path.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "selected_records": len(rows),
        "written_records": sum(row["status"] == "written" for row in rows),
        "reused_records": sum(row["status"] == "reused" for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
