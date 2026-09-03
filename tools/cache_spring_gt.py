#!/usr/bin/env python3
"""Create an independent Spring ground-truth supervision cache.

The tensor keys retain the training dataset's historical ``teacher_*`` names,
but the cache identity is ``spring-ground-truth`` and cannot be consumed by a
legacy FFS pseudo-teacher configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.cache_dataset import (  # noqa: E402
    CacheIdentity,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import load_manifest  # noqa: E402
from data.spring import (  # noqa: E402
    SPRING_FLOW_LIBRARY_COMMIT,
    SPRING_GT_COMPONENT,
    SPRING_GT_TARGET_TYPE,
    load_spring_disparity,
)


def _safe(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"invalid path component: {value!r}")
    return normalized


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _resolve_manifest_path(path_text: str, manifest_directory: Path) -> Path:
    """Resolve a manifest path relative to the JSONL file that owns it."""

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def _record_gt_path(record: Any, manifest_directory: Path) -> Path:
    path = _resolve_manifest_path(record.gt_disparity_path or "", manifest_directory)
    if not path.is_file():
        raise FileNotFoundError(f"Spring GT disparity is missing: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dtype", choices=["float16", "float32"], default="float32"
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid start-index/limit")
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = load_manifest(manifest)
    selected = records[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("manifest selection is empty")
    if len(selected) != len(records):
        raise ValueError(
            "GT teacher cache must cover the complete manifest; create a sliced manifest "
            "instead of using --start-index/--limit"
        )
    manifest_directory = manifest.parent

    manifest_sha = sha256_file(manifest)
    config = {
        "identity_version": 2,
        "dataset": "Spring",
        "dataset_version": "2.0",
        "supervision_source": "Spring_GT",
        "target_type": SPRING_GT_TARGET_TYPE,
        "resolution": "image",
        "sampling": "dsp5[::2,::2]",
        "disparity_value_scaling": 1.0,
        "disparity_unit": "full_hd_pixels",
        "invalid_policy": "finite_and_positive_only",
        "cache_dtype": args.cache_dtype,
    }
    identity = CacheIdentity(
        component=SPRING_GT_COMPONENT,
        upstream_commit=SPRING_FLOW_LIBRARY_COMMIT,
        checkpoint_sha256=canonical_json_sha256(
            # The dense Spring GT source is a dataset-level immutable
            # teacher, not a model trained separately for each split.  Keep
            # the cache identity split-independent so a checkpoint trained
            # on one sequence-disjoint manifest can be evaluated against a
            # validation manifest without a false lineage mismatch.  The
            # per-manifest hash remains in the receipt and every record's
            # source metadata for exact coverage/audit checks.
            {
                "dataset": "Spring",
                "source": "disp1_left",
                "encoding": "positive_left_reference_magnitude",
                "resolution": "image",
                "disparity_unit": "full_hd_pixels",
                "config": config,
            }
        ),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(config),
    )
    root = args.output.expanduser().resolve() / "teacher"
    receipt_path = root / "run_receipt.json"
    if receipt_path.is_file() and not args.overwrite:
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            old.get("identity") != identity.to_dict()
            or old.get("manifest_sha256") != manifest_sha
        ):
            raise RuntimeError(
                f"existing cache receipt identity differs: {receipt_path}"
            )

    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    rows: list[dict[str, Any]] = []
    valid_pixel_count = 0
    total_pixel_count = 0
    maximum_disparity_hr_px = 0.0
    started = time.perf_counter()
    for selection_index, record in enumerate(selected, start=args.start_index):
        gt_path = _record_gt_path(record, manifest_directory)
        image_left = _resolve_manifest_path(record.left_path, manifest_directory)
        image_right = _resolve_manifest_path(record.right_path, manifest_directory)
        if not image_left.is_file() or not image_right.is_file():
            raise FileNotFoundError(
                f"Spring stereo images are missing for {record.sequence_id}/{record.frame_id}"
            )
        with (
            Image.open(image_left) as left_image,
            Image.open(image_right) as right_image,
        ):
            if left_image.size != right_image.size:
                raise RuntimeError(
                    f"Spring stereo image sizes differ: {image_left} vs {image_right}"
                )
            image_width, image_height = left_image.size
        path = root / _safe(record.sequence_id) / f"{_safe(record.frame_id)}.pt"
        left_hash, right_hash, gt_hash = (
            sha256_file(image_left),
            sha256_file(image_right),
            sha256_file(gt_path),
        )
        if path.is_file() and not args.overwrite:
            payload = load_cache_record(path, expected_identity=identity)
            source = payload["metadata"].get("source", {})
            if (
                source.get("left_sha256") != left_hash
                or source.get("right_sha256") != right_hash
                or source.get("gt_sha256") != gt_hash
            ):
                raise RuntimeError(f"cache source mismatch: {path}")
            tensors = payload["tensors"]
            cached_disparity = tensors.get("teacher_disparity_hr_px")
            cached_valid = tensors.get("teacher_valid_mask")
            if not isinstance(cached_disparity, torch.Tensor) or not isinstance(
                cached_valid, torch.Tensor
            ):
                raise RuntimeError(f"malformed Spring GT cache tensors: {path}")
            valid_values = cached_disparity[cached_valid.to(dtype=torch.bool)]
            valid_pixel_count += int(valid_values.numel())
            total_pixel_count += int(cached_disparity.numel())
            if valid_values.numel():
                maximum_disparity_hr_px = max(
                    maximum_disparity_hr_px, float(valid_values.max().item())
                )
            rows.append(
                {
                    "selection_index": selection_index,
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "cache_path": str(path),
                    "cache_sha256": sha256_file(path),
                    "status": "reused",
                    "source": payload["metadata"].get("source"),
                }
            )
            continue

        disparity = load_spring_disparity(gt_path, resolution="image", sign="positive")
        if disparity.ndim != 2 or not np.isfinite(disparity).any():
            raise ValueError(f"invalid Spring GT disparity: {gt_path}")
        if disparity.shape != (image_height, image_width):
            raise ValueError(
                "Spring GT/image shape mismatch for "
                f"{record.sequence_id}/{record.frame_id}: "
                f"GT={disparity.shape}, image={(image_height, image_width)}"
            )
        valid = np.isfinite(disparity) & (disparity > 0)
        disparity = np.where(valid, disparity, 0.0).astype(np.float32, copy=False)
        disparity_t = torch.from_numpy(disparity).unsqueeze(0).to(dtype)
        valid_t = torch.from_numpy(valid).unsqueeze(0)
        confidence_t = valid_t.to(dtype)
        tensors = {
            "teacher_disparity_hr_px": disparity_t,
            "teacher_confidence": confidence_t,
            "teacher_entropy": torch.where(
                valid_t, torch.zeros_like(confidence_t), torch.ones_like(confidence_t)
            ),
            "teacher_last_update_magnitude_hr_px": torch.zeros_like(confidence_t),
            "teacher_valid_mask": valid_t,
            "teacher_trusted_mask": valid_t,
        }
        metadata = {
            "source": {
                "manifest_path": str(manifest),
                "manifest_record": record.to_dict(),
                "left_sha256": left_hash,
                "right_sha256": right_hash,
                "gt_path": str(gt_path),
                "gt_sha256": gt_hash,
                "hr_shape_bchw": [
                    1,
                    3,
                    int(disparity.shape[0]),
                    int(disparity.shape[1]),
                ],
            },
            "checkpoint": {
                "label": "Spring_GT",
                "path": str(gt_path),
                "sha256": gt_hash,
            },
            "config": config,
            "supervision": {
                "target_type": SPRING_GT_TARGET_TYPE,
                "cache_component": SPRING_GT_COMPONENT,
                "paper_ground_truth": True,
                "synthetic_ground_truth": True,
                "sampling": "dsp5[::2,::2] without value scaling",
            },
            "adapter": {
                "frozen": True,
                "inference_mode": True,
                "source": "dataset_ground_truth",
            },
            "units": {
                "teacher_disparity_hr_px": "Spring full-HD pixels",
                "teacher_confidence": "dimensionless",
                "teacher_valid_mask": "mask",
                "teacher_trusted_mask": "mask",
            },
        }
        save_cache_record(path, tensors=tensors, metadata=metadata, identity=identity)
        valid_values = disparity_t[valid_t]
        valid_pixel_count += int(valid_values.numel())
        total_pixel_count += int(disparity_t.numel())
        if valid_values.numel():
            maximum_disparity_hr_px = max(
                maximum_disparity_hr_px, float(valid_values.max().item())
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

    cache_manifest_path = root / "cache_manifest.jsonl"
    _atomic_jsonl(cache_manifest_path, rows)
    receipt = {
        "schema_version": 1,
        "identity_version": int(config["identity_version"]),
        "identity": identity.to_dict(),
        "config": config,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "cache_manifest": str(cache_manifest_path.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "selected_records": len(rows),
        "written_records": sum(row["status"] == "written" for row in rows),
        "reused_records": sum(row["status"] == "reused" for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "target_type": SPRING_GT_TARGET_TYPE,
        "paper_ground_truth": True,
        "synthetic_ground_truth": True,
        "statistics": {
            "valid_pixels": valid_pixel_count,
            "invalid_pixels": total_pixel_count - valid_pixel_count,
            "valid_fraction": (
                float(valid_pixel_count) / float(total_pixel_count)
                if total_pixel_count
                else 0.0
            ),
            "maximum_disparity_hr_px": maximum_disparity_hr_px,
            "maximum_disparity_lr_px_at_x2": maximum_disparity_hr_px / 2.0,
        },
        "lineage_note": "Spring GT supervision; not an FFS pseudo-teacher",
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
