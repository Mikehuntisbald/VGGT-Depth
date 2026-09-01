#!/usr/bin/env python3
"""Cache Spring v2 ``disp1_left`` as real HR supervision tensors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image

from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import ManifestRecord, load_manifest
from data.spring import (
    SPRING_FLOW_LIBRARY_COMMIT,
    SPRING_GT_COMPONENT,
    SPRING_GT_TARGET_TYPE,
    read_spring_disparity,
)
from data.training_dataset import cache_path_for_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Cache root written directly as sequence/frame.pt plus run_receipt.json",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                json.dump(row, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve(path_text: str, manifest: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _source_identity(
    record: ManifestRecord, manifest: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    left = _resolve(record.left_path, manifest)
    right = _resolve(record.right_path, manifest)
    if record.gt_disparity_path is None:
        raise CacheMismatchError(
            f"Spring ground truth is missing for {record.sequence_id}/{record.frame_id}"
        )
    disparity = _resolve(record.gt_disparity_path, manifest)
    for path in (left, right, disparity):
        if not path.is_file():
            raise FileNotFoundError(f"Spring cache source is missing: {path}")
    return left, right, disparity, {
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
        "ground_truth_sha256": sha256_file(disparity),
    }


def main() -> int:
    args = _parse_args()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.start_index < 0 or args.limit is not None and args.limit <= 0:
        raise ValueError("start-index must be non-negative and limit must be positive")
    records = load_manifest(manifest)
    selected = records[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("Spring ground-truth selection is empty")
    manifest_sha256 = sha256_file(manifest)
    config = {
        "dataset": "Spring",
        "dataset_version": "2.0",
        "source_format": "HDF5 .dsp5 dataset=disparity",
        "target_type": SPRING_GT_TARGET_TYPE,
        "target_grid": "RGB Full-HD image grid",
        "sampling": "dsp5[::2,::2]",
        "disparity_value_scaling": 1.0,
        "invalid_policy": "zero/NaN/Inf -> zero value and false validity",
        "cache_dtype": "float32",
        "legacy_checkpoint_sha256_field_owner": "source manifest SHA-256",
    }
    identity = CacheIdentity(
        component=SPRING_GT_COMPONENT,
        upstream_commit=SPRING_FLOW_LIBRARY_COMMIT,
        # CacheIdentity v1 predates non-model supervision.  Bind this legacy
        # field to the immutable source manifest rather than inventing a model.
        checkpoint_sha256=manifest_sha256,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(config),
    )
    canonical_receipt = output / "run_receipt.json"
    if canonical_receipt.is_file():
        existing = json.loads(canonical_receipt.read_text(encoding="utf-8"))
        if existing.get("identity") != identity.to_dict():
            raise CacheMismatchError(
                "Spring GT cache identity differs; choose a new output root"
            )
        if existing.get("manifest_sha256") != manifest_sha256:
            raise CacheMismatchError(
                "Spring GT cache manifest differs; choose a new output root"
            )

    rows: list[dict[str, Any]] = []
    valid_pixel_count = 0
    total_pixel_count = 0
    maximum_disparity_hr_px = 0.0
    started = time.perf_counter()
    for manifest_index, record in enumerate(
        selected, start=args.start_index
    ):
        left, right, gt_path, hashes = _source_identity(record, manifest)
        cache_path = cache_path_for_record(output, record)
        expected_source = {
            "manifest_path": str(manifest),
            "manifest_index": manifest_index,
            "manifest_record": record.to_dict(),
            **hashes,
        }
        if cache_path.is_file() and not args.overwrite:
            payload = load_cache_record(cache_path, expected_identity=identity)
            if payload["metadata"].get("source") != expected_source:
                raise CacheMismatchError(
                    f"Spring GT cache source differs: {cache_path}"
                )
            status = "reused_identity_match"
            cached_disparity = payload["tensors"].get(
                "teacher_disparity_hr_px"
            )
            cached_valid = payload["tensors"].get("teacher_valid_mask")
            if not isinstance(cached_disparity, torch.Tensor) or not isinstance(
                cached_valid, torch.Tensor
            ):
                raise CacheMismatchError(
                    f"Spring GT cache tensors are malformed: {cache_path}"
                )
            valid_values = cached_disparity[cached_valid.to(dtype=torch.bool)]
            valid_pixel_count += int(valid_values.numel())
            total_pixel_count += int(cached_disparity.numel())
            if valid_values.numel():
                maximum_disparity_hr_px = max(
                    maximum_disparity_hr_px,
                    float(valid_values.max().item()),
                )
        else:
            with Image.open(left) as left_image, Image.open(right) as right_image:
                if left_image.size != right_image.size:
                    raise CacheMismatchError(
                        f"Spring stereo image sizes differ: {left} vs {right}"
                    )
                width, height = left_image.size
            gt = read_spring_disparity(gt_path, image_size_hw=(height, width))
            disparity = torch.from_numpy(gt.disparity_hr_px).unsqueeze(0)
            valid = torch.from_numpy(gt.valid_mask).unsqueeze(0)
            confidence = valid.to(dtype=torch.float32)
            tensors = {
                "teacher_disparity_hr_px": disparity,
                "teacher_confidence": confidence,
                "teacher_valid_mask": valid,
                "teacher_trusted_mask": valid.clone(),
            }
            metadata = {
                "source": expected_source,
                "supervision": {
                    "target_type": SPRING_GT_TARGET_TYPE,
                    "paper_ground_truth": True,
                    "synthetic_rendered_ground_truth": True,
                    "source_size_hw": list(gt.source_size_hw),
                    "target_size_hw": list(gt.target_size_hw),
                    "disparity_unit": "Full-HD image pixels",
                    "sampling": "dsp5[::2,::2] without value scaling",
                },
                "config": config,
                "units": {
                    "teacher_disparity_hr_px": "HR image pixels",
                    "teacher_confidence": "dimensionless",
                    "teacher_valid_mask": "mask/dimensionless",
                    "teacher_trusted_mask": "mask/dimensionless",
                },
            }
            save_cache_record(
                cache_path,
                tensors=tensors,
                metadata=metadata,
                identity=identity,
            )
            status = "written"
            valid_values = disparity[valid]
            valid_pixel_count += int(valid_values.numel())
            total_pixel_count += int(disparity.numel())
            if valid_values.numel():
                maximum_disparity_hr_px = max(
                    maximum_disparity_hr_px,
                    float(valid_values.max().item()),
                )
        rows.append(
            {
                "selection_index": manifest_index,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "status": status,
                "source": expected_source,
            }
        )
        print(f"[{len(rows)}/{len(selected)}] {status} {cache_path}")

    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "config": config,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "selected_records": len(selected),
        "written_records": sum(row["status"] == "written" for row in rows),
        "reused_records": sum(row["status"].startswith("reused") for row in rows),
        "elapsed_seconds": elapsed,
        "target_type": SPRING_GT_TARGET_TYPE,
        "paper_ground_truth": True,
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
    }
    selection_end = args.start_index + len(selected) - 1
    tag = f"records_{args.start_index:06d}_{selection_end:06d}"
    _atomic_jsonl(output / "runs" / f"{tag}.jsonl", rows)
    _atomic_json(output / "runs" / f"{tag}.json", receipt)
    if args.start_index == 0 and len(selected) == len(records):
        _atomic_jsonl(output / "cache_manifest.jsonl", rows)
        _atomic_json(canonical_receipt, receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(rows),
                "canonical_receipt": (
                    str(canonical_receipt)
                    if canonical_receipt.is_file()
                    else None
                ),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
