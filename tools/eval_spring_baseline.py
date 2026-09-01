#!/usr/bin/env python3
"""Evaluate the Spring S0 bilinear FFS baseline.

The regular :mod:`eval` entry point always loads a trainable checkpoint.  S0
is deliberately checkpoint-free, so this small evaluator reads the frozen FFS
observation cache, upsamples its HR-pixel disparity to the Spring image grid,
and compares it with the dense Spring ``disp1_left`` ground truth.  Optional
Spring ``maps`` files are consumed when present; missing maps are reported as
unavailable rather than silently being treated as evidence.

This is a screening report only.  It never claims paper accuracy and leaves
temporal fields unavailable because S0 has no temporal prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import load_cache_record, sha256_file  # noqa: E402
from data.manifest import load_manifest  # noqa: E402
from data.spring import load_spring_disparity  # noqa: E402
from metrics.spring_arms import disparity_metrics  # noqa: E402
from metrics.boundary import disparity_boundary_mask  # noqa: E402


def _safe(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError(f"invalid path component: {value!r}")
    return text


def _cache_path(root: Path, sequence_id: str, frame_id: int) -> Path:
    return root / _safe(sequence_id) / f"{_safe(frame_id)}.pt"


def _extract_observation(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("observation cache has no tensors mapping")
    value = tensors.get("observation_disparity_hr_px")
    trusted = tensors.get("observation_trusted_mask")
    if not isinstance(value, torch.Tensor) or not isinstance(trusted, torch.Tensor):
        raise ValueError("observation cache lacks observation_disparity_hr_px")
    if value.ndim == 4 and tuple(value.shape[:2]) == (1, 1):
        value = value[0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            "observation_disparity_hr_px must resolve to [1,H,W], got "
            f"{tuple(value.shape)}"
        )
    if not value.is_floating_point():
        value = value.float()
    if trusted.ndim == 4 and tuple(trusted.shape[:2]) == (1, 1):
        trusted = trusted[0]
    elif trusted.ndim == 2:
        trusted = trusted.unsqueeze(0)
    if trusted.ndim != 3 or trusted.shape != value.shape:
        raise ValueError("observation_trusted_mask shape does not match disparity")
    return value.float(), trusted.to(dtype=torch.bool)


def _resize_mask(path: Path, target_hw: tuple[int, int], *, match: bool = False) -> np.ndarray:
    """Read a Spring map and normalize it to the image-resolution grid."""

    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if match and array.ndim == 3:
        array = np.any(array[..., : min(2, array.shape[-1])] > 0, axis=-1)
    else:
        if array.ndim == 3:
            array = array[..., 0]
        array = array > 0
    if array.shape == target_hw:
        return array.astype(bool, copy=False)
    # Ground-truth auxiliary maps are normally stored at 4K and averaged over
    # each 2x2 block by the official evaluator.  Nearest-neighbour would make
    # a single high-resolution detail bit disappear, so use a block mean for
    # exact 4K->HD semantics and nearest for any other compatible shape.
    h, w = target_hw
    if array.shape == (2 * h, 2 * w):
        return (
            array.reshape(h, 2, w, 2).mean(axis=(1, 3)) >= 0.5
        )
    source = torch.from_numpy(array.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(source, size=target_hw, mode="nearest")[0, 0]
    return resized.numpy().astype(bool, copy=False)


def _sequence_root(gt_path: Path) -> Path:
    # ``.../<sequence>/disp1_left/disp1_left_####.dsp5``.
    return gt_path.parent.parent


def _map_path(record: Any, name: str) -> Path:
    gt_path = Path(str(record.gt_disparity_path)).expanduser().resolve()
    sequence_root = _sequence_root(gt_path)
    frame_id = int(record.frame_id)
    return sequence_root / "maps" / name / f"{name}_{frame_id:04d}.png"


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"frames": len(rows)}
    numeric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {"frame_id", "timestamp", "image_pixel_count"}
        }
    )
    # EPE/rate fields are reduced over frames here.  The exact Spring website
    # evaluator uses pixel-weighted numerators; frame weighting is the safe
    # fallback when the compact per-frame helper does not retain numerators.
    for name in numeric_names:
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and math.isfinite(float(row[name]))
        ]
        result[name] = None if not values else float(np.mean(values))
    # Rates emitted by disparity_metrics can be made pixel-weighted exactly.
    image_pixels = sum(int(row.get("image_pixel_count", 0)) for row in rows)
    for name in ("negative_rate", "zero_rate", "invalid_rate"):
        if image_pixels:
            result[name] = sum(
                float(row.get(name, 0.0)) * int(row.get("image_pixel_count", 0))
                for row in rows
            ) / image_pixels
        else:
            result[name] = None
    for name in ("valid_count", "unmatched_count", "ffs_trusted_count"):
        result[name] = int(sum(int(row.get(name, 0) or 0) for row in rows))
    return result


def _write_csv(path: Path, aggregate: Mapping[str, Any]) -> None:
    fields = ["metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(aggregate):
            if key == "frames" or key.endswith("_count"):
                continue
            writer.writerow({"metric": key, "value": aggregate[key]})


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate(
    *,
    manifest_path: Path,
    observation_root: Path,
    output_dir: Path,
    seed: int = 42,
    start: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    records = load_manifest(manifest_path)
    if start < 0 or start >= len(records):
        raise ValueError(f"start={start} is outside manifest with {len(records)} records")
    selected = records[start:] if limit is None else records[start : start + limit]
    if not selected:
        raise ValueError("selection is empty")

    rows: list[dict[str, Any]] = []
    map_counts = {"detail": 0, "match": 0}
    for record in selected:
        cache_path = _cache_path(observation_root, record.sequence_id, record.frame_id)
        payload = load_cache_record(cache_path)
        disparity_lr, trusted_lr = _extract_observation(payload)
        gt = load_spring_disparity(record.gt_disparity_path or "", resolution="image", sign="positive")
        target_hw = tuple(int(value) for value in gt.shape)
        prediction = F.interpolate(
            torch.nan_to_num(disparity_lr[None], nan=0.0, posinf=0.0, neginf=0.0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        trusted_hr = F.interpolate(
            trusted_lr[None].float(), size=target_hw, mode="nearest"
        )[0, 0].numpy().astype(bool)
        detail_path = _map_path(record, "detailmap_disp1_left")
        match_path = _map_path(record, "matchmap_disp1_left")
        detail = None
        matched = None
        if detail_path.is_file():
            detail = _resize_mask(detail_path, target_hw)
            map_counts["detail"] += 1
        if match_path.is_file():
            matched = _resize_mask(match_path, target_hw, match=True)
            map_counts["match"] += 1
        metrics = disparity_metrics(
            prediction,
            gt,
            detail_mask=detail,
            match_mask=matched,
            ffs_trusted_mask=trusted_hr,
            ffs_prediction=prediction,
            boundary_mask=disparity_boundary_mask(
                torch.from_numpy(gt).float(),
                gradient_threshold_px=1.0,
                radius_px=1,
            ).numpy(),
        )
        row: dict[str, Any] = {
            "record_id": f"{record.sequence_id}/{record.frame_id}",
            "sequence_id": record.sequence_id,
            "frame_id": int(record.frame_id),
            "timestamp": float(record.timestamp),
            "image_pixel_count": int(gt.size),
            "detail_map_available": detail is not None,
            "match_map_available": matched is not None,
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                row[name] = None if isinstance(value, float) and not math.isfinite(value) else value
        rows.append(row)

    aggregate = _aggregate(rows)
    aggregate["detail_map_coverage"] = map_counts["detail"] / len(rows)
    aggregate["match_map_coverage"] = map_counts["match"] / len(rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "SCREENING_ONLY",
        "seed": int(seed),
        "arm": "S0",
        "target": {
            "type": "Spring_GT",
            "paper_accuracy": False,
            "disparity_unit": "full_hd_pixels",
            "resolution": "image",
        },
        "pose_source": "none",
        "temporal_metrics": {
            "rigid_temporal_residual_error": None,
            "non_rigid_temporal_residual_error": None,
        },
        "selection": {"start": start, "limit": limit, "records": len(rows)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "observation_root": str(observation_root.resolve()),
        "metrics": aggregate,
        "per_record": rows,
        "map_coverage": map_counts,
        "notes": [
            "S0 is the LR FFS observation bilinearly upsampled to the Spring image grid.",
            "High-detail and matched fields are valid only when the corresponding Spring maps are present.",
            "FFS trusted measurement error is computed on the nearest-upsampled trusted FFS mask.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "metrics.json", report)
    (output_dir / "per_record_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", aggregate)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-cache-root", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    report = evaluate(
        manifest_path=args.manifest.expanduser().resolve(),
        observation_root=args.observation_cache_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        seed=args.seed,
        start=args.start,
        limit=args.limit,
    )
    print(json.dumps({"status": report["status"], "metrics": str(args.output_dir / "metrics.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
