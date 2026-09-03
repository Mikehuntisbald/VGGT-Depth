#!/usr/bin/env python3
"""Evaluate the Spring F0/F1 frozen-FFS baselines.

The regular :mod:`eval` entry point always loads a trainable checkpoint.  F0
and F1 are deliberately checkpoint-free, so this evaluator reads a frozen FFS
observation cache and compares it with dense Spring ``disp1_left`` ground
truth. F0 consumes full-resolution FFS directly; F1 bilinearly reconstructs a
half-resolution FFS observation. A manifest-bound endpoint list and explicit
fixed crop put both arms on the same domain as the temporal arms.

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
import time
from collections.abc import Mapping
from dataclasses import dataclass
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
from data.endpoint_selection import EndpointSelection, load_endpoint_index  # noqa: E402
from data.manifest import load_manifest  # noqa: E402
from data.spring import (  # noqa: E402
    SPRING_GT_COMPONENT,
    SPRING_GT_TARGET_TYPE,
    load_spring_disparity,
)
from metrics.spring_arms import disparity_metrics  # noqa: E402
from metrics.boundary import disparity_boundary_mask  # noqa: E402
from utils.checkpoint import repository_git_hash  # noqa: E402


COMMON_CROP_ORIGIN_XY = (576, 348)
COMMON_CROP_SIZE_HW = (384, 768)


@dataclass(frozen=True, slots=True)
class BaselineMode:
    name: str
    arm: str
    scale: int
    cache_component: str
    reconstruction: str


BASELINE_MODES = {
    "full": BaselineMode(
        name="full",
        arm="F0",
        scale=1,
        cache_component="ffs-observation-full-resolution",
        reconstruction="identity",
    ),
    "half": BaselineMode(
        name="half",
        arm="F1",
        scale=2,
        cache_component="ffs-observation",
        reconstruction="bilinear_align_corners_false",
    ),
}


METRIC_DENOMINATORS = {
    "overall_epe": "valid_count",
    "overall_1px": "valid_count",
    "high_detail_epe": "high_detail_count",
    "high_detail_1px": "high_detail_count",
    "low_detail_epe": "low_detail_count",
    "low_detail_1px": "low_detail_count",
    "matched_epe": "matched_count",
    "matched_1px": "matched_count",
    "unmatched_completion_1px": "unmatched_count",
    "unmatched_completion_2px": "unmatched_count",
    "boundary_epe": "boundary_count",
    "ffs_trusted_measurement_error": "ffs_trusted_count",
    "negative_rate": "image_pixel_count",
    "zero_rate": "image_pixel_count",
    "invalid_rate": "image_pixel_count",
}


def _safe(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError(f"invalid path component: {value!r}")
    return text


def _cache_path(root: Path, sequence_id: str, frame_id: int) -> Path:
    return root / _safe(sequence_id) / f"{_safe(frame_id)}.pt"


def _baseline_mode(name: str) -> BaselineMode:
    try:
        return BASELINE_MODES[name]
    except KeyError as exc:
        raise ValueError(
            f"baseline mode must be one of {sorted(BASELINE_MODES)}"
        ) from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache_lineage(
    root: Path,
    *,
    manifest_path: Path,
    mode: BaselineMode,
) -> dict[str, Any]:
    """Load and strictly bind an F0/F1 cache receipt to this evaluation."""

    receipt_path = root / "run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"observation cache receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read observation cache receipt: {receipt_path}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("observation cache receipt must be a JSON object")
    expected_manifest_sha = sha256_file(manifest_path)
    if receipt.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError("observation cache receipt is bound to a different manifest")
    identity = receipt.get("identity")
    config = receipt.get("config")
    if not isinstance(identity, Mapping) or not isinstance(config, Mapping):
        raise ValueError("observation cache receipt lacks identity/config mappings")
    if identity.get("component") != mode.cache_component:
        raise ValueError(
            "observation cache component does not match baseline mode: "
            f"expected {mode.cache_component!r}, got {identity.get('component')!r}"
        )
    if config.get("role") != "observation" or config.get("scale") != mode.scale:
        raise ValueError(
            "observation cache role/scale does not match baseline mode: "
            f"expected observation/scale={mode.scale}"
        )
    declared_mode = config.get("resolution_mode")
    if declared_mode is not None and declared_mode != mode.name:
        raise ValueError(
            "observation cache resolution_mode does not match evaluator mode: "
            f"{declared_mode!r} vs {mode.name!r}"
        )
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "identity": dict(identity),
        "config": dict(config),
    }


def _extract_observation(
    payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _observation_on_image_grid(
    disparity: torch.Tensor,
    trusted: torch.Tensor,
    *,
    target_hw: tuple[int, int],
    mode: BaselineMode,
) -> tuple[np.ndarray, np.ndarray]:
    expected_hw = tuple(int(value // mode.scale) for value in target_hw)
    if any(value % mode.scale for value in target_hw):
        raise ValueError(
            f"target shape {target_hw} is not divisible by baseline scale {mode.scale}"
        )
    if tuple(disparity.shape[-2:]) != expected_hw:
        raise ValueError(
            f"{mode.name}-resolution cache grid must be {expected_hw}, got "
            f"{tuple(disparity.shape[-2:])}"
        )
    finite = disparity[None]
    if mode.scale == 1:
        prediction = finite[0, 0]
        trusted_image = trusted[0]
    else:
        prediction = F.interpolate(
            finite,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        trusted_image = F.interpolate(
            trusted[None].float(), size=target_hw, mode="nearest"
        )[0, 0].to(dtype=torch.bool)
    return prediction.numpy(), trusted_image.numpy().astype(bool, copy=False)


def _crop_xywh(
    *,
    crop_mode: str,
    crop_origin_xy: tuple[int, int] | None,
    crop_size_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if crop_mode == "full":
        if crop_origin_xy is not None:
            raise ValueError("crop-origin requires crop-mode=fixed")
        return None
    if crop_mode != "fixed":
        raise ValueError("crop-mode must be fixed or full")
    origin = COMMON_CROP_ORIGIN_XY if crop_origin_xy is None else crop_origin_xy
    if len(origin) != 2 or len(crop_size_hw) != 2:
        raise ValueError("crop origin/size must each contain two integers")
    x, y = (int(value) for value in origin)
    height, width = (int(value) for value in crop_size_hw)
    image_height, image_width = image_hw
    if x < 0 or y < 0 or height <= 0 or width <= 0:
        raise ValueError("crop origin must be non-negative and crop size positive")
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"fixed crop {(x, y, width, height)} is outside image shape {image_hw}"
        )
    return x, y, width, height


def _crop_array(
    value: np.ndarray, crop: tuple[int, int, int, int] | None
) -> np.ndarray:
    if crop is None:
        return value
    x, y, width, height = crop
    return value[y : y + height, x : x + width]


def _resize_mask(
    path: Path, target_hw: tuple[int, int], *, match: bool = False
) -> np.ndarray:
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
        return array.reshape(h, 2, w, 2).mean(axis=(1, 3)) >= 0.5
    source = torch.from_numpy(array.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(source, size=target_hw, mode="nearest")[0, 0]
    return resized.numpy().astype(bool, copy=False)


def _sequence_root(gt_path: Path) -> Path:
    # ``.../<sequence>/disp1_left/disp1_left_####.dsp5``.
    return gt_path.parent.parent


def _manifest_relative_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _map_path(record: Any, name: str, *, manifest_path: Path) -> Path:
    gt_path = _manifest_relative_path(str(record.gt_disparity_path), manifest_path)
    sequence_root = _sequence_root(gt_path)
    frame_id = int(record.frame_id)
    return sequence_root / "maps" / name / f"{name}_{frame_id:04d}.png"


def _support_counts(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    detail_mask: np.ndarray | None,
    match_mask: np.ndarray | None,
    trusted_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> dict[str, int]:
    gt_support = np.isfinite(ground_truth) & (ground_truth > 0)
    valid = gt_support & np.isfinite(prediction)
    counts = {
        "image_pixel_count": int(ground_truth.size),
        "valid_count": int(valid.sum()),
        "high_detail_count": 0,
        "low_detail_count": 0,
        "matched_count": 0,
        "unmatched_count": 0,
        "boundary_count": int((valid & boundary_mask).sum()),
        "ffs_trusted_count": int(
            (gt_support & trusted_mask & np.isfinite(prediction)).sum()
        ),
    }
    if detail_mask is not None:
        counts["high_detail_count"] = int((valid & detail_mask).sum())
        counts["low_detail_count"] = int((valid & ~detail_mask).sum())
    if match_mask is not None:
        counts["matched_count"] = int((valid & match_mask).sum())
        # Completion keeps invalid predictions in the denominator.
        counts["unmatched_count"] = int((gt_support & ~match_mask).sum())
    return counts


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty Spring baseline result")
    result: dict[str, Any] = {
        "frames": len(rows),
        "aggregation": "global_numerator_over_pixel_count",
    }
    numerators: dict[str, float | None] = {}
    denominators: dict[str, int] = {}
    for metric, count_name in METRIC_DENOMINATORS.items():
        numerator = 0.0
        denominator = 0
        for row in rows:
            value = row.get(metric)
            count = row.get(count_name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            ):
                numerator += float(value) * count
                denominator += count
        result[metric] = None if denominator == 0 else numerator / denominator
        numerators[metric] = None if denominator == 0 else numerator
        denominators[metric] = denominator
    count_names = sorted(set(METRIC_DENOMINATORS.values()))
    for name in count_names:
        result[name] = int(sum(int(row.get(name, 0) or 0) for row in rows))
    result["numerators"] = numerators
    result["denominators"] = denominators
    return result


def _write_csv(path: Path, aggregate: Mapping[str, Any]) -> None:
    fields = ["metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(aggregate):
            if (
                key == "frames"
                or key.endswith("_count")
                or not isinstance(aggregate[key], (int, float, type(None)))
            ):
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
    mode: str = "half",
    endpoint_index_list: Path | None = None,
    crop_mode: str = "full",
    crop_origin: tuple[int, int] | None = None,
    crop_size: tuple[int, int] = COMMON_CROP_SIZE_HW,
) -> dict[str, Any]:
    started = time.perf_counter()
    if seed < 0:
        raise ValueError("seed must be non-negative")
    baseline_mode = _baseline_mode(mode)
    records = load_manifest(manifest_path)
    lineage = _load_cache_lineage(
        observation_root, manifest_path=manifest_path, mode=baseline_mode
    )
    if start < 0 or start > len(records):
        raise ValueError(
            f"start={start} is outside manifest with {len(records)} records"
        )
    selected_indices: list[int]
    endpoint_selection: EndpointSelection | None = None
    if endpoint_index_list is not None:
        endpoint_selection = load_endpoint_index(
            endpoint_index_list, manifest_path=manifest_path
        )
        selected_indices = list(endpoint_selection.manifest_indices)
        if start:
            selected_indices = selected_indices[start:]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            selected_indices = selected_indices[:limit]
    else:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        selected_indices = list(range(start, len(records)))
        if limit is not None:
            selected_indices = selected_indices[:limit]
    selected = [records[index] for index in selected_indices]
    if not selected:
        raise ValueError("selection is empty")

    rows: list[dict[str, Any]] = []
    map_counts = {"detail": 0, "match": 0}
    for record in selected:
        cache_path = _cache_path(observation_root, record.sequence_id, record.frame_id)
        payload = load_cache_record(cache_path)
        if payload.get("identity") != lineage["identity"]:
            raise ValueError(
                f"cache record identity differs from run receipt: {cache_path}"
            )
        disparity_lr, trusted_lr = _extract_observation(payload)
        gt_path = _manifest_relative_path(record.gt_disparity_path or "", manifest_path)
        gt = load_spring_disparity(gt_path, resolution="image", sign="positive")
        target_hw_full = tuple(int(value) for value in gt.shape)
        prediction, trusted_hr = _observation_on_image_grid(
            disparity_lr,
            trusted_lr,
            target_hw=target_hw_full,
            mode=baseline_mode,
        )
        crop = _crop_xywh(
            crop_mode=crop_mode,
            crop_origin_xy=crop_origin,
            crop_size_hw=crop_size,
            image_hw=target_hw_full,
        )
        gt = _crop_array(gt, crop)
        prediction = _crop_array(prediction, crop)
        trusted_hr = _crop_array(trusted_hr, crop)
        target_hw = tuple(int(value) for value in gt.shape)
        detail_path = _map_path(
            record, "detailmap_disp1_left", manifest_path=manifest_path
        )
        match_path = _map_path(
            record, "matchmap_disp1_left", manifest_path=manifest_path
        )
        detail = None
        matched = None
        if detail_path.is_file():
            detail = _resize_mask(detail_path, target_hw_full)
            detail = _crop_array(detail, crop)
            map_counts["detail"] += 1
        if match_path.is_file():
            matched = _resize_mask(match_path, target_hw_full, match=True)
            matched = _crop_array(matched, crop)
            map_counts["match"] += 1
        boundary = disparity_boundary_mask(
            torch.from_numpy(gt).float(),
            gradient_threshold_px=1.0,
            radius_px=1,
        ).numpy()
        metrics = disparity_metrics(
            prediction,
            gt,
            detail_mask=detail,
            match_mask=matched,
            ffs_trusted_mask=trusted_hr,
            ffs_prediction=prediction,
            boundary_mask=boundary,
        )
        if detail is None:
            for name in (
                "high_detail_epe",
                "high_detail_1px",
                "low_detail_epe",
                "low_detail_1px",
            ):
                metrics[name] = float("nan")
        if matched is None:
            for name in (
                "matched_epe",
                "matched_1px",
                "unmatched_completion_1px",
                "unmatched_completion_2px",
            ):
                metrics[name] = float("nan")
        metrics.update(
            _support_counts(
                prediction,
                gt,
                detail_mask=detail,
                match_mask=matched,
                trusted_mask=trusted_hr,
                boundary_mask=boundary,
            )
        )
        row: dict[str, Any] = {
            "record_id": f"{record.sequence_id}/{record.frame_id}",
            "manifest_index": int(selected_indices[len(rows)]),
            "sequence_id": record.sequence_id,
            "frame_id": int(record.frame_id),
            "timestamp": float(record.timestamp),
            "detail_map_available": detail is not None,
            "match_map_available": matched is not None,
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                row[name] = (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
        rows.append(row)

    aggregate = _aggregate(rows)
    aggregate["detail_map_coverage"] = map_counts["detail"] / len(rows)
    aggregate["match_map_coverage"] = map_counts["match"] / len(rows)
    endpoint_report = (
        None
        if endpoint_selection is None
        else endpoint_selection.to_report(
            available_endpoint_count=len(records),
            evaluated_manifest_indices=selected_indices,
        )
    )
    resolved_crop = _crop_xywh(
        crop_mode=crop_mode,
        crop_origin_xy=crop_origin,
        crop_size_hw=crop_size,
        image_hw=target_hw_full,
    )
    evaluation_lineage = {
        "manifest_sha256": sha256_file(manifest_path),
        "endpoint_id_sha256": (
            None
            if endpoint_report is None
            else endpoint_report["evaluated_endpoint_id_sha256"]
        ),
        "endpoint_count": len(selected_indices),
        "crop_mode": crop_mode,
        "crop_hr_xywh": None if resolved_crop is None else list(resolved_crop),
        "baseline_mode": baseline_mode.name,
        "cache_identity": lineage["identity"],
    }
    evaluator_provenance = {
        "git_hash": repository_git_hash(PROJECT_ROOT),
        "eval_py_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_module_sha256": sha256_file(
            (SRC_ROOT / "metrics/spring_arms.py").resolve()
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    elapsed_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "SCREENING_ONLY",
        "seed": int(seed),
        "arm": baseline_mode.arm,
        "baseline_mode": baseline_mode.name,
        "resolution_contract": {
            "scale": baseline_mode.scale,
            "cache_component": baseline_mode.cache_component,
            "reconstruction": baseline_mode.reconstruction,
            "max_disp_hr_equivalent_px": lineage["config"].get(
                "max_disp_hr_equivalent_px"
            ),
        },
        "target": {
            "type": SPRING_GT_TARGET_TYPE,
            "component": SPRING_GT_COMPONENT,
            "paper_gt": True,
            "synthetic_ground_truth": True,
            "paper_accuracy": False,
            "disparity_unit": "full_hd_pixels",
            "resolution": "image",
        },
        "evaluator": evaluator_provenance,
        "device": "cpu",
        "elapsed_seconds": elapsed_seconds,
        "pose_source": "none",
        "temporal_metrics": {
            "rigid_temporal_residual_error": None,
            "non_rigid_temporal_residual_error": None,
        },
        "selection": {
            "start": start,
            "limit": limit,
            "records": len(rows),
            "first_manifest_index": selected_indices[0],
            "last_manifest_index": selected_indices[-1],
            "endpoint_index_list": endpoint_report,
            "crop_mode": crop_mode,
            "crop_origin_xy": (
                None if resolved_crop is None else list(resolved_crop[:2])
            ),
            "crop_size_hw": (
                None if resolved_crop is None else [resolved_crop[3], resolved_crop[2]]
            ),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "observation_root": str(observation_root.resolve()),
        "observation_lineage": lineage,
        "evaluation_lineage": evaluation_lineage,
        "evaluation_lineage_sha256": _canonical_sha256(evaluation_lineage),
        "metrics": aggregate,
        "per_record": rows,
        "map_coverage": map_counts,
        "notes": [
            (
                "F0 is full-resolution FFS with identity reconstruction."
                if baseline_mode.name == "full"
                else "F1 is the half-resolution FFS observation bilinearly upsampled to the Spring image grid."
            ),
            "This SCREENING_ONLY report does not claim official Spring benchmark accuracy.",
            "High-detail and matched fields are valid only when the corresponding Spring maps are present.",
            "FFS trusted measurement error is computed on the nearest-upsampled trusted FFS mask.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "metrics.json", report)
    (output_dir / "per_record_metrics.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", aggregate)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-cache-root", type=Path, required=True)
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir", type=Path, required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(BASELINE_MODES)),
        default="half",
        help="full (F0) or half (F1; default) observation grid",
    )
    parser.add_argument(
        "--endpoint-index-list",
        "--spring-endpoint-index-list",
        dest="endpoint_index_list",
        type=Path,
        help="manifest-bound common endpoint JSON",
    )
    parser.add_argument("--crop-mode", choices=("full", "fixed"), default="full")
    parser.add_argument(
        "--crop-origin",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="fixed crop origin in model/HD pixels (default 576 348)",
    )
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=COMMON_CROP_SIZE_HW,
        help="fixed crop size in model/HD pixels (default 384 768)",
    )
    args = parser.parse_args(argv)
    report = evaluate(
        manifest_path=args.manifest.expanduser().resolve(),
        observation_root=args.observation_cache_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        seed=args.seed,
        start=args.start,
        limit=args.limit,
        mode=args.mode,
        endpoint_index_list=(
            None
            if args.endpoint_index_list is None
            else args.endpoint_index_list.expanduser().resolve()
        ),
        crop_mode=args.crop_mode,
        crop_origin=(None if args.crop_origin is None else tuple(args.crop_origin)),
        crop_size=tuple(args.crop_size),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": str(args.output_dir / "metrics.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
