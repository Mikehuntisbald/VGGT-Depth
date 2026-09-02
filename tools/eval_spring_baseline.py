#!/usr/bin/env python3
"""Evaluate a frozen FFS Spring baseline on a common endpoint domain.

The regular :mod:`eval` entry point always loads a trainable checkpoint.  S0
is deliberately checkpoint-free, so this small evaluator reads a frozen FFS
observation (half-resolution) or teacher (full-resolution) cache, upsamples
its HR-pixel disparity to the Spring image grid, and compares it with the dense
Spring ``disp1_left`` ground truth.  Optional
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

from data.cache_dataset import CacheIdentity, load_cache_record, sha256_file  # noqa: E402
from data.endpoint_selection import (  # noqa: E402
    ENDPOINT_SELECTION_KIND,
    load_endpoint_index,
)
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


def _load_cache_lineage(
    root: Path,
    manifest_path: Path,
    *,
    role: str,
    arm: str,
) -> tuple[dict[str, Any], CacheIdentity]:
    """Load and validate the cache receipt before scoring any records.

    Baseline metrics are only useful when the frozen tensor source is
    attributable to the declared checkpoint/scale recipe.  In particular,
    F0 and F1 intentionally use different resolution contracts; accepting a
    manifest-matching cache with the wrong ``max_disp`` would silently mix
    those arms.
    """

    root = root.expanduser().resolve()
    receipt_path = root / "run_receipt.json"
    cache_manifest_path = root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not cache_manifest_path.is_file():
        raise FileNotFoundError(
            f"cache receipt and manifest are required under {root}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse cache receipt: {receipt_path}") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("cache receipt must be a JSON object")
    manifest_path = manifest_path.expanduser().resolve()
    manifest_sha256 = sha256_file(manifest_path)
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("cache receipt is bound to a different manifest")
    recorded_manifest = receipt.get("manifest")
    if recorded_manifest is not None and Path(str(recorded_manifest)).expanduser().resolve() != manifest_path:
        raise ValueError("cache receipt manifest path differs from evaluation manifest")
    expected_count = sum(
        1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    try:
        selected_count = int(receipt.get("selected_records", -1))
        written_count = int(receipt.get("written_records", -1))
        reused_count = int(receipt.get("reused_records", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("cache receipt record counts are malformed") from exc
    if selected_count != expected_count or written_count + reused_count != expected_count:
        raise ValueError("cache receipt does not cover the complete manifest")
    identity_value = receipt.get("identity")
    if not isinstance(identity_value, Mapping):
        raise ValueError("cache receipt identity is missing or malformed")
    expected_component = f"ffs-{role}"
    if identity_value.get("component") != expected_component:
        raise ValueError(
            f"cache component mismatch: expected {expected_component!r}, "
            f"got {identity_value.get('component')!r}"
        )
    required_identity = (
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "config_sha256",
    )
    if any(not isinstance(identity_value.get(name), str) for name in required_identity):
        raise ValueError("cache receipt identity fields are malformed")
    cuda_value = identity_value.get("cuda_version")
    if cuda_value is not None and not isinstance(cuda_value, str):
        raise ValueError("cache receipt cuda_version is malformed")
    config_value = receipt.get("config")
    if not isinstance(config_value, Mapping):
        raise ValueError("cache receipt config is missing or malformed")
    arm_name = str(arm).strip().upper()
    if role == "observation":
        if config_value.get("role") != "observation":
            raise ValueError("observation cache receipt role is malformed")
        expected_scale = 1 if arm_name == "F0" else 2 if arm_name == "F1" else None
        if expected_scale is not None:
            expected_resolution = (
                "full_resolution_observation" if expected_scale == 1 else "mvp"
            )
            if config_value.get("scale") != expected_scale:
                raise ValueError(
                    f"{arm_name} requires observation scale={expected_scale}"
                )
            if config_value.get("resolution_mode") != expected_resolution:
                raise ValueError(
                    f"{arm_name} requires resolution_mode={expected_resolution!r}"
                )
            expected_max_disp = 416 if expected_scale == 1 else 192
            if config_value.get("max_disp") != expected_max_disp:
                raise ValueError(
                    f"{arm_name} requires max_disp={expected_max_disp}"
                )
    elif role == "teacher":
        if config_value.get("teacher_source") != "Spring_GT":
            raise ValueError("teacher cache receipt is not Spring_GT")
    try:
        cache_rows = [
            json.loads(line)
            for line in cache_manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse cache manifest: {cache_manifest_path}") from exc
    if len(cache_rows) != expected_count:
        raise ValueError("cache manifest does not cover the complete manifest")
    for row in cache_rows:
        if not isinstance(row, Mapping):
            raise ValueError("cache manifest row is malformed")
        cache_path = Path(str(row.get("cache_path", ""))).expanduser().resolve()
        try:
            cache_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("cache manifest points outside its cache root") from exc
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
    identity = CacheIdentity(
        component=str(identity_value["component"]),
        upstream_commit=str(identity_value["upstream_commit"]),
        checkpoint_sha256=str(identity_value["checkpoint_sha256"]),
        torch_version=str(identity_value["torch_version"]),
        cuda_version=None if cuda_value is None else str(cuda_value),
        config_sha256=str(identity_value["config_sha256"]),
    )
    lineage = {
        "root": str(root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "identity": dict(identity_value),
        "config": dict(config_value),
    }
    return lineage, identity


def _extract_cache_prediction(
    payload: Mapping[str, Any], *, role: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one frozen-cache disparity/mask pair.

    ``cache_ffs.py`` uses the observation keys for the half-resolution FFS
    path and the teacher keys for the full-resolution path.  Keeping this
    distinction explicit prevents F0 (full-resolution FFS) from accidentally
    being scored as the F1 bilinear baseline.
    """

    if role not in {"observation", "teacher"}:
        raise ValueError("role must be 'observation' or 'teacher'")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError(f"{role} cache has no tensors mapping")
    prefix = "observation" if role == "observation" else "teacher"
    value = tensors.get(f"{prefix}_disparity_hr_px")
    trusted = tensors.get(f"{prefix}_trusted_mask")
    if not isinstance(value, torch.Tensor) or not isinstance(trusted, torch.Tensor):
        raise ValueError(f"{role} cache lacks {prefix}_disparity_hr_px")
    if value.ndim == 4 and tuple(value.shape[:2]) == (1, 1):
        value = value[0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            f"{prefix}_disparity_hr_px must resolve to [1,H,W], got "
            f"{tuple(value.shape)}"
        )
    if not value.is_floating_point():
        value = value.float()
    if trusted.ndim == 4 and tuple(trusted.shape[:2]) == (1, 1):
        trusted = trusted[0]
    elif trusted.ndim == 2:
        trusted = trusted.unsqueeze(0)
    if trusted.ndim != 3 or trusted.shape != value.shape:
        raise ValueError(f"{prefix}_trusted_mask shape does not match disparity")
    return value.float(), trusted.to(dtype=torch.bool)


def _extract_observation(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Backwards-compatible observation-only helper."""

    return _extract_cache_prediction(payload, role="observation")


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


def _fixed_crop_slices(
    target_hw: tuple[int, int],
    *,
    crop_size_hr_hw: tuple[int, int],
    crop_origin_hr_xy: tuple[int, int] | None,
    spatial_scale: int = 2,
) -> tuple[slice, slice, dict[str, int]]:
    """Resolve a deterministic, scale-aligned HR crop.

    Temporal F4--F6 evaluation uses the same fixed 384x768 crop to keep the
    Top-K hidden warp within a 24-GiB GPU.  Frozen F0/F1 must be scored on
    exactly that spatial domain when a crop is requested; the old full-image
    behavior remains the default for backwards compatibility.
    """

    if len(target_hw) != 2 or any(int(value) <= 0 for value in target_hw):
        raise ValueError(f"target_hw must contain positive dimensions, got {target_hw}")
    crop_h, crop_w = (int(crop_size_hr_hw[0]), int(crop_size_hr_hw[1]))
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("crop dimensions must be positive")
    height, width = (int(target_hw[0]), int(target_hw[1]))
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"crop {(crop_h, crop_w)} exceeds target {(height, width)}"
        )
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")
    if crop_origin_hr_xy is None:
        # Match the training datasets: center the crop and align its origin to
        # the LR sampling grid so bilinear FFS centers remain phase-consistent.
        x0 = ((width - crop_w) // 2 // spatial_scale) * spatial_scale
        y0 = ((height - crop_h) // 2 // spatial_scale) * spatial_scale
    else:
        x0, y0 = (int(crop_origin_hr_xy[0]), int(crop_origin_hr_xy[1]))
    if x0 < 0 or y0 < 0 or x0 + crop_w > width or y0 + crop_h > height:
        raise ValueError(
            f"crop origin {(x0, y0)} with size {(crop_h, crop_w)} is outside {(height, width)}"
        )
    if x0 % spatial_scale or y0 % spatial_scale:
        raise ValueError("crop origin must be aligned to spatial_scale")
    if crop_h % spatial_scale or crop_w % spatial_scale:
        raise ValueError("crop dimensions must be divisible by spatial_scale")
    return (
        slice(y0, y0 + crop_h),
        slice(x0, x0 + crop_w),
        {"x_hr": x0, "y_hr": y0, "width_hr": crop_w, "height_hr": crop_h},
    )


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frames": len(rows),
        "aggregation": (
            "global_numerator_count_pixel_weighted_with_legacy_frame_mean_fallback"
        ),
    }
    # New Spring rows retain a numerator/count pair for every metric domain.
    # Reduce those pairs globally; retain a frame-mean fallback for old cache
    # rows so historical reports remain readable and explicitly marked.
    scalar_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not key.endswith("_numerator")
            and not key.endswith("_count")
            and key not in {"frame_id", "timestamp", "image_pixel_count"}
        }
    )
    fallback_used = False
    for name in scalar_names:
        numerator_name = f"{name}_numerator"
        count_name = f"{name}_count"
        pair_rows = [
            row
            for row in rows
            if numerator_name in row or count_name in row
        ]
        if pair_rows:
            numerator = 0.0
            count = 0
            # A mixed legacy/new set cannot be reduced by silently dropping
            # the legacy rows.  Fall back to the explicit frame-mean path for
            # that metric unless every row carries its pair.
            valid_pairs = len(pair_rows) == len(rows)
            for row in pair_rows:
                raw_num = row.get(numerator_name)
                raw_count = row.get(count_name)
                # An empty per-frame domain is not an invalid aggregate; it
                # contributes no pixels (e.g. no matched/boundary pixels in a
                # particular Spring frame).  Skip the 0/NaN pair and retain
                # the other frames' exact numerators.
                if (
                    isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count == 0
                    and (
                        raw_num is None
                        or (
                            isinstance(raw_num, (int, float))
                            and not isinstance(raw_num, bool)
                            and (
                                not math.isfinite(float(raw_num))
                                or float(raw_num) == 0.0
                            )
                        )
                    )
                ):
                    continue
                if (
                    isinstance(raw_num, (int, float))
                    and not isinstance(raw_num, bool)
                    and math.isfinite(float(raw_num))
                    and isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count > 0
                ):
                    numerator += float(raw_num)
                    count += int(raw_count)
                else:
                    valid_pairs = False
            if valid_pairs and count > 0:
                result[name] = numerator / count
                result[numerator_name] = numerator
                result[count_name] = count
                continue
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and math.isfinite(float(row[name]))
        ]
        result[name] = None if not values else float(np.mean(values))
        result[numerator_name] = None
        result[count_name] = 0
        fallback_used = True
    # Explicit output-health rates are always defined over every output pixel;
    # retain a global pixel denominator even when a legacy row lacks pairs.
    image_pixels = sum(int(row.get("image_pixel_count", 0) or 0) for row in rows)
    result["image_pixel_count"] = image_pixels
    for name in ("negative_rate", "zero_rate", "invalid_rate"):
        if image_pixels:
            numerator = 0.0
            count = 0
            for row in rows:
                raw_num = row.get(f"{name}_numerator")
                raw_count = row.get(f"{name}_count")
                if (
                    isinstance(raw_num, (int, float))
                    and not isinstance(raw_num, bool)
                    and math.isfinite(float(raw_num))
                    and isinstance(raw_count, int)
                    and raw_count > 0
                ):
                    numerator += float(raw_num)
                    count += int(raw_count)
                else:
                    value = row.get(name)
                    pixels = int(row.get("image_pixel_count", 0) or 0)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        numerator += float(value) * pixels
                        count += pixels
                        fallback_used = True
            if count:
                result[name] = numerator / count
                result[f"{name}_numerator"] = numerator
                result[f"{name}_count"] = count
            else:
                result[name] = None
        else:
            result[name] = None
    for name in ("valid_count", "unmatched_count", "ffs_trusted_count"):
        result[name] = int(sum(int(row.get(name, 0) or 0) for row in rows))
    result["aggregation_fallback_used"] = fallback_used
    return result


def _write_csv(path: Path, aggregate: Mapping[str, Any]) -> None:
    fields = ["metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(aggregate):
            if (
                key in {"frames", "image_pixel_count"}
                or key.endswith("_count")
                or key.endswith("_numerator")
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
    cache_role: str = "observation",
    arm: str = "S0",
    endpoint_index_list: Path | None = None,
    crop_mode: str = "full",
    crop_size_hr_hw: tuple[int, int] = (384, 768),
    crop_origin_hr_xy: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if cache_role not in {"observation", "teacher"}:
        raise ValueError("cache_role must be 'observation' or 'teacher'")
    if not str(arm).strip():
        raise ValueError("arm must be non-empty")
    crop_mode = str(crop_mode).strip().lower()
    if crop_mode not in {"full", "fixed"}:
        raise ValueError("crop_mode must be 'full' or 'fixed'")
    if len(crop_size_hr_hw) != 2:
        raise ValueError("crop_size_hr_hw must contain (height,width)")
    crop_size_hr_hw = (int(crop_size_hr_hw[0]), int(crop_size_hr_hw[1]))
    if any(value <= 0 for value in crop_size_hr_hw):
        raise ValueError("crop_size_hr_hw values must be positive")
    if crop_origin_hr_xy is not None:
        if len(crop_origin_hr_xy) != 2:
            raise ValueError("crop_origin_hr_xy must contain (x,y)")
        crop_origin_hr_xy = (int(crop_origin_hr_xy[0]), int(crop_origin_hr_xy[1]))
        if any(value < 0 for value in crop_origin_hr_xy):
            raise ValueError("crop_origin_hr_xy values must be non-negative")
        if crop_mode != "fixed":
            raise ValueError("crop_origin_hr_xy requires crop_mode='fixed'")
    records = load_manifest(manifest_path)
    cache_lineage, cache_identity = _load_cache_lineage(
        observation_root,
        manifest_path,
        role=cache_role,
        arm=arm,
    )
    endpoint_selection = None
    if endpoint_index_list is not None:
        endpoint_selection = load_endpoint_index(
            endpoint_index_list.expanduser().resolve(),
            manifest_path=manifest_path,
        )
        selected = [records[index] for index in endpoint_selection.manifest_indices]
    else:
        selected = list(records)
    if start < 0 or start >= len(selected):
        raise ValueError(f"start={start} is outside selected manifest with {len(selected)} records")
    selected = selected[start:] if limit is None else selected[start : start + limit]
    if not selected:
        raise ValueError("selection is empty")

    rows: list[dict[str, Any]] = []
    map_counts = {"detail": 0, "match": 0}
    prediction_resolution: str | None = None
    resolved_crop_origins: set[tuple[int, int]] = set()
    for record in selected:
        cache_path = _cache_path(observation_root, record.sequence_id, record.frame_id)
        payload = load_cache_record(cache_path, expected_identity=cache_identity)
        source = payload.get("metadata", {}).get("source")
        if isinstance(source, Mapping):
            source_record = source.get("manifest_record")
            if isinstance(source_record, Mapping) and source_record != record.to_dict():
                raise ValueError(
                    f"cache source manifest record differs for "
                    f"{record.sequence_id}/{record.frame_id}"
                )
        disparity_lr, trusted_lr = _extract_cache_prediction(
            payload, role=cache_role
        )
        gt = load_spring_disparity(record.gt_disparity_path or "", resolution="image", sign="positive")
        target_hw = tuple(int(value) for value in gt.shape)
        current_resolution = (
            "full"
            if tuple(disparity_lr.shape[-2:]) == target_hw
            else "half"
        )
        if prediction_resolution is None:
            prediction_resolution = current_resolution
        elif prediction_resolution != current_resolution:
            raise ValueError("cache prediction resolution changed within a manifest")
        prediction = F.interpolate(
            torch.nan_to_num(disparity_lr[None], nan=0.0, posinf=0.0, neginf=0.0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        if tuple(trusted_lr.shape[-2:]) == target_hw:
            trusted_hr = trusted_lr[0].numpy().astype(bool)
        else:
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

        # Restrict every metric input to one identical HR window.  Crop after
        # cache upsampling and auxiliary-map normalization so full- and
        # half-resolution FFS observations use exactly the same pixel domain.
        if crop_mode == "fixed":
            y_slice, x_slice, crop_metadata = _fixed_crop_slices(
                target_hw,
                crop_size_hr_hw=crop_size_hr_hw,
                crop_origin_hr_xy=crop_origin_hr_xy,
                spatial_scale=2,
            )
            resolved_crop_origins.add(
                (int(crop_metadata["x_hr"]), int(crop_metadata["y_hr"]))
            )
            prediction = prediction[y_slice, x_slice]
            gt = gt[y_slice, x_slice]
            trusted_hr = trusted_hr[y_slice, x_slice]
            if detail is not None:
                detail = detail[y_slice, x_slice]
            if matched is not None:
                matched = matched[y_slice, x_slice]
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
        if crop_mode == "fixed":
            row["crop_origin_hr_xy"] = [
                int(crop_metadata["x_hr"]), int(crop_metadata["y_hr"])
            ]
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                row[name] = None if isinstance(value, float) and not math.isfinite(value) else value
        rows.append(row)

    if crop_mode == "fixed" and len(resolved_crop_origins) != 1:
        raise ValueError(
            "fixed evaluation resolved to multiple crop origins; "
            "use an explicit --crop-origin for a common domain"
        )

    aggregate = _aggregate(rows)
    aggregate["detail_map_coverage"] = map_counts["detail"] / len(rows)
    aggregate["match_map_coverage"] = map_counts["match"] / len(rows)
    common_domain_complete = bool(
        endpoint_selection is not None
        and endpoint_selection.kind == ENDPOINT_SELECTION_KIND
        and endpoint_selection.count == 1302
        and start == 0
        and len(rows) == endpoint_selection.count
    )
    full_validation_selection = bool(
        endpoint_selection is None
        and start == 0
        and limit is None
        and len(rows) == len(records)
    )
    coverage_complete = bool(common_domain_complete or full_validation_selection)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "SCREENING_ONLY",
        "seed": int(seed),
        "arm": str(arm),
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
        "selection": {
            "start": start,
            "limit": limit,
            "records": len(rows),
            "endpoint_index_list": (
                None
                if endpoint_selection is None
                else endpoint_selection.to_report(
                    available_endpoint_count=len(records),
                    evaluated_manifest_indices=(
                        endpoint_selection.manifest_indices[start:]
                        if limit is None
                        else endpoint_selection.manifest_indices[
                            start : start + limit
                        ]
                    ),
                )
            ),
        },
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "observation_root": str(observation_root.resolve()),
        "cache_lineage": cache_lineage,
        "cache_role": cache_role,
        "resolution_mode": prediction_resolution,
        "coverage_scope": (
            "common_domain"
            if common_domain_complete
            else "full_validation"
            if full_validation_selection
            else "limited_subset"
        ),
        "coverage_selection": (
            "common_domain"
            if common_domain_complete
            else "full_validation"
            if full_validation_selection
            else "limited_subset"
        ),
        "coverage_eligible": coverage_complete,
        "full_validation_selection": full_validation_selection,
        "common_domain_selection": common_domain_complete,
        "common_domain_complete": common_domain_complete,
        "common_domain_coverage_eligible": common_domain_complete,
        "common_domain_endpoint_count": (
            None if endpoint_selection is None else endpoint_selection.count
        ),
        "common_domain_expected_endpoint_count": 1302,
        "native_metric_contract": {
            "boundary_domain": (
                "native Spring GT disparity boundary mask; gradient threshold=1px, "
                "radius=1px"
            ),
            "boundary_pseudo_gt_override": False,
            "aggregation": (
                "global_numerator_count_pixel_weighted_with_legacy_frame_mean_fallback"
            ),
        },
        # Keep both the compact fields used by the trainable evaluator and a
        # self-describing contract for downstream lineage/audit tooling.
        "crop_mode": crop_mode,
        "hr_crop": list(crop_size_hr_hw) if crop_mode == "fixed" else None,
        "fixed_crop_origin_hr_xy": (
            None
            if crop_mode != "fixed"
            else sorted([list(origin) for origin in resolved_crop_origins])
        ),
        "crop_contract": {
            "evaluation_crop_mode": crop_mode,
            "size_hr_hw": list(crop_size_hr_hw) if crop_mode == "fixed" else None,
            "requested_origin_hr_xy": (
                None
                if crop_origin_hr_xy is None
                else list(crop_origin_hr_xy)
            ),
            "resolved_origins_hr_xy": sorted(
                [list(origin) for origin in resolved_crop_origins]
            ),
            "spatial_scale": 2,
            "all_metric_inputs_same_crop": (
                crop_mode == "full" or len(resolved_crop_origins) == 1
            ),
        },
        "metrics": aggregate,
        "per_record": rows,
        "map_coverage": map_counts,
        "notes": [
            (
                "Full-resolution FFS observation evaluated on the Spring image grid."
                if prediction_resolution == "full"
                else "Half-resolution FFS observation bilinearly upsampled to the Spring image grid."
            ),
            "High-detail and matched fields are valid only when the corresponding Spring maps are present.",
            "FFS trusted measurement error is computed on the nearest-upsampled trusted FFS mask.",
            (
                "All metric inputs were scored on the fixed HR crop."
                if crop_mode == "fixed"
                else "Metrics were scored on the full Spring image grid."
            ),
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
    parser.add_argument(
        "--cache-role",
        choices=("observation", "teacher"),
        default="observation",
        help=(
            "frozen cache tensor role; observation is the normal half/full FFS "
            "path, teacher reads teacher_disparity_hr_px"
        ),
    )
    parser.add_argument(
        "--arm",
        default="S0",
        help="label written to the screening report (for example F0 or F1)",
    )
    parser.add_argument(
        "--spring-endpoint-index-list",
        type=Path,
        help="optional manifest-bound common Spring endpoint selection",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("full", "fixed"),
        default="full",
        help="score the full Spring image or one deterministic fixed HR crop",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(384, 768),
        help="fixed HR crop size (default: 384 768)",
    )
    parser.add_argument(
        "--crop-origin",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="optional scale-aligned HR crop origin; default is center-aligned",
    )
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
        cache_role=args.cache_role,
        arm=args.arm,
        endpoint_index_list=args.spring_endpoint_index_list,
        crop_mode=args.crop_mode,
        crop_size_hr_hw=tuple(args.crop_size),
        crop_origin_hr_xy=(
            None if args.crop_origin is None else tuple(args.crop_origin)
        ),
    )
    print(json.dumps({"status": report["status"], "metrics": str(args.output_dir / "metrics.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
