#!/usr/bin/env python3
"""Batch-derive versioned metric geometry from raw VGGT cache manifests.

Every raw VGGT causal window is joined to the exact FFS observation record at
``FFS_ROOT/sequence_id/frame_id.pt``.  Identity, image lineage, target frame,
and causal ordering are validated before tensors are combined.  Existing
derived records are reusable only when their complete identity and source
mapping match.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.stereo_calibration import (
    RECTIFIED_CALIBRATION_CONTRACT,
    RectifiedCalibrationIndex,
    RectifiedCalibrationRecord,
    calibration_window_sha256,
    load_rectified_calibration_sidecar,
)
from geometry.pose_quality import validate_raw_cache_pair
from tools.derive_geometry_cache import (
    CALIBRATED_DERIVED_ALGORITHM,
    CALIBRATED_DERIVED_COMPONENT,
    CALIBRATED_DERIVED_SCHEMA_VERSION,
    DERIVED_SCHEMA_VERSION,
    GeometryThresholds,
    _source_metadata,
    derive_geometry,
)


@dataclass(frozen=True, slots=True)
class RawVGGTManifestEntry:
    """One validated row from ``vggt/cache_manifest.jsonl``."""

    selection_index: int
    target_manifest_index: int
    sequence_id: str
    frame_id: int
    timestamp: float
    cache_path: Path


DIAGNOSTIC_PATHS: dict[str, tuple[str, str]] = {
    "depth_weighted_mae_hr_px": ("depth_consistency", "weighted_mae_hr_px"),
    "depth_median_absolute_error_hr_px": (
        "depth_consistency",
        "median_absolute_error_hr_px",
    ),
    "depth_median_relative_error": (
        "depth_consistency",
        "median_relative_error",
    ),
    "baseline_coefficient_of_variation": (
        "baseline",
        "baseline_coefficient_of_variation",
    ),
    "stereo_rotation_error_max_deg": (
        "baseline",
        "stereo_rotation_error_max_deg",
    ),
    "photometric_median_absolute_rgb_residual": (
        "photometric",
        "median_absolute_rgb_residual",
    ),
    "photometric_valid_fraction": ("photometric", "valid_fraction"),
}


def _safe_component(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"cannot create a safe path component from {value!r}")
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _finite_diagnostics(quality: Mapping[str, Any]) -> dict[str, float]:
    """Extract every available finite scalar, including rejected windows."""

    result: dict[str, float] = {}
    for name, (section_name, value_name) in DIAGNOSTIC_PATHS.items():
        section = quality.get(section_name)
        value = section.get(value_name) if isinstance(section, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if torch.isfinite(torch.tensor(numeric)).item():
                result[name] = numeric
    return result


def _percentile_summary(values: Sequence[float], *, total_windows: int) -> dict[str, Any]:
    """Return deterministic linear percentiles without an optional dependency."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "available_count": 0,
            "missing_count": total_windows,
            "mean": None,
            "p00": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p100": None,
        }

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "available_count": len(ordered),
        "missing_count": total_windows - len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p00": percentile(0),
        "p05": percentile(5),
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p95": percentile(95),
        "p100": percentile(100),
    }


def audit_safe_zero_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reload every derived record and enforce validity/zero-fill invariants."""

    audited = 0
    pose_rejected_zero = 0
    static_rejected_zero = 0
    all_float_tensors_finite = 0
    calibrated_records = 0
    calibrated_pose_rejected_zero = 0
    for row in rows:
        cache_path = Path(str(row["cache_path"]))
        payload = load_cache_record(cache_path)
        tensors = payload["tensors"]
        quality = payload["metadata"].get("pose_quality")
        if not isinstance(quality, Mapping):
            raise CacheMismatchError(f"safe-zero audit: missing pose_quality in {cache_path}")
        alignment = quality.get("alignment")
        if not isinstance(alignment, Mapping):
            raise CacheMismatchError(f"safe-zero audit: missing alignment in {cache_path}")
        pose_tensor = tensors.get("temporal_pose_valid")
        static_tensor = tensors.get("static_prior_valid")
        if not isinstance(pose_tensor, torch.Tensor) or not isinstance(
            static_tensor, torch.Tensor
        ):
            raise CacheMismatchError(f"safe-zero audit: validity tensors missing in {cache_path}")
        pose_valid = bool(pose_tensor.item())
        static_valid = bool(static_tensor.item())
        if not (
            pose_valid is (quality.get("pose_valid") is True)
            and pose_valid is (row.get("pose_valid") is True)
            and static_valid is (alignment.get("static_prior_valid") is True)
            and static_valid is (row.get("static_prior_valid") is True)
        ):
            raise CacheMismatchError(
                f"safe-zero audit: manifest/metadata/tensor validity mismatch in {cache_path}"
            )
        floating = [
            value
            for value in tensors.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        if not all(bool(torch.isfinite(value).all().item()) for value in floating):
            raise CacheMismatchError(
                f"safe-zero audit: non-finite derived tensor in {cache_path}"
            )
        all_float_tensors_finite += 1
        temporal_extrinsics = tensors.get(
            "vggt_extrinsics_camera_from_world_metric_temporal"
        )
        if not isinstance(temporal_extrinsics, torch.Tensor):
            raise CacheMismatchError(
                f"safe-zero audit: temporal extrinsics missing in {cache_path}"
            )
        if not pose_valid:
            if bool(torch.count_nonzero(temporal_extrinsics).item()):
                raise CacheMismatchError(
                    f"safe-zero audit: rejected temporal pose is non-zero in {cache_path}"
                )
            pose_rejected_zero += 1
        component = payload.get("identity", {}).get("component")
        if component == CALIBRATED_DERIVED_COMPONENT:
            calibrated_records += 1
            stereo_valid = tensors.get("stereo_calibration_valid")
            transform = tensors.get("T_right_rectified_from_left_rectified_m")
            constrained = tensors.get(
                "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
            )
            if (
                not isinstance(stereo_valid, torch.Tensor)
                or stereo_valid.dtype != torch.bool
                or stereo_valid.numel() != 1
                or not bool(stereo_valid.item())
                or not isinstance(transform, torch.Tensor)
                or transform.shape != (4, 4)
                or not isinstance(constrained, torch.Tensor)
                or constrained.shape != (10, 3, 4)
            ):
                raise CacheMismatchError(
                    f"safe-zero audit: calibrated stereo tensors malformed in {cache_path}"
                )
            if not pose_valid:
                if bool(torch.count_nonzero(constrained).item()):
                    raise CacheMismatchError(
                        "safe-zero audit: rejected constrained pose is non-zero in "
                        f"{cache_path}"
                    )
                calibrated_pose_rejected_zero += 1
        if not static_valid:
            aligned = tensors.get("vggt_disparity_current_left_aligned_hr_px")
            aligned_confidence = tensors.get("vggt_aligned_confidence")
            aligned_mask = tensors.get("vggt_aligned_valid_mask")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (aligned, aligned_confidence, aligned_mask)
            ):
                raise CacheMismatchError(
                    f"safe-zero audit: static-prior tensors missing in {cache_path}"
                )
            if (
                bool(torch.count_nonzero(aligned).item())
                or bool(torch.count_nonzero(aligned_confidence).item())
                or bool(aligned_mask.any().item())
            ):
                raise CacheMismatchError(
                    f"safe-zero audit: rejected static prior is non-zero in {cache_path}"
                )
            static_rejected_zero += 1
        audited += 1
    result = {
        "passed": True,
        "records_audited": audited,
        "weights_only_safe_load_records": audited,
        "manifest_metadata_tensor_validity_consistent_records": audited,
        "all_float_tensors_finite_records": all_float_tensors_finite,
        "pose_rejected_zero_temporal_extrinsics": pose_rejected_zero,
        "static_rejected_zero_prior_tensors": static_rejected_zero,
    }
    if calibrated_records:
        result.update(
            {
                "calibrated_stereo_records": calibrated_records,
                "pose_rejected_zero_stereo_constrained_extrinsics": (
                    calibrated_pose_rejected_zero
                ),
            }
        )
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _receipt_selected_windows(path: Path) -> int | None:
    """Read completeness from an existing receipt or return None if absent."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["counts"]
        selected = counts["selected"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CacheMismatchError(
            f"existing canonical receipt is malformed; refusing overwrite: {path}"
        ) from exc
    if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise CacheMismatchError(
            f"existing canonical receipt has invalid selected count: {path}"
        )
    return selected


def load_raw_vggt_manifest(
    manifest_path: Path, *, vggt_root: Path
) -> list[RawVGGTManifestEntry]:
    """Load a canonical raw manifest and reject duplicates/path escapes."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw VGGT cache manifest is missing: {manifest_path}")
    entries: list[RawVGGTManifestEntry] = []
    seen_targets: set[tuple[str, int]] = set()
    seen_selection_indices: set[int] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {manifest_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest row {line_number} must be an object")
            try:
                selection_index = int(row["selection_index"])
                target_manifest_index = int(row["target_manifest_index"])
                sequence_id = str(row["sequence_id"])
                frame_id = int(row["frame_id"])
                timestamp = float(row["timestamp"])
                cache_path = Path(row["cache_path"]).expanduser().resolve()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed raw VGGT manifest row {line_number}: {row!r}"
                ) from exc
            if selection_index < 0 or target_manifest_index < 0 or not sequence_id:
                raise ValueError(f"invalid indices/sequence at manifest row {line_number}")
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"raw VGGT cache from row {line_number} is missing: {cache_path}"
                )
            if not _is_within(cache_path, vggt_root):
                raise CacheMismatchError(
                    f"raw VGGT manifest path escapes --vggt-root: {cache_path}"
                )
            target = (sequence_id, frame_id)
            if target in seen_targets:
                raise CacheMismatchError(f"duplicate VGGT target in manifest: {target}")
            if selection_index in seen_selection_indices:
                raise CacheMismatchError(
                    f"duplicate VGGT selection_index: {selection_index}"
                )
            seen_targets.add(target)
            seen_selection_indices.add(selection_index)
            entries.append(
                RawVGGTManifestEntry(
                    selection_index=selection_index,
                    target_manifest_index=target_manifest_index,
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    cache_path=cache_path,
                )
            )
    if not entries:
        raise ValueError("raw VGGT cache manifest is empty")
    if [item.selection_index for item in entries] != sorted(
        item.selection_index for item in entries
    ):
        raise CacheMismatchError("raw VGGT cache manifest is not selection-index ordered")
    return entries


def validate_causal_window_unchanged(
    vggt_payload: Mapping[str, Any], entry: RawVGGTManifestEntry
) -> None:
    """Make the no-future contract executable in the derived batch stage."""

    metadata = vggt_payload.get("metadata")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping) or source.get("causal") is not True:
        raise CacheMismatchError("raw VGGT cache does not assert causal=True")
    records = source.get("manifest_records")
    if not isinstance(records, list) or len(records) != 5:
        raise CacheMismatchError("raw VGGT cache must retain exactly five source records")
    sequence_ids = [record.get("sequence_id") for record in records]
    frame_ids = [record.get("frame_id") for record in records]
    timestamps = [record.get("timestamp") for record in records]
    if any(sequence_id != entry.sequence_id for sequence_id in sequence_ids):
        raise CacheMismatchError("raw VGGT causal window crosses sequence boundaries")
    try:
        frames_int = [int(value) for value in frame_ids]
        times_float = [float(value) for value in timestamps]
    except (TypeError, ValueError) as exc:
        raise CacheMismatchError("raw VGGT causal frame/timestamp values are malformed") from exc
    if any(current <= previous for previous, current in zip(frames_int, frames_int[1:])):
        raise CacheMismatchError("raw VGGT causal frame IDs are not strictly increasing")
    if any(current <= previous for previous, current in zip(times_float, times_float[1:])):
        raise CacheMismatchError("raw VGGT causal timestamps are not strictly increasing")
    if (
        frames_int[-1] != entry.frame_id
        or times_float[-1] != entry.timestamp
        or source.get("target_manifest_index") != entry.target_manifest_index
        or source.get("target_sequence_id") != entry.sequence_id
        or source.get("target_frame_id") != entry.frame_id
        or float(source.get("target_timestamp")) != entry.timestamp
    ):
        raise CacheMismatchError("raw manifest row and cached causal target disagree")
    if any(timestamp > entry.timestamp for timestamp in times_float):
        raise CacheMismatchError("raw VGGT causal window contains a future frame")


def _derived_config(
    thresholds: GeometryThresholds,
    *,
    cache_dtype: str,
    calibration_index: RectifiedCalibrationIndex | None = None,
) -> dict[str, Any]:
    calibrated = calibration_index is not None
    result: dict[str, Any] = {
        "schema_version": (
            CALIBRATED_DERIVED_SCHEMA_VERSION if calibrated else DERIVED_SCHEMA_VERSION
        ),
        "algorithm": (
            CALIBRATED_DERIVED_ALGORITHM
            if calibrated
            else "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
        ),
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "thresholds": thresholds.as_dict(),
        "cache_dtype": cache_dtype,
        "missing_diagnostic_policy": "invalid",
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    if calibration_index is not None:
        result["rectified_stereo_calibration"] = {
            "component": CALIBRATED_DERIVED_COMPONENT,
            "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
            "sidecar_path": str(calibration_index.sidecar_path),
            "sidecar_sha256": calibration_index.sidecar_sha256,
            "receipt_path": str(calibration_index.receipt_path),
            "receipt_sha256": calibration_index.receipt_sha256,
            "pixel_audit_path": str(calibration_index.pixel_audit_path),
            "pixel_audit_sha256": calibration_index.pixel_audit_sha256,
        }
    return result


def _derived_identity(
    vggt_payload: Mapping[str, Any],
    ffs_payload: Mapping[str, Any],
    *,
    vggt_cache_sha256: str,
    ffs_cache_sha256: str,
    config: Mapping[str, Any],
    calibration_window: Sequence[RectifiedCalibrationRecord] | None = None,
) -> CacheIdentity:
    vggt_identity = vggt_payload["identity"]
    ffs_identity = ffs_payload["identity"]
    return CacheIdentity(
        component=(
            CALIBRATED_DERIVED_COMPONENT
            if calibration_window is not None
            else "vggt-ffs-derived-geometry"
        ),
        upstream_commit=canonical_json_sha256(
            {
                "vggt": vggt_identity["upstream_commit"],
                "ffs": ffs_identity["upstream_commit"],
            }
        ),
        checkpoint_sha256=canonical_json_sha256(
            {
                "vggt_raw_cache_sha256": vggt_cache_sha256,
                "ffs_raw_cache_sha256": ffs_cache_sha256,
                **(
                    {
                        "rectified_calibration_window_sha256": (
                            calibration_window_sha256(calibration_window)
                        )
                    }
                    if calibration_window is not None
                    else {}
                ),
            }
        ),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256=canonical_json_sha256(dict(config)),
    )


def derive_geometry_manifest(
    *,
    vggt_root: Path,
    ffs_root: Path,
    output_root: Path,
    report_path: Path | None = None,
    thresholds: GeometryThresholds | None = None,
    cache_dtype: str = "float32",
    start_window: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
    rectified_calibration_sidecar: Path | None = None,
    rectified_calibration_receipt: Path | None = None,
) -> dict[str, Any]:
    """Derive a selected raw-manifest slice and return its run receipt."""

    vggt_root = vggt_root.expanduser().resolve()
    ffs_root = ffs_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not vggt_root.is_dir() or not ffs_root.is_dir():
        raise FileNotFoundError("--vggt-root and --ffs-root must be existing directories")
    if start_window < 0 or limit is not None and limit <= 0:
        raise ValueError("start-window must be non-negative and limit must be positive")
    if cache_dtype not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be float16 or float32")
    thresholds = thresholds or GeometryThresholds()
    thresholds.validate()
    if rectified_calibration_receipt is not None and (
        rectified_calibration_sidecar is None
    ):
        raise ValueError(
            "rectified_calibration_receipt requires rectified_calibration_sidecar"
        )
    calibration_index: RectifiedCalibrationIndex | None = None
    if rectified_calibration_sidecar is not None:
        calibration_index = load_rectified_calibration_sidecar(
            rectified_calibration_sidecar,
            receipt_path=rectified_calibration_receipt,
        )
    raw_manifest_path = vggt_root / "cache_manifest.jsonl"
    all_entries = load_raw_vggt_manifest(raw_manifest_path, vggt_root=vggt_root)
    raw_canonical_receipt_path = vggt_root / "run_receipt.json"
    canonical_vggt_identity: Mapping[str, Any] | None = None
    raw_canonical_receipt_sha256: str | None = None
    raw_canonical_contract_verified = False
    source_manifest_path: str | None = None
    source_manifest_sha256: str | None = None
    if raw_canonical_receipt_path.exists():
        try:
            raw_canonical_receipt = json.loads(
                raw_canonical_receipt_path.read_text(encoding="utf-8")
            )
            raw_selected = int(raw_canonical_receipt["selected_windows"])
            raw_written = int(raw_canonical_receipt["written_records"])
            raw_reused = int(raw_canonical_receipt["reused_records"])
            raw_available = int(raw_canonical_receipt["available_windows"])
            canonical_vggt_identity = raw_canonical_receipt["identity"]
            # Older test/fixture receipts may omit the optional source
            # manifest binding.  Parse it lazily so a count inconsistency is
            # reported as the stronger complete-coverage failure first; real
            # producer receipts always include both fields and are checked
            # below after coverage has passed.
            source_manifest_value = raw_canonical_receipt.get("manifest")
            source_manifest_sha_value = raw_canonical_receipt.get("manifest_sha256")
            source_manifest_path = (
                str(source_manifest_value)
                if isinstance(source_manifest_value, str)
                else None
            )
            source_manifest_sha256 = (
                str(source_manifest_sha_value)
                if isinstance(source_manifest_sha_value, str)
                else None
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheMismatchError(
                f"raw VGGT canonical receipt is malformed: {raw_canonical_receipt_path}"
            ) from exc
        if (
            raw_selected != len(all_entries)
            or raw_written + raw_reused != raw_selected
            or raw_available != raw_selected
            or not isinstance(canonical_vggt_identity, Mapping)
        ):
            raise CacheMismatchError(
                "raw VGGT canonical receipt does not prove complete manifest coverage"
            )
        if source_manifest_path is None or source_manifest_sha256 is None:
            raise CacheMismatchError(
                "raw VGGT canonical receipt is not bound to a valid source manifest"
            )
        source_manifest = Path(source_manifest_path).expanduser().resolve()
        if (
            not source_manifest.is_file()
            or source_manifest_sha256 != sha256_file(source_manifest)
        ):
            raise CacheMismatchError(
                "raw VGGT canonical receipt is not bound to a valid source manifest"
            )
        raw_canonical_receipt_sha256 = sha256_file(raw_canonical_receipt_path)
        raw_canonical_contract_verified = True
    selected = all_entries[start_window:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("selected raw VGGT cache window range is empty")

    config = _derived_config(
        thresholds,
        cache_dtype=cache_dtype,
        calibration_index=calibration_index,
    )
    tensor_dtype = torch.float16 if cache_dtype == "float16" else torch.float32
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    run_id = f"{started_wall.strftime('%Y%m%dT%H%M%S.%fZ')}-{time.time_ns()}"
    rows: list[dict[str, Any]] = []
    failure_histogram: Counter[str] = Counter()
    sequence_counts: dict[str, Counter[str]] = {}
    diagnostics_all: dict[str, list[float]] = {
        name: [] for name in DIAGNOSTIC_PATHS
    }
    diagnostics_pose_rejected: dict[str, list[float]] = {
        name: [] for name in DIAGNOSTIC_PATHS
    }
    pose_valid_count = 0
    static_prior_valid_count = 0
    written_count = 0
    reused_count = 0
    expected_vggt_identity: Mapping[str, Any] | None = canonical_vggt_identity
    expected_ffs_identity: Mapping[str, Any] | None = None
    raw_weights_only_safe_load_count = 0
    raw_identity_match_count = 0
    raw_causal_target_valid_count = 0
    raw_all_float_tensors_finite_count = 0

    for batch_index, entry in enumerate(selected, start=1):
        vggt_payload = load_cache_record(entry.cache_path)
        calibration_window: tuple[RectifiedCalibrationRecord, ...] | None = None
        if calibration_index is not None:
            vggt_metadata = vggt_payload.get("metadata")
            vggt_source = (
                vggt_metadata.get("source")
                if isinstance(vggt_metadata, Mapping)
                else None
            )
            if not isinstance(vggt_source, Mapping):
                raise CacheMismatchError(
                    f"raw VGGT source metadata is missing at {entry.cache_path}"
                )
            calibration_window = calibration_index.records_for_vggt_source(
                vggt_source
            )
        vggt_identity = vggt_payload["identity"]
        if expected_vggt_identity is None:
            expected_vggt_identity = vggt_identity
        if vggt_identity != expected_vggt_identity:
            raise CacheMismatchError(
                f"raw VGGT identity mismatch at {entry.cache_path}"
            )
        raw_weights_only_safe_load_count += 1
        raw_identity_match_count += 1
        validate_causal_window_unchanged(vggt_payload, entry)
        raw_causal_target_valid_count += 1
        if not all(
            not value.is_floating_point() or bool(torch.isfinite(value).all().item())
            for value in vggt_payload["tensors"].values()
        ):
            raise CacheMismatchError(
                f"raw VGGT cache contains non-finite tensors: {entry.cache_path}"
            )
        raw_all_float_tensors_finite_count += 1
        ffs_path = (
            ffs_root
            / _safe_component(entry.sequence_id)
            / f"{_safe_component(entry.frame_id)}.pt"
        )
        if not ffs_path.is_file():
            raise FileNotFoundError(
                f"matching FFS observation cache is missing for "
                f"{entry.sequence_id}/{entry.frame_id}: {ffs_path}"
            )
        ffs_payload = load_cache_record(ffs_path)
        ffs_identity = ffs_payload["identity"]
        if expected_ffs_identity is None:
            expected_ffs_identity = ffs_identity
        if ffs_identity != expected_ffs_identity:
            raise CacheMismatchError(f"FFS observation identity mismatch at {ffs_path}")
        linkage = validate_raw_cache_pair(vggt_payload, ffs_payload)
        if (
            linkage["target_sequence_id"] != entry.sequence_id
            or linkage["target_frame_id"] != entry.frame_id
            or linkage["target_timestamp"] != entry.timestamp
        ):
            raise CacheMismatchError("joined raw caches disagree with manifest target")
        vggt_cache_sha256 = sha256_file(entry.cache_path)
        ffs_cache_sha256 = sha256_file(ffs_path)
        source = _source_metadata(
            vggt_cache=entry.cache_path,
            ffs_cache=ffs_path,
            vggt_cache_sha256=vggt_cache_sha256,
            ffs_cache_sha256=ffs_cache_sha256,
            linkage=linkage,
            rectified_calibration_index=calibration_index,
            rectified_calibration_window=calibration_window,
        )
        identity = _derived_identity(
            vggt_payload,
            ffs_payload,
            vggt_cache_sha256=vggt_cache_sha256,
            ffs_cache_sha256=ffs_cache_sha256,
            config=config,
            calibration_window=calibration_window,
        )
        output_path = (
            output_root
            / _safe_component(entry.sequence_id)
            / f"{_safe_component(entry.frame_id)}.pt"
        )
        if output_path.exists() and not overwrite:
            existing = load_cache_record(output_path, expected_identity=identity)
            if existing["metadata"].get("source") != source:
                raise CacheMismatchError(
                    f"derived cache source mismatch for {entry.sequence_id}/{entry.frame_id}"
                )
            metadata = existing["metadata"]
            status = "reused_identity_and_source_match"
            reused_count += 1
        else:
            tensors, derived_metadata = derive_geometry(
                vggt_payload,
                ffs_payload,
                thresholds=thresholds,
                cache_dtype=tensor_dtype,
                rectified_calibration_window=calibration_window,
            )
            metadata = {"source": source, "config": config, **derived_metadata}
            save_cache_record(
                output_path,
                tensors=tensors,
                metadata=metadata,
                identity=identity,
            )
            status = "written"
            written_count += 1

        quality = metadata.get("pose_quality")
        if not isinstance(quality, Mapping):
            raise CacheMismatchError("derived record has no pose_quality metadata")
        pose_valid = quality.get("pose_valid") is True
        alignment = quality.get("alignment")
        static_prior_valid = (
            isinstance(alignment, Mapping)
            and alignment.get("static_prior_valid") is True
        )
        failure_reasons = quality.get("failure_reasons")
        if not isinstance(failure_reasons, list) or not all(
            isinstance(reason, str) for reason in failure_reasons
        ):
            raise CacheMismatchError("derived failure_reasons must be a string list")
        failure_histogram.update(failure_reasons)
        diagnostics = _finite_diagnostics(quality)
        for name, value in diagnostics.items():
            diagnostics_all[name].append(value)
            if not pose_valid:
                diagnostics_pose_rejected[name].append(value)
        pose_valid_count += int(pose_valid)
        static_prior_valid_count += int(static_prior_valid)
        per_sequence = sequence_counts.setdefault(entry.sequence_id, Counter())
        per_sequence["selected"] += 1
        per_sequence["pose_valid"] += int(pose_valid)
        per_sequence["pose_rejected"] += int(not pose_valid)
        per_sequence["static_prior_valid"] += int(static_prior_valid)
        per_sequence["static_prior_rejected"] += int(not static_prior_valid)
        rows.append(
            {
                "selection_index": entry.selection_index,
                "target_manifest_index": entry.target_manifest_index,
                "sequence_id": entry.sequence_id,
                "frame_id": entry.frame_id,
                "timestamp": entry.timestamp,
                "vggt_cache_path": str(entry.cache_path),
                "vggt_cache_sha256": vggt_cache_sha256,
                "ffs_cache_path": str(ffs_path),
                "ffs_cache_sha256": ffs_cache_sha256,
                "cache_path": str(output_path),
                "cache_sha256": sha256_file(output_path),
                "status": status,
                "pose_valid": pose_valid,
                "static_prior_valid": static_prior_valid,
                "failure_reasons": failure_reasons,
                "diagnostics": diagnostics,
                **(
                    {
                        "rectified_calibration_window_sha256": (
                            calibration_window_sha256(calibration_window)
                        )
                    }
                    if calibration_window is not None
                    else {}
                ),
            }
        )
        print(
            f"[{batch_index}/{len(selected)}] {entry.sequence_id}/{entry.frame_id} "
            f"status={status} pose_valid={pose_valid} "
            f"static_prior_valid={static_prior_valid}"
        )

    total = len(rows)
    cache_manifest_path = output_root / "cache_manifest.jsonl"
    canonical_receipt_path = output_root / "run_receipt.json"
    existing_canonical_selected = _receipt_selected_windows(canonical_receipt_path)
    preserve_more_complete_canonical = (
        existing_canonical_selected is not None
        and existing_canonical_selected > total
    )
    run_manifest_path = output_root / "run_manifests" / f"{run_id}.jsonl"
    _atomic_jsonl(run_manifest_path, rows)
    if preserve_more_complete_canonical:
        if not cache_manifest_path.is_file():
            raise CacheMismatchError(
                "a more-complete canonical receipt exists without its cache manifest"
            )
    else:
        _atomic_jsonl(cache_manifest_path, rows)
    safe_zero_audit = audit_safe_zero_contract(rows)
    elapsed_seconds = time.perf_counter() - started
    completed_wall = datetime.now(timezone.utc)
    receipt: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "component": (
            f"{CALIBRATED_DERIVED_COMPONENT}-batch"
            if calibration_index is not None
            else "vggt-ffs-derived-geometry-batch"
        ),
        "run_id": run_id,
        "started_at_utc": started_wall.isoformat(),
        "completed_at_utc": completed_wall.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config": config,
        "inputs": {
            "vggt_root": str(vggt_root),
            "vggt_cache_manifest": str(raw_manifest_path),
            "vggt_cache_manifest_sha256": sha256_file(raw_manifest_path),
            "manifest": source_manifest_path,
            "manifest_sha256": source_manifest_sha256,
            "vggt_available_windows": len(all_entries),
            "ffs_root": str(ffs_root),
            **(
                {
                    "rectified_calibration_sidecar": str(
                        calibration_index.sidecar_path
                    ),
                    "rectified_calibration_sidecar_sha256": (
                        calibration_index.sidecar_sha256
                    ),
                    "rectified_calibration_receipt": str(
                        calibration_index.receipt_path
                    ),
                    "rectified_calibration_receipt_sha256": (
                        calibration_index.receipt_sha256
                    ),
                    "pixel_rectification_audit": str(
                        calibration_index.pixel_audit_path
                    ),
                    "pixel_rectification_audit_sha256": (
                        calibration_index.pixel_audit_sha256
                    ),
                }
                if calibration_index is not None
                else {}
            ),
        },
        "raw_input_audit": {
            "passed": True,
            "canonical_receipt": (
                str(raw_canonical_receipt_path)
                if raw_canonical_receipt_path.exists()
                else None
            ),
            "canonical_receipt_sha256": raw_canonical_receipt_sha256,
            "canonical_receipt_complete_manifest_coverage": (
                raw_canonical_contract_verified
            ),
            "weights_only_safe_load_records": raw_weights_only_safe_load_count,
            "vggt_identity_match_records": raw_identity_match_count,
            "ffs_identity_match_records": len(rows),
            "causal_target_valid_records": raw_causal_target_valid_count,
            "all_float_tensors_finite_records": (
                raw_all_float_tensors_finite_count
            ),
            "vggt_identity": dict(expected_vggt_identity or {}),
            "ffs_identity": dict(expected_ffs_identity or {}),
        },
        "output": {
            "root": str(output_root),
            "cache_manifest": str(cache_manifest_path),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
            "run_cache_manifest": str(run_manifest_path),
            "run_cache_manifest_sha256": sha256_file(run_manifest_path),
        },
        "selection": {
            "start_window": start_window,
            "limit": limit,
            "selected_windows": total,
        },
        # Keep the source frame-manifest binding at the top level as well as
        # under inputs.  Consumers such as the Spring arm runner can then
        # reject a geometry cache generated from a different bounded split
        # without having to follow the raw VGGT receipt indirection.
        "manifest": source_manifest_path,
        "manifest_sha256": source_manifest_sha256,
        "counts": {
            "selected": total,
            "written": written_count,
            "reused": reused_count,
            "pose_valid": pose_valid_count,
            "pose_rejected": total - pose_valid_count,
            "static_prior_valid": static_prior_valid_count,
            "static_prior_rejected": total - static_prior_valid_count,
        },
        "rates": {
            "pose_valid": pose_valid_count / total,
            "pose_rejected": (total - pose_valid_count) / total,
            "static_prior_valid": static_prior_valid_count / total,
            "static_prior_rejected": (total - static_prior_valid_count) / total,
        },
        "by_sequence": {
            sequence_id: {
                "counts": {
                    name: counts[name]
                    for name in (
                        "selected",
                        "pose_valid",
                        "pose_rejected",
                        "static_prior_valid",
                        "static_prior_rejected",
                    )
                },
                "rates": {
                    "pose_valid": counts["pose_valid"] / counts["selected"],
                    "pose_rejected": counts["pose_rejected"] / counts["selected"],
                    "static_prior_valid": (
                        counts["static_prior_valid"] / counts["selected"]
                    ),
                    "static_prior_rejected": (
                        counts["static_prior_rejected"] / counts["selected"]
                    ),
                },
            }
            for sequence_id, counts in sorted(sequence_counts.items())
        },
        "failure_reason_histogram": dict(sorted(failure_histogram.items())),
        "diagnostic_percentiles": {
            name: {
                "all_available_windows": _percentile_summary(
                    diagnostics_all[name], total_windows=total
                ),
                "pose_rejected_available_windows": _percentile_summary(
                    diagnostics_pose_rejected[name],
                    total_windows=total - pose_valid_count,
                ),
            }
            for name in DIAGNOSTIC_PATHS
        },
        "quality_gate_semantics": {
            "depth_median_relative_error": {
                "enabled": thresholds.max_depth_median_relative_error is not None,
                "threshold": thresholds.max_depth_median_relative_error,
                "diagnostic_is_still_aggregated_when_gate_disabled": True,
            }
        },
        "safe_zero_audit": safe_zero_audit,
        "claim_policy": (
            "Counts are observed gate results; rejected windows remain explicit and "
            "are never reported as passes."
        ),
    }
    preserved_receipt_path = output_root / "run_receipts" / f"{run_id}.json"
    receipt["output"]["canonical_run_receipt"] = str(canonical_receipt_path)
    receipt["output"]["preserved_run_receipt"] = str(preserved_receipt_path)
    if thresholds == GeometryThresholds():
        verification_arguments = [
            sys.executable,
            str((PROJECT_ROOT / "tools" / "derive_geometry_manifest.py").resolve()),
            "--vggt-root",
            str(vggt_root),
            "--ffs-root",
            str(ffs_root),
            "--output",
            str(output_root),
            "--cache-dtype",
            cache_dtype,
            "--start-window",
            str(start_window),
        ]
        if limit is not None:
            verification_arguments.extend(("--limit", str(limit)))
        if report_path is not None:
            verification_arguments.extend(
                ("--report", str(report_path.expanduser().resolve()))
            )
        if calibration_index is not None:
            verification_arguments.extend(
                (
                    "--rectified-calibration-sidecar",
                    str(calibration_index.sidecar_path),
                    "--rectified-calibration-receipt",
                    str(calibration_index.receipt_path),
                )
            )
        receipt["safe_zero_audit"]["verification_command"] = shlex.join(
            verification_arguments
        )
    else:
        receipt["safe_zero_audit"]["verification_command"] = None
        receipt["safe_zero_audit"]["verification_note"] = (
            "Non-default test thresholds were supplied through the Python API."
        )
    receipt["canonical_update"] = {
        "existing_selected_windows": existing_canonical_selected,
        "current_selected_windows": total,
        "status": (
            "preserved_more_complete_existing"
            if preserve_more_complete_canonical
            else "updated_with_current_run"
        ),
    }
    _atomic_json(preserved_receipt_path, receipt)
    if not preserve_more_complete_canonical:
        _atomic_json(canonical_receipt_path, receipt)
    if report_path is not None:
        resolved_report = report_path.expanduser().resolve()
        existing_report_selected = _receipt_selected_windows(resolved_report)
        if existing_report_selected is None or existing_report_selected <= total:
            _atomic_json(resolved_report, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-derive metric geometry for a raw VGGT cache manifest."
    )
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--ffs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rectified-calibration-sidecar", type=Path)
    parser.add_argument("--rectified-calibration-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = derive_geometry_manifest(
        vggt_root=args.vggt_root,
        ffs_root=args.ffs_root,
        output_root=args.output,
        report_path=args.report,
        cache_dtype=args.cache_dtype,
        start_window=args.start_window,
        limit=args.limit,
        overwrite=args.overwrite,
        rectified_calibration_sidecar=args.rectified_calibration_sidecar,
        rectified_calibration_receipt=args.rectified_calibration_receipt,
    )
    print(
        json.dumps(
            {
                "selected": receipt["counts"]["selected"],
                "written": receipt["counts"]["written"],
                "reused": receipt["counts"]["reused"],
                "pose_valid": receipt["counts"]["pose_valid"],
                "pose_rejected": receipt["counts"]["pose_rejected"],
                "static_prior_valid": receipt["counts"]["static_prior_valid"],
                "failure_reason_histogram": receipt["failure_reason_histogram"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
