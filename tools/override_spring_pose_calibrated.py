#!/usr/bin/env python3
"""Create a calibrated-v2 Spring GT-pose view of a derived geometry cache.

The producer cache owns the FastFS/VGGT depth and alignment tensors.  This
adapter replaces only the temporal pose with the exact Spring ``cam_data``
pose and composes the right-camera views through the active calibrated
stereo sidecar.  It is intentionally separate from the legacy pose override
so a GT-pose arm can never overwrite or masquerade as a VGGT-pose cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.cache_dataset import (  # noqa: E402
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import ManifestRecord, load_manifest  # noqa: E402
from data.stereo_calibration import (  # noqa: E402
    RECTIFIED_CALIBRATION_CONTRACT,
    RectifiedCalibrationIndex,
    load_rectified_calibration_sidecar,
)


CALIBRATED_DERIVED_COMPONENT = "vggt-ffs-derived-geometry-calibrated-stereo-v2"
DERIVED_BATCH_COMPONENT = f"{CALIBRATED_DERIVED_COMPONENT}-batch"
DERIVED_ALGORITHM = (
    "baseline_metric_scale+scale_only_alignment+strict_pose_quality+"
    "calibrated_stereo_constraint_v2"
)
# This marker is deliberately explicit and machine-checkable.  A GT-pose
# override may replace a rejected VGGT transport pose with an authoritative
# Spring pose, but the producer decision must remain auditable in the derived
# record and manifest.  Consumers use this marker to distinguish that policy
# from a cache whose quality diagnostics were silently discarded.
QUALITY_SCORE_OVERRIDE = "authoritative_gt_pose"


def _safe(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError(f"invalid path component: {value!r}")
    return text


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise CacheMismatchError(f"blank derived manifest row {path}:{line_number}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise CacheMismatchError(f"derived manifest row is not an object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise CacheMismatchError(f"derived manifest is empty: {path}")
    return rows


def _gt_pose(record: ManifestRecord) -> np.ndarray:
    value = record.extras.get("gt_extrinsics_camera_from_world")
    if value is None:
        value = record.extras.get("gt_pose_camera_from_world")
    if value is None:
        raise CacheMismatchError(
            f"Spring GT pose is missing for {record.sequence_id}/{record.frame_id}"
        )
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise CacheMismatchError(f"invalid Spring GT pose for {record.sequence_id}/{record.frame_id}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=2e-5, rtol=0.0):
        raise CacheMismatchError("Spring GT pose homogeneous row is malformed")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise CacheMismatchError("Spring GT pose rotation is not orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=2e-5):
        raise CacheMismatchError("Spring GT pose rotation determinant is not +1")
    return pose


def _gt_pose_context(
    records: Sequence[ManifestRecord],
    endpoint_index: int,
    calibration: RectifiedCalibrationIndex,
) -> torch.Tensor:
    endpoint = records[endpoint_index]
    causal = [
        index
        for index, record in enumerate(records)
        if index <= endpoint_index and record.sequence_id == endpoint.sequence_id
    ]
    if not causal:
        raise CacheMismatchError("cannot construct an empty Spring causal pose context")
    selected = causal[-5:]
    selected = [selected[0]] * (5 - len(selected)) + selected
    views: list[torch.Tensor] = []
    for index in selected:
        record = records[index]
        left = _gt_pose(record)
        cal = calibration.record_for_identity(
            record.sequence_id, record.frame_id, record.timestamp
        )
        right_h = cal.as_tensor(dtype=torch.float64) @ torch.from_numpy(left)
        if not bool(torch.isfinite(right_h).all()):
            raise CacheMismatchError("composed Spring right pose is non-finite")
        views.extend(
            (
                torch.from_numpy(left[:3, :4].astype(np.float32, copy=False)),
                right_h[:3, :4].to(dtype=torch.float32),
            )
        )
    result = torch.stack(views, dim=0).contiguous()
    if tuple(result.shape) != (10, 3, 4):
        raise CacheMismatchError(f"Spring GT pose context has shape {tuple(result.shape)}")
    return result


def _source_identity(payload: Mapping[str, Any]) -> CacheIdentity:
    value = payload.get("identity")
    if not isinstance(value, Mapping):
        raise CacheMismatchError("source calibrated identity is missing")
    if value.get("component") != CALIBRATED_DERIVED_COMPONENT:
        raise CacheMismatchError(
            "source cache must be calibrated_stereo_v2; refusing legacy pose override"
        )
    return CacheIdentity(
        component=str(value["component"]),
        upstream_commit=str(value["upstream_commit"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        torch_version=str(value["torch_version"]),
        cuda_version=None if value["cuda_version"] is None else str(value["cuda_version"]),
        config_sha256=str(value["config_sha256"]),
    )


def override_spring_pose_calibrated(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    calibration_sidecar: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    records = load_manifest(manifest)
    source_rows = _read_rows(source / "cache_manifest.jsonl")
    calibration = load_rectified_calibration_sidecar(
        calibration_sidecar, expected_manifest_path=manifest
    )
    source_receipt_path = source / "run_receipt.json"
    if not source_receipt_path.is_file():
        raise FileNotFoundError(source_receipt_path)
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(source_receipt, Mapping):
        raise CacheMismatchError("source calibrated receipt is malformed")
    if source_receipt.get("schema_version") != 2 or source_receipt.get(
        "component"
    ) != DERIVED_BATCH_COMPONENT:
        raise CacheMismatchError("source receipt is not a calibrated-v2 batch receipt")
    source_config = source_receipt.get("config")
    if not isinstance(source_config, Mapping) or source_config.get("algorithm") != DERIVED_ALGORITHM:
        raise CacheMismatchError("source receipt calibrated algorithm differs")
    source_manifest_sha = sha256_file(source / "cache_manifest.jsonl")
    source_receipt_sha = sha256_file(source_receipt_path)
    config = dict(source_config)
    config.update(
        {
            "pose_source": "Spring_GT_pose",
            "depth_source": "copied_from_vggt_derived",
            "quality_score_override": QUALITY_SCORE_OVERRIDE,
            "source_derived_manifest_sha256": source_manifest_sha,
            "source_derived_receipt_sha256": source_receipt_sha,
        }
    )
    config_sha = canonical_json_sha256(config)
    output.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    written = reused = 0
    for row_index, row in enumerate(source_rows):
        try:
            target_index = int(row["target_manifest_index"])
            sequence_id = str(row["sequence_id"])
            frame_id = int(row["frame_id"])
            timestamp = float(row["timestamp"])
            source_path = Path(str(row["cache_path"])).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheMismatchError(f"malformed source row {row_index}") from exc
        if target_index < 0 or target_index >= len(records):
            raise CacheMismatchError(f"source target index out of range: {target_index}")
        record = records[target_index]
        if (sequence_id, frame_id, timestamp) != (
            record.sequence_id,
            record.frame_id,
            record.timestamp,
        ):
            raise CacheMismatchError(f"source row target differs at manifest index {target_index}")
        source_payload = load_cache_record(source_path)
        source_identity = _source_identity(source_payload)
        source_sha = sha256_file(source_path)
        identity = CacheIdentity(
            component=CALIBRATED_DERIVED_COMPONENT,
            upstream_commit=f"{source_identity.upstream_commit}+Spring_GT_pose",
            checkpoint_sha256=canonical_json_sha256(
                {
                    "source_cache_sha256": source_sha,
                    "source_receipt_sha256": source_receipt_sha,
                    "manifest_sha256": sha256_file(manifest),
                    "calibration_sidecar_sha256": calibration.sidecar_sha256,
                    "target_manifest_index": target_index,
                    "pose_source": "Spring_GT_pose",
                }
            ),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            config_sha256=config_sha,
        )
        out_path = output / _safe(sequence_id) / f"{_safe(frame_id)}.pt"
        if out_path.is_file() and not overwrite:
            existing = load_cache_record(out_path, expected_identity=identity)
            metadata = existing.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("pose_override", {}).get(
                "pose_source"
            ) != "Spring_GT_pose":
                raise CacheMismatchError(f"existing output is not a Spring GT-pose override: {out_path}")
            status = "reused"
            reused += 1
        else:
            tensors = dict(source_payload["tensors"])
            gt_context = _gt_pose_context(records, target_index, calibration)
            tensors["vggt_extrinsics_camera_from_world_metric_temporal"] = gt_context
            tensors[
                "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
            ] = gt_context
            tensors["temporal_pose_valid"] = torch.tensor(True, dtype=torch.bool)
            metadata = copy.deepcopy(source_payload["metadata"])
            if not isinstance(metadata, dict):
                raise CacheMismatchError(f"source metadata is malformed: {source_path}")
            quality = metadata.get("pose_quality")
            if not isinstance(quality, dict):
                raise CacheMismatchError(f"source pose_quality is malformed: {source_path}")
            # Preserve the producer's audited VGGT decision verbatim before
            # installing the authoritative Spring GT temporal pose.  A GT
            # override may legitimately make the temporal warp usable, but
            # it must never erase evidence that the source VGGT pose/prior
            # was rejected (the F6 VGGT-pose arm still consumes that source
            # decision).
            source_quality = copy.deepcopy(quality)
            raw_source_pose_valid = source_quality.get("pose_valid")
            if not isinstance(raw_source_pose_valid, bool):
                raise CacheMismatchError(
                    f"source pose_valid is malformed: {source_path}"
                )
            source_pose_valid = raw_source_pose_valid
            source_failure_reasons = copy.deepcopy(
                source_quality.get("failure_reasons", [])
            )
            if not isinstance(source_failure_reasons, list) or not all(
                isinstance(reason, str) for reason in source_failure_reasons
            ):
                raise CacheMismatchError(
                    f"source failure_reasons are malformed: {source_path}"
                )
            source_alignment = source_quality.get("alignment")
            if not isinstance(source_alignment, Mapping) or not isinstance(
                source_alignment.get("static_prior_valid"), bool
            ):
                raise CacheMismatchError(
                    f"source static_prior_valid is malformed: {source_path}"
                )
            source_static_prior_valid = bool(
                source_alignment["static_prior_valid"]
            )
            quality["pose_valid"] = True
            quality["pose_source"] = "Spring_GT_pose"
            # The temporal pose is now exact Spring GT.  The source VGGT
            # residuals may be unavailable for records rejected by the raw
            # quality gates, so the calibrated temporal loader must use an
            # explicit authoritative-GT score rather than interpreting those
            # missing diagnostics as malformed data.
            quality["quality_score_override"] = "authoritative_gt_pose"
            quality["failure_reasons"] = []
            quality["source_vggt_pose_valid"] = source_pose_valid
            quality["source_vggt_failure_reasons"] = source_failure_reasons
            quality["source_vggt_pose_quality"] = source_quality
            quality["source_vggt_static_prior_valid"] = source_static_prior_valid
            baseline = quality.get("baseline")
            if isinstance(baseline, dict):
                baseline.update(
                    {
                        "valid": True,
                        "failure_reason": None,
                        "baseline_coefficient_of_variation": 0.0,
                        "stereo_rotation_error_max_deg": 0.0,
                        "stereo_rotation_error_median_deg": 0.0,
                    }
                )
            metadata["pose_quality"] = quality
            metadata["config"] = dict(config)
            stereo = metadata.get("stereo_calibration")
            if isinstance(stereo, dict):
                stereo["hybrid_pose_valid"] = True
                metadata["stereo_calibration"] = stereo
            metadata["pose_override"] = {
                "pose_source": "Spring_GT_pose",
                "source_pose_source": "VGGT_pose",
                "depth_source": "copied_from_vggt_derived",
                "source_cache_path": str(source_path),
                "source_cache_sha256": source_sha,
                "source_receipt_sha256": source_receipt_sha,
                "manifest_path": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "calibration_sidecar_path": str(calibration.sidecar_path),
                "calibration_sidecar_sha256": calibration.sidecar_sha256,
                "target_manifest_index": target_index,
                "source_pose_valid": source_pose_valid,
                "source_static_prior_valid": source_static_prior_valid,
                "source_failure_reasons": source_failure_reasons,
                "quality_score_override": QUALITY_SCORE_OVERRIDE,
            }
            save_cache_record(out_path, tensors=tensors, metadata=metadata, identity=identity)
            status = "written"
            written += 1
        out_rows.append(
            {
                "selection_index": int(row.get("selection_index", row_index)),
                "target_manifest_index": target_index,
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "cache_path": str(out_path.resolve()),
                "cache_sha256": sha256_file(out_path),
                "status": status,
                "pose_valid": True,
                "static_prior_valid": bool(row.get("static_prior_valid", False)),
                "source_pose_valid": bool(
                    row.get("source_pose_valid", row.get("pose_valid", False))
                ),
                "source_static_prior_valid": bool(
                    row.get(
                        "source_static_prior_valid",
                        row.get("static_prior_valid", False),
                    )
                ),
                "source_failure_reasons": copy.deepcopy(
                    row.get("source_failure_reasons", row.get("failure_reasons", []))
                ),
                "quality_score_override": QUALITY_SCORE_OVERRIDE,
                "failure_reasons": [],
            }
        )
    out_rows.sort(key=lambda item: int(item["selection_index"]))
    manifest_out = output / "cache_manifest.jsonl"
    _atomic_jsonl(manifest_out, out_rows)
    manifest_out_sha = sha256_file(manifest_out)
    receipt = {
        "schema_version": 2,
        "component": DERIVED_BATCH_COMPONENT,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "config": config,
        "inputs": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "source_derived_root": str(source),
            "source_derived_receipt": str(source_receipt_path),
            "source_derived_receipt_sha256": source_receipt_sha,
            "source_derived_manifest": str(source / "cache_manifest.jsonl"),
            "source_derived_manifest_sha256": source_manifest_sha,
            "pose_source": "Spring_GT_pose",
            "depth_source": "copied_from_vggt_derived",
            "rectified_calibration_sidecar": str(calibration.sidecar_path),
            "rectified_calibration_sidecar_sha256": calibration.sidecar_sha256,
            "rectified_calibration_receipt": str(calibration.receipt_path),
            "rectified_calibration_receipt_sha256": calibration.receipt_sha256,
        },
        "selection": {"start_window": 0, "limit": None, "selected_windows": len(out_rows)},
        "counts": {
            "selected": len(out_rows),
            "written": written,
            "reused": reused,
            "pose_valid": len(out_rows),
            "pose_rejected": 0,
            "source_pose_valid": sum(bool(row["source_pose_valid"]) for row in out_rows),
            "source_pose_rejected": sum(
                not bool(row["source_pose_valid"]) for row in out_rows
            ),
            "source_static_prior_valid": sum(
                bool(row["source_static_prior_valid"]) for row in out_rows
            ),
            "source_static_prior_rejected": sum(
                not bool(row["source_static_prior_valid"]) for row in out_rows
            ),
            "static_prior_valid": sum(bool(row["static_prior_valid"]) for row in out_rows),
            "static_prior_rejected": sum(not bool(row["static_prior_valid"]) for row in out_rows),
        },
        "output": {
            "root": str(output),
            "cache_manifest": str(manifest_out),
            "cache_manifest_sha256": manifest_out_sha,
        },
        "pose_override": {
            "pose_source": "Spring_GT_pose",
            "source_pose_source": "VGGT_pose",
            "depth_source": "copied_from_vggt_derived",
            "quality_score_override": QUALITY_SCORE_OVERRIDE,
            "source_receipt_sha256": source_receipt_sha,
            "calibration_sidecar_sha256": calibration.sidecar_sha256,
        },
    }
    _atomic_json(output / "run_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--calibration-sidecar", type=Path, required=True)
    parser.add_argument("--output", "--output-root", dest="output_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    receipt = override_spring_pose_calibrated(
        manifest_path=args.manifest,
        source_root=args.source_root,
        calibration_sidecar=args.calibration_sidecar,
        output_root=args.output_root,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "pose_source": "Spring_GT_pose",
                "selected": receipt["counts"]["selected"],
                "written": receipt["counts"]["written"],
                "reused": receipt["counts"]["reused"],
                "output": str(args.output_root.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
