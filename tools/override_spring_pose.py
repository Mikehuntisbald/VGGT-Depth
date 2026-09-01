#!/usr/bin/env python3
"""Create a Spring GT-pose view of an existing VGGT-derived cache.

The S4 ablation needs the *same* VGGT depth/alignment as S5 while replacing
only the temporal camera pose with Spring's calibrated manifest pose.  This
tool performs that operation on a complete derived-cache tree and records a
new cache identity/receipt.  It never edits the source cache in place and it
never changes the depth-prior tensors, so a comparison between S4 and S5 is
attributable to pose source alone.

The output intentionally retains the project's legacy derived-cache contract
(``vggt-ffs-derived-geometry`` / schema version 1); the additional
``pose_override`` fields are provenance metadata, not a replacement contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.cache_dataset import (  # noqa: E402
    CacheIdentity,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import ManifestRecord, load_manifest  # noqa: E402


DERIVED_COMPONENT = "vggt-ffs-derived-geometry"
DERIVED_BATCH_COMPONENT = "vggt-ffs-derived-geometry-batch"
DERIVED_ALGORITHM = (
    "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
)


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
                    json.dumps(
                        dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gt_pose(record: ManifestRecord) -> np.ndarray:
    value = record.extras.get("gt_extrinsics_camera_from_world")
    if value is None:
        value = record.extras.get("gt_pose_camera_from_world")
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid Spring GT pose for {record.sequence_id}/{record.frame_id}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise ValueError(f"non-orthonormal Spring GT pose for {record.sequence_id}/{record.frame_id}")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=2e-5):
        raise ValueError(f"invalid Spring GT rotation for {record.sequence_id}/{record.frame_id}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"invalid Spring GT homogeneous row for {record.sequence_id}/{record.frame_id}")
    return pose


def gt_pose_context(records: Sequence[ManifestRecord], endpoint_index: int) -> torch.Tensor:
    """Return the exact causal five-pair Spring pose context [10,3,4]."""

    if endpoint_index < 0 or endpoint_index >= len(records):
        raise IndexError(endpoint_index)
    endpoint = records[endpoint_index]
    causal = [
        index
        for index, record in enumerate(records)
        if index <= endpoint_index and record.sequence_id == endpoint.sequence_id
    ]
    if not causal:
        raise ValueError("cannot construct an empty causal pose context")
    selected = causal[-5:]
    selected = [selected[0]] * (5 - len(selected)) + selected
    views: list[torch.Tensor] = []
    for index in selected:
        record = records[index]
        left = _gt_pose(record)
        right_from_left = np.eye(4, dtype=np.float64)
        # Spring's rectified right camera: X_right = X_left + [-B,0,0].
        right_from_left[0, 3] = -float(record.baseline_m)
        right = right_from_left @ left
        views.append(torch.from_numpy(left[:3, :4].astype(np.float32, copy=False)))
        views.append(torch.from_numpy(right[:3, :4].astype(np.float32, copy=False)))
    result = torch.stack(views, dim=0).contiguous()
    if tuple(result.shape) != (10, 3, 4) or not bool(torch.isfinite(result).all()):
        raise ValueError("Spring GT pose context must be finite [10,3,4]")
    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"derived cache manifest is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"blank row in derived cache manifest: {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"derived cache row is not an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"derived cache manifest is empty: {path}")
    return rows


def _source_identity(payload: Mapping[str, Any]) -> CacheIdentity:
    value = payload.get("identity")
    if not isinstance(value, Mapping):
        raise ValueError("source cache identity is missing")
    required = {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
    if set(value) != required:
        raise ValueError("source cache identity fields are malformed")
    if value.get("component") != DERIVED_COMPONENT:
        raise ValueError(
            f"source derived component must be {DERIVED_COMPONENT!r}, got {value.get('component')!r}"
        )
    return CacheIdentity(
        component=str(value["component"]),
        upstream_commit=str(value["upstream_commit"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        torch_version=str(value["torch_version"]),
        cuda_version=None if value["cuda_version"] is None else str(value["cuda_version"]),
        config_sha256=str(value["config_sha256"]),
    )


def _validate_source_receipt(
    source_root: Path, manifest: Path, selected_count: int
) -> dict[str, Any]:
    receipt_path = source_root / "run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"source derived receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("source derived receipt is not an object")
    if receipt.get("component") != DERIVED_BATCH_COMPONENT:
        raise ValueError("source derived receipt has an unexpected component")
    if receipt.get("schema_version") != 1:
        raise ValueError("pose override currently supports legacy_v1 derived caches only")
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("source derived receipt has no config")
    required = {
        "algorithm": DERIVED_ALGORITHM,
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"source derived receipt config {key!r} differs")
    if receipt.get("manifest") and Path(str(receipt["manifest"])).resolve() != manifest.resolve():
        raise ValueError("source derived receipt manifest differs from --manifest")
    if receipt.get("inputs", {}).get("manifest_sha256") not in (None, sha256_file(manifest)):
        raise ValueError("source derived receipt manifest SHA-256 differs")
    counts = receipt.get("counts")
    selection = receipt.get("selection")
    if not isinstance(counts, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("source derived receipt coverage fields are missing")
    # A VGGT-derived cache is intentionally sparse with respect to the source
    # frame manifest: five-pair causal inference can only emit endpoints from
    # frame 5 onward.  The cache must cover its *own* complete selected-window
    # inventory, while the temporal dataset later decides which T=3 windows
    # have all three derived student endpoints.  Requiring one derived row
    # per manifest frame would reject every valid bounded Spring run (for a
    # seven-frame split the source has three VGGT endpoints and one full T=3
    # window).
    if (
        int(counts.get("selected", -1)) != selected_count
        or int(selection.get("selected_windows", -1)) != selected_count
    ):
        raise ValueError("source derived cache coverage is incomplete")
    if int(counts.get("written", -1)) + int(counts.get("reused", -1)) != selected_count:
        raise ValueError("source derived receipt written/reused coverage is malformed")
    for valid_name, rejected_name in (("pose_valid", "pose_rejected"), ("static_prior_valid", "static_prior_rejected")):
        if int(counts.get(valid_name, -1)) + int(counts.get(rejected_name, -1)) != selected_count:
            raise ValueError(f"source derived receipt {valid_name}/{rejected_name} coverage is malformed")
    manifest_path = source_root / "cache_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source derived manifest is missing: {manifest_path}")
    output = receipt.get("output")
    if isinstance(output, Mapping) and output.get("cache_manifest_sha256") not in (None, sha256_file(manifest_path)):
        raise ValueError("source derived receipt/cache manifest SHA-256 differs")
    return receipt


def override_spring_pose(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a complete derived cache while substituting Spring GT poses."""

    manifest = Path(manifest_path).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not source.is_dir():
        raise FileNotFoundError(f"source derived cache root does not exist: {source}")
    records = load_manifest(manifest)
    source_rows = _read_rows(source / "cache_manifest.jsonl")
    if not source_rows:
        raise ValueError("source derived cache manifest is empty")
    source_receipt = _validate_source_receipt(source, manifest, len(source_rows))
    source_receipt_sha = sha256_file(source / "run_receipt.json")
    source_manifest_sha = sha256_file(source / "cache_manifest.jsonl")
    # Preserve every source threshold/algorithm field.  Only the explicit
    # source selectors and immutable lineage hashes are added below; dropping
    # a threshold would make an otherwise valid derived record impossible to
    # audit against the producer that generated its depth prior.
    source_config = source_receipt.get("config")
    if not isinstance(source_config, Mapping):  # guarded by _validate_source_receipt
        raise ValueError("source derived receipt config is malformed")
    config = dict(source_config)
    config.update(
        {
            "schema_version": 1,
            "algorithm": DERIVED_ALGORITHM,
            "extrinsics_convention": "camera-from-world",
            "previous_left_view_index": 6,
            "current_left_view_index": 8,
            "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
            "pose_source": "Spring_GT_pose",
            "depth_source": "copied_from_vggt_derived",
            "source_derived_receipt_sha256": source_receipt_sha,
            "source_derived_manifest_sha256": source_manifest_sha,
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
            raise ValueError(f"malformed source derived row {row_index}") from exc
        if target_index < 0 or target_index >= len(records):
            raise ValueError(f"source target index out of range: {target_index}")
        record = records[target_index]
        if (sequence_id, frame_id, timestamp) != (record.sequence_id, record.frame_id, record.timestamp):
            raise ValueError(f"source row target differs from manifest at index {target_index}")
        if not source_path.is_file():
            raise FileNotFoundError(f"source derived record is missing: {source_path}")
        source_payload = load_cache_record(source_path)
        source_identity = _source_identity(source_payload)
        out_path = output / _safe(sequence_id) / f"{_safe(frame_id)}.pt"
        override_identity = CacheIdentity(
            component=DERIVED_COMPONENT,
            upstream_commit=(
                f"{source_identity.upstream_commit}+Spring_GT_pose"
            ),
            checkpoint_sha256=canonical_json_sha256(
                {
                    "source_cache_sha256": sha256_file(source_path),
                    "source_receipt_sha256": source_receipt_sha,
                    "manifest_sha256": sha256_file(manifest),
                    "target_manifest_index": target_index,
                    "pose_source": "Spring_GT_pose",
                }
            ),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            config_sha256=config_sha,
        )
        if out_path.is_file() and not overwrite:
            existing = load_cache_record(out_path, expected_identity=override_identity)
            metadata = existing.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"override metadata is malformed: {out_path}")
            override = metadata.get("pose_override")
            if not isinstance(override, Mapping) or override.get("pose_source") != "Spring_GT_pose":
                raise ValueError(f"existing output is not a Spring GT-pose override: {out_path}")
            status = "reused"
            reused += 1
        else:
            tensors = dict(source_payload["tensors"])
            tensors["vggt_extrinsics_camera_from_world_metric_temporal"] = gt_pose_context(
                records, target_index
            )
            tensors["temporal_pose_valid"] = torch.tensor(True, dtype=torch.bool)
            metadata = copy.deepcopy(source_payload["metadata"])
            if not isinstance(metadata, dict):
                raise ValueError(f"source metadata is malformed: {source_path}")
            quality = metadata.get("pose_quality")
            if not isinstance(quality, dict):
                raise ValueError(f"source pose_quality is malformed: {source_path}")
            quality["pose_valid"] = True
            quality["failure_reasons"] = []
            quality["pose_source"] = "Spring_GT_pose"
            baseline = quality.get("baseline")
            if isinstance(baseline, dict):
                baseline["valid"] = True
                baseline["failure_reason"] = None
                baseline["baseline_coefficient_of_variation"] = 0.0
                baseline["stereo_rotation_error_max_deg"] = 0.0
                baseline["stereo_rotation_error_median_deg"] = 0.0
            metadata["pose_quality"] = quality
            metadata["config"] = dict(config)
            metadata["pose_override"] = {
                "pose_source": "Spring_GT_pose",
                "source_pose_source": "VGGT_pose",
                "depth_source": "copied_from_vggt_derived",
                "source_cache_path": str(source_path),
                "source_cache_sha256": sha256_file(source_path),
                "manifest_path": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "target_manifest_index": target_index,
            }
            save_cache_record(
                out_path,
                tensors=tensors,
                metadata=metadata,
                identity=override_identity,
            )
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
                "failure_reasons": [],
            }
        )

    out_rows.sort(key=lambda item: int(item["selection_index"]))
    _atomic_jsonl(output / "cache_manifest.jsonl", out_rows)
    output_manifest_sha = sha256_file(output / "cache_manifest.jsonl")
    receipt = {
        "schema_version": 1,
        "component": DERIVED_BATCH_COMPONENT,
        # Keep the source frame manifest binding at the receipt top level, in
        # the same form emitted by derive_geometry_manifest.  The Spring arm
        # runner uses this field for strict split/endpoint coverage checks;
        # the nested inputs copy below remains for older consumers.
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "config": config,
        "inputs": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "source_derived_root": str(source),
            "source_derived_receipt": str(source / "run_receipt.json"),
            "source_derived_receipt_sha256": source_receipt_sha,
            "source_derived_manifest": str(source / "cache_manifest.jsonl"),
            "source_derived_manifest_sha256": source_manifest_sha,
            "pose_source": "Spring_GT_pose",
            "depth_source": "copied_from_vggt_derived",
        },
        "selection": {"start_window": 0, "limit": None, "selected_windows": len(out_rows)},
        "counts": {
            "selected": len(out_rows),
            "written": written,
            "reused": reused,
            "pose_valid": len(out_rows),
            "pose_rejected": 0,
            "static_prior_valid": sum(bool(row["static_prior_valid"]) for row in out_rows),
            "static_prior_rejected": sum(not bool(row["static_prior_valid"]) for row in out_rows),
        },
        "output": {
            "root": str(output),
            "cache_manifest": str(output / "cache_manifest.jsonl"),
            "cache_manifest_sha256": output_manifest_sha,
        },
        "pose_override": {
            "pose_source": "Spring_GT_pose",
            "source_pose_source": "VGGT_pose",
            "depth_source": "copied_from_vggt_derived",
            "source_receipt_sha256": source_receipt_sha,
        },
        "lineage_note": "S4 control: actual VGGT depth/alignment with independent Spring GT pose",
    }
    _atomic_json(output / "run_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", "--output-root", dest="output_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = override_spring_pose(
        manifest_path=args.manifest,
        source_root=args.source_root,
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
