#!/usr/bin/env python3
"""Build a GT-pose/no-VGGT-prior geometry cache for Spring arm screening.

S2--S4 intentionally isolate temporal history and the VGGT depth prior.  The
Spring manifest contains calibrated camera poses, so those arms do not need to
run the 1B VGGT model: this producer stores the manifest GT pose in the exact
``[10,3,4]`` temporal field and zero-fills the optional VGGT disparity prior.
The lineage is explicitly marked ``Spring_GT_pose``; it must never be reused
for an arm whose pose source is ``vggt``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import tempfile
import time
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
from data.manifest import load_manifest  # noqa: E402


LEGACY_ALGORITHM = "baseline_metric_scale+scale_only_alignment+strict_pose_quality"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _safe(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("._")
    if not text:
        raise ValueError(f"invalid path component: {value!r}")
    return text


def _gt_pose(record: Any) -> np.ndarray:
    value = record.extras.get("gt_extrinsics_camera_from_world")
    if value is None:
        value = record.extras.get("gt_pose_camera_from_world")
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid GT pose for {record.sequence_id}/{record.frame_id}")
    return pose


def _gt_context(records: list[Any], endpoint_index: int) -> torch.Tensor:
    endpoint = records[endpoint_index]
    indices = [
        index
        for index, record in enumerate(records)
        if index <= endpoint_index and record.sequence_id == endpoint.sequence_id
    ]
    if not indices:
        raise ValueError("cannot build an empty GT pose context")
    selected = indices[-5:]
    selected = [selected[0]] * (5 - len(selected)) + selected
    views: list[torch.Tensor] = []
    for index in selected:
        record = records[index]
        left = _gt_pose(record)
        right_from_left = np.eye(4, dtype=np.float64)
        right_from_left[0, 3] = -float(record.baseline_m)
        right = right_from_left @ left
        views.extend(
            (
                torch.from_numpy(left[:3, :4].astype(np.float32, copy=False)),
                torch.from_numpy(right[:3, :4].astype(np.float32, copy=False)),
            )
        )
    result = torch.stack(views, dim=0).contiguous()
    if tuple(result.shape) != (10, 3, 4) or not bool(torch.isfinite(result).all()):
        raise ValueError("GT pose context must be finite [10,3,4]")
    return result


def _selected_records(
    records: Sequence[Any], *, sequence_warmup: int
) -> list[tuple[int, Any]]:
    """Keep original manifest indices after a per-sequence positional warmup."""

    if isinstance(sequence_warmup, bool) or not isinstance(sequence_warmup, int):
        raise TypeError("sequence_warmup must be an integer")
    if sequence_warmup < 0:
        raise ValueError("sequence_warmup must be non-negative")
    positions: dict[str, int] = defaultdict(int)
    selected: list[tuple[int, Any]] = []
    for manifest_index, record in enumerate(records):
        sequence_id = str(record.sequence_id)
        position = positions[sequence_id]
        positions[sequence_id] += 1
        if position >= sequence_warmup:
            selected.append((manifest_index, record))
    if not selected:
        raise ValueError("sequence warmup removed every manifest record")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequence-warmup",
        type=int,
        default=0,
        help=(
            "skip this many leading records per sequence; F3 uses 4 so its "
            "three-frame derived window has the same frame-7 endpoint floor "
            "as calibrated VGGT geometry"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    observation_root = args.observation_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    records = load_manifest(manifest)
    if not records:
        raise ValueError("manifest is empty")
    selected_records = _selected_records(
        records, sequence_warmup=args.sequence_warmup
    )
    manifest_sha = sha256_file(manifest)
    config = {
        "schema_version": 1,
        "algorithm": LEGACY_ALGORITHM,
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
        "pose_source": "Spring_GT_pose",
        "depth_prior_source": "disabled_zero_fill",
        "sequence_warmup": args.sequence_warmup,
        "selection_policy": "per_sequence_manifest_position",
    }
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for selection_index, (manifest_index, record) in enumerate(selected_records):
        obs_path = (
            observation_root
            / _safe(record.sequence_id)
            / f"{_safe(record.frame_id)}.pt"
        )
        if not obs_path.is_file():
            raise FileNotFoundError(f"observation cache missing: {obs_path}")
        obs_sha = sha256_file(obs_path)
        obs_payload = load_cache_record(obs_path)
        obs_tensors = obs_payload.get("tensors", {})
        trusted = obs_tensors.get("observation_trusted_mask")
        disparity = obs_tensors.get("observation_disparity_hr_px")
        if not isinstance(trusted, torch.Tensor) or not isinstance(disparity, torch.Tensor):
            raise ValueError(f"observation cache lacks required tensors: {obs_path}")
        if disparity.ndim == 4:
            disparity = disparity[0]
        if trusted.ndim == 4:
            trusted = trusted[0]
        if disparity.ndim != 3 or trusted.shape != disparity.shape:
            raise ValueError(f"observation cache grid malformed: {obs_path}")
        grid = tuple(int(value) for value in disparity.shape)
        gt_pose = _gt_context(records, manifest_index)
        identity = CacheIdentity(
            component="vggt-ffs-derived-geometry",
            upstream_commit="Spring_GT_pose:cam_data/extrinsics.txt",
            checkpoint_sha256=canonical_json_sha256(
                {"manifest_sha256": manifest_sha, "observation_sha256": obs_sha}
            ),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            config_sha256=canonical_json_sha256(config),
        )
        cache_path = output_root / _safe(record.sequence_id) / f"{_safe(record.frame_id)}.pt"
        if cache_path.is_file() and not args.overwrite:
            payload = load_cache_record(cache_path, expected_identity=identity)
            status = "reused"
            metadata = payload["metadata"]
        else:
            zeros = torch.zeros(grid, dtype=torch.float32)
            false_mask = torch.zeros(grid, dtype=torch.bool)
            tensors = {
                "vggt_extrinsics_camera_from_world_metric_diagnostic_only": gt_pose,
                "vggt_extrinsics_camera_from_world_metric_temporal": gt_pose,
                "vggt_depth_current_left_metric_m": zeros,
                "vggt_disparity_current_left_aligned_hr_px": zeros,
                "vggt_aligned_confidence": zeros,
                "vggt_depth_metric_valid_mask": false_mask,
                "vggt_aligned_valid_mask": false_mask,
                "ffs_trusted_mask": trusted.to(torch.bool),
                "temporal_pose_valid": torch.tensor(True, dtype=torch.bool),
                "static_prior_valid": torch.tensor(False, dtype=torch.bool),
            }
            linkage = {
                "target_sequence_id": record.sequence_id,
                "target_frame_id": record.frame_id,
                "target_timestamp": record.timestamp,
                "target_manifest_record": record.to_dict(),
                "pose_source": "Spring_GT_pose",
                "ffs_raw_identity": obs_payload.get("identity"),
            }
            metadata = {
                "source": {
                    "ffs_cache_path": str(obs_path),
                    "ffs_cache_sha256": obs_sha,
                    "ffs_raw_identity": obs_payload.get("identity"),
                    "linkage": linkage,
                    "manifest_path": str(manifest),
                    "manifest_sha256": manifest_sha,
                },
                "config": config,
                "target": {
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "timestamp": record.timestamp,
                },
                "pose_quality": {
                    "pose_valid": True,
                    "failure_reasons": [],
                    "pose_source": "Spring_GT_pose",
                    "alignment": {
                        "valid": False,
                        "static_prior_valid": False,
                        "failure_reason": "disabled_by_arm_control",
                    },
                },
                "tensor_semantics": {
                    "metric_pose_temporal": "Spring GT camera-from-world pose",
                    "metric_depth": "zero-filled; VGGT depth disabled",
                    "aligned_disparity": "zero-filled; VGGT depth disabled",
                },
            }
            save_cache_record(cache_path, tensors=tensors, metadata=metadata, identity=identity)
            status = "written"
        counts["selected"] += 1
        counts["written"] += int(status == "written")
        counts["reused"] += int(status == "reused")
        counts["pose_valid"] += 1
        counts["static_prior_rejected"] += 1
        rows.append(
            {
                "selection_index": selection_index,
                "target_manifest_index": manifest_index,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": sha256_file(cache_path),
                "status": status,
                "pose_valid": True,
                "static_prior_valid": False,
                "failure_reasons": [],
            }
        )
        print(
            f"[{len(rows)}/{len(selected_records)}] "
            f"{record.sequence_id}/{record.frame_id} {status}",
            flush=True,
        )

    cache_manifest = output_root / "cache_manifest.jsonl"
    _atomic_jsonl(cache_manifest, rows)
    total = len(rows)
    receipt = {
        "schema_version": 1,
        "component": "vggt-ffs-derived-geometry-batch",
        # Keep the manifest binding at the receipt root, matching the other
        # derived-cache producers and the runner's strict receipt contract.
        # The same fields remain under ``inputs`` for backwards-readable
        # provenance.
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "config": config,
        "inputs": {
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha,
            "observation_root": str(observation_root),
            "vggt_root": None,
            "pose_source": "Spring_GT_pose",
        },
        "output": {
            "root": str(output_root),
            "cache_manifest": str(cache_manifest),
            "cache_manifest_sha256": sha256_file(cache_manifest),
        },
        "selection": {
            "start_window": 0,
            "limit": None,
            "selected_windows": total,
            "sequence_warmup": args.sequence_warmup,
            "policy": "per_sequence_manifest_position",
        },
        "counts": {
            "selected": total,
            "written": counts["written"],
            "reused": counts["reused"],
            "pose_valid": total,
            "pose_rejected": 0,
            "static_prior_valid": 0,
            "static_prior_rejected": total,
        },
        "rates": {
            "pose_valid": 1.0,
            "pose_rejected": 0.0,
            "static_prior_valid": 0.0,
            "static_prior_rejected": 1.0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "lineage_note": "GT pose control cache; never valid for temporal_pose_source=vggt",
    }
    _atomic_json(output_root / "run_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
