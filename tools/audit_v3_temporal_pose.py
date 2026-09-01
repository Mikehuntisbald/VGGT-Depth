#!/usr/bin/env python3
"""Audit whether calibrated v3 temporal-pose inputs vary on a derived split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


COMPONENT = "v3-temporal-pose-variation-audit"
SCHEMA_VERSION = 1
POSE_TENSOR = (
    "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
)
LEFT_VIEW_BY_AGE = {1: 6, 2: 4}
CURRENT_LEFT_VIEW = 8


class TemporalPoseAuditError(ValueError):
    """Raised when cache or pose evidence violates the v3 audit contract."""


def _record_id(sequence_id: str, frame_id: int) -> str:
    return f"{sequence_id}/{frame_id}"


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TemporalPoseAuditError("formal endpoint identity is not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _load_validation_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TemporalPoseAuditError(f"cannot read validation manifest: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise TemporalPoseAuditError("validation manifest is empty or contains blank rows")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    last_timestamp: dict[str, float] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalPoseAuditError(
                f"malformed validation-manifest row {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise TemporalPoseAuditError(
                f"validation-manifest row {line_number} is not an object"
            )
        sequence_id = row.get("sequence_id")
        frame_id = row.get("frame_id")
        timestamp = row.get("timestamp")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise TemporalPoseAuditError(
                f"validation-manifest identity is malformed on row {line_number}"
            )
        key = (sequence_id, frame_id)
        if key in seen:
            raise TemporalPoseAuditError(f"duplicate validation record {key!r}")
        previous = last_timestamp.get(sequence_id)
        if previous is not None and float(timestamp) <= previous:
            raise TemporalPoseAuditError(
                f"validation timestamps are not increasing for {sequence_id!r}"
            )
        seen.add(key)
        last_timestamp[sequence_id] = float(timestamp)
        records.append(
            {
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "timestamp": float(timestamp),
                "manifest_index": line_number - 1,
            }
        )
    return records


def _formal_temporal_endpoint_ids(
    records: Sequence[Mapping[str, Any]], derived_indices: set[int]
) -> list[str]:
    """Mirror the T=3/five-context dataset's exact derived-entry filter."""

    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_sequence[str(record["sequence_id"])].append(index)
    endpoints: list[tuple[int, str]] = []
    for sequence_id, indices in by_sequence.items():
        for position in range(4, len(indices)):
            student = indices[position - 2 : position + 1]
            if all(index in derived_indices for index in student):
                endpoint_index = indices[position]
                record = records[endpoint_index]
                endpoints.append(
                    (
                        endpoint_index,
                        _record_id(sequence_id, int(record["frame_id"])),
                    )
                )
    return [record_id for _, record_id in sorted(endpoints)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporalPoseAuditError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise TemporalPoseAuditError(f"{name} must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
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


def _homogeneous(extrinsics_camera_from_world: Tensor) -> Tensor:
    if extrinsics_camera_from_world.shape != (3, 4):
        raise TemporalPoseAuditError("each camera-from-world pose must be [3,4]")
    result = torch.eye(4, dtype=torch.float64)
    result[:3] = extrinsics_camera_from_world.to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise TemporalPoseAuditError("temporal pose contains non-finite values")
    rotation = result[:3, :3]
    if not torch.allclose(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(3, dtype=torch.float64),
        atol=2e-3,
        rtol=0.0,
    ) or not math.isclose(float(torch.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise TemporalPoseAuditError("temporal pose rotation is not SO(3)")
    return result


def _relative_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise TemporalPoseAuditError("cannot summarize an empty pose sample")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "minimum": float(tensor.min()),
        "mean": float(tensor.mean()),
        "maximum": float(tensor.max()),
        "std_population": float(tensor.std(unbiased=False)),
    }


def audit_temporal_pose_variation(
    derived_root: str | Path,
    *,
    validation_manifest_path: str | Path | None = None,
    minimum_valid_windows: int = 30,
    minimum_translation_std_over_baseline: float = 0.01,
    minimum_rotation_std_deg: float = 0.1,
) -> dict[str, Any]:
    """Return an immutable-input audit for causal age-1/age-2 pose variation."""

    root = Path(derived_root).expanduser().resolve()
    manifest_path = root / "cache_manifest.jsonl"
    receipt_path = root / "run_receipt.json"
    if minimum_valid_windows < 2:
        raise TemporalPoseAuditError("minimum_valid_windows must be at least two")
    for name, value in (
        ("minimum_translation_std_over_baseline", minimum_translation_std_over_baseline),
        ("minimum_rotation_std_deg", minimum_rotation_std_deg),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise TemporalPoseAuditError(f"{name} must be finite and positive")
    receipt = _load_json(receipt_path, "derived run receipt")
    if receipt.get("schema_version") != 2 or receipt.get("component") != (
        "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
    ):
        raise TemporalPoseAuditError("derived run receipt is not calibrated-stereo-v2")
    output = receipt.get("output")
    manifest_sha256 = _sha256(manifest_path)
    if not isinstance(output, Mapping) or output.get(
        "cache_manifest_sha256"
    ) != manifest_sha256:
        raise TemporalPoseAuditError("derived receipt does not bind cache manifest SHA-256")
    try:
        raw_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TemporalPoseAuditError(f"cannot read cache manifest: {manifest_path}") from exc
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise TemporalPoseAuditError("cache manifest is empty or contains blank rows")

    translations: dict[int, list[tuple[str, float]]] = {1: [], 2: []}
    rotations: dict[int, list[tuple[str, float]]] = {1: [], 2: []}
    baselines: list[tuple[str, float]] = []
    seen_targets: set[tuple[str, int]] = set()
    row_by_manifest_index: dict[int, Mapping[str, Any]] = {}
    valid_target_ids: set[str] = set()
    pose_valid_rows = 0
    for line_number, line in enumerate(raw_lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalPoseAuditError(
                f"malformed cache-manifest row {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise TemporalPoseAuditError(f"cache-manifest row {line_number} is not an object")
        sequence_id = row.get("sequence_id")
        frame_id = row.get("frame_id")
        if not isinstance(sequence_id, str) or isinstance(frame_id, bool) or not isinstance(
            frame_id, int
        ):
            raise TemporalPoseAuditError(f"invalid target identity on row {line_number}")
        target = (sequence_id, frame_id)
        if target in seen_targets:
            raise TemporalPoseAuditError(f"duplicate cache target {target!r}")
        seen_targets.add(target)
        target_manifest_index = row.get("target_manifest_index")
        if (
            isinstance(target_manifest_index, bool)
            or not isinstance(target_manifest_index, int)
            or target_manifest_index < 0
            or target_manifest_index in row_by_manifest_index
        ):
            raise TemporalPoseAuditError(
                f"invalid/duplicate target_manifest_index on row {line_number}"
            )
        row_by_manifest_index[target_manifest_index] = row
        if row.get("pose_valid") is not True:
            continue
        cache_path_value = row.get("cache_path")
        cache_sha256 = row.get("cache_sha256")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            raise TemporalPoseAuditError(f"pose-valid row {line_number} lacks cache_path")
        cache_path = Path(cache_path_value).expanduser().resolve()
        if (
            not isinstance(cache_sha256, str)
            or len(cache_sha256) != 64
            or any(character not in "0123456789abcdef" for character in cache_sha256)
        ):
            raise TemporalPoseAuditError(
                f"pose-valid row {line_number} has invalid cache_sha256"
            )
        if _sha256(cache_path) != cache_sha256:
            raise TemporalPoseAuditError(f"derived cache SHA mismatch: {cache_path}")
        try:
            artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError) as exc:
            raise TemporalPoseAuditError(f"cannot load derived cache: {cache_path}") from exc
        if not isinstance(artifact, Mapping) or not isinstance(
            artifact.get("tensors"), Mapping
        ):
            raise TemporalPoseAuditError(f"malformed derived cache: {cache_path}")
        tensors = artifact["tensors"]
        valid_tensor = tensors.get("temporal_pose_valid")
        poses = tensors.get(POSE_TENSOR)
        if not isinstance(valid_tensor, Tensor) or valid_tensor.shape != () or (
            valid_tensor.dtype != torch.bool or valid_tensor.item() is not True
        ):
            raise TemporalPoseAuditError(
                f"manifest/cache temporal_pose_valid mismatch: {cache_path}"
            )
        if not isinstance(poses, Tensor) or poses.shape != (10, 3, 4):
            raise TemporalPoseAuditError(f"{POSE_TENSOR} must be [10,3,4]: {cache_path}")
        metadata = artifact.get("metadata")
        source = metadata.get("source") if isinstance(metadata, Mapping) else None
        linkage = source.get("linkage") if isinstance(source, Mapping) else None
        record = (
            linkage.get("target_manifest_record")
            if isinstance(linkage, Mapping)
            else None
        )
        baseline = record.get("baseline_m") if isinstance(record, Mapping) else None
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or (
            not math.isfinite(float(baseline)) or float(baseline) <= 0.0
        ):
            raise TemporalPoseAuditError(f"invalid baseline in {cache_path}")
        target_id = _record_id(sequence_id, frame_id)
        baselines.append((target_id, float(baseline)))
        valid_target_ids.add(target_id)
        current = _homogeneous(poses[CURRENT_LEFT_VIEW])
        for age, history_index in LEFT_VIEW_BY_AGE.items():
            history = _homogeneous(poses[history_index])
            relative = current @ torch.linalg.inv(history)
            translation_norm = float(torch.linalg.vector_norm(relative[:3, 3]))
            cosine = max(
                -1.0,
                min(1.0, (float(torch.trace(relative[:3, :3])) - 1.0) / 2.0),
            )
            translations[age].append((target_id, translation_norm))
            rotations[age].append((target_id, math.degrees(math.acos(cosine))))
        pose_valid_rows += 1

    formal_endpoint_ids: list[str] | None = None
    formal_pose_valid_ids: list[str] | None = None
    validation_manifest_identity: dict[str, Any] | None = None
    if validation_manifest_path is not None:
        validation_path = Path(validation_manifest_path).expanduser().resolve()
        records = _load_validation_manifest(validation_path)
        for manifest_index, row in row_by_manifest_index.items():
            if manifest_index >= len(records):
                raise TemporalPoseAuditError(
                    f"derived target_manifest_index is outside validation manifest: {manifest_index}"
                )
            record = records[manifest_index]
            timestamp = row.get("timestamp")
            if (
                row.get("sequence_id") != record["sequence_id"]
                or row.get("frame_id") != record["frame_id"]
                or isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isclose(
                    float(timestamp),
                    float(record["timestamp"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise TemporalPoseAuditError(
                    f"derived/validation identity mismatch at manifest index {manifest_index}"
                )
        formal_endpoint_ids = _formal_temporal_endpoint_ids(
            records, set(row_by_manifest_index)
        )
        formal_set = set(formal_endpoint_ids)
        formal_pose_valid_ids = sorted(valid_target_ids & formal_set)
        validation_manifest_identity = {
            "path": str(validation_path),
            "sha256": _sha256(validation_path),
            "records": len(records),
        }

    formal_set = None if formal_endpoint_ids is None else set(formal_endpoint_ids)
    selected_baselines = [
        value for record_id, value in baselines if formal_set is None or record_id in formal_set
    ]
    baseline_reference = (
        sum(selected_baselines) / len(selected_baselines)
        if selected_baselines
        else None
    )
    age_reports: dict[str, Any] = {}
    ages_vary: list[bool] = []
    for age in (1, 2):
        translation_values = [
            value
            for record_id, value in translations[age]
            if formal_set is None or record_id in formal_set
        ]
        rotation_values = [
            value
            for record_id, value in rotations[age]
            if formal_set is None or record_id in formal_set
        ]
        translation = _relative_statistics(translation_values)
        rotation = _relative_statistics(rotation_values)
        std_over_baseline = (
            None
            if baseline_reference is None
            else translation["std_population"] / baseline_reference
        )
        varies = bool(
            std_over_baseline is not None
            and (
                std_over_baseline >= minimum_translation_std_over_baseline
                or rotation["std_population"] >= minimum_rotation_std_deg
            )
        )
        ages_vary.append(varies)
        age_reports[str(age)] = {
            "translation_norm_m": translation,
            "translation_std_over_baseline": std_over_baseline,
            "rotation_angle_deg": rotation,
            "varies": varies,
        }
    effective_valid_count = (
        len(formal_pose_valid_ids)
        if formal_pose_valid_ids is not None
        else pose_valid_rows
    )
    count_valid = effective_valid_count >= minimum_valid_windows
    temporal_pose_varies = count_valid and all(ages_vary)
    return {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS" if temporal_pose_varies else "FAIL",
        "temporal_pose_varies": temporal_pose_varies,
        "convention": (
            "T_current_from_history = E_current_camera_from_world @ "
            "inverse(E_history_camera_from_world); left views current=8, age1=6, age2=4"
        ),
        "counts": {
            "manifest_windows": len(raw_lines),
            "pose_valid_windows": pose_valid_rows,
            "formal_temporal_endpoints": (
                None if formal_endpoint_ids is None else len(formal_endpoint_ids)
            ),
            "formal_windows": (
                None if formal_endpoint_ids is None else len(formal_endpoint_ids)
            ),
            "formal_pose_valid_windows": (
                None if formal_pose_valid_ids is None else len(formal_pose_valid_ids)
            ),
            "minimum_valid_windows": minimum_valid_windows,
        },
        "formal_endpoint_binding": {
            "available": formal_endpoint_ids is not None,
            "record_ids": formal_endpoint_ids,
            "record_ids_sha256": (
                None
                if formal_endpoint_ids is None
                else _canonical_sha256(formal_endpoint_ids)
            ),
            "pose_valid_record_ids": formal_pose_valid_ids,
        },
        "baseline_reference_m": baseline_reference,
        "ages": age_reports,
        "thresholds": {
            "minimum_translation_std_over_baseline": (
                minimum_translation_std_over_baseline
            ),
            "minimum_rotation_std_deg": minimum_rotation_std_deg,
            "policy": "each age must exceed at least one variation threshold",
        },
        "inputs": {
            "derived_root": str(root),
            "cache_manifest_path": str(manifest_path),
            "cache_manifest_sha256": manifest_sha256,
            "run_receipt_path": str(receipt_path),
            "run_receipt_sha256": _sha256(receipt_path),
            "validation_manifest": validation_manifest_identity,
        },
        "claim_boundary": (
            "This proves that the cached temporal-pose conditioner input varies; "
            "it does not prove pose accuracy or conditioning benefit."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        "--manifest",
        dest="validation_manifest",
        type=Path,
        help="bind variation to the exact formal T=3 validation endpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-valid-windows", type=int, default=30)
    parser.add_argument(
        "--minimum-translation-std-over-baseline", type=float, default=0.01
    )
    parser.add_argument("--minimum-rotation-std-deg", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_temporal_pose_variation(
        args.derived_root,
        validation_manifest_path=args.validation_manifest,
        minimum_valid_windows=args.minimum_valid_windows,
        minimum_translation_std_over_baseline=(
            args.minimum_translation_std_over_baseline
        ),
        minimum_rotation_std_deg=args.minimum_rotation_std_deg,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPONENT",
    "TemporalPoseAuditError",
    "audit_temporal_pose_variation",
]
