#!/usr/bin/env python3
"""Read-only end-to-end audit of formal Stage-B cached training inputs.

The audit is deliberately independent from the training Dataset.  It validates
the manifests, canonical receipts, every cache record, and the causal endpoint
join directly.  Cache records are loaded with ``torch.load(weights_only=True)``;
no model is constructed and no GPU work is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


AUDIT_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
STUDENT_SEQUENCE_LENGTH = 3
VGGT_CONTEXT_PAIRS = 5
VIEW_ORDER = (
    "L[t-4]",
    "R[t-4]",
    "L[t-3]",
    "R[t-3]",
    "L[t-2]",
    "R[t-2]",
    "L[t-1]",
    "R[t-1]",
    "L[t]",
    "R[t]",
)
IDENTITY_FIELDS = {
    "component",
    "upstream_commit",
    "checkpoint_sha256",
    "torch_version",
    "cuda_version",
    "config_sha256",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TemporalInputAuditError(RuntimeError):
    """Raised when formal Stage-B input lineage cannot be proven."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    index: int
    sequence_id: str
    frame_id: int
    timestamp: float
    left_path: Path
    right_path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CausalEndpoint:
    sequence_id: str
    endpoint_index: int
    student_indices: tuple[int, int, int]
    vggt_indices: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ManifestSnapshot:
    path: Path
    sha256: str
    byte_size: int
    records: tuple[ManifestEntry, ...]
    sequence_order: tuple[str, ...]
    indices_by_sequence: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class CacheAuditResult:
    report: Mapping[str, Any]
    identity: Mapping[str, Any]
    file_sha256_by_index: Mapping[int, str]
    tensor_shape_by_index: Mapping[int, tuple[int, ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TemporalInputAuditError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_constant(value: str) -> None:
    raise TemporalInputAuditError(f"strict JSON contains {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, name: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_strict_constant,
            object_pairs_hook=_strict_object,
        )
    except TemporalInputAuditError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise TemporalInputAuditError(f"cannot parse strict JSON {name}: {exc}") from exc


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file(), f"{name} is missing: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TemporalInputAuditError(f"{name} is not UTF-8: {exc}") from exc
    value = _strict_json_loads(text, name)
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    _finite_tree(value, name)
    return value, payload


def _read_jsonl(path: Path, name: str) -> tuple[list[dict[str, Any]], bytes]:
    _require(path.is_file(), f"{name} is missing: {path}")
    payload = path.read_bytes()
    _require(payload.endswith(b"\n"), f"{name} has an unterminated final line")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TemporalInputAuditError(f"{name} is not UTF-8: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"{name} line {line_number} is blank")
        value = _strict_json_loads(line, f"{name}:{line_number}")
        _require(isinstance(value, dict), f"{name} line {line_number} is not an object")
        _finite_tree(value, f"{name}[{line_number}]")
        rows.append(value)
    _require(bool(rows), f"{name} is empty")
    return rows, payload


def _finite_tree(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"{name} contains non-finite tensor values")
        return
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
        return
    raise TemporalInputAuditError(f"{name} contains unsupported {type(value).__name__}")


def _safe_component(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    _require(bool(normalized), f"cannot form a safe cache component from {value!r}")
    return normalized


def _cache_path(root: Path, record: ManifestEntry) -> Path:
    return root / _safe_component(record.sequence_id) / f"{_safe_component(record.frame_id)}.pt"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_safe_cache(path: Path, name: str) -> tuple[dict[str, Any], str, int]:
    _require(path.is_file(), f"{name} is missing: {path}")
    payload_bytes = path.read_bytes()
    try:
        payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - safe-load failures are audit evidence
        raise TemporalInputAuditError(f"weights_only load failed for {name}: {exc}") from exc
    _require(isinstance(payload, dict), f"{name} payload is not a dictionary")
    _require(
        set(payload) == {"schema_version", "identity", "metadata", "tensors"},
        f"{name} cache envelope schema is malformed",
    )
    _require(payload["schema_version"] == CACHE_SCHEMA_VERSION, f"{name} cache schema mismatch")
    _require(isinstance(payload["identity"], dict), f"{name} identity is malformed")
    _require(isinstance(payload["metadata"], dict), f"{name} metadata is malformed")
    _require(isinstance(payload["tensors"], dict) and payload["tensors"], f"{name} tensors are malformed")
    _finite_tree(payload["identity"], f"{name}.identity")
    _finite_tree(payload["metadata"], f"{name}.metadata")
    return payload, _sha256_bytes(payload_bytes), len(payload_bytes)


def _validate_identity(identity: Any, component: str, name: str) -> dict[str, Any]:
    _require(isinstance(identity, dict), f"{name} identity is not a mapping")
    _require(set(identity) == IDENTITY_FIELDS, f"{name} identity fields are malformed")
    _require(identity["component"] == component, f"{name} identity component mismatch")
    for field in ("upstream_commit", "checkpoint_sha256", "torch_version", "config_sha256"):
        _require(isinstance(identity[field], str) and identity[field], f"{name} identity {field} is invalid")
    _require(SHA256_PATTERN.fullmatch(identity["checkpoint_sha256"]) is not None, f"{name} checkpoint SHA is invalid")
    _require(SHA256_PATTERN.fullmatch(identity["config_sha256"]) is not None, f"{name} config SHA is invalid")
    _require(identity["cuda_version"] is None or isinstance(identity["cuda_version"], str), f"{name} CUDA version is invalid")
    return identity


def _parse_manifest(path: Path, name: str) -> ManifestSnapshot:
    rows, payload = _read_jsonl(path, name)
    records: list[ManifestEntry] = []
    seen_targets: set[tuple[str, int]] = set()
    sequence_order: list[str] = []
    closed_sequences: set[str] = set()
    active_sequence: str | None = None
    indices_by_sequence: dict[str, list[int]] = defaultdict(list)
    last_timestamp: dict[str, float] = {}
    last_frame: dict[str, int] = {}
    for index, row in enumerate(rows):
        for field in ("sequence_id", "frame_id", "timestamp", "left_path", "right_path", "K", "baseline_m"):
            _require(field in row, f"{name} row {index} is missing {field}")
        sequence_id = row["sequence_id"]
        frame_id = row["frame_id"]
        timestamp = _finite_number(row["timestamp"], f"{name}[{index}].timestamp")
        _require(isinstance(sequence_id, str) and sequence_id, f"{name} row {index} sequence_id is invalid")
        _require(_is_int(frame_id), f"{name} row {index} frame_id is invalid")
        _require(row.get("rectified") is True, f"{name} row {index} is not explicitly rectified")
        baseline = _finite_number(row["baseline_m"], f"{name}[{index}].baseline_m")
        _require(baseline > 0.0, f"{name} row {index} baseline is not positive")
        K = row["K"]
        _require(
            isinstance(K, list)
            and len(K) == 3
            and all(isinstance(line, list) and len(line) == 3 for line in K),
            f"{name} row {index} K is not 3x3",
        )
        _finite_tree(K, f"{name}[{index}].K")
        left = Path(str(row["left_path"])).expanduser().resolve()
        right = Path(str(row["right_path"])).expanduser().resolve()
        _require(left.is_file() and right.is_file(), f"{name} row {index} source stereo file is missing")
        _require(left != right, f"{name} row {index} left/right paths are identical")
        target = (sequence_id, int(frame_id))
        _require(target not in seen_targets, f"{name} contains duplicate target {target}")
        seen_targets.add(target)
        if active_sequence != sequence_id:
            if active_sequence is not None:
                closed_sequences.add(active_sequence)
            _require(sequence_id not in closed_sequences, f"{name} sequence {sequence_id!r} is non-contiguous")
            active_sequence = sequence_id
            sequence_order.append(sequence_id)
        if sequence_id in last_timestamp:
            _require(timestamp > last_timestamp[sequence_id], f"{name} timestamps are not increasing in {sequence_id}")
            _require(int(frame_id) > last_frame[sequence_id], f"{name} frame IDs are not increasing in {sequence_id}")
        last_timestamp[sequence_id] = timestamp
        last_frame[sequence_id] = int(frame_id)
        if "source_frame_index" in row:
            _require(row["source_frame_index"] == frame_id, f"{name} source_frame_index mismatch at {index}")
        if "source_time_sec" in row:
            _require(row["source_time_sec"] == row["timestamp"], f"{name} source_time_sec mismatch at {index}")
        records.append(
            ManifestEntry(
                index=index,
                sequence_id=sequence_id,
                frame_id=int(frame_id),
                timestamp=timestamp,
                left_path=left,
                right_path=right,
                raw=row,
            )
        )
        indices_by_sequence[sequence_id].append(index)
    return ManifestSnapshot(
        path=path.resolve(),
        sha256=_sha256_bytes(payload),
        byte_size=len(payload),
        records=tuple(records),
        sequence_order=tuple(sequence_order),
        indices_by_sequence={key: tuple(value) for key, value in indices_by_sequence.items()},
    )


def _build_causal_endpoints(manifest: ManifestSnapshot) -> tuple[CausalEndpoint, ...]:
    windows: list[CausalEndpoint] = []
    for sequence_id in manifest.sequence_order:
        indices = manifest.indices_by_sequence[sequence_id]
        for position in range(VGGT_CONTEXT_PAIRS - 1, len(indices)):
            endpoint = indices[position]
            student = tuple(indices[position - STUDENT_SEQUENCE_LENGTH + 1 : position + 1])
            vggt = tuple(indices[position - VGGT_CONTEXT_PAIRS + 1 : position + 1])
            _require(len(student) == 3 and len(vggt) == 5, "internal causal-window length mismatch")
            windows.append(
                CausalEndpoint(
                    sequence_id=sequence_id,
                    endpoint_index=endpoint,
                    student_indices=student,  # type: ignore[arg-type]
                    vggt_indices=vggt,  # type: ignore[arg-type]
                )
            )
    windows.sort(key=lambda item: item.endpoint_index)
    return tuple(windows)


def _record_set(root: Path) -> set[Path]:
    return {path.resolve() for path in root.rglob("*.pt") if path.is_file()}


def _validate_float_tensors(
    tensors: Mapping[str, Any],
    name: str,
    *,
    semantic_lr_error_mask: tuple[str, str] | None = None,
) -> int:
    sentinel_count = 0
    for tensor_name, tensor in tensors.items():
        _require(isinstance(tensor, torch.Tensor), f"{name}.{tensor_name} is not a tensor")
        if not tensor.is_floating_point() and not tensor.is_complex():
            continue
        if bool(torch.isfinite(tensor).all()):
            continue
        allowed_name = None if semantic_lr_error_mask is None else semantic_lr_error_mask[0]
        valid_name = None if semantic_lr_error_mask is None else semantic_lr_error_mask[1]
        _require(tensor_name == allowed_name, f"{name}.{tensor_name} contains non-finite values")
        valid = tensors.get(valid_name)
        _require(isinstance(valid, torch.Tensor) and valid.dtype == torch.bool, f"{name} valid mask is malformed")
        _require(valid.shape == tensor.shape, f"{name} LR-error/valid-mask shape mismatch")
        _require(bool(torch.isfinite(tensor[valid]).all()), f"{name} LR error is non-finite inside valid mask")
        nonfinite = ~torch.isfinite(tensor)
        _require(not bool((nonfinite & valid).any()), f"{name} LR sentinel appears inside valid mask")
        _require(not bool(torch.isnan(tensor).any()), f"{name} LR error contains NaN")
        _require(not bool(torch.isneginf(tensor).any()), f"{name} LR error contains -inf")
        sentinel_count += int(nonfinite.sum().item())
    return sentinel_count


def _identity_digest(rows: Sequence[tuple[int, str, str]]) -> str:
    payload = json.dumps(rows, sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(payload)


def _validate_ffs_cache(
    *,
    root: Path,
    role: str,
    manifest: ManifestSnapshot,
    source_hashes: dict[Path, str],
) -> CacheAuditResult:
    _require(root.is_dir(), f"FFS {role} cache root is missing: {root}")
    receipt_path = root / "run_receipt.json"
    receipt, receipt_bytes = _read_json(receipt_path, f"FFS {role} receipt")
    _require(receipt.get("schema_version") == 1, f"FFS {role} receipt schema mismatch")
    _require(receipt.get("manifest_sha256") == manifest.sha256, f"FFS {role} manifest hash mismatch")
    _require(Path(str(receipt.get("manifest", ""))).expanduser().resolve() == manifest.path, f"FFS {role} manifest path mismatch")
    selected = receipt.get("selected_records")
    written = receipt.get("written_records")
    reused = receipt.get("reused_records")
    _require(all(_is_int(value) and value >= 0 for value in (selected, written, reused)), f"FFS {role} coverage counts are invalid")
    _require(selected == len(manifest.records), f"FFS {role} selected coverage is incomplete")
    _require(written + reused == selected, f"FFS {role} written+reused coverage mismatch")
    component = f"ffs-{role}"
    identity = _validate_identity(receipt.get("identity"), component, f"FFS {role} receipt")
    config = receipt.get("config")
    _require(isinstance(config, dict), f"FFS {role} receipt config is missing")
    expected_config = {
        "observation": {"role": "observation", "scale": 2, "iterations": 4},
        "teacher": {"role": "teacher", "scale": 1, "iterations": 8},
    }[role]
    for key, value in expected_config.items():
        _require(config.get(key) == value, f"FFS {role} config {key} mismatch")
    _require(config.get("right_left_check") is True, f"FFS {role} lacks right-left check")
    _require(config.get("provisional_checkpoint_role") is False, f"FFS {role} uses a provisional checkpoint role")
    expected_paths = {_cache_path(root, record).resolve() for record in manifest.records}
    actual_paths = _record_set(root)
    _require(actual_paths == expected_paths, f"FFS {role} cache file coverage is not exact")

    file_hashes: dict[int, str] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    digest_rows: list[tuple[int, str, str]] = []
    total_bytes = 0
    lr_sentinel_count = 0
    for record in manifest.records:
        path = _cache_path(root, record).resolve()
        payload, cache_sha, byte_size = _load_safe_cache(path, f"FFS {role}[{record.index}]")
        total_bytes += byte_size
        file_hashes[record.index] = cache_sha
        digest_rows.append((record.index, str(path.relative_to(root)), cache_sha))
        _require(payload["identity"] == identity, f"FFS {role} identity mismatch at index {record.index}")
        metadata = payload["metadata"]
        _require(metadata.get("config") == config, f"FFS {role} config metadata mismatch at index {record.index}")
        source = metadata.get("source")
        _require(isinstance(source, Mapping), f"FFS {role} source metadata missing at {record.index}")
        _require(source.get("manifest_record") == record.raw, f"FFS {role} source record mismatch at {record.index}")
        _require(Path(str(source.get("manifest_path", ""))).expanduser().resolve() == manifest.path, f"FFS {role} source manifest path mismatch at {record.index}")
        for side, source_path in (("left", record.left_path), ("right", record.right_path)):
            if source_path not in source_hashes:
                source_hashes[source_path] = _sha256_file(source_path)
            _require(source.get(f"{side}_sha256") == source_hashes[source_path], f"FFS {role} {side} source hash mismatch at {record.index}")
        tensors = payload["tensors"]
        if role == "observation":
            primary = "observation_disparity_hr_px"
            lr_error = ("observation_left_right_error_lr_px", "observation_valid_mask")
        else:
            primary = "teacher_disparity_hr_px"
            lr_error = ("teacher_left_right_error_hr_px", "teacher_valid_mask")
        _require(isinstance(tensors.get(primary), torch.Tensor), f"FFS {role} primary disparity is missing")
        shapes[record.index] = tuple(tensors[primary].shape)
        lr_sentinel_count += _validate_float_tensors(
            tensors,
            f"FFS {role}[{record.index}]",
            semantic_lr_error_mask=lr_error,
        )
    return CacheAuditResult(
        report={
            "root": str(root),
            "canonical_receipt": str(receipt_path),
            "canonical_receipt_sha256": _sha256_bytes(receipt_bytes),
            "manifest_sha256": manifest.sha256,
            "selected_records": selected,
            "written_records": written,
            "reused_records": reused,
            "weights_only_safe_load_records": len(file_hashes),
            "finite_or_semantic_sentinel_records": len(file_hashes),
            "positive_infinity_lr_sentinel_elements_outside_valid_mask": lr_sentinel_count,
            "current_source_image_hashes_verified": len(manifest.records) * 2,
            "cache_bytes_read": total_bytes,
            "record_identity_digest_sha256": _identity_digest(digest_rows),
            "identity": identity,
            "config": config,
        },
        identity=identity,
        file_sha256_by_index=file_hashes,
        tensor_shape_by_index=shapes,
    )


def _validate_raw_vggt(
    *,
    root: Path,
    manifest: ManifestSnapshot,
    endpoints: Sequence[CausalEndpoint],
    source_hashes: dict[Path, str],
) -> tuple[CacheAuditResult, list[dict[str, Any]], str, str]:
    _require(root.is_dir(), f"raw VGGT cache root is missing: {root}")
    receipt_path = root / "run_receipt.json"
    receipt, receipt_bytes = _read_json(receipt_path, "raw VGGT receipt")
    receipt_sha = _sha256_bytes(receipt_bytes)
    _require(receipt.get("schema_version") == 1, "raw VGGT receipt schema mismatch")
    _require(receipt.get("manifest_sha256") == manifest.sha256, "raw VGGT manifest hash mismatch")
    _require(Path(str(receipt.get("manifest", ""))).expanduser().resolve() == manifest.path, "raw VGGT manifest path mismatch")
    available = receipt.get("available_windows")
    selected = receipt.get("selected_windows")
    written = receipt.get("written_records")
    reused = receipt.get("reused_records")
    _require(all(_is_int(value) and value >= 0 for value in (available, selected, written, reused)), "raw VGGT coverage counts are invalid")
    _require(available == selected == len(endpoints), "raw VGGT selected windows do not equal all available windows")
    _require(written + reused == selected, "raw VGGT written+reused coverage mismatch")
    identity = _validate_identity(receipt.get("identity"), "vggt-omega", "raw VGGT receipt")
    config = receipt.get("config")
    _require(isinstance(config, dict), "raw VGGT config is missing")
    _require(config.get("causal") is True, "raw VGGT receipt is not causal")
    _require(config.get("context_pairs") == VGGT_CONTEXT_PAIRS, "raw VGGT context is not five pairs")
    _require(config.get("view_order") == list(VIEW_ORDER), "raw VGGT L/R view order mismatch")
    _require(config.get("current_left_view_index") == 8, "raw VGGT current-left index mismatch")

    cache_manifest_path = root / "cache_manifest.jsonl"
    rows, cache_manifest_bytes = _read_jsonl(cache_manifest_path, "raw VGGT cache manifest")
    cache_manifest_sha = _sha256_bytes(cache_manifest_bytes)
    _require(len(rows) == len(endpoints), "raw VGGT cache manifest coverage mismatch")
    expected_paths: set[Path] = set()
    file_hashes: dict[int, str] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    digest_rows: list[tuple[int, str, str]] = []
    status_written = 0
    status_reused = 0
    total_bytes = 0
    for selection_index, (row, endpoint) in enumerate(zip(rows, endpoints, strict=True)):
        target = manifest.records[endpoint.endpoint_index]
        _require(row.get("selection_index") == selection_index, f"raw VGGT selection index mismatch at {selection_index}")
        _require(row.get("target_manifest_index") == endpoint.endpoint_index, f"raw VGGT target index mismatch at {selection_index}")
        for key, expected in (("sequence_id", target.sequence_id), ("frame_id", target.frame_id), ("timestamp", target.raw["timestamp"])):
            _require(row.get(key) == expected, f"raw VGGT {key} mismatch at {selection_index}")
        status = row.get("status")
        _require(isinstance(status, str) and (status == "written" or status.startswith("reused")), f"raw VGGT status is invalid at {selection_index}")
        status_written += int(status == "written")
        status_reused += int(status.startswith("reused"))
        path = Path(str(row.get("cache_path", ""))).expanduser().resolve()
        canonical = _cache_path(root, target).resolve()
        _require(path == canonical and _is_within(path, root), f"raw VGGT cache path mismatch at {selection_index}")
        expected_paths.add(path)
        payload, cache_sha, byte_size = _load_safe_cache(path, f"raw VGGT[{selection_index}]")
        total_bytes += byte_size
        file_hashes[endpoint.endpoint_index] = cache_sha
        digest_rows.append((endpoint.endpoint_index, str(path.relative_to(root)), cache_sha))
        _require(payload["identity"] == identity, f"raw VGGT identity mismatch at {selection_index}")
        metadata = payload["metadata"]
        _require(metadata.get("config") == config, f"raw VGGT config metadata mismatch at {selection_index}")
        source = metadata.get("source")
        _require(isinstance(source, Mapping), f"raw VGGT source metadata missing at {selection_index}")
        _require(source.get("manifest_sha256") == manifest.sha256, f"raw VGGT source manifest hash mismatch at {selection_index}")
        _require(Path(str(source.get("manifest_path", ""))).expanduser().resolve() == manifest.path, f"raw VGGT source manifest path mismatch at {selection_index}")
        _require(source.get("manifest_indices") == list(endpoint.vggt_indices), f"raw VGGT context indices mismatch at {selection_index}")
        _require(source.get("manifest_records") == [manifest.records[index].raw for index in endpoint.vggt_indices], f"raw VGGT context records mismatch at {selection_index}")
        for key, expected in (("target_manifest_index", endpoint.endpoint_index), ("target_sequence_id", target.sequence_id), ("target_frame_id", target.frame_id), ("target_timestamp", target.raw["timestamp"]), ("view_order", list(VIEW_ORDER)), ("causal", True)):
            _require(source.get(key) == expected, f"raw VGGT source {key} mismatch at {selection_index}")
        ordered_images = source.get("ordered_images")
        _require(isinstance(ordered_images, list) and len(ordered_images) == 10, f"raw VGGT ordered images malformed at {selection_index}")
        expected_images: list[tuple[str, Path]] = []
        for context_index in endpoint.vggt_indices:
            context_record = manifest.records[context_index]
            expected_images.extend((("left", context_record.left_path), ("right", context_record.right_path)))
        for view_index, (image, view_label, (side, path_expected)) in enumerate(zip(ordered_images, VIEW_ORDER, expected_images, strict=True)):
            _require(isinstance(image, Mapping), f"raw VGGT ordered image {view_index} malformed")
            _require(image.get("view_index") == view_index and image.get("view_label") == view_label, f"raw VGGT view order mismatch at {selection_index}:{view_index}")
            _require(Path(str(image.get("path", ""))).expanduser().resolve() == path_expected, f"raw VGGT image path mismatch at {selection_index}:{view_index}")
            if path_expected not in source_hashes:
                source_hashes[path_expected] = _sha256_file(path_expected)
            _require(image.get("sha256") == source_hashes[path_expected], f"raw VGGT image SHA mismatch at {selection_index}:{view_index}")
            _require(image.get("size_bytes") == path_expected.stat().st_size, f"raw VGGT image size mismatch at {selection_index}:{view_index}")
        tensors = payload["tensors"]
        _validate_float_tensors(tensors, f"raw VGGT[{selection_index}]")
        depth = tensors.get("vggt_depth_current_left_arbitrary")
        extrinsics = tensors.get("vggt_extrinsics_camera_from_world")
        _require(isinstance(depth, torch.Tensor) and depth.ndim == 3 and depth.shape[0] == 1, f"raw VGGT current depth shape mismatch at {selection_index}")
        _require(isinstance(extrinsics, torch.Tensor) and tuple(extrinsics.shape) == (10, 3, 4), f"raw VGGT extrinsics shape mismatch at {selection_index}")
        shapes[endpoint.endpoint_index] = tuple(depth.shape)
    _require(status_written == written and status_reused == reused, "raw VGGT manifest status counts disagree with receipt")
    _require(_record_set(root) == expected_paths, "raw VGGT cache file coverage is not exact")
    return (
        CacheAuditResult(
            report={
                "root": str(root),
                "canonical_receipt": str(receipt_path),
                "canonical_receipt_sha256": receipt_sha,
                "cache_manifest": str(cache_manifest_path),
                "cache_manifest_sha256": cache_manifest_sha,
                "manifest_binding": "derived receipt binds both this canonical receipt SHA and cache-manifest SHA",
                "available_windows": available,
                "selected_windows": selected,
                "written_records": written,
                "reused_records": reused,
                "weights_only_safe_finite_records": len(file_hashes),
                "cache_bytes_read": total_bytes,
                "record_identity_digest_sha256": _identity_digest(digest_rows),
                "identity": identity,
                "config": config,
            },
            identity=identity,
            file_sha256_by_index=file_hashes,
            tensor_shape_by_index=shapes,
        ),
        rows,
        receipt_sha,
        cache_manifest_sha,
    )


def _scalar_bool(tensors: Mapping[str, Any], name: str, record_name: str) -> bool:
    value = tensors.get(name)
    _require(isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.numel() == 1, f"{record_name}.{name} is not a boolean scalar")
    return bool(value.item())


def _validate_derived_receipt(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    manifest_sha: str,
    raw_root: Path,
    observation_root: Path,
    raw_receipt_sha: str,
    raw_manifest_sha: str,
    raw_identity: Mapping[str, Any],
    observation_identity: Mapping[str, Any],
    available: int,
) -> None:
    _require(receipt.get("schema_version") == 1, "derived receipt schema mismatch")
    _require(receipt.get("component") == "vggt-ffs-derived-geometry-batch", "derived receipt component mismatch")
    selection = receipt.get("selection")
    counts = receipt.get("counts")
    _require(isinstance(selection, Mapping) and isinstance(counts, Mapping), "derived coverage fields are missing")
    _require(selection.get("start_window") == 0, "derived canonical selection does not start at zero")
    _require(selection.get("limit") is None, "derived canonical selection has a limit")
    _require(selection.get("selected_windows") == available, "derived selection does not cover all available windows")
    selected = counts.get("selected")
    written = counts.get("written")
    reused = counts.get("reused")
    _require(all(_is_int(value) and value >= 0 for value in (selected, written, reused)), "derived count values are invalid")
    _require(selected == available and written + reused == selected, "derived written+reused coverage mismatch")
    for valid, rejected in (("pose_valid", "pose_rejected"), ("static_prior_valid", "static_prior_rejected")):
        _require(_is_int(counts.get(valid)) and _is_int(counts.get(rejected)), f"derived {valid} counts are invalid")
        _require(counts[valid] + counts[rejected] == selected, f"derived {valid}/{rejected} coverage mismatch")
    inputs = receipt.get("inputs")
    output = receipt.get("output")
    _require(isinstance(inputs, Mapping) and isinstance(output, Mapping), "derived input/output lineage is missing")
    _require(Path(str(inputs.get("vggt_root", ""))).expanduser().resolve() == raw_root, "derived raw VGGT root mismatch")
    _require(Path(str(inputs.get("ffs_root", ""))).expanduser().resolve() == observation_root, "derived FFS root mismatch")
    _require(inputs.get("vggt_available_windows") == available, "derived raw available-window count mismatch")
    _require(inputs.get("vggt_cache_manifest_sha256") == raw_manifest_sha, "derived receipt does not bind raw VGGT manifest")
    _require(Path(str(output.get("root", ""))).expanduser().resolve() == root, "derived output root mismatch")
    _require(output.get("cache_manifest_sha256") == manifest_sha, "derived receipt manifest hash mismatch")
    _require(output.get("run_cache_manifest_sha256") == manifest_sha, "derived run-manifest hash mismatch")
    run_manifest = Path(str(output.get("run_cache_manifest", ""))).expanduser().resolve()
    _require(run_manifest.is_file() and _sha256_file(run_manifest) == manifest_sha, "derived preserved run manifest cannot prove canonical content")
    raw_audit = receipt.get("raw_input_audit")
    safe_zero = receipt.get("safe_zero_audit")
    _require(isinstance(raw_audit, Mapping) and raw_audit.get("passed") is True, "derived raw_input_audit did not pass")
    _require(isinstance(safe_zero, Mapping) and safe_zero.get("passed") is True, "derived safe_zero_audit did not pass")
    _require(raw_audit.get("canonical_receipt_complete_manifest_coverage") is True, "derived receipt lacks raw canonical coverage proof")
    _require(raw_audit.get("canonical_receipt_sha256") == raw_receipt_sha, "derived receipt does not bind raw canonical receipt")
    _require(raw_audit.get("ffs_identity") == observation_identity, "derived receipt FFS identity mismatch")
    _require(raw_audit.get("vggt_identity") == raw_identity, "derived receipt raw VGGT identity mismatch")
    for field in ("all_float_tensors_finite_records", "causal_target_valid_records", "ffs_identity_match_records", "vggt_identity_match_records", "weights_only_safe_load_records"):
        _require(raw_audit.get(field) == available, f"derived raw_input_audit {field} is incomplete")
    for field in ("all_float_tensors_finite_records", "manifest_metadata_tensor_validity_consistent_records", "records_audited", "weights_only_safe_load_records"):
        _require(safe_zero.get(field) == available, f"derived safe_zero_audit {field} is incomplete")
    _require(safe_zero.get("pose_rejected_zero_temporal_extrinsics") == counts["pose_rejected"], "derived pose safe-zero count mismatch")
    _require(safe_zero.get("static_rejected_zero_prior_tensors") == counts["static_prior_rejected"], "derived prior safe-zero count mismatch")
    canonical_update = receipt.get("canonical_update")
    _require(isinstance(canonical_update, Mapping), "derived canonical_update proof is missing")
    _require(canonical_update.get("current_selected_windows") == available, "derived current canonical coverage mismatch")
    _require(canonical_update.get("existing_selected_windows") == available, "derived previous canonical coverage mismatch")
    config = receipt.get("config")
    _require(isinstance(config, Mapping), "derived config is missing")
    required_config = {
        "algorithm": "baseline_metric_scale+scale_only_alignment+strict_pose_quality",
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    for key, expected in required_config.items():
        _require(config.get(key) == expected, f"derived geometry config {key} mismatch")
    _require(receipt_path.resolve() == root / "run_receipt.json", "derived receipt is not canonical")


def _validate_derived(
    *,
    root: Path,
    manifest: ManifestSnapshot,
    endpoints: Sequence[CausalEndpoint],
    observation_root: Path,
    observation: CacheAuditResult,
    raw_root: Path,
    raw: CacheAuditResult,
    raw_receipt_sha: str,
    raw_manifest_sha: str,
) -> tuple[Mapping[str, Any], Mapping[int, tuple[bool, bool]], list[dict[str, Any]]]:
    _require(root.is_dir(), f"derived cache root is missing: {root}")
    receipt_path = root / "run_receipt.json"
    receipt, receipt_bytes = _read_json(receipt_path, "derived receipt")
    cache_manifest_path = root / "cache_manifest.jsonl"
    rows, manifest_bytes = _read_jsonl(cache_manifest_path, "derived cache manifest")
    manifest_sha = _sha256_bytes(manifest_bytes)
    _require(len(rows) == len(endpoints), "derived cache manifest coverage mismatch")
    _validate_derived_receipt(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
        manifest_sha=manifest_sha,
        raw_root=raw_root,
        observation_root=observation_root,
        raw_receipt_sha=raw_receipt_sha,
        raw_manifest_sha=raw_manifest_sha,
        raw_identity=raw.identity,
        observation_identity=observation.identity,
        available=len(endpoints),
    )
    config = receipt["config"]
    expected_paths: set[Path] = set()
    gates: dict[int, tuple[bool, bool]] = {}
    digest_rows: list[tuple[int, str, str]] = []
    total_bytes = 0
    status_written = 0
    status_reused = 0
    by_sequence: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "selected": 0,
            "pose_valid": 0,
            "pose_rejected": 0,
            "static_prior_valid": 0,
            "static_prior_rejected": 0,
        }
    )
    for selection_index, (row, endpoint) in enumerate(zip(rows, endpoints, strict=True)):
        record = manifest.records[endpoint.endpoint_index]
        _require(row.get("selection_index") == selection_index, f"derived selection index mismatch at {selection_index}")
        _require(row.get("target_manifest_index") == endpoint.endpoint_index, f"derived target index mismatch at {selection_index}")
        for key, expected in (("sequence_id", record.sequence_id), ("frame_id", record.frame_id), ("timestamp", record.raw["timestamp"])):
            _require(row.get(key) == expected, f"derived {key} mismatch at {selection_index}")
        pose_valid = row.get("pose_valid")
        static_valid = row.get("static_prior_valid")
        _require(isinstance(pose_valid, bool) and isinstance(static_valid, bool), f"derived gate flags are malformed at {selection_index}")
        gates[endpoint.endpoint_index] = (pose_valid, static_valid)
        sequence_counts = by_sequence[record.sequence_id]
        sequence_counts["selected"] += 1
        sequence_counts["pose_valid" if pose_valid else "pose_rejected"] += 1
        sequence_counts["static_prior_valid" if static_valid else "static_prior_rejected"] += 1
        status = row.get("status")
        _require(isinstance(status, str) and (status == "written" or status.startswith("reused")), f"derived status is invalid at {selection_index}")
        status_written += int(status == "written")
        status_reused += int(status.startswith("reused"))
        cache_path = Path(str(row.get("cache_path", ""))).expanduser().resolve()
        observation_path = _cache_path(observation_root, record).resolve()
        raw_path = _cache_path(raw_root, record).resolve()
        _require(cache_path == _cache_path(root, record).resolve() and _is_within(cache_path, root), f"derived cache path mismatch at {selection_index}")
        _require(Path(str(row.get("ffs_cache_path", ""))).expanduser().resolve() == observation_path, f"derived manifest FFS path mismatch at {selection_index}")
        _require(Path(str(row.get("vggt_cache_path", ""))).expanduser().resolve() == raw_path, f"derived manifest VGGT path mismatch at {selection_index}")
        _require(row.get("ffs_cache_sha256") == observation.file_sha256_by_index[endpoint.endpoint_index], f"derived manifest FFS SHA mismatch at {selection_index}")
        _require(row.get("vggt_cache_sha256") == raw.file_sha256_by_index[endpoint.endpoint_index], f"derived manifest VGGT SHA mismatch at {selection_index}")
        expected_paths.add(cache_path)
        payload, cache_sha, byte_size = _load_safe_cache(cache_path, f"derived[{selection_index}]")
        total_bytes += byte_size
        _require(row.get("cache_sha256") == cache_sha, f"derived cache SHA mismatch at {selection_index}")
        digest_rows.append((endpoint.endpoint_index, str(cache_path.relative_to(root)), cache_sha))
        _validate_identity(payload["identity"], "vggt-ffs-derived-geometry", f"derived[{selection_index}]")
        metadata = payload["metadata"]
        _require(metadata.get("config") == config, f"derived config metadata mismatch at {selection_index}")
        _require(metadata.get("target") == {"sequence_id": record.sequence_id, "frame_id": record.frame_id, "timestamp": record.raw["timestamp"]}, f"derived target metadata mismatch at {selection_index}")
        source = metadata.get("source")
        _require(isinstance(source, Mapping), f"derived source metadata missing at {selection_index}")
        _require(Path(str(source.get("ffs_cache_path", ""))).expanduser().resolve() == observation_path, f"derived source FFS path mismatch at {selection_index}")
        _require(Path(str(source.get("vggt_cache_path", ""))).expanduser().resolve() == raw_path, f"derived source VGGT path mismatch at {selection_index}")
        _require(source.get("ffs_cache_sha256") == observation.file_sha256_by_index[endpoint.endpoint_index], f"derived source FFS SHA mismatch at {selection_index}")
        _require(source.get("vggt_cache_sha256") == raw.file_sha256_by_index[endpoint.endpoint_index], f"derived source VGGT SHA mismatch at {selection_index}")
        linkage = source.get("linkage")
        _require(isinstance(linkage, Mapping), f"derived linkage missing at {selection_index}")
        _require(linkage.get("target_manifest_record") == record.raw, f"derived linked manifest record mismatch at {selection_index}")
        for key, expected in (("target_sequence_id", record.sequence_id), ("target_frame_id", record.frame_id), ("target_timestamp", record.raw["timestamp"]), ("vggt_raw_identity", raw.identity), ("ffs_raw_identity", observation.identity)):
            _require(linkage.get(key) == expected, f"derived linkage {key} mismatch at {selection_index}")
        previous = manifest.records[endpoint.vggt_indices[-2]]
        previous_left = linkage.get("previous_left")
        current_left = linkage.get("current_left")
        _require(isinstance(previous_left, Mapping) and isinstance(current_left, Mapping), f"derived left-view linkage missing at {selection_index}")
        _require(previous_left.get("view_index") == 6 and previous_left.get("view_label") == "L[t-1]", f"derived previous-left view index mismatch at {selection_index}")
        _require(current_left.get("view_index") == 8 and current_left.get("view_label") == "L[t]", f"derived current-left view index mismatch at {selection_index}")
        _require(Path(str(previous_left.get("path", ""))).expanduser().resolve() == previous.left_path, f"derived previous-left path mismatch at {selection_index}")
        _require(Path(str(current_left.get("path", ""))).expanduser().resolve() == record.left_path, f"derived current-left path mismatch at {selection_index}")
        quality = metadata.get("pose_quality")
        alignment = quality.get("alignment") if isinstance(quality, Mapping) else None
        _require(isinstance(quality, Mapping) and isinstance(alignment, Mapping), f"derived quality metadata missing at {selection_index}")
        _require(quality.get("pose_valid") is pose_valid, f"derived pose gate metadata mismatch at {selection_index}")
        _require(alignment.get("static_prior_valid") is static_valid, f"derived static gate metadata mismatch at {selection_index}")
        tensors = payload["tensors"]
        _validate_float_tensors(tensors, f"derived[{selection_index}]")
        tensor_pose = _scalar_bool(tensors, "temporal_pose_valid", f"derived[{selection_index}]")
        tensor_static = _scalar_bool(tensors, "static_prior_valid", f"derived[{selection_index}]")
        _require(tensor_pose == pose_valid and tensor_static == static_valid, f"derived tensor/manifest gate mismatch at {selection_index}")
        extrinsics = tensors.get("vggt_extrinsics_camera_from_world_metric_temporal")
        disparity = tensors.get("vggt_disparity_current_left_aligned_hr_px")
        confidence = tensors.get("vggt_aligned_confidence")
        valid_mask = tensors.get("vggt_aligned_valid_mask")
        _require(isinstance(extrinsics, torch.Tensor) and tuple(extrinsics.shape) == (10, 3, 4), f"derived temporal pose shape mismatch at {selection_index}")
        _require(isinstance(disparity, torch.Tensor) and isinstance(confidence, torch.Tensor) and isinstance(valid_mask, torch.Tensor), f"derived prior tensors missing at {selection_index}")
        observation_shape = observation.tensor_shape_by_index[endpoint.endpoint_index]
        _require(tuple(disparity.shape[-2:]) == tuple(observation_shape[-2:]), f"derived/FFS LR grid mismatch at {selection_index}")
        if not pose_valid:
            _require(not bool((extrinsics != 0).any()), f"derived rejected pose is not zero-filled at {selection_index}")
        if not static_valid:
            _require(not bool((disparity != 0).any()), f"derived rejected disparity prior is not zero-filled at {selection_index}")
            _require(not bool((confidence != 0).any()), f"derived rejected confidence is not zero-filled at {selection_index}")
            _require(not bool(valid_mask.any()), f"derived rejected valid mask is not empty at {selection_index}")
    counts = receipt["counts"]
    _require(status_written == counts["written"] and status_reused == counts["reused"], "derived manifest status counts disagree with receipt")
    _require(_record_set(root) == expected_paths, "derived cache file coverage is not exact")
    receipt_by_sequence = receipt.get("by_sequence")
    _require(isinstance(receipt_by_sequence, Mapping), "derived per-sequence receipt counts are missing")
    for sequence_id, computed in by_sequence.items():
        item = receipt_by_sequence.get(sequence_id)
        _require(isinstance(item, Mapping) and item.get("counts") == computed, f"derived per-sequence counts mismatch for {sequence_id}")
    _require(set(receipt_by_sequence) == set(by_sequence), "derived receipt has unexpected sequence counts")
    return (
        {
            "root": str(root),
            "canonical_receipt": str(receipt_path),
            "canonical_receipt_sha256": _sha256_bytes(receipt_bytes),
            "cache_manifest": str(cache_manifest_path),
            "cache_manifest_sha256": manifest_sha,
            "selection": receipt["selection"],
            "counts": counts,
            "weights_only_safe_finite_records": len(rows),
            "cache_bytes_read": total_bytes,
            "record_identity_digest_sha256": _identity_digest(digest_rows),
            "raw_input_audit": receipt["raw_input_audit"],
            "safe_zero_audit": receipt["safe_zero_audit"],
        },
        gates,
        rows,
    )


def _endpoint_digest(endpoints: Sequence[CausalEndpoint]) -> str:
    rows = [
        {
            "sequence_id": endpoint.sequence_id,
            "endpoint_index": endpoint.endpoint_index,
            "student_indices": list(endpoint.student_indices),
            "vggt_indices": list(endpoint.vggt_indices),
        }
        for endpoint in endpoints
    ]
    return _sha256_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _availability_report(
    manifest: ManifestSnapshot,
    endpoints: Sequence[CausalEndpoint],
    gates: Mapping[int, tuple[bool, bool]],
) -> tuple[dict[str, Any], tuple[CausalEndpoint, ...], dict[str, int]]:
    derived_indices = set(gates)
    evaluable = tuple(
        endpoint
        for endpoint in endpoints
        if all(index in derived_indices for index in endpoint.student_indices)
    )
    _require(bool(evaluable), "no Stage-B T=3 window has derived geometry for all three times")
    by_sequence: dict[str, Any] = {}
    total_evaluable_current_pose = 0
    total_evaluable_current_static = 0
    total_time_pose = 0
    total_time_static = 0
    for sequence_id in manifest.sequence_order:
        indices = manifest.indices_by_sequence[sequence_id]
        sequence_endpoints = [item for item in endpoints if item.sequence_id == sequence_id]
        sequence_evaluable = [item for item in evaluable if item.sequence_id == sequence_id]
        all_pose = sum(gates[item.endpoint_index][0] for item in sequence_endpoints)
        all_static = sum(gates[item.endpoint_index][1] for item in sequence_endpoints)
        current_pose = sum(gates[item.endpoint_index][0] for item in sequence_evaluable)
        current_static = sum(gates[item.endpoint_index][1] for item in sequence_evaluable)
        time_pose = sum(gates[index][0] for item in sequence_evaluable for index in item.student_indices)
        time_static = sum(gates[index][1] for item in sequence_evaluable for index in item.student_indices)
        total_evaluable_current_pose += current_pose
        total_evaluable_current_static += current_static
        total_time_pose += time_pose
        total_time_static += time_static
        first_record = manifest.records[indices[0]]
        last_record = manifest.records[indices[-1]]
        by_sequence[sequence_id] = {
            "manifest_records": len(indices),
            "manifest_index_range": [indices[0], indices[-1]],
            "frame_id_range": [first_record.frame_id, last_record.frame_id],
            "raw_and_derived_endpoints": len(sequence_endpoints),
            "evaluable_t3_endpoints": len(sequence_evaluable),
            "evaluable_endpoint_index_range": (
                None
                if not sequence_evaluable
                else [sequence_evaluable[0].endpoint_index, sequence_evaluable[-1].endpoint_index]
            ),
            "all_derived_endpoint_gates": {
                "pose_valid": all_pose,
                "pose_rejected": len(sequence_endpoints) - all_pose,
                "static_prior_valid": all_static,
                "static_prior_rejected": len(sequence_endpoints) - all_static,
            },
            "evaluable_current_endpoint_gates": {
                "pose_valid": current_pose,
                "pose_rejected": len(sequence_evaluable) - current_pose,
                "static_prior_valid": current_static,
                "static_prior_rejected": len(sequence_evaluable) - current_static,
            },
            "evaluable_t3_time_slice_gates": {
                "time_slices": 3 * len(sequence_evaluable),
                "pose_valid": time_pose,
                "pose_rejected": 3 * len(sequence_evaluable) - time_pose,
                "static_prior_valid": time_static,
                "static_prior_rejected": 3 * len(sequence_evaluable) - time_static,
            },
        }
    totals = {
        "evaluable_current_pose_valid": total_evaluable_current_pose,
        "evaluable_current_static_prior_valid": total_evaluable_current_static,
        "evaluable_time_slice_pose_valid": total_time_pose,
        "evaluable_time_slice_static_prior_valid": total_time_static,
    }
    return by_sequence, evaluable, totals


def audit_temporal_inputs(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    observation_root: str | Path,
    teacher_root: str | Path,
    raw_vggt_root: str | Path,
    derived_root: str | Path,
) -> dict[str, Any]:
    """Perform an exhaustive, read-only audit and return strict JSON data."""

    train = _parse_manifest(Path(train_manifest).expanduser().resolve(), "train manifest")
    validation = _parse_manifest(Path(validation_manifest).expanduser().resolve(), "validation manifest")
    overlap = sorted(set(train.sequence_order).intersection(validation.sequence_order))
    _require(not overlap, f"train/validation sequence leakage: {overlap}")
    endpoints = _build_causal_endpoints(train)
    _require(bool(endpoints), "train manifest has no causal five-pair endpoints")
    source_hashes: dict[Path, str] = {}
    observation_path = Path(observation_root).expanduser().resolve()
    teacher_path = Path(teacher_root).expanduser().resolve()
    raw_path = Path(raw_vggt_root).expanduser().resolve()
    derived_path = Path(derived_root).expanduser().resolve()
    roots = (observation_path, teacher_path, raw_path, derived_path)
    _require(len(set(roots)) == 4, "cache roots must be distinct")

    observation = _validate_ffs_cache(
        root=observation_path,
        role="observation",
        manifest=train,
        source_hashes=source_hashes,
    )
    teacher = _validate_ffs_cache(
        root=teacher_path,
        role="teacher",
        manifest=train,
        source_hashes=source_hashes,
    )
    raw, raw_rows, raw_receipt_sha, raw_manifest_sha = _validate_raw_vggt(
        root=raw_path,
        manifest=train,
        endpoints=endpoints,
        source_hashes=source_hashes,
    )
    derived, gates, derived_rows = _validate_derived(
        root=derived_path,
        manifest=train,
        endpoints=endpoints,
        observation_root=observation_path,
        observation=observation,
        raw_root=raw_path,
        raw=raw,
        raw_receipt_sha=raw_receipt_sha,
        raw_manifest_sha=raw_manifest_sha,
    )
    _require(
        [row["target_manifest_index"] for row in raw_rows]
        == [row["target_manifest_index"] for row in derived_rows],
        "raw/derived endpoint ordering differs",
    )
    availability, evaluable, gate_totals = _availability_report(train, endpoints, gates)
    for endpoint in evaluable:
        target = train.records[endpoint.endpoint_index]
        for index in (*endpoint.student_indices, *endpoint.vggt_indices):
            source = train.records[index]
            _require(source.sequence_id == target.sequence_id, "causal window crosses a sequence")
            _require(index <= endpoint.endpoint_index, "causal window accesses a future manifest index")
            _require(source.timestamp <= target.timestamp, "causal window accesses a future timestamp")

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": "stage-b-training-input-audit",
        "status": "PASS",
        "read_only": True,
        "manifests": {
            "train": {
                "path": str(train.path),
                "sha256": train.sha256,
                "byte_size": train.byte_size,
                "record_count": len(train.records),
                "sequence_order": list(train.sequence_order),
                "sequence_counts": {
                    sequence_id: len(train.indices_by_sequence[sequence_id])
                    for sequence_id in train.sequence_order
                },
                "strict_sequence_ordering": True,
            },
            "validation": {
                "path": str(validation.path),
                "sha256": validation.sha256,
                "byte_size": validation.byte_size,
                "record_count": len(validation.records),
                "sequence_order": list(validation.sequence_order),
            },
            "train_validation_sequence_disjoint": True,
            "overlapping_sequences": [],
        },
        "caches": {
            "ffs_observation": observation.report,
            "ffs_teacher": teacher.report,
            "raw_vggt": raw.report,
            "derived_geometry": derived,
        },
        "causal_contract": {
            "student_sequence_length": STUDENT_SEQUENCE_LENGTH,
            "vggt_context_pairs": VGGT_CONTEXT_PAIRS,
            "view_order": list(VIEW_ORDER),
            "expected_raw_and_derived_endpoints": len(endpoints),
            "expected_endpoint_digest_sha256": _endpoint_digest(endpoints),
            "evaluable_t3_endpoints": len(evaluable),
            "evaluable_endpoint_digest_sha256": _endpoint_digest(evaluable),
            "no_future_manifest_index_access": True,
            "no_future_timestamp_access": True,
            "no_sequence_boundary_crossing": True,
            "availability_by_sequence": availability,
            "gate_totals": gate_totals,
        },
        "lineage_closure": {
            "source_images_current_hash_verified": True,
            "source_image_files_hashed": len(source_hashes),
            "ffs_receipts_bind_train_manifest": True,
            "raw_vggt_receipt_binds_train_manifest": True,
            "derived_receipt_binds_raw_receipt_sha256": True,
            "derived_receipt_binds_raw_manifest_sha256": True,
            "derived_manifest_binds_per_record_ffs_and_raw_sha256": True,
            "per_record_weights_only_safe_load": True,
            "per_record_tensor_finiteness_or_declared_ffs_sentinel": True,
            "coverage_proven_by_receipts_and_rows_not_file_count_alone": True,
        },
    }
    _finite_tree(report, "audit report")
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def _inside_any(path: Path, roots: Sequence[Path]) -> bool:
    return any(_is_within(path, root) for root in roots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--raw-vggt-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(
        path.expanduser().resolve()
        for path in (
            args.observation_root,
            args.teacher_root,
            args.raw_vggt_root,
            args.derived_root,
        )
    )
    output = args.json_out.expanduser().resolve()
    try:
        _require(not _inside_any(output, roots), "--json-out must be outside every audited cache root")
        report = audit_temporal_inputs(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            observation_root=roots[0],
            teacher_root=roots[1],
            raw_vggt_root=roots[2],
            derived_root=roots[3],
        )
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print("audit:stage-b-inputs: PASS")
        print(f"receipt: {output}")
        return 0
    except (OSError, TemporalInputAuditError) as exc:
        failure = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "component": "stage-b-training-input-audit",
            "status": "FAIL",
            "error": str(exc),
        }
        sys.stderr.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
