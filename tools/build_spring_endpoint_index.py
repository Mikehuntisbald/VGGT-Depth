#!/usr/bin/env python3
"""Generate a manifest-bound common endpoint index for Spring evaluation.

With no ``--source`` arguments, the script emits every endpoint for which a
causal T=3/VGGT-5 window can be formed.  ``--derived-cache-root`` can further
restrict this to endpoints whose three student frames are present in a frozen
VGGT-derived cache (the S4/S5/S6 common set).  When source endpoint lists are
given, it emits their strict intersection.  Source lists must be bound to the
same manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.endpoint_selection import (  # noqa: E402
    ENDPOINT_ID_HASH_ALGORITHM,
    EndpointSelectionError,
    load_endpoint_index,
    write_endpoint_index,
)
from data.manifest import load_manifest  # noqa: E402
from data.training_dataset import build_causal_windows  # noqa: E402


DEFAULT_SEQUENCE_WARMUP = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write a small immutable receipt without exposing a partial JSON file."""

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
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_warmup_indices(
    records: Sequence[object], indices: Sequence[int], warmup: int
) -> list[int]:
    """Filter endpoint indices by *per-sequence* manifest position.

    The manifest is the authority for sequence order.  We deliberately count
    positions over all records, rather than over the already-filtered source
    set, so ``--sequence-warmup 6`` always means “discard frames 1--6” even
    when a caller intersects a prior endpoint list or a derived cache.
    """

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("--sequence-warmup must be a non-negative integer")
    positions: dict[str, int] = {}
    allowed: set[int] = set()
    for manifest_index, record in enumerate(records):
        sequence_id = str(getattr(record, "sequence_id"))
        position = positions.get(sequence_id, 0)
        positions[sequence_id] = position + 1
        if position >= warmup:
            allowed.add(manifest_index)
    return [index for index in indices if index in allowed]


def _source_identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a manifest-bound Spring common endpoint index"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="existing endpoint index; repeat to emit the strict intersection",
    )
    parser.add_argument(
        "--source-records",
        type=Path,
        action="append",
        help=(
            "evaluator per_record_metrics.jsonl; repeat to intersect endpoint "
            "identities recovered from prior arms"
        ),
    )
    parser.add_argument(
        "--derived-cache-root",
        type=Path,
        help=(
            "optional full derived-cache root; keep only causal windows whose "
            "three student endpoint records have derived entries"
        ),
    )
    parser.add_argument(
        "--sequence-warmup",
        type=int,
        default=DEFAULT_SEQUENCE_WARMUP,
        help=(
            "discard this many leading manifest positions independently per "
            "sequence (formal Spring common domain: 6)"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="optional receipt path (default: OUTPUT.with_suffix('.receipt.json'))",
    )
    return parser


def _read_source_indices(paths: Sequence[Path], manifest: Path) -> set[int]:
    selected: set[int] | None = None
    for path in paths:
        source = load_endpoint_index(path, manifest_path=manifest)
        current = set(source.manifest_indices)
        selected = current if selected is None else selected.intersection(current)
    return set() if selected is None else selected


def _read_source_record_indices(paths: Sequence[Path], manifest: Path) -> set[int]:
    """Recover unique endpoint IDs from evaluator per-record JSONL artifacts."""

    records = load_manifest(manifest)
    by_key = {
        (record.sequence_id, record.frame_id): index
        for index, record in enumerate(records)
    }
    if len(by_key) != len(records):
        raise EndpointSelectionError(
            "manifest has duplicate sequence_id/frame_id endpoint keys"
        )
    selected: set[int] | None = None
    for path in paths:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        current: set[int] = set()
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise EndpointSelectionError(
                        f"malformed source-records JSONL row {source}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EndpointSelectionError(
                        f"source-records row is not an object {source}:{line_number}"
                    )
                raw_index = row.get("manifest_index")
                if raw_index is not None:
                    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                        raise EndpointSelectionError(
                            f"source-records manifest_index is not integer {source}:{line_number}"
                        )
                    index = raw_index
                else:
                    key = (row.get("sequence_id"), row.get("frame_id"))
                    if key not in by_key:
                        raise EndpointSelectionError(
                            f"source-records row lacks a manifest endpoint key {source}:{line_number}"
                        )
                    index = by_key[key]
                if index < 0 or index >= len(records):
                    raise EndpointSelectionError(
                        f"source-records endpoint index is outside manifest {source}:{line_number}"
                    )
                record = records[index]
                if row.get("sequence_id") is not None and row["sequence_id"] != record.sequence_id:
                    raise EndpointSelectionError(
                        f"source-records sequence_id mismatch {source}:{line_number}"
                    )
                if row.get("frame_id") is not None and row["frame_id"] != record.frame_id:
                    raise EndpointSelectionError(
                        f"source-records frame_id mismatch {source}:{line_number}"
                    )
                current.add(index)
        if not current:
            raise EndpointSelectionError(f"source-records file is empty: {source}")
        selected = current if selected is None else selected.intersection(current)
    return set() if selected is None else selected


def _read_derived_indices(root: Path) -> set[int]:
    """Read target manifest indices from a complete derived cache manifest."""

    path = root.expanduser().resolve() / "cache_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise EndpointSelectionError(f"blank derived cache row: {path}:{line_number}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EndpointSelectionError(
                    f"invalid derived cache JSON: {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict) or "target_manifest_index" not in value:
                raise EndpointSelectionError(
                    f"derived cache row lacks target_manifest_index: {path}:{line_number}"
                )
            raw_index = value["target_manifest_index"]
            if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
                raise EndpointSelectionError(
                    f"derived target_manifest_index is malformed: {path}:{line_number}"
                )
            indices.add(raw_index)
    if not indices:
        raise EndpointSelectionError(f"derived cache manifest is empty: {path}")
    return indices


def run(args: argparse.Namespace) -> int:
    manifest = args.manifest.expanduser().resolve()
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if (
        isinstance(args.sequence_warmup, bool)
        or not isinstance(args.sequence_warmup, int)
        or args.sequence_warmup < 0
    ):
        raise ValueError("--sequence-warmup must be a non-negative integer")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = load_manifest(manifest)
    if not records:
        raise EndpointSelectionError("manifest is empty")
    source_sets: list[set[int]] = []
    source_identities: list[dict[str, object]] = []
    if args.source:
        source_sets.append(_read_source_indices(args.source, manifest))
        source_identities.extend(
            _source_identity(path.expanduser().resolve()) for path in args.source
        )
    if args.source_records:
        source_sets.append(_read_source_record_indices(args.source_records, manifest))
        source_identities.extend(
            _source_identity(path.expanduser().resolve())
            for path in args.source_records
        )
    if source_sets:
        indices_set = source_sets[0]
        for source_set in source_sets[1:]:
            indices_set = indices_set.intersection(source_set)
        indices = sorted(indices_set)
    else:
        indices = [
            window.endpoint_index
            for window in build_causal_windows(
                records, student_sequence_length=3, vggt_context_pairs=5
            )
        ]
    unfiltered_count = len(indices)
    derived_identity: dict[str, object] | None = None
    if args.derived_cache_root is not None:
        derived_root = args.derived_cache_root.expanduser().resolve()
        derived_manifest = derived_root / "cache_manifest.jsonl"
        derived_indices = _read_derived_indices(args.derived_cache_root)
        derived_identity = {
            "root": str(derived_root),
            "cache_manifest": _source_identity(derived_manifest),
        }
        by_index = {index for index in derived_indices if index < len(records)}
        indices = [
            window.endpoint_index
            for window in build_causal_windows(
                records, student_sequence_length=3, vggt_context_pairs=5
            )
            if all(index in by_index for index in window.student_indices)
        ]
        if source_sets:
            source_indices = set.intersection(*source_sets)
            indices = [index for index in indices if index in source_indices]
    indices = _sequence_warmup_indices(records, indices, args.sequence_warmup)
    warmup_filtered_count = len(indices)
    indices = indices[args.start :]
    if args.limit is not None:
        indices = indices[: args.limit]
    if not indices:
        raise EndpointSelectionError("endpoint selection is empty after filtering")
    selection = write_endpoint_index(
        args.output,
        manifest_path=manifest,
        manifest_indices=indices,
    )
    output_path = args.output.expanduser().resolve()
    receipt_path = (
        output_path.with_suffix(".receipt.json")
        if args.receipt is None
        else args.receipt.expanduser().resolve()
    )
    positions: dict[str, int] = {}
    sequence_counts: dict[str, int] = {}
    sequence_first_last: dict[str, dict[str, int]] = {}
    for record in records:
        position = positions.get(record.sequence_id, 0)
        positions[record.sequence_id] = position + 1
    for index in indices:
        record = records[index]
        sequence_counts[record.sequence_id] = sequence_counts.get(record.sequence_id, 0) + 1
        bounds = sequence_first_last.setdefault(
            record.sequence_id,
            {"first_manifest_index": index, "first_frame_id": int(record.frame_id)},
        )
        bounds["last_manifest_index"] = index
        bounds["last_frame_id"] = int(record.frame_id)
    receipt_payload: dict[str, object] = {
        "schema_version": 1,
        "component": "spring-common-endpoint-index-builder",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "protocol": {
            "kind": "spring_common_endpoint_index",
            "endpoint_id_hash_algorithm": ENDPOINT_ID_HASH_ALGORITHM,
            "student_sequence_length": 3,
            "vggt_context_pairs": 5,
            "sequence_warmup": int(args.sequence_warmup),
            "warmup_policy": "per_sequence_manifest_position",
            "timestamp_contract": "bound_to_source_manifest",
        },
        "input": {
            "manifest": _source_identity(manifest),
            "record_count": len(records),
            "source_indices": source_identities,
            "derived_cache": derived_identity,
        },
        "selection": {
            "causal_candidates_before_filters": unfiltered_count,
            "after_warmup": warmup_filtered_count,
            "start": int(args.start),
            "limit": args.limit,
            "selected": selection.count,
            "sequence_counts": sequence_counts,
            "sequence_bounds": sequence_first_last,
        },
        "output": {
            "path": str(output_path),
            "file_sha256": selection.file_sha256,
            "manifest_sha256": selection.manifest_sha256,
            "endpoint_count": selection.count,
            "endpoint_id_sha256": selection.entries_sha256,
        },
    }
    _atomic_json(receipt_path, receipt_payload)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "path": str(selection.path),
                "manifest_sha256": selection.manifest_sha256,
                "endpoint_id_sha256": selection.entries_sha256,
                "endpoint_id_hash_algorithm": ENDPOINT_ID_HASH_ALGORITHM,
                "endpoint_count": selection.count,
                "file_sha256": selection.file_sha256,
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
