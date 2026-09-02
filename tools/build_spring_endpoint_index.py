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
import json
import sys
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
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
    records = load_manifest(manifest)
    source_sets: list[set[int]] = []
    if args.source:
        source_sets.append(_read_source_indices(args.source, manifest))
    if args.source_records:
        source_sets.append(_read_source_record_indices(args.source_records, manifest))
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
    if args.derived_cache_root is not None:
        derived_indices = _read_derived_indices(args.derived_cache_root)
        by_index = {index: records[index] for index in derived_indices if index < len(records)}
        indices = [
            window.endpoint_index
            for window in build_causal_windows(
                records, student_sequence_length=3, vggt_context_pairs=5
            )
            if all(index in by_index for index in window.student_indices)
        ]
        if args.source:
            source_indices = set(_read_source_indices(args.source, manifest))
            indices = [index for index in indices if index in source_indices]
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
