"""Manifest-bound endpoint selection for comparable Spring evaluations.

An endpoint list is deliberately separate from the source JSONL manifest.  It
contains only endpoint identities (the original manifest index plus sequence,
frame and timestamp) and a digest of those identities.  Evaluators construct
the *complete* spatial/temporal dataset first, then use the list to select
dataset positions.  Consequently a selected T=3 sample still retains all of
its causal history and VGGT context.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache_dataset import sha256_file
from .manifest import ManifestRecord, load_manifest


ENDPOINT_SELECTION_SCHEMA_VERSION = 1
ENDPOINT_SELECTION_KIND = "spring_common_endpoint_index"
ENDPOINT_ID_HASH_ALGORITHM = (
    "sha256(canonical_json([manifest_index,sequence_id,frame_id,timestamp]))"
)


class EndpointSelectionError(ValueError):
    """Raised when a selection list cannot be bound to an evaluation corpus."""


def _canonical_entries(entries: Sequence["EndpointIdentity"]) -> bytes:
    payload = [entry.to_dict() for entry in entries]
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _entries_sha256(entries: Sequence["EndpointIdentity"]) -> str:
    return hashlib.sha256(_canonical_entries(entries)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EndpointSelectionError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise EndpointSelectionError(f"{name} is not hexadecimal SHA-256") from exc
    return value.lower()


@dataclass(frozen=True, slots=True)
class EndpointIdentity:
    """One endpoint identity in original manifest coordinates."""

    manifest_index: int
    sequence_id: str
    frame_id: int
    timestamp: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.manifest_index, bool)
            or not isinstance(self.manifest_index, int)
            or self.manifest_index < 0
        ):
            raise EndpointSelectionError("manifest_index must be a non-negative integer")
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise EndpointSelectionError("sequence_id must be a non-empty string")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise EndpointSelectionError("frame_id must be a non-negative integer")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise EndpointSelectionError("timestamp must be numeric")
        if not math.isfinite(float(self.timestamp)):
            raise EndpointSelectionError("timestamp must be finite")
        object.__setattr__(self, "timestamp", float(self.timestamp))

    @classmethod
    def from_record(cls, manifest_index: int, record: ManifestRecord) -> "EndpointIdentity":
        if not isinstance(record, ManifestRecord):
            raise TypeError("record must be a ManifestRecord")
        return cls(
            manifest_index=int(manifest_index),
            sequence_id=record.sequence_id,
            frame_id=record.frame_id,
            timestamp=record.timestamp,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointIdentity":
        if not isinstance(value, Mapping):
            raise EndpointSelectionError("endpoint entry must be a JSON object")
        # ``endpoint_index`` is accepted as a readable alias for hand-authored
        # lists, but serialized output always uses the unambiguous manifest name.
        raw_index = value.get("manifest_index", value.get("endpoint_index"))
        missing = [
            name
            for name, raw in (
                ("manifest_index", raw_index),
                ("sequence_id", value.get("sequence_id")),
                ("frame_id", value.get("frame_id")),
                ("timestamp", value.get("timestamp")),
            )
            if raw is None
        ]
        if missing:
            raise EndpointSelectionError(
                f"endpoint entry is missing required fields: {missing}"
            )
        return cls(
            manifest_index=raw_index,
            sequence_id=value["sequence_id"],
            frame_id=value["frame_id"],
            timestamp=value["timestamp"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_index": self.manifest_index,
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class EndpointSelection:
    """Validated endpoint list and its provenance digests."""

    path: Path
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[EndpointIdentity, ...]
    entries_sha256: str
    file_sha256: str
    kind: str = ENDPOINT_SELECTION_KIND
    schema_version: int = ENDPOINT_SELECTION_SCHEMA_VERSION

    @property
    def manifest_indices(self) -> tuple[int, ...]:
        return tuple(entry.manifest_index for entry in self.entries)

    @property
    def count(self) -> int:
        return len(self.entries)

    def to_report(
        self,
        *,
        available_endpoint_count: int | None = None,
        evaluated_manifest_indices: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Return compact report metadata; the full list remains an artifact."""

        report: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "endpoint_id_sha256": self.entries_sha256,
            "entries_sha256": self.entries_sha256,
            "endpoint_id_hash_algorithm": ENDPOINT_ID_HASH_ALGORITHM,
            "endpoint_count": self.count,
            "first_manifest_index": self.entries[0].manifest_index,
            "last_manifest_index": self.entries[-1].manifest_index,
        }
        if available_endpoint_count is not None:
            report["available_endpoint_count"] = int(available_endpoint_count)
        if evaluated_manifest_indices is not None:
            evaluated = tuple(int(index) for index in evaluated_manifest_indices)
            if not evaluated:
                raise EndpointSelectionError(
                    "evaluated endpoint selection cannot be empty"
                )
            selected_entries = {
                entry.manifest_index: entry for entry in self.entries
            }
            try:
                evaluated_entries = tuple(selected_entries[index] for index in evaluated)
            except KeyError as exc:
                raise EndpointSelectionError(
                    "evaluated endpoint is outside the declared endpoint list"
                ) from exc
            report["evaluated_endpoint_count"] = len(evaluated_entries)
            report["evaluated_endpoint_id_sha256"] = _entries_sha256(
                evaluated_entries
            )
        return report


def _validate_entries_order(entries: Sequence[EndpointIdentity]) -> None:
    if not entries:
        raise EndpointSelectionError("endpoint selection is empty")
    previous = -1
    for entry in entries:
        if entry.manifest_index <= previous:
            raise EndpointSelectionError(
                "endpoint manifest_index values must be strictly increasing"
            )
        previous = entry.manifest_index


def _selection_payload(selection: EndpointSelection) -> dict[str, Any]:
    return {
        "schema_version": ENDPOINT_SELECTION_SCHEMA_VERSION,
        "kind": ENDPOINT_SELECTION_KIND,
        "manifest_path": str(selection.manifest_path),
        "manifest_sha256": selection.manifest_sha256,
        "endpoint_count": selection.count,
        "endpoint_id_sha256": selection.entries_sha256,
        "endpoint_id_hash_algorithm": ENDPOINT_ID_HASH_ALGORITHM,
        "entries": [entry.to_dict() for entry in selection.entries],
    }


def write_endpoint_index(
    path: str | Path,
    *,
    manifest_path: str | Path,
    manifest_indices: Iterable[int],
) -> EndpointSelection:
    """Write a manifest-bound endpoint JSON list and return its metadata."""

    output_path = Path(path).expanduser().resolve()
    source_manifest = Path(manifest_path).expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if output_path == source_manifest:
        raise EndpointSelectionError(
            "endpoint output path must differ from the source manifest"
        )
    records = load_manifest(source_manifest)
    raw_indices = list(manifest_indices)
    if not raw_indices:
        raise EndpointSelectionError("endpoint selection is empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_indices
    ):
        raise EndpointSelectionError("manifest indices must be integers")
    indices = list(raw_indices)
    if indices != sorted(set(indices)):
        raise EndpointSelectionError(
            "manifest indices must be unique and strictly increasing"
        )
    if any(index < 0 or index >= len(records) for index in indices):
        raise EndpointSelectionError(
            f"endpoint manifest index is outside manifest range [0,{len(records)})"
        )
    entries = tuple(
        EndpointIdentity.from_record(index, records[index]) for index in indices
    )
    entry_digest = _entries_sha256(entries)
    provisional = EndpointSelection(
        path=output_path,
        manifest_path=source_manifest,
        manifest_sha256=sha256_file(source_manifest),
        entries=entries,
        entries_sha256=entry_digest,
        file_sha256="",
    )
    payload = _selection_payload(provisional)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return load_endpoint_index(output_path, manifest_path=source_manifest)


def load_endpoint_index(
    path: str | Path,
    *,
    manifest_path: str | Path,
) -> EndpointSelection:
    """Read and strictly bind an endpoint list to ``manifest_path``."""

    selection_path = Path(path).expanduser().resolve()
    source_manifest = Path(manifest_path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointSelectionError(f"cannot read endpoint list: {selection_path}") from exc
    if not isinstance(payload, Mapping):
        raise EndpointSelectionError("endpoint list root must be a JSON object")
    if payload.get("schema_version") != ENDPOINT_SELECTION_SCHEMA_VERSION:
        raise EndpointSelectionError(
            f"unsupported endpoint list schema: {payload.get('schema_version')!r}"
        )
    if payload.get("kind") != ENDPOINT_SELECTION_KIND:
        raise EndpointSelectionError(
            f"endpoint list kind must be {ENDPOINT_SELECTION_KIND!r}"
        )
    declared_algorithm = payload.get("endpoint_id_hash_algorithm")
    if declared_algorithm is not None and declared_algorithm != ENDPOINT_ID_HASH_ALGORITHM:
        raise EndpointSelectionError(
            "endpoint list endpoint_id_hash_algorithm is unsupported"
        )
    actual_manifest_sha = sha256_file(source_manifest)
    declared_manifest_sha = _require_sha256(
        payload.get("manifest_sha256"), "manifest_sha256"
    )
    if declared_manifest_sha != actual_manifest_sha:
        raise EndpointSelectionError(
            "endpoint list is bound to a different manifest SHA-256: "
            f"expected {actual_manifest_sha}, got {declared_manifest_sha}"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise EndpointSelectionError("endpoint list entries must be a JSON array")
    records = load_manifest(source_manifest)
    try:
        entries = tuple(
            _identity_from_payload(item, records) for item in raw_entries
        )
    except (TypeError, EndpointSelectionError) as exc:
        if isinstance(exc, EndpointSelectionError):
            raise
        raise EndpointSelectionError("malformed endpoint list entry") from exc
    _validate_entries_order(entries)
    declared_count = payload.get("endpoint_count", payload.get("count"))
    if declared_count is not None and declared_count != len(entries):
        raise EndpointSelectionError(
            f"endpoint_count disagrees with entries: {declared_count!r} vs {len(entries)}"
        )
    actual_entries_sha = _entries_sha256(entries)
    declared_entries_sha = _require_sha256(
        payload.get("endpoint_id_sha256", payload.get("entries_sha256")),
        "endpoint_id_sha256",
    )
    if declared_entries_sha != actual_entries_sha:
        raise EndpointSelectionError(
            "endpoint list endpoint_id_sha256 does not match its entries"
        )
    validate_endpoint_selection(entries, records)
    return EndpointSelection(
        path=selection_path,
        manifest_path=source_manifest,
        manifest_sha256=actual_manifest_sha,
        entries=entries,
        entries_sha256=actual_entries_sha,
        file_sha256=sha256_file(selection_path),
        kind=str(payload["kind"]),
        schema_version=int(payload["schema_version"]),
    )


def _identity_from_payload(
    value: Mapping[str, Any], records: Sequence[ManifestRecord]
) -> EndpointIdentity:
    """Resolve an entry supplied as manifest index *or* sequence/frame key.

    Index-only entries and sequence/frame-only entries are accepted for
    interoperability with existing reports.  Missing identity fields are
    filled from the bound manifest, while supplied fields are checked exactly.
    """

    if not isinstance(value, Mapping):
        raise EndpointSelectionError("endpoint entry must be a JSON object")
    raw_index = value.get("manifest_index", value.get("endpoint_index"))
    raw_sequence = value.get("sequence_id")
    raw_frame = value.get("frame_id")
    raw_timestamp = value.get("timestamp")
    if raw_index is None:
        if not isinstance(raw_sequence, str) or not raw_sequence.strip():
            raise EndpointSelectionError(
                "endpoint entry needs manifest_index or sequence_id/frame_id"
            )
        if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
            raise EndpointSelectionError(
                "sequence/frame endpoint entry requires integer frame_id"
            )
        matches = [
            index
            for index, record in enumerate(records)
            if record.sequence_id == raw_sequence and record.frame_id == raw_frame
        ]
        if len(matches) != 1:
            raise EndpointSelectionError(
                "sequence/frame endpoint key does not identify exactly one "
                f"manifest record: {raw_sequence!r}/{raw_frame}"
            )
        manifest_index = matches[0]
    else:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise EndpointSelectionError("manifest_index must be an integer")
        manifest_index = raw_index
    if manifest_index < 0 or manifest_index >= len(records):
        raise EndpointSelectionError(
            f"endpoint manifest index is outside manifest range [0,{len(records)})"
        )
    record = records[manifest_index]
    if raw_sequence is not None and raw_sequence != record.sequence_id:
        raise EndpointSelectionError(
            f"endpoint sequence_id disagrees at manifest index {manifest_index}"
        )
    if raw_frame is not None and (
        isinstance(raw_frame, bool)
        or not isinstance(raw_frame, int)
        or raw_frame != record.frame_id
    ):
        raise EndpointSelectionError(
            f"endpoint frame_id disagrees at manifest index {manifest_index}"
        )
    if raw_timestamp is not None:
        if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, (int, float)):
            raise EndpointSelectionError("endpoint timestamp must be numeric")
        if not math.isfinite(float(raw_timestamp)) or float(raw_timestamp) != float(record.timestamp):
            raise EndpointSelectionError(
                f"endpoint timestamp disagrees at manifest index {manifest_index}"
            )
    return EndpointIdentity.from_record(manifest_index, record)


def validate_endpoint_selection(
    entries: Sequence[EndpointIdentity], records: Sequence[ManifestRecord]
) -> None:
    """Ensure every selected identity exactly matches the source manifest."""

    _validate_entries_order(entries)
    for entry in entries:
        if entry.manifest_index >= len(records):
            raise EndpointSelectionError(
                "endpoint manifest index is missing from evaluation manifest: "
                f"{entry.manifest_index}"
            )
        record = records[entry.manifest_index]
        if (
            record.sequence_id != entry.sequence_id
            or record.frame_id != entry.frame_id
            or float(record.timestamp) != float(entry.timestamp)
        ):
            raise EndpointSelectionError(
                "endpoint identity disagrees with evaluation manifest at index "
                f"{entry.manifest_index}"
            )


def resolve_endpoint_dataset_indices(
    selection: EndpointSelection,
    endpoint_manifest_indices: Iterable[int],
) -> tuple[int, ...]:
    """Map manifest endpoint IDs to positions in a fully-built dataset.

    Missing IDs are fatal.  This is intentionally fail-closed: silently
    dropping an endpoint would make two arms report different populations.
    """

    positions: dict[int, int] = {}
    for position, raw_index in enumerate(endpoint_manifest_indices):
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise EndpointSelectionError("dataset endpoint indices must be integers")
        if raw_index in positions:
            raise EndpointSelectionError(
                f"dataset exposes duplicate endpoint manifest index {raw_index}"
            )
        positions[raw_index] = position
    missing = [index for index in selection.manifest_indices if index not in positions]
    if missing:
        preview = ", ".join(str(value) for value in missing[:16])
        suffix = "..." if len(missing) > 16 else ""
        raise EndpointSelectionError(
            "evaluation arm is missing required endpoint manifest indices "
            f"({len(missing)}): {preview}{suffix}"
        )
    return tuple(positions[index] for index in selection.manifest_indices)


__all__ = [
    "ENDPOINT_SELECTION_KIND",
    "ENDPOINT_SELECTION_SCHEMA_VERSION",
    "ENDPOINT_ID_HASH_ALGORITHM",
    "EndpointIdentity",
    "EndpointSelection",
    "EndpointSelectionError",
    "load_endpoint_index",
    "resolve_endpoint_dataset_indices",
    "validate_endpoint_selection",
    "write_endpoint_index",
]
