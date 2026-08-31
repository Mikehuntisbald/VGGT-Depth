"""Validated JSONL manifest interface for rectified stereo sequences."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geometry.camera import PinholeIntrinsics, validate_intrinsics


class ManifestValidationError(ValueError):
    """Raised when a manifest record violates the stereo data contract."""


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ManifestValidationError(f"missing required manifest field {key!r}")
    return payload[key]


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{name} must be a non-empty string")
    return value


def _parse_intrinsics(value: Any) -> tuple[tuple[float, float, float], ...]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
        validate_intrinsics(matrix)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"invalid K: {exc}") from exc
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One calibrated, rectified stereo frame in a causal sequence.

    ``K`` is the real calibration in original/HR pixel coordinates and
    ``baseline_m`` is the physical stereo baseline in metres. Paths are stored
    exactly as written and may be absolute or relative to the manifest file.
    """

    sequence_id: str
    frame_id: int
    timestamp: float
    left_path: str
    right_path: str
    K: tuple[tuple[float, float, float], ...]
    baseline_m: float
    gt_disparity_path: str | None = None
    rectified: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _nonempty_string(self.sequence_id, "sequence_id")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise ManifestValidationError("frame_id must be an integer")
        if self.frame_id < 0:
            raise ManifestValidationError("frame_id must be non-negative")
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise ManifestValidationError("timestamp must be numeric")
        if not math.isfinite(float(self.timestamp)):
            raise ManifestValidationError("timestamp must be finite")
        _nonempty_string(self.left_path, "left_path")
        _nonempty_string(self.right_path, "right_path")
        if self.left_path == self.right_path:
            raise ManifestValidationError("left_path and right_path must differ")
        _parse_intrinsics(self.K)
        if isinstance(self.baseline_m, bool) or not isinstance(
            self.baseline_m, (int, float)
        ):
            raise ManifestValidationError("baseline_m must be numeric")
        if not math.isfinite(float(self.baseline_m)) or self.baseline_m <= 0.0:
            raise ManifestValidationError("baseline_m must be finite and > 0")
        if self.gt_disparity_path is not None:
            _nonempty_string(self.gt_disparity_path, "gt_disparity_path")
        if self.rectified is not True:
            raise ManifestValidationError(
                "only rectified stereo pairs are supported; rectified must be true"
            )
        if not isinstance(self.extras, Mapping):
            raise ManifestValidationError("extras must be a mapping")
        reserved = set(self.required_field_names()) | {"rectified"}
        overlap = reserved.intersection(self.extras)
        if overlap:
            raise ManifestValidationError(
                f"extras cannot replace reserved fields: {sorted(overlap)}"
            )
        try:
            json.dumps(dict(self.extras), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"extras must be finite JSON-compatible values: {exc}"
            ) from exc
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "baseline_m", float(self.baseline_m))
        object.__setattr__(self, "K", _parse_intrinsics(self.K))
        # Keep a private copy so caller-owned mappings cannot mutate this
        # record. A plain dict remains pickleable for DataLoader workers.
        object.__setattr__(self, "extras", dict(self.extras))

    @staticmethod
    def required_field_names() -> tuple[str, ...]:
        """Return the stable required JSON field names."""

        return (
            "sequence_id",
            "frame_id",
            "timestamp",
            "left_path",
            "right_path",
            "K",
            "baseline_m",
            "gt_disparity_path",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestRecord":
        """Validate and construct a record from one decoded JSON object."""

        if not isinstance(payload, Mapping):
            raise ManifestValidationError("manifest record must be a JSON object")
        known = set(cls.required_field_names()) | {"rectified"}
        extras = {key: value for key, value in payload.items() if key not in known}
        return cls(
            sequence_id=_required(payload, "sequence_id"),
            frame_id=_required(payload, "frame_id"),
            timestamp=_required(payload, "timestamp"),
            left_path=_required(payload, "left_path"),
            right_path=_required(payload, "right_path"),
            K=_parse_intrinsics(_required(payload, "K")),
            baseline_m=_required(payload, "baseline_m"),
            gt_disparity_path=_required(payload, "gt_disparity_path"),
            # The base schema predates this explicit flag. Missing means the
            # producer asserts the required rectified-input contract.
            rectified=payload.get("rectified", True),
            extras=extras,
        )

    @property
    def intrinsics(self) -> PinholeIntrinsics:
        """Return calibrated scalar intrinsics in original/HR pixel units."""

        return PinholeIntrinsics.from_matrix(self.K)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with rectification explicit."""

        payload: dict[str, Any] = {
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "left_path": self.left_path,
            "right_path": self.right_path,
            "K": [list(row) for row in self.K],
            "baseline_m": self.baseline_m,
            "gt_disparity_path": self.gt_disparity_path,
            "rectified": self.rectified,
        }
        payload.update(self.extras)
        return payload


def iter_manifest(manifest_path: str | os.PathLike[str]) -> Iterator[ManifestRecord]:
    """Yield validated records from a non-empty UTF-8 JSONL manifest.

    Blank lines are rejected rather than silently changing frame indexing. Any
    error includes the source path and one-based line number.
    """

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist or is not a file: {path}")
    found_record = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ManifestValidationError(
                    f"{path}:{line_number}: blank JSONL records are not allowed"
                )
            try:
                payload = json.loads(raw_line)
                record = ManifestRecord.from_dict(payload)
            except (json.JSONDecodeError, ManifestValidationError) as exc:
                raise ManifestValidationError(
                    f"{path}:{line_number}: {exc}"
                ) from exc
            found_record = True
            yield record
    if not found_record:
        raise ManifestValidationError(f"manifest is empty: {path}")


def load_manifest(manifest_path: str | os.PathLike[str]) -> list[ManifestRecord]:
    """Load all validated records from a JSONL manifest."""

    return list(iter_manifest(manifest_path))


def write_manifest(
    manifest_path: str | os.PathLike[str],
    records: Iterable[ManifestRecord],
) -> None:
    """Atomically write one or more validated records as UTF-8 JSONL.

    Empty manifests are rejected because downstream causal-window construction
    cannot distinguish them from a failed dataset scan.
    """

    path = Path(manifest_path)
    materialized = list(records)
    if not materialized:
        raise ManifestValidationError("refusing to write an empty manifest")
    for index, record in enumerate(materialized):
        if not isinstance(record, ManifestRecord):
            raise TypeError(
                f"records[{index}] must be ManifestRecord, got {type(record).__name__}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for record in materialized:
                json.dump(record.to_dict(), handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
