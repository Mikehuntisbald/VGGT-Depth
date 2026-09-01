"""Immutable calibrated stereo sidecars for stored rectified image pairs.

The project manifests and raw FFS/VGGT caches predate an explicit stereo
extrinsic field.  Rewriting those manifests would invalidate their exact
source hashes, so calibrated geometry is carried in a separately versioned
JSONL sidecar.  Every row is bound to one original manifest row and to the
pixel-level rectification audit which owns the stored-image coordinate frame.

All transforms in this module use column vectors.  In particular,
``T_right_rectified_from_left_rectified_m`` satisfies
``X_right = T_right_left @ X_left`` and is expressed in metres.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor

from .cache_dataset import CacheMismatchError, canonical_json_sha256, sha256_file
from .manifest import ManifestRecord, iter_manifest


RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION = 1
RECTIFIED_CALIBRATION_COMPONENT = "rectified-stereo-calibration"
RECTIFIED_CALIBRATION_CONTRACT = "stored_rectified_virtual_cameras_v1"
RECTIFIED_PIXEL_CONTRACT = "audited_same_row_rectified_pixels_v1"
RECTIFIED_EXTRINSICS_CONVENTION = (
    "right-camera-from-left-camera; X_right=T_right_left@X_left"
)

_ROW_FIELDS = {
    "schema_version",
    "component",
    "contract_version",
    "sequence_id",
    "frame_id",
    "timestamp",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_manifest_index",
    "source_record_sha256",
    "metadata_path",
    "metadata_sha256",
    "left_image_frame",
    "right_image_frame",
    "extrinsics_convention",
    "T_right_rectified_from_left_rectified_m",
    "derivation",
    "pixel_coordinate_contract",
    "pixel_audit_path",
    "pixel_audit_sha256",
    "diagnostic_T_right_raw_from_left_raw_m",
    "calibration_record_sha256",
}


def _plain_matrix(value: Any, *, name: str, rows: int, columns: int) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CacheMismatchError(f"{name} is not a numeric matrix") from exc
    if matrix.shape != (rows, columns):
        raise CacheMismatchError(
            f"{name} must have shape [{rows},{columns}], got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise CacheMismatchError(f"{name} contains NaN or infinity")
    return matrix


def _rotation_matrix(value: Any, *, name: str) -> np.ndarray:
    rotation = _plain_matrix(value, name=name, rows=3, columns=3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise CacheMismatchError(f"{name} is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise CacheMismatchError(f"{name} determinant is not +1")
    return rotation


def _flat_rotation(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CacheMismatchError(f"{name} is not numeric") from exc
    if array.size != 9:
        raise CacheMismatchError(f"{name} must contain exactly 9 values")
    return _rotation_matrix(array.reshape(3, 3), name=name)


def _flat_matrix(
    value: Any, *, name: str, rows: int, columns: int
) -> np.ndarray:
    """Parse a flattened or nested calibration matrix without shape ambiguity."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CacheMismatchError(f"{name} is not numeric") from exc
    if array.size != rows * columns:
        raise CacheMismatchError(
            f"{name} must contain exactly {rows * columns} values"
        )
    return _plain_matrix(
        array.reshape(rows, columns), name=name, rows=rows, columns=columns
    )


def _homogeneous_transform(value: Any, *, name: str) -> np.ndarray:
    transform = _plain_matrix(value, name=name, rows=4, columns=4)
    _rotation_matrix(transform[:3, :3], name=f"{name} rotation")
    if not np.allclose(
        transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-12, rtol=0.0
    ):
        raise CacheMismatchError(f"{name} has a malformed homogeneous row")
    return transform


def _record_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("calibration_record_sha256", None)
    return canonical_json_sha256(unsigned)


@dataclass(frozen=True, slots=True)
class RectifiedCalibrationRecord:
    """One manifest-bound rectified stereo calibration row."""

    sequence_id: str
    frame_id: int
    timestamp: float
    source_manifest_index: int
    source_record_sha256: str
    metadata_path: str
    metadata_sha256: str
    T_right_rectified_from_left_rectified_m: tuple[tuple[float, ...], ...]
    diagnostic_T_right_raw_from_left_raw_m: tuple[tuple[float, ...], ...]
    calibration_record_sha256: str

    def as_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return the runtime rectified ``right-from-left`` transform."""

        return torch.tensor(
            self.T_right_rectified_from_left_rectified_m,
            dtype=dtype,
            device=device,
        )


@dataclass(frozen=True, slots=True)
class RectifiedCalibrationIndex:
    """Validated immutable sidecar indexed by manifest row and frame identity."""

    sidecar_path: Path
    sidecar_sha256: str
    receipt_path: Path
    receipt_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    pixel_audit_path: Path
    pixel_audit_sha256: str
    records: tuple[RectifiedCalibrationRecord, ...]

    def record_for_manifest_index(self, index: int) -> RectifiedCalibrationRecord:
        """Return the calibration at an exact original manifest index."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("manifest index must be an integer")
        if index < 0 or index >= len(self.records):
            raise CacheMismatchError(f"calibration manifest index is out of range: {index}")
        record = self.records[index]
        if record.source_manifest_index != index:
            raise CacheMismatchError(
                f"calibration sidecar index mismatch: slot={index}, "
                f"record={record.source_manifest_index}"
            )
        return record

    def record_for_identity(
        self, sequence_id: str, frame_id: int, timestamp: float
    ) -> RectifiedCalibrationRecord:
        """Resolve a unique row and verify timestamp without fuzzy fallback."""

        matches = [
            record
            for record in self.records
            if record.sequence_id == sequence_id and record.frame_id == frame_id
        ]
        if len(matches) != 1:
            raise CacheMismatchError(
                "calibration identity must resolve exactly once: "
                f"{sequence_id}/{frame_id}, matches={len(matches)}"
            )
        record = matches[0]
        if record.timestamp != float(timestamp):
            raise CacheMismatchError(
                f"calibration timestamp mismatch for {sequence_id}/{frame_id}: "
                f"expected {timestamp}, got {record.timestamp}"
            )
        return record

    def records_for_vggt_source(
        self, source: Mapping[str, Any]
    ) -> tuple[RectifiedCalibrationRecord, ...]:
        """Join the exact five manifest rows embedded in a raw VGGT cache."""

        indices = source.get("manifest_indices")
        manifest_records = source.get("manifest_records")
        if not isinstance(indices, list) or len(indices) != 5:
            raise CacheMismatchError(
                "calibrated derivation requires five VGGT source manifest_indices"
            )
        if not isinstance(manifest_records, list) or len(manifest_records) != 5:
            raise CacheMismatchError(
                "calibrated derivation requires five VGGT source manifest_records"
            )
        result: list[RectifiedCalibrationRecord] = []
        for index_value, manifest_record in zip(indices, manifest_records, strict=True):
            if isinstance(index_value, bool) or not isinstance(index_value, int):
                raise CacheMismatchError("VGGT source manifest index is not an integer")
            if not isinstance(manifest_record, Mapping):
                raise CacheMismatchError("VGGT source manifest record is malformed")
            calibration = self.record_for_manifest_index(index_value)
            if calibration.source_record_sha256 != canonical_json_sha256(
                dict(manifest_record)
            ):
                raise CacheMismatchError(
                    "calibration/source manifest record hash mismatch at index "
                    f"{index_value}"
                )
            if (
                calibration.sequence_id != manifest_record.get("sequence_id")
                or calibration.frame_id != manifest_record.get("frame_id")
                or calibration.timestamp != manifest_record.get("timestamp")
            ):
                raise CacheMismatchError(
                    f"calibration/source identity mismatch at index {index_value}"
                )
            result.append(calibration)
        return tuple(result)


def _audit_manifest_entry(
    audit: Mapping[str, Any], *, manifest_path: Path, manifest_sha256: str
) -> Mapping[str, Any]:
    if audit.get("schema_version") != 1:
        raise CacheMismatchError("unsupported pixel audit schema")
    if audit.get("component") != "pixel-level-epipolar-rectification-audit":
        raise CacheMismatchError("pixel audit component mismatch")
    if audit.get("status") != "PASS":
        raise CacheMismatchError("pixel audit did not pass")
    if audit.get("published_contract") != RECTIFIED_PIXEL_CONTRACT:
        raise CacheMismatchError("pixel audit rectification contract mismatch")
    checks = audit.get("threshold_checks")
    if not isinstance(checks, list) or not checks or any(
        not isinstance(item, Mapping) or item.get("passed") is not True
        for item in checks
    ):
        raise CacheMismatchError("pixel audit threshold checks are incomplete")
    manifests = audit.get("manifests")
    if not isinstance(manifests, Mapping):
        raise CacheMismatchError("pixel audit manifest bindings are missing")
    resolved = manifest_path.resolve()
    matches = [
        value
        for value in manifests.values()
        if isinstance(value, Mapping)
        and Path(str(value.get("path", ""))).expanduser().resolve() == resolved
        and value.get("sha256") == manifest_sha256
    ]
    if len(matches) != 1:
        raise CacheMismatchError(
            "pixel audit must bind the exact source manifest path and SHA-256"
        )
    return matches[0]


def _rectified_calibration_payload(
    record: ManifestRecord,
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_manifest_index: int,
    pixel_audit_path: Path,
    pixel_audit_sha256: str,
) -> dict[str, Any]:
    if record.rectified is not True:
        raise CacheMismatchError("rectified stereo calibration requires rectified=true")
    extras = record.extras
    for name in ("K_right", "P_left", "P_right", "metadata_path", "metadata_sha256"):
        if name not in extras:
            raise CacheMismatchError(f"manifest calibration field {name!r} is missing")
    k_left = _plain_matrix(record.K, name="K_left", rows=3, columns=3)
    k_right = _plain_matrix(extras["K_right"], name="K_right", rows=3, columns=3)
    p_left = _plain_matrix(extras["P_left"], name="P_left", rows=3, columns=4)
    p_right = _plain_matrix(extras["P_right"], name="P_right", rows=3, columns=4)
    if not np.allclose(p_left[:, :3], k_left, atol=1e-9, rtol=0.0):
        raise CacheMismatchError("P_left[:,:3] does not equal K_left")
    if not np.allclose(p_right[:, :3], k_right, atol=1e-9, rtol=0.0):
        raise CacheMismatchError("P_right[:,:3] does not equal K_right")
    if not np.allclose(p_left[:, 3], 0.0, atol=1e-12, rtol=0.0):
        raise CacheMismatchError("rectified P_left translation is not zero")
    try:
        t_rectified = np.linalg.solve(p_right[:, :3], p_right[:, 3])
    except np.linalg.LinAlgError as exc:
        raise CacheMismatchError("P_right calibration matrix is singular") from exc
    baseline = float(record.baseline_m)
    expected_translation = np.asarray([-baseline, 0.0, 0.0])
    if not np.allclose(t_rectified, expected_translation, atol=1e-9, rtol=0.0):
        raise CacheMismatchError(
            "P_right translation is not the required rectified [-baseline,0,0]"
        )

    metadata_path = Path(str(extras["metadata_path"])).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"stereo metadata is missing: {metadata_path}")
    metadata_sha256 = sha256_file(metadata_path)
    if metadata_sha256 != extras["metadata_sha256"]:
        raise CacheMismatchError(
            f"stereo metadata SHA-256 mismatch for {metadata_path}"
        )
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CacheMismatchError(f"cannot parse stereo metadata {metadata_path}") from exc
    if not isinstance(metadata, Mapping) or metadata.get("rectified") is not True:
        raise CacheMismatchError("stereo metadata does not assert rectified=true")
    try:
        left_info = metadata["left_rect_camera_info"]
        right_info = metadata["right_rect_camera_info"]
        left_frame = str(metadata["left_frame_id"])
        right_frame = str(metadata["right_frame_id"])
    except (KeyError, TypeError) as exc:
        raise CacheMismatchError("rectified metadata camera fields are missing") from exc
    if not isinstance(left_info, Mapping) or not isinstance(right_info, Mapping):
        raise CacheMismatchError("rectified camera info is malformed")
    rotation_left = _flat_rotation(
        left_info.get("r"), name="left rectification rotation"
    )
    rotation_right = _flat_rotation(
        right_info.get("r"), name="right rectification rotation"
    )
    metadata_k_left = _flat_matrix(
        left_info.get("k"), name="metadata left rectified K", rows=3, columns=3
    )
    metadata_k_right = _flat_matrix(
        right_info.get("k"), name="metadata right rectified K", rows=3, columns=3
    )
    metadata_p_left = _flat_matrix(
        left_info.get("p"), name="metadata left rectified P", rows=3, columns=4
    )
    metadata_p_right = _flat_matrix(
        right_info.get("p"), name="metadata right rectified P", rows=3, columns=4
    )
    for name, metadata_value, manifest_value in (
        ("left rectified K", metadata_k_left, k_left),
        ("right rectified K", metadata_k_right, k_right),
        ("left rectified P", metadata_p_left, p_left),
        ("right rectified P", metadata_p_right, p_right),
    ):
        if not np.allclose(metadata_value, manifest_value, atol=1e-9, rtol=0.0):
            raise CacheMismatchError(f"metadata/manifest {name} mismatch")
    metadata_baseline = metadata.get("stereo_baseline_m")
    if (
        isinstance(metadata_baseline, bool)
        or not isinstance(metadata_baseline, (int, float))
        or not math.isfinite(float(metadata_baseline))
        or not math.isclose(
            float(metadata_baseline), baseline, abs_tol=1e-9, rel_tol=0.0
        )
    ):
        raise CacheMismatchError("metadata/manifest stereo baseline mismatch")

    transform_rectified = np.eye(4, dtype=np.float64)
    # Use the calibrated scalar after proving that projection factorisation is
    # identical.  This keeps the serialized physical owner explicit.
    transform_rectified[0, 3] = -baseline
    transform_raw = np.eye(4, dtype=np.float64)
    transform_raw[:3, :3] = rotation_right.T @ rotation_left
    transform_raw[:3, 3] = rotation_right.T @ expected_translation
    _homogeneous_transform(transform_raw, name="diagnostic raw stereo transform")
    if not math.isclose(
        float(np.linalg.norm(transform_raw[:3, 3])), baseline, abs_tol=1e-9
    ):
        raise CacheMismatchError("diagnostic raw stereo baseline is inconsistent")

    source_record = record.to_dict()
    payload: dict[str, Any] = {
        "schema_version": RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION,
        "component": RECTIFIED_CALIBRATION_COMPONENT,
        "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
        "sequence_id": record.sequence_id,
        "frame_id": record.frame_id,
        "timestamp": record.timestamp,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_index": source_manifest_index,
        "source_record_sha256": canonical_json_sha256(source_record),
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "left_image_frame": f"{left_frame}:rectified_virtual",
        "right_image_frame": f"{right_frame}:rectified_virtual",
        "extrinsics_convention": RECTIFIED_EXTRINSICS_CONVENTION,
        "T_right_rectified_from_left_rectified_m": transform_rectified.tolist(),
        "derivation": {
            "method": "rectified_projection_factorization_v1",
            "P_left": p_left.tolist(),
            "P_right": p_right.tolist(),
            "baseline_m": baseline,
            "runtime_right_vertical_intrinsics_policy": (
                "pixel audit owns rows; diagnostic K_right cy is not applied"
            ),
        },
        "pixel_coordinate_contract": RECTIFIED_PIXEL_CONTRACT,
        "pixel_audit_path": str(pixel_audit_path.resolve()),
        "pixel_audit_sha256": pixel_audit_sha256,
        "diagnostic_T_right_raw_from_left_raw_m": transform_raw.tolist(),
    }
    payload["calibration_record_sha256"] = _record_sha256(payload)
    return payload


def _atomic_immutable_text(path: Path, text: str) -> str:
    """Create ``path`` atomically or verify an existing byte-identical file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise CacheMismatchError(f"immutable calibration artifact differs: {path}")
        return "reused_identical"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "written"


def build_rectified_calibration_sidecar(
    manifest_path: str | Path,
    pixel_audit_path: str | Path,
    output_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an immutable per-row calibration sidecar and receipt."""

    manifest = Path(manifest_path).expanduser().resolve()
    audit_path = Path(pixel_audit_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    receipt = (
        output.with_suffix(".receipt.json")
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if output == receipt:
        raise ValueError("sidecar and receipt paths must differ")
    if not manifest.is_file() or not audit_path.is_file():
        raise FileNotFoundError("manifest and pixel audit must exist")
    manifest_sha256 = sha256_file(manifest)
    audit_sha256 = sha256_file(audit_path)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheMismatchError(f"cannot parse pixel audit {audit_path}") from exc
    if not isinstance(audit, Mapping):
        raise CacheMismatchError("pixel audit root is not a mapping")
    audit_manifest = _audit_manifest_entry(
        audit, manifest_path=manifest, manifest_sha256=manifest_sha256
    )
    records = list(iter_manifest(manifest))
    if audit_manifest.get("record_count") != len(records):
        raise CacheMismatchError("pixel audit manifest record count mismatch")
    rows = [
        _rectified_calibration_payload(
            record,
            source_manifest_path=manifest,
            source_manifest_sha256=manifest_sha256,
            source_manifest_index=index,
            pixel_audit_path=audit_path,
            pixel_audit_sha256=audit_sha256,
        )
        for index, record in enumerate(records)
    ]
    sidecar_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    sidecar_status = _atomic_immutable_text(output, sidecar_text)
    sidecar_sha256 = sha256_file(output)
    unique_calibrations = {
        canonical_json_sha256(
            {
                "rectified": row["T_right_rectified_from_left_rectified_m"],
                "raw": row["diagnostic_T_right_raw_from_left_raw_m"],
            }
        )
        for row in rows
    }
    receipt_payload = {
        "schema_version": RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION,
        "component": f"{RECTIFIED_CALIBRATION_COMPONENT}-receipt",
        "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
        "status": "PASS",
        "source": {
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha256,
            "pixel_audit_path": str(audit_path),
            "pixel_audit_sha256": audit_sha256,
            "pixel_coordinate_contract": RECTIFIED_PIXEL_CONTRACT,
        },
        "output": {
            "sidecar_path": str(output),
            "sidecar_sha256": sidecar_sha256,
        },
        "counts": {
            "records": len(rows),
            "sequences": len({row["sequence_id"] for row in rows}),
            "unique_calibrations": len(unique_calibrations),
        },
        "extrinsics_convention": RECTIFIED_EXTRINSICS_CONVENTION,
    }
    receipt_text = json.dumps(
        receipt_payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    receipt_status = _atomic_immutable_text(receipt, receipt_text)
    result = dict(receipt_payload)
    result["receipt_path"] = str(receipt)
    result["receipt_sha256"] = sha256_file(receipt)
    result["immutable_write_status"] = sidecar_status
    result["immutable_receipt_status"] = receipt_status
    return result


def _parse_sidecar_record(payload: Mapping[str, Any]) -> RectifiedCalibrationRecord:
    if set(payload) != _ROW_FIELDS:
        raise CacheMismatchError(
            "calibration sidecar row fields mismatch: "
            f"expected={sorted(_ROW_FIELDS)}, got={sorted(payload)}"
        )
    if payload.get("schema_version") != RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION:
        raise CacheMismatchError("calibration sidecar row schema mismatch")
    if payload.get("component") != RECTIFIED_CALIBRATION_COMPONENT:
        raise CacheMismatchError("calibration sidecar component mismatch")
    if payload.get("contract_version") != RECTIFIED_CALIBRATION_CONTRACT:
        raise CacheMismatchError("calibration sidecar contract mismatch")
    if payload.get("extrinsics_convention") != RECTIFIED_EXTRINSICS_CONVENTION:
        raise CacheMismatchError("calibration sidecar extrinsics convention mismatch")
    if payload.get("pixel_coordinate_contract") != RECTIFIED_PIXEL_CONTRACT:
        raise CacheMismatchError("calibration pixel-coordinate contract mismatch")
    if payload.get("calibration_record_sha256") != _record_sha256(payload):
        raise CacheMismatchError("calibration sidecar row SHA-256 mismatch")
    for name in ("sequence_id", "metadata_path", "metadata_sha256"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise CacheMismatchError(f"calibration sidecar {name} is malformed")
    for name in ("frame_id", "source_manifest_index"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CacheMismatchError(f"calibration sidecar {name} is malformed")
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise CacheMismatchError("calibration sidecar timestamp is malformed")
    transform_rectified = _homogeneous_transform(
        payload["T_right_rectified_from_left_rectified_m"],
        name="T_right_rectified_from_left_rectified_m",
    )
    derivation = payload.get("derivation")
    if not isinstance(derivation, Mapping) or derivation.get("method") != (
        "rectified_projection_factorization_v1"
    ):
        raise CacheMismatchError("calibration sidecar derivation is malformed")
    baseline = derivation.get("baseline_m")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not math.isfinite(baseline) or baseline <= 0:
        raise CacheMismatchError("calibration sidecar baseline is malformed")
    expected = np.eye(4)
    expected[0, 3] = -float(baseline)
    if not np.allclose(transform_rectified, expected, atol=1e-9, rtol=0.0):
        raise CacheMismatchError("rectified calibration is not exactly [I|-baseline]")
    transform_raw = _homogeneous_transform(
        payload["diagnostic_T_right_raw_from_left_raw_m"],
        name="diagnostic_T_right_raw_from_left_raw_m",
    )
    if not math.isclose(
        float(np.linalg.norm(transform_raw[:3, 3])), float(baseline), abs_tol=1e-9
    ):
        raise CacheMismatchError("diagnostic raw calibration baseline mismatch")
    return RectifiedCalibrationRecord(
        sequence_id=str(payload["sequence_id"]),
        frame_id=int(payload["frame_id"]),
        timestamp=float(timestamp),
        source_manifest_index=int(payload["source_manifest_index"]),
        source_record_sha256=str(payload["source_record_sha256"]),
        metadata_path=str(payload["metadata_path"]),
        metadata_sha256=str(payload["metadata_sha256"]),
        T_right_rectified_from_left_rectified_m=tuple(
            tuple(float(item) for item in row) for row in transform_rectified
        ),
        diagnostic_T_right_raw_from_left_raw_m=tuple(
            tuple(float(item) for item in row) for row in transform_raw
        ),
        calibration_record_sha256=str(payload["calibration_record_sha256"]),
    )


def load_rectified_calibration_sidecar(
    sidecar_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    expected_manifest_path: str | Path | None = None,
) -> RectifiedCalibrationIndex:
    """Load and fully validate an immutable rectified-calibration sidecar."""

    sidecar = Path(sidecar_path).expanduser().resolve()
    receipt = (
        sidecar.with_suffix(".receipt.json")
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if not sidecar.is_file() or not receipt.is_file():
        raise FileNotFoundError("calibration sidecar and receipt must both exist")
    sidecar_sha256 = sha256_file(sidecar)
    receipt_sha256 = sha256_file(receipt)
    try:
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheMismatchError(f"cannot parse calibration receipt {receipt}") from exc
    if not isinstance(receipt_payload, Mapping):
        raise CacheMismatchError("calibration receipt is not a mapping")
    expected_receipt = {
        "schema_version": RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION,
        "component": f"{RECTIFIED_CALIBRATION_COMPONENT}-receipt",
        "contract_version": RECTIFIED_CALIBRATION_CONTRACT,
        "status": "PASS",
        "extrinsics_convention": RECTIFIED_EXTRINSICS_CONVENTION,
    }
    for name, expected in expected_receipt.items():
        if receipt_payload.get(name) != expected:
            raise CacheMismatchError(f"calibration receipt {name} mismatch")
    source = receipt_payload.get("source")
    output = receipt_payload.get("output")
    counts = receipt_payload.get("counts")
    if not all(isinstance(value, Mapping) for value in (source, output, counts)):
        raise CacheMismatchError("calibration receipt source/output/counts are malformed")
    assert isinstance(source, Mapping) and isinstance(output, Mapping) and isinstance(counts, Mapping)
    if output.get("sidecar_path") != str(sidecar) or output.get("sidecar_sha256") != sidecar_sha256:
        raise CacheMismatchError("calibration receipt does not bind the sidecar")
    manifest_path = Path(str(source.get("manifest_path", ""))).expanduser().resolve()
    manifest_sha256 = str(source.get("manifest_sha256", ""))
    audit_path = Path(str(source.get("pixel_audit_path", ""))).expanduser().resolve()
    audit_sha256 = str(source.get("pixel_audit_sha256", ""))
    if source.get("pixel_coordinate_contract") != RECTIFIED_PIXEL_CONTRACT:
        raise CacheMismatchError("calibration receipt pixel contract mismatch")
    if expected_manifest_path is not None and manifest_path != Path(
        expected_manifest_path
    ).expanduser().resolve():
        raise CacheMismatchError("calibration receipt source manifest path mismatch")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha256:
        raise CacheMismatchError("calibration source manifest is missing or changed")
    if not audit_path.is_file() or sha256_file(audit_path) != audit_sha256:
        raise CacheMismatchError("calibration pixel audit is missing or changed")

    manifest_records = list(iter_manifest(manifest_path))
    rows: list[Mapping[str, Any]] = []
    for line_number, text in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            raise CacheMismatchError(f"blank calibration sidecar row {line_number}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CacheMismatchError(
                f"invalid calibration JSON at line {line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise CacheMismatchError(f"calibration row {line_number} is not a mapping")
        rows.append(value)
    record_count = counts.get("records")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count != len(rows) or record_count != len(manifest_records):
        raise CacheMismatchError("calibration receipt/sidecar/manifest coverage mismatch")
    parsed: list[RectifiedCalibrationRecord] = []
    seen_identity: set[tuple[str, int]] = set()
    live_metadata_sha256: dict[Path, str] = {}
    for index, (payload, manifest_record) in enumerate(
        zip(rows, manifest_records, strict=True)
    ):
        if payload.get("source_manifest_path") != str(manifest_path) or payload.get(
            "source_manifest_sha256"
        ) != manifest_sha256:
            raise CacheMismatchError(f"calibration row {index} manifest binding mismatch")
        if payload.get("pixel_audit_path") != str(audit_path) or payload.get(
            "pixel_audit_sha256"
        ) != audit_sha256:
            raise CacheMismatchError(f"calibration row {index} audit binding mismatch")
        record = _parse_sidecar_record(payload)
        if record.source_manifest_index != index:
            raise CacheMismatchError(f"calibration row order mismatch at index {index}")
        if record.source_record_sha256 != canonical_json_sha256(manifest_record.to_dict()):
            raise CacheMismatchError(f"calibration row source hash mismatch at index {index}")
        metadata_path = Path(record.metadata_path).expanduser().resolve()
        manifest_metadata_path = Path(
            str(manifest_record.extras.get("metadata_path", ""))
        ).expanduser().resolve()
        manifest_metadata_sha256 = manifest_record.extras.get("metadata_sha256")
        if metadata_path != manifest_metadata_path or (
            record.metadata_sha256 != manifest_metadata_sha256
        ):
            raise CacheMismatchError(
                f"calibration row metadata binding mismatch at index {index}"
            )
        if not metadata_path.is_file():
            raise CacheMismatchError(
                f"calibration metadata is missing at index {index}: {metadata_path}"
            )
        actual_metadata_sha256 = live_metadata_sha256.get(metadata_path)
        if actual_metadata_sha256 is None:
            actual_metadata_sha256 = sha256_file(metadata_path)
            live_metadata_sha256[metadata_path] = actual_metadata_sha256
        if actual_metadata_sha256 != record.metadata_sha256:
            raise CacheMismatchError(
                f"calibration metadata SHA-256 mismatch at index {index}"
            )
        if (
            record.sequence_id != manifest_record.sequence_id
            or record.frame_id != manifest_record.frame_id
            or record.timestamp != manifest_record.timestamp
        ):
            raise CacheMismatchError(f"calibration row identity mismatch at index {index}")
        identity = (record.sequence_id, record.frame_id)
        if identity in seen_identity:
            raise CacheMismatchError(f"duplicate calibration identity: {identity}")
        seen_identity.add(identity)
        parsed.append(record)
    return RectifiedCalibrationIndex(
        sidecar_path=sidecar,
        sidecar_sha256=sidecar_sha256,
        receipt_path=receipt,
        receipt_sha256=receipt_sha256,
        source_manifest_path=manifest_path,
        source_manifest_sha256=manifest_sha256,
        pixel_audit_path=audit_path,
        pixel_audit_sha256=audit_sha256,
        records=tuple(parsed),
    )


def calibration_window_sha256(
    records: Sequence[RectifiedCalibrationRecord],
) -> str:
    """Hash an ordered stereo-calibration window for cache identity binding."""

    if not records:
        raise ValueError("calibration window cannot be empty")
    return canonical_json_sha256(
        {"ordered_calibration_record_sha256": [item.calibration_record_sha256 for item in records]}
    )


__all__ = [
    "RECTIFIED_CALIBRATION_COMPONENT",
    "RECTIFIED_CALIBRATION_CONTRACT",
    "RECTIFIED_CALIBRATION_SIDECAR_SCHEMA_VERSION",
    "RECTIFIED_EXTRINSICS_CONVENTION",
    "RECTIFIED_PIXEL_CONTRACT",
    "RectifiedCalibrationIndex",
    "RectifiedCalibrationRecord",
    "build_rectified_calibration_sidecar",
    "calibration_window_sha256",
    "load_rectified_calibration_sidecar",
]
