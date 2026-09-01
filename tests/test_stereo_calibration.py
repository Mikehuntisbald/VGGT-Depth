from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from data.cache_dataset import CacheMismatchError, sha256_file
from data.manifest import ManifestRecord, write_manifest
from data.stereo_calibration import (
    RECTIFIED_CALIBRATION_COMPONENT,
    RECTIFIED_CALIBRATION_CONTRACT,
    RECTIFIED_PIXEL_CONTRACT,
    build_rectified_calibration_sidecar,
    load_rectified_calibration_sidecar,
)


BASELINE_M = 0.12
K_LEFT = [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]
K_RIGHT = [[100.0, 0.0, 4.0], [0.0, 100.0, 8.4], [0.0, 0.0, 1.0]]


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def _write_metadata(path: Path) -> None:
    rotation_left = _rotation_z(0.02)
    rotation_right = _rotation_z(-0.01)
    payload = {
        "rectified": True,
        "left_frame_id": "left_optical",
        "right_frame_id": "right_optical",
        "left_rect_camera_info": {"r": rotation_left.reshape(-1).tolist()},
        "right_rect_camera_info": {"r": rotation_right.reshape(-1).tolist()},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _records(tmp_path: Path, *, count: int = 5) -> list[ManifestRecord]:
    metadata = tmp_path / "meta.yaml"
    _write_metadata(metadata)
    p_left = [row + [0.0] for row in K_LEFT]
    p_right = [
        [100.0, 0.0, 4.0, -100.0 * BASELINE_M],
        [0.0, 100.0, 8.4, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return [
        ManifestRecord(
            sequence_id="sequence",
            frame_id=index,
            timestamp=0.2 * index,
            left_path=str(tmp_path / f"left_{index}.png"),
            right_path=str(tmp_path / f"right_{index}.png"),
            K=tuple(tuple(value for value in row) for row in K_LEFT),
            baseline_m=BASELINE_M,
            gt_disparity_path=None,
            extras={
                "K_right": K_RIGHT,
                "P_left": p_left,
                "P_right": p_right,
                "metadata_path": str(metadata.resolve()),
                "metadata_sha256": sha256_file(metadata),
            },
        )
        for index in range(count)
    ]


def _write_audit(path: Path, manifest: Path, *, record_count: int) -> None:
    payload = {
        "schema_version": 1,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS",
        "published_contract": RECTIFIED_PIXEL_CONTRACT,
        "threshold_checks": [
            {"scope": "global", "metric": "test", "passed": True}
        ],
        "manifests": {
            "train": {
                "path": str(manifest.resolve()),
                "sha256": sha256_file(manifest),
                "record_count": record_count,
            }
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build(tmp_path: Path) -> tuple[Path, Path, Path, list[ManifestRecord]]:
    manifest = tmp_path / "manifest.jsonl"
    records = _records(tmp_path)
    write_manifest(manifest, records)
    audit = tmp_path / "pixel_audit.json"
    _write_audit(audit, manifest, record_count=len(records))
    sidecar = tmp_path / "calibration.jsonl"
    receipt = tmp_path / "calibration.receipt.json"
    result = build_rectified_calibration_sidecar(
        manifest, audit, sidecar, receipt_path=receipt
    )
    assert result["status"] == "PASS"
    return manifest, sidecar, receipt, records


def test_build_and_load_sidecar_uses_explicit_rectified_transform(
    tmp_path: Path,
) -> None:
    manifest, sidecar, receipt, records = _build(tmp_path)
    index = load_rectified_calibration_sidecar(
        sidecar, receipt_path=receipt, expected_manifest_path=manifest
    )
    assert len(index.records) == len(records)
    record = index.record_for_manifest_index(3)
    assert record is index.record_for_identity("sequence", 3, records[3].timestamp)
    transform = record.as_tensor().numpy()
    expected = np.eye(4)
    expected[0, 3] = -BASELINE_M
    np.testing.assert_allclose(transform, expected, atol=1e-8, rtol=0.0)
    raw = np.asarray(record.diagnostic_T_right_raw_from_left_raw_m)
    assert np.linalg.norm(raw[:3, 3]) == pytest.approx(BASELINE_M)
    assert np.linalg.det(raw[:3, :3]) == pytest.approx(1.0)

    row = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
    assert row["component"] == RECTIFIED_CALIBRATION_COMPONENT
    assert row["contract_version"] == RECTIFIED_CALIBRATION_CONTRACT
    assert row["derivation"]["runtime_right_vertical_intrinsics_policy"].startswith(
        "pixel audit owns rows"
    )


def test_immutable_rebuild_reuses_identical_bytes_and_rejects_drift(
    tmp_path: Path,
) -> None:
    manifest, sidecar, receipt, _ = _build(tmp_path)
    audit = tmp_path / "pixel_audit.json"
    second = build_rectified_calibration_sidecar(
        manifest, audit, sidecar, receipt_path=receipt
    )
    assert second["immutable_write_status"] == "reused_identical"
    assert second["immutable_receipt_status"] == "reused_identical"

    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CacheMismatchError, match="does not bind the sidecar"):
        load_rectified_calibration_sidecar(sidecar, receipt_path=receipt)
    with pytest.raises(CacheMismatchError, match="immutable calibration artifact differs"):
        build_rectified_calibration_sidecar(
            manifest, audit, sidecar, receipt_path=receipt
        )


def test_builder_rejects_projection_or_audit_mismatch(tmp_path: Path) -> None:
    records = _records(tmp_path)
    bad = records[0].to_dict()
    bad["P_right"][0][3] = abs(bad["P_right"][0][3])
    records[0] = ManifestRecord.from_dict(bad)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)
    audit = tmp_path / "audit.json"
    _write_audit(audit, manifest, record_count=len(records))
    with pytest.raises(CacheMismatchError, match=r"\[-baseline,0,0\]"):
        build_rectified_calibration_sidecar(
            manifest, audit, tmp_path / "sidecar.jsonl"
        )

    good_manifest = tmp_path / "good_manifest.jsonl"
    good_records = _records(tmp_path)
    write_manifest(good_manifest, good_records)
    _write_audit(audit, good_manifest, record_count=len(good_records))
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["status"] = "FAIL"
    audit.write_text(json.dumps(audit_payload), encoding="utf-8")
    with pytest.raises(CacheMismatchError, match="did not pass"):
        build_rectified_calibration_sidecar(
            good_manifest, audit, tmp_path / "sidecar2.jsonl"
        )
