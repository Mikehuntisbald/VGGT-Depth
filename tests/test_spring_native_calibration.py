from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.cache_dataset import sha256_file
from data.manifest import ManifestRecord, write_manifest
from data.stereo_calibration import (
    RECTIFIED_PIXEL_CONTRACT,
    build_rectified_calibration_sidecar,
    load_rectified_calibration_sidecar,
)


def test_spring_native_sidecar_preserves_legacy_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "spring.jsonl"
    record = ManifestRecord(
        sequence_id="spring",
        frame_id=1,
        timestamp=0.0,
        left_path=str(tmp_path / "left.png"),
        right_path=str(tmp_path / "right.png"),
        K=((100.0, 0.0, 3.0), (0.0, 100.0, 2.0), (0.0, 0.0, 1.0)),
        baseline_m=0.065,
        gt_disparity_path=None,
        extras={"dataset": "spring", "image_shape_hw": [4, 6]},
    )
    write_manifest(manifest, [record])
    before = manifest.read_bytes()
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "component": "pixel-level-epipolar-rectification-audit",
                "status": "PASS",
                "published_contract": RECTIFIED_PIXEL_CONTRACT,
                "threshold_checks": [{"passed": True}],
                "manifests": {
                    "train": {
                        "path": str(manifest.resolve()),
                        "sha256": sha256_file(manifest),
                        "record_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    sidecar = tmp_path / "calibration.jsonl"
    receipt = tmp_path / "calibration.receipt.json"
    build_rectified_calibration_sidecar(
        manifest,
        audit,
        sidecar,
        receipt_path=receipt,
        spring_native=True,
        spring_metadata_root=tmp_path / "metadata",
    )
    assert manifest.read_bytes() == before
    index = load_rectified_calibration_sidecar(
        sidecar, receipt_path=receipt, expected_manifest_path=manifest
    )
    assert index.spring_native is True
    assert index.records[0].as_tensor()[0, 3].item() == pytest.approx(-0.065)
