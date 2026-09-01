from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch
import numpy as np

from data.cache_dataset import (
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    sha256_file,
)
from data.manifest import ManifestRecord, write_manifest
from data.stereo_calibration import RectifiedCalibrationRecord
from tests.test_derive_geometry_manifest import (
    _prepare_single_window,
    _save_payload,
    _small_thresholds,
)
from tests.test_pose_quality_pipeline import _raw_payloads
from tests.test_stereo_calibration import _write_audit, _write_metadata
from tools.derive_geometry_cache import (
    CALIBRATED_DERIVED_COMPONENT,
    GeometryThresholds,
    derive_geometry,
)
from tools.derive_geometry_manifest import derive_geometry_manifest
from data.stereo_calibration import build_rectified_calibration_sidecar


def _calibration_window(vggt_payload: dict) -> tuple[RectifiedCalibrationRecord, ...]:
    records = vggt_payload["metadata"]["source"]["manifest_records"]
    result = []
    for index, record in enumerate(records):
        baseline = float(record["baseline_m"])
        transform = (
            (1.0, 0.0, 0.0, -baseline),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        result.append(
            RectifiedCalibrationRecord(
                sequence_id=str(record["sequence_id"]),
                frame_id=int(record["frame_id"]),
                timestamp=float(record["timestamp"]),
                source_manifest_index=index,
                source_record_sha256=canonical_json_sha256(record),
                metadata_path=f"metadata-{index}.yaml",
                metadata_sha256=f"metadata-hash-{index}",
                T_right_rectified_from_left_rectified_m=transform,
                diagnostic_T_right_raw_from_left_raw_m=transform,
                calibration_record_sha256=canonical_json_sha256(
                    {"synthetic-calibration-index": index}
                ),
            )
        )
    return tuple(result)


def _thresholds() -> GeometryThresholds:
    return GeometryThresholds(
        min_alignment_pixels=20,
        min_photometric_samples=20,
        min_photometric_valid_fraction=0.9,
    )


def test_calibrated_constraint_preserves_raw_gates_and_replaces_only_right_pose(
    tmp_path: Path,
) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=64, current_value=64
    )
    # Inject a finite raw right-camera y error which remains under the original
    # quality gates. The calibrated output must remove it without rewriting
    # the diagnostic tensor or its raw quality evidence.
    raw_extrinsics = vggt_payload["tensors"][
        "vggt_extrinsics_camera_from_world"
    ]
    raw_extrinsics[1::2, 1, 3] = 0.02
    calibration = _calibration_window(vggt_payload)

    legacy_tensors, legacy_metadata = derive_geometry(
        vggt_payload, ffs_payload, thresholds=_thresholds()
    )
    tensors, metadata = derive_geometry(
        vggt_payload,
        ffs_payload,
        thresholds=_thresholds(),
        rectified_calibration_window=calibration,
    )
    assert bool(tensors["temporal_pose_valid"].item())
    assert bool(tensors["stereo_calibration_valid"].item())
    torch.testing.assert_close(
        tensors["vggt_extrinsics_camera_from_world_metric_diagnostic_only"],
        legacy_tensors["vggt_extrinsics_camera_from_world_metric_diagnostic_only"],
    )
    assert metadata["pose_quality"] == legacy_metadata["pose_quality"]

    constrained = tensors[
        "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
    ]
    diagnostic = tensors[
        "vggt_extrinsics_camera_from_world_metric_diagnostic_only"
    ]
    transform = tensors["T_right_rectified_from_left_rectified_m"]
    assert tuple(transform.shape) == (4, 4)
    for pair_index in range(5):
        left_index = 2 * pair_index
        right_index = left_index + 1
        torch.testing.assert_close(constrained[left_index], diagnostic[left_index])
        left_homogeneous = torch.eye(4)
        left_homogeneous[:3] = diagnostic[left_index]
        expected_right = (transform @ left_homogeneous)[:3]
        torch.testing.assert_close(constrained[right_index], expected_right)
        assert constrained[right_index, 1, 3].item() == pytest.approx(0.0)
        assert diagnostic[right_index, 1, 3].item() != pytest.approx(0.0)
    assert metadata["stereo_calibration"]["quality_policy"].startswith(
        "raw VGGT stereo residuals own"
    )


def test_calibrated_constraint_is_zero_when_original_pose_gate_rejects(
    tmp_path: Path,
) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=0, current_value=255
    )
    tensors, metadata = derive_geometry(
        vggt_payload,
        ffs_payload,
        thresholds=_thresholds(),
        rectified_calibration_window=_calibration_window(vggt_payload),
    )
    assert not bool(tensors["temporal_pose_valid"].item())
    assert torch.count_nonzero(
        tensors[
            "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
        ]
    ).item() == 0
    assert metadata["stereo_calibration"]["hybrid_pose_valid"] is False


def test_calibrated_constraint_rejects_source_or_baseline_mismatch(
    tmp_path: Path,
) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=64, current_value=64
    )
    calibration = list(_calibration_window(vggt_payload))
    first = calibration[0]
    calibration[0] = RectifiedCalibrationRecord(
        sequence_id=first.sequence_id,
        frame_id=first.frame_id,
        timestamp=first.timestamp,
        source_manifest_index=first.source_manifest_index,
        source_record_sha256="wrong-source",
        metadata_path=first.metadata_path,
        metadata_sha256=first.metadata_sha256,
        T_right_rectified_from_left_rectified_m=(
            (1.0, 0.0, 0.0, -0.2),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        diagnostic_T_right_raw_from_left_raw_m=(
            (1.0, 0.0, 0.0, -0.2),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        calibration_record_sha256=first.calibration_record_sha256,
    )
    with pytest.raises(CacheMismatchError, match="record/source hash mismatch"):
        derive_geometry(
            vggt_payload,
            ffs_payload,
            thresholds=_thresholds(),
            rectified_calibration_window=calibration,
        )


def test_batch_derivation_uses_new_component_and_binds_sidecar(
    tmp_path: Path,
) -> None:
    (
        vggt_root,
        ffs_root,
        output_root,
        vggt_path,
        vggt_payload,
        ffs_payload,
    ) = _prepare_single_window(tmp_path, previous_value=64, current_value=64)
    metadata_path = tmp_path / "calibration_meta.yaml"
    _write_metadata(metadata_path)
    baseline = 0.1
    k_left = [[40.0, 0.0, 24.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
    k_right = [[40.0, 0.0, 24.0], [0.0, 40.0, 21.4], [0.0, 0.0, 1.0]]
    p_left = [row + [0.0] for row in k_left]
    p_right = [
        [40.0, 0.0, 24.0, -4.0],
        [0.0, 40.0, 21.4, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    records: list[ManifestRecord] = []
    ordered_images: list[dict] = []
    source_dir = tmp_path / "calibrated_sources"
    source_dir.mkdir()
    for index in range(5):
        left_path = source_dir / f"left_{index}.png"
        right_path = source_dir / f"right_{index}.png"
        Image.fromarray(np.full((32, 48, 3), 64, dtype=np.uint8)).save(left_path)
        Image.fromarray(np.full((32, 48, 3), 64, dtype=np.uint8)).save(right_path)
        record = ManifestRecord(
            sequence_id="synthetic_sequence",
            frame_id=index,
            timestamp=index * 0.2,
            left_path=str(left_path.resolve()),
            right_path=str(right_path.resolve()),
            K=tuple(tuple(value for value in row) for row in k_left),
            baseline_m=baseline,
            gt_disparity_path=None,
            extras={
                "K_right": k_right,
                "P_left": p_left,
                "P_right": p_right,
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
            },
        )
        records.append(record)
        for view_index, path in ((2 * index, left_path), (2 * index + 1, right_path)):
            ordered_images.append(
                {
                    "view_index": view_index,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    manifest_path = tmp_path / "calibrated_manifest.jsonl"
    write_manifest(manifest_path, records)
    audit_path = tmp_path / "pixel_audit.json"
    _write_audit(audit_path, manifest_path, record_count=5)
    sidecar_path = tmp_path / "calibration.jsonl"
    sidecar_receipt = tmp_path / "calibration.receipt.json"
    build_rectified_calibration_sidecar(
        manifest_path,
        audit_path,
        sidecar_path,
        receipt_path=sidecar_receipt,
    )

    vggt_source = vggt_payload["metadata"]["source"]
    vggt_source["manifest_records"] = [record.to_dict() for record in records]
    vggt_source["manifest_indices"] = list(range(5))
    vggt_source["ordered_images"] = ordered_images
    vggt_source["target_sequence_id"] = records[-1].sequence_id
    vggt_source["target_frame_id"] = records[-1].frame_id
    vggt_source["target_timestamp"] = records[-1].timestamp
    ffs_source = ffs_payload["metadata"]["source"]
    ffs_source["manifest_record"] = records[-1].to_dict()
    ffs_source["left_sha256"] = ordered_images[8]["sha256"]
    ffs_source["right_sha256"] = ordered_images[9]["sha256"]
    _save_payload(vggt_path, vggt_payload)
    ffs_path = ffs_root / "synthetic_sequence" / "4.pt"
    _save_payload(ffs_path, ffs_payload)

    receipt = derive_geometry_manifest(
        vggt_root=vggt_root,
        ffs_root=ffs_root,
        output_root=output_root,
        thresholds=_small_thresholds(),
        rectified_calibration_sidecar=sidecar_path,
        rectified_calibration_receipt=sidecar_receipt,
    )
    assert receipt["schema_version"] == 2
    assert receipt["component"] == f"{CALIBRATED_DERIVED_COMPONENT}-batch"
    assert receipt["counts"]["pose_valid"] == 1
    assert receipt["safe_zero_audit"]["calibrated_stereo_records"] == 1
    payload = load_cache_record(output_root / "synthetic_sequence" / "4.pt")
    assert payload["identity"]["component"] == CALIBRATED_DERIVED_COMPONENT
    assert payload["tensors"][
        "T_right_rectified_from_left_rectified_m"
    ].shape == (4, 4)
    assert payload["tensors"][
        "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
    ].shape == (10, 3, 4)
    calibration_source = payload["metadata"]["source"][
        "rectified_stereo_calibration"
    ]
    assert calibration_source["sidecar_sha256"] == sha256_file(sidecar_path)
