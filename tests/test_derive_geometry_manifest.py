from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.cache_dataset import CacheIdentity, CacheMismatchError, save_cache_record
from tests.test_pose_quality_pipeline import _raw_payloads
from tools.derive_geometry_cache import GeometryThresholds
from tools.derive_geometry_manifest import derive_geometry_manifest


def _save_payload(path: Path, payload: dict) -> None:
    save_cache_record(
        path,
        tensors=payload["tensors"],
        metadata=payload["metadata"],
        identity=CacheIdentity(**payload["identity"]),
    )


def _prepare_single_window(
    tmp_path: Path, *, previous_value: int, current_value: int
) -> tuple[Path, Path, Path, Path, dict, dict]:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    vggt_payload, ffs_payload = _raw_payloads(
        source_dir,
        previous_value=previous_value,
        current_value=current_value,
    )
    source = vggt_payload["metadata"]["source"]
    source.update(
        {
            "causal": True,
            "target_manifest_index": 4,
        }
    )
    vggt_root = tmp_path / "vggt"
    ffs_root = tmp_path / "ffs"
    output_root = tmp_path / "derived"
    sequence_id = "synthetic_sequence"
    vggt_path = vggt_root / sequence_id / "4.pt"
    ffs_path = ffs_root / sequence_id / "4.pt"
    _save_payload(vggt_path, vggt_payload)
    _save_payload(ffs_path, ffs_payload)
    row = {
        "selection_index": 0,
        "target_manifest_index": 4,
        "sequence_id": sequence_id,
        "frame_id": 4,
        "timestamp": 0.8,
        "cache_path": str(vggt_path.resolve()),
        "status": "written",
    }
    (vggt_root / "cache_manifest.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (
        vggt_root,
        ffs_root,
        output_root,
        vggt_path,
        vggt_payload,
        ffs_payload,
    )


def _small_thresholds() -> GeometryThresholds:
    return GeometryThresholds(
        min_alignment_pixels=20,
        min_photometric_samples=20,
        min_photometric_valid_fraction=0.9,
    )


def test_batch_writes_reuses_and_rejects_changed_raw_identity(tmp_path: Path) -> None:
    (
        vggt_root,
        ffs_root,
        output_root,
        vggt_path,
        vggt_payload,
        _,
    ) = _prepare_single_window(tmp_path, previous_value=64, current_value=64)
    report = tmp_path / "report.json"
    first = derive_geometry_manifest(
        vggt_root=vggt_root,
        ffs_root=ffs_root,
        output_root=output_root,
        report_path=report,
        thresholds=_small_thresholds(),
    )
    assert first["counts"] == {
        "selected": 1,
        "written": 1,
        "reused": 0,
        "pose_valid": 1,
        "pose_rejected": 0,
        "static_prior_valid": 1,
        "static_prior_rejected": 0,
    }
    assert first["by_sequence"]["synthetic_sequence"]["counts"][
        "selected"
    ] == 1
    assert first["by_sequence"]["synthetic_sequence"]["counts"][
        "pose_valid"
    ] == 1
    assert report.is_file()
    assert (output_root / "synthetic_sequence" / "4.pt").is_file()
    assert first["diagnostic_percentiles"]["depth_weighted_mae_hr_px"][
        "all_available_windows"
    ]["p50"] == pytest.approx(0.0, abs=1e-5)
    assert first["safe_zero_audit"]["passed"]
    assert first["safe_zero_audit"]["records_audited"] == 1
    assert first["safe_zero_audit"]["verification_command"] is None
    assert first["raw_input_audit"]["weights_only_safe_load_records"] == 1
    assert first["raw_input_audit"]["vggt_identity_match_records"] == 1
    assert not first["raw_input_audit"][
        "canonical_receipt_complete_manifest_coverage"
    ]

    second = derive_geometry_manifest(
        vggt_root=vggt_root,
        ffs_root=ffs_root,
        output_root=output_root,
        thresholds=_small_thresholds(),
    )
    assert second["counts"]["written"] == 0
    assert second["counts"]["reused"] == 1
    assert len(list((output_root / "run_receipts").glob("*.json"))) == 2

    canonical_path = output_root / "run_receipt.json"
    more_complete = json.loads(canonical_path.read_text(encoding="utf-8"))
    more_complete["counts"]["selected"] = 2
    canonical_path.write_text(
        json.dumps(more_complete, sort_keys=True) + "\n", encoding="utf-8"
    )
    subset = derive_geometry_manifest(
        vggt_root=vggt_root,
        ffs_root=ffs_root,
        output_root=output_root,
        thresholds=_small_thresholds(),
    )
    assert subset["canonical_update"]["status"] == "preserved_more_complete_existing"
    assert json.loads(canonical_path.read_text(encoding="utf-8"))["counts"][
        "selected"
    ] == 2

    vggt_payload["identity"]["config_sha256"] = "changed-vggt-config"
    _save_payload(vggt_path, vggt_payload)
    with pytest.raises(CacheMismatchError, match="identity mismatch"):
        derive_geometry_manifest(
            vggt_root=vggt_root,
            ffs_root=ffs_root,
            output_root=output_root,
            thresholds=_small_thresholds(),
        )


def test_batch_reports_real_rejection_and_rejected_diagnostics(tmp_path: Path) -> None:
    vggt_root, ffs_root, output_root, *_ = _prepare_single_window(
        tmp_path, previous_value=0, current_value=255
    )
    receipt = derive_geometry_manifest(
        vggt_root=vggt_root,
        ffs_root=ffs_root,
        output_root=output_root,
        thresholds=_small_thresholds(),
    )
    assert receipt["counts"]["pose_valid"] == 0
    assert receipt["counts"]["pose_rejected"] == 1
    assert receipt["counts"]["static_prior_valid"] == 1
    assert receipt["safe_zero_audit"]["pose_rejected_zero_temporal_extrinsics"] == 1
    assert receipt["safe_zero_audit"]["static_rejected_zero_prior_tensors"] == 0
    assert receipt["failure_reason_histogram"][
        "photometric:photometric_residual_exceeds_threshold"
    ] == 1
    photo = receipt["diagnostic_percentiles"][
        "photometric_median_absolute_rgb_residual"
    ]["pose_rejected_available_windows"]
    assert photo["available_count"] == 1
    assert photo["p50"] == pytest.approx(1.0)
    relative_semantics = receipt["quality_gate_semantics"][
        "depth_median_relative_error"
    ]
    assert not relative_semantics["enabled"]
    assert relative_semantics["threshold"] is None
    assert relative_semantics["diagnostic_is_still_aggregated_when_gate_disabled"]


def test_batch_rejects_modified_future_window(tmp_path: Path) -> None:
    (
        vggt_root,
        ffs_root,
        output_root,
        vggt_path,
        vggt_payload,
        _,
    ) = _prepare_single_window(tmp_path, previous_value=64, current_value=64)
    vggt_payload["metadata"]["source"]["manifest_records"][-2]["timestamp"] = 1.0
    _save_payload(vggt_path, vggt_payload)
    with pytest.raises(CacheMismatchError, match="causal timestamps"):
        derive_geometry_manifest(
            vggt_root=vggt_root,
            ffs_root=ffs_root,
            output_root=output_root,
            thresholds=_small_thresholds(),
        )
