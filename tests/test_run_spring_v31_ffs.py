from __future__ import annotations

import json
from pathlib import Path

from tools.run_spring_v31_ffs import (
    GT_POSE_QUALITY_SCORE_OVERRIDE,
    _cache_complete,
    _eval_command,
    _override_complete,
    _sha256,
)
from tools.eval_spring_baseline import _load_cache_lineage


def _write_cache_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "0001",
                "frame_id": 1,
                "timestamp": 0.0,
                "left_path": "left.png",
                "right_path": "right.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "cache" / "observation"
    (root / "0001").mkdir(parents=True)
    cache_path = root / "0001" / "1.pt"
    cache_path.write_bytes(b"fixture")
    (root / "cache_manifest.jsonl").write_text(
        json.dumps(
            {
                "selection_index": 0,
                "sequence_id": "0001",
                "frame_id": 1,
                "timestamp": 0.0,
                "cache_path": str(cache_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import hashlib

    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = {
        "component": "ffs-observation",
        "checkpoint_sha256": "checkpoint-sha",
        "upstream_commit": "fastfs-commit",
        "torch_version": "torch-test",
        "cuda_version": None,
        "config_sha256": "config-sha",
    }
    config = {
        "role": "observation",
        "scale": 2,
        "resolution_mode": "mvp",
        "iterations": 4,
        "max_disp": 192,
        "volume_backend": "pytorch1",
        "right_left_check": True,
        "checkpoint_label": "20-30-48",
        "expected_checkpoint_label": "20-30-48",
        "provisional_checkpoint_role": False,
        "missing_normalize": "error",
    }
    (root / "run_receipt.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "selected_records": 1,
                "written_records": 1,
                "reused_records": 0,
                "identity": identity,
                "config": config,
            }
        ),
        encoding="utf-8",
    )
    return root, manifest, config, identity


def test_spring_v31_cache_reuse_requires_exact_recipe(tmp_path: Path) -> None:
    root, manifest, config, identity = _write_cache_fixture(tmp_path)

    assert _cache_complete(
        root,
        manifest,
        component="ffs-observation",
        expected_config=config,
        expected_identity=identity,
    )

    wrong_config = dict(config)
    wrong_config["max_disp"] = 416
    assert not _cache_complete(
        root,
        manifest,
        component="ffs-observation",
        expected_config=wrong_config,
        expected_identity=identity,
    )

    wrong_identity = dict(identity)
    wrong_identity["checkpoint_sha256"] = "different-checkpoint"
    assert not _cache_complete(
        root,
        manifest,
        component="ffs-observation",
        expected_config=config,
        expected_identity=wrong_identity,
    )


def test_baseline_lineage_rejects_wrong_f1_resolution_recipe(tmp_path: Path) -> None:
    root, manifest, config, identity = _write_cache_fixture(tmp_path)
    lineage, parsed_identity = _load_cache_lineage(
        root, manifest, role="observation", arm="F1"
    )
    assert lineage["identity"] == identity
    assert parsed_identity.checkpoint_sha256 == "checkpoint-sha"

    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    receipt["config"] = {**config, "max_disp": 416}
    (root / "run_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    try:
        _load_cache_lineage(root, manifest, role="observation", arm="F1")
    except ValueError as exc:
        assert "max_disp=192" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("wrong F1 max-disp recipe was accepted")


def _write_calibrated_override_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create the smallest receipt/index pair for override-reuse auditing."""

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "0001",
                "frame_id": 1,
                "timestamp": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    source_manifest = source_root / "cache_manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "selection_index": 0,
                "target_manifest_index": 0,
                "sequence_id": "0001",
                "frame_id": 1,
                "timestamp": 0.0,
                "pose_valid": False,
                "static_prior_valid": False,
                "failure_reasons": ["baseline_rotation:baseline_cv_exceeds_threshold"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_receipt = {
        "schema_version": 2,
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "manifest_sha256": _sha256(manifest),
        "config": {"algorithm": "calibrated-test"},
        "counts": {
            "selected": 1,
            "written": 1,
            "reused": 0,
            "pose_valid": 0,
            "pose_rejected": 1,
            "static_prior_valid": 0,
            "static_prior_rejected": 1,
        },
    }
    (source_root / "run_receipt.json").write_text(
        json.dumps(source_receipt), encoding="utf-8"
    )
    cache_path = output_root / "0001" / "1.pt"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"override-cache")
    sidecar = tmp_path / "calibration.jsonl"
    sidecar.write_text("calibration\n", encoding="utf-8")
    sidecar_receipt = sidecar.with_suffix(".receipt.json")
    sidecar_receipt.write_text("receipt\n", encoding="utf-8")
    output_manifest = output_root / "cache_manifest.jsonl"
    output_manifest.write_text(
        json.dumps(
            {
                "selection_index": 0,
                "target_manifest_index": 0,
                "sequence_id": "0001",
                "frame_id": 1,
                "timestamp": 0.0,
                "cache_path": str(cache_path),
                "pose_valid": True,
                "static_prior_valid": False,
                "source_pose_valid": False,
                "source_static_prior_valid": False,
                "source_failure_reasons": [
                    "baseline_rotation:baseline_cv_exceeds_threshold"
                ],
                "failure_reasons": [],
                "quality_score_override": GT_POSE_QUALITY_SCORE_OVERRIDE,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calibration = {
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": _sha256(sidecar),
        "receipt_path": str(sidecar_receipt.resolve()),
        "receipt_sha256": _sha256(sidecar_receipt),
    }
    output_receipt = {
        "schema_version": 2,
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "manifest_sha256": _sha256(manifest),
        "config": {
            "pose_source": "Spring_GT_pose",
            "depth_source": "copied_from_vggt_derived",
            "quality_score_override": GT_POSE_QUALITY_SCORE_OVERRIDE,
            "source_derived_manifest_sha256": _sha256(source_manifest),
            "source_derived_receipt_sha256": _sha256(source_root / "run_receipt.json"),
            "rectified_stereo_calibration": calibration,
        },
        "counts": {
            "selected": 1,
            "written": 1,
            "reused": 0,
            "pose_valid": 1,
            "pose_rejected": 0,
            "source_pose_valid": 0,
            "source_pose_rejected": 1,
            "source_static_prior_valid": 0,
            "source_static_prior_rejected": 1,
        },
        "pose_override": {
            "quality_score_override": GT_POSE_QUALITY_SCORE_OVERRIDE
        },
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(output_receipt), encoding="utf-8"
    )
    return output_root, manifest, source_root, sidecar


def test_calibrated_override_reuse_requires_marker_and_source_rejection_audit(
    tmp_path: Path,
) -> None:
    output, manifest, source, sidecar = _write_calibrated_override_fixture(tmp_path)
    assert _override_complete(
        output, manifest, source, calibrated=True, sidecar=sidecar
    )

    receipt_path = output / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["config"].pop("quality_score_override")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not _override_complete(
        output, manifest, source, calibrated=True, sidecar=sidecar
    )


def test_calibrated_override_reuse_rejects_tampered_source_rejection_fields(
    tmp_path: Path,
) -> None:
    output, manifest, source, sidecar = _write_calibrated_override_fixture(tmp_path)
    output_manifest = output / "cache_manifest.jsonl"
    row = json.loads(output_manifest.read_text(encoding="utf-8"))
    row["source_pose_valid"] = True
    output_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert not _override_complete(
        output, manifest, source, calibrated=True, sidecar=sidecar
    )


def test_temporal_eval_command_keeps_formal_holdout_lineage(tmp_path: Path) -> None:
    paths = {
        "val_manifest": tmp_path / "validation.jsonl",
        "val_ffs_half": tmp_path / "ffs_half",
        "val_teacher": tmp_path / "teacher",
        "endpoint_list": tmp_path / "common_endpoints.json",
    }
    command = _eval_command(
        paths,
        arm="F4",
        config=tmp_path / "F4.yaml",
        checkpoint=tmp_path / "F4.pt",
        output=tmp_path / "eval",
        device="cuda:0",
        derived=tmp_path / "derived",
        sidecar=tmp_path / "calibration.jsonl",
        limit=1302,
    )
    assert "--allow-non-holdout-smoke" not in command
    assert command[command.index("--crop-mode") + 1] == "fixed"
    assert command[command.index("--crop-origin") + 1 : command.index("--crop-origin") + 3] == [
        "576",
        "348",
    ]
    assert command[command.index("--limit") + 1] == "1302"
