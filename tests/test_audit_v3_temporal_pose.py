from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from tools.audit_v3_temporal_pose import (
    TemporalPoseAuditError,
    audit_temporal_pose_variation,
)


def _write_audit_cache(root: Path, translations_m: list[float]) -> None:
    root.mkdir(parents=True)
    rows = []
    for index, translation_m in enumerate(translations_m):
        poses = torch.zeros((10, 3, 4), dtype=torch.float32)
        poses[:, :3, :3] = torch.eye(3)
        # E_current @ inv(E_history) gives +translation for the chosen history.
        poses[6, 0, 3] = -translation_m
        poses[4, 0, 3] = -2.0 * translation_m
        cache_path = root / f"{index}.pt"
        torch.save(
            {
                "metadata": {
                    "source": {
                        "linkage": {
                            "target_manifest_record": {"baseline_m": 0.1}
                        }
                    }
                },
                "tensors": {
                    "temporal_pose_valid": torch.tensor(True),
                    (
                        "vggt_extrinsics_camera_from_world_metric_temporal_"
                        "stereo_constrained"
                    ): poses,
                },
            },
            cache_path,
        )
        rows.append(
            {
                "sequence_id": "sequence",
                "frame_id": index,
                "timestamp": float(index),
                "target_manifest_index": index,
                "pose_valid": True,
                "cache_path": str(cache_path),
                "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            }
        )
    manifest_path = root / "cache_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "component": (
                    "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
                ),
                "output": {
                    "cache_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )


def test_temporal_pose_audit_detects_both_age_variations(tmp_path: Path) -> None:
    root = tmp_path / "derived"
    _write_audit_cache(root, [0.01, 0.03, 0.07])

    report = audit_temporal_pose_variation(
        root,
        minimum_valid_windows=2,
        minimum_translation_std_over_baseline=0.01,
        minimum_rotation_std_deg=0.1,
    )

    assert report["status"] == "PASS"
    assert report["temporal_pose_varies"] is True
    assert report["counts"]["pose_valid_windows"] == 3
    assert report["ages"]["1"]["varies"] is True
    assert report["ages"]["2"]["varies"] is True


def test_temporal_pose_audit_fails_closed_for_constant_input(tmp_path: Path) -> None:
    root = tmp_path / "derived"
    _write_audit_cache(root, [0.02, 0.02, 0.02])

    report = audit_temporal_pose_variation(root, minimum_valid_windows=2)

    assert report["status"] == "FAIL"
    assert report["temporal_pose_varies"] is False


def test_temporal_pose_audit_rejects_manifest_cache_validity_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "derived"
    _write_audit_cache(root, [0.01, 0.03])
    artifact = torch.load(root / "0.pt", map_location="cpu", weights_only=False)
    artifact["tensors"]["temporal_pose_valid"] = torch.tensor(False)
    torch.save(artifact, root / "0.pt")
    rows = [
        json.loads(line)
        for line in (root / "cache_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[0]["cache_sha256"] = hashlib.sha256((root / "0.pt").read_bytes()).hexdigest()
    manifest_path = root / "cache_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    receipt["output"]["cache_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    (root / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TemporalPoseAuditError, match="manifest/cache"):
        audit_temporal_pose_variation(root, minimum_valid_windows=2)


def test_temporal_pose_audit_uses_exact_formal_t3_endpoints(tmp_path: Path) -> None:
    root = tmp_path / "derived"
    # The first six records vary but are not formal T=3 endpoints. The last
    # four are constant, so a correctly endpoint-bound audit must fail.
    _write_audit_cache(
        root,
        [0.01, 0.03, 0.07, 0.11, 0.17, 0.23, 0.05, 0.05, 0.05, 0.05],
    )
    derived_manifest = root / "cache_manifest.jsonl"
    derived_rows = derived_manifest.read_text(encoding="utf-8").splitlines()[4:]
    derived_manifest.write_text("\n".join(derived_rows) + "\n", encoding="utf-8")
    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    receipt["output"]["cache_manifest_sha256"] = hashlib.sha256(
        derived_manifest.read_bytes()
    ).hexdigest()
    (root / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        "".join(
            json.dumps(
                {
                    "sequence_id": "sequence",
                    "frame_id": index,
                    "timestamp": float(index),
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )

    report = audit_temporal_pose_variation(
        root,
        validation_manifest_path=validation,
        minimum_valid_windows=2,
    )

    assert report["status"] == "FAIL"
    assert report["counts"]["formal_temporal_endpoints"] == 4
    assert report["counts"]["formal_pose_valid_windows"] == 4
    assert report["formal_endpoint_binding"]["record_ids"] == [
        f"sequence/{index}" for index in range(6, 10)
    ]
