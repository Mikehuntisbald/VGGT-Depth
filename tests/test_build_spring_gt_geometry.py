from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
import torch

from data.cache_dataset import (
    CacheIdentity,
    canonical_json_sha256,
    save_cache_record,
    sha256_file,
)
from tools.build_spring_gt_geometry import _selected_records, build_parser, main


@dataclass(frozen=True)
class _Record:
    sequence_id: str
    frame_id: int


def test_sequence_warmup_preserves_original_indices_and_common_t3_floor() -> None:
    records = [
        *(_Record("a", frame_id) for frame_id in range(1, 8)),
        *(_Record("b", frame_id) for frame_id in range(1, 8)),
    ]
    selected = _selected_records(records, sequence_warmup=4)

    assert [
        (index, record.sequence_id, record.frame_id) for index, record in selected
    ] == [
        (4, "a", 5),
        (5, "a", 6),
        (6, "a", 7),
        (11, "b", 5),
        (12, "b", 6),
        (13, "b", 7),
    ]
    # Three retained derived frames are first available at frame 7.
    assert selected[2][1].frame_id == 7
    assert selected[5][1].frame_id == 7


def test_sequence_warmup_validation_and_cli_default() -> None:
    records = [_Record("a", 1)]
    assert (
        build_parser()
        .parse_args(["--manifest", "m", "--observation-root", "o", "--output", "x"])
        .sequence_warmup
        == 0
    )
    with pytest.raises(ValueError, match="non-negative"):
        _selected_records(records, sequence_warmup=-1)
    with pytest.raises(ValueError, match="removed every"):
        _selected_records(records, sequence_warmup=1)


def _write_observation_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    manifest = tmp_path / "manifest.jsonl"
    manifest_record = {
        "sequence_id": "sequence",
        "frame_id": 1,
        "timestamp": 0.0,
        "left_path": "left.png",
        "right_path": "right.png",
        "K": [[10.0, 0.0, 2.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        "baseline_m": 0.1,
        "gt_disparity_path": None,
        "gt_extrinsics_camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    manifest.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")

    observation_root = tmp_path / "observation"
    observation_path = observation_root / "sequence" / "1.pt"
    config = {"role": "observation", "scale": 2}
    identity = CacheIdentity(
        component="ffs-observation",
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="test",
        cuda_version=None,
        config_sha256=canonical_json_sha256(config),
    )
    save_cache_record(
        observation_path,
        tensors={
            "observation_disparity_hr_px": torch.ones(1, 2, 3),
            "observation_trusted_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        },
        metadata={},
        identity=identity,
    )
    cache_manifest = observation_root / "cache_manifest.jsonl"
    cache_manifest.write_text(
        json.dumps(
            {
                "selection_index": 0,
                "sequence_id": "sequence",
                "frame_id": 1,
                "cache_path": str(observation_path.resolve()),
                "cache_sha256": sha256_file(observation_path),
                "status": "written",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation_receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "config": config,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "cache_manifest": str(cache_manifest.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "selected_records": 1,
        "written_records": 1,
        "reused_records": 0,
    }
    (observation_root / "run_receipt.json").write_text(
        json.dumps(observation_receipt), encoding="utf-8"
    )
    return manifest, observation_root, tmp_path / "derived", identity.to_dict()


def test_receipt_binds_current_observation_receipt_inventory_and_identity(
    tmp_path: Path,
) -> None:
    manifest, observation_root, output, identity = _write_observation_fixture(tmp_path)

    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--observation-root",
                str(observation_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    receipt = json.loads((output / "run_receipt.json").read_text(encoding="utf-8"))
    inputs = receipt["inputs"]
    observation_receipt = observation_root / "run_receipt.json"
    observation_manifest = observation_root / "cache_manifest.jsonl"
    assert inputs["observation_run_receipt"] == str(observation_receipt.resolve())
    assert inputs["observation_run_receipt_sha256"] == sha256_file(observation_receipt)
    assert inputs["observation_cache_manifest"] == str(observation_manifest.resolve())
    assert inputs["observation_cache_manifest_sha256"] == sha256_file(
        observation_manifest
    )
    assert inputs["observation_identity"] == identity


def test_observation_inventory_drift_is_rejected_before_geometry_generation(
    tmp_path: Path,
) -> None:
    manifest, observation_root, output, _ = _write_observation_fixture(tmp_path)
    (observation_root / "cache_manifest.jsonl").write_text(
        '{"mutated":true}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="receipt/inventory lineage differs"):
        main(
            [
                "--manifest",
                str(manifest),
                "--observation-root",
                str(observation_root),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
