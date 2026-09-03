from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import eval as eval_cli
import train
from data.cache_dataset import CacheIdentity, load_cache_record
from data.manifest import ManifestRecord, write_manifest
from data.spring import SPRING_GT_COMPONENT, SPRING_GT_TARGET_TYPE
from evaluation import validate_checkpoint_lineage
from utils.checkpoint import CheckpointMismatchError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _identity(component: str) -> dict[str, object]:
    return CacheIdentity(
        component=component,
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        config_sha256="c" * 64,
    ).to_dict()


def _supervision_config() -> dict[str, object]:
    return {
        "enabled": True,
        "protocol_version": train.SPRING_SUPERVISION_PROTOCOL,
        "target_type": SPRING_GT_TARGET_TYPE,
        "teacher_cache_component": SPRING_GT_COMPONENT,
        "paper_ground_truth": True,
        "synthetic_ground_truth": True,
    }


def _spatial_checkpoint_config() -> dict[str, object]:
    observation = _identity("ffs-observation")
    target = _identity(SPRING_GT_COMPONENT)
    return {
        "data": {
            "sequence_length": 1,
            "derived_contract": "legacy_v1",
            "observation_cache_identity": observation,
            "teacher_cache_identity": target,
        },
        "train": {"stage": "spatial"},
        "model": {"use_history": False, "use_vggt_pose": False},
        "vggt": {},
        "calibration_conditioning_v3": {
            "enabled": False,
            "protocol_version": "disabled",
            "use_rays": False,
            "use_stereo_pose": False,
            "use_temporal_pose": False,
        },
        "supervision": _supervision_config(),
    }


def test_spring_v31_base_configs_bind_real_gt_without_arm_pose_override() -> None:
    sidecar_override = ["data.calibration_sidecar_path=/tmp/spring-sidecar.jsonl"]
    spatial = eval_cli.resolve_evaluation_config(
        "configs/spring_v3_1/base_spatial_gt.yaml", sidecar_override
    )
    temporal = eval_cli.resolve_evaluation_config(
        "configs/spring_v3_1/base_temporal_gt.yaml", sidecar_override
    )

    for config in (spatial, temporal):
        target = train.supervision_target_from_config(config)
        assert target.target_type == SPRING_GT_TARGET_TYPE
        assert target.cache_component == SPRING_GT_COMPONENT
        assert target.paper_ground_truth
        assert target.synthetic_ground_truth
    assert temporal.data.temporal_pose_source == "vggt"
    assert temporal.model.use_vggt_depth is True
    assert eval_cli.validate_evaluation_config(spatial) == "spatial"
    assert eval_cli.validate_evaluation_config(temporal) == "temporal"


def test_supervision_parser_rejects_partial_or_mislabeled_contract() -> None:
    assert train.supervision_target_from_config({}) == train.SupervisionTarget()
    partial = {"supervision": {"enabled": True}}
    with pytest.raises(ValueError, match="supervision fields differ"):
        train.supervision_target_from_config(partial)
    mislabeled = _supervision_config()
    mislabeled["teacher_cache_component"] = "ffs-teacher"
    with pytest.raises(ValueError, match="Spring supervision identity differs"):
        train.supervision_target_from_config({"supervision": mislabeled})


def test_checkpoint_lineage_binds_exact_supervision_and_cache_component() -> None:
    config = _spatial_checkpoint_config()
    observation = _identity("ffs-observation")
    target = _identity(SPRING_GT_COMPONENT)
    result = validate_checkpoint_lineage(
        {"training_config": config},
        required_stage="spatial",
        observation_cache_identity=observation,
        teacher_cache_identity=target,
        evaluation_config=copy.deepcopy(config),
    )
    assert result["supervision"] == _supervision_config()

    changed = copy.deepcopy(config)
    changed["supervision"]["target_type"] = "wrong"  # type: ignore[index]
    with pytest.raises(CheckpointMismatchError, match="supervision config differs"):
        validate_checkpoint_lineage(
            {"training_config": config},
            required_stage="spatial",
            observation_cache_identity=observation,
            teacher_cache_identity=target,
            evaluation_config=changed,
        )

    wrong_target = _identity("ffs-teacher")
    wrong_config = copy.deepcopy(config)
    wrong_config["data"]["teacher_cache_identity"] = wrong_target  # type: ignore[index]
    with pytest.raises(CheckpointMismatchError, match="target cache component"):
        validate_checkpoint_lineage(
            {"training_config": wrong_config},
            required_stage="spatial",
            observation_cache_identity=observation,
            teacher_cache_identity=wrong_target,
            evaluation_config=wrong_config,
        )


def _write_gt_fixture(tmp_path: Path) -> tuple[Path, ManifestRecord]:
    h5py = pytest.importorskip("h5py")
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(left)
    Image.fromarray(np.ones((2, 3, 3), dtype=np.uint8)).save(right)
    disparity_path = tmp_path / "disp1_left" / "disp1_left_0001.dsp5"
    disparity_path.parent.mkdir()
    stored = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    stored[0, 0] = 0.0
    stored[2, 2] = np.nan
    with h5py.File(disparity_path, "w") as handle:
        handle["disparity"] = stored
    record = ManifestRecord(
        sequence_id="spring-sequence",
        frame_id=1,
        timestamp=0.0,
        left_path=str(left),
        right_path=str(right),
        K=((100.0, 0.0, 1.0), (0.0, 100.0, 1.0), (0.0, 0.0, 1.0)),
        baseline_m=0.065,
        gt_disparity_path=str(disparity_path),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    return manifest, record


def test_spring_gt_cache_has_independent_component_and_real_gt_receipt(
    tmp_path: Path,
) -> None:
    manifest, record = _write_gt_fixture(tmp_path)
    output = tmp_path / "cache"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "cache_spring_gt.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    root = output / "teacher"
    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["identity_version"] == 2
    assert receipt["identity"]["component"] == SPRING_GT_COMPONENT
    assert receipt["target_type"] == SPRING_GT_TARGET_TYPE
    assert receipt["paper_ground_truth"] is True
    assert receipt["synthetic_ground_truth"] is True
    identity = train.load_receipt_identity(
        root,
        expected_component=SPRING_GT_COMPONENT,
        manifest_path=manifest,
    )
    with pytest.raises(ValueError, match="component mismatch"):
        train.load_receipt_identity(
            root,
            expected_component="ffs-teacher",
            manifest_path=manifest,
        )
    payload = load_cache_record(
        root / record.sequence_id / f"{record.frame_id}.pt",
        expected_identity=identity,
    )
    assert payload["tensors"]["teacher_disparity_hr_px"].dtype == torch.float32
    assert payload["metadata"]["config"]["target_type"] == SPRING_GT_TARGET_TYPE
    assert receipt["statistics"]["valid_pixels"] == 4
