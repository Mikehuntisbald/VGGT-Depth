from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

import train
import eval as eval_cli
from data.cache_dataset import load_cache_record
from data.manifest import ManifestRecord, load_manifest, write_manifest
from data.spring import (
    SPRING_BASELINE_M,
    SPRING_GT_COMPONENT,
    SPRING_GT_TARGET_TYPE,
    SpringDatasetError,
    build_spring_manifest_records,
    read_spring_disparity,
    read_spring_intrinsics,
)
from data.stereo_calibration import (
    build_rectified_calibration_sidecar,
    load_rectified_calibration_sidecar,
)
from data.training_dataset import cache_path_for_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dsp5(path: Path, value: np.ndarray, *, key: str = "disparity") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=value, compression="gzip")
    return path


def _png(path: Path, *, size: tuple[int, int], value: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)
    return path


def test_spring_dsp5_uses_image_grid_without_value_scaling(tmp_path: Path) -> None:
    stored = np.arange(24, dtype=np.float32).reshape(4, 6)
    stored[0, 0] = 0.0
    stored[2, 2] = np.nan
    source = _dsp5(tmp_path / "disp.dsp5", stored)

    result = read_spring_disparity(source, image_size_hw=(2, 3))

    expected = stored[::2, ::2]
    valid = np.isfinite(expected) & (expected > 0)
    np.testing.assert_array_equal(result.valid_mask, valid)
    np.testing.assert_allclose(
        result.disparity_hr_px[valid], expected[valid], rtol=0.0, atol=0.0
    )
    assert result.disparity_hr_px[0, 0] == 0.0
    assert result.disparity_hr_px[1, 1] == 0.0
    assert result.source_size_hw == (4, 6)
    assert result.target_size_hw == (2, 3)


def test_spring_dsp5_and_intrinsics_fail_closed(tmp_path: Path) -> None:
    bad = _dsp5(tmp_path / "bad.dsp5", np.ones((4, 6), np.float32), key="flow")
    with pytest.raises(SpringDatasetError, match="lacks the 'disparity'"):
        read_spring_disparity(bad, image_size_hw=(2, 3))

    intrinsics = tmp_path / "intrinsics.txt"
    intrinsics.write_text("100 101 50 40\n110 111 51 41\n", encoding="utf-8")
    values = read_spring_intrinsics(intrinsics)
    np.testing.assert_allclose(values[1], [110.0, 111.0, 51.0, 41.0])
    intrinsics.write_text("100 101 50\n", encoding="utf-8")
    with pytest.raises(SpringDatasetError, match=r"shape \[N,4\]"):
        read_spring_intrinsics(intrinsics)


def _official_layout(root: Path, *, frames: int = 2) -> Path:
    sequence = root / "train" / "0001"
    intrinsics = sequence / "cam_data" / "intrinsics.txt"
    intrinsics.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.write_text(
        "".join(
            f"{1000 + index} {1001 + index} 960 540\n"
            for index in range(frames)
        ),
        encoding="utf-8",
    )
    for frame_id in range(1, frames + 1):
        _png(
            sequence / "frame_left" / f"frame_left_{frame_id:04d}.png",
            size=(1920, 1080),
            value=20 + frame_id,
        )
        _png(
            sequence / "frame_right" / f"frame_right_{frame_id:04d}.png",
            size=(1920, 1080),
            value=30 + frame_id,
        )
        # Manifest discovery validates coverage/path only; the dedicated dsp5
        # reader and cache test below own numeric HDF5 semantics.
        disparity = sequence / "disp1_left" / f"disp1_left_{frame_id:04d}.dsp5"
        disparity.parent.mkdir(parents=True, exist_ok=True)
        disparity.write_bytes(b"fixture")
    return root


def test_spring_manifest_and_calibration_sidecar(tmp_path: Path) -> None:
    dataset_root = _official_layout(tmp_path / "spring")
    records, summaries = build_spring_manifest_records(
        dataset_root, split="train"
    )

    assert len(records) == 2
    assert summaries[0].frames == 2
    first = records[0]
    assert first.sequence_id == "spring_train_0001"
    assert first.baseline_m == pytest.approx(SPRING_BASELINE_M)
    assert first.gt_disparity_path is not None
    assert first.K[0][0] == pytest.approx(1000.0)
    assert first.extras["P_right"][0][3] == pytest.approx(
        -1000.0 * SPRING_BASELINE_M
    )
    assert records[1].K[0][0] == pytest.approx(1001.0)

    manifest = tmp_path / "spring_train.jsonl"
    pixel_audit = tmp_path / "spring_train.pixel_audit.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_spring_manifest.py"),
            "--dataset-root",
            str(dataset_root),
            "--split",
            "train",
            "--output",
            str(manifest),
            "--pixel-audit-output",
            str(pixel_audit),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    loaded = load_manifest(manifest)
    assert len(loaded) == 2
    sidecar = tmp_path / "spring_train.calibration.jsonl"
    build_rectified_calibration_sidecar(manifest, pixel_audit, sidecar)
    calibration = load_rectified_calibration_sidecar(
        sidecar, expected_manifest_path=manifest
    )
    transform = calibration.record_for_manifest_index(1).as_tensor()
    torch.testing.assert_close(transform[:3, :3], torch.eye(3))
    assert transform[0, 3].item() == pytest.approx(-SPRING_BASELINE_M)


def test_spring_ground_truth_cache_and_supervision_config(tmp_path: Path) -> None:
    left = _png(tmp_path / "left.png", size=(6, 4), value=10)
    right = _png(tmp_path / "right.png", size=(6, 4), value=20)
    stored = np.arange(1, 97, dtype=np.float32).reshape(8, 12)
    stored[0, 0] = 0.0
    stored[2, 2] = np.nan
    disparity = _dsp5(tmp_path / "disp.dsp5", stored)
    record = ManifestRecord(
        sequence_id="spring_train_0001",
        frame_id=1,
        timestamp=0.0,
        left_path=str(left),
        right_path=str(right),
        K=((100.0, 0.0, 3.0), (0.0, 100.0, 2.0), (0.0, 0.0, 1.0)),
        baseline_m=SPRING_BASELINE_M,
        gt_disparity_path=str(disparity),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    output = tmp_path / "gt_cache"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "cache_spring_gt.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    receipt = json.loads((output / "run_receipt.json").read_text())
    assert receipt["identity"]["component"] == SPRING_GT_COMPONENT
    assert receipt["target_type"] == SPRING_GT_TARGET_TYPE
    assert receipt["statistics"]["maximum_disparity_hr_px"] == pytest.approx(
        float(np.nanmax(stored[::2, ::2]))
    )
    payload = load_cache_record(cache_path_for_record(output, record))
    tensors = payload["tensors"]
    expected = stored[::2, ::2]
    valid = np.isfinite(expected) & (expected > 0)
    torch.testing.assert_close(
        tensors["teacher_disparity_hr_px"][0][torch.from_numpy(valid)],
        torch.from_numpy(expected[valid]),
    )
    assert not tensors["teacher_valid_mask"][0, 0, 0]
    assert not tensors["teacher_valid_mask"][0, 1, 1]
    assert tensors["teacher_confidence"].dtype == torch.float32

    spring_config = train.resolve_config(
        "configs/spring_mvp_x2_v3_1.yaml",
        ["data.calibration_sidecar_path=/tmp/spring-sidecar.jsonl"],
    )
    supervision = train.supervision_target_from_config(spring_config)
    assert supervision.target_type == SPRING_GT_TARGET_TYPE
    assert supervision.cache_component == SPRING_GT_COMPONENT
    assert supervision.paper_ground_truth
    temporal_contract = eval_cli._temporal_metric_contract(
        temporal_metric_v2=True,
        target_type=supervision.target_type,
        paper_ground_truth=supervision.paper_ground_truth,
    )
    assert temporal_contract["reference"] == SPRING_GT_TARGET_TYPE
    assert temporal_contract["paper_gt"] is True
    legacy = train.resolve_config("configs/mvp_x2_v3_1.yaml")
    assert train.supervision_target_from_config(legacy) == train.SupervisionTarget()
