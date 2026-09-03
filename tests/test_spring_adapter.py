from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.manifest import load_manifest
from data.spring import (
    SPRING_BASELINE_M,
    SPRING_FLOW_LIBRARY_COMMIT,
    SPRING_INTRINSICS_FORMAT,
    SpringFormatError,
    build_spring_manifest,
    load_spring_sequence,
    relative_spring_pose,
    spring_gt_pose_from_manifest,
    spring_manifest_records,
)


def _write_sequence(root: Path, *, with_pixels: bool = False) -> Path:
    sequence = root / "spring" / "train" / "0001"
    cam = sequence / "cam_data"
    cam.mkdir(parents=True)
    (cam / "intrinsics.txt").write_text(
        "100 101 960 540\n110 111 960 540\n", encoding="utf-8"
    )
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[0, 3] = -1.0
    np.savetxt(cam / "extrinsics.txt", np.stack((pose0, pose1)).reshape(2, 16))
    (cam / "focaldistance.txt").write_text("10\n100\n", encoding="utf-8")
    if with_pixels:
        for side in ("left", "right"):
            (sequence / f"frame_{side}").mkdir()
            (sequence / f"disp1_{side}").mkdir()
            for frame in (1, 2):
                (sequence / f"frame_{side}" / f"frame_{side}_{frame:04d}.png").touch()
                (sequence / f"disp1_{side}" / f"disp1_{side}_{frame:04d}.dsp5").touch()
    return root


def test_spring_sidecars_parse_and_relative_pose_is_camera_from_world(
    tmp_path: Path,
) -> None:
    root = _write_sequence(tmp_path)
    sequence = load_spring_sequence(
        root / "spring" / "train" / "0001",
        require_images=False,
        require_disparity=False,
    )
    assert len(sequence.frames) == 2
    first, second = sequence.frames
    assert first.frame_id == 1
    assert second.K[0][0] == pytest.approx(110.0)
    assert first.focal_distance_m == pytest.approx(10.0)
    np.testing.assert_allclose(first.right_pose()[0, 3], -SPRING_BASELINE_M)
    relative = relative_spring_pose(first, second)
    np.testing.assert_allclose(relative[:3, :3], np.eye(3))
    np.testing.assert_allclose(relative[:3, 3], [-1.0, 0.0, 0.0])


def test_spring_manifest_contains_explicit_gt_pose_and_disparity_contract(
    tmp_path: Path,
) -> None:
    root = _write_sequence(tmp_path)
    records = spring_manifest_records(
        root,
        require_images=False,
        require_disparity=False,
        timestamp_fps=10.0,
    )
    assert [record.frame_id for record in records] == [1, 2]
    assert [record.timestamp for record in records] == [0.0, 0.1]
    assert records[0].baseline_m == pytest.approx(0.065)
    assert records[0].extras["gt_pose_convention"] == "world_to_camera_opencv"
    assert records[0].extras["gt_disparity_unit"] == "full_hd_pixels"
    extras = records[0].extras
    np.testing.assert_allclose(extras["K_right"], records[0].K)
    np.testing.assert_allclose(extras["P_left"], np.c_[records[0].K, np.zeros(3)])
    assert extras["P_right"][0][3] == pytest.approx(
        -records[0].K[0][0] * SPRING_BASELINE_M
    )
    assert extras["metadata_path"].endswith("cam_data/intrinsics.txt")
    assert len(extras["metadata_sha256"]) == 64
    assert extras["calibration_metadata_format"] == SPRING_INTRINSICS_FORMAT
    assert extras["calibration_metadata_row"] == 0
    assert extras["spring_flow_library_commit"] == SPRING_FLOW_LIBRARY_COMMIT
    np.testing.assert_allclose(
        records[1].extras["gt_extrinsics_camera_from_world"][0],
        [1.0, 0.0, 0.0, -1.0],
    )

    output = tmp_path / "manifest.jsonl"
    build_spring_manifest(
        root,
        output,
        require_images=False,
        require_disparity=False,
    )
    loaded = load_manifest(output)
    assert len(loaded) == 2
    assert loaded[0].extras["dataset"] == "spring"
    np.testing.assert_allclose(
        spring_gt_pose_from_manifest(loaded[1]),
        np.asarray(loaded[1].extras["gt_extrinsics_camera_from_world"]),
    )


def test_spring_strict_pixel_presence_fails_closed(tmp_path: Path) -> None:
    root = _write_sequence(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing Spring image"):
        spring_manifest_records(root)


def test_official_spring_manifest_rejects_noncanonical_baseline(
    tmp_path: Path,
) -> None:
    root = _write_sequence(tmp_path)
    with pytest.raises(ValueError, match="official Spring manifests require baseline"):
        spring_manifest_records(
            root,
            require_images=False,
            require_disparity=False,
            baseline_m=0.12,
        )


def test_spring_sidecar_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _write_sequence(tmp_path)
    focal = root / "spring" / "train" / "0001" / "cam_data" / "focaldistance.txt"
    focal.write_text("10\n", encoding="utf-8")
    with pytest.raises(SpringFormatError, match="row counts disagree"):
        load_spring_sequence(
            root / "spring" / "train" / "0001",
            require_images=False,
            require_disparity=False,
        )


def test_spring_dsp5_image_sampling_keeps_full_hd_disparity_units(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    from data.spring import load_spring_disparity

    path = tmp_path / "disp1_left" / "disp1_left_0001.dsp5"
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        handle["disparity"] = np.arange(16, dtype=np.float16).reshape(4, 4)
    sampled = load_spring_disparity(path, resolution="image")
    np.testing.assert_array_equal(sampled, [[0.0, 2.0], [8.0, 10.0]])
    np.testing.assert_array_equal(
        load_spring_disparity(path, sign="left_to_right"),
        -np.arange(16, dtype=np.float32).reshape(4, 4),
    )
