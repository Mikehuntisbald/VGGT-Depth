from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.manifest import ManifestRecord, write_manifest
from data.raw_stereo_video_dataset import (
    RawStereoVideoClipDataset,
    collate_raw_stereo_video_samples,
)


def _write_rgb(path: Path, *, height: int, width: int, offset: int) -> None:
    _, x = np.mgrid[:height, :width]
    values = np.broadcast_to(x + offset, (height, width)).astype(np.uint8)
    image = np.repeat(values[..., None], 3, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _make_manifest(
    tmp_path: Path,
    *,
    frame_count: int = 5,
    write_all_gt: bool = True,
    dynamic_frame_ids: set[int] | None = None,
) -> Path:
    height, width = 6, 10
    records: list[ManifestRecord] = []
    for index in range(frame_count):
        left = tmp_path / "pixels" / f"left-{index}.png"
        right = tmp_path / "pixels" / f"right-{index}.png"
        gt_left = tmp_path / "pixels" / f"disp-left-{index}.npy"
        gt_right = tmp_path / "pixels" / f"disp-right-{index}.npy"
        _write_rgb(left, height=height, width=width, offset=10 * index)
        _write_rgb(right, height=height, width=width, offset=100 + 10 * index)
        if write_all_gt or index in {1, 2}:
            np.save(gt_left, np.full((height, width), 4.0 + index, np.float32))
        if write_all_gt or index == 2:
            np.save(gt_right, np.full((height, width), 5.0 + index, np.float32))

        K_left = np.asarray([[100.0, 0.0, 5.0], [0.0, 120.0, 3.0], [0.0, 0.0, 1.0]])
        K_right = np.asarray([[110.0, 0.0, 5.0], [0.0, 121.0, 3.0], [0.0, 0.0, 1.0]])
        P_left = np.concatenate((K_left, np.zeros((3, 1))), axis=1)
        P_right = np.concatenate((K_right, np.zeros((3, 1))), axis=1)
        P_right[0, 3] = -K_right[0, 0] * 0.2
        pose = np.eye(4)
        pose[0, 3] = 0.25 * index
        records.append(
            ManifestRecord(
                sequence_id="sequence-a",
                frame_id=index,
                timestamp=0.1 * index,
                left_path=str(left.relative_to(tmp_path)),
                right_path=str(right.relative_to(tmp_path)),
                K=tuple(tuple(float(item) for item in row) for row in K_left),
                baseline_m=0.2,
                gt_disparity_path=str(gt_left.relative_to(tmp_path)),
                extras={
                    "dataset": "spring",
                    "K_right": K_right.tolist(),
                    "P_left": P_left.tolist(),
                    "P_right": P_right.tolist(),
                    "gt_disparity_right_path": str(gt_right.relative_to(tmp_path)),
                    "gt_extrinsics_camera_from_world": pose.tolist(),
                    "gt_pose_convention": "world_to_camera_opencv",
                },
            )
        )
        if dynamic_frame_ids is not None and index in dynamic_frame_ids:
            dynamic_image = np.zeros((height, width), dtype=bool)
            dynamic_image[1:5, 2:5] = True
            dynamic_2x = np.repeat(np.repeat(dynamic_image, 2, axis=0), 2, axis=1)
            dynamic_root = tmp_path / "maps" / "rigidmap_BW_left"
            dynamic_root.mkdir(parents=True, exist_ok=True)
            Image.fromarray(dynamic_2x).save(
                dynamic_root / f"rigidmap_BW_left_{index:04d}.png"
            )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)
    return manifest


def _dataset(manifest: Path, length: int) -> RawStereoVideoClipDataset:
    return RawStereoVideoClipDataset(
        manifest,
        clip_length=length,
        crop_size_hw=(4, 6),
        resize_size_hw=(2, 3),
        crop_mode="fixed",
        fixed_crop_origin_xy=(2, 1),
    )


def test_raw_clip_contract_crop_resize_geometry_and_last_target(tmp_path: Path) -> None:
    # Frame zero and frame-one right GT deliberately do not exist.  The default
    # policy reads only endpoint L/R plus the immediately previous left GT.
    manifest = _make_manifest(tmp_path, write_all_gt=False, dynamic_frame_ids={2})
    dataset = _dataset(manifest, 3)

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample.frame_ids == (0, 1, 2)
    assert sample.manifest_indices == (0, 1, 2)
    assert sample.rgb.shape == (3, 2, 3, 2, 3)
    assert sample.K.shape == (3, 2, 3, 3)
    assert sample.T_right_from_left.shape == (3, 4, 4)
    assert sample.T_current_from_previous.shape == (3, 4, 4)
    assert sample.disparity_gt_left_px.shape == (3, 1, 2, 3)
    assert sample.disparity_gt_px.shape == (3, 2, 1, 2, 3)

    expected_left_K = torch.tensor(
        [[50.0, 0.0, 1.25], [0.0, 60.0, 0.75], [0.0, 0.0, 1.0]]
    )
    expected_right_K = torch.tensor(
        [[55.0, 0.0, 1.25], [0.0, 60.5, 0.75], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(sample.K[:, 0], expected_left_K.expand(3, -1, -1))
    torch.testing.assert_close(sample.K[:, 1], expected_right_K.expand(3, -1, -1))
    torch.testing.assert_close(
        sample.T_right_from_left[:, 0, 3], torch.full((3,), -0.2)
    )
    torch.testing.assert_close(
        sample.T_current_from_previous[1:, 0, 3], torch.full((2,), 0.25)
    )
    torch.testing.assert_close(sample.T_current_from_previous[0], torch.eye(4))
    assert sample.temporal_transform_valid.tolist() == [False, True, True]
    assert sample.camera_pose_valid.tolist() == [True, True, True]

    assert sample.target_time_mask.tolist() == [False, False, True]
    assert torch.count_nonzero(sample.valid_gt_left[:2]) == 0
    assert torch.count_nonzero(sample.disparity_gt_left_px[:2]) == 0
    torch.testing.assert_close(
        sample.disparity_gt_left_px[-1], torch.full((1, 2, 3), 3.0)
    )
    torch.testing.assert_close(
        sample.disparity_gt_right_px[-1], torch.full((1, 2, 3), 3.5)
    )
    assert bool(sample.valid_gt_left[-1].all())
    assert sample.identity_metadata["crop_xywh"] == [2, 1, 6, 4]
    assert sample.identity_metadata["resize_coordinate_convention"] == (
        "align_corners_false_half_pixel"
    )
    assert sample.identity_metadata["source_paths"][0]["gt_disparity_left"] is None

    assert sample.previous_disparity_gt_available.item()
    assert sample.previous_disparity_gt_left_px.shape == (1, 2, 3)
    torch.testing.assert_close(
        sample.previous_disparity_gt_left_px, torch.full((1, 2, 3), 2.5)
    )
    assert bool(sample.previous_valid_gt_left.all())
    assert sample.previous_disparity_gt_left_valid is sample.previous_valid_gt_left
    previous_source = sample.identity_metadata["source_paths"][1]
    assert previous_source["gt_disparity_right"] is None
    assert previous_source["temporal_previous_gt_disparity_left"].endswith(
        "disp-left-1.npy"
    )

    assert sample.dynamic_mask_available.item()
    assert sample.dynamic_mask_current.shape == (1, 2, 3)
    assert sample.dynamic_mask_current.tolist() == [
        [[True, False, False], [True, False, False]]
    ]
    dynamic_contract = sample.identity_metadata["dynamic_mask_contract"]
    assert dynamic_contract["semantic"] == "white_or_true_is_dynamic_and_excluded"
    assert dynamic_contract["observed_pillow_mode"] == "1"
    assert dynamic_contract["observed_size_hw"] == [12, 20]
    assert dynamic_contract["image_grid_reduction"].startswith("2x2_majority")

    # A synchronized geometric resize preserves the constant view offset.
    torch.testing.assert_close(
        sample.rgb[:, 1] - sample.rgb[:, 0],
        torch.full_like(sample.rgb[:, 0], 100.0 / 255.0),
        atol=2e-6,
        rtol=0.0,
    )


def test_variable_length_collate_left_pads_and_aligns_endpoints(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path, write_all_gt=True)
    short = _dataset(manifest, 2)[0]
    long = _dataset(manifest, 4)[0]
    batch = collate_raw_stereo_video_samples([short, long])

    assert batch["rgb"].shape == (2, 4, 2, 3, 2, 3)
    assert batch["K"].shape == (2, 4, 2, 3, 3)
    assert batch["disparity_gt_px"].shape == (2, 4, 2, 1, 2, 3)
    assert batch["clip_lengths"].tolist() == [2, 4]
    assert batch["time_valid_mask"].tolist() == [
        [False, False, True, True],
        [True, True, True, True],
    ]
    assert batch["frame_ids"].tolist() == [[-1, -1, 0, 1], [0, 1, 2, 3]]
    assert batch["target_time_mask"].tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]
    assert batch["previous_disparity_gt_left_px"].shape == (2, 1, 2, 3)
    assert batch["previous_valid_gt_left"].shape == (2, 1, 2, 3)
    assert batch["previous_disparity_gt_available"].tolist() == [True, True]
    assert batch["dynamic_mask_current"].shape == (2, 1, 2, 3)
    assert batch["dynamic_mask_available"].tolist() == [False, False]
    assert batch["previous_disparity_gt_left_valid"] is batch["previous_valid_gt_left"]
    assert torch.count_nonzero(batch["rgb"][0, :2]) == 0
    torch.testing.assert_close(batch["rgb"][0, -1], short.rgb[-1])
    torch.testing.assert_close(batch["rgb"][1, -1], long.rgb[-1])
    assert batch["T_current_from_previous_valid"] is batch["temporal_transform_valid"]


def test_variable_clip_length_is_deterministic_and_stays_causal(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path, frame_count=7, write_all_gt=True)
    dataset = RawStereoVideoClipDataset(
        manifest,
        clip_length=(2, 4),
        crop_size_hw=(4, 6),
        crop_mode="fixed",
    )
    first = dataset[-1]
    repeated = dataset[-1]
    assert 2 <= first.clip_length <= 4
    assert first.frame_ids == repeated.frame_ids
    assert first.frame_ids == tuple(sorted(first.frame_ids))
    assert first.frame_ids[-1] == 6
    assert all(timestamp <= first.timestamps[-1] for timestamp in first.timestamps)
    dataset.set_epoch(3)
    assert 2 <= dataset[-1].clip_length <= 4


def test_all_target_mode_loads_dense_supervision(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path, frame_count=3, write_all_gt=True)
    dataset = RawStereoVideoClipDataset(
        manifest,
        clip_length=3,
        crop_size_hw=(4, 6),
        crop_mode="fixed",
        target_mode="all",
    )
    sample = dataset[0]
    assert sample.target_time_mask.tolist() == [True, True, True]
    assert bool(sample.valid_gt_left.all())
    assert bool(sample.valid_gt_right.all())
    assert not sample.dynamic_mask_available.item()
    assert torch.count_nonzero(sample.dynamic_mask_current) == 0


def test_ambiguous_temporal_transform_fails_closed(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path, frame_count=2, write_all_gt=True)
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("gt_extrinsics_camera_from_world")
        payload.pop("gt_pose_convention")
        if payload["frame_id"] == 1:
            payload["T_temporal"] = np.eye(4).tolist()
        records.append(ManifestRecord.from_dict(payload))
    write_manifest(manifest, records)
    dataset = RawStereoVideoClipDataset(
        manifest,
        clip_length=2,
        crop_size_hw=(4, 6),
        crop_mode="fixed",
    )
    with pytest.raises(ValueError, match="T_temporal requires"):
        _ = dataset[0]
