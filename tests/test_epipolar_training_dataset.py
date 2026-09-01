from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.cache_dataset import CacheMismatchError
from data.epipolar_training_dataset import (
    EpipolarTrainingDataset,
    _cropped_right_intrinsics,
    collate_epipolar_training_samples,
)
from data.manifest import ManifestRecord
from data.temporal_training_dataset import CachedTemporalTrainingDataset
from test_temporal_training_dataset import _make_temporal_cache


def _dataset(tmp_path: Path) -> EpipolarTrainingDataset:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(tmp_path)
    temporal = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
        fixed_crop_origin_hr_xy=(2, 2),
    )
    return EpipolarTrainingDataset(temporal)


def test_right_rgb_uses_exact_temporal_endpoint_crop(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sample = dataset[0]

    assert sample.temporal.frame_ids == (4, 5, 6)
    assert sample.crop_hr_px == {
        "x": 2,
        "y": 2,
        "width": 8,
        "height": 4,
        "spatial_scale": 2,
    }
    assert sample.rgb_right_hr.shape == (3, 4, 8)
    assert sample.rgb_right_hr.dtype == torch.float32
    with Image.open(sample.right_path) as image:
        expected = np.asarray(image.convert("RGB"), dtype=np.uint8)[2:6, 2:10]
    expected_tensor = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() / 255.0
    torch.testing.assert_close(sample.rgb_right_hr, expected_tensor)
    assert sample.temporal.rgb_hr_sequence[-1].shape == sample.rgb_right_hr.shape
    torch.testing.assert_close(sample.K_right_hr, sample.temporal.K_hr_sequence[-1])
    assert sample.right_intrinsics_source == "manifest.K"


def test_collate_retains_temporal_fields_and_right_provenance(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    batch = collate_epipolar_training_samples([sample, sample])

    assert batch["rgb_hr_sequence"].shape == (2, 3, 3, 4, 8)
    assert batch["rgb_right_hr"].shape == (2, 3, 4, 8)
    assert batch["K_right_hr"].shape == (2, 3, 3)
    torch.testing.assert_close(batch["epipolar_right_row_scale"], torch.ones(2))
    torch.testing.assert_close(
        batch["epipolar_right_row_offset_hr_px"], torch.zeros(2)
    )
    assert batch["right_path"] == [sample.right_path, sample.right_path]
    assert batch["right_sha256"] == [sample.right_sha256, sample.right_sha256]
    assert batch["epipolar_crop_hr_px"] == [
        dict(sample.crop_hr_px),
        dict(sample.crop_hr_px),
    ]


def test_modified_right_source_is_rejected_by_existing_lineage(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = dataset[0]
    right_path = Path(first.right_path)
    with Image.open(right_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[0, 0] = 255 - array[0, 0]
    Image.fromarray(array, mode="RGB").save(right_path)

    with pytest.raises(CacheMismatchError, match="source mismatch|SHA-256"):
        _ = dataset[0]


def test_empty_epipolar_collate_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_epipolar_training_samples([])


def test_right_intrinsics_preserve_vertical_offset_after_shared_crop() -> None:
    record = ManifestRecord(
        sequence_id="seq",
        frame_id=1,
        timestamp=0.0,
        left_path="left.png",
        right_path="right.png",
        K=((800.0, 0.0, 300.0), (0.0, 800.0, 100.0), (0.0, 0.0, 1.0)),
        baseline_m=0.1,
        extras={
            "K_right": [
                [800.0, 0.0, 300.0],
                [0.0, 800.0, 105.4],
                [0.0, 0.0, 1.0],
            ]
        },
    )
    crop = {"x": 20, "y": 40, "width": 80, "height": 60, "spatial_scale": 2}

    right, source = _cropped_right_intrinsics(record, crop)

    assert source == "manifest.K_right"
    assert right[0, 2].item() == pytest.approx(280.0)
    assert right[1, 2].item() == pytest.approx(65.4)
    left_cy_after_crop = 100.0 - 40.0
    assert right[1, 2].item() - left_cy_after_crop == pytest.approx(5.4)
