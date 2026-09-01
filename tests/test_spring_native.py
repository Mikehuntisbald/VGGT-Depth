from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from metrics.spring_arms import (
    aggregate_spring_rows,
    spring_disparity_row,
    spring_map_bundle,
)


def _record(root: Path, frame_id: int = 2) -> dict[str, object]:
    sequence = root / "spring" / "train" / "0001"
    gt = sequence / "disp1_left" / f"disp1_left_{frame_id:04d}.dsp5"
    return {
        "sequence_id": "0001",
        "frame_id": frame_id,
        "gt_disparity_path": str(gt),
    }


def _write_map(root: Path, name: str, frame_id: int, array: np.ndarray) -> None:
    path = root / "spring" / "train" / "0001" / "maps" / name
    path.mkdir(parents=True)
    Image.fromarray(array).save(path / f"{name}_{frame_id:04d}.png")


def test_spring_map_bundle_applies_official_rigid_downsample_and_crop(tmp_path: Path) -> None:
    frame_id = 2
    maps_root = tmp_path / "spring" / "train" / "0001" / "maps"
    detail = np.zeros((4, 6), dtype=np.uint8)
    detail[1:3, 2:4] = 255
    match = np.zeros((4, 6, 3), dtype=np.uint8)
    match[1:3, 2:4, 0] = 255
    rigid = np.zeros((8, 12), dtype=np.uint8)
    rigid[2:4, 4:8] = 255
    _write_map(tmp_path, "detailmap_disp1_left", frame_id, detail)
    _write_map(tmp_path, "matchmap_disp1_left", frame_id, match)
    _write_map(tmp_path, "rigidmap_BW_left", frame_id, rigid)
    record = _record(tmp_path, frame_id)
    bundle = spring_map_bundle(
        record,
        target_hw=(2, 3),
        crop_hr_xywh=(1, 1, 3, 2),
        require_rigid=True,
    )
    assert bundle["detail"].shape == (2, 3)
    assert bundle["matched"].shape == (2, 3)
    assert bundle["rigid"].shape == (2, 3)
    # The rigid source is 4K; majority reduction then crop is deterministic.
    assert bool(bundle["rigid"].sum())


def test_spring_native_rows_keep_invalid_unmatched_denominator() -> None:
    gt = np.full((1, 3), 2.0)
    pred = np.array([[2.25, 0.0, np.nan]])
    row = spring_disparity_row(
        pred,
        gt,
        detail_mask=np.ones_like(gt, dtype=bool),
        match_mask=np.zeros_like(gt, dtype=bool),
        ffs_trusted_mask=np.ones_like(gt, dtype=bool),
        ffs_prediction=pred,
    )
    assert row["unmatched_count"] == 3
    assert row["unmatched_completion_1px"] == 1 / 3


def test_aggregate_spring_rows_pixel_weights_output_health_rates() -> None:
    rows = [
        {"overall_epe": 1.0, "negative_rate": 0.0, "zero_rate": 0.0, "invalid_rate": 0.0, "image_pixel_count": 1},
        {"overall_epe": 3.0, "negative_rate": 1.0, "zero_rate": 0.0, "invalid_rate": 1.0, "image_pixel_count": 3},
    ]
    aggregate = aggregate_spring_rows(rows)
    assert aggregate["overall_epe"] == 2.0
    assert aggregate["negative_rate"] == 0.75
    assert aggregate["invalid_rate"] == 0.75
