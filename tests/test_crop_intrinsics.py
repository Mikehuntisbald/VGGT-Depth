import json
import pickle

import numpy as np
import pytest

from data.crop import CropWindow, sample_aligned_crop, validate_crop_origin
from data.manifest import (
    ManifestRecord,
    ManifestValidationError,
    load_manifest,
    write_manifest,
)
from geometry.camera import PinholeIntrinsics, resize_intrinsics


def test_crop_then_x2_downsample_updates_intrinsics_without_mutation() -> None:
    intrinsics_hr = np.asarray(
        [[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]]
    )
    original = intrinsics_hr.copy()
    crop = CropWindow(
        x_px=100,
        y_px=40,
        width_px=768,
        height_px=384,
        spatial_scale=2,
    )

    intrinsics_crop_hr = crop.crop_intrinsics(intrinsics_hr)
    intrinsics_crop_lr = resize_intrinsics(intrinsics_crop_hr, 1 / 2)

    np.testing.assert_array_equal(intrinsics_hr, original)
    np.testing.assert_allclose(
        intrinsics_crop_hr,
        [[800.0, 0.0, 540.0], [0.0, 810.0, 320.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        intrinsics_crop_lr,
        [[400.0, 0.0, 270.0], [0.0, 405.0, 160.0], [0.0, 0.0, 1.0]],
    )
    assert crop.lr_size_hw == (192, 384)


def test_anisotropic_resize_scales_each_intrinsics_row() -> None:
    intrinsics = np.asarray(
        [[1000.0, 2.0, 600.0], [0.0, 900.0, 300.0], [0.0, 0.0, 1.0]]
    )

    resized = resize_intrinsics(intrinsics, scale_x=0.5, scale_y=0.25)

    np.testing.assert_allclose(
        resized,
        [[500.0, 1.0, 300.0], [0.0, 225.0, 75.0], [0.0, 0.0, 1.0]],
    )


@pytest.mark.parametrize("origin", [(1, 0), (0, 3), (-2, 0)])
def test_misaligned_or_negative_crop_origin_is_rejected(
    origin: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        validate_crop_origin(origin[0], origin[1], spatial_scale=2)


def test_seeded_sampler_returns_an_aligned_in_bounds_crop() -> None:
    crop = sample_aligned_crop(
        image_height_px=720,
        image_width_px=1280,
        crop_height_px=384,
        crop_width_px=768,
        spatial_scale=2,
        generator=np.random.default_rng(42),
    )

    assert crop.x_px % 2 == 0
    assert crop.y_px % 2 == 0
    assert crop.x_stop_px <= 1280
    assert crop.y_stop_px <= 720


def test_pinhole_crop_checks_source_bounds() -> None:
    intrinsics = PinholeIntrinsics(
        fx_px=800.0,
        fy_px=800.0,
        cx_px=640.0,
        cy_px=360.0,
        width_px=1280,
        height_px=720,
    )

    with pytest.raises(ValueError, match="exceeds"):
        intrinsics.cropped(1000, 400, 768, 384)


def test_manifest_round_trip_preserves_calibration_and_extras(tmp_path) -> None:
    record = ManifestRecord.from_dict(
        {
            "sequence_id": "seq001",
            "frame_id": 12,
            "timestamp": 0.4,
            "left_path": "left/000012.png",
            "right_path": "right/000012.png",
            "K": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
            "baseline_m": 0.12,
            "gt_disparity_path": None,
            "rectified": True,
            "split": "train",
        }
    )
    manifest_path = tmp_path / "train.jsonl"

    write_manifest(manifest_path, [record])
    loaded = load_manifest(manifest_path)

    assert loaded == [record]
    assert loaded[0].intrinsics.fx_px == 800.0
    assert loaded[0].extras["split"] == "train"
    assert pickle.loads(pickle.dumps(loaded[0])) == loaded[0]


def test_manifest_rejects_unrectified_and_empty_inputs(tmp_path) -> None:
    payload = {
        "sequence_id": "seq001",
        "frame_id": 0,
        "timestamp": 0.0,
        "left_path": "left.png",
        "right_path": "right.png",
        "K": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
        "baseline_m": 0.12,
        "gt_disparity_path": None,
        "rectified": False,
    }
    with pytest.raises(ManifestValidationError, match="rectified"):
        ManifestRecord.from_dict(payload)

    manifest_path = tmp_path / "empty.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="empty"):
        load_manifest(manifest_path)


def test_manifest_error_reports_jsonl_line_number(tmp_path) -> None:
    manifest_path = tmp_path / "broken.jsonl"
    manifest_path.write_text(json.dumps({"sequence_id": "missing fields"}) + "\n")

    with pytest.raises(ManifestValidationError, match=r"broken\.jsonl:1"):
        load_manifest(manifest_path)
