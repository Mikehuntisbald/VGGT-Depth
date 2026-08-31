from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backbones.vggt_omega_adapter import (
    ImagePreprocessMetadata,
    VGGTOmegaOutput,
)
from data.cache_dataset import CacheMismatchError
from data.manifest import ManifestRecord
from tools.cache_vggt import (
    CURRENT_LEFT_VIEW_INDEX,
    VIEW_ORDER,
    build_causal_stereo_windows,
    cache_tensors_from_output,
    validate_cached_source,
)


K_LEFT = (
    (100.0, 0.0, 20.0),
    (0.0, 101.0, 10.0),
    (0.0, 0.0, 1.0),
)
K_RIGHT = (
    (102.0, 0.0, 21.0),
    (0.0, 103.0, 11.0),
    (0.0, 0.0, 1.0),
)


def _record(sequence: str, index: int, *, timestamp: float | None = None) -> ManifestRecord:
    return ManifestRecord(
        sequence_id=sequence,
        frame_id=index * 10,
        timestamp=float(index if timestamp is None else timestamp),
        left_path=f"{sequence}/left_{index}.png",
        right_path=f"{sequence}/right_{index}.png",
        K=K_LEFT,
        baseline_m=0.12,
        extras={"K_right": K_RIGHT},
    )


def test_causal_windows_never_cross_sequences_or_use_future_frames(tmp_path: Path) -> None:
    records: list[ManifestRecord] = []
    for index in range(6):
        records.extend((_record("a", index), _record("b", index)))

    windows = build_causal_stereo_windows(records)
    assert len(windows) == 4
    assert [window.target_manifest_index for window in windows] == [8, 9, 10, 11]
    for window in windows:
        assert len({record.sequence_id for record in window.records}) == 1
        assert all(
            record.timestamp <= window.target.timestamp for record in window.records
        )
        assert window.manifest_indices[-1] == window.target_manifest_index
        paths = window.ordered_image_paths(tmp_path / "manifest.jsonl")
        assert len(paths) == 10
        assert paths[0].name == "left_0.png" or paths[0].name == "left_1.png"
        assert paths[-1].name.startswith("right_")

    assert VIEW_ORDER == (
        "L[t-4]",
        "R[t-4]",
        "L[t-3]",
        "R[t-3]",
        "L[t-2]",
        "R[t-2]",
        "L[t-1]",
        "R[t-1]",
        "L[t]",
        "R[t]",
    )


def test_causal_window_rejects_out_of_order_source() -> None:
    records = [_record("a", index) for index in range(5)]
    records[3] = _record("a", 3, timestamp=1.5)
    with pytest.raises(ValueError, match="not strictly timestamp-ordered"):
        build_causal_stereo_windows(records)


def test_intrinsics_alternate_left_and_right_calibration() -> None:
    window = build_causal_stereo_windows(
        [_record("a", index) for index in range(5)]
    )[0]
    calibrated = window.calibrated_intrinsics_ordered()
    assert calibrated.shape == (10, 3, 3)
    torch.testing.assert_close(calibrated[0], torch.tensor(K_LEFT))
    torch.testing.assert_close(calibrated[1], torch.tensor(K_RIGHT))
    torch.testing.assert_close(calibrated[8], torch.tensor(K_LEFT))
    torch.testing.assert_close(calibrated[9], torch.tensor(K_RIGHT))


def _metadata(index: int) -> ImagePreprocessMetadata:
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    return ImagePreprocessMetadata(
        source_path=f"/{index}.png",
        original_size_hw=(2, 3),
        crop_xyxy=(0, 0, 3, 2),
        cropped_size_hw=(2, 3),
        resized_size_hw=(2, 3),
        resize_scale_xy=(1.0, 1.0),
        padding_lrtb=(0, 0, 0, 0),
        model_size_hw=(2, 3),
        original_to_model_3x3=identity,
        model_to_original_3x3=identity,
    )


def _output() -> VGGTOmegaOutput:
    depth = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1, 1)
    depth = depth.expand(1, 10, 2, 3, 1) + 1.0
    confidence = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)
    confidence = confidence.expand(1, 10, 2, 3) + 1.0
    calibrated = torch.tensor(K_LEFT).repeat(1, 10, 1, 1)
    return VGGTOmegaOutput(
        depth=depth,
        depth_conf=confidence,
        pose_enc=torch.zeros(1, 10, 9),
        extrinsics=torch.zeros(1, 10, 3, 4),
        intrinsics_pred=torch.eye(3).repeat(1, 10, 1, 1),
        camera_tokens=torch.ones(1, 10, 1, 8),
        register_tokens=torch.ones(1, 10, 16, 8) * 2,
        preprocessing=tuple(_metadata(index) for index in range(10)),
        intrinsics_calibrated_original=calibrated,
        intrinsics_calibrated_model=calibrated.clone(),
        metadata={},
    )


def test_cache_selects_current_left_and_preserves_geometry_precision() -> None:
    window = build_causal_stereo_windows(
        [_record("a", index) for index in range(5)]
    )[0]
    tensors = cache_tensors_from_output(
        _output(),
        window,
        cache_dtype=torch.float16,
        all_view_dense=False,
    )

    assert CURRENT_LEFT_VIEW_INDEX == 8
    assert tensors["vggt_depth_current_left_arbitrary"].shape == (1, 2, 3)
    assert tensors["vggt_depth_current_left_arbitrary"].dtype == torch.float16
    assert tensors["vggt_depth_current_left_arbitrary"].unique().item() == 9.0
    assert tensors["vggt_depth_conf_current_left_unbounded"].unique().item() == 9.0
    assert tensors["vggt_camera_tokens"].dtype == torch.float16
    assert tensors["vggt_register_tokens"].dtype == torch.float16
    assert tensors["vggt_extrinsics_camera_from_world"].dtype == torch.float32
    assert tensors["calibrated_intrinsics_original_px"].dtype == torch.float32
    assert "vggt_depth_all_views_arbitrary" not in tensors

    all_view = cache_tensors_from_output(
        _output(),
        window,
        cache_dtype=torch.float16,
        all_view_dense=True,
    )
    assert all_view["vggt_depth_all_views_arbitrary"].shape == (10, 1, 2, 3)
    assert all_view["vggt_depth_conf_all_views_unbounded"].shape == (10, 1, 2, 3)

    overflowing = _output()
    overflowing.depth_conf.fill_(1.0e8)
    with pytest.raises(ValueError, match="use --cache-dtype float32"):
        cache_tensors_from_output(
            overflowing,
            window,
            cache_dtype=torch.float16,
            all_view_dense=False,
        )


def test_source_identity_mismatch_is_never_silently_reused() -> None:
    expected = {
        "manifest_sha256": "manifest-a",
        "ordered_images": [{"sha256": "image-a"}],
        "causal": True,
    }
    validate_cached_source(dict(expected), expected)
    with pytest.raises(CacheMismatchError, match="cache source mismatch"):
        validate_cached_source(
            expected | {"ordered_images": [{"sha256": "image-b"}]},
            expected,
        )
