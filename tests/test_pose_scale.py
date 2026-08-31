from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.pose_scale import (
    assess_pose_quality,
    camera_centers_from_extrinsics,
    estimate_baseline_metric_scale,
    metric_scale_vggt_geometry,
    relative_stereo_rotations,
    scale_vggt_translations_and_depth,
    stereo_baselines_from_extrinsics,
)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _camera_from_world(rotation: np.ndarray, center_world: np.ndarray) -> np.ndarray:
    translation = -rotation @ center_world
    return np.concatenate((rotation, translation[:, None]), axis=1)


def _stereo_window(
    predicted_baselines: list[float],
    *,
    right_rotation_degrees: float = 0.0,
) -> np.ndarray:
    poses: list[np.ndarray] = []
    for frame_index, baseline in enumerate(predicted_baselines):
        left_center = np.asarray([0.03 * frame_index, 0.01 * frame_index, 1.0])
        right_center = left_center + np.asarray([baseline, 0.0, 0.0])
        poses.extend(
            (
                _camera_from_world(np.eye(3), left_center),
                _camera_from_world(_rotation_z(right_rotation_degrees), right_center),
            )
        )
    return np.stack(poses)


def test_camera_center_uses_camera_from_world_convention() -> None:
    rotation = _rotation_z(37.0)
    expected_center = np.asarray([1.25, -0.4, 3.0])
    extrinsic = _camera_from_world(rotation, expected_center)

    recovered_center = camera_centers_from_extrinsics(extrinsic)

    np.testing.assert_allclose(recovered_center, expected_center, atol=1e-12)


def test_five_pair_median_baseline_recovers_and_applies_one_metric_scale() -> None:
    # Median predicted baseline is 0.20 VGGT units; real baseline is 0.12 m.
    extrinsics = _stereo_window([0.19, 0.20, 0.20, 0.20, 0.21])
    original_extrinsics = extrinsics.copy()
    depth_vggt_unit = np.asarray([1.0, 2.0, np.nan, 4.0])

    estimate = estimate_baseline_metric_scale(
        extrinsics,
        calibrated_baseline_m=0.12,
        max_baseline_cv=0.10,
    )

    assert estimate.valid
    assert estimate.alpha_m_per_vggt_unit == pytest.approx(0.6)
    assert estimate.median_predicted_baseline_vggt_unit == pytest.approx(0.2)
    assert estimate.quality.baseline_cv == pytest.approx(
        np.std([0.19, 0.20, 0.20, 0.20, 0.21]) / 0.2
    )

    scaled_extrinsics, depth_m = scale_vggt_translations_and_depth(
        extrinsics,
        depth_vggt_unit,
        estimate.alpha,
    )
    np.testing.assert_array_equal(extrinsics, original_extrinsics)
    np.testing.assert_allclose(
        scaled_extrinsics[..., :3, :3], extrinsics[..., :3, :3]
    )
    np.testing.assert_allclose(
        scaled_extrinsics[..., :3, 3], extrinsics[..., :3, 3] * 0.6
    )
    np.testing.assert_allclose(depth_m[:2], [0.6, 1.2])
    assert np.isnan(depth_m[2])
    assert depth_m[3] == pytest.approx(2.4)
    np.testing.assert_allclose(
        stereo_baselines_from_extrinsics(scaled_extrinsics),
        np.asarray([0.19, 0.20, 0.20, 0.20, 0.21]) * 0.6,
    )


def test_relative_stereo_rotation_is_right_from_left_and_quality_is_gated() -> None:
    extrinsics = _stereo_window([0.2] * 5, right_rotation_degrees=2.0)

    relative = relative_stereo_rotations(extrinsics)
    for pair_rotation in relative:
        np.testing.assert_allclose(pair_rotation, _rotation_z(2.0), atol=1e-12)

    accepted = assess_pose_quality(
        extrinsics,
        max_stereo_rotation_error_deg=2.1,
    )
    rejected = assess_pose_quality(
        extrinsics,
        max_stereo_rotation_error_deg=1.9,
    )

    assert accepted.valid
    assert accepted.stereo_rotation_error_deg == pytest.approx(2.0, abs=1e-10)
    assert not rejected.valid
    assert "stereo_rotation_error_exceeds_threshold" in rejected.failure_reasons


def test_bad_baseline_cv_and_nan_windows_return_no_scaled_geometry() -> None:
    inconsistent = _stereo_window([0.05, 0.2, 0.2, 0.2, 0.35])
    result = metric_scale_vggt_geometry(
        inconsistent,
        np.ones((2, 3)),
        calibrated_baseline_m=0.12,
        max_baseline_cv=0.1,
    )

    assert not result.valid
    assert result.extrinsics_camera_from_world_metric is None
    assert result.depth_m is None
    assert "baseline_cv_exceeds_threshold" in result.scale.quality.failure_reasons

    non_finite = _stereo_window([0.2] * 5)
    non_finite[3, 0, 3] = np.nan
    estimate = estimate_baseline_metric_scale(non_finite, 0.12)
    assert not estimate.valid
    assert math.isnan(estimate.alpha)
    assert "non_finite_baseline" in estimate.quality.failure_reasons


def test_torch_pose_scaling_preserves_device_dtype_and_homogeneous_row() -> None:
    torch = pytest.importorskip("torch")
    extrinsics_3x4 = torch.as_tensor(_stereo_window([0.2] * 5), dtype=torch.float32)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(10, 1, 4)
    extrinsics_4x4 = torch.cat((extrinsics_3x4, bottom), dim=1)
    depth = torch.tensor([1.0, 2.0], dtype=torch.float16)

    estimate = estimate_baseline_metric_scale(extrinsics_4x4, 0.12)
    scaled_extrinsics, scaled_depth = scale_vggt_translations_and_depth(
        extrinsics_4x4, depth, estimate.alpha
    )

    assert estimate.valid
    assert scaled_extrinsics.dtype == extrinsics_4x4.dtype
    assert scaled_depth.dtype == depth.dtype
    assert scaled_extrinsics.device == extrinsics_4x4.device
    torch.testing.assert_close(scaled_extrinsics[:, 3], bottom[:, 0])
    torch.testing.assert_close(scaled_depth, torch.tensor([0.6, 1.2], dtype=torch.float16))


@pytest.mark.parametrize("camera_count", [0, 1, 9])
def test_invalid_interleaved_camera_counts_are_rejected(camera_count: int) -> None:
    poses = np.empty((camera_count, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="camera count|expected 5"):
        stereo_baselines_from_extrinsics(poses)

