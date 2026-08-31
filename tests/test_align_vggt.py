from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.align_vggt import (
    align_vggt_depth_to_ffs_disparity,
    ffs_trusted_mask,
    robust_scale_only_irls,
)


def test_ffs_trusted_mask_uses_strict_thresholds_and_rejects_nonfinite() -> None:
    disparity_hr_px = np.asarray([10.0, 10.0, 10.0, 0.0, np.nan, 10.0])
    confidence = np.asarray([0.81, 0.80, 0.90, 0.99, 0.99, np.inf])
    lr_error_hr_px = np.asarray([0.99, 0.2, 1.0, 0.1, 0.1, 0.1])

    trusted = ffs_trusted_mask(
        disparity_hr_px,
        confidence,
        lr_error_hr_px,
    )

    assert trusted.tolist() == [True, False, False, False, False, False]


def test_huber_irls_recovers_scale_despite_large_outlier() -> None:
    depth_m = np.linspace(1.0, 8.0, 64)
    inverse_depth_per_m = 1.0 / depth_m
    disparity_hr_px = 48.0 * inverse_depth_per_m
    disparity_hr_px[7] += 500.0

    estimate = robust_scale_only_irls(
        inverse_depth_per_m,
        disparity_hr_px,
        min_samples=32,
        huber_delta_hr_px=0.25,
    )

    assert estimate.valid
    assert estimate.sample_count == 64
    assert estimate.scale == pytest.approx(48.0, abs=0.02)
    assert estimate.iterations >= 1


def test_alignment_emits_prior_for_all_valid_depth_but_fits_trusted_only() -> None:
    depth_m = np.linspace(1.0, 4.0, 48).reshape(6, 8)
    disparity_ffs_hr_px = 72.0 / depth_m
    reliable = np.ones_like(depth_m, dtype=bool)
    reliable[:, -2:] = False
    # Values outside the fit mask cannot change the recovered scale.
    disparity_ffs_hr_px[:, -2:] = 10000.0
    depth_m[0, -1] = np.nan

    result = align_vggt_depth_to_ffs_disparity(
        disparity_ffs_hr_px,
        depth_m,
        reliable_ffs_mask=reliable,
        min_reliable_pixels=32,
    )

    assert result.valid
    assert result.scale_px_m == pytest.approx(72.0, rel=1e-6)
    assert result.reliable_pixel_count == 36
    expected_valid = np.isfinite(depth_m) & (depth_m > 0)
    np.testing.assert_array_equal(result.valid_mask, expected_valid)
    np.testing.assert_allclose(
        result.disparity_vggt_aligned_hr_px[expected_valid],
        72.0 / depth_m[expected_valid],
        rtol=1e-6,
    )
    assert np.isnan(result.disparity_vggt_aligned_hr_px[0, -1])


def test_insufficient_or_nonfinite_reliable_pixels_returns_invalid_nan_prior() -> None:
    depth_m = np.asarray([[2.0, np.nan], [0.0, 4.0]])
    disparity_hr_px = np.asarray([[20.0, 10.0], [5.0, np.inf]])

    result = align_vggt_depth_to_ffs_disparity(
        disparity_hr_px,
        depth_m,
        min_reliable_pixels=2,
    )

    assert not result.valid
    assert result.reliable_pixel_count == 1
    assert result.failure_reason == "insufficient_reliable_pixels"
    assert math.isnan(result.scale_px_m)
    assert np.isnan(result.disparity_vggt_aligned_hr_px).all()
    assert not result.valid_mask.any()


def test_empty_arrays_are_nonthrowing_and_explicitly_invalid() -> None:
    result = align_vggt_depth_to_ffs_disparity(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        min_reliable_pixels=1,
    )

    assert not result.valid
    assert result.reliable_pixel_count == 0
    assert result.disparity_vggt_aligned_hr_px.shape == (0, 4)
    assert result.valid_mask.shape == (0, 4)


def test_zero_weight_pixels_are_excluded_from_fit() -> None:
    inverse_depth_per_m = np.ones(8)
    disparity_hr_px = np.asarray([20.0] * 7 + [1000.0])
    weights = np.asarray([1.0] * 7 + [0.0])

    result = robust_scale_only_irls(
        inverse_depth_per_m,
        disparity_hr_px,
        weights=weights,
        min_samples=7,
    )

    assert result.valid
    assert result.sample_count == 7
    assert result.scale == pytest.approx(20.0)


def test_torch_alignment_preserves_shape_device_dtype_and_is_nan_safe() -> None:
    torch = pytest.importorskip("torch")
    depth_m = torch.linspace(1.0, 4.0, 40, dtype=torch.float32).reshape(5, 8)
    disparity_hr_px = 36.0 / depth_m
    disparity_hr_px[0, 0] = torch.nan

    result = align_vggt_depth_to_ffs_disparity(
        disparity_hr_px,
        depth_m,
        min_reliable_pixels=32,
    )

    assert result.valid
    assert result.disparity_vggt_aligned_hr_px.shape == depth_m.shape
    assert result.disparity_vggt_aligned_hr_px.device == depth_m.device
    assert result.disparity_vggt_aligned_hr_px.dtype == depth_m.dtype
    assert result.scale_px_m == pytest.approx(36.0, rel=1e-5)
    assert torch.isfinite(result.disparity_vggt_aligned_hr_px).all()


@pytest.mark.parametrize("bad_minimum", [0, -1])
def test_nonpositive_minimum_reliable_pixel_count_is_rejected(bad_minimum: int) -> None:
    with pytest.raises(ValueError, match="min_samples"):
        robust_scale_only_irls(
            np.ones(4),
            np.ones(4),
            min_samples=bad_minimum,
        )
