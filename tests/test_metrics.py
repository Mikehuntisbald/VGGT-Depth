from __future__ import annotations

import math

import pytest
import torch

from metrics.boundary import boundary_epe, disparity_boundary_mask
from metrics.disparity import (
    bad_pixel_rate,
    disparity_metrics,
    end_point_error,
    invalid_negative_nan_rate,
    invalid_region_completeness_improvement,
    low_confidence_region_epe,
)
from metrics.pointcloud import disparity_to_point_cloud, point_to_plane_error
from metrics.temporal import temporal_disparity_error, trusted_region_degradation


def test_epe_bad1_bad2_have_exact_aggregation_terms() -> None:
    prediction_hr_px = torch.tensor([10.0, 11.5, 13.0, 999.0, 999.0])
    target_hr_px = torch.tensor([10.0, 10.0, 10.0, 0.0, float("nan")])

    report = disparity_metrics(prediction_hr_px, target_hr_px)

    assert report.epe_px.valid
    assert report.epe_px.count == 3
    assert report.epe_px.numerator == pytest.approx(4.5)
    assert report.epe_px.value == pytest.approx(1.5)
    assert report.bad_1.value == pytest.approx(2.0 / 3.0)
    assert report.bad_1.numerator == 2
    assert report.bad_2.value == pytest.approx(1.0 / 3.0)


def test_epe_empty_mask_and_nonfinite_prediction_are_not_silent_zero() -> None:
    target_hr_px = torch.tensor([4.0, 5.0])
    empty = end_point_error(
        torch.tensor([4.0, 5.0]),
        target_hr_px,
        valid_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert not empty.valid
    assert empty.count == 0
    assert math.isnan(empty.value)

    nonfinite = end_point_error(torch.tensor([4.0, float("nan")]), target_hr_px)
    assert not nonfinite.valid
    assert nonfinite.count == 2
    assert math.isnan(nonfinite.value)
    # Bad-N remains computable and counts that invalid prediction as bad.
    assert bad_pixel_rate(
        torch.tensor([4.0, float("nan")]), target_hr_px, 1.0
    ).value == pytest.approx(0.5)


def test_low_confidence_region_epe_uses_only_finite_below_threshold() -> None:
    prediction_hr_px = torch.tensor([2.0, 5.0, 30.0, 40.0])
    target_hr_px = torch.tensor([1.0, 3.0, 10.0, 10.0])
    confidence = torch.tensor([0.2, 0.79, 0.8, float("nan")])

    result = low_confidence_region_epe(
        prediction_hr_px,
        target_hr_px,
        confidence,
        confidence_threshold=0.8,
    )

    assert result.valid
    assert result.count == 2
    assert result.numerator == pytest.approx(3.0)
    assert result.value == pytest.approx(1.5)


def test_invalid_region_completeness_reports_absolute_and_relative_change() -> None:
    candidate_hr_px = torch.tensor([1.0, float("nan"), -1.0, 2.0])
    baseline_hr_px = torch.tensor([float("nan"), 1.0, 0.0, float("nan")])
    holes = torch.ones(4, dtype=torch.bool)

    report = invalid_region_completeness_improvement(
        candidate_hr_px,
        baseline_hr_px,
        holes,
    )

    assert report.valid and report.relative_valid
    assert report.candidate.value == pytest.approx(0.5)
    assert report.baseline.value == pytest.approx(0.25)
    assert report.absolute_improvement == pytest.approx(0.25)
    assert report.absolute_improvement_percentage_points == pytest.approx(25.0)
    assert report.relative_improvement_percent == pytest.approx(100.0)


def test_invalid_region_completeness_handles_empty_and_zero_baseline() -> None:
    candidate_hr_px = torch.tensor([1.0])
    baseline_hr_px = torch.tensor([float("nan")])
    empty = invalid_region_completeness_improvement(
        candidate_hr_px,
        baseline_hr_px,
        torch.tensor([False]),
    )
    assert not empty.valid and not empty.relative_valid
    assert math.isnan(empty.absolute_improvement)

    zero_baseline = invalid_region_completeness_improvement(
        candidate_hr_px,
        baseline_hr_px,
        torch.tensor([True]),
    )
    assert zero_baseline.valid
    assert not zero_baseline.relative_valid
    assert zero_baseline.absolute_improvement == pytest.approx(1.0)
    assert math.isnan(zero_baseline.relative_improvement_percent)


def test_invalid_negative_nan_rates_have_explicit_denominator() -> None:
    disparity_hr_px = torch.tensor([1.0, 0.0, -1.0, float("nan"), float("inf")])

    report = invalid_negative_nan_rate(disparity_hr_px)

    assert report.total_count == 5
    assert report.invalid.numerator == 4
    assert report.invalid_rate == pytest.approx(0.8)
    assert report.negative_rate == pytest.approx(0.2)
    assert report.nan_rate == pytest.approx(0.2)
    assert report.infinite.value == pytest.approx(0.2)
    assert report.zero.value == pytest.approx(0.2)

    empty = invalid_negative_nan_rate(
        disparity_hr_px,
        evaluation_mask=torch.zeros(5, dtype=torch.bool),
    )
    assert not empty.invalid.valid
    assert math.isnan(empty.invalid_rate)


def test_boundary_mask_marks_both_sides_and_configurable_radius() -> None:
    target_hr_px = torch.tensor([[[[1.0, 1.0, 5.0, 5.0]]]])

    exact = disparity_boundary_mask(
        target_hr_px,
        gradient_threshold_px=2.0,
        radius_px=0,
    )
    assert torch.equal(
        exact,
        torch.tensor([[[[False, True, True, False]]]]),
    )
    expanded = disparity_boundary_mask(
        target_hr_px,
        gradient_threshold_px=2.0,
        radius_px=1,
    )
    assert expanded.all()

    prediction_hr_px = target_hr_px.clone()
    prediction_hr_px[..., 1] += 2.0
    prediction_hr_px[..., 2] += 4.0
    result = boundary_epe(
        prediction_hr_px,
        target_hr_px,
        gradient_threshold_px=2.0,
        radius_px=0,
    )
    assert result.count == 2
    assert result.value == pytest.approx(3.0)


def test_boundary_ignores_gradient_pairs_touching_invalid_gt() -> None:
    target_hr_px = torch.tensor([[1.0, float("nan"), 10.0]])
    mask = disparity_boundary_mask(
        target_hr_px,
        gradient_threshold_px=0.5,
        radius_px=0,
    )
    assert not mask.any()
    result = boundary_epe(
        target_hr_px.clone(),
        target_hr_px,
        gradient_threshold_px=0.5,
        radius_px=0,
    )
    assert not result.valid and result.count == 0


def test_temporal_error_uses_supplied_safe_mask_and_valid_history() -> None:
    current_hr_px = torch.tensor([5.0, 8.0, 50.0, 7.0])
    warped_history_hr_px = torch.tensor([4.0, 10.0, 1.0, float("nan")])
    safe = torch.tensor([True, True, False, True])

    result = temporal_disparity_error(
        current_hr_px,
        warped_history_hr_px,
        safe_mask=safe,
    )

    assert result.count == 2
    assert result.numerator == pytest.approx(3.0)
    assert result.value == pytest.approx(1.5)


def test_trusted_region_degradation_has_relative_percent_and_empty_handling() -> None:
    target_hr_px = torch.tensor([10.0, 10.0, 10.0])
    baseline_hr_px = torch.tensor([9.0, 11.0, 999.0])
    candidate_hr_px = torch.tensor([8.0, 12.0, 999.0])
    trusted = torch.tensor([True, True, False])

    report = trusted_region_degradation(
        candidate_hr_px,
        baseline_hr_px,
        target_hr_px,
        trusted_mask=trusted,
    )
    assert report.valid and report.relative_valid
    assert report.baseline_epe_px.value == pytest.approx(1.0)
    assert report.candidate_epe_px.value == pytest.approx(2.0)
    assert report.absolute_change_px == pytest.approx(1.0)
    assert report.relative_change_percent == pytest.approx(100.0)

    empty = trusted_region_degradation(
        candidate_hr_px,
        baseline_hr_px,
        target_hr_px,
        trusted_mask=torch.zeros(3, dtype=torch.bool),
    )
    assert not empty.valid and not empty.relative_valid


def test_disparity_to_point_cloud_uses_calibrated_pixel_geometry() -> None:
    disparity_hr_px = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    K_hr_px = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]
    )

    result = disparity_to_point_cloud(disparity_hr_px, K_hr_px, baseline_m=0.5)

    assert result.points_camera_m.shape == (1, 2, 2, 3)
    assert result.valid_mask.shape == (1, 2, 2)
    assert torch.allclose(
        result.points_camera_m[0, 0, 0],
        torch.tensor([-0.5, -0.25, 1.0]),
    )
    assert torch.allclose(
        result.points_camera_m[0, 0, 1],
        torch.tensor([0.0, -0.25, 1.0]),
    )
    assert not result.valid_mask[0, 1, 1]
    assert torch.isnan(result.points_camera_m[0, 1, 1]).all()


def test_disparity_to_point_cloud_supports_batched_calibration() -> None:
    disparity_hr_px = torch.tensor([[[[2.0]]], [[[4.0]]]])
    K_hr_px = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    result = disparity_to_point_cloud(
        disparity_hr_px,
        K_hr_px,
        baseline_m=torch.tensor([0.5, 0.25]),
    )
    assert torch.allclose(result.points_camera_m[:, 0, 0, 2], torch.tensor([1.0, 0.5]))


def test_point_to_plane_uses_only_explicit_matched_correspondences() -> None:
    source_m = torch.tensor([[10.0, 0.0, 1.0], [7.0, 3.0, 2.0]])
    target_m = torch.zeros_like(source_m)
    # Non-unit normals prove that normalization is applied.
    normals = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])

    result = point_to_plane_error(source_m, target_m, normals)

    assert result.valid
    assert result.count == 2
    assert result.numerator == pytest.approx(3.0)
    assert result.value == pytest.approx(1.5)
    masked = point_to_plane_error(
        source_m,
        target_m,
        normals,
        correspondence_mask=torch.tensor([True, False]),
    )
    assert masked.count == 1
    assert masked.value == pytest.approx(1.0)


def test_point_to_plane_empty_valid_correspondence_is_nan() -> None:
    points = torch.zeros((2, 3))
    result = point_to_plane_error(points, points, torch.zeros_like(points))
    assert not result.valid
    assert result.count == 0
    assert math.isnan(result.value)
