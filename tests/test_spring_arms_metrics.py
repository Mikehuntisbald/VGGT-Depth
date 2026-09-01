import numpy as np

from metrics.spring_arms import (
    disparity_metrics,
    temporal_residual_metrics,
    topk_diagnostics,
)


def test_spring_disparity_metrics_cover_required_fields() -> None:
    gt = np.array([[2.0, 4.0], [0.0, 3.0]])
    pred = np.array([[2.5, 6.0], [0.0, 1.0]])
    out = disparity_metrics(
        pred,
        gt,
        detail_mask=np.array([[True, False], [False, True]]),
        match_mask=np.array([[True, False], [False, True]]),
        ffs_trusted_mask=np.array([[True, False], [False, False]]),
        ffs_prediction=np.array([[2.25, 0.0], [0.0, 0.0]]),
        boundary_mask=np.array([[False, True], [False, False]]),
    )
    assert out["valid_count"] == 3
    assert "overall_epe" in out
    assert "unmatched_completion_1px" in out
    assert "negative_rate" in out and "invalid_rate" in out
    assert out["ffs_trusted_measurement_error"] == 0.25


def test_unmatched_completion_counts_invalid_predictions_as_failures() -> None:
    gt = np.full((1, 4), 2.0)
    pred = np.array([[2.25, 0.0, -1.0, np.nan]])
    out = disparity_metrics(
        pred,
        gt,
        match_mask=np.zeros_like(gt, dtype=bool),
    )
    # Only the first prediction is within 1/2 px; zero, negative, and NaN
    # predictions remain in the unmatched denominator and count as failures.
    assert out["unmatched_count"] == 4
    assert out["unmatched_completion_1px"] == 0.25
    assert out["unmatched_completion_2px"] == 0.25


def test_temporal_and_topk_diagnostics() -> None:
    out = temporal_residual_metrics(
        np.array([[1.0, 2.0]]),
        np.array([[0.5, 1.0]]),
        valid_mask=np.ones((1, 2), dtype=bool),
        rigid_mask=np.array([[True, False]]),
    )
    assert out["rigid_temporal_residual_error"] == 0.5
    topk = topk_diagnostics(
        age=np.array([[[1.0, 2.0]], [[2.0, 1.0]]]),
        phase=np.zeros((2, 2, 1, 2)),
        depth=np.ones((2, 1, 2)),
        weights=np.ones((2, 1, 2)),
        valid=np.ones((2, 1, 2), dtype=bool),
    )
    assert topk["topk_valid_count"] == 4
    assert 0.0 <= topk["attention_entropy"] <= 1.0


def test_topk_diagnostics_use_valid_pixel_denominators_and_distinct_ages() -> None:
    # Two populated slots from age 1 are not multi-frame complementarity.
    age = np.array([[[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 2.0]]])
    phase = np.zeros((3, 2, 1, 2), dtype=np.float64)
    phase[2, 0, 0, 1] = 0.5
    depth = np.ones((3, 1, 2), dtype=np.float64)
    weights = np.ones_like(depth)
    valid = np.array([[[True, True]], [[True, True]], [[False, True]]])
    out = topk_diagnostics(
        age=age,
        phase=phase,
        depth=depth,
        weights=weights,
        valid=valid,
        age2_available=np.array([[False, True]]),
    )
    assert out["unique_age_fraction"] == 0.5
    assert out["age_2_survival_rate"] == 1.0
    assert out["topk_valid_count"] == 5
