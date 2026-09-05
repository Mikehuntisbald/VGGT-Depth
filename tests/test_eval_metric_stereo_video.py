from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metrics.metric_stereo_video import (  # noqa: E402
    AccuracyCoverageHistogram,
    MetricAccumulator,
    MetricValue,
    endpoint_metric_values,
    temporal_residual_metric_values,
)
from tools.eval_metric_stereo_video import _selected_variants


def _example() -> dict[str, torch.Tensor]:
    pred_depth = torch.tensor([[[[2.0, 4.0], [2.0, 4.0]]]])
    gt_disp = torch.tensor([[[[1.0, 0.5], [1.0, 0.5]]]])
    K = torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]])
    baseline = torch.tensor([1.0])
    return dict(
        predicted_depth_m=pred_depth,
        predicted_disparity_px=1.0 / pred_depth,
        predicted_valid_mask=torch.ones_like(pred_depth, dtype=torch.bool),
        predicted_valid_probability=torch.full_like(pred_depth, 0.9),
        predicted_uncertainty=torch.ones_like(pred_depth),
        gt_disparity_px=gt_disp,
        gt_valid_mask=torch.ones_like(gt_disp, dtype=torch.bool),
        intrinsics_left=K,
        baseline_m=baseline,
        dynamic_mask=torch.tensor([[[[False, True], [False, True]]]]),
        dynamic_available=torch.tensor([True]),
    )


def test_endpoint_metrics_are_metric_and_empty_safe() -> None:
    values = endpoint_metric_values(**_example())
    assert values["metric_depth_absrel"].value == pytest.approx(0.0)
    assert values["metric_depth_rmse_squared"].value == pytest.approx(0.0)
    assert values["scale_error_log"].value == pytest.approx(0.0)
    assert values["dynamic_epe_px"].count == 2
    assert values["static_epe_px"].count == 2
    assert values["uncertainty_1sigma_coverage"].value == pytest.approx(1.0)

    empty = _example()
    empty["gt_valid_mask"] = torch.zeros_like(empty["gt_valid_mask"])
    empty_values = endpoint_metric_values(**empty)
    assert empty_values["metric_depth_absrel"].count == 0
    assert empty_values["scale_error_log"].count == 0
    assert empty_values["uncertainty_ece"].count == 0


def test_metric_accumulator_merges_global_numerators_and_derives_rmse() -> None:
    accumulator = MetricAccumulator()
    accumulator.update({"metric_depth_rmse_squared": MetricValue(4.0, 2)})
    accumulator.update({"metric_depth_rmse_squared": MetricValue(5.0, 3)})
    result = accumulator.finalize()
    assert result["metric_depth_rmse_squared"]["value"] == pytest.approx(1.8)
    assert result["metric_depth_rmse"]["value"] == pytest.approx(1.8**0.5)


def test_all_gt_metrics_penalize_learned_validity_rejection() -> None:
    example = _example()
    example["predicted_disparity_px"] = torch.tensor(
        [[[[1.0, 99.0], [1.0, 0.5]]]]
    )
    example["predicted_valid_mask"] = torch.tensor(
        [[[[True, False], [True, True]]]]
    )
    detail = torch.tensor([[[[False, True], [False, True]]]])
    matched = torch.tensor([[[[True, False], [True, False]]]])
    boundary = torch.ones_like(detail)
    values = endpoint_metric_values(
        **example,
        detail_mask=detail,
        matched_mask=matched,
        boundary_mask=boundary,
        invalid_penalty_px=10.0,
    )

    assert values["common_valid_epe_px"].value == pytest.approx(0.0)
    assert values["all_gt_penalized_epe_px"].value == pytest.approx(2.5)
    assert values["prediction_coverage"].value == pytest.approx(0.75)
    assert values["completeness_1px"].value == pytest.approx(0.75)
    assert values["spring_high_detail_epe_px"].value == pytest.approx(5.0)
    assert values["spring_unmatched_completion_1px"].value == pytest.approx(0.5)
    assert values["validity_confidence_brier_0p1_log_depth"].count == 4
    assert values["validity_confidence_ece_0p1_log_depth"].count == 4


def test_evaluator_defaults_to_trained_configuration() -> None:
    assert _selected_variants(None) == ["trained_configuration"]
    assert _selected_variants(
        "diagnostic_A1_vggt_gauge_only, diagnostic_A2_vggt_dense_feature_only"
    ) == [
        "diagnostic_A1_vggt_gauge_only",
        "diagnostic_A2_vggt_dense_feature_only",
    ]
    with pytest.raises(ValueError, match="cannot be empty"):
        _selected_variants(" , ")


def test_accuracy_coverage_curve_ranks_all_gt_pixels_by_confidence() -> None:
    histogram = AccuracyCoverageHistogram(bins=101, device=torch.device("cpu"))
    histogram.update(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([0.0, 1.0, 2.0, 9.0]),
        torch.ones(4, dtype=torch.bool),
        error_cap_px=10.0,
    )
    result = histogram.finalize([0.5, 1.0])
    assert result["gt_valid_count"] == 4
    assert result["points"][0]["epe_px"] == pytest.approx(0.5)
    assert result["points"][1]["epe_px"] == pytest.approx(3.0)


def test_temporal_delta_and_disocclusion_have_separate_domains() -> None:
    shape = (1, 1, 1, 3)
    values = temporal_residual_metric_values(
        current_prediction_disparity_px=torch.tensor([[[[4.0, 8.0, 2.0]]]]),
        warped_previous_prediction_disparity_px=torch.tensor([[[[3.0, 7.0, 0.0]]]]),
        current_gt_disparity_px=torch.tensor([[[[5.0, 9.0, 2.0]]]]),
        warped_previous_gt_disparity_px=torch.tensor([[[[4.0, 7.0, 0.0]]]]),
        current_prediction_valid=torch.ones(shape, dtype=torch.bool),
        warped_prediction_valid=torch.tensor([[[[True, True, False]]]]),
        current_gt_valid=torch.ones(shape, dtype=torch.bool),
        warped_gt_valid=torch.tensor([[[[True, True, False]]]]),
        dynamic_mask=torch.tensor([[[[False, True, False]]]]),
        dynamic_available=torch.tensor([True]),
        invalid_penalty_px=10.0,
    )

    assert values["temporal_matched_penalized_delta_epe_px"].value == pytest.approx(0.5)
    assert values["rigid_temporal_residual_epe_px"].value == pytest.approx(0.0)
    assert values["non_rigid_temporal_residual_epe_px"].value == pytest.approx(1.0)
    assert values["temporal_disocclusion_current_epe_px"].value == pytest.approx(0.0)
    assert values["temporal_disocclusion_completion_1px"].value == pytest.approx(1.0)
