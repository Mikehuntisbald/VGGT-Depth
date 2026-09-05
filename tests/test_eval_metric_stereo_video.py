from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metrics.metric_stereo_video import (  # noqa: E402
    MetricAccumulator,
    MetricValue,
    endpoint_metric_values,
)


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
