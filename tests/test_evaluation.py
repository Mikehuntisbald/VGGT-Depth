from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from evaluation import (
    AggregateMetric,
    PSEUDO_GT_LABEL,
    aggregate_metric_results,
    comparison_from_aggregates,
    compute_sample_metrics,
    load_model_for_evaluation,
    upsample_ffs_inputs_to_hr,
)
from metrics.disparity import MetricResult
from utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointMismatchError


def test_aggregate_uses_global_numerator_and_count_not_image_means() -> None:
    result = aggregate_metric_results(
        (
            MetricResult(value=1.0, numerator=1.0, count=1, valid=True),
            MetricResult(value=3.0, numerator=9.0, count=3, valid=True),
        )
    )

    assert result.valid
    assert result.numerator == pytest.approx(10.0)
    assert result.count == 4
    assert result.value == pytest.approx(2.5)


def test_aggregate_empty_and_selected_invalid_are_never_zero() -> None:
    empty = aggregate_metric_results((MetricResult.invalid(),))
    assert empty == AggregateMetric(value=None, numerator=None, count=0, valid=False)

    invalid_selected = aggregate_metric_results(
        (
            MetricResult(value=2.0, numerator=4.0, count=2, valid=True),
            MetricResult.invalid(count=3),
        )
    )
    assert not invalid_selected.valid
    assert invalid_selected.count == 5
    assert invalid_selected.value is None
    assert invalid_selected.numerator is None


def test_ffs_upsampling_preserves_hr_units_and_uses_nearest_masks() -> None:
    disparity_hr_px_lr_grid = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    confidence_lr = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    valid_lr = torch.tensor([[[[True, False], [False, True]]]])
    trusted_lr = torch.tensor([[[[False, True], [True, False]]]])

    disparity, confidence, valid, trusted = upsample_ffs_inputs_to_hr(
        disparity_hr_px_lr_grid,
        confidence_lr,
        valid_lr,
        trusted_lr,
        output_size_hw=(4, 4),
    )

    # HR-pixel disparity is interpolated, not multiplied by x2 a second time.
    assert disparity[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert disparity[0, 0, -1, -1].item() == pytest.approx(8.0)
    assert 0.0 < confidence[0, 0, 1, 1].item() < 1.0
    assert valid.dtype == torch.bool and trusted.dtype == torch.bool
    assert valid[0, 0, :2, :2].all()
    assert not valid[0, 0, :2, 2:].any()
    assert trusted[0, 0, :2, 2:].all()
    assert not trusted[0, 0, :2, :2].any()


def test_compute_sample_metrics_uses_trusted_pseudo_gt_domains() -> None:
    target = torch.ones((1, 1, 2, 3)) * 10.0
    prediction = target.clone()
    prediction[..., 0, 0] = 12.0
    target_trusted = torch.tensor([[[[True, True, False], [False, False, False]]]])
    confidence = torch.tensor([[[[0.2, 0.9, 0.1], [0.1, 0.1, 0.1]]]])
    valid_ffs = torch.tensor([[[[True, True, True], [True, True, True]]]])
    trusted_ffs = torch.tensor([[[[True, False, True], [True, True, True]]]])

    metrics = compute_sample_metrics(
        prediction,
        target,
        target_trusted_mask=target_trusted,
        ffs_confidence_hr=confidence,
        ffs_valid_mask_hr=valid_ffs,
        ffs_trusted_mask_hr=trusted_ffs,
        boundary_radius_px=0,
    )

    assert metrics["epe_px"].count == 2
    assert metrics["epe_px"].value == pytest.approx(1.0)
    assert metrics["low_confidence_epe_px"].count == 1
    assert metrics["low_confidence_epe_px"].value == pytest.approx(2.0)
    assert metrics["trusted_region_epe_px"].count == 1
    # No FFS-invalid pixel overlaps trusted pseudo-GT: null domain, not zero.
    assert not metrics["invalid_region_completeness"].valid
    assert metrics["invalid_region_completeness"].count == 0


def test_comparison_is_derived_from_dataset_aggregates() -> None:
    baseline = {
        "trusted_region_epe_px": AggregateMetric(1.0, 10.0, 10, True),
        "low_confidence_epe_px": AggregateMetric(2.0, 20.0, 10, True),
        "invalid_region_completeness": AggregateMetric(0.4, 4.0, 10, True),
    }
    candidate = {
        "trusted_region_epe_px": AggregateMetric(1.02, 10.2, 10, True),
        "low_confidence_epe_px": AggregateMetric(1.6, 16.0, 10, True),
        "invalid_region_completeness": AggregateMetric(0.5, 5.0, 10, True),
    }

    comparison = comparison_from_aggregates(baseline, candidate)

    assert comparison["trusted_region_degradation"]["relative_change_percent"] == pytest.approx(2.0)
    assert comparison["low_confidence_epe_change"][
        "relative_change_percent"
    ] == pytest.approx(-20.0)
    assert comparison["invalid_region_completeness_change"][
        "relative_change_percent"
    ] == pytest.approx(25.0)


def test_evaluation_checkpoint_loader_is_strict_and_checks_parameter_count(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    count = sum(parameter.numel() for parameter in source.parameters())
    checkpoint = tmp_path / "stage_a.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": source.state_dict(),
            "parameter_count": count,
            "step": 7,
            "config": {"stage": "A"},
            "git_hash": "a" * 40,
        },
        checkpoint,
    )

    metadata = load_model_for_evaluation(
        checkpoint, target, expected_parameter_count=count
    )
    assert metadata["step"] == 7
    assert metadata["parameter_count"] == count
    torch.testing.assert_close(source.weight, target.weight)
    torch.testing.assert_close(source.bias, target.bias)
    assert PSEUDO_GT_LABEL in "trusted_hr_ffs_teacher_pseudo_gt"

    with pytest.raises(CheckpointMismatchError, match="parameter count mismatch"):
        load_model_for_evaluation(
            checkpoint, nn.Linear(2, 1), expected_parameter_count=count + 1
        )


def test_empty_metric_result_nan_does_not_leak_into_json_aggregate() -> None:
    raw = MetricResult.invalid()
    assert math.isnan(raw.value)
    aggregate = aggregate_metric_results((raw,))
    assert aggregate.to_dict() == {
        "value": None,
        "numerator": None,
        "count": 0,
        "valid": False,
    }
