from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from metrics.spring_arms import (
    aggregate_spring_rows,
    disparity_metrics,
    spring_disparity_row,
    spring_map_bundle,
    temporal_residual_metrics,
)
from tools.compose_spring_screening_report import (
    _protocol_training,
    _window_count,
    map_arm,
)
from tools.run_spring_arms import _extract_metrics


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


def test_spring_map_bundle_applies_official_rigid_downsample_and_crop(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("valid_mask", np.ones((2, 1), dtype=bool), "valid_mask shape mismatch"),
        ("detail_mask", np.ones((2, 1), dtype=bool), "detail_mask shape mismatch"),
        ("match_mask", np.ones((2, 1), dtype=bool), "match_mask shape mismatch"),
        (
            "boundary_mask",
            np.ones((2, 1), dtype=bool),
            "boundary_mask shape mismatch",
        ),
    ],
)
def test_spring_disparity_metrics_reject_mismatched_masks(
    keyword: str, value: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        disparity_metrics(
            np.ones((1, 2)),
            np.ones((1, 2)),
            **{keyword: value},
        )


def test_spring_disparity_metrics_reject_mismatched_ffs_inputs() -> None:
    shape = (1, 2)
    with pytest.raises(ValueError, match="ffs_trusted_mask shape mismatch"):
        disparity_metrics(
            np.ones(shape),
            np.ones(shape),
            ffs_trusted_mask=np.ones((2, 1), dtype=bool),
            ffs_prediction=np.ones(shape),
        )
    with pytest.raises(ValueError, match="ffs_prediction shape mismatch"):
        disparity_metrics(
            np.ones(shape),
            np.ones(shape),
            ffs_trusted_mask=np.ones(shape, dtype=bool),
            ffs_prediction=np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    ("reference", "valid_mask", "rigid_mask", "message"),
    [
        (
            np.ones((2, 1)),
            np.ones((1, 2), dtype=bool),
            None,
            "prediction/reference shapes must match",
        ),
        (
            np.ones((1, 2)),
            np.ones((2, 1), dtype=bool),
            None,
            "valid_mask shape must match inputs",
        ),
        (
            np.ones((1, 2)),
            np.ones((1, 2), dtype=bool),
            np.ones((2, 1), dtype=bool),
            "rigid_mask shape must match inputs",
        ),
    ],
)
def test_temporal_residual_metrics_reject_mismatched_shapes(
    reference: np.ndarray,
    valid_mask: np.ndarray,
    rigid_mask: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        temporal_residual_metrics(
            np.ones((1, 2)),
            reference,
            valid_mask=valid_mask,
            rigid_mask=rigid_mask,
        )


def test_spring_metrics_emit_numerators_and_counts() -> None:
    gt = np.full((1, 3), 2.0)
    pred = np.array([[2.25, 0.0, np.nan]])
    out = disparity_metrics(
        pred,
        gt,
        detail_mask=np.ones_like(gt, dtype=bool),
        match_mask=np.zeros_like(gt, dtype=bool),
        ffs_trusted_mask=np.ones_like(gt, dtype=bool),
        ffs_prediction=pred,
        boundary_mask=np.ones_like(gt, dtype=bool),
    )
    names = (
        "overall_epe",
        "overall_1px",
        "high_detail_epe",
        "high_detail_1px",
        "low_detail_epe",
        "low_detail_1px",
        "matched_epe",
        "matched_1px",
        "unmatched_completion_1px",
        "unmatched_completion_2px",
        "boundary_epe",
        "ffs_trusted_measurement_error",
        "negative_rate",
        "zero_rate",
        "invalid_rate",
    )
    for name in names:
        assert f"{name}_numerator" in out
        assert f"{name}_count" in out
    assert out["unmatched_completion_1px_numerator"] == 1
    assert out["unmatched_completion_1px_count"] == 3

    temporal = temporal_residual_metrics(
        np.array([[1.0, 3.0]]),
        np.array([[0.0, 1.0]]),
        valid_mask=np.ones((1, 2), dtype=bool),
        rigid_mask=np.array([[True, False]]),
    )
    assert temporal["rigid_temporal_residual_error_numerator"] == 1
    assert temporal["rigid_temporal_residual_error_count"] == 1
    assert temporal["non_rigid_temporal_residual_error_numerator"] == 2
    assert temporal["non_rigid_temporal_residual_error_count"] == 1


def test_aggregate_spring_rows_pixel_weights_output_health_rates() -> None:
    rows = [
        {
            "overall_epe": 1.0,
            "negative_rate": 0.0,
            "zero_rate": 0.0,
            "invalid_rate": 0.0,
            "image_pixel_count": 1,
        },
        {
            "overall_epe": 3.0,
            "negative_rate": 1.0,
            "zero_rate": 0.0,
            "invalid_rate": 1.0,
            "image_pixel_count": 3,
        },
    ]
    aggregate = aggregate_spring_rows(rows)
    assert aggregate["overall_epe"] == 2.0
    assert aggregate["negative_rate"] == 0.75
    assert aggregate["invalid_rate"] == 0.75


def test_aggregate_spring_rows_uses_global_metric_numerators_and_counts() -> None:
    rows = [
        {
            "overall_epe": 1.0,
            "overall_epe_numerator": 1.0,
            "overall_epe_count": 1,
            "image_pixel_count": 1,
        },
        {
            "overall_epe": 3.0,
            "overall_epe_numerator": 9.0,
            "overall_epe_count": 3,
            "image_pixel_count": 3,
        },
    ]
    aggregate = aggregate_spring_rows(rows)
    assert aggregate["overall_epe"] == 2.5
    assert aggregate["overall_epe_numerator"] == 10.0
    assert aggregate["overall_epe_count"] == 4
    assert aggregate["aggregation_fallback_used"] is False


def test_native_boundary_uses_spring_gt_and_ignores_legacy_override() -> None:
    gt = np.asarray([[1.0, 1.0, 4.0, 4.0]])
    prediction = np.asarray([[1.0, 1.0, 3.0, 4.0]])
    mask = np.ones_like(gt, dtype=bool)
    row = spring_disparity_row(
        prediction,
        gt,
        detail_mask=mask,
        match_mask=mask,
        ffs_trusted_mask=mask,
        ffs_prediction=prediction,
        boundary_epe=999.0,
    )
    assert row["boundary_source"] == "spring_gt_disparity_boundary_mask"
    assert row["boundary_override_ignored"] is True
    assert row["boundary_epe"] != 999.0
    assert row["boundary_epe_count"] > 0


def test_composer_prefers_complete_native_side_channel_and_keeps_buckets() -> None:
    report = {
        "spring_native_metrics": {
            "status": "AVAILABLE",
            "methods": {
                "T3": {
                    "overall_epe": 1.0,
                    "high_detail_epe": 2.0,
                    "matched_epe": 3.0,
                }
            },
            "topk_diagnostics": {
                "age_2_survival_rate": 0.5,
                "gain_by_fractional_phase_bucket": {"phase_lt_0.25": 0.1},
            },
        }
    }
    mapped = map_arm("S3", report)
    assert mapped["overall_epe"] == 1.0
    assert mapped["high_detail_epe"] == 2.0
    assert mapped["matched_epe"] == 3.0
    assert mapped["age_2_survival_rate"] == 0.5
    assert mapped["gain_by_fractional_phase_bucket"] == {"phase_lt_0.25": 0.1}


def test_runner_extracts_native_metrics_before_legacy_aliases() -> None:
    report = {
        "spring_native_metrics": {
            "status": "AVAILABLE",
            "methods": {"T3": {"high_detail_epe": 4.0, "overall_epe": 2.0}},
            "topk_diagnostics": {"age_2_survival_rate": 0.75},
        },
        "methods": {"T3": {"epe_px": {"value": 99.0}}},
    }
    metrics = _extract_metrics(report, preferred_method="T3")
    assert metrics["overall_epe"] == 2.0
    assert metrics["high_detail_epe"] == 4.0
    assert metrics["age_2_survival_rate"] == 0.75


def test_composer_counts_windows_per_sequence() -> None:
    summary = {
        "records": 10,
        "sequence_lengths": {"a": 7, "b": 3},
    }
    # Five-pair VGGT context: (7-4) + (3-4 => 0), not 10-4.
    assert _window_count(summary, 4) == 3
    assert _window_count(summary, 6) == 1


def test_composer_preserves_corrected_per_arm_schedule() -> None:
    reports = {
        "S1": {
            "checkpoint_training_completion": {
                "configured_steps": 5000,
                "actual_step": 5000,
                "canonical_schedule": True,
            }
        },
        "S2": {
            "checkpoint_training_completion": {
                "configured_steps": 15000,
                "actual_step": 1038,
                "canonical_schedule": False,
            }
        },
    }
    protocol = _protocol_training(reports, {"S3": 15000})
    assert protocol["training_steps"] is None
    assert protocol["configured_training_steps_by_arm"] == {
        "S1": 5000,
        "S2": 15000,
        "S3": 15000,
    }
    assert protocol["actual_training_steps_by_arm"] == {"S1": 5000, "S2": 1038}
