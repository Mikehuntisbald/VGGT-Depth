from __future__ import annotations

import json

import pytest

from tools.audit_epipolar_rectification import (
    EpipolarAuditError,
    FrameMeasurement,
    RectificationThresholds,
    aggregate_measurements,
    deterministic_balanced_positions,
    evaluate_rectification_contract,
    summarize_displacements,
)


def _measurement(
    *,
    sequence: str,
    index: int,
    dy: tuple[float, ...],
    covered: bool = True,
    metadata_delta: float = 5.4,
) -> FrameMeasurement:
    inliers = len(dy)
    return FrameMeasurement(
        split="train",
        manifest_index=index,
        sequence_id=sequence,
        frame_id=100 + index,
        timestamp=float(index),
        metadata_right_minus_left_cy_px=metadata_delta,
        left_path=f"/left/{index}.png",
        right_path=f"/right/{index}.png",
        left_sha256="a" * 64,
        right_sha256="b" * 64,
        metadata_path=f"/meta/{index}.yaml",
        metadata_sha256_verified="c" * 64,
        image_size_wh=(1280, 800),
        left_keypoints=100,
        right_keypoints=100,
        knn_pairs=100,
        ratio_matches=80,
        positive_horizontal_matches=75,
        plausible_prefilter_matches=70,
        ransac_inliers=inliers,
        covered=covered,
        failure_reasons=() if covered else ("insufficient_ransac_inliers",),
        dy_right_minus_left_inliers_px=dy,
        horizontal_disparity_left_minus_right_inliers_px=tuple(
            10.0 + value for value in dy
        ),
        fundamental_matrix=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
    )


def test_balanced_sampling_is_deterministic_unique_and_endpoint_inclusive() -> None:
    assert deterministic_balanced_positions(10, 4) == (0, 3, 6, 9)
    assert deterministic_balanced_positions(3, 8) == (0, 1, 2)
    assert deterministic_balanced_positions(9, 1) == (4,)
    first = deterministic_balanced_positions(1166, 32)
    second = deterministic_balanced_positions(1166, 32)
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert first[0] == 0 and first[-1] == 1165


def test_displacement_summary_is_strict_and_uses_signed_and_absolute_values() -> None:
    summary = summarize_displacements((-2.0, -0.5, 0.0, 0.5, 2.0))
    assert summary is not None
    assert summary["signed"]["p50"] == 0.0
    assert summary["absolute"]["p50"] == 0.5
    assert summary["absolute"]["p95"] == pytest.approx(2.0)
    assert summarize_displacements(()) is None
    with pytest.raises(EpipolarAuditError, match="non-finite"):
        summarize_displacements((0.0, float("nan")))


def test_contract_passes_only_when_every_sequence_and_global_scope_pass() -> None:
    seq_a = [
        _measurement(sequence="a", index=0, dy=(-0.5, 0.0, 0.5)),
        _measurement(sequence="a", index=1, dy=(-0.4, 0.1, 0.6)),
    ]
    seq_b = [
        _measurement(sequence="b", index=2, dy=(-0.8, 0.0, 0.8)),
        _measurement(sequence="b", index=3, dy=(-0.7, 0.1, 0.9)),
    ]
    aggregates = {
        "train:a": aggregate_measurements(seq_a),
        "validation:b": aggregate_measurements(seq_b),
    }
    global_aggregate = aggregate_measurements([*seq_a, *seq_b])

    passed, checks = evaluate_rectification_contract(
        sequence_aggregates=aggregates,
        global_aggregate=global_aggregate,
        thresholds=RectificationThresholds(),
    )

    assert passed is True
    assert len(checks) == 9
    assert all(check["passed"] for check in checks)
    frame_report = seq_a[0].to_report()
    assert frame_report["observed_minus_metadata_cy_delta_px"] == pytest.approx(-5.4)
    json.dumps({"aggregates": aggregates, "checks": checks}, allow_nan=False)


def test_contract_fails_closed_for_coverage_or_vertical_tail() -> None:
    good = _measurement(sequence="a", index=0, dy=(-0.1, 0.0, 0.1))
    uncovered = _measurement(sequence="a", index=1, dy=(), covered=False)
    high_tail = _measurement(sequence="b", index=2, dy=(0.0, 0.1, 4.0))
    aggregates = {
        "train:a": aggregate_measurements([good, uncovered]),
        "validation:b": aggregate_measurements([high_tail]),
    }

    passed, checks = evaluate_rectification_contract(
        sequence_aggregates=aggregates,
        global_aggregate=aggregate_measurements([good, uncovered, high_tail]),
        thresholds=RectificationThresholds(),
    )

    assert passed is False
    failed = {(check["scope"], check["metric"]) for check in checks if not check["passed"]}
    assert ("sequence:train:a", "frame_coverage_fraction") in failed
    assert ("sequence:validation:b", "p95_abs_dy_right_minus_left_px") in failed


def test_threshold_configuration_rejects_nonfinite_or_inverted_counts() -> None:
    with pytest.raises(EpipolarAuditError, match="cannot exceed"):
        RectificationThresholds(
            min_ratio_matches_per_frame=32,
            min_plausible_matches_per_frame=48,
        ).validate()
    with pytest.raises(EpipolarAuditError, match="finite"):
        RectificationThresholds(max_p95_abs_dy_px=float("inf")).validate()
