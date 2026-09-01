import math

import pytest

torch = pytest.importorskip("torch")

from geometry.topk_splat import (
    TopKSplatResult,
    merge_topk_splat_results,
    topk_z_aware_splat,
)
from geometry.zbuffer_reproject import zbuffer_reproject


def _intrinsics(fx_px: float = 2.0) -> torch.Tensor:
    return torch.tensor(
        [[fx_px, 0.0, 0.0], [0.0, fx_px, 0.0], [0.0, 0.0, 1.0]]
    )


def _camera_from_world_at_x(camera_center_world_x_m: float) -> torch.Tensor:
    extrinsics = torch.eye(4)
    extrinsics[0, 3] = -camera_center_world_x_m
    return extrinsics


def _finite_fields(result) -> list[torch.Tensor]:
    fields = [
        result.disparity_hr_px,
        result.depth_m,
        result.confidence,
        result.temporal_age_frames,
        result.footprint_weight,
        result.projected_uv_grid_px,
        result.fractional_offset_grid_px,
        result.source_uv_grid_px,
        result.z_aware_weights,
        result.weighted_disparity_hr_px,
        result.weighted_depth_m,
        result.weighted_confidence,
        result.weighted_fractional_offset_grid_px,
        result.weighted_temporal_age_frames,
    ]
    if result.warped_hidden_feature is not None:
        fields.append(result.warped_hidden_feature)
    if result.weighted_hidden_feature is not None:
        fields.append(result.weighted_hidden_feature)
    return fields


def test_identity_bilinear_splat_preserves_all_pixels_and_hidden_feature() -> None:
    disparity = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    depth = 12.0 / disparity
    confidence = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]])
    hidden = torch.arange(24.0).reshape(1, 4, 2, 3).requires_grad_()

    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        torch.eye(4),
        torch.eye(4),
        top_k=3,
        temporal_age_frames=1,
        previous_hidden_feature=hidden,
    )

    torch.testing.assert_close(result.disparity_hr_px[:, :1], disparity)
    torch.testing.assert_close(result.depth_m[:, :1], depth)
    torch.testing.assert_close(result.confidence[:, :1], confidence)
    assert bool(result.valid_mask[:, :1].all())
    assert not bool(result.valid_mask[:, 1:].any())
    torch.testing.assert_close(
        result.footprint_weight[:, :1], torch.ones_like(disparity)
    )
    torch.testing.assert_close(result.z_aware_weights[:, :1], torch.ones_like(disparity))
    torch.testing.assert_close(result.weighted_hidden_feature, hidden)
    result.weighted_hidden_feature.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


def test_fractional_projection_uses_four_neighbor_bilinear_footprint() -> None:
    disparity = torch.zeros((1, 1, 1, 4))
    depth = torch.zeros_like(disparity)
    confidence = torch.zeros_like(disparity)
    hidden = torch.zeros((1, 1, 1, 4))
    disparity[0, 0, 0, 1] = 2.0
    depth[0, 0, 0, 1] = 1.0
    confidence[0, 0, 0, 1] = 1.0
    hidden[0, 0, 0, 1] = 7.0

    # Moving the current camera centre -0.125m shifts u by +0.25 at fx=2,Z=1.
    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(-0.125),
        top_k=1,
        previous_hidden_feature=hidden,
    )

    assert result.valid_mask[0, 0, 0].tolist() == [False, True, True, False]
    assert result.footprint_weight[0, 0, 0, 1].item() == pytest.approx(0.75)
    assert result.footprint_weight[0, 0, 0, 2].item() == pytest.approx(0.25)
    assert result.fractional_offset_grid_px[0, 0, 0, 0, 1].item() == pytest.approx(
        0.25
    )
    assert result.fractional_offset_grid_px[0, 0, 0, 0, 2].item() == pytest.approx(
        -0.75
    )
    assert result.warped_hidden_feature[0, 0, 0, 0, 1].item() == 7.0
    assert result.warped_hidden_feature[0, 0, 0, 0, 2].item() == 7.0
    assert result.weighted_hidden_feature[0, 0, 0, 1].item() == 7.0
    assert result.weighted_hidden_feature[0, 0, 0, 2].item() == 7.0


def test_collision_keeps_topk_in_depth_order_and_marks_all_retained() -> None:
    # With camera motion +1m and fx=2, (u,Z)=(1,2),(2,1),(3,2/3)
    # all project exactly to target u=0.
    disparity = torch.zeros((1, 1, 1, 4))
    depth = torch.zeros_like(disparity)
    confidence = torch.zeros_like(disparity)
    disparity[0, 0, 0, 1:] = torch.tensor([1.0, 2.0, 3.0])
    depth[0, 0, 0, 1:] = torch.tensor([2.0, 1.0, 2.0 / 3.0])
    confidence[0, 0, 0, 1:] = torch.tensor([0.2, 0.5, 0.9])

    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(1.0),
        top_k=2,
    )

    assert result.candidate_count[0, 0, 0, 0].item() == 3
    torch.testing.assert_close(
        result.depth_m[0, :, 0, 0], torch.tensor([2.0 / 3.0, 1.0])
    )
    assert result.source_linear_index[0, :, 0, 0].tolist() == [3, 2]
    assert result.visibility_mask[0, :, 0, 0].tolist() == [True, False]
    assert result.collision_mask[0, :, 0, 0].tolist() == [True, True]


def test_exact_depth_tie_uses_lower_source_index_stably() -> None:
    disparity = torch.zeros((1, 1, 1, 4))
    depth = torch.zeros_like(disparity)
    confidence = torch.zeros_like(disparity)
    disparity[0, 0, 0, 1:3] = 2.0
    depth[0, 0, 0, 1:3] = 1.0
    confidence[0, 0, 0, 1:3] = 1.0

    # Both sources contribute footprint 0.5 to target 1 at equal Z.
    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(0.25),
        top_k=2,
    )

    assert result.source_linear_index[0, :, 0, 1].tolist() == [1, 2]
    torch.testing.assert_close(
        result.footprint_weight[0, :, 0, 1], torch.tensor([0.5, 0.5])
    )


def test_merge_weights_explicitly_include_bilinear_footprint() -> None:
    disparity = torch.zeros((1, 1, 1, 4))
    depth = torch.zeros_like(disparity)
    confidence = torch.zeros_like(disparity)
    disparity[0, 0, 0, 1] = 2.0
    depth[0, 0, 0, 1] = 1.0
    confidence[0, 0, 0, 1] = 1.0
    previous = _camera_from_world_at_x(0.0)
    high_footprint = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        previous,
        _camera_from_world_at_x(-0.125),  # projected u=1.25 -> 0.75 at target 1
        top_k=2,
        temporal_age_frames=1,
    )
    low_footprint = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        previous,
        _camera_from_world_at_x(-0.375),  # projected u=1.75 -> 0.25 at target 1
        top_k=2,
        temporal_age_frames=1,
    )

    merged = merge_topk_splat_results(
        [high_footprint, low_footprint], top_k=2
    )
    torch.testing.assert_close(
        merged.footprint_weight[0, :, 0, 1], torch.tensor([0.75, 0.25])
    )
    torch.testing.assert_close(
        merged.z_aware_weights[0, :, 0, 1], torch.tensor([0.75, 0.25])
    )
    assert merged.weighted_fractional_offset_grid_px[0, 0, 0, 1].item() == pytest.approx(
        0.375
    )


def test_multi_age_merge_reorders_depth_and_preserves_metadata_and_gradients() -> None:
    disparity = torch.tensor([[[[2.0]]]])
    confidence = torch.tensor([[[[1.0]]]])
    hidden_age1 = torch.tensor([[[[2.0]], [[4.0]]]], requires_grad=True)
    hidden_age2 = torch.tensor([[[[6.0]], [[8.0]]]], requires_grad=True)
    common = (_intrinsics(), torch.eye(4), torch.eye(4))
    age1 = topk_z_aware_splat(
        disparity,
        torch.tensor([[[[2.0]]]]),
        confidence,
        *common,
        top_k=2,
        temporal_age_frames=1,
        previous_hidden_feature=hidden_age1,
    )
    age2 = topk_z_aware_splat(
        disparity,
        torch.tensor([[[[1.0]]]]),
        confidence,
        *common,
        top_k=2,
        temporal_age_frames=2,
        previous_hidden_feature=hidden_age2,
    )
    merged = merge_topk_splat_results([age1, age2], top_k=2)

    # Depth dominates age: the age-2 point is geometrically first.
    torch.testing.assert_close(merged.depth_m[0, :, 0, 0], torch.tensor([1.0, 2.0]))
    assert merged.temporal_age_frames[0, :, 0, 0].tolist() == [2.0, 1.0]
    assert merged.source_sequence_index[0, :, 0, 0].tolist() == [1, 0]
    expected_unnormalised = torch.tensor(
        [math.exp(-2.0 / 3.0), math.exp(-1.0 / 0.25) * math.exp(-1.0 / 3.0)]
    )
    expected_weights = expected_unnormalised / expected_unnormalised.sum()
    torch.testing.assert_close(
        merged.z_aware_weights[0, :, 0, 0], expected_weights, rtol=1e-5, atol=1e-6
    )
    expected_hidden = (
        expected_weights[0] * hidden_age2[0, :, 0, 0]
        + expected_weights[1] * hidden_age1[0, :, 0, 0]
    )
    torch.testing.assert_close(merged.weighted_hidden_feature[0, :, 0, 0], expected_hidden)
    assert merged.weighted_temporal_age_frames[0, 0, 0, 0].item() == pytest.approx(
        float((expected_weights * torch.tensor([2.0, 1.0])).sum()), rel=1e-5
    )
    merged.weighted_hidden_feature.sum().backward()
    assert hidden_age1.grad is not None and bool(torch.isfinite(hidden_age1.grad).all())
    assert hidden_age2.grad is not None and bool(torch.isfinite(hidden_age2.grad).all())


def test_invalid_oov_nonfinite_feature_and_zero_weight_fail_closed() -> None:
    disparity = torch.tensor([[[[1.0, 1.0, 1.0, float("nan")]]]])
    depth = torch.ones_like(disparity)
    confidence = torch.tensor([[[[1.0, 0.0, 1.0, 1.0]]]])
    hidden = torch.ones((1, 2, 1, 4))
    hidden[0, 0, 0, 2] = float("nan")
    valid = torch.tensor([[[[True, True, True, False]]]])

    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        _intrinsics(),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(10.0),
        top_k=2,
        previous_hidden_feature=hidden,
        source_valid_mask=valid,
    )
    assert not bool(result.valid_mask.any())
    assert not bool(result.aggregate_valid_mask.any())
    assert not bool(result.z_aware_weights.any())
    for value in _finite_fields(result):
        assert bool(torch.isfinite(value).all())

    zero_weight = topk_z_aware_splat(
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[0.0]]]]),
        _intrinsics(),
        torch.eye(4),
        torch.eye(4),
        top_k=1,
    )
    assert bool(zero_weight.valid_mask.any())
    assert not bool(zero_weight.aggregate_valid_mask.any())
    assert zero_weight.z_aware_weights.eq(0).all()


def test_propagated_source_visibility_and_collision_are_not_discarded() -> None:
    scalar = torch.ones((1, 1, 1, 1))
    result = topk_z_aware_splat(
        scalar,
        scalar,
        scalar,
        _intrinsics(),
        torch.eye(4),
        torch.eye(4),
        top_k=1,
        source_visibility_mask=torch.zeros_like(scalar, dtype=torch.bool),
        source_collision_mask=torch.ones_like(scalar, dtype=torch.bool),
    )

    assert bool(result.valid_mask.all())
    assert not bool(result.source_visibility_mask.any())
    assert bool(result.source_collision_mask.all())
    assert not bool(result.aggregate_valid_mask.any())
    assert result.z_aware_weights.eq(0).all()


def test_k1_nearest_mode_matches_canonical_single_winner_numerically() -> None:
    disparity = torch.tensor([[[[0.0, 1.0, 2.0, 0.0, 3.0]]]], dtype=torch.float64)
    depth = torch.tensor([[[[0.0, 2.0, 1.0, 0.0, 0.75]]]], dtype=torch.float64)
    confidence = torch.tensor([[[[0.0, 0.2, 0.9, 0.0, 0.7]]]], dtype=torch.float64)
    intrinsics = _intrinsics(fx_px=2.0).double()
    previous = _camera_from_world_at_x(0.0).double()
    current = _camera_from_world_at_x(0.2).double()

    canonical = zbuffer_reproject(
        disparity, depth, confidence, intrinsics, previous, current
    )
    v2 = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        intrinsics,
        previous,
        current,
        top_k=1,
        splat_footprint="nearest",
    ).as_single_winner()

    for field in (
        "disparity_hr_px",
        "depth_m",
        "confidence",
        "projected_uv",
        "fractional_offset",
        "source_uv",
    ):
        torch.testing.assert_close(getattr(v2, field), getattr(canonical, field))
    for field in ("valid_mask", "visibility_mask", "collision_mask"):
        assert torch.equal(getattr(v2, field), getattr(canonical, field))


def test_dual_calibration_projects_with_target_k_and_returns_target_disparity() -> None:
    disparity = torch.tensor([[[[0.1, 0.0, 0.0]]]], dtype=torch.float64)
    depth = torch.tensor([[[[2.0, 0.0, 0.0]]]], dtype=torch.float64)
    confidence = torch.ones_like(depth)
    hidden = torch.tensor([[[[7.0, 0.0, 0.0]]]], requires_grad=True)
    source_k = _intrinsics(fx_px=2.0).double()
    target_k = torch.tensor(
        [[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    identity = torch.eye(4, dtype=torch.float64)
    result = topk_z_aware_splat(
        disparity,
        depth,
        confidence,
        source_k,
        identity,
        identity,
        intrinsics_current_grid_3x3=target_k,
        intrinsics_previous_hr_3x3=source_k,
        intrinsics_current_hr_3x3=target_k,
        baseline_previous_m=torch.tensor([0.1], dtype=torch.float64),
        baseline_current_m=torch.tensor([0.2], dtype=torch.float64),
        top_k=1,
        splat_footprint="nearest",
        previous_hidden_feature=hidden,
    )

    assert result.valid_mask.tolist() == [[[[False, True, False]]]]
    assert result.disparity_hr_px[0, 0, 0, 1].item() == pytest.approx(0.4)
    assert result.projected_uv_grid_px[0, 0, 0, 0, 1].item() == pytest.approx(
        1.0
    )
    assert result.weighted_hidden_feature[0, 0, 0, 1].item() == pytest.approx(
        7.0
    )
    result.weighted_hidden_feature.sum().backward()
    assert hidden.grad is not None
    assert hidden.grad[0, 0, 0, 0].item() == pytest.approx(1.0)


def test_explicit_same_calibration_matches_legacy_topk() -> None:
    disparity = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float64)
    depth = 1.0 / disparity
    confidence = torch.tensor([[[[0.2, 0.5, 0.9]]]], dtype=torch.float64)
    calibration = _intrinsics(fx_px=5.0).double()
    baseline = torch.tensor([0.2], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    common = (
        disparity,
        depth,
        confidence,
        calibration,
        identity,
        identity,
    )
    legacy = topk_z_aware_splat(
        *common, top_k=2, splat_footprint="bilinear"
    )
    explicit = topk_z_aware_splat(
        *common,
        intrinsics_current_grid_3x3=calibration,
        intrinsics_previous_hr_3x3=calibration,
        intrinsics_current_hr_3x3=calibration,
        baseline_previous_m=baseline,
        baseline_current_m=baseline,
        top_k=2,
        splat_footprint="bilinear",
    )
    for name in TopKSplatResult.__dataclass_fields__:
        legacy_value = getattr(legacy, name)
        explicit_value = getattr(explicit, name)
        if legacy_value is None:
            assert explicit_value is None
        else:
            torch.testing.assert_close(explicit_value, legacy_value)


def test_invalid_arguments_and_merge_contract_fail_loudly() -> None:
    scalar = torch.ones((1, 1, 1, 1))
    common = (scalar, scalar, scalar, _intrinsics(), torch.eye(4), torch.eye(4))
    with pytest.raises(ValueError, match="splat_footprint"):
        topk_z_aware_splat(*common, splat_footprint="round")
    with pytest.raises(ValueError, match="non-negative"):
        topk_z_aware_splat(*common, temporal_age_frames=-1)
    result = topk_z_aware_splat(*common, top_k=1)
    with pytest.raises(ValueError, match="smaller"):
        merge_topk_splat_results([result], top_k=2)
    with pytest.raises(ValueError, match="K=1"):
        topk_z_aware_splat(*common, top_k=2).as_single_winner()
