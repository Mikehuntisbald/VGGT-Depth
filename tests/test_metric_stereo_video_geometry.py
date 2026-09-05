from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geometry.causal_memory import (  # noqa: E402
    TemporalGeometryState,
    VisibilityAwareTemporalMemory,
)
from models.metric_stereo_video_geometry import (  # noqa: E402
    CausalMetricStereoVideoGeometry,
    MetricStereoFrameInput,
    StereoBackboneFeatures,
    VGGTCausalGeometryFeatures,
    align_vggt_inverse_depth_to_metric_stereo,
)


def _intrinsics(batch: int = 1) -> torch.Tensor:
    value = torch.tensor(
        [[10.0, 0.0, 4.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]
    )
    return value.unsqueeze(0).repeat(batch, 1, 1)


def _stereo_transform(batch: int = 1) -> torch.Tensor:
    value = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    value[:, 0, 3] = -0.2
    return value


def _frame(
    time_index: int,
    *,
    seed: int,
    feature_grad: bool = False,
    context_time_indices: tuple[int, ...] | None = None,
) -> MetricStereoFrameInput:
    generator = torch.Generator().manual_seed(seed)
    stereo_feature = torch.rand(1, 4, 4, 5, generator=generator)
    vggt_feature = torch.rand(1, 6, 2, 3, generator=generator)
    stereo_feature.requires_grad_(feature_grad)
    vggt_feature.requires_grad_(feature_grad)
    return MetricStereoFrameInput(
        left_rgb=torch.rand(1, 3, 8, 10, generator=generator),
        right_rgb=torch.rand(1, 3, 8, 10, generator=generator),
        intrinsics_left_3x3=_intrinsics(),
        T_right_from_left_m=_stereo_transform(),
        T_current_from_previous_m=(
            None if time_index == 0 else torch.eye(4).unsqueeze(0)
        ),
        # fx * baseline is 2 m*px, so this is exactly 1 / metre.
        lowres_disparity_left_px=torch.full((1, 1, 4, 5), 2.0),
        lowres_disparity_valid_mask=torch.ones(1, 1, 4, 5, dtype=torch.bool),
        lowres_disparity_confidence=torch.ones(1, 1, 4, 5),
        stereo_features=StereoBackboneFeatures(stereo_feature, time_index),
        vggt_features=VGGTCausalGeometryFeatures(
            feature_map=vggt_feature,
            inverse_depth_relative=torch.full((1, 1, 2, 3), 3.0),
            confidence=torch.ones(1, 1, 2, 3),
            context_time_indices=(
                tuple(range(time_index + 1))
                if context_time_indices is None
                else context_time_indices
            ),
            time_index=time_index,
        ),
        time_index=time_index,
    )


def _model(*, checkpointing: bool = False) -> CausalMetricStereoVideoGeometry:
    torch.manual_seed(7)
    return CausalMetricStereoVideoGeometry(
        stereo_feature_channels=4,
        vggt_feature_channels=6,
        hidden_channels=16,
        residual_blocks=1,
        minimum_gauge_overlap=4,
        activation_checkpointing=checkpointing,
    )


def test_metric_gauge_recovers_scale_only_alignment() -> None:
    relative = torch.tensor(
        [[[[2.0, 4.0], [6.0, 8.0]]], [[[1.0, 2.0], [3.0, 4.0]]]]
    )
    expected_scale = torch.tensor([0.25, 2.0]).reshape(2, 1, 1, 1)
    metric = relative * expected_scale
    confidence = torch.ones_like(relative)
    valid = torch.ones_like(relative, dtype=torch.bool)

    result = align_vggt_inverse_depth_to_metric_stereo(
        relative,
        metric,
        relative_confidence=confidence,
        metric_confidence=confidence,
        relative_valid_mask=valid,
        metric_valid_mask=valid,
        minimum_overlap=4,
    )

    torch.testing.assert_close(
        result.scale_m_inv_per_relative_unit, expected_scale
    )
    torch.testing.assert_close(result.inverse_depth_m_inv, metric)
    assert result.valid_mask.tolist() == [True, True]
    assert result.overlap_count.tolist() == [4, 4]


def test_output_has_one_consistent_metric_geometry_owner() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model.forward_step(_frame(0, seed=1))

    expected_shape = (1, 1, 8, 10)
    for value in (
        output.inverse_depth_m_inv,
        output.depth_m,
        output.disparity_left_px,
        output.valid_logits,
        output.valid_probability,
        output.log_variance,
        output.uncertainty,
        output.confidence,
    ):
        assert value.shape == expected_shape
        assert bool(torch.isfinite(value).all())
    torch.testing.assert_close(
        output.depth_m * output.inverse_depth_m_inv,
        torch.ones_like(output.depth_m),
        atol=1e-6,
        rtol=1e-6,
    )
    # fx=10 px and baseline=0.2 m.
    torch.testing.assert_close(
        output.disparity_left_px,
        2.0 * output.inverse_depth_m_inv,
        atol=1e-6,
        rtol=1e-6,
    )
    assert output.disparity_right_px is None
    assert output.valid_mask.dtype == torch.bool
    assert bool(output.gauge.valid_mask.all())
    torch.testing.assert_close(
        output.gauge.scale_m_inv_per_relative_unit,
        torch.full((1, 1, 1, 1), 1.0 / 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not output.temporal.used_history


def test_injected_backbone_features_receive_gradients_with_checkpointing() -> None:
    model = _model(checkpointing=True).train()
    frame = _frame(0, seed=2, feature_grad=True)

    output = model.forward_step(frame)
    loss = (
        output.inverse_depth_m_inv.mean()
        + output.valid_logits.square().mean()
        + output.log_variance.mean()
    )
    loss.backward()

    stereo_grad = frame.stereo_features.feature_map.grad
    vggt_grad = frame.vggt_features.feature_map.grad
    assert stereo_grad is not None and bool(torch.isfinite(stereo_grad).all())
    assert vggt_grad is not None and bool(torch.isfinite(vggt_grad).all())
    assert stereo_grad.abs().sum() > 0
    assert vggt_grad.abs().sum() > 0


def test_causal_prefix_output_is_invariant_to_future_frame() -> None:
    model = _model().eval()
    current = _frame(0, seed=3)
    future_a = _frame(1, seed=4)
    future_b = _frame(1, seed=99)

    with torch.no_grad():
        prefix = model((current,)).frames[0]
        clip_a = model((current, future_a)).frames[0]
        clip_b = model((current, future_b)).frames[0]

    torch.testing.assert_close(prefix.inverse_depth_m_inv, clip_a.inverse_depth_m_inv)
    torch.testing.assert_close(prefix.inverse_depth_m_inv, clip_b.inverse_depth_m_inv)
    torch.testing.assert_close(prefix.valid_logits, clip_b.valid_logits)
    torch.testing.assert_close(prefix.log_variance, clip_b.log_variance)


def test_future_vggt_context_fails_closed() -> None:
    model = _model().eval()
    frame = _frame(1, seed=5, context_time_indices=(0, 1, 2))

    with pytest.raises(ValueError, match="current frame|future"):
        model.forward_step(frame)


def test_visibility_memory_preserves_gradients_and_rejects_depth_mismatch() -> None:
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    feature.requires_grad_()
    inverse_depth = torch.full((1, 1, 2, 3), 0.5)
    valid = torch.ones_like(inverse_depth, dtype=torch.bool)
    state = TemporalGeometryState(
        feature=feature,
        inverse_depth_m_inv=inverse_depth,
        confidence=torch.ones_like(inverse_depth),
        valid_mask=valid,
        intrinsics_hr_3x3=torch.tensor(
            [[[6.0, 0.0, 2.5], [0.0, 6.0, 1.5], [0.0, 0.0, 1.0]]]
        ),
        baseline_m=torch.tensor([0.2]),
        image_size_hw=(4, 6),
        time_index=0,
    )
    memory = VisibilityAwareTemporalMemory(
        relative_depth_tolerance=0.01,
        absolute_depth_tolerance_m=0.01,
    )
    identity = torch.eye(4).unsqueeze(0)

    matching = memory(
        state,
        intrinsics_current_hr_3x3=state.intrinsics_hr_3x3,
        baseline_current_m=state.baseline_m,
        image_size_current_hw=state.image_size_hw,
        T_current_from_previous_m=identity,
        current_inverse_depth_m_inv=inverse_depth,
        current_valid_mask=valid,
    )
    assert bool(matching.valid_mask.all())
    torch.testing.assert_close(matching.feature, feature)
    matching.feature.sum().backward()
    torch.testing.assert_close(feature.grad, torch.ones_like(feature))

    mismatching = memory(
        state.detach(),
        intrinsics_current_hr_3x3=state.intrinsics_hr_3x3,
        baseline_current_m=state.baseline_m,
        image_size_current_hw=state.image_size_hw,
        T_current_from_previous_m=identity,
        current_inverse_depth_m_inv=torch.full_like(inverse_depth, 0.1),
        current_valid_mask=valid,
    )
    assert not bool(mismatching.valid_mask.any())
    assert not bool(mismatching.depth_consistent_mask.any())
    assert torch.count_nonzero(mismatching.feature) == 0


def test_second_frame_uses_pose_warped_causal_state() -> None:
    model = _model().eval()
    first = _frame(0, seed=6)
    second = _frame(1, seed=7)

    with torch.no_grad():
        result = model((first, second))

    assert len(result.frames) == 2
    assert result.frames[1].temporal.used_history
    assert bool(result.frames[1].temporal.valid_mask.all())
    assert bool(result.frames[1].temporal.zbuffer_visible_mask.all())
    assert bool(
        (result.frames[1].temporal.warped_inverse_depth_pre_consistency_m_inv > 0).all()
    )
    assert result.final_state.time_index == 1


def test_negative_residual_keeps_causal_inverse_depth_cache_at_fp32_floor() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.lowres_head.weight.zero_()
        model.lowres_head.bias.zero_()
        model.lowres_head.bias[0] = -100.0

    first = _frame(0, seed=8)
    second = _frame(1, seed=9)
    first.lowres_disparity_left_px.fill_(2e-8)
    second.lowres_disparity_left_px.fill_(2e-8)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        result = model((first, second))

    # tanh(-100) gives the minimum -0.25 residual.  Applying the floor only
    # before exp(-0.25) would cache 7.79e-9 and make depth/disparity disagree
    # in the next frame's metric witness check.
    assert result.frames[0].state.inverse_depth_m_inv.dtype == torch.float32
    assert bool((result.frames[0].state.inverse_depth_m_inv >= 1e-8).all())
    assert bool((result.final_state.inverse_depth_m_inv >= 1e-8).all())
    assert result.frames[1].temporal.used_history
