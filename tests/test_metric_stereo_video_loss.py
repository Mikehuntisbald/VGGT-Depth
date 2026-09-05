from __future__ import annotations

import math

import pytest
import torch

from losses.metric_stereo_video import (
    MetricStereoVideoLoss,
    MetricStereoVideoLossWeights,
    combine_metric_stereo_video_losses,
    depth_from_inverse_depth_m_inv,
    disparity_from_inverse_depth_m_inv,
    inverse_depth_scale_alignment_loss,
    laplace_inverse_depth_uncertainty_loss,
    left_right_consistency_loss,
    metric_stereo_video_loss,
    normalized_log_depth_loss,
    photometric_reprojection_loss,
    pose_residual_loss,
    robust_disparity_loss,
    robust_multiscale_disparity_loss,
    robust_multiscale_pixel_disparity_loss,
    scale_alignment_loss,
    validity_classification_loss,
    visibility_aware_temporal_residual_loss,
    warp_right_disparity_to_left,
)


def test_canonical_inverse_depth_derives_metric_depth_and_disparity() -> None:
    inverse_depth = torch.tensor([[[[[0.5]]], [[[0.25]]]], [[[[1.0]]], [[[2.0]]]]])
    fx_px = torch.tensor([[100.0, 200.0], [50.0, 25.0]])
    disparity = disparity_from_inverse_depth_m_inv(
        inverse_depth, fx_px=fx_px, baseline_m=0.2
    )
    torch.testing.assert_close(
        disparity.flatten(), torch.tensor([10.0, 10.0, 10.0, 10.0])
    )
    torch.testing.assert_close(
        depth_from_inverse_depth_m_inv(inverse_depth).flatten(),
        torch.tensor([2.0, 4.0, 1.0, 0.5]),
    )

    with pytest.raises(ValueError, match="baseline_m must be finite and > 0"):
        disparity_from_inverse_depth_m_inv(inverse_depth, fx_px=100.0, baseline_m=0.0)


def test_depth_conversion_preserves_invalidity_instead_of_epsilon_filling() -> None:
    inverse_depth = torch.tensor([[[[1.0, 0.0, -1.0, float("nan")]]]])
    depth = depth_from_inverse_depth_m_inv(inverse_depth)
    assert depth[0, 0, 0, 0].item() == 1.0
    assert bool(torch.isnan(depth[0, 0, 0, 1:]).all())


def test_multiscale_disparity_uses_full_resolution_pixel_units_and_normalized_weights() -> (
    None
):
    target = torch.full((1, 1, 4, 6), 10.0)
    coarse_correct = torch.full((1, 1, 2, 3), 0.5)
    fine_wrong = torch.full((1, 1, 4, 6), 0.6, requires_grad=True)
    loss = robust_multiscale_disparity_loss(
        (coarse_correct, fine_wrong),
        target,
        fx_px=100.0,
        baseline_m=0.2,
        scale_weights=(1.0, 1.0),
        epsilon_px=1e-6,
    )
    # The coarse prediction is converted with the full-resolution focal length,
    # while the fine prediction has a 2 px error. Normalized weights give 1 px.
    assert loss.item() == pytest.approx(1.0, abs=1e-5)
    loss.backward()
    assert fine_wrong.grad is not None
    assert bool(torch.isfinite(fine_wrong.grad).all())


def test_multiscale_pixel_disparity_does_not_rescale_values_during_resize() -> None:
    target = torch.full((1, 1, 4, 6), 10.0)
    coarse = torch.full((1, 1, 2, 3), 10.0, requires_grad=True)
    loss = robust_multiscale_pixel_disparity_loss(coarse, target)
    assert loss.item() == 0.0
    loss.backward()
    assert coarse.grad is not None
    assert bool(torch.isfinite(coarse.grad).all())


def test_multiscale_and_log_depth_losses_are_empty_and_target_nonfinite_safe() -> None:
    prediction = torch.full((1, 1, 1, 2), 0.5, requires_grad=True)
    target_disparity = torch.tensor([[[[10.0, float("inf")]]]])
    empty = torch.zeros_like(target_disparity, dtype=torch.bool)
    disparity_loss = robust_multiscale_disparity_loss(
        prediction,
        target_disparity,
        fx_px=100.0,
        baseline_m=0.2,
        valid_mask=empty,
    )
    depth_loss = normalized_log_depth_loss(
        prediction,
        torch.tensor([[[[0.5, float("nan")]]]]),
        valid_mask=empty,
    )
    total = disparity_loss + depth_loss
    assert total.item() == 0.0
    assert torch.isfinite(total)
    total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_nonfinite_prediction_fails_fast_even_on_an_empty_cpu_domain() -> None:
    prediction = torch.tensor([[[[float("nan")]]]], requires_grad=True)
    with pytest.raises(ValueError, match="non-finite predictions"):
        robust_multiscale_disparity_loss(
            prediction,
            torch.ones_like(prediction),
            fx_px=1.0,
            baseline_m=1.0,
            valid_mask=torch.zeros_like(prediction, dtype=torch.bool),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nonfinite_cuda_prediction_poison_guard_does_not_synchronize_away() -> None:
    prediction = torch.tensor([[[[float("nan")]]]], device="cuda", requires_grad=True)
    loss = robust_multiscale_disparity_loss(
        prediction,
        torch.ones_like(prediction),
        fx_px=1.0,
        baseline_m=1.0,
        valid_mask=torch.zeros_like(prediction, dtype=torch.bool),
    )
    assert not bool(torch.isfinite(loss).item())


def test_log_depth_is_dimensionless_but_retains_metric_scale_error() -> None:
    target = torch.ones(2, 1, 1, 2)
    prediction = target * 2.0
    loss = normalized_log_depth_loss(prediction, target, epsilon=1e-6)
    # No scene-wise centering: a global 2x inverse-depth scale error is visible.
    assert loss.item() == pytest.approx(math.log(2.0), abs=2e-6)


def test_temporal_residual_excludes_dynamic_and_occluded_pixels() -> None:
    current = torch.tensor([[[[2.0, 8.0, 32.0, 64.0]]]], requires_grad=True)
    previous_warped = torch.ones_like(current)
    target_current = torch.tensor([[[[2.0, 4.0, 4.0, 4.0]]]])
    target_previous_warped = torch.ones_like(current)
    visible = torch.ones_like(current, dtype=torch.bool)
    dynamic = torch.tensor([[[[False, False, True, False]]]])
    occluded = torch.zeros_like(dynamic)
    collision = torch.tensor([[[[False, False, False, True]]]])
    loss = visibility_aware_temporal_residual_loss(
        current,
        previous_warped,
        target_current,
        target_previous_warped,
        visibility_mask=visible,
        dynamic_mask=dynamic,
        occlusion_mask=occluded,
        collision_mask=collision,
        epsilon=1e-6,
    )
    assert loss.item() == pytest.approx(math.log(2.0) / 2.0, abs=2e-6)
    loss.backward()
    assert current.grad is not None
    assert current.grad[0, 0, 0, 1].item() > 0
    assert current.grad[0, 0, 0, 2].item() == 0
    assert current.grad[0, 0, 0, 3].item() == 0


def test_photometric_reprojection_honors_occlusion_dynamic_and_nan_masks() -> None:
    reference = torch.zeros(1, 3, 1, 4)
    reprojected = torch.ones_like(reference, requires_grad=True)
    occlusion = torch.tensor([[[[False, True, False, False]]]])
    dynamic = torch.tensor([[[[False, False, True, False]]]])
    valid = torch.tensor([[[[1.0, 1.0, 1.0, float("nan")]]]])
    loss = photometric_reprojection_loss(
        reference,
        reprojected,
        valid_mask=valid,
        occlusion_mask=occlusion,
        dynamic_mask=dynamic,
        ssim_weight=0.0,
        epsilon=1e-6,
    )
    assert loss.item() == pytest.approx(1.0, abs=2e-6)
    loss.backward()
    assert reprojected.grad is not None
    assert bool((reprojected.grad[..., 0] > 0).all())
    assert torch.equal(
        reprojected.grad[..., 1:], torch.zeros_like(reprojected.grad[..., 1:])
    )


def test_left_right_consistency_uses_x_left_minus_disparity() -> None:
    left = torch.ones(1, 1, 1, 5)
    right = torch.tensor([[[[1.0, 1.0, 2.0, 1.0, 1.0]]]])
    sampled, valid = warp_right_disparity_to_left(left, right)
    assert valid.tolist() == [[[[False, True, True, True, True]]]]
    torch.testing.assert_close(sampled[0, 0, 0, 1:], torch.tensor([1.0, 1.0, 2.0, 1.0]))

    loss = left_right_consistency_loss(left, right, epsilon_px=1e-6)
    assert loss.item() == pytest.approx(0.25, abs=2e-6)
    occlusion = torch.tensor([[[[False, False, False, True, False]]]])
    assert left_right_consistency_loss(
        left, right, occlusion_mask=occlusion, epsilon_px=1e-6
    ).item() == pytest.approx(0.0, abs=1e-7)


def test_pose_and_scale_losses_use_dimensionless_normalization() -> None:
    pose = torch.tensor([[0.1, 0.0, 0.0, 0.1, 0.0, 0.0]])
    pose_loss = pose_residual_loss(
        pose,
        translation_normalizer_m=0.1,
        rotation_normalizer_rad=0.1,
        epsilon=1e-6,
    )
    assert pose_loss.item() == pytest.approx(1.0 / 3.0, abs=2e-6)
    assert scale_alignment_loss(
        torch.tensor([math.log(2.0)]), torch.zeros(1), epsilon=1e-6
    ).item() == pytest.approx(math.log(2.0), abs=2e-6)


def test_scale_alignment_can_use_stereo_metric_inverse_depth_directly() -> None:
    unscaled = torch.ones(1, 2, 1, 1, 2)
    metric_target = unscaled * 2.0
    predicted_log_scale = torch.full((1, 2), math.log(2.0), requires_grad=True)
    loss = inverse_depth_scale_alignment_loss(
        unscaled,
        metric_target,
        predicted_log_scale,
        valid_mask=torch.ones_like(unscaled, dtype=torch.bool),
        epsilon=1e-6,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert predicted_log_scale.grad is not None
    assert bool(torch.isfinite(predicted_log_scale.grad).all())


def test_laplace_uncertainty_uses_log_variance_and_detaches_geometry() -> None:
    prediction = torch.full((1, 1, 1, 1), 2.0, requires_grad=True)
    target = torch.ones_like(prediction)
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    loss = laplace_inverse_depth_uncertainty_loss(prediction, target, log_variance)
    assert loss.item() == pytest.approx(math.sqrt(2.0) * math.log(2.0), abs=2e-6)
    loss.backward()
    assert prediction.grad is None
    assert log_variance.grad is not None
    assert bool(torch.isfinite(log_variance.grad).all())


def test_laplace_bounds_match_model_head_without_an_upper_dead_zone() -> None:
    prediction = torch.full((1, 1, 1, 1), 2.0)
    log_variance = torch.full_like(prediction, 8.0, requires_grad=True)
    loss = laplace_inverse_depth_uncertainty_loss(
        prediction, torch.ones_like(prediction), log_variance
    )
    loss.backward()
    assert log_variance.grad is not None
    assert log_variance.grad.item() != 0.0


def test_residual_confidence_weights_are_detached_to_prevent_gaming() -> None:
    prediction = torch.tensor([[[[2.0]]]], requires_grad=True)
    confidence = torch.ones_like(prediction, requires_grad=True)
    loss = robust_disparity_loss(
        prediction,
        torch.ones_like(prediction),
        confidence=confidence,
    )
    loss.backward()
    assert prediction.grad is not None
    assert confidence.grad is None


def test_validity_loss_supervises_invalid_pixels_and_is_empty_safe() -> None:
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    target = torch.tensor([[[[True, False]]]])
    loss = validity_classification_loss(logits, target)
    assert loss.item() == pytest.approx(math.log(2.0))

    empty = validity_classification_loss(
        logits,
        target,
        supervision_mask=torch.zeros_like(target),
    )
    assert empty.item() == 0.0 and torch.isfinite(empty)
    empty.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_composition_averages_related_subterms_instead_of_double_counting() -> None:
    one = torch.tensor(1.0, requires_grad=True)
    two = one * 2
    four = one * 4
    weights = MetricStereoVideoLossWeights(
        disparity=1,
        depth=1,
        temporal=1,
        reprojection=1,
        left_right_consistency=1,
        pose_scale=1,
        uncertainty=1,
        validity=1,
    )
    result = combine_metric_stereo_video_losses(
        disparity=one,
        depth=one,
        temporal=one,
        stereo_reprojection=two,
        temporal_reprojection=four,
        left_right_consistency=one,
        pose=two,
        scale=four,
        uncertainty=one,
        validity=one,
        weights=weights,
    )
    assert result.reprojection.item() == 3.0
    assert result.pose_scale.item() == 3.0
    assert result.total.item() == 12.0


def test_composition_does_not_dilute_a_nonempty_optional_component() -> None:
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    result = combine_metric_stereo_video_losses(
        disparity=zero,
        depth=zero,
        temporal=zero,
        stereo_reprojection=one,
        temporal_reprojection=zero,
        stereo_reprojection_active=True,
        temporal_reprojection_active=False,
        left_right_consistency=zero,
        pose=one,
        scale=zero,
        pose_active=True,
        scale_active=False,
        uncertainty=zero,
        validity=zero,
    )
    assert result.reprojection.item() == 1.0
    assert result.pose_scale.item() == 1.0


def test_joint_mapping_api_runs_all_terms_and_backpropagates() -> None:
    inverse_depth = torch.full((1, 2, 1, 2, 5), 0.05, requires_grad=True)
    target_disparity = torch.ones_like(inverse_depth)
    valid = torch.ones_like(inverse_depth, dtype=torch.bool)
    rgb = torch.zeros(1, 2, 3, 2, 5)
    predictions = {
        "inverse_depth_m_inv": inverse_depth,
        "right_inverse_depth_m_inv": inverse_depth.detach().clone(),
        "valid_logits": torch.zeros_like(inverse_depth, requires_grad=True),
        "log_variance": torch.zeros_like(inverse_depth, requires_grad=True),
        "pose_residual": torch.zeros(1, 2, 6, requires_grad=True),
        "log_scale": torch.zeros(1, 2, requires_grad=True),
    }
    targets = {
        "disparity_left_px": target_disparity,
        "valid": valid,
        "left_rgb": rgb,
        "pose_residual": torch.zeros(1, 2, 6),
        "log_scale": torch.zeros(1, 2),
    }
    geometry = {
        "fx_px": torch.tensor([[100.0, 100.0]]),
        "baseline_m": 0.2,
        "temporal_warped_inverse_depth_m_inv": inverse_depth.detach().clone(),
        "temporal_warped_target_inverse_depth_m_inv": inverse_depth.detach().clone(),
        "stereo_reprojected_left_rgb": rgb.clone(),
        "temporal_reprojected_left_rgb": rgb.clone(),
    }
    masks = {
        "temporal_visibility": valid,
        "temporal_dynamic": torch.zeros_like(valid),
        "temporal_occlusion": torch.zeros_like(valid),
        "stereo_reprojection_valid": valid,
        "stereo_occlusion": torch.zeros_like(valid),
        "temporal_reprojection_valid": valid,
    }
    result = MetricStereoVideoLoss()(predictions, targets, geometry, masks)
    assert torch.isfinite(result.total)
    assert set(result.active_terms) == {
        "disparity",
        "depth",
        "temporal",
        "stereo_reprojection",
        "temporal_reprojection",
        "left_right_consistency",
        "pose",
        "scale",
        "uncertainty",
        "validity",
    }
    result.total.backward()
    assert inverse_depth.grad is not None
    assert bool(torch.isfinite(inverse_depth.grad).all())


def test_joint_temporal_loss_honors_dataset_dynamic_availability() -> None:
    inverse_depth = torch.tensor([[[[1.0]]], [[[2.0]]]], requires_grad=True)
    target_disparity = torch.ones_like(inverse_depth)
    valid = torch.ones_like(inverse_depth, dtype=torch.bool)
    result = metric_stereo_video_loss(
        {"inverse_depth_m_inv": inverse_depth},
        {"disparity_left_px": target_disparity, "valid": valid},
        {
            "fx_px": 1.0,
            "baseline_m": 1.0,
            "temporal_warped_inverse_depth_m_inv": torch.ones_like(inverse_depth),
            "temporal_warped_target_inverse_depth_m_inv": torch.ones_like(
                inverse_depth
            ),
        },
        {
            "temporal_visibility": valid,
            "dynamic_mask_current": torch.zeros_like(valid),
            "dynamic_mask_available": torch.tensor([True, False]),
            "temporal_occlusion": torch.zeros_like(valid),
        },
    )
    # The second sample has a residual but no dynamic annotation, so it cannot
    # be assumed static and is excluded from temporal supervision.
    assert result.temporal.item() == 0.0


def test_dataset_dynamic_mask_requires_an_explicit_availability_flag() -> None:
    inverse_depth = torch.ones(1, 1, 1, 1)
    valid = torch.ones_like(inverse_depth, dtype=torch.bool)
    with pytest.raises(KeyError, match="dynamic_mask_available"):
        metric_stereo_video_loss(
            {"inverse_depth_m_inv": inverse_depth},
            {"disparity_left_px": inverse_depth, "valid": valid},
            {
                "fx_px": 1.0,
                "baseline_m": 1.0,
                "temporal_warped_inverse_depth_m_inv": inverse_depth,
                "temporal_warped_target_inverse_depth_m_inv": inverse_depth,
            },
            {
                "temporal_visibility": valid,
                "dynamic_mask_current": torch.zeros_like(valid),
                "temporal_occlusion": torch.zeros_like(valid),
            },
        )


def test_joint_api_rejects_a_noncanonical_disparity_alias() -> None:
    inverse_depth = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="violates d=fx"):
        metric_stereo_video_loss(
            {
                "inverse_depth_m_inv": inverse_depth,
                "disparity_left_px": torch.ones_like(inverse_depth),
            },
            {
                "disparity_left_px": torch.full_like(inverse_depth, 20.0),
                "valid": torch.ones_like(inverse_depth, dtype=torch.bool),
            },
            {"fx_px": 100.0, "baseline_m": 0.2},
        )


def test_joint_api_converts_right_disparity_from_lr_to_hr_pixel_units() -> None:
    inverse_depth = torch.full((1, 1, 2, 6), 0.05)
    result = metric_stereo_video_loss(
        {
            "inverse_depth_m_inv": inverse_depth,
            "disparity_right_px": None,
            # 0.5 LR pixels becomes the canonical 1.0 HR pixel disparity.
            "disparity_right_lr_px": torch.full((1, 1, 1, 3), 0.5),
        },
        {
            "disparity_left_px": torch.ones_like(inverse_depth),
            "disparity_right_px": torch.ones_like(inverse_depth),
            "valid": torch.ones_like(inverse_depth, dtype=torch.bool),
            "valid_right": torch.ones_like(inverse_depth, dtype=torch.bool),
        },
        {
            "fx_px": 100.0,
            "baseline_m": 0.2,
            "stereo_lr_to_hr_scale": 2.0,
        },
    )
    assert result.left_right_consistency.item() == pytest.approx(0.0, abs=1e-7)
    assert "left_right_consistency" in result.active_terms
    assert "right_disparity_supervision" in result.active_terms


def test_joint_disparity_averages_nonempty_left_and_right_supervision() -> None:
    inverse_depth = torch.full((1, 1, 1, 6), 0.6)
    result = metric_stereo_video_loss(
        {
            "inverse_depth_m_inv": inverse_depth,
            "disparity_right_px": torch.full_like(inverse_depth, 14.0),
        },
        {
            "disparity_left_px": torch.full_like(inverse_depth, 10.0),
            "disparity_right_px": torch.full_like(inverse_depth, 10.0),
            "valid": torch.ones_like(inverse_depth, dtype=torch.bool),
            "valid_right": torch.ones_like(inverse_depth, dtype=torch.bool),
        },
        {"fx_px": 20.0, "baseline_m": 1.0},
    )
    # Left error is 2 px and right error is 4 px; L_disp is the view mean.
    assert result.disparity.item() == pytest.approx(3.0, abs=2e-3)


def test_empty_right_supervision_does_not_dilute_left_disparity_loss() -> None:
    inverse_depth = torch.full((1, 1, 1, 6), 0.6, requires_grad=True)
    result = metric_stereo_video_loss(
        {
            "inverse_depth_m_inv": inverse_depth,
            "disparity_right_px": torch.full_like(inverse_depth, 14.0),
        },
        {
            "disparity_left_px": torch.full_like(inverse_depth, 10.0),
            "disparity_right_px": torch.full_like(inverse_depth, 10.0),
            "valid": torch.ones_like(inverse_depth, dtype=torch.bool),
            "valid_right": torch.zeros_like(inverse_depth, dtype=torch.bool),
        },
        {"fx_px": 20.0, "baseline_m": 1.0},
    )
    assert result.disparity.item() == pytest.approx(2.0, abs=2e-3)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert inverse_depth.grad is not None
    assert bool(torch.isfinite(inverse_depth.grad).all())


def test_empty_left_supervision_does_not_dilute_right_disparity_loss() -> None:
    inverse_depth = torch.full((1, 1, 1, 6), 0.6)
    result = metric_stereo_video_loss(
        {
            "inverse_depth_m_inv": inverse_depth,
            "disparity_right_px": torch.full_like(inverse_depth, 14.0),
        },
        {
            "disparity_left_px": torch.full_like(inverse_depth, 10.0),
            "disparity_right_px": torch.full_like(inverse_depth, 10.0),
            "valid": torch.zeros_like(inverse_depth, dtype=torch.bool),
            "valid_right": torch.ones_like(inverse_depth, dtype=torch.bool),
        },
        {"fx_px": 20.0, "baseline_m": 1.0},
    )
    assert result.disparity.item() == pytest.approx(4.0, abs=2e-3)


def test_bfloat16_losses_accumulate_in_float32() -> None:
    inverse_depth = torch.full((1, 1, 2, 2), 0.5, dtype=torch.bfloat16)
    target_disparity = torch.full((1, 1, 2, 2), 10.0, dtype=torch.bfloat16)
    loss = robust_multiscale_disparity_loss(
        inverse_depth,
        target_disparity,
        fx_px=100.0,
        baseline_m=0.2,
    )
    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
