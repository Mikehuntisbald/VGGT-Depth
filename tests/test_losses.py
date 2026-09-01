from __future__ import annotations

import pytest
import torch

from losses import (
    LossWeights,
    combine_loss_terms,
    disparity_loss,
    ffs_gate_regularizer,
    gradient_loss,
    laplace_uncertainty_nll,
    lower_bound_penalty,
    measurement_consistency_loss,
    sample_hr_at_lr_centers,
    temporal_consistency_loss,
    temporal_residual_consistency_loss,
)


def test_empty_and_nan_masks_return_finite_differentiable_zero() -> None:
    prediction = torch.tensor([[[[float("nan"), 2.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, float("nan")]]]])
    loss = disparity_loss(
        prediction,
        target,
        valid_mask=torch.zeros_like(target, dtype=torch.bool),
    )
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_hr_center_sampling_and_measurement_unit_conversion() -> None:
    disparity_hr_px = torch.tensor(
        [[[[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]]]]
    )
    sampled_hr_px = sample_hr_at_lr_centers(disparity_hr_px, scale=2)
    torch.testing.assert_close(sampled_hr_px, torch.tensor([[[[3.0, 7.0]]]]))
    observation_lr_px = sampled_hr_px / 2.0
    loss = measurement_consistency_loss(
        disparity_hr_px,
        observation_lr_px,
        torch.ones_like(observation_lr_px, dtype=torch.bool),
        scale=2,
    )
    assert loss.item() == 0.0


def test_gradient_loss_is_zero_for_equal_edges() -> None:
    target = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    assert gradient_loss(target.clone(), target).item() == 0.0


def test_lower_bound_penalty_is_a_finite_zero_floor_hinge() -> None:
    disparity = torch.tensor(
        [[[[1.0, 0.0, -2.0, float("nan")]]]], requires_grad=True
    )
    penalty = lower_bound_penalty(disparity, lower_bound_hr_px=0.0)
    # The finite elements are [1, 0, -2], therefore mean([0, 0, 4]).
    assert penalty.item() == pytest.approx(4.0 / 3.0)
    penalty.backward()
    assert disparity.grad is not None
    assert bool(torch.isfinite(disparity.grad).all())
    assert disparity.grad[0, 0, 0, 0].item() == 0.0
    assert disparity.grad[0, 0, 0, 1].item() == 0.0
    assert disparity.grad[0, 0, 0, 2].item() < 0.0


def test_temporal_loss_excludes_collision_and_photometric_failure() -> None:
    current = torch.tensor([[[[1.0, 10.0, 20.0]]]])
    history = torch.tensor([[[[2.0, 0.1, 0.1]]]])
    ones = torch.ones_like(current, dtype=torch.bool)
    collision = torch.tensor([[[[False, True, False]]]])
    photo = torch.tensor([[[[0.0, 0.0, 2.0]]]])
    loss = temporal_consistency_loss(
        current,
        history,
        static_mask=ones,
        visibility_mask=ones,
        collision_mask=collision,
        photometric_residual=photo,
        max_photometric_residual=1.0,
        geometry_consistent_mask=ones,
    )
    assert 0.999 < loss.item() < 1.001


def test_temporal_residual_loss_cancels_bias_and_penalizes_flicker() -> None:
    teacher_previous_warped = torch.tensor([[[[10.0, 10.0]]]])
    teacher_current = torch.tensor([[[[12.0, 12.0]]]])
    prediction_previous_warped = torch.tensor([[[[13.0, 13.0]]]])
    prediction_current = torch.tensor(
        [[[[15.0, 17.0]]]], requires_grad=True
    )
    valid = torch.ones_like(teacher_current, dtype=torch.bool)
    loss = temporal_residual_consistency_loss(
        prediction_current,
        prediction_previous_warped,
        teacher_current,
        teacher_previous_warped,
        static_mask=valid,
        visibility_mask=valid,
        collision_mask=torch.zeros_like(valid),
        photometric_residual=torch.zeros_like(teacher_current),
        max_photometric_residual=0.1,
        geometry_consistent_mask=valid,
        current_reference_valid_mask=valid,
        warped_previous_reference_valid_mask=valid,
        epsilon=1e-6,
    )
    assert loss.item() == pytest.approx(1.0, abs=1e-5)
    loss.backward()
    assert prediction_current.grad is not None
    assert prediction_current.grad[..., 0].item() == pytest.approx(0.0, abs=1e-5)
    assert prediction_current.grad[..., 1].item() > 0.0


def test_temporal_residual_loss_is_empty_safe() -> None:
    prediction = torch.ones((1, 1, 1, 1), requires_grad=True)
    reference = torch.ones_like(prediction)
    empty = torch.zeros_like(prediction, dtype=torch.bool)
    loss = temporal_residual_consistency_loss(
        prediction,
        prediction.detach(),
        reference,
        reference,
        static_mask=empty,
        visibility_mask=torch.ones_like(empty),
        collision_mask=empty,
        photometric_residual=torch.zeros_like(prediction),
        max_photometric_residual=0.1,
        geometry_consistent_mask=torch.ones_like(empty),
        current_reference_valid_mask=torch.ones_like(empty),
        warped_previous_reference_valid_mask=torch.ones_like(empty),
    )
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))


def test_uncertainty_nll_and_gate_regularizer_are_empty_safe() -> None:
    prediction = torch.ones(1, 1, 2, 2)
    target = prediction.clone()
    log_variance = torch.zeros_like(prediction)
    assert laplace_uncertainty_nll(prediction, target, log_variance).item() == 0.0

    weights = torch.full((1, 3, 2, 2), 1.0 / 3.0)
    confidence = torch.ones(1, 1, 2, 2)
    invalid = torch.zeros_like(confidence, dtype=torch.bool)
    assert ffs_gate_regularizer(weights, confidence, invalid).item() == 0.0


def test_gate_regularizer_rewards_ffs_ownership_only_when_trusted() -> None:
    confidence = torch.ones(1, 1, 1, 1)
    valid = torch.ones_like(confidence, dtype=torch.bool)
    ffs_owned = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    shared = torch.full((1, 3, 1, 1), 1.0 / 3.0)
    assert ffs_gate_regularizer(ffs_owned, confidence, valid).item() == 0.0
    assert ffs_gate_regularizer(shared, confidence, valid).item() > 0.6


def test_composite_uses_declared_mvp_coefficients() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    breakdown = combine_loss_terms(
        disparity=value,
        measurement=value,
        gradient=value,
        temporal=value,
        epipolar=value,
        uncertainty_nll=value,
        gate_regularizer=value,
        weights=LossWeights(),
    )
    assert breakdown.total.item() == pytest.approx(1.88)
    breakdown.total.backward()
    assert value.grad.item() == pytest.approx(1.88)
