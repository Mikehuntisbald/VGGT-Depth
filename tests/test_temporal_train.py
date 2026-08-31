from __future__ import annotations

import torch

import train
from models.ffs_omega_tsr import ModelOutput


def _identity_extrinsics(batch: int) -> torch.Tensor:
    value = torch.zeros(batch, 10, 3, 4)
    value[:, :, :3, :3] = torch.eye(3)
    return value


def _output(disparity_value: float = 4.0) -> ModelOutput:
    disparity = torch.full((1, 1, 4, 6), disparity_value, requires_grad=True)
    log_variance = torch.zeros_like(disparity, requires_grad=True)
    source_weights = torch.softmax(
        torch.zeros(1, 3, 2, 3, requires_grad=True), dim=1
    )
    return ModelOutput(
        disparity_hr_px=disparity,
        disparity_raw_hr_px=disparity,
        source_weights=source_weights,
        log_variance=log_variance,
        uncertainty=torch.exp(log_variance),
        hidden_state=(torch.ones(1, 2, 2, 3),),
        anchor_gate=torch.ones_like(disparity),
        source_valid_mask=torch.ones(1, 3, 2, 3, dtype=torch.bool),
    )


def _transport(pose_valid: bool) -> train.TemporalTransport:
    rgb = torch.zeros(1, 3, 4, 6)
    K_hr = torch.tensor(
        [[[10.0, 0.0, 2.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]]
    )
    return train.build_temporal_transport(
        previous_output=_output(),
        previous_rgb_hr=rgb,
        current_rgb_hr=rgb.clone(),
        current_ffs_disparity_hr_px=torch.full((1, 1, 2, 3), 4.0),
        current_ffs_confidence=torch.full((1, 1, 2, 3), 0.5),
        intrinsics_current_hr=K_hr,
        baseline_current_m=torch.tensor([0.1]),
        temporal_extrinsics_camera_from_world=_identity_extrinsics(1),
        temporal_pose_valid=torch.tensor([pose_valid]),
        scale=2,
    )


def test_pose_invalid_strictly_zeroes_temporal_history() -> None:
    transport = _transport(False)
    assert not bool(transport.valid_history.any())
    assert not bool(transport.visibility_mask.any())
    assert transport.disparity_history_hr_px.eq(0).all()
    assert transport.confidence_history.eq(0).all()
    assert transport.fractional_offset_px.eq(0).all()


def test_pose_invalid_resets_convgru_state_for_only_rejected_batch_items() -> None:
    state = (torch.arange(12.0).reshape(2, 2, 1, 3),)
    reset = train._reset_hidden_where_pose_invalid(
        state, torch.tensor([True, False])
    )
    assert reset is not None
    torch.testing.assert_close(reset[0][0], state[0][0])
    assert reset[0][1].eq(0).all()


def test_identity_pose_and_equal_rgb_produce_valid_zero_residual_history() -> None:
    transport = _transport(True)
    assert bool(transport.valid_history.all())
    assert bool(transport.visibility_mask.all())
    torch.testing.assert_close(
        transport.disparity_history_hr_px,
        torch.full_like(transport.disparity_history_hr_px, 4.0),
    )
    assert transport.photometric_residual.eq(0).all()
    assert bool(transport.static_mask.all())
    assert bool(transport.geometry_consistent_mask.all())
    assert transport.disparity_history_loss_hr_px.shape == (1, 1, 4, 6)
    assert transport.disparity_history_hr_px.shape == (1, 1, 2, 3)


def test_transport_resolves_thin_hr_collision_before_lr_sampling() -> None:
    disparity = torch.zeros(1, 1, 2, 4)
    # With fx=2, B=1 and camera motion +1 m:
    # u=1,d=1 (Z=2) and u=2,d=2 (Z=1) both land at HR u=0.
    disparity[0, 0, 0, 1] = 1.0
    disparity[0, 0, 0, 2] = 2.0
    log_variance = torch.zeros_like(disparity)
    previous_output = ModelOutput(
        disparity_hr_px=disparity,
        disparity_raw_hr_px=disparity,
        source_weights=torch.ones(1, 3, 1, 2) / 3.0,
        log_variance=log_variance,
        uncertainty=torch.ones_like(disparity),
        hidden_state=(),
        anchor_gate=torch.ones_like(disparity),
        source_valid_mask=torch.ones(1, 3, 1, 2, dtype=torch.bool),
    )
    extrinsics = _identity_extrinsics(1)
    extrinsics[:, 8, 0, 3] = -1.0
    transport = train.build_temporal_transport(
        previous_output=previous_output,
        previous_rgb_hr=torch.zeros(1, 3, 2, 4),
        current_rgb_hr=torch.zeros(1, 3, 2, 4),
        current_ffs_disparity_hr_px=torch.full((1, 1, 1, 2), 2.0),
        current_ffs_confidence=torch.zeros(1, 1, 1, 2),
        intrinsics_current_hr=torch.tensor(
            [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
        baseline_current_m=torch.tensor([1.0]),
        temporal_extrinsics_camera_from_world=extrinsics,
        temporal_pose_valid=torch.tensor([True]),
        scale=2,
    )
    # If disparity were reduced to LR first, the two one-pixel-wide surfaces
    # would be blended and this HR collision would disappear.
    assert bool(transport.collision_mask_hr[0, 0, 0, 0])
    assert bool(transport.collision_mask[0, 0, 0, 0])
    assert transport.fractional_offset_px.shape == (1, 2, 1, 2)


def test_temporal_step_loss_uses_only_effective_transport_mask() -> None:
    output = _output(5.0)
    target = torch.full((1, 1, 4, 6), 5.0)
    batch = {
        "teacher_disparity_hr_px": target,
        "teacher_confidence": torch.ones_like(target),
        "teacher_trusted_mask": torch.ones_like(target, dtype=torch.bool),
        "observation_disparity_lr_px": torch.full((1, 1, 2, 3), 2.5),
        "observation_confidence": torch.full((1, 1, 2, 3), 0.5),
        "observation_trusted_mask": torch.ones(1, 1, 2, 3, dtype=torch.bool),
    }
    valid_transport = _transport(True)
    valid_loss = train.compute_stage_b_step_loss(
        output, batch, transport=valid_transport
    )
    assert valid_loss.temporal.item() > 0

    invalid_transport = _transport(False)
    invalid_loss = train.compute_stage_b_step_loss(
        output, batch, transport=invalid_transport
    )
    assert invalid_loss.temporal.item() == 0.0
    invalid_loss.total.backward()
    assert output.disparity_hr_px.grad is not None
