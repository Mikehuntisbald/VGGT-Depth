from __future__ import annotations

import torch

import train
from models.ffs_omega_tsr import ModelOutput


def _identity_window_extrinsics() -> torch.Tensor:
    extrinsics = torch.zeros(1, 10, 3, 4)
    extrinsics[:, :, :3, :3] = torch.eye(3)
    return extrinsics


def _single_lr_source_model_output() -> tuple[ModelOutput, torch.Tensor]:
    # Under x2 align_corners=False sampling, LR u=1 reads the HR centre at
    # u=2.5. Only that LR source location has a positive metric disparity.
    disparity_hr_px = torch.zeros(1, 1, 4, 6)
    disparity_hr_px[:, :, 0:2, 2:4] = 4.0
    hidden = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    hidden.requires_grad_()
    second_hidden = torch.ones_like(hidden, requires_grad=True)
    output = ModelOutput(
        disparity_hr_px=disparity_hr_px,
        disparity_raw_hr_px=disparity_hr_px,
        source_weights=torch.full((1, 3, 2, 3), 1.0 / 3.0),
        log_variance=torch.zeros_like(disparity_hr_px),
        uncertainty=torch.ones_like(disparity_hr_px),
        hidden_state=(hidden, second_hidden),
        anchor_gate=torch.ones_like(disparity_hr_px),
        source_valid_mask=torch.ones(1, 3, 2, 3, dtype=torch.bool),
    )
    return output, hidden


def test_corrected_v3_dual_k_b_hidden_transport_and_fractional_phase() -> None:
    output, hidden = _single_lr_source_model_output()
    # Corrected LR intrinsics are:
    #   source fx=2,cx=.5; target fx=4,cx=.5.
    # Source u=1 therefore projects to target u=1.5 at identity pose. The
    # bilinear candidates land at target u={1,2} with phases {+.5,-.5}.
    K_source_hr = torch.tensor(
        [[[4.0, 0.0, 1.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    K_target_hr = torch.tensor(
        [[[8.0, 0.0, 1.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    memory = [
        train.TemporalMemoryEntry(
            output=output,
            rgb_hr=torch.zeros(1, 3, 4, 6),
            time_index=0,
            intrinsics_hr=K_source_hr,
            baseline_m=torch.tensor([0.1]),
        )
    ]
    contract = train.TemporalHistoryV2(
        enabled=True,
        top_k=4,
        memory_frames=2,
        splat_footprint="bilinear",
        depth_temperature_m=0.25,
        age_temperature_frames=1.0,
        source_collision_penalty=0.5,
        candidate_feature_channels=32,
    )
    transport = train.build_topk_temporal_transport(
        memory=memory,
        current_time_index=1,
        current_rgb_hr=torch.zeros(1, 3, 4, 6),
        current_ffs_disparity_hr_px=torch.full((1, 1, 2, 3), 16.0),
        current_ffs_confidence=torch.zeros(1, 1, 2, 3),
        intrinsics_current_hr=K_target_hr,
        baseline_current_m=torch.tensor([0.2]),
        temporal_extrinsics_camera_from_world=_identity_window_extrinsics(),
        temporal_pose_valid=torch.tensor([True]),
        contract=contract,
        scale=2,
        align_corners_false_pixel_centers=True,
        reject_conflict_hr_px=100.0,
        geometry_threshold_hr_px=100.0,
    )

    assert transport.topk_valid_mask is not None
    assert transport.topk_disparity_history_hr_px is not None
    assert transport.topk_fractional_offset_px is not None
    valid = transport.topk_valid_mask[0, 0, 0]
    torch.testing.assert_close(valid, torch.tensor([False, True, True]))
    torch.testing.assert_close(
        transport.topk_disparity_history_hr_px[0, 0, 0, 1:],
        torch.tensor([16.0, 16.0]),
    )
    torch.testing.assert_close(
        transport.topk_fractional_offset_px[0, 0, 0, 0, 1:],
        torch.tensor([0.5, -0.5]),
    )
    torch.testing.assert_close(
        transport.fractional_offset_px[0, 0, 0, 1:],
        torch.tensor([0.5, -0.5]),
    )

    assert transport.warped_hidden_state is not None
    torch.testing.assert_close(
        transport.warped_hidden_state[0][0, 0, 0, 1:],
        torch.tensor([1.0, 1.0]),
    )
    sum(state.sum() for state in transport.warped_hidden_state).backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())
    assert hidden.grad[0, 0, 0, 1] > 0


def test_v3_identity_hidden_transport_has_zero_lr_fractional_phase() -> None:
    output, _ = _single_lr_source_model_output()
    K_hr = torch.tensor(
        [[[4.0, 0.0, 1.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    contract = train.TemporalHistoryV2(
        enabled=True,
        top_k=2,
        memory_frames=1,
        splat_footprint="nearest",
        candidate_feature_channels=32,
    )
    transport = train.build_topk_temporal_transport(
        memory=[
            train.TemporalMemoryEntry(
                output=output,
                rgb_hr=torch.zeros(1, 3, 4, 6),
                time_index=0,
                intrinsics_hr=K_hr,
                baseline_m=torch.tensor([0.1]),
            )
        ],
        current_time_index=1,
        current_rgb_hr=torch.zeros(1, 3, 4, 6),
        current_ffs_disparity_hr_px=torch.full((1, 1, 2, 3), 4.0),
        current_ffs_confidence=torch.zeros(1, 1, 2, 3),
        intrinsics_current_hr=K_hr,
        baseline_current_m=torch.tensor([0.1]),
        temporal_extrinsics_camera_from_world=_identity_window_extrinsics(),
        temporal_pose_valid=torch.tensor([True]),
        contract=contract,
        scale=2,
        align_corners_false_pixel_centers=True,
        reject_conflict_hr_px=100.0,
        geometry_threshold_hr_px=100.0,
    )
    assert transport.topk_fractional_offset_px is not None
    assert transport.topk_valid_mask is not None
    phase = transport.topk_fractional_offset_px[
        transport.topk_valid_mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)
    ]
    torch.testing.assert_close(phase, torch.zeros_like(phase), atol=0.0, rtol=0.0)
