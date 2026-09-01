from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

import train
from losses.disparity import charbonnier
from losses.temporal import temporal_residual_consistency_loss
from models.ffs_omega_tsr import FFSOmegaTSR, ModelOutput


_HEIGHT_HR = 4
_WIDTH_HR = 6
_HEIGHT_LR = _HEIGHT_HR // 2
_WIDTH_LR = _WIDTH_HR // 2


def _identity_extrinsics(batch_size: int = 1) -> torch.Tensor:
    extrinsics = torch.zeros(batch_size, 10, 3, 4)
    extrinsics[:, :, :3, :3] = torch.eye(3)
    return extrinsics


def _intrinsics_hr(batch_size: int = 1) -> torch.Tensor:
    intrinsics = torch.tensor(
        [[10.0, 0.0, 2.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]
    )
    return intrinsics.unsqueeze(0).repeat(batch_size, 1, 1)


def _history_contract() -> train.TemporalHistoryV2:
    return train.TemporalHistoryV2(
        enabled=True,
        top_k=4,
        memory_frames=2,
        splat_footprint="nearest",
        depth_temperature_m=0.25,
        age_temperature_frames=1.0,
        source_collision_penalty=0.5,
        candidate_feature_channels=32,
    )


def _model_output(
    *,
    hidden_layer_values: tuple[float, float],
) -> tuple[ModelOutput, tuple[torch.Tensor, torch.Tensor]]:
    disparity = torch.full((1, 1, _HEIGHT_HR, _WIDTH_HR), 4.0)
    log_variance = torch.zeros_like(disparity)
    hidden_state = tuple(
        torch.full(
            (1, 2, _HEIGHT_LR, _WIDTH_LR),
            value,
            requires_grad=True,
        )
        for value in hidden_layer_values
    )
    output = ModelOutput(
        disparity_hr_px=disparity,
        disparity_raw_hr_px=disparity,
        source_weights=torch.full(
            (1, 3, _HEIGHT_LR, _WIDTH_LR), 1.0 / 3.0
        ),
        log_variance=log_variance,
        uncertainty=torch.ones_like(disparity),
        hidden_state=hidden_state,
        anchor_gate=torch.ones_like(disparity),
        source_valid_mask=torch.ones(
            1, 3, _HEIGHT_LR, _WIDTH_LR, dtype=torch.bool
        ),
    )
    return output, hidden_state


def _memory() -> tuple[list[train.TemporalMemoryEntry], tuple[torch.Tensor, ...]]:
    age_two_output, age_two_hidden = _model_output(
        hidden_layer_values=(20.0, 40.0)
    )
    age_one_output, age_one_hidden = _model_output(
        hidden_layer_values=(10.0, 30.0)
    )
    memory = [
        train.TemporalMemoryEntry(
            output=age_two_output,
            rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
            time_index=0,
        ),
        train.TemporalMemoryEntry(
            output=age_one_output,
            rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
            time_index=1,
        ),
    ]
    return memory, (*age_one_hidden, *age_two_hidden)


def _topk_transport(*, pose_valid: bool) -> train.TemporalTransport:
    memory, _ = _memory()
    return train.build_topk_temporal_transport(
        memory=memory,
        current_time_index=2,
        current_rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
        current_ffs_disparity_hr_px=torch.full(
            (1, 1, _HEIGHT_LR, _WIDTH_LR), 4.0
        ),
        current_ffs_confidence=torch.full(
            (1, 1, _HEIGHT_LR, _WIDTH_LR), 0.5
        ),
        intrinsics_current_hr=_intrinsics_hr(),
        baseline_current_m=torch.tensor([0.1]),
        temporal_extrinsics_camera_from_world=_identity_extrinsics(),
        temporal_pose_valid=torch.tensor([pose_valid]),
        contract=_history_contract(),
        scale=2,
    )


def test_topk_identity_transport_keeps_age_phase_and_warps_hidden_state() -> None:
    memory, hidden_inputs = _memory()
    transport = train.build_topk_temporal_transport(
        memory=memory,
        current_time_index=2,
        current_rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
        current_ffs_disparity_hr_px=torch.full(
            (1, 1, _HEIGHT_LR, _WIDTH_LR), 4.0
        ),
        current_ffs_confidence=torch.full(
            (1, 1, _HEIGHT_LR, _WIDTH_LR), 0.5
        ),
        intrinsics_current_hr=_intrinsics_hr(),
        baseline_current_m=torch.tensor([0.1]),
        temporal_extrinsics_camera_from_world=_identity_extrinsics(),
        temporal_pose_valid=torch.tensor([True]),
        contract=_history_contract(),
        scale=2,
    )

    assert transport.topk_disparity_history_hr_px is not None
    assert transport.topk_confidence_history is not None
    assert transport.topk_fractional_offset_px is not None
    assert transport.topk_temporal_age_frames is not None
    assert transport.topk_z_aware_weights is not None
    assert transport.topk_valid_mask is not None
    assert transport.warped_hidden_state is not None
    assert transport.topk_disparity_history_hr_px.shape == (
        1,
        4,
        _HEIGHT_LR,
        _WIDTH_LR,
    )
    assert transport.topk_fractional_offset_px.shape == (
        1,
        4,
        2,
        _HEIGHT_LR,
        _WIDTH_LR,
    )
    assert transport.topk_valid_mask.sum(dim=1).eq(2).all()
    observed_ages = torch.unique(
        transport.topk_temporal_age_frames[transport.topk_valid_mask]
    )
    torch.testing.assert_close(observed_ages, torch.tensor([1.0, 2.0]))
    assert transport.topk_fractional_offset_px[
        transport.topk_valid_mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)
    ].eq(0).all()
    torch.testing.assert_close(
        transport.topk_z_aware_weights.sum(dim=1),
        torch.ones(1, _HEIGHT_LR, _WIDTH_LR),
    )
    assert bool(transport.valid_history.all())
    assert bool(transport.visibility_mask.all())

    age_one_weight = math.exp(-1.0) / (math.exp(-1.0) + math.exp(-2.0))
    age_two_weight = 1.0 - age_one_weight
    expected_layer_values = (
        age_one_weight * 10.0 + age_two_weight * 20.0,
        age_one_weight * 30.0 + age_two_weight * 40.0,
    )
    assert len(transport.warped_hidden_state) == 2
    for state, expected in zip(
        transport.warped_hidden_state, expected_layer_values, strict=True
    ):
        assert state.shape == (1, 2, _HEIGHT_LR, _WIDTH_LR)
        torch.testing.assert_close(state, torch.full_like(state, expected))

    sum(state.sum() for state in transport.warped_hidden_state).backward()
    assert all(value.grad is not None for value in hidden_inputs)
    assert all(bool(torch.isfinite(value.grad).all()) for value in hidden_inputs)


def test_pose_invalid_topk_transport_is_exact_zero_including_hidden_state() -> None:
    transport = _topk_transport(pose_valid=False)
    assert transport.warped_hidden_state is not None
    assert transport.topk_valid_mask is not None
    assert not bool(transport.valid_history.any())
    assert not bool(transport.visibility_mask.any())
    assert not bool(transport.topk_valid_mask.any())
    for value in (
        transport.disparity_history_hr_px,
        transport.confidence_history,
        transport.fractional_offset_px,
        transport.topk_disparity_history_hr_px,
        transport.topk_confidence_history,
        transport.topk_fractional_offset_px,
        transport.topk_temporal_age_frames,
        transport.topk_z_aware_weights,
        *transport.warped_hidden_state,
    ):
        assert value is not None
        assert value.eq(0).all()


def _topk_model_inputs() -> dict[str, torch.Tensor]:
    scalar_shape = (1, 4, _HEIGHT_LR, _WIDTH_LR)
    valid = torch.zeros(scalar_shape, dtype=torch.bool)
    valid[:, :2] = True
    weights = torch.zeros(scalar_shape)
    weights[:, 0] = 0.75
    weights[:, 1] = 0.25
    disparity = torch.zeros(scalar_shape)
    disparity[:, 0] = 4.0
    disparity[:, 1] = 5.0
    confidence = valid.to(dtype=torch.float32)
    age = torch.zeros(scalar_shape)
    age[:, 0] = 1.0
    age[:, 1] = 2.0
    return {
        "history_topk_disparity_hr_px": disparity,
        "history_topk_confidence": confidence,
        "history_topk_fractional_offset_px": torch.zeros(
            1, 4, 2, _HEIGHT_LR, _WIDTH_LR
        ),
        "history_topk_age_frames": age,
        "history_topk_weights": weights,
        "history_topk_valid_mask": valid,
    }


def test_topk_model_is_under_budget_and_legacy_model_rejects_candidates() -> None:
    model = FFSOmegaTSR(temporal_history_top_k=4).eval()
    assert 0 < model.trainable_parameter_count < 12_000_000
    rgb_hr = torch.rand(1, 3, _HEIGHT_HR, _WIDTH_HR)
    disparity_ffs_hr_px = torch.full(
        (1, 1, _HEIGHT_LR, _WIDTH_LR), 4.0
    )
    confidence_ffs = torch.ones_like(disparity_ffs_hr_px)
    candidates = _topk_model_inputs()
    with torch.no_grad():
        output = model(
            rgb_hr,
            disparity_ffs_hr_px,
            confidence_ffs,
            **candidates,
        )
    assert output.disparity_hr_px.shape == (1, 1, _HEIGHT_HR, _WIDTH_HR)
    assert output.history_topk_effective_weights is not None
    assert output.history_topk_valid_mask is not None
    torch.testing.assert_close(
        output.history_topk_effective_weights.sum(dim=1),
        torch.ones(1, _HEIGHT_LR, _WIDTH_LR),
    )
    assert output.history_topk_valid_mask.sum(dim=1).eq(2).all()

    legacy_model = FFSOmegaTSR().eval()
    with pytest.raises(ValueError, match="require temporal_history_top_k"):
        legacy_model(
            rgb_hr,
            disparity_ffs_hr_px,
            confidence_ffs,
            **candidates,
        )


def test_reference_identity_warp_and_teacher_temporal_residual_loss() -> None:
    previous_teacher = torch.full((1, 1, _HEIGHT_HR, _WIDTH_HR), 5.0)
    reference_warp = train.build_reference_temporal_warp(
        previous_reference_disparity_hr_px=previous_teacher,
        previous_reference_confidence=torch.ones_like(previous_teacher),
        previous_reference_valid_mask=torch.ones_like(
            previous_teacher, dtype=torch.bool
        ),
        intrinsics_current_hr=_intrinsics_hr(),
        baseline_current_m=torch.tensor([0.1]),
        temporal_extrinsics_camera_from_world=_identity_extrinsics(),
        temporal_pose_valid=torch.tensor([True]),
        contract=_history_contract(),
    )
    torch.testing.assert_close(reference_warp.disparity_hr_px, previous_teacher)
    assert bool(reference_warp.valid_mask_hr.all())
    assert bool(reference_warp.visibility_mask_hr.all())
    assert not bool(reference_warp.collision_mask_hr.any())

    current_teacher = previous_teacher + 1.0
    warped_previous_prediction = previous_teacher + 2.0
    current_prediction_same_bias = current_teacher + 2.0
    common = {
        "static_mask": torch.ones_like(previous_teacher, dtype=torch.bool),
        "visibility_mask": reference_warp.visibility_mask_hr,
        "collision_mask": reference_warp.collision_mask_hr,
        "photometric_residual": torch.zeros_like(previous_teacher),
        "max_photometric_residual": 0.1,
        "geometry_consistent_mask": torch.ones_like(
            previous_teacher, dtype=torch.bool
        ),
        "current_reference_valid_mask": torch.ones_like(
            previous_teacher, dtype=torch.bool
        ),
        "warped_previous_reference_valid_mask": reference_warp.valid_mask_hr,
    }
    cancelled = temporal_residual_consistency_loss(
        current_prediction_same_bias,
        warped_previous_prediction,
        current_teacher,
        reference_warp.disparity_hr_px,
        **common,
    )
    assert cancelled.item() == 0.0

    one_pixel_temporal_drift = temporal_residual_consistency_loss(
        current_prediction_same_bias + 1.0,
        warped_previous_prediction,
        current_teacher,
        reference_warp.disparity_hr_px,
        **common,
    )
    torch.testing.assert_close(
        one_pixel_temporal_drift,
        charbonnier(torch.tensor(1.0)),
    )


def _history_v2_config_section() -> dict[str, object]:
    return {
        "enabled": True,
        "protocol_version": train.TEMPORAL_HISTORY_V2_PROTOCOL,
        "top_k": 4,
        "memory_frames": 2,
        "splat_footprint": "nearest",
        "depth_temperature_m": 0.25,
        "age_temperature_frames": 3.0,
        "source_collision_penalty": 0.5,
        "candidate_feature_channels": 32,
    }


def test_temporal_v2_config_parser_fails_closed() -> None:
    malformed = {"temporal_history_v2": _history_v2_config_section()}
    malformed["temporal_history_v2"].pop("protocol_version")
    with pytest.raises(ValueError, match="protocol version mismatch"):
        train.temporal_history_v2_from_config(malformed)

    half_enabled = OmegaConf.create(train.DEFAULT_CONFIG)
    half_enabled.temporal_history_v2 = _history_v2_config_section()
    with pytest.raises(ValueError, match="must be enabled together"):
        train._validate_common_training_config(half_enabled, total_steps=1)

    invalid_k = OmegaConf.create(train.DEFAULT_CONFIG)
    invalid_k.temporal_history_v2 = {
        **_history_v2_config_section(),
        "top_k": 1,
    }
    invalid_k.temporal_residual_v2 = {
        "enabled": True,
        "protocol_version": train.TEMPORAL_RESIDUAL_V2_PROTOCOL,
        "reference": "teacher",
    }
    with pytest.raises(ValueError, match="at least K=2"):
        train._validate_common_training_config(invalid_k, total_steps=1)
