from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import eval as eval_cli
import train
from models.ffs_omega_tsr import ModelOutput
from data.temporal_training_dataset import _continuous_pose_quality_score


_HEIGHT_HR = 8
_WIDTH_HR = 12
_HEIGHT_LR = _HEIGHT_HR // 2
_WIDTH_LR = _WIDTH_HR // 2
_TOP_K = 4
_LEGACY_V3_STATE_KEY_SHA256 = (
    "05a3ee28aa80bf88a821ceb92019101a0cbea1570085d0e06d55662f7c0d68bb"
)
_V31_PARAMETER_COUNT = 1_778_244
_V31_STATE_KEY_COUNT = 106
_V31_STATE_KEY_SHA256 = (
    "0dcdd40ba461b14d58278711d94c586e169e20848e3262691f7957d0318a6cc7"
)


def _resolved(path: str):
    return train.resolve_config(
        path,
        ["data.calibration_sidecar_path=/tmp/v31-calibration.jsonl"],
    )


def _intrinsics_hr() -> torch.Tensor:
    return torch.tensor(
        [[[20.0, 0.0, 5.5], [0.0, 20.0, 3.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


def _stereo_transform() -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    transform[:, 0, 3] = -0.1
    return transform


def _identity_temporal_transforms() -> torch.Tensor:
    return (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 4, 4)
        .repeat(1, 2, 1, 1)
    )


def _identity_vggt_extrinsics() -> torch.Tensor:
    extrinsics = torch.zeros(1, 10, 3, 4, dtype=torch.float32)
    extrinsics[:, :, :3, :3] = torch.eye(3, dtype=torch.float32)
    return extrinsics


def _model_output(
    hidden_values: tuple[float, float],
) -> tuple[ModelOutput, tuple[torch.Tensor, torch.Tensor]]:
    disparity = torch.full((1, 1, _HEIGHT_HR, _WIDTH_HR), 4.0)
    hidden_state = tuple(
        torch.full(
            (1, 96, _HEIGHT_LR, _WIDTH_LR),
            value,
            requires_grad=True,
        )
        for value in hidden_values
    )
    output = ModelOutput(
        disparity_hr_px=disparity,
        disparity_raw_hr_px=disparity,
        source_weights=torch.full(
            (1, 3, _HEIGHT_LR, _WIDTH_LR), 1.0 / 3.0
        ),
        log_variance=torch.zeros_like(disparity),
        uncertainty=torch.ones_like(disparity),
        hidden_state=hidden_state,
        anchor_gate=torch.ones_like(disparity),
        source_valid_mask=torch.ones(
            1, 3, _HEIGHT_LR, _WIDTH_LR, dtype=torch.bool
        ),
        history_value_feature=hidden_state[-1][:, :8],
    )
    return output, hidden_state


def _temporal_memory() -> tuple[
    list[train.TemporalMemoryEntry],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    age_two_output, age_two_hidden = _model_output((20.0, 21.0))
    age_one_output, age_one_hidden = _model_output((10.0, 11.0))
    memory = [
        train.TemporalMemoryEntry(
            output=age_two_output,
            rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
            time_index=0,
            intrinsics_hr=_intrinsics_hr(),
            baseline_m=torch.tensor([0.1]),
        ),
        train.TemporalMemoryEntry(
            output=age_one_output,
            rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
            time_index=1,
            intrinsics_hr=_intrinsics_hr(),
            baseline_m=torch.tensor([0.1]),
        ),
    ]
    return memory, age_one_hidden, age_two_hidden


def _calibration_kwargs() -> dict[str, torch.Tensor]:
    return {
        "K_left_hr_px": _intrinsics_hr(),
        "baseline_m": torch.tensor([0.1], dtype=torch.float32),
        "T_right_rectified_from_left_rectified_m": _stereo_transform(),
        "T_current_from_history_m": _identity_temporal_transforms(),
        "temporal_pose_valid": torch.ones(1, 2, dtype=torch.bool),
    }


def _build_v31_transport() -> tuple[
    object,
    train.TemporalTransport,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    config = _resolved("configs/temporal_x2_v3_1.yaml")
    memory, age_one_hidden, age_two_hidden = _temporal_memory()
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
        temporal_extrinsics_camera_from_world=_identity_vggt_extrinsics(),
        temporal_pose_valid=torch.tensor([True]),
        contract=train.temporal_history_v2_from_config(config),
        temporal_pose_quality_score=torch.tensor([0.75]),
        candidate_contract=train.temporal_candidate_fusion_v3_1_from_config(
            config
        ),
        scale=2,
        align_corners_false_pixel_centers=True,
    )
    return config, transport, age_one_hidden, age_two_hidden


def _candidate_model_inputs() -> dict[str, torch.Tensor]:
    scalar_shape = (1, _TOP_K, _HEIGHT_LR, _WIDTH_LR)
    valid = torch.zeros(scalar_shape, dtype=torch.bool)
    valid[:, :2] = True
    front = torch.zeros_like(valid)
    front[:, 0] = True
    context = torch.zeros_like(valid)
    context[:, 1] = True
    weights = torch.zeros(scalar_shape)
    weights[:, 0] = 0.25
    weights[:, 1] = 0.75
    disparity = torch.zeros(scalar_shape)
    disparity[:, 0] = 4.0
    disparity[:, 1] = 40.0
    depth = torch.zeros(scalar_shape)
    depth[:, 0] = 1.0
    depth[:, 1] = 2.0
    age = torch.zeros(scalar_shape)
    age[:, 0] = 1.0
    age[:, 1] = 2.0
    depth_layer = torch.full(scalar_shape, -1, dtype=torch.int64)
    depth_layer[:, 0] = 0
    depth_layer[:, 1] = 1
    return {
        "disparity_history_hr_px": torch.full(
            (1, 1, _HEIGHT_LR, _WIDTH_LR), 40.0
        ),
        "confidence_history": torch.ones(
            1, 1, _HEIGHT_LR, _WIDTH_LR
        ),
        "history_visibility": torch.ones(
            1, 1, _HEIGHT_LR, _WIDTH_LR
        ),
        "valid_history": torch.ones(
            1, 1, _HEIGHT_LR, _WIDTH_LR, dtype=torch.bool
        ),
        "history_topk_disparity_hr_px": disparity,
        "history_topk_confidence": valid.float(),
        "history_topk_fractional_offset_px": torch.zeros(
            1, _TOP_K, 2, _HEIGHT_LR, _WIDTH_LR
        ),
        "history_topk_age_frames": age,
        "history_topk_weights": weights,
        "history_topk_valid_mask": valid,
        "history_topk_depth_m": depth,
        "history_topk_pose_quality": valid.float(),
        "history_topk_depth_layer_index": depth_layer,
        "history_topk_front_surface_mask": front,
        "history_topk_context_only_mask": context,
        "history_topk_warped_hidden_feature": torch.zeros(
            1, _TOP_K, 8, _HEIGHT_LR, _WIDTH_LR
        ),
    }


def _state_key_sha256(model: torch.nn.Module) -> str:
    return hashlib.sha256("\n".join(model.state_dict()).encode()).hexdigest()


def test_v31_configs_enable_all_protocols_and_keep_legacy_v3_keys() -> None:
    v31_models = []
    for path in (
        "configs/mvp_x2_v3_1.yaml",
        "configs/temporal_x2_v3_1.yaml",
    ):
        config = _resolved(path)
        calibration = train.calibration_conditioning_v3_from_config(config)
        measurement = train.measurement_ownership_v3_1_from_config(config)
        candidates = train.temporal_candidate_fusion_v3_1_from_config(config)
        assert calibration.enabled
        assert calibration.align_corners_false_pixel_centers
        assert measurement.enabled
        assert candidates.enabled
        model = train.build_model(config)
        assert model.align_corners_false_pixel_centers
        assert model.measurement_ownership_v3_1
        assert model.current_conditioned_history_v3_1
        assert model.current_conditioned_history is not None
        assert model.trainable_parameter_count == _V31_PARAMETER_COUNT
        assert len(model.state_dict()) == _V31_STATE_KEY_COUNT
        assert _state_key_sha256(model) == _V31_STATE_KEY_SHA256
        assert model.trainable_parameter_count < 12_000_000
        v31_models.append(model)
    assert tuple(v31_models[0].state_dict()) == tuple(v31_models[1].state_dict())
    assert (
        v31_models[0].trainable_parameter_count
        == v31_models[1].trainable_parameter_count
    )

    for path in ("configs/mvp_x2_v3.yaml", "configs/temporal_x2_v3.yaml"):
        config = _resolved(path)
        calibration = train.calibration_conditioning_v3_from_config(config)
        assert calibration.enabled
        assert not calibration.align_corners_false_pixel_centers
        assert not train.measurement_ownership_v3_1_from_config(config).enabled
        assert not train.temporal_candidate_fusion_v3_1_from_config(config).enabled
        model = train.build_model(config)
        assert not model.measurement_ownership_v3_1
        assert model.current_conditioned_history is None
        assert len(model.state_dict()) == 92
        assert _state_key_sha256(model) == _LEGACY_V3_STATE_KEY_SHA256


def test_calibrated_pose_quality_is_continuous_and_rejected_pose_is_zero() -> None:
    thresholds = {
        "max_baseline_cv": 0.10,
        "max_stereo_rotation_error_deg": 5.0,
        "max_photometric_median_absolute_rgb": 0.12,
        "max_depth_weighted_mae_hr_px": 2.0,
        "max_depth_median_absolute_error_hr_px": 2.0,
    }
    quality = {
        "baseline": {
            "baseline_coefficient_of_variation": 0.05,
            "stereo_rotation_error_max_deg": 1.0,
        },
        "photometric": {"median_absolute_rgb_residual": 0.03},
        "depth_consistency": {
            "weighted_mae_hr_px": 0.5,
            "median_absolute_error_hr_px": 1.0,
        },
    }
    score = _continuous_pose_quality_score(
        quality,
        pose_valid=True,
        derived_contract="calibrated_stereo_v2",
        thresholds=thresholds,
        cache_path=Path("quality.pt"),
    )
    expected_ratios = (0.5, 0.2, 0.25, 0.25, 0.5)
    assert score == pytest.approx(
        math.exp(-sum(expected_ratios) / len(expected_ratios))
    )
    assert 0.0 < score < 1.0
    assert _continuous_pose_quality_score(
        {},
        pose_valid=False,
        derived_contract="calibrated_stereo_v2",
        thresholds=thresholds,
        cache_path=Path("rejected.pt"),
    ) == 0.0


def test_stage_a_v31_forward_is_finite_with_no_temporal_candidates() -> None:
    config = _resolved("configs/mvp_x2_v3_1.yaml")
    model = train.build_model(config).eval()
    rgb_hr = torch.rand(1, 3, _HEIGHT_HR, _WIDTH_HR)
    disparity_ffs = torch.full((1, 1, _HEIGHT_LR, _WIDTH_LR), 4.0)
    confidence_ffs = torch.ones_like(disparity_ffs)
    spatial_batch: Mapping[str, torch.Tensor] = {
        "rgb_hr": rgb_hr,
        "K_hr": _intrinsics_hr(),
        "baseline_m": torch.tensor([0.1]),
        "T_right_rectified_from_left_rectified_m": _stereo_transform(),
    }
    calibration_kwargs = train.calibration_model_kwargs_spatial(
        spatial_batch,
        train.calibration_conditioning_v3_from_config(config),
    )

    with torch.no_grad():
        output = model(
            rgb_hr,
            disparity_ffs,
            confidence_ffs,
            valid_ffs=torch.ones_like(disparity_ffs, dtype=torch.bool),
            **calibration_kwargs,
        )

    assert torch.isfinite(output.disparity_hr_px).all()
    assert output.history_topk_effective_weights is not None
    assert output.history_topk_valid_mask is not None
    assert output.history_topk_context_weights is not None
    assert output.history_metric_disparity_hr_px is not None
    assert output.history_metric_confidence is not None
    assert output.history_metric_valid_mask is not None
    assert output.history_topk_effective_weights.eq(0).all()
    assert not output.history_topk_valid_mask.any()
    assert output.history_topk_context_weights.eq(0).all()
    assert output.history_metric_disparity_hr_px.eq(0).all()
    assert output.history_metric_confidence.eq(0).all()
    assert not output.history_metric_valid_mask.any()


def test_v31_two_age_transport_populates_candidate_contract_and_hidden_gradients() -> None:
    _, transport, age_one_hidden, age_two_hidden = _build_v31_transport()

    eval_kwargs = eval_cli._history_model_kwargs(
        transport,
        rgb_dtype=torch.float32,
        temporal_v2=True,
        temporal_candidate_v31=True,
    )
    assert eval_kwargs["history_topk_depth_m"] is transport.topk_depth_m
    assert (
        eval_kwargs["history_topk_depth_layer_index"]
        is transport.topk_depth_layer_index
    )
    assert (
        eval_kwargs["history_topk_front_surface_mask"]
        is transport.topk_front_surface_mask
    )
    assert (
        eval_kwargs["history_topk_context_only_mask"]
        is transport.topk_context_only_mask
    )
    assert (
        eval_kwargs["history_topk_warped_hidden_feature"]
        is transport.topk_warped_hidden_feature
    )

    assert transport.topk_valid_mask is not None
    expected_scalar = (1, _TOP_K, _HEIGHT_LR, _WIDTH_LR)
    for value in (
        transport.topk_depth_m,
        transport.topk_depth_layer_index,
        transport.topk_front_surface_mask,
        transport.topk_context_only_mask,
    ):
        assert value is not None and value.shape == expected_scalar
    assert transport.topk_age2_depth_consistent_available_mask is not None
    assert transport.topk_age2_depth_consistent_available_mask.shape == (
        1,
        1,
        _HEIGHT_LR,
        _WIDTH_LR,
    )
    assert transport.topk_age2_depth_consistent_available_mask.all()
    assert transport.topk_warped_hidden_feature is not None
    assert transport.topk_warped_hidden_feature.shape == (
        1,
        _TOP_K,
        8,
        _HEIGHT_LR,
        _WIDTH_LR,
    )
    assert transport.topk_warped_hidden_feature.requires_grad
    assert transport.topk_temporal_age_frames is not None
    observed_ages = torch.unique(
        transport.topk_temporal_age_frames[transport.topk_valid_mask]
    )
    torch.testing.assert_close(observed_ages, torch.tensor([1.0, 2.0]))
    assert transport.topk_front_surface_mask is not None
    assert (
        transport.topk_front_surface_mask[transport.topk_valid_mask]
    ).all()
    assert transport.topk_context_only_mask is not None
    assert not transport.topk_context_only_mask.any()

    assert transport.warped_hidden_state is not None
    hidden_loss = transport.topk_warped_hidden_feature.sum() + sum(
        state.sum() for state in transport.warped_hidden_state
    )
    hidden_loss.backward()
    assert all(state.grad is not None for state in age_one_hidden)
    assert age_two_hidden[-1].grad is not None
    for gradient in (
        age_one_hidden[0].grad,
        age_one_hidden[1].grad,
        age_two_hidden[-1].grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    # Only each memory entry's final layer is carried as an attention value;
    # age two must never initialize the recurrent state directly.
    assert age_two_hidden[0].grad is None


def test_eval_v31_transport_kwargs_model_forward_and_diagnostics_are_bound() -> None:
    config, transport, _, _ = _build_v31_transport()
    model = train.build_model(config).eval()
    rgb_hr = torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR)
    disparity_ffs = torch.full((1, 1, _HEIGHT_LR, _WIDTH_LR), 4.0)
    confidence_ffs = torch.full_like(disparity_ffs, 0.5)
    history_kwargs = eval_cli._history_model_kwargs(
        transport,
        rgb_dtype=rgb_hr.dtype,
        temporal_v2=True,
        temporal_candidate_v31=True,
    )

    with torch.no_grad():
        output = model(
            rgb_hr,
            disparity_ffs,
            confidence_ffs,
            valid_ffs=torch.ones_like(disparity_ffs, dtype=torch.bool),
            hidden_state=transport.warped_hidden_state,
            **history_kwargs,
            **_calibration_kwargs(),
        )
    diagnostics = eval_cli._topk_v31_diagnostics(
        transport,
        output,
        teacher_disparity_hr_px=torch.full(
            (1, 1, _HEIGHT_HR, _WIDTH_HR), 4.0
        ),
        teacher_trusted_mask_hr=torch.ones(
            1, 1, _HEIGHT_HR, _WIDTH_HR, dtype=torch.bool
        ),
        scale=2,
    )

    assert set(diagnostics) == set(eval_cli.TOPK_V31_DIAGNOSTIC_NAMES)
    assert all(metric.valid for metric in diagnostics.values())
    assert diagnostics["unique_age_fraction"].value == pytest.approx(1.0)
    assert diagnostics["age2_survival_rate"].value == pytest.approx(1.0)
    assert diagnostics["rank0_disparity_epe_hr_px"].value == pytest.approx(0.0)
    assert diagnostics["weighted_disparity_epe_hr_px"].value == pytest.approx(
        0.0
    )


def test_topk_diagnostics_separate_retention_prior_and_learned_attention() -> None:
    batch, candidates, height, width = 1, 4, 1, 1
    valid = torch.ones(batch, candidates, height, width, dtype=torch.bool)
    phases = torch.zeros(batch, candidates, 2, height, width)
    phases[:, :, 0] = torch.tensor([0.0, 0.5, 0.0, 0.5]).reshape(
        batch, candidates, height, width
    )
    ages = torch.tensor([1.0, 1.0, 2.0, 2.0]).reshape(
        batch, candidates, height, width
    )
    disparities = torch.tensor([4.0, 6.0, 4.0, 6.0]).reshape(
        batch, candidates, height, width
    )
    uniform = torch.full_like(disparities, 0.25)
    one_hot = torch.zeros_like(disparities)
    one_hot[:, 0] = 1.0
    transport = SimpleNamespace(
        topk_disparity_history_hr_px=disparities,
        topk_depth_m=torch.tensor([1.0, 1.1, 1.2, 1.3]).reshape_as(disparities),
        topk_fractional_offset_px=phases,
        topk_temporal_age_frames=ages,
        topk_valid_mask=valid,
        topk_metric_prior_weights=uniform,
        topk_front_surface_mask=valid,
        topk_age2_depth_consistent_available_mask=torch.ones(
            batch, 1, height, width, dtype=torch.bool
        ),
    )
    output = SimpleNamespace(
        history_topk_context_weights=one_hot,
        history_topk_effective_weights=one_hot,
        history_metric_disparity_hr_px=torch.full(
            (batch, 1, height, width), 4.0
        ),
        history_metric_valid_mask=torch.ones(
            batch, 1, height, width, dtype=torch.bool
        ),
    )
    diagnostics = eval_cli._topk_v31_diagnostics(
        transport,
        output,
        teacher_disparity_hr_px=torch.full((batch, 1, 2, 2), 4.0),
        teacher_trusted_mask_hr=torch.ones(
            batch, 1, 2, 2, dtype=torch.bool
        ),
        scale=2,
    )
    assert diagnostics["fractional_phase_variance"].value == pytest.approx(0.5)
    assert diagnostics["attended_fractional_phase_variance"].value == pytest.approx(
        0.0
    )
    assert diagnostics["topk_weight_entropy"].value == pytest.approx(math.log(4.0))
    assert diagnostics["context_attention_weight_entropy"].value == pytest.approx(
        0.0
    )
    assert diagnostics["weighted_disparity_epe_hr_px"].value == pytest.approx(1.0)
    assert diagnostics["attention_weighted_disparity_epe_hr_px"].value == pytest.approx(
        0.0
    )
    assert diagnostics["age2_survival_rate"].value == pytest.approx(1.0)

    output.history_topk_context_weights = torch.zeros_like(one_hot)
    zero_context = eval_cli._topk_v31_diagnostics(
        transport,
        output,
        teacher_disparity_hr_px=torch.full((batch, 1, 2, 2), 4.0),
        teacher_trusted_mask_hr=torch.ones(
            batch, 1, 2, 2, dtype=torch.bool
        ),
        scale=2,
    )
    assert not zero_context["attended_fractional_phase_variance"].valid
    assert not zero_context["context_attention_weight_entropy"].valid


def test_model_uses_front_surface_proposal_and_back_layer_only_as_context() -> None:
    config = _resolved("configs/temporal_x2_v3_1.yaml")
    model = train.build_model(config).eval()
    rgb_hr = torch.rand(1, 3, _HEIGHT_HR, _WIDTH_HR)
    disparity_ffs = torch.full((1, 1, _HEIGHT_LR, _WIDTH_LR), 10.0)
    confidence_ffs = torch.ones_like(disparity_ffs)
    candidate_inputs = _candidate_model_inputs()

    with torch.no_grad():
        output = model(
            rgb_hr,
            disparity_ffs,
            confidence_ffs,
            valid_ffs=torch.ones_like(disparity_ffs, dtype=torch.bool),
            **candidate_inputs,
            **_calibration_kwargs(),
        )

    assert output.history_metric_disparity_hr_px is not None
    assert output.history_metric_confidence is not None
    assert output.history_metric_valid_mask is not None
    torch.testing.assert_close(
        output.history_metric_disparity_hr_px,
        torch.full_like(output.history_metric_disparity_hr_px, 4.0),
    )
    assert output.history_metric_valid_mask.all()
    assert output.history_topk_effective_weights is not None
    assert output.history_topk_effective_weights[:, 1].eq(0).all()
    assert output.history_topk_context_weights is not None
    assert (output.history_topk_context_weights[:, 1] > 0).all()

    assert output.disparity_source_mix_hr_px_lr_grid is not None
    expected_mix = (
        output.source_weights[:, 0:1] * disparity_ffs
        + output.source_weights[:, 2:3]
        * output.history_metric_disparity_hr_px
    )
    torch.testing.assert_close(
        output.disparity_source_mix_hr_px_lr_grid,
        expected_mix,
    )
    coarse_pre_attention_history = candidate_inputs["disparity_history_hr_px"]
    assert not torch.allclose(
        output.disparity_source_mix_hr_px_lr_grid,
        output.source_weights[:, 0:1] * disparity_ffs
        + output.source_weights[:, 2:3] * coarse_pre_attention_history,
    )


def test_v31_config_transport_and_model_inputs_fail_closed() -> None:
    broken_config = OmegaConf.create(
        OmegaConf.to_container(
            _resolved("configs/mvp_x2_v3_1.yaml"), resolve=False
        )
    )
    del broken_config.calibration_conditioning_v3.pixel_center_contract
    with pytest.raises(ValueError, match="corrected half-pixel"):
        train.validate_stage_a_config(broken_config)

    temporal_config = _resolved("configs/temporal_x2_v3_1.yaml")
    with pytest.raises(ValueError, match="align_corners_false pixel centres"):
        train.build_topk_temporal_transport(
            memory=[],
            current_time_index=2,
            current_rgb_hr=torch.zeros(1, 3, _HEIGHT_HR, _WIDTH_HR),
            current_ffs_disparity_hr_px=torch.ones(
                1, 1, _HEIGHT_LR, _WIDTH_LR
            ),
            current_ffs_confidence=torch.ones(
                1, 1, _HEIGHT_LR, _WIDTH_LR
            ),
            intrinsics_current_hr=_intrinsics_hr(),
            baseline_current_m=torch.tensor([0.1]),
            temporal_extrinsics_camera_from_world=_identity_vggt_extrinsics(),
            temporal_pose_valid=torch.tensor([True]),
            contract=train.temporal_history_v2_from_config(temporal_config),
            temporal_pose_quality_score=torch.tensor([0.75]),
            candidate_contract=(
                train.temporal_candidate_fusion_v3_1_from_config(
                    temporal_config
                )
            ),
            align_corners_false_pixel_centers=False,
        )

    model = train.build_model(temporal_config).eval()
    candidate_inputs = _candidate_model_inputs()
    del candidate_inputs["history_topk_warped_hidden_feature"]
    with pytest.raises(ValueError, match="all six v3.1"):
        model(
            torch.rand(1, 3, _HEIGHT_HR, _WIDTH_HR),
            torch.full((1, 1, _HEIGHT_LR, _WIDTH_LR), 10.0),
            torch.ones(1, 1, _HEIGHT_LR, _WIDTH_LR),
            valid_ffs=torch.ones(
                1, 1, _HEIGHT_LR, _WIDTH_LR, dtype=torch.bool
            ),
            **candidate_inputs,
            **_calibration_kwargs(),
        )
