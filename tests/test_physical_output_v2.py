from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf
from torch import nn

import train
from losses import validity_completion_loss
from models.epipolar_refiner import HREpipolarRefiner
from models.ffs_omega_tsr import FFSOmegaTSR


def _tsr_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    return (
        torch.rand(1, 3, 8, 12, generator=generator),
        1.0 + torch.rand(1, 1, 4, 6, generator=generator),
        torch.rand(1, 1, 4, 6, generator=generator),
    )


def test_legacy_default_state_and_forward_are_exactly_unchanged() -> None:
    torch.manual_seed(11)
    legacy = FFSOmegaTSR().eval()
    torch.manual_seed(11)
    explicit = FFSOmegaTSR(physical_output_v2=False).eval()
    assert tuple(legacy.state_dict()) == tuple(explicit.state_dict())
    for name in legacy.state_dict():
        torch.testing.assert_close(
            legacy.state_dict()[name], explicit.state_dict()[name], rtol=0, atol=0
        )
    inputs = _tsr_inputs()
    with torch.no_grad():
        before = legacy(*inputs)
        after = explicit(*inputs)
    for name in (
        "disparity_hr_px",
        "disparity_raw_hr_px",
        "source_weights",
        "log_variance",
        "anchor_gate",
    ):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    assert after.valid_probability is None
    assert after.completion_probability is None
    assert after.output_valid_mask is None


def test_v2_all_invalid_is_exact_zero_and_never_fake_completion() -> None:
    model = FFSOmegaTSR(physical_output_v2=True).eval()
    rgb, disparity, confidence = _tsr_inputs()
    valid = torch.zeros_like(disparity, dtype=torch.bool)
    disparity.fill_(-4.0)
    with torch.no_grad():
        output = model(rgb, disparity, confidence, valid_ffs=valid)
    assert output.valid_probability is not None
    assert output.completion_probability is not None
    assert output.output_valid_mask is not None
    assert output.completion_mask is not None
    torch.testing.assert_close(output.disparity_hr_px, torch.zeros_like(output.disparity_hr_px))
    torch.testing.assert_close(
        output.disparity_raw_hr_px, torch.zeros_like(output.disparity_raw_hr_px)
    )
    torch.testing.assert_close(
        output.valid_probability, torch.zeros_like(output.valid_probability)
    )
    torch.testing.assert_close(
        output.completion_probability, torch.zeros_like(output.completion_probability)
    )
    assert not output.output_valid_mask.any()
    assert not output.completion_mask.any()


def test_v2_nonnegative_zero_invalid_and_trusted_ffs_is_conserved_exactly() -> None:
    model = FFSOmegaTSR(physical_output_v2=True).eval()
    rgb, disparity, confidence = _tsr_inputs()
    confidence.fill_(1.0)
    valid = torch.ones_like(disparity, dtype=torch.bool)
    with torch.no_grad():
        final_layer = model.hr_output_head[-1]
        assert isinstance(final_layer, nn.Conv2d)
        final_layer.bias[0] = -100.0
        output = model(rgb, disparity, confidence, valid_ffs=valid)
    expected = torch.nn.functional.interpolate(
        disparity, scale_factor=2, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(output.disparity_hr_px, expected, rtol=0, atol=0)
    assert output.output_valid_mask is not None
    assert output.output_valid_mask.all()
    assert torch.all(output.disparity_hr_px >= 0)
    assert not torch.any((output.disparity_hr_px == 0) & output.output_valid_mask)


def test_validity_completion_loss_is_soft_empty_safe_and_has_finite_gradients() -> None:
    valid_logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
    completion_logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
    teacher_valid = torch.tensor([[[[True, False, True], [False, True, True]]]])
    teacher_confidence = torch.tensor([[[[1.0, 0.8, 0.3], [0.0, 0.7, 1.0]]]])
    observation_valid = torch.ones_like(teacher_valid)
    loss = validity_completion_loss(
        valid_logits=valid_logits,
        completion_logits=completion_logits,
        valid_probability=torch.sigmoid(valid_logits),
        completion_probability=torch.zeros_like(completion_logits),
        teacher_valid_mask=teacher_valid,
        teacher_confidence=teacher_confidence,
        observation_valid_mask_hr=observation_valid,
    )
    assert loss.valid_pixel_count == 6
    assert loss.completion_pixel_count == 0
    assert loss.completion_bce.item() == 0.0
    total = loss.valid_bce + loss.completion_bce + loss.calibration
    total.backward()
    assert valid_logits.grad is not None and torch.isfinite(valid_logits.grad).all()
    assert completion_logits.grad is not None
    assert torch.count_nonzero(completion_logits.grad) == 0


def test_v2_config_build_and_training_loss_wire_all_three_terms() -> None:
    config = OmegaConf.create(train.DEFAULT_CONFIG)
    config.physical_output_v2 = {
        "enabled": True,
        "protocol_version": "explicit_valid_completion_nonnegative_v2",
        "valid_threshold": 0.5,
        "completion_threshold": 0.5,
        "trusted_ffs_confidence_threshold": 0.8,
        "valid_bce_weight": 0.05,
        "completion_bce_weight": 0.05,
        "calibration_weight": 0.01,
    }
    contract = train.physical_output_v2_from_config(config)
    model = train.build_model(config).train()
    assert model.validity_completion_head is not None
    assert model.trainable_parameter_count > FFSOmegaTSR().trainable_parameter_count
    assert model.trainable_parameter_count < 12_000_000
    rgb, disparity, confidence = _tsr_inputs()
    valid_ffs = torch.ones_like(disparity, dtype=torch.bool)
    valid_ffs[..., 0, 0] = False
    output = model(rgb, disparity, confidence, valid_ffs=valid_ffs)
    target = output.disparity_hr_px.detach().clamp_min(0.25)
    batch = {
        "teacher_disparity_hr_px": target,
        "teacher_confidence": torch.full_like(target, 0.8),
        "teacher_valid_mask": torch.ones_like(target, dtype=torch.bool),
        "teacher_trusted_mask": torch.ones_like(target, dtype=torch.bool),
        "observation_disparity_lr_px": disparity / 2.0,
        "observation_confidence": confidence,
        "observation_trusted_mask": valid_ffs,
        "valid_ffs": valid_ffs,
    }
    breakdown = train.compute_stage_a_loss(
        output, batch, physical_output_v2=contract
    )
    assert breakdown.valid_bce is not None
    assert breakdown.completion_bce is not None
    assert breakdown.validity_calibration is not None
    assert set(breakdown.detached_scalars()).issuperset(
        {"valid_bce", "completion_bce", "validity_calibration"}
    )
    breakdown.total.backward()
    head_gradients = [
        parameter.grad
        for parameter in model.validity_completion_head.parameters()
        if parameter.grad is not None
    ]
    assert head_gradients
    assert all(torch.isfinite(value).all() for value in head_gradients)


def test_v2_config_rejects_legacy_positivity_lineage_mixing() -> None:
    config = OmegaConf.create(train.DEFAULT_CONFIG)
    config.physical_output_v2 = {
        "enabled": True,
        "protocol_version": "explicit_valid_completion_nonnegative_v2",
        "valid_threshold": 0.5,
        "completion_threshold": 0.5,
        "trusted_ffs_confidence_threshold": 0.8,
        "valid_bce_weight": 0.05,
        "completion_bce_weight": 0.05,
        "calibration_weight": 0.01,
    }
    config.positivity_ablation = {
        "enabled": True,
        "sanitize_invalid_sources": True,
        "lower_bound_hr_px": 0.0,
        "lr_negative_penalty_weight": 0.1,
        "raw_negative_penalty_weight": 0.01,
    }
    with pytest.raises(ValueError, match="separate lineages"):
        train.physical_output_v2_from_config(config)


def test_stage_c_v2_step_zero_is_exact_noop_and_state_is_opt_in_only() -> None:
    legacy = HREpipolarRefiner(feature_channels=8, correlation_groups=2, head_channels=12)
    v2 = HREpipolarRefiner(
        feature_channels=8,
        correlation_groups=2,
        head_channels=12,
        base_aware_noop_v2=True,
    ).eval()
    assert not any(name.startswith("no_op_gate_head") for name in legacy.state_dict())
    assert any(name.startswith("no_op_gate_head") for name in v2.state_dict())
    rgb = torch.rand(1, 3, 4, 12)
    base = torch.full((1, 1, 4, 12), 1.25)
    with torch.no_grad():
        output = v2(rgb, rgb.clone(), base)
    torch.testing.assert_close(output.corrected_disparity_hr_px, base, rtol=0, atol=0)
    torch.testing.assert_close(output.correction_hr_px, torch.zeros_like(base), rtol=0, atol=0)
    assert output.no_op_mask is not None and output.no_op_mask.all()
    assert output.output_valid_mask is not None and output.output_valid_mask.all()


def test_stage_c_v2_fp32_lower_bound_and_zero_semantics() -> None:
    model = HREpipolarRefiner(
        feature_channels=8,
        correlation_groups=2,
        head_channels=12,
        base_aware_noop_v2=True,
    ).eval()
    correction_layer = model.correction_head[-1]
    gate_layer = model.no_op_gate_head
    assert isinstance(correction_layer, nn.Conv2d)
    assert isinstance(gate_layer, nn.Conv2d)
    with torch.no_grad():
        correction_layer.bias.fill_(-100.0)
        gate_layer.bias.fill_(-100.0)
    rgb = torch.rand(1, 3, 4, 12)
    base = torch.full((1, 1, 4, 12), 0.25)
    base[..., 0] = 0.0
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(rgb, rgb.clone(), base)
    assert output.pre_lower_bound_correction_hr_px is not None
    assert output.pre_lower_bound_disparity_hr_px is not None
    assert output.correction_hr_px.dtype == torch.float32
    assert torch.all(output.correction_hr_px >= -base.float())
    assert torch.all(output.corrected_disparity_hr_px >= 0)
    assert torch.all(output.corrected_disparity_hr_px[..., 0] == 0)
    assert output.output_valid_mask is not None
    assert not output.output_valid_mask[..., 0].any()


def test_stage_c_v2_ste_keeps_pre_bound_correction_gradient_finite() -> None:
    model = HREpipolarRefiner(
        feature_channels=8,
        correlation_groups=2,
        head_channels=12,
        base_aware_noop_v2=True,
    ).train()
    rgb_left = torch.rand(1, 3, 4, 12)
    rgb_right = torch.rand(1, 3, 4, 12)
    base = torch.full((1, 1, 4, 12), 1.0)
    output = model(rgb_left, rgb_right, base)
    assert output.pre_lower_bound_correction_hr_px is not None
    # The forward hard gate is no-op, but the soft backward path must reach the
    # correction head at initialization.
    loss = (output.pre_lower_bound_disparity_hr_px - 1.2).abs().mean()
    loss.backward()
    correction_layer = model.correction_head[-1]
    assert isinstance(correction_layer, nn.Conv2d)
    assert correction_layer.bias.grad is not None
    assert torch.isfinite(correction_layer.bias.grad).all()
    assert correction_layer.bias.grad.abs().item() > 0
    for parameter in model.no_op_gate_head.parameters():  # type: ignore[union-attr]
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
