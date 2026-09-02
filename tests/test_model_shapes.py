from __future__ import annotations

import torch

from losses import lower_bound_penalty
from models.convex_upsampler import ConvexUpsampler
from models.ffs_omega_tsr import FFSOmegaTSR, count_trainable_parameters
from models.source_gating import masked_source_softmax


def _base_inputs(
    *, batch: int = 1, height_lr: int = 6, width_lr: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    rgb_hr = torch.rand(
        batch, 3, 2 * height_lr, 2 * width_lr, generator=generator
    )
    disparity_ffs_hr_px = 2.0 + 20.0 * torch.rand(
        batch, 1, height_lr, width_lr, generator=generator
    )
    confidence_ffs = torch.rand(
        batch, 1, height_lr, width_lr, generator=generator
    )
    return rgb_hr, disparity_ffs_hr_px, confidence_ffs


def test_default_model_shapes_are_x2_and_parameter_budget_is_below_12m() -> None:
    model = FFSOmegaTSR().eval()
    parameter_count = count_trainable_parameters(model)
    print(f"FFSOmegaTSR trainable parameters: {parameter_count:,}")
    assert parameter_count == model.trainable_parameter_count
    assert 0 < parameter_count < 12_000_000

    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs(batch=2)
    height_lr, width_lr = disparity_ffs_hr_px.shape[-2:]
    with torch.no_grad():
        output = model(rgb_hr, disparity_ffs_hr_px, confidence_ffs)

    assert output.disparity_hr_px.shape == (2, 1, 2 * height_lr, 2 * width_lr)
    assert output.disparity_raw_hr_px.shape == output.disparity_hr_px.shape
    assert output.disparity_source_mix_hr_px_lr_grid is not None
    assert output.disparity_post_lr_residual_hr_px_lr_grid is not None
    assert output.disparity_post_convex_hr_px is not None
    assert output.disparity_source_mix_hr_px_lr_grid.shape == (
        2,
        1,
        height_lr,
        width_lr,
    )
    assert output.disparity_post_lr_residual_hr_px_lr_grid.shape == (
        2,
        1,
        height_lr,
        width_lr,
    )
    assert output.disparity_post_convex_hr_px.shape == output.disparity_hr_px.shape
    assert output.source_weights.shape == (2, 3, height_lr, width_lr)
    assert output.log_variance.shape == output.disparity_hr_px.shape
    assert output.uncertainty.shape == output.disparity_hr_px.shape
    assert output.anchor_gate.shape == output.disparity_hr_px.shape
    assert output.source_valid_mask.shape == (2, 3, height_lr, width_lr)
    assert len(output.hidden_state) == 2
    assert all(state.shape == (2, 96, height_lr, width_lr) for state in output.hidden_state)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (
            output.disparity_hr_px,
            output.disparity_raw_hr_px,
            output.disparity_source_mix_hr_px_lr_grid,
            output.disparity_post_lr_residual_hr_px_lr_grid,
            output.disparity_post_convex_hr_px,
            output.source_weights,
            output.log_variance,
            output.uncertainty,
        )
    )

    # Diagnostic taps are plain outputs: they add no state and therefore keep
    # legacy checkpoints strictly loadable.
    reloaded = FFSOmegaTSR().eval()
    incompatible = reloaded.load_state_dict(model.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(reloaded.state_dict()) == tuple(model.state_dict())


def test_convex_upsampling_preserves_hr_pixel_units_and_constant_borders() -> None:
    upsampler = ConvexUpsampler(scale=2)
    disparity_hr_px_lr_grid = torch.full((1, 1, 3, 4), 17.5)
    arbitrary_mask_logits = torch.randn(1, upsampler.mask_channels, 3, 4)

    disparity_hr_px = upsampler(disparity_hr_px_lr_grid, arbitrary_mask_logits)

    assert disparity_hr_px.shape == (1, 1, 6, 8)
    # No x2 value multiplication: values already use HR-pixel disparity units.
    torch.testing.assert_close(
        disparity_hr_px, torch.full_like(disparity_hr_px, 17.5), atol=1e-5, rtol=1e-5
    )


def test_convex_bilinear_initialization_matches_align_corners_false_resize() -> None:
    upsampler = ConvexUpsampler(scale=2)
    generator = torch.Generator().manual_seed(123)
    disparity_hr_px_lr_grid = torch.randn(2, 1, 5, 7, generator=generator)
    # The helper returns one bias vector in the same flattened channel order
    # as ConvexUpsampler's [3x3, phase_y, phase_x] mask layout.
    mask_logits = upsampler.bilinear_mask_logits().reshape(1, -1, 1, 1)
    mask_logits = mask_logits.expand(
        disparity_hr_px_lr_grid.shape[0],
        upsampler.mask_channels,
        disparity_hr_px_lr_grid.shape[-2],
        disparity_hr_px_lr_grid.shape[-1],
    )
    actual = upsampler(disparity_hr_px_lr_grid, mask_logits)
    expected = torch.nn.functional.interpolate(
        disparity_hr_px_lr_grid,
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(
        torch.softmax(mask_logits[:1, :, :1, :1].reshape(9, 2, 2), dim=0).sum(dim=0),
        torch.ones(2, 2),
        rtol=0.0,
        atol=1e-6,
    )


def test_model_bilinear_convex_initialization_preserves_initial_spatial_output() -> None:
    model = FFSOmegaTSR(convex_initialization="bilinear").eval()
    rgb, disparity, confidence = _base_inputs()
    confidence.fill_(1.0)
    valid = torch.ones_like(confidence, dtype=torch.bool)
    with torch.no_grad():
        output = model(rgb, disparity, confidence, valid_ffs=valid)
    expected = torch.nn.functional.interpolate(
        disparity, scale_factor=2, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(output.disparity_hr_px, expected, rtol=0.0, atol=2e-5)


def test_masked_softmax_excludes_sources_and_has_finite_all_invalid_fallback() -> None:
    logits = torch.tensor(
        [[[[1.0, 1.0]], [[100.0, 2.0]], [[3.0, 100.0]]]], dtype=torch.float32
    )
    valid_mask = torch.tensor(
        [[[[True, False]], [[False, False]], [[True, False]]]]
    )

    weights = masked_source_softmax(logits, valid_mask)

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1, 2))
    assert weights[0, 1, 0, 0].item() == 0.0
    # At the all-invalid second pixel, deterministic FFS fallback is one-hot.
    torch.testing.assert_close(weights[0, :, 0, 1], torch.tensor([1.0, 0.0, 0.0]))


def test_model_masks_invalid_sources_and_sanitizes_an_all_invalid_pixel() -> None:
    model = FFSOmegaTSR().eval()
    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs()
    disparity_vggt_hr_px = disparity_ffs_hr_px + 3.0
    confidence_vggt = torch.ones_like(confidence_ffs)
    disparity_history_hr_px = disparity_ffs_hr_px - 1.0
    confidence_history = torch.ones_like(confidence_ffs)
    history_visibility = torch.ones_like(confidence_ffs)

    valid_ffs = torch.ones_like(confidence_ffs, dtype=torch.bool)
    valid_vggt = torch.zeros_like(valid_ffs)
    valid_history = torch.ones_like(valid_ffs)
    # This pixel is invalid for every source and contains non-finite measurements.
    disparity_ffs_hr_px[..., 0, 0] = float("nan")
    disparity_vggt_hr_px[..., 0, 0] = float("inf")
    disparity_history_hr_px[..., 0, 0] = float("nan")
    valid_ffs[..., 0, 0] = False
    valid_history[..., 0, 0] = False

    with torch.no_grad():
        output = model(
            rgb_hr,
            disparity_ffs_hr_px,
            confidence_ffs,
            disparity_vggt_hr_px=disparity_vggt_hr_px,
            confidence_vggt=confidence_vggt,
            disparity_history_hr_px=disparity_history_hr_px,
            confidence_history=confidence_history,
            history_visibility=history_visibility,
            valid_ffs=valid_ffs,
            valid_vggt=valid_vggt,
            valid_history=valid_history,
        )

    assert torch.count_nonzero(output.source_weights[:, 1]).item() == 0
    torch.testing.assert_close(
        output.source_weights[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0])
    )
    assert not output.source_valid_mask[0, :, 0, 0].any()
    assert torch.isfinite(output.disparity_hr_px).all()
    assert torch.isfinite(output.uncertainty).all()


def test_ffs_confidence_anchor_applies_exact_correction_bounds() -> None:
    model = FFSOmegaTSR().eval()
    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs()
    confidence_ffs.fill_(1.0)

    with torch.no_grad():
        trusted_output = model(rgb_hr, disparity_ffs_hr_px, confidence_ffs)

    baseline_hr_px = torch.nn.functional.interpolate(
        disparity_ffs_hr_px, scale_factor=2, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(
        trusted_output.anchor_gate, torch.full_like(trusted_output.anchor_gate, 0.1)
    )
    torch.testing.assert_close(
        trusted_output.disparity_hr_px - baseline_hr_px,
        0.1 * (trusted_output.disparity_raw_hr_px - baseline_hr_px),
    )

    confidence_ffs.zero_()
    with torch.no_grad():
        low_confidence_output = model(
            rgb_hr,
            disparity_ffs_hr_px,
            confidence_ffs,
            valid_ffs=torch.ones_like(confidence_ffs, dtype=torch.bool),
        )
    torch.testing.assert_close(
        low_confidence_output.anchor_gate,
        torch.ones_like(low_confidence_output.anchor_gate),
    )
    torch.testing.assert_close(
        low_confidence_output.disparity_hr_px,
        low_confidence_output.disparity_raw_hr_px,
    )


def test_d025_positivity_opt_in_sanitizes_all_invalid_without_epsilon_fill() -> None:
    baseline = FFSOmegaTSR().eval()
    ablation = FFSOmegaTSR(
        sanitize_invalid_source_disparities=True,
        positivity_floor_hr_px=0.0,
    ).eval()
    incompatible = ablation.load_state_dict(baseline.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    rgb_hr = torch.rand(1, 3, 4, 6)
    negative_ffs = torch.full((1, 1, 2, 3), -3.0)
    confidence = torch.ones_like(negative_ffs)
    valid = torch.zeros_like(negative_ffs, dtype=torch.bool)
    with torch.no_grad():
        baseline_output = baseline(
            rgb_hr, negative_ffs, confidence, valid_ffs=valid
        )
        ablation_output = ablation(
            rgb_hr, negative_ffs, confidence, valid_ffs=valid
        )

    assert (baseline_output.disparity_hr_px < 0).all()
    assert not ablation_output.source_valid_mask.any()
    torch.testing.assert_close(
        ablation_output.source_weights,
        torch.tensor([[[[1.0] * 3] * 2, [[0.0] * 3] * 2, [[0.0] * 3] * 2]]),
    )
    for tensor in (
        ablation_output.disparity_source_mix_hr_px_lr_grid,
        ablation_output.disparity_post_lr_residual_hr_px_lr_grid,
        ablation_output.disparity_post_convex_hr_px,
        ablation_output.disparity_raw_hr_px,
        ablation_output.disparity_hr_px,
    ):
        assert tensor is not None
        # Exactly zero remains an invalid measurement; no epsilon or softplus
        # creates a fake positive completion at the all-invalid pixels.
        torch.testing.assert_close(tensor, torch.zeros_like(tensor))


def test_d025_positivity_keeps_high_confidence_anchor_and_has_finite_gradient() -> None:
    model = FFSOmegaTSR(
        sanitize_invalid_source_disparities=True,
        positivity_floor_hr_px=0.0,
    ).train()
    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs()
    confidence_ffs.fill_(1.0)
    with torch.no_grad():
        model.disparity_residual_head.bias.fill_(-1.0)
    output = model(rgb_hr, disparity_ffs_hr_px, confidence_ffs)

    assert output.disparity_pre_lower_bound_hr_px_lr_grid is not None
    assert output.disparity_pre_lower_bound_raw_hr_px is not None
    torch.testing.assert_close(
        output.anchor_gate, torch.full_like(output.anchor_gate, 0.1)
    )
    baseline_hr_px = torch.nn.functional.interpolate(
        disparity_ffs_hr_px, scale_factor=2, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(
        output.disparity_hr_px - baseline_hr_px,
        0.1 * (output.disparity_raw_hr_px - baseline_hr_px),
    )
    penalty = lower_bound_penalty(
        output.disparity_pre_lower_bound_hr_px_lr_grid, lower_bound_hr_px=0.0
    ) + lower_bound_penalty(
        output.disparity_pre_lower_bound_raw_hr_px, lower_bound_hr_px=0.0
    )
    penalty.backward()
    assert model.disparity_residual_head.bias.grad is not None
    assert bool(torch.isfinite(model.disparity_residual_head.bias.grad).all())
    assert model.disparity_residual_head.bias.grad.abs().item() > 0.0


def test_default_positivity_options_leave_the_forward_path_bitwise_unchanged() -> None:
    reference = FFSOmegaTSR().eval()
    explicit_default = FFSOmegaTSR(
        sanitize_invalid_source_disparities=False,
        positivity_floor_hr_px=None,
    ).eval()
    explicit_default.load_state_dict(reference.state_dict(), strict=True)
    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs()
    with torch.no_grad():
        before = reference(rgb_hr, disparity_ffs_hr_px, confidence_ffs)
        after = explicit_default(rgb_hr, disparity_ffs_hr_px, confidence_ffs)
    for name in (
        "disparity_hr_px",
        "disparity_raw_hr_px",
        "source_weights",
        "log_variance",
        "disparity_source_mix_hr_px_lr_grid",
        "disparity_post_lr_residual_hr_px_lr_grid",
        "disparity_post_convex_hr_px",
    ):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    assert after.disparity_pre_lower_bound_hr_px_lr_grid is None
    assert after.disparity_pre_lower_bound_raw_hr_px is None


def test_recurrence_is_deterministic_on_reset_and_advances_with_prior_state() -> None:
    torch.manual_seed(7)
    model = FFSOmegaTSR().eval()
    rgb_hr, disparity_ffs_hr_px, confidence_ffs = _base_inputs()

    with torch.no_grad():
        first = model(rgb_hr, disparity_ffs_hr_px, confidence_ffs)
        reset_repeat = model(rgb_hr, disparity_ffs_hr_px, confidence_ffs)
        second = model(
            rgb_hr,
            disparity_ffs_hr_px,
            confidence_ffs,
            hidden_state=first.hidden_state,
        )

    for first_state, repeated_state in zip(
        first.hidden_state, reset_repeat.hidden_state, strict=True
    ):
        torch.testing.assert_close(first_state, repeated_state)
    assert any(
        not torch.allclose(first_state, second_state)
        for first_state, second_state in zip(first.hidden_state, second.hidden_state, strict=True)
    )
    assert all(torch.isfinite(state).all() for state in second.hidden_state)
