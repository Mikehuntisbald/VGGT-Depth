from __future__ import annotations

import torch

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
            output.source_weights,
            output.log_variance,
            output.uncertainty,
        )
    )


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
