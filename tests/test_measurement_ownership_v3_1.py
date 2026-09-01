from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.ffs_omega_tsr import (
    FFSOmegaTSR,
    project_ffs_measurement_ownership_v3_1,
)


def _project(
    proposal: torch.Tensor,
    disparity_lr_hr_px: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return project_ffs_measurement_ownership_v3_1(
        proposal,
        disparity_lr_hr_px,
        confidence,
        valid,
        scale=2,
        trusted_confidence_threshold=0.8,
        minimum_subpixel_residual_hr_px=1.0,
        maximum_subpixel_residual_hr_px=8.0,
        boundary_relative_scale=0.10,
    )


def test_trusted_lr_measurement_is_exact_but_hr_nullspace_detail_survives() -> None:
    disparity_lr_hr_px = torch.tensor([[[[10.0, 12.0]]]])
    confidence = torch.ones_like(disparity_lr_hr_px)
    valid = torch.ones_like(disparity_lr_hr_px, dtype=torch.bool)
    target_nearest = F.interpolate(disparity_lr_hr_px, scale_factor=2, mode="nearest")
    detail = torch.tensor(
        [[[[0.75, -0.75, 0.40, -0.40], [-0.25, 0.25, -0.20, 0.20]]]]
    )
    proposal = (target_nearest + detail).requires_grad_()

    output, _, trusted_hr = _project(
        proposal, disparity_lr_hr_px, confidence, valid
    )
    sampled = F.interpolate(
        output,
        size=disparity_lr_hr_px.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(sampled, disparity_lr_hr_px, rtol=0, atol=1.0e-6)
    assert trusted_hr.all()
    assert torch.count_nonzero(output - target_nearest) > 0
    # Exact ownership is a differentiable projection, not a detached overwrite.
    output.square().mean().backward()
    assert proposal.grad is not None
    assert torch.isfinite(proposal.grad).all()
    assert torch.count_nonzero(proposal.grad) > 0


def test_confidence_and_ffs_boundary_set_a_finite_per_cell_detail_limit() -> None:
    disparity_lr_hr_px = torch.tensor(
        [[[[10.0, 10.0, 10.0, 20.0, 20.0, 20.0]]]]
    )
    confidence = torch.ones_like(disparity_lr_hr_px)
    confidence[..., -1] = 0.0
    valid = torch.ones_like(disparity_lr_hr_px, dtype=torch.bool)
    bilinear = F.interpolate(
        disparity_lr_hr_px, scale_factor=2, mode="bilinear", align_corners=False
    )
    proposal = bilinear + 100.0

    output, limit_hr, trusted_hr = _project(
        proposal, disparity_lr_hr_px, confidence, valid
    )
    limit_lr = limit_hr[..., ::2, ::2]
    # High-confidence flat interior uses the strict floor; a depth boundary and
    # a low-confidence cell get the relaxed but still finite ceiling.
    torch.testing.assert_close(limit_lr[..., 0], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(limit_lr[..., 2], torch.tensor([[[8.0]]]))
    torch.testing.assert_close(limit_lr[..., -1], torch.tensor([[[8.0]]]))
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)
    # The low-confidence final cell is not projected to exact ownership, but
    # its correction around bilinear FFS remains hard bounded.
    low_trust = ~trusted_hr
    assert low_trust.any()
    assert torch.all((output - bilinear).abs()[low_trust] <= limit_hr[low_trust] + 1.0e-6)


def test_v3_1_hole_without_vggt_or_history_metric_support_is_exact_zero() -> None:
    model = FFSOmegaTSR(
        physical_output_v2=True,
        calibration_conditioning_v3=True,
        use_rays=False,
        use_stereo_pose=False,
        use_temporal_pose=False,
        align_corners_false_pixel_centers=True,
        measurement_ownership_v3_1=True,
    ).eval()
    rgb = torch.rand(1, 3, 8, 12)
    disparity = torch.full((1, 1, 4, 6), -4.0)
    confidence = torch.ones_like(disparity)
    valid = torch.zeros_like(disparity, dtype=torch.bool)
    validity_layer = model.validity_completion_head
    assert isinstance(validity_layer, nn.Sequential)
    final = validity_layer[-1]
    assert isinstance(final, nn.Conv2d)
    with torch.no_grad():
        # Even an arbitrarily confident learned completion request cannot make
        # RGB-only geometry metric-valid.
        final.bias[0] = 100.0
        final.bias[1] = 100.0
        output = model(rgb, disparity, confidence, valid_ffs=valid)
    torch.testing.assert_close(
        output.disparity_hr_px, torch.zeros_like(output.disparity_hr_px)
    )
    assert output.completion_probability is not None
    assert output.completion_probability.eq(0).all()
    assert output.completion_mask is not None and not output.completion_mask.any()
    assert output.output_valid_mask is not None and not output.output_valid_mask.any()


def test_measurement_v3_1_is_opt_in_and_preserves_v2_state_and_output() -> None:
    torch.manual_seed(123)
    legacy_v2 = FFSOmegaTSR(physical_output_v2=True).eval()
    torch.manual_seed(123)
    explicit_disabled = FFSOmegaTSR(
        physical_output_v2=True,
        measurement_ownership_v3_1=False,
    ).eval()
    assert tuple(legacy_v2.state_dict()) == tuple(explicit_disabled.state_dict())
    rgb = torch.rand(1, 3, 8, 12)
    disparity = 1.0 + torch.rand(1, 1, 4, 6)
    confidence = torch.rand(1, 1, 4, 6)
    with torch.no_grad():
        expected = legacy_v2(rgb, disparity, confidence)
        actual = explicit_disabled(rgb, disparity, confidence)
    for name in (
        "disparity_hr_px",
        "disparity_raw_hr_px",
        "source_weights",
        "log_variance",
        "anchor_gate",
        "valid_probability",
        "completion_probability",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0, atol=0
        )


def test_measurement_v3_1_rejects_use_without_physical_output_contract() -> None:
    try:
        FFSOmegaTSR(measurement_ownership_v3_1=True)
    except ValueError as error:
        assert "requires physical_output_v2" in str(error)
    else:
        raise AssertionError("v3.1 ownership must fail closed outside physical v2")
