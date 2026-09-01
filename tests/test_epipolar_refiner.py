from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from models.epipolar_refiner import (
    HREpipolarRefiner,
    groupwise_epipolar_correlation,
    rectified_vertical_affine_from_intrinsics,
)


def _grid_sample_same_row_reference(
    feature_left_hr: torch.Tensor,
    feature_right_hr: torch.Tensor,
    predicted_disparity_hr_px: torch.Tensor,
    *,
    candidate_offsets_hr_px: tuple[float, ...],
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Previous grid-sample formulation, retained only as a test oracle."""

    batch, channels, height, width = feature_left_hr.shape
    offsets = torch.tensor(
        candidate_offsets_hr_px,
        dtype=feature_left_hr.dtype,
        device=feature_left_hr.device,
    )
    candidate_count = offsets.numel()
    column = torch.arange(width, dtype=feature_left_hr.dtype).view(1, 1, 1, width)
    row = torch.arange(height, dtype=feature_left_hr.dtype).view(1, 1, height, 1)
    source_u = (
        column
        - predicted_disparity_hr_px[:, 0].unsqueeze(1)
        - offsets.view(1, candidate_count, 1, 1)
    )
    source_v = row.expand(batch, candidate_count, height, width)
    valid = (
        torch.isfinite(source_u)
        & (source_u >= 0.0)
        & (source_u <= float(width - 1))
    )
    grid_x = 2.0 * (source_u + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (source_v + 0.5) / float(height) - 1.0
    packed_grid = torch.stack((grid_x, grid_y), dim=-1).permute(
        0, 2, 1, 3, 4
    ).reshape(batch, height, candidate_count * width, 2)
    sampled = functional.grid_sample(
        feature_right_hr,
        packed_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, channels, height, candidate_count, width).permute(
        0, 1, 3, 2, 4
    )
    channels_per_group = channels // num_groups
    grouped_left = feature_left_hr.reshape(
        batch, num_groups, channels_per_group, height, width
    )
    grouped_right = sampled.reshape(
        batch, num_groups, channels_per_group, candidate_count, height, width
    )
    correlation = (grouped_left.unsqueeze(3) * grouped_right).mean(dim=2)
    return correlation.masked_fill(~valid.unsqueeze(1), 0.0), valid


@pytest.mark.parametrize(
    ("true_disparity_hr_px", "predicted_disparity_hr_px", "expected_delta_index"),
    [
        (2.0, 2.0, 2),  # the zero-correction candidate
        (3.0, 1.0, 4),  # delta=+2 reaches x_right=x_left-3
        (1.0, 3.0, 0),  # delta=-2 reaches x_right=x_left-1
    ],
)
def test_groupwise_correlation_peak_has_expected_disparity_correction_sign(
    true_disparity_hr_px: float,
    predicted_disparity_hr_px: float,
    expected_delta_index: int,
) -> None:
    width = 12
    # One-hot horizontal position descriptors make the true match exact and
    # every incorrect in-bounds candidate orthogonal.
    feature_right_hr = torch.eye(width).T.reshape(1, width, 1, width)
    feature_left_hr = torch.zeros_like(feature_right_hr)
    true_disparity = int(true_disparity_hr_px)
    feature_left_hr[..., true_disparity:] = feature_right_hr[..., :-true_disparity]
    predicted = torch.full((1, 1, 1, width), predicted_disparity_hr_px)

    correlation, valid = groupwise_epipolar_correlation(
        feature_left_hr,
        feature_right_hr,
        predicted,
        num_groups=1,
    )

    interior = slice(5, 10)
    assert valid[0, expected_delta_index, 0, interior].all()
    peak = correlation.mean(dim=1).argmax(dim=1)
    torch.testing.assert_close(
        peak[0, 0, interior],
        torch.full_like(peak[0, 0, interior], expected_delta_index),
    )


def test_fractional_disparity_uses_bilinear_right_feature_sampling() -> None:
    width = 8
    feature_left_hr = torch.ones(1, 1, 1, width)
    feature_right_hr = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, width)
    predicted_disparity_hr_px = torch.full((1, 1, 1, width), 1.5)

    correlation, valid = groupwise_epipolar_correlation(
        feature_left_hr,
        feature_right_hr,
        predicted_disparity_hr_px,
        candidate_offsets_hr_px=(0.0,),
        num_groups=1,
    )

    assert valid[0, 0, 0, 4]
    # At left x=4, the right feature is sampled at x=2.5.
    torch.testing.assert_close(correlation[0, 0, 0, 0, 4], torch.tensor(2.5))


def test_same_row_gather_matches_grid_sample_values_and_gradients() -> None:
    """The deterministic sampler is mathematically the old HR sampler."""

    generator = torch.Generator().manual_seed(2026)
    shape = (2, 4, 3, 11)
    left_gather = torch.rand(shape, generator=generator, dtype=torch.float64).requires_grad_()
    right_gather = torch.rand(shape, generator=generator, dtype=torch.float64).requires_grad_()
    column = torch.arange(shape[-1], dtype=torch.float64).view(1, 1, 1, -1)
    disparity_gather = (1.25 + 0.03 * column).expand(2, 1, 3, -1).clone().requires_grad_()
    offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)

    correlation_gather, valid_gather = groupwise_epipolar_correlation(
        left_gather,
        right_gather,
        disparity_gather,
        candidate_offsets_hr_px=offsets,
        num_groups=2,
    )
    weights = torch.rand(
        correlation_gather.shape, generator=generator, dtype=torch.float64
    )
    gradients_gather = torch.autograd.grad(
        (correlation_gather * weights).sum(),
        (left_gather, right_gather, disparity_gather),
    )

    left_grid = left_gather.detach().clone().requires_grad_()
    right_grid = right_gather.detach().clone().requires_grad_()
    disparity_grid = disparity_gather.detach().clone().requires_grad_()
    correlation_grid, valid_grid = _grid_sample_same_row_reference(
        left_grid,
        right_grid,
        disparity_grid,
        candidate_offsets_hr_px=offsets,
        num_groups=2,
    )
    gradients_grid = torch.autograd.grad(
        (correlation_grid * weights).sum(),
        (left_grid, right_grid, disparity_grid),
    )

    assert torch.equal(valid_gather, valid_grid)
    torch.testing.assert_close(correlation_gather, correlation_grid, rtol=0, atol=1e-12)
    for actual, expected in zip(gradients_gather, gradients_grid, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_rectified_vertical_principal_point_offset_samples_correct_right_row() -> None:
    height = 9
    feature_right_hr = torch.eye(height).T.reshape(1, height, height, 1)
    feature_left_hr = torch.zeros_like(feature_right_hr)
    # A left row v corresponds to right row v+2.
    feature_left_hr[:, :, :-2] = feature_right_hr[:, :, 2:]
    disparity_hr_px = torch.zeros(1, 1, height, 1)

    correlation, valid = groupwise_epipolar_correlation(
        feature_left_hr,
        feature_right_hr,
        disparity_hr_px,
        candidate_offsets_hr_px=(0.0,),
        num_groups=1,
        right_row_offset_hr_px=torch.tensor([2.0]),
    )

    assert valid[0, 0, :-2, 0].all()
    assert not valid[0, 0, -2:, 0].any()
    assert torch.all(correlation[0, 0, 0, :-2, 0] > 0)


def test_vertical_affine_uses_cropped_left_and_right_intrinsics_with_sign() -> None:
    intrinsics_left = torch.tensor(
        [[[800.0, 0.0, 300.0], [0.0, 800.0, 100.0], [0.0, 0.0, 1.0]]]
    )
    intrinsics_right = torch.tensor(
        [[[800.0, 0.0, 300.0], [0.0, 800.0, 105.4], [0.0, 0.0, 1.0]]]
    )

    scale, offset = rectified_vertical_affine_from_intrinsics(
        intrinsics_left, intrinsics_right
    )

    torch.testing.assert_close(scale, torch.tensor([1.0]))
    torch.testing.assert_close(offset, torch.tensor([5.4]))


def test_groupwise_correlation_averages_channels_inside_each_group_only() -> None:
    feature_left_hr = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    feature_right_hr = torch.tensor([5.0, 6.0, 7.0, 8.0]).reshape(1, 4, 1, 1)
    disparity_hr_px = torch.zeros(1, 1, 1, 1)

    correlation, valid = groupwise_epipolar_correlation(
        feature_left_hr,
        feature_right_hr,
        disparity_hr_px,
        candidate_offsets_hr_px=(0.0,),
        num_groups=2,
    )

    assert valid.item()
    expected = torch.tensor(
        [
            (1.0 * 5.0 + 2.0 * 6.0) / 2.0,
            (3.0 * 7.0 + 4.0 * 8.0) / 2.0,
        ]
    )
    torch.testing.assert_close(correlation[0, :, 0, 0, 0], expected)


def test_candidate_out_of_bounds_mask_is_strict_and_correlation_is_zeroed() -> None:
    left = torch.ones(1, 4, 1, 5)
    right = torch.ones_like(left)
    disparity_hr_px = torch.full((1, 1, 1, 5), 2.0)

    correlation, valid = groupwise_epipolar_correlation(
        left,
        right,
        disparity_hr_px,
        candidate_offsets_hr_px=(-2.0, 0.0, 2.0),
        num_groups=2,
    )

    # At x=0: delta=-2 samples x=0 and is valid; delta=0/+2 are negative.
    torch.testing.assert_close(valid[0, :, 0, 0], torch.tensor([True, False, False]))
    # At x=4: all source coordinates 4,2,0 are valid, including boundaries.
    torch.testing.assert_close(valid[0, :, 0, 4], torch.tensor([True, True, True]))
    assert torch.count_nonzero(correlation[:, :, 1:, :, 0]).item() == 0


def test_refiner_cpu_shapes_parameter_budget_and_initial_no_op() -> None:
    torch.manual_seed(42)
    model = HREpipolarRefiner().eval()
    assert 0 < model.trainable_parameter_count < 500_000
    rgb_left_hr = torch.rand(2, 3, 9, 13)
    rgb_right_hr = torch.rand_like(rgb_left_hr)
    predicted_disparity_hr_px = 1.0 + 3.0 * torch.rand(2, 1, 9, 13)

    with torch.no_grad():
        output = model(rgb_left_hr, rgb_right_hr, predicted_disparity_hr_px)

    assert output.corrected_disparity_hr_px.shape == (2, 1, 9, 13)
    assert output.correction_hr_px.shape == (2, 1, 9, 13)
    assert output.correlation.shape == (2, 8, 5, 9, 13)
    assert output.candidate_valid_mask.shape == (2, 5, 9, 13)
    assert output.candidate_valid_mask.dtype == torch.bool
    assert output.confidence.shape == (2, 1, 9, 13)
    torch.testing.assert_close(output.right_row_scale, torch.ones(2))
    torch.testing.assert_close(output.right_row_offset_hr_px, torch.zeros(2))
    torch.testing.assert_close(output.correction_hr_px, torch.zeros_like(output.correction_hr_px))
    torch.testing.assert_close(
        output.corrected_disparity_hr_px, predicted_disparity_hr_px
    )
    assert torch.isfinite(output.correlation).all()
    assert torch.isfinite(output.confidence).all()


@pytest.mark.parametrize("bias", [-100.0, 100.0])
def test_correction_is_bounded_to_two_hr_pixels(bias: float) -> None:
    model = HREpipolarRefiner().eval()
    final_layer = model.correction_head[-1]
    assert isinstance(final_layer, torch.nn.Conv2d)
    with torch.no_grad():
        final_layer.bias.fill_(bias)
    rgb = torch.rand(1, 3, 5, 9)
    predicted_disparity_hr_px = torch.ones(1, 1, 5, 9)

    with torch.no_grad():
        output = model(rgb, rgb.clone(), predicted_disparity_hr_px)

    valid_correction = output.correction_hr_px[
        output.candidate_valid_mask.any(dim=1, keepdim=True)
    ]
    assert valid_correction.numel() > 0
    assert valid_correction.abs().max().item() <= 2.0
    assert torch.all(valid_correction.sign() == (1.0 if bias > 0 else -1.0))


def test_all_invalid_search_has_zero_confidence_and_zero_correction() -> None:
    model = HREpipolarRefiner().eval()
    final_layer = model.correction_head[-1]
    assert isinstance(final_layer, torch.nn.Conv2d)
    with torch.no_grad():
        final_layer.bias.fill_(10.0)
    rgb = torch.rand(1, 3, 3, 7)
    predicted_disparity_hr_px = torch.full((1, 1, 3, 7), 100.0)

    with torch.no_grad():
        output = model(rgb, rgb.clone(), predicted_disparity_hr_px)

    assert not output.candidate_valid_mask.any()
    torch.testing.assert_close(output.confidence, torch.zeros_like(output.confidence))
    torch.testing.assert_close(output.correction_hr_px, torch.zeros_like(output.correction_hr_px))
    torch.testing.assert_close(
        output.corrected_disparity_hr_px, predicted_disparity_hr_px
    )


def test_opt_in_positivity_bounds_correction_without_creating_points() -> None:
    model = HREpipolarRefiner(positivity_floor_hr_px=0.0).eval()
    final_layer = model.correction_head[-1]
    assert isinstance(final_layer, torch.nn.Conv2d)
    with torch.no_grad():
        final_layer.bias.fill_(-100.0)
    rgb = torch.rand(1, 3, 3, 9)
    base = torch.full((1, 1, 3, 9), 0.25)
    base[..., 0] = 0.0

    with torch.no_grad():
        output = model(rgb, rgb.clone(), base)

    assert output.pre_lower_bound_correction_hr_px is not None
    assert output.pre_lower_bound_disparity_hr_px is not None
    assert torch.any(output.pre_lower_bound_disparity_hr_px < 0)
    assert torch.all(output.corrected_disparity_hr_px >= 0)
    assert torch.all(output.corrected_disparity_hr_px[..., 0] == 0)
    assert not torch.any(output.corrected_disparity_hr_px > base)
    any_valid = output.candidate_valid_mask.any(dim=1, keepdim=True)
    assert torch.all(output.correction_hr_px[~any_valid] == 0)


def test_opt_in_positivity_projection_stays_fp32_under_bfloat16_autocast() -> None:
    model = HREpipolarRefiner(positivity_floor_hr_px=0.0).eval()
    final_layer = model.correction_head[-1]
    assert isinstance(final_layer, torch.nn.Conv2d)
    with torch.no_grad():
        final_layer.bias.fill_(-100.0)
    rgb = torch.rand(1, 3, 4, 12)
    base = torch.full((1, 1, 4, 12), 0.03125)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(rgb, rgb.clone(), base)

    assert output.corrected_disparity_hr_px.dtype == torch.float32
    assert output.correction_hr_px.dtype == torch.float32
    assert output.pre_lower_bound_disparity_hr_px is not None
    assert output.pre_lower_bound_disparity_hr_px.dtype == torch.float32
    assert torch.all(output.corrected_disparity_hr_px >= 0)


def test_opt_in_positivity_rejects_nonphysical_frozen_base() -> None:
    model = HREpipolarRefiner(positivity_floor_hr_px=0.0).eval()
    rgb = torch.rand(1, 3, 3, 9)
    negative_base = torch.ones(1, 1, 3, 9)
    negative_base[..., 4] = -0.01

    with pytest.raises(FloatingPointError, match="non-negative frozen base"):
        model(rgb, rgb.clone(), negative_base)


def test_positivity_floor_forbids_epsilon_fill() -> None:
    with pytest.raises(ValueError, match="exactly 0.0"):
        HREpipolarRefiner(positivity_floor_hr_px=1e-6)


def test_refiner_backward_gradients_are_finite() -> None:
    torch.manual_seed(7)
    model = HREpipolarRefiner().train()
    rgb_left_hr = torch.rand(1, 3, 6, 10, requires_grad=True)
    rgb_right_hr = torch.rand(1, 3, 6, 10, requires_grad=True)
    predicted_disparity_hr_px = (
        1.0 + torch.rand(1, 1, 6, 10)
    ).requires_grad_()

    output = model(rgb_left_hr, rgb_right_hr, predicted_disparity_hr_px)
    loss = (
        output.corrected_disparity_hr_px.mean()
        + output.correlation.square().mean()
        + output.confidence.mean()
    )
    loss.backward()

    for tensor in (rgb_left_hr, rgb_right_hr, predicted_disparity_hr_px):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_refiner_cuda_bfloat16_autocast_smoke() -> None:
    model = HREpipolarRefiner().cuda().train()
    rgb_left_hr = torch.rand(1, 3, 32, 48, device="cuda", requires_grad=True)
    rgb_right_hr = torch.rand(1, 3, 32, 48, device="cuda", requires_grad=True)
    disparity_hr_px = (
        2.0 + 4.0 * torch.rand(1, 1, 32, 48, device="cuda")
    ).requires_grad_()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(rgb_left_hr, rgb_right_hr, disparity_hr_px)
        loss = output.corrected_disparity_hr_px.mean() + output.confidence.mean()
    loss.backward()

    assert torch.isfinite(output.corrected_disparity_hr_px).all()
    assert torch.isfinite(output.correlation).all()
    assert disparity_hr_px.grad is not None
    assert torch.isfinite(disparity_hr_px.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_same_row_cuda_backward_passes_strict_deterministic_algorithms() -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(42)
        model = HREpipolarRefiner().cuda().train()
        rgb_left_hr = torch.rand(1, 3, 24, 40, device="cuda")
        rgb_right_hr = torch.rand_like(rgb_left_hr)
        disparity_hr_px = torch.full(
            (1, 1, 24, 40), 7.25, device="cuda", requires_grad=True
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(rgb_left_hr, rgb_right_hr, disparity_hr_px)
            loss = (
                output.corrected_disparity_hr_px.mean()
                + output.correlation.square().mean()
                + output.confidence.mean()
            )
        loss.backward()

        assert torch.isfinite(loss)
        assert disparity_hr_px.grad is not None
        assert torch.isfinite(disparity_hr_px.grad).all()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )
