from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from backbones.ffs_adapter import (
    FFSAdapter,
    disparity_lr_to_hr_pixels,
    left_right_consistency_error,
    make_right_reference_pair,
    normalized_cost_entropy,
    padding_to_multiple,
    quarter_grid_delta_to_input_pixels,
    restore_right_reference_disparity,
)


def test_padding_matches_upstream_symmetric_divisible_by_32_contract() -> None:
    tensor = torch.arange(35 * 66, dtype=torch.float32).reshape(1, 1, 35, 66)
    padding = padding_to_multiple(35, 66)

    assert padding.as_tuple == (15, 15, 14, 15)
    padded = padding.apply(tensor)
    assert padded.shape[-2:] == (64, 96)
    torch.testing.assert_close(padding.remove(padded), tensor)


def test_disparity_and_update_delta_units_are_explicit() -> None:
    disparity_lr_px = torch.tensor([[[[1.25, 7.0]]]])
    torch.testing.assert_close(
        disparity_lr_to_hr_pixels(disparity_lr_px, 2),
        torch.tensor([[[[2.5, 14.0]]]]),
    )
    delta_quarter_px = torch.tensor([[[[-0.5, 0.75]]]])
    torch.testing.assert_close(
        quarter_grid_delta_to_input_pixels(delta_quarter_px),
        torch.tensor([[[[-2.0, 3.0]]]]),
    )


def test_normalized_entropy_has_correct_extremes_and_shape() -> None:
    uniform_logits = torch.zeros(2, 1, 4, 3, 5)
    uniform_entropy = normalized_cost_entropy(uniform_logits)
    assert uniform_entropy.shape == (2, 1, 3, 5)
    torch.testing.assert_close(uniform_entropy, torch.ones_like(uniform_entropy))

    peaked_logits = torch.full((1, 1, 4, 1, 1), -100.0)
    peaked_logits[:, :, 2] = 100.0
    peaked_entropy = normalized_cost_entropy(peaked_logits)
    assert peaked_entropy.item() == pytest.approx(0.0, abs=1e-6)

    one_bin = normalized_cost_entropy(torch.randn(1, 1, 1, 2, 3))
    torch.testing.assert_close(one_bin, torch.zeros_like(one_bin))


def test_right_reference_requires_swap_and_horizontal_flip() -> None:
    left = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    right = torch.tensor([[[[4.0, 5.0, 6.0]]]])

    right_reference, left_target = make_right_reference_pair(left, right)
    torch.testing.assert_close(right_reference, torch.tensor([[[[6.0, 5.0, 4.0]]]]))
    torch.testing.assert_close(left_target, torch.tensor([[[[3.0, 2.0, 1.0]]]]))
    torch.testing.assert_close(
        restore_right_reference_disparity(right_reference), right
    )


def test_left_right_error_samples_at_x_minus_left_disparity() -> None:
    # d_left(x)=1 samples d_right at x-1.  The first pixel is out of view.
    left = torch.ones(1, 1, 1, 5)
    right = torch.tensor([[[[1.0, 1.0, 2.0, 1.0, 1.0]]]])
    error, valid = left_right_consistency_error(left, right)

    assert valid.tolist() == [[[[False, True, True, True, True]]]]
    assert math.isinf(error[0, 0, 0, 0].item())
    torch.testing.assert_close(error[0, 0, 0, 1:], torch.tensor([0.0, 0.0, 1.0, 0.0]))


class _FakeClassifier(nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        return logits


class _FakeUpdateBlock(nn.Module):
    def forward(self, delta: Tensor) -> tuple[None, None, Tensor]:
        return None, None, delta


class _FakeFFS(nn.Module):
    """Small module exposing the exact hook surfaces used by upstream FFS."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.classifier = _FakeClassifier()
        self.update_block = _FakeUpdateBlock()
        self.calls: list[tuple[Tensor, Tensor, bool]] = []

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        iters: int,
        test_mode: bool,
        optimize_build_volume: str,
    ) -> Tensor:
        del optimize_build_volume
        self.calls.append((left.detach().clone(), right.detach().clone(), torch.is_grad_enabled()))
        batch, _, height, width = left.shape
        logits = torch.full(
            (batch, 1, 4, height // 4, width // 4),
            -10.0,
            dtype=left.dtype,
            device=left.device,
        )
        logits[:, :, 0] = 10.0
        self.classifier(logits)
        for iteration in range(iters):
            delta = torch.full(
                (batch, 1, height // 4, width // 4),
                0.25 if iteration == iters - 1 else 0.0,
                dtype=left.dtype,
                device=left.device,
            )
            self.update_block(delta)
        assert test_mode
        return torch.ones(batch, 1, height, width, dtype=left.dtype, device=left.device)


def test_adapter_freezes_runs_inference_unpads_and_computes_aux() -> None:
    model = _FakeFFS()
    adapter = FFSAdapter(
        model,
        spatial_scale=2,
        iterations=2,
        update_tau_input_px=2.0,
    )
    adapter.train(True)
    left = torch.arange(3 * 35 * 67, dtype=torch.float32).reshape(1, 3, 35, 67)
    right = left + 1

    output = adapter(left, right, right_left_check=True)

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert [call[0].shape[-2:] for call in model.calls] == [(64, 96), (64, 96)]
    assert all(not grad_enabled for _, _, grad_enabled in model.calls)
    assert output.disparity_lr_px.shape == (1, 1, 35, 67)
    torch.testing.assert_close(output.disparity_lr_px, torch.ones_like(output.disparity_lr_px))
    torch.testing.assert_close(output.disparity_hr_px, torch.full_like(output.disparity_hr_px, 2.0))
    torch.testing.assert_close(
        output.last_update_magnitude_input_px,
        torch.ones_like(output.last_update_magnitude_input_px),
    )
    torch.testing.assert_close(
        output.left_right_error_lr_px[..., 1:],
        torch.zeros_like(output.left_right_error_lr_px[..., 1:]),
    )
    assert not output.valid_mask[..., 0].any()
    assert output.valid_mask[..., 1:].all()
    expected_confidence = math.exp(-0.5)
    assert output.confidence[..., 1:].mean().item() == pytest.approx(
        expected_confidence, rel=1e-5
    )
    assert output.metadata["update_delta_source_unit"] == "1/4-grid pixels"

    # The second pass must be FFS(flip(R), flip(L)).  Unpad before comparing
    # because the fake model records its padded inputs.
    second_left, second_right, _ = model.calls[1]
    padding = padding_to_multiple(35, 67)
    torch.testing.assert_close(padding.remove(second_left), torch.flip(right, (-1,)))
    torch.testing.assert_close(padding.remove(second_right), torch.flip(left, (-1,)))
