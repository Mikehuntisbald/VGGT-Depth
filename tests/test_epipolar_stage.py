from __future__ import annotations

import pytest
import torch
from torch import nn

from models.epipolar_refiner import HREpipolarRefiner
from models.epipolar_stage import (
    FrozenTemporalEpipolarStage,
    compute_epipolar_stage_loss,
)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1)


def _stage() -> tuple[FrozenTemporalEpipolarStage, list[tuple[bool, bool]]]:
    calls: list[tuple[bool, bool]] = []

    def predictor(base: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        calls.append((torch.is_grad_enabled(), base.training))
        assert isinstance(base, _Base)
        return base.projection(batch["rgb_hr_sequence"][:, -1]).abs() + 1.0

    stage = FrozenTemporalEpipolarStage(
        _Base(),
        HREpipolarRefiner(
            feature_channels=8,
            correlation_groups=2,
            head_channels=12,
        ),
        predictor,
    )
    return stage, calls


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]
    )
    return {
        "rgb_hr_sequence": torch.rand(batch_size, 3, 3, 6, 10),
        "rgb_right_hr": torch.rand(batch_size, 3, 6, 10),
        "K_hr_sequence": intrinsics.reshape(1, 1, 3, 3).repeat(
            batch_size, 3, 1, 1
        ),
        "K_right_hr": intrinsics.reshape(1, 3, 3).repeat(batch_size, 1, 1),
        "epipolar_right_row_scale": torch.ones(batch_size),
        "epipolar_right_row_offset_hr_px": torch.zeros(batch_size),
        "epipolar_right_row_mapping_source": [
            "audited_same_row_rectified_pixels_v1"
        ]
        * batch_size,
    }


def test_stage_freezes_base_and_only_refiner_receives_gradients() -> None:
    stage, calls = _stage()
    stage.train()
    assert not stage.base_model.training
    assert stage.refiner.training
    assert all(not parameter.requires_grad for parameter in stage.base_model.parameters())
    assert stage.trainable_parameter_count == stage.refiner.trainable_parameter_count

    output = stage(_batch())
    assert calls == [(False, False)]
    assert output.base_disparity_hr_px.shape == (2, 1, 6, 10)
    assert output.refined_disparity_hr_px.shape == (2, 1, 6, 10)
    assert output.correction_hr_px.shape == (2, 1, 6, 10)
    torch.testing.assert_close(output.refinement.right_row_scale, torch.ones(2))
    torch.testing.assert_close(
        output.refinement.right_row_offset_hr_px, torch.zeros(2)
    )
    assert not output.base_disparity_hr_px.requires_grad

    target = torch.ones_like(output.refined_disparity_hr_px) * 2.0
    loss = compute_epipolar_stage_loss(
        output,
        target,
        torch.ones_like(target, dtype=torch.bool),
    )
    loss.total.backward()
    assert all(parameter.grad is None for parameter in stage.base_model.parameters())
    assert any(
        parameter.grad is not None for parameter in stage.refiner.parameters()
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in stage.refiner.parameters()
    )


def test_initial_stage_is_exact_no_op_and_reports_bounded_search() -> None:
    stage, _ = _stage()
    output = stage.eval()(_batch(batch_size=1))

    torch.testing.assert_close(
        output.refined_disparity_hr_px, output.base_disparity_hr_px
    )
    torch.testing.assert_close(
        output.correction_hr_px, torch.zeros_like(output.correction_hr_px)
    )
    torch.testing.assert_close(
        stage.refiner.candidate_offsets_hr_px,
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
    )
    assert output.refinement.candidate_valid_mask.shape == (1, 5, 6, 10)


def test_empty_teacher_mask_returns_differentiable_finite_zero() -> None:
    stage, _ = _stage()
    output = stage(_batch(batch_size=1))
    target = torch.ones_like(output.refined_disparity_hr_px)
    loss = compute_epipolar_stage_loss(
        output,
        target,
        torch.zeros_like(target, dtype=torch.bool),
        correction_regularizer_weight=0.5,
    )

    assert loss.valid_pixel_count == 0
    assert loss.total.item() == pytest.approx(0.0)
    assert loss.disparity.item() == pytest.approx(0.0)
    assert loss.correction_regularizer.item() == pytest.approx(0.0)
    assert loss.total.requires_grad
    loss.total.backward()


def test_stage_rejects_mismatched_right_crop_shape() -> None:
    stage, _ = _stage()
    batch = _batch(batch_size=1)
    batch["rgb_right_hr"] = torch.rand(1, 3, 5, 10)
    with pytest.raises(ValueError, match="right RGB shape"):
        stage(batch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epipolar_right_row_scale", 1.001),
        ("epipolar_right_row_offset_hr_px", 0.001),
    ],
)
def test_formal_stage_rejects_non_same_row_runtime_mapping(
    field: str, value: float
) -> None:
    stage, _ = _stage()
    batch = _batch(batch_size=1)
    batch[field].fill_(value)

    with pytest.raises(ValueError, match="exact same-row"):
        stage(batch)


def test_loss_rejects_negative_regularizer_weight() -> None:
    stage, _ = _stage()
    output = stage(_batch(batch_size=1))
    target = torch.ones_like(output.refined_disparity_hr_px)
    with pytest.raises(ValueError, match=">= 0"):
        compute_epipolar_stage_loss(
            output,
            target,
            torch.ones_like(target, dtype=torch.bool),
            correction_regularizer_weight=-1.0,
        )
