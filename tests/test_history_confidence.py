from __future__ import annotations

import math

import pytest
import torch

from geometry.history_confidence import history_confidence


def test_history_confidence_matches_declared_exponential_decay() -> None:
    previous = torch.tensor([[[[0.8]]]])
    visible = torch.ones_like(previous, dtype=torch.bool)
    collision = torch.zeros_like(visible)
    photo = torch.tensor([[[[0.1]]]])
    history = torch.tensor([[[[12.0]]]])
    current = torch.tensor([[[[11.0]]]])
    current_confidence = torch.tensor([[[[0.5]]]])
    result = history_confidence(
        previous,
        visible,
        collision,
        photo,
        history,
        current,
        current_confidence,
        photometric_temperature=0.1,
        disparity_temperature_hr_px=2.0,
    )
    expected = 0.8 * math.exp(-1.0) * math.exp(-0.5)
    assert result.confidence.item() == pytest.approx(expected, rel=1e-6)
    assert bool(result.valid_mask.item())


def test_collision_and_trusted_ffs_conflict_are_hard_rejections() -> None:
    shape = (1, 1, 1, 2)
    previous = torch.ones(shape)
    visible = torch.ones(shape, dtype=torch.bool)
    collision = torch.tensor([[[[True, False]]]])
    photo = torch.zeros(shape)
    history = torch.tensor([[[[10.0, 20.0]]]])
    current = torch.tensor([[[[10.0, 10.0]]]])
    current_confidence = torch.ones(shape)
    result = history_confidence(
        previous,
        visible,
        collision,
        photo,
        history,
        current,
        current_confidence,
        reject_conflict_hr_px=2.0,
    )
    assert not bool(result.valid_mask.any())
    assert result.confidence.eq(0).all()
    assert not result.rejected_current_ffs_conflict[0, 0, 0, 0]
    assert result.rejected_current_ffs_conflict[0, 0, 0, 1]


def test_nonfinite_history_is_zero_and_never_propagates_nan() -> None:
    value = torch.tensor([[[[float("nan")]]]])
    result = history_confidence(
        torch.ones_like(value),
        torch.ones_like(value, dtype=torch.bool),
        torch.zeros_like(value, dtype=torch.bool),
        torch.zeros_like(value),
        value,
        torch.ones_like(value),
        torch.zeros_like(value),
    )
    assert not bool(result.valid_mask.item())
    assert result.confidence.item() == 0.0
    assert torch.isfinite(result.confidence).all()
