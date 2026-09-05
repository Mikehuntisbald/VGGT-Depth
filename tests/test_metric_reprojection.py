from __future__ import annotations

import torch

from geometry.metric_reprojection import (
    stereo_reproject_right_to_left,
    temporal_reproject_previous_to_current,
)


def _K(batch: int, height: int, width: int) -> torch.Tensor:
    matrix = torch.tensor(
        [[8.0, 0.0, (width - 1) / 2], [0.0, 8.0, (height - 1) / 2], [0.0, 0.0, 1.0]]
    )
    return matrix.unsqueeze(0).repeat(batch, 1, 1)


def test_stereo_reprojection_uses_positive_left_disparity() -> None:
    left = torch.zeros(1, 3, 2, 5)
    right = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5).expand(1, 3, 2, 5)
    disparity = torch.ones(1, 1, 2, 5, requires_grad=True)
    result = stereo_reproject_right_to_left(left, right, disparity)
    assert result.valid_mask[0, 0, 0].tolist() == [False, True, True, True, True]
    torch.testing.assert_close(result.image[0, 0, 0, 1:], torch.arange(4, dtype=torch.float32))
    result.image.sum().backward()
    assert disparity.grad is not None


def test_identity_temporal_reprojection_preserves_image_and_depth_gradient() -> None:
    image = torch.rand(1, 3, 4, 6)
    inverse = torch.full((1, 1, 4, 6), 0.5, requires_grad=True)
    result = temporal_reproject_previous_to_current(
        image,
        image,
        inverse,
        _K(1, 4, 6),
        _K(1, 4, 6),
        torch.eye(4).unsqueeze(0),
        previous_inverse_depth_m_inv=inverse.detach(),
    )
    assert bool(result.valid_mask.all())
    assert not bool(result.occlusion_mask.any())
    torch.testing.assert_close(result.image, image, atol=1e-6, rtol=1e-6)
    (result.image.square().mean() + result.projected_depth_m.mean()).backward()
    assert inverse.grad is not None and bool(torch.isfinite(inverse.grad).all())


def test_temporal_reprojection_marks_behind_surface_occluded() -> None:
    image = torch.rand(1, 3, 2, 4)
    current_inverse = torch.full((1, 1, 2, 4), 0.25)
    previous_inverse = torch.full((1, 1, 2, 4), 1.0)
    result = temporal_reproject_previous_to_current(
        image,
        image,
        current_inverse,
        _K(1, 2, 4),
        _K(1, 2, 4),
        torch.eye(4).unsqueeze(0),
        previous_inverse_depth_m_inv=previous_inverse,
        relative_depth_tolerance=0.01,
        absolute_depth_tolerance_m=0.01,
    )
    assert bool(result.occlusion_mask.all())
    assert not bool(result.valid_mask.any())
