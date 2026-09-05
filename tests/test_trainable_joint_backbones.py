from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from backbones.trainable_stereo import (
    TrainableFastFoundationStereo,
    half_resolution_stereo_images,
)
from backbones.trainable_vggt_omega import TrainableVGGTOmega


class _DummyStereoFeature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=1)
        self.normalization = nn.BatchNorm2d(8)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        fine = F.avg_pool2d(self.normalization(self.projection(value)), 4)
        return [fine, F.avg_pool2d(fine, 2)]


class _DummyStereo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = _DummyStereoFeature()
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        iters: int,
        test_mode: bool,
        low_memory: bool,
        optimize_build_volume: str,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del test_mode, low_memory, optimize_build_volume
        features = self.feature(torch.cat((left, right)))
        disparity = (left[:, :1] - right[:, :1]).abs() / 255.0 + self.bias
        initial = F.avg_pool2d(disparity, 4)
        return initial, [disparity * float(index + 1) / iters for index in range(iters)]


class _PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim


class _DummyAggregator(nn.Module):
    patch_size = 4

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(embed_dim)
        self.projection = nn.Linear(3, 2 * embed_dim)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor | None], int]:
        batch, frames, _, height, width = images.shape
        pooled = F.avg_pool2d(
            images.reshape(batch * frames, 3, height, width), 4
        ).reshape(batch, frames, 3, -1)
        patches = self.projection(pooled.transpose(2, 3))
        camera = patches.new_zeros(batch, frames, 1, patches.shape[-1])
        tokens = torch.cat((camera, patches), dim=2)
        return [tokens for _ in range(24)], 1


class _DummyDenseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        layers: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layers, patch_token_start, frames_chunk_size
        depth = (images.mean(dim=2, keepdim=False) + 1.0) * self.scale
        return depth.unsqueeze(-1), torch.ones_like(depth)


class _DummyVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _DummyAggregator()
        self.dense_head = _DummyDenseHead()


def test_trainable_stereo_preserves_clip_and_gradient_contract() -> None:
    left = torch.rand(2, 3, 3, 64, 128)
    right = torch.rand_like(left)
    left_lr, right_lr = half_resolution_stereo_images(left, right)
    wrapper = TrainableFastFoundationStereo(
        _DummyStereo(), iterations=2, max_disp=192, predict_right=True
    )
    output = wrapper(left_lr, right_lr)

    assert output.disparity_left_lr_px.shape == (2, 3, 1, 32, 64)
    assert output.disparity_right_lr_px is not None
    assert output.disparity_right_lr_px.shape == output.disparity_left_lr_px.shape
    assert output.left_features[0].shape == (2, 3, 8, 8, 16)
    assert len(output.iteration_disparities_left_lr_px) == 2
    (output.disparity_left_lr_px.mean() + output.left_features[0].mean()).backward()
    assert wrapper.model.bias.grad is not None
    assert wrapper.model.feature.projection.weight.grad is not None
    assert wrapper.model.feature.normalization.weight.grad is not None


def test_trainable_stereo_history_is_invariant_to_later_prefix_frames() -> None:
    wrapper = TrainableFastFoundationStereo(
        _DummyStereo(), iterations=2, max_disp=192, predict_right=True
    ).train()
    assert not wrapper.model.feature.normalization.training
    assert wrapper.model.feature.normalization.weight.requires_grad

    left = torch.rand(1, 3, 3, 32, 64)
    right = torch.rand_like(left)
    changed_left = left.clone()
    changed_right = right.clone()
    changed_left[:, 1:] = torch.rand_like(changed_left[:, 1:])
    changed_right[:, 1:] = torch.rand_like(changed_right[:, 1:])

    first = wrapper(left, right)
    changed = wrapper(changed_left, changed_right)

    torch.testing.assert_close(first.disparity_left_lr_px[:, 0], changed.disparity_left_lr_px[:, 0])
    torch.testing.assert_close(first.disparity_right_lr_px[:, 0], changed.disparity_right_lr_px[:, 0])
    for first_feature, changed_feature in zip(
        first.left_features, changed.left_features, strict=True
    ):
        torch.testing.assert_close(first_feature[:, 0], changed_feature[:, 0])


def test_trainable_vggt_returns_only_causal_prefix_endpoint() -> None:
    wrapper = TrainableVGGTOmega(_DummyVGGT(), geometry_channels=16)
    images = torch.rand(2, 4, 3, 16, 24)
    output = wrapper(images)

    assert output.depth_current_arbitrary.shape == (2, 1, 16, 24)
    assert output.confidence_current_unbounded.shape == (2, 1, 16, 24)
    assert output.geometry_current.shape == (2, 16, 4, 6)
    assert output.patch_grid_hw == (4, 6)
    output.depth_current_arbitrary.mean().backward()
    assert wrapper.model.dense_head.scale.grad is not None
