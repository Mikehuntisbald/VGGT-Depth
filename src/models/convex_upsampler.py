"""Learned convex x2 upsampling without changing disparity units."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class ConvexUpsampler(nn.Module):
    """RAFT-style normalized 3x3 convex upsampling.

    The input values are disparity in HR-pixel units sampled on an LR grid.
    Consequently the spatial grid grows by ``scale`` but disparity values are
    **not** multiplied by ``scale`` here.
    """

    neighborhood_size = 3

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"the MVP ConvexUpsampler supports x2 only, got x{scale}")
        self.scale = scale

    @property
    def mask_channels(self) -> int:
        return self.neighborhood_size**2 * self.scale**2

    def forward(self, disparity_hr_px_lr_grid: Tensor, mask_logits: Tensor) -> Tensor:
        """Upsample ``[B,1,H,W]`` HR-pixel disparity to ``[B,1,2H,2W]``."""

        if disparity_hr_px_lr_grid.ndim != 4 or disparity_hr_px_lr_grid.shape[1] != 1:
            raise ValueError(
                "disparity_hr_px_lr_grid must have shape [B,1,H,W], got "
                f"{disparity_hr_px_lr_grid.shape}"
            )
        batch, _, height, width = disparity_hr_px_lr_grid.shape
        expected_mask_shape = (batch, self.mask_channels, height, width)
        if mask_logits.shape != expected_mask_shape:
            raise ValueError(
                f"mask_logits must have shape {expected_mask_shape}, got {mask_logits.shape}"
            )

        weights = mask_logits.reshape(
            batch,
            1,
            self.neighborhood_size**2,
            self.scale,
            self.scale,
            height,
            width,
        )
        weights = torch.softmax(weights, dim=2)

        # Replicate padding avoids introducing artificial zero-disparity borders.
        padded = functional.pad(disparity_hr_px_lr_grid, (1, 1, 1, 1), mode="replicate")
        neighborhoods = functional.unfold(padded, kernel_size=self.neighborhood_size)
        neighborhoods = neighborhoods.reshape(
            batch,
            1,
            self.neighborhood_size**2,
            1,
            1,
            height,
            width,
        )
        upsampled = torch.sum(weights * neighborhoods, dim=2)
        upsampled = upsampled.permute(0, 1, 4, 2, 5, 3).contiguous()
        return upsampled.reshape(batch, 1, height * self.scale, width * self.scale)
