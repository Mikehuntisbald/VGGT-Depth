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

    @classmethod
    def bilinear_weights(
        cls,
        scale: int = 2,
        *,
        epsilon: float = 1.0e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return per-subpixel 3x3 weights equivalent to x2 bilinear resize.

        The returned tensor is laid out as ``[3*3, scale, scale]`` so it can
        be flattened directly into the convex-mask channel order used by
        :meth:`forward`.  ``align_corners=False`` maps the first/second HR
        samples to LR offsets ``-0.25``/``+0.25``; replicate padding in
        :meth:`forward` then gives the same edge behaviour as PyTorch's
        bilinear interpolation.  Zero support entries receive a tiny floor
        before normalization by softmax, keeping the initialization finite
        and the corresponding mask channels trainable.

        ``epsilon`` is intentionally small: the resulting output differs from
        ``torch.nn.functional.interpolate(..., mode="bilinear")`` only at
        floating-point noise level while avoiding ``-inf`` mask biases.
        """

        if scale != 2:
            raise ValueError(f"the MVP ConvexUpsampler supports x2 only, got x{scale}")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not torch.isfinite(torch.tensor(float(epsilon)))
            or float(epsilon) <= 0.0
        ):
            raise ValueError("epsilon must be finite and positive")

        # For HR samples r=0,1, the align_corners=False source coordinate is
        # i-0.25 and i+0.25 respectively.  The corresponding 1-D weights for
        # offsets (-1, 0, +1) are therefore (.25,.75,0) and (0,.75,.25).
        one_dimensional = torch.tensor(
            [[0.25, 0.75, 0.0], [0.0, 0.75, 0.25]],
            dtype=dtype,
            device=device,
        )
        weights = torch.empty(
            cls.neighborhood_size**2,
            scale,
            scale,
            dtype=dtype,
            device=device,
        )
        for row_phase in range(scale):
            for column_phase in range(scale):
                # ``outer`` is row-major in the same order as unfold's 3x3
                # neighborhood: (-1,-1), (-1,0), ..., (+1,+1).
                weights[:, row_phase, column_phase] = torch.outer(
                    one_dimensional[row_phase], one_dimensional[column_phase]
                ).reshape(-1)
        # Softmax normalizes over the neighborhood dimension.  Add the floor
        # before taking logs so unsupported taps remain finite/trainable, then
        # normalize explicitly so callers of ``bilinear_weights`` see proper
        # convex weights as well.
        weights = weights + float(epsilon)
        return weights / weights.sum(dim=0, keepdim=True)

    @classmethod
    def bilinear_mask_logits(
        cls,
        scale: int = 2,
        *,
        epsilon: float = 1.0e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return finite mask logits whose softmax is bilinear-equivalent."""

        weights = cls.bilinear_weights(
            scale,
            epsilon=epsilon,
            dtype=dtype,
            device=device,
        )
        return torch.log(weights)

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
