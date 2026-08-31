"""Masked source selection for FFS, VGGT, and warped history disparity."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_source_softmax(
    logits: Tensor,
    valid_mask: Tensor,
    *,
    fallback_source: int = 0,
) -> Tensor:
    """Apply a source-axis softmax while assigning invalid sources zero weight.

    Args:
        logits: Source logits shaped ``[B, S, H, W]``.
        valid_mask: Boolean-compatible source validity shaped ``[B, S, H, W]``.
        fallback_source: Source selected at pixels where every source is invalid.
            Source zero is FFS in this project. Its disparity is sanitized before
            mixing, so even a completely invalid pixel has a finite result.

    Returns:
        Normalized weights shaped ``[B, S, H, W]``. Invalid sources receive
        exactly zero. The selected fallback receives one at all-invalid pixels.
    """

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B,S,H,W], got {logits.shape}")
    if valid_mask.shape != logits.shape:
        raise ValueError(
            f"valid_mask must match logits shape {logits.shape}, got {valid_mask.shape}"
        )
    source_count = logits.shape[1]
    if not 0 <= fallback_source < source_count:
        raise ValueError(
            f"fallback_source must be in [0,{source_count}), got {fallback_source}"
        )

    valid = valid_mask.to(dtype=torch.bool)
    finite_logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    any_valid = valid.any(dim=1, keepdim=True)

    fallback_valid = torch.zeros_like(valid)
    fallback_valid[:, fallback_source : fallback_source + 1] = True
    effective_valid = torch.where(any_valid, valid, fallback_valid)

    negative_large = torch.finfo(finite_logits.dtype).min
    masked_logits = finite_logits.masked_fill(~effective_valid, negative_large)
    weights = torch.softmax(masked_logits, dim=1)
    # Multiplication and renormalization make exact-zero masking explicit even
    # on low precision devices where a very negative exp implementation varies.
    weights = weights * effective_valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    return weights / denominator


class SourceGatingHead(nn.Module):
    """Predict per-pixel logits for ``[FFS, VGGT, history]`` sources."""

    source_count = 3

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.logits = nn.Conv2d(in_channels, self.source_count, kernel_size=3, padding=1)
        # Metric FFS is the owner. A modest initial prior favors it while masks
        # still exclude invalid FFS pixels exactly.
        nn.init.zeros_(self.logits.weight)
        with torch.no_grad():
            self.logits.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))

    def forward(self, feature_lr: Tensor, valid_mask: Tensor) -> Tensor:
        """Return masked source weights shaped ``[B,3,H,W]``."""

        logits = self.logits(feature_lr)
        return masked_source_softmax(logits, valid_mask, fallback_source=0)
