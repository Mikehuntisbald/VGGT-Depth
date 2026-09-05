"""Online, differentiable VGGT-Omega geometry features for joint training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TrainableVGGTOmegaOutput:
    """Current-frame scale-ambiguous geometry inferred from a causal prefix."""

    depth_current_arbitrary: Tensor
    confidence_current_unbounded: Tensor
    geometry_current: Tensor
    patch_grid_hw: tuple[int, int]


def load_vggt_omega(
    checkpoint: str | Path,
    repo: str | Path,
    *,
    enable_camera: bool = False,
) -> nn.Module:
    """Load the official 1B checkpoint while retaining a differentiable graph."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    repo_path = Path(repo).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint_path}")
    if not (repo_path / "vggt_omega" / "models" / "vggt_omega.py").is_file():
        raise FileNotFoundError(f"VGGT-Omega source is missing: {repo_path}")
    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    from vggt_omega.models import VGGTOmega  # type: ignore[import-not-found]

    model = VGGTOmega(enable_camera=enable_camera, enable_depth=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("camera_head.",) if not enable_camera else ()
    if incompatible.missing_keys:
        raise RuntimeError(f"VGGT checkpoint is missing keys: {incompatible.missing_keys[:8]}")
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"VGGT checkpoint has unexpected keys: {unexpected[:8]}")
    model.requires_grad_(True)
    model.train()
    return model


class TrainableVGGTOmega(nn.Module):
    """Extract the final causal target's dense depth and geometry tokens.

    Inter-frame attention inside upstream VGGT is bidirectional over the input
    set.  This wrapper preserves causality by accepting a prefix ending at the
    target and returning only that final target.  Callers must never append
    future frames to ``left_rgb_prefix``.
    """

    def __init__(self, model: nn.Module, *, geometry_channels: int = 192) -> None:
        super().__init__()
        if geometry_channels <= 0:
            raise ValueError("geometry_channels must be positive")
        if not hasattr(model, "aggregator") or not hasattr(model, "dense_head"):
            raise TypeError("VGGT model must expose aggregator and dense_head")
        if model.dense_head is None:
            raise TypeError("VGGT model was constructed without a dense depth head")
        self.model = model
        token_channels = 2 * int(model.aggregator.patch_embed.embed_dim)
        self.geometry_projection = nn.Sequential(
            nn.Conv2d(token_channels, geometry_channels, kernel_size=1),
            nn.GroupNorm(8, geometry_channels),
            nn.GELU(),
        )
        self.geometry_channels = int(geometry_channels)

    def forward(self, left_rgb_prefix: Tensor) -> TrainableVGGTOmegaOutput:
        if left_rgb_prefix.ndim != 5 or left_rgb_prefix.shape[2] != 3:
            raise ValueError(
                "left_rgb_prefix must have shape [B,T,3,H,W], got "
                f"{tuple(left_rgb_prefix.shape)}"
            )
        if not left_rgb_prefix.is_floating_point():
            raise TypeError("VGGT RGB must be floating point")
        batch, frames, _, height, width = left_rgb_prefix.shape
        patch_size = int(self.model.aggregator.patch_size)
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"VGGT height/width must be divisible by patch size {patch_size}"
            )
        if frames <= 0:
            raise ValueError("causal prefix must contain at least one frame")

        autocast_enabled = left_rgb_prefix.device.type == "cuda"
        with torch.autocast(
            device_type=left_rgb_prefix.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            aggregated, patch_token_start = self.model.aggregator(left_rgb_prefix)
        final_tokens = aggregated[-1]
        if final_tokens is None:
            raise RuntimeError("VGGT aggregator did not retain its final tokens")

        # Dense depth needs the four cached scales, but decoding only the final
        # frame avoids retaining unnecessary per-frame decoder activations.
        target_layers = [
            None if layer is None else layer[:, -1:].contiguous()
            for layer in aggregated
        ]
        target_images = left_rgb_prefix[:, -1:].contiguous()
        with torch.autocast(
            device_type=left_rgb_prefix.device.type,
            enabled=False,
        ):
            depth, confidence = self.model.dense_head(
                target_layers,
                images=target_images.float(),
                patch_token_start=patch_token_start,
                frames_chunk_size=1,
            )

        patch_height, patch_width = height // patch_size, width // patch_size
        target_patch_tokens = final_tokens[:, -1, patch_token_start:]
        expected_tokens = patch_height * patch_width
        if target_patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"VGGT patch count mismatch: {target_patch_tokens.shape[1]} != {expected_tokens}"
            )
        feature = target_patch_tokens.transpose(1, 2).reshape(
            batch, target_patch_tokens.shape[-1], patch_height, patch_width
        )
        feature = self.geometry_projection(feature)
        return TrainableVGGTOmegaOutput(
            depth_current_arbitrary=depth[:, 0].permute(0, 3, 1, 2).contiguous(),
            confidence_current_unbounded=confidence[:, 0].unsqueeze(1).contiguous(),
            geometry_current=feature,
            patch_grid_hw=(patch_height, patch_width),
        )


__all__ = [
    "TrainableVGGTOmega",
    "TrainableVGGTOmegaOutput",
    "load_vggt_omega",
]
