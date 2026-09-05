"""Differentiable Fast-FoundationStereo wrapper for joint video training.

This module is deliberately separate from :mod:`backbones.ffs_adapter`.  The
cache adapter is frozen by contract; joint training needs the upstream graph,
its iterative predictions, and the left matching features to remain attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ffs_adapter import make_right_reference_pair, restore_right_reference_disparity


@dataclass(frozen=True, slots=True)
class TrainableStereoOutput:
    """Online stereo predictions for a causal clip.

    Disparities are measured in pixels of the stereo input grid.  Feature
    tensors retain gradients and are ordered from fine to coarse resolution.
    """

    disparity_left_lr_px: Tensor
    disparity_right_lr_px: Tensor | None
    left_features: tuple[Tensor, ...]
    iteration_disparities_left_lr_px: tuple[Tensor, ...]


def load_fast_foundation_stereo(
    checkpoint: str | Path,
    repo: str | Path,
    *,
    iterations: int = 4,
    max_disp: int = 192,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load the pinned serialized FFS model without freezing it."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if max_disp <= 0 or max_disp % 4:
        raise ValueError("max_disp must be positive and divisible by four")
    if amp_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    repo_path = Path(repo).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FFS checkpoint does not exist: {checkpoint_path}")
    if not (repo_path / "core" / "foundation_stereo.py").is_file():
        raise FileNotFoundError(f"Fast-FoundationStereo source is missing: {repo_path}")

    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    import Utils as upstream_utils  # type: ignore[import-not-found]
    import core.foundation_stereo  # type: ignore[import-not-found] # noqa: F401

    # Upstream uses a module-level AMP dtype inside nested autocast scopes.
    upstream_utils.AMP_DTYPE = amp_dtype
    submodule = sys.modules.get("core.submodule")
    if submodule is not None:
        setattr(submodule, "AMP_DTYPE", amp_dtype)

    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"serialized FFS checkpoint did not contain a module: {type(model)!r}")
    if not hasattr(model, "args") or not hasattr(model, "feature"):
        raise TypeError("serialized FFS model lacks args/feature interfaces")
    if model.args.get("normalize") is None:
        raise ValueError("FFS checkpoint lacks args.normalize; joint training refuses to guess")
    model.args.valid_iters = int(iterations)
    model.args.max_disp = int(max_disp)
    model.args.mixed_precision = True
    model.requires_grad_(True)
    model.train()
    return model


class TrainableFastFoundationStereo(nn.Module):
    """Run one shared FFS model over all stereo times in a causal clip."""

    def __init__(
        self,
        model: nn.Module,
        *,
        iterations: int = 4,
        max_disp: int = 192,
        predict_right: bool = True,
        volume_backend: str = "pytorch1",
    ) -> None:
        super().__init__()
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if max_disp <= 0 or max_disp % 4:
            raise ValueError("max_disp must be positive and divisible by four")
        if volume_backend not in {"pytorch1", "triton"}:
            raise ValueError("volume_backend must be pytorch1 or triton")
        if not hasattr(model, "feature"):
            raise TypeError("FFS model must expose its feature module")
        self.model = model
        self.iterations = int(iterations)
        self.max_disp = int(max_disp)
        self.predict_right = bool(predict_right)
        self.volume_backend = volume_backend
        # The serialized FFS checkpoint contains SyncBatchNorm layers.  Frames
        # are flattened into B*T for one efficient backbone call, so training
        # batch statistics would otherwise couple a historical frame to later
        # frames in the causal prefix.  Keep pretrained running statistics
        # fixed while leaving BN affine parameters and the full backbone
        # differentiable.
        self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> "TrainableFastFoundationStereo":
        super().train(mode)
        if mode:
            self._freeze_batch_norm_statistics()
        return self

    @staticmethod
    def _validate_clip(rgb: Tensor) -> tuple[int, int, int, int]:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError(f"stereo clip must have shape [B,T,3,H,W], got {tuple(rgb.shape)}")
        if not rgb.is_floating_point():
            raise TypeError("stereo RGB must be floating point")
        batch, frames, _, height, width = rgb.shape
        if min(batch, frames, height, width) <= 0:
            raise ValueError("stereo clip dimensions must be non-empty")
        if height % 32 or width % 32:
            raise ValueError(
                f"FFS input height/width must be divisible by 32, got {(height, width)}"
            )
        return batch, frames, height, width

    def _run_once(
        self, left: Tensor, right: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        captured: list[Any] = []

        def capture_features(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(output)

        handle = self.model.feature.register_forward_hook(capture_features)
        try:
            result = self.model(
                left,
                right,
                iters=self.iterations,
                test_mode=False,
                low_memory=False,
                optimize_build_volume=self.volume_backend,
            )
        finally:
            handle.remove()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("training-mode FFS must return (initial, predictions)")
        _initial, predictions = result
        if not isinstance(predictions, (list, tuple)) or not predictions:
            raise TypeError("training-mode FFS returned no iterative predictions")
        if len(captured) != 1 or not isinstance(captured[0], (list, tuple)):
            raise RuntimeError("FFS feature hook did not capture one feature pyramid")
        batch = left.shape[0]
        left_features = tuple(feature[:batch] for feature in captured[0])
        prediction_tuple = tuple(prediction for prediction in predictions)
        return prediction_tuple[-1], left_features, prediction_tuple

    def forward(self, left_rgb: Tensor, right_rgb: Tensor) -> TrainableStereoOutput:
        """Predict both reference directions from RGB clips in ``[0,1]``."""

        left_shape = self._validate_clip(left_rgb)
        right_shape = self._validate_clip(right_rgb)
        if left_shape != right_shape or left_rgb.shape != right_rgb.shape:
            raise ValueError("left and right clips must have identical shapes")
        if left_rgb.device != right_rgb.device:
            raise ValueError("left and right clips must share one device")
        batch, frames, height, width = left_shape
        left_flat = left_rgb.reshape(batch * frames, 3, height, width) * 255.0
        right_flat = right_rgb.reshape(batch * frames, 3, height, width) * 255.0

        left_disparity, features, predictions = self._run_once(left_flat, right_flat)
        left_disparity = left_disparity.reshape(batch, frames, 1, height, width)
        clip_features = tuple(
            feature.reshape(batch, frames, *feature.shape[1:]) for feature in features
        )
        iteration_predictions = tuple(
            prediction.reshape(batch, frames, 1, height, width)
            for prediction in predictions
        )

        right_disparity: Tensor | None = None
        if self.predict_right:
            right_reference, left_target = make_right_reference_pair(left_flat, right_flat)
            right_flipped, _features, _predictions = self._run_once(
                right_reference, left_target
            )
            right_disparity = restore_right_reference_disparity(right_flipped).reshape(
                batch, frames, 1, height, width
            )

        return TrainableStereoOutput(
            disparity_left_lr_px=left_disparity,
            disparity_right_lr_px=right_disparity,
            left_features=clip_features,
            iteration_disparities_left_lr_px=iteration_predictions,
        )


def half_resolution_stereo_images(left_rgb: Tensor, right_rgb: Tensor) -> tuple[Tensor, Tensor]:
    """Create the x2 observation grid with the project's half-pixel convention."""

    if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 5:
        raise ValueError("left/right RGB must be matching [B,T,3,H,W] tensors")
    height, width = left_rgb.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("HR height and width must be even")
    flat_shape = (-1, 3, height, width)
    left = F.interpolate(
        left_rgb.reshape(flat_shape),
        size=(height // 2, width // 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    right = F.interpolate(
        right_rgb.reshape(flat_shape),
        size=(height // 2, width // 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        left.reshape(*left_rgb.shape[:2], 3, height // 2, width // 2),
        right.reshape(*right_rgb.shape[:2], 3, height // 2, width // 2),
    )


__all__ = [
    "TrainableFastFoundationStereo",
    "TrainableStereoOutput",
    "half_resolution_stereo_images",
    "load_fast_foundation_stereo",
]
