"""Adapter for the frozen Fast-FoundationStereo (FFS) backbone.

The upstream model is intentionally treated as an opaque, frozen dependency.
Auxiliary tensors are observed with forward hooks; this module never patches or
modifies files below ``third_party/Fast-FoundationStereo``.

Coordinate and unit contract
----------------------------
``disparity_lr_px`` is expressed in pixels of the image passed to FFS.  For an
FFS input downsampled from the target image by ``spatial_scale``, the target-HR
unit value is ``disparity_hr_px = spatial_scale * disparity_lr_px``.  The
upstream recurrent update lives on a 1/4-resolution grid, so its disparity delta
is multiplied by four before it is reported in input-image pixel units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Padding:
    """Symmetric spatial padding in ``torch.nn.functional.pad`` order."""

    left: int
    right: int
    top: int
    bottom: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.right, self.top, self.bottom

    def apply(self, tensor: Tensor) -> Tensor:
        """Replicate-pad a ``[..., H, W]`` tensor."""

        if tensor.ndim < 2:
            raise ValueError(f"expected at least 2 dimensions, got {tensor.shape}")
        if min(tensor.shape[-2:]) <= 0:
            raise ValueError(f"spatial dimensions must be non-empty, got {tensor.shape}")
        if not any(self.as_tuple):
            return tensor
        return F.pad(tensor, self.as_tuple, mode="replicate")

    def remove(self, tensor: Tensor) -> Tensor:
        """Remove this padding from a ``[..., H, W]`` tensor."""

        if tensor.ndim < 2:
            raise ValueError(f"expected at least 2 dimensions, got {tensor.shape}")
        height, width = tensor.shape[-2:]
        y_stop = height - self.bottom
        x_stop = width - self.right
        if self.top > y_stop or self.left > x_stop:
            raise ValueError(
                f"padding {self.as_tuple} is larger than tensor shape {tensor.shape}"
            )
        return tensor[..., self.top:y_stop, self.left:x_stop]


@dataclass(frozen=True)
class FFSOutput:
    """Full-resolution-on-the-FFS-grid output from :class:`FFSAdapter`.

    All dense tensors have shape ``[B, 1, H_lr, W_lr]``.  Confidence and
    entropy are dimensionless.  The remaining numeric suffixes state their
    disparity unit explicitly.
    """

    disparity_lr_px: Tensor
    disparity_hr_px: Tensor
    confidence: Tensor
    entropy: Tensor
    last_update_magnitude_input_px: Tensor
    left_right_error_lr_px: Tensor | None
    valid_mask: Tensor
    metadata: Mapping[str, Any]

    # Compatibility aliases for the names in the original design document.
    @property
    def disparity_lr(self) -> Tensor:
        return self.disparity_lr_px

    @property
    def disparity_hr_unit(self) -> Tensor:
        return self.disparity_hr_px

    @property
    def last_update_magnitude(self) -> Tensor:
        return self.last_update_magnitude_input_px

    @property
    def left_right_error(self) -> Tensor | None:
        return self.left_right_error_lr_px


def padding_to_multiple(
    height: int,
    width: int,
    *,
    divisor: int = 32,
) -> Padding:
    """Return symmetric padding that makes ``height`` and ``width`` divisible.

    When an odd number of pixels is needed, the extra pixel is placed on the
    bottom or right, matching upstream FFS ``InputPadder(mode="sintel")``.
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    pad_height = (-height) % divisor
    pad_width = (-width) % divisor
    return Padding(
        left=pad_width // 2,
        right=pad_width - pad_width // 2,
        top=pad_height // 2,
        bottom=pad_height - pad_height // 2,
    )


def disparity_lr_to_hr_pixels(disparity_lr_px: Tensor, spatial_scale: float) -> Tensor:
    """Convert disparity from FFS-input pixels to target-HR pixels."""

    if not math.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError(f"spatial_scale must be finite and positive, got {spatial_scale}")
    return disparity_lr_px * spatial_scale


def quarter_grid_delta_to_input_pixels(delta_quarter_px: Tensor) -> Tensor:
    """Convert signed upstream 1/4-grid disparity delta to FFS-input pixels."""

    return delta_quarter_px * 4.0


def normalized_cost_entropy(cost_logits: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Compute normalized entropy from raw upstream classifier logits.

    Args:
        cost_logits: Upstream classifier output with shape ``[B, 1, D, H, W]``.
        eps: Lower numerical bound used inside the logarithm.

    Returns:
        Tensor of shape ``[B, 1, H, W]`` in ``[0, 1]``.  A degenerate
        one-bin distribution has entropy zero.
    """

    if cost_logits.ndim != 5 or cost_logits.shape[1] != 1:
        raise ValueError(
            "classifier logits must have shape [B, 1, D, H, W], "
            f"got {tuple(cost_logits.shape)}"
        )
    disparity_bins = cost_logits.shape[2]
    if disparity_bins <= 0:
        raise ValueError("classifier logits must contain at least one disparity bin")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if disparity_bins == 1:
        return torch.zeros_like(cost_logits[:, :, 0], dtype=torch.float32)

    probability = torch.softmax(cost_logits.float().squeeze(1), dim=1)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    entropy = entropy / math.log(disparity_bins)
    return entropy.clamp(0.0, 1.0)


def make_right_reference_pair(left_rgb: Tensor, right_rgb: Tensor) -> tuple[Tensor, Tensor]:
    """Create the FFS pair used to estimate right-view positive disparity.

    For rectified stereo with positive left disparity, right-view inference must
    swap the cameras *and* flip both images horizontally.  A plain ``FFS(R,L)``
    asks the positive-disparity model to search in the wrong direction.
    """

    if left_rgb.shape != right_rgb.shape:
        raise ValueError(
            f"left/right shapes must match, got {left_rgb.shape} and {right_rgb.shape}"
        )
    return torch.flip(right_rgb, dims=(-1,)), torch.flip(left_rgb, dims=(-1,))


def restore_right_reference_disparity(disparity_flipped_px: Tensor) -> Tensor:
    """Map disparity inferred in horizontally flipped coordinates back."""

    return torch.flip(disparity_flipped_px, dims=(-1,))


def left_right_consistency_error(
    disparity_left_lr_px: Tensor,
    disparity_right_lr_px: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate right disparity at ``x_right = x_left - d_left``.

    Args:
        disparity_left_lr_px: Positive left-reference disparity ``[B,1,H,W]``
            in FFS-input pixels.
        disparity_right_lr_px: Positive right-reference disparity ``[B,1,H,W]``
            in the same units, already restored from flipped coordinates.

    Returns:
        ``(absolute_error_lr_px, valid_mask)``.  Invalid samples have ``inf``
        error and a false mask.  Sampling is horizontal linear interpolation at
        the same image row.
    """

    if disparity_left_lr_px.ndim != 4 or disparity_left_lr_px.shape[1] != 1:
        raise ValueError(
            "left disparity must have shape [B, 1, H, W], "
            f"got {tuple(disparity_left_lr_px.shape)}"
        )
    if disparity_right_lr_px.shape != disparity_left_lr_px.shape:
        raise ValueError(
            "right disparity shape must match left disparity, "
            f"got {tuple(disparity_right_lr_px.shape)} and "
            f"{tuple(disparity_left_lr_px.shape)}"
        )

    batch, _, height, width = disparity_left_lr_px.shape
    x_left = torch.arange(
        width,
        device=disparity_left_lr_px.device,
        dtype=disparity_left_lr_px.dtype,
    ).view(1, 1, 1, width)
    x_right = x_left - disparity_left_lr_px
    coordinate_valid = torch.isfinite(x_right) & (x_right >= 0) & (x_right <= width - 1)
    x_safe = torch.where(coordinate_valid, x_right, torch.zeros_like(x_right))
    x_floor = torch.floor(x_safe)
    fraction = x_safe - x_floor
    index_floor = x_floor.long().clamp(0, width - 1)
    index_ceil = (index_floor + 1).clamp(0, width - 1)

    right_floor = torch.gather(disparity_right_lr_px, dim=3, index=index_floor)
    right_ceil = torch.gather(disparity_right_lr_px, dim=3, index=index_ceil)
    # Avoid ``inf * 0 -> nan`` at exact integer coordinates.
    sampled_right = torch.where(
        fraction == 0,
        right_floor,
        right_floor * (1.0 - fraction) + right_ceil * fraction,
    )

    # At integer coordinates the ceil sample has zero weight and need not be
    # finite.  This matters when invalid values are represented as NaN/Inf.
    support_valid = torch.isfinite(right_floor) & (
        (fraction == 0) | torch.isfinite(right_ceil)
    )
    valid = (
        coordinate_valid
        & support_valid
        & torch.isfinite(disparity_left_lr_px)
        & (disparity_left_lr_px > 0)
    )
    error = (disparity_left_lr_px - sampled_right).abs()
    error = torch.where(valid, error, torch.full_like(error, torch.inf))
    assert error.shape == (batch, 1, height, width)
    return error, valid


class FFSAdapter(nn.Module):
    """Frozen adapter around an instantiated upstream FFS model.

    The input tensors are RGB ``[B,3,H_lr,W_lr]`` in the upstream range
    ``[0,255]``.  The adapter pads to a multiple of 32, invokes the model under
    :func:`torch.inference_mode`, captures cost/update auxiliaries with hooks,
    and returns tensors unpadded to ``H_lr x W_lr``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        spatial_scale: float = 2.0,
        iterations: int = 4,
        volume_backend: str = "pytorch1",
        update_tau_input_px: float = 1.0,
        lr_sigma_lr_px: float = 1.0,
        entropy_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not hasattr(model, "classifier"):
            raise TypeError("upstream FFS model has no 'classifier' module for hook capture")
        if not hasattr(model, "update_block"):
            raise TypeError("upstream FFS model has no 'update_block' module for hook capture")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if update_tau_input_px <= 0 or lr_sigma_lr_px <= 0:
            raise ValueError("confidence temperature/sigma values must be positive")
        # Validate scale at construction time while keeping conversion logic in
        # one public helper.
        disparity_lr_to_hr_pixels(torch.tensor(0.0), spatial_scale)

        self.model = model
        self.spatial_scale = float(spatial_scale)
        self.iterations = int(iterations)
        self.volume_backend = volume_backend
        self.update_tau_input_px = float(update_tau_input_px)
        self.lr_sigma_lr_px = float(lr_sigma_lr_px)
        self.entropy_eps = float(entropy_eps)

        self._classifier_outputs: list[Tensor] = []
        self._update_deltas: list[Tensor] = []
        self._classifier_hook = self.model.classifier.register_forward_hook(
            self._capture_classifier
        )
        self._update_hook = self.model.update_block.register_forward_hook(
            self._capture_update
        )
        self.model.requires_grad_(False)
        self.model.eval()

    def _capture_classifier(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError(f"FFS classifier hook expected Tensor, got {type(output)!r}")
        self._classifier_outputs.append(output.detach())

    def _capture_update(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise TypeError(
                "FFS update_block hook expected tuple/list whose final item is delta_disp"
            )
        delta_quarter_px = output[-1]
        if not isinstance(delta_quarter_px, Tensor):
            raise TypeError(
                "FFS update_block final output must be a Tensor, "
                f"got {type(delta_quarter_px)!r}"
            )
        # Do not retain recurrent hidden states from every iteration.  Only the
        # delta is part of this adapter's confidence contract.
        self._update_deltas.append(delta_quarter_px.detach())

    def train(self, mode: bool = True) -> "FFSAdapter":
        """Keep this cache-only adapter and its upstream model in eval mode."""

        super().train(False)
        self.model.eval()
        return self

    def close(self) -> None:
        """Remove hooks when the adapter will no longer be used."""

        self._classifier_hook.remove()
        self._update_hook.remove()

    @staticmethod
    def _validate_images(left_rgb: Tensor, right_rgb: Tensor) -> None:
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(
                f"left/right shapes must match, got {left_rgb.shape} and {right_rgb.shape}"
            )
        if left_rgb.ndim != 4 or left_rgb.shape[1] != 3:
            raise ValueError(
                "FFS RGB inputs must have shape [B, 3, H, W], "
                f"got {tuple(left_rgb.shape)}"
            )
        if not left_rgb.is_floating_point() or not right_rgb.is_floating_point():
            raise TypeError("FFS RGB inputs must be floating point tensors in [0, 255]")
        if left_rgb.device != right_rgb.device:
            raise ValueError("left and right RGB inputs must be on the same device")
        if not torch.isfinite(left_rgb).all() or not torch.isfinite(right_rgb).all():
            raise ValueError("FFS RGB inputs must be finite")

    def _reset_captures(self) -> None:
        self._classifier_outputs.clear()
        self._update_deltas.clear()

    def _extract_auxiliaries(
        self,
        *,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        if len(self._classifier_outputs) != 1:
            raise RuntimeError(
                "expected exactly one FFS classifier call, captured "
                f"{len(self._classifier_outputs)}"
            )
        if len(self._update_deltas) != self.iterations:
            raise RuntimeError(
                f"expected {self.iterations} FFS update calls, captured "
                f"{len(self._update_deltas)}"
            )

        entropy_quarter = normalized_cost_entropy(
            self._classifier_outputs[-1], eps=self.entropy_eps
        )
        delta_quarter_px = self._update_deltas[-1]
        if delta_quarter_px.ndim != 4 or delta_quarter_px.shape[1] != 1:
            raise ValueError(
                "FFS delta_disp must have shape [B,1,H/4,W/4], "
                f"got {tuple(delta_quarter_px.shape)}"
            )

        last_update_magnitude_input_px_quarter = (
            quarter_grid_delta_to_input_pixels(delta_quarter_px.detach()).abs().float()
        )
        entropy = F.interpolate(
            entropy_quarter,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        update_magnitude = F.interpolate(
            last_update_magnitude_input_px_quarter,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return entropy, update_magnitude

    def _infer_once(
        self,
        left_rgb: Tensor,
        right_rgb: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Padding]:
        padding = padding_to_multiple(*left_rgb.shape[-2:], divisor=32)
        left_padded = padding.apply(left_rgb)
        right_padded = padding.apply(right_rgb)
        self._reset_captures()
        self.model.eval()
        disparity_padded = self.model(
            left_padded,
            right_padded,
            iters=self.iterations,
            test_mode=True,
            optimize_build_volume=self.volume_backend,
        )
        if not isinstance(disparity_padded, Tensor):
            raise TypeError(
                "FFS test_mode=True must return a disparity Tensor, "
                f"got {type(disparity_padded)!r}"
            )
        expected_shape = (left_rgb.shape[0], 1, *left_padded.shape[-2:])
        if tuple(disparity_padded.shape) != expected_shape:
            raise ValueError(
                f"FFS disparity shape must be {expected_shape}, "
                f"got {tuple(disparity_padded.shape)}"
            )
        entropy_padded, update_magnitude_padded = self._extract_auxiliaries(
            output_size=left_padded.shape[-2:]
        )
        return (
            padding.remove(disparity_padded.float()),
            padding.remove(entropy_padded),
            padding.remove(update_magnitude_padded),
            padding,
        )

    @torch.inference_mode()
    def forward(
        self,
        left_rgb: Tensor,
        right_rgb: Tensor,
        *,
        right_left_check: bool = False,
    ) -> FFSOutput:
        """Run frozen FFS and optionally compute a right-left consistency map."""

        self._validate_images(left_rgb, right_rgb)
        disparity_lr_px, entropy, update_magnitude_input_px, padding = self._infer_once(
            left_rgb, right_rgb
        )
        cost_confidence = (1.0 - entropy).clamp(0.0, 1.0)
        update_confidence = torch.exp(
            -update_magnitude_input_px / self.update_tau_input_px
        )

        width = disparity_lr_px.shape[-1]
        x = torch.arange(
            width,
            dtype=disparity_lr_px.dtype,
            device=disparity_lr_px.device,
        ).view(1, 1, 1, width)
        valid_mask = (
            torch.isfinite(disparity_lr_px)
            & (disparity_lr_px > 0)
            & ((x - disparity_lr_px) >= 0)
        )

        left_right_error_lr_px: Tensor | None = None
        lr_confidence = torch.ones_like(cost_confidence)
        if right_left_check:
            right_reference, left_target = make_right_reference_pair(left_rgb, right_rgb)
            disparity_right_flipped_px, _, _, _ = self._infer_once(
                right_reference, left_target
            )
            disparity_right_lr_px = restore_right_reference_disparity(
                disparity_right_flipped_px
            )
            left_right_error_lr_px, lr_valid = left_right_consistency_error(
                disparity_lr_px, disparity_right_lr_px
            )
            lr_confidence = torch.where(
                lr_valid,
                torch.exp(-left_right_error_lr_px / self.lr_sigma_lr_px),
                torch.zeros_like(left_right_error_lr_px),
            )
            valid_mask = valid_mask & lr_valid

        confidence = cost_confidence * update_confidence * lr_confidence
        confidence = torch.where(
            valid_mask & torch.isfinite(confidence),
            confidence.clamp(0.0, 1.0),
            torch.zeros_like(confidence),
        )
        disparity_hr_px = disparity_lr_to_hr_pixels(
            disparity_lr_px, self.spatial_scale
        )
        metadata: dict[str, Any] = {
            "input_shape_bchw": list(left_rgb.shape),
            "padding_lrtb": list(padding.as_tuple),
            "padded_shape_hw": [
                left_rgb.shape[-2] + padding.top + padding.bottom,
                left_rgb.shape[-1] + padding.left + padding.right,
            ],
            "padding_divisor": 32,
            "spatial_scale": self.spatial_scale,
            "disparity_lr_unit": "FFS input pixels",
            "disparity_hr_unit": "target HR pixels",
            "update_delta_source_unit": "1/4-grid pixels",
            "last_update_magnitude_unit": "FFS input pixels",
            "update_grid_to_input_scale": 4.0,
            "iterations": self.iterations,
            "volume_backend": self.volume_backend,
            "right_left_check": right_left_check,
            "frozen": True,
            "inference_mode": True,
        }
        return FFSOutput(
            disparity_lr_px=disparity_lr_px,
            disparity_hr_px=disparity_hr_px,
            confidence=confidence,
            entropy=entropy,
            last_update_magnitude_input_px=update_magnitude_input_px,
            left_right_error_lr_px=left_right_error_lr_px,
            valid_mask=valid_mask,
            metadata=metadata,
        )


__all__ = [
    "FFSAdapter",
    "FFSOutput",
    "Padding",
    "disparity_lr_to_hr_pixels",
    "left_right_consistency_error",
    "make_right_reference_pair",
    "normalized_cost_entropy",
    "padding_to_multiple",
    "quarter_grid_delta_to_input_pixels",
    "restore_right_reference_disparity",
]
