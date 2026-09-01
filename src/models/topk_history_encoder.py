"""Shared per-candidate encoder for opt-in top-K temporal history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .rgb_encoder import ConvNormAct


@dataclass(frozen=True, slots=True)
class TopKHistoryEncoding:
    """Encoded top-K candidates and their fail-closed weighted aggregate.

    Attributes:
        candidate_feature: Shared-encoder output ``[B,K,32,H,W]``. Invalid
            candidate slots are exactly zero, including after affine layers
            have learned non-zero biases.
        aggregate_feature: Explicit z-aware weighted sum ``[B,32,H,W]``.
            Pixels with no positive valid weight are exactly zero.
        effective_weights: Sanitized, valid-masked, and per-pixel normalized
            source weights ``[B,K,H,W]``. No uniform fallback is introduced.
        aggregate_valid_mask: At least one candidate had a finite positive
            weight after masking, boolean ``[B,1,H,W]``.
    """

    candidate_feature: Tensor
    aggregate_feature: Tensor
    effective_weights: Tensor
    aggregate_valid_mask: Tensor


class TopKHistoryEncoder(nn.Module):
    """Encode arbitrary-K history candidates with one shared 7-channel CNN.

    The seven per-candidate input channels are ordered as disparity (HR pixel
    units), confidence, phase ``du,dv`` (grid pixel units), temporal age
    (frames), explicit z-aware weight, and validity. Candidate count ``K`` is a
    runtime dimension and never changes parameters or state-dict keys.

    This module is intentionally not installed in :class:`FFSOmegaTSR` by
    default. Merely importing or constructing it cannot alter a canonical
    model checkpoint.
    """

    input_channels = 7

    def __init__(
        self,
        *,
        output_channels: int = 32,
        maximum_disparity_hr_px: float = 2048.0,
        maximum_age_frames: float = 32.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(output_channels, bool)
            or not isinstance(output_channels, int)
            or output_channels <= 0
        ):
            raise ValueError("output_channels must be a positive integer")
        for name, value in (
            ("maximum_disparity_hr_px", maximum_disparity_hr_px),
            ("maximum_age_frames", maximum_age_frames),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not torch.isfinite(torch.tensor(float(value)))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and > 0")
        self.output_channels = output_channels
        self.maximum_disparity_hr_px = float(maximum_disparity_hr_px)
        self.maximum_age_frames = float(maximum_age_frames)
        self.candidate_encoder = nn.Sequential(
            ConvNormAct(self.input_channels, output_channels),
            ConvNormAct(output_channels, output_channels),
        )

    @staticmethod
    def _check_scalar_candidates(
        name: str,
        value: Tensor,
        expected_shape: tuple[int, int, int, int],
        *,
        device: torch.device,
    ) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be a real floating-point tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}")

    @staticmethod
    def _sanitize(value: Tensor, *, minimum: float, maximum: float) -> Tensor:
        return torch.nan_to_num(
            value,
            nan=0.0,
            posinf=maximum,
            neginf=minimum,
        ).clamp(minimum, maximum)

    def forward(
        self,
        disparity_hr_px: Tensor,
        confidence: Tensor,
        fractional_phase_grid_px: Tensor,
        temporal_age_frames: Tensor,
        z_aware_weights: Tensor,
        valid_mask: Tensor,
    ) -> TopKHistoryEncoding:
        """Encode and aggregate top-K history on one LR or HR grid.

        Args:
            disparity_hr_px: Candidate disparity ``[B,K,H,W]`` in HR pixels.
            confidence: Candidate confidence ``[B,K,H,W]``.
            fractional_phase_grid_px: Candidate ``du,dv`` phase
                ``[B,K,2,H,W]`` in the active grid's pixels.
            temporal_age_frames: Candidate age ``[B,K,H,W]`` in frames.
            z_aware_weights: Explicit candidate weights ``[B,K,H,W]``. They
                are sanitized to ``[0,1]``, masked, and renormalized while
                preserving their relative values.
            valid_mask: Strict boolean candidate mask ``[B,K,H,W]``.

        Returns:
            :class:`TopKHistoryEncoding`. Input NaN/Inf is replaced or bounded
            before convolution. Invalid candidates and all-invalid aggregates
            are exact zero; the function never substitutes uniform weights.
        """

        if not isinstance(disparity_hr_px, Tensor):
            raise TypeError("disparity_hr_px must be a torch.Tensor")
        if (
            disparity_hr_px.ndim != 4
            or disparity_hr_px.shape[0] <= 0
            or disparity_hr_px.shape[1] <= 0
            or disparity_hr_px.shape[2] <= 0
            or disparity_hr_px.shape[3] <= 0
        ):
            raise ValueError(
                "disparity_hr_px must have positive shape [B,K,H,W], got "
                f"{tuple(disparity_hr_px.shape)}"
            )
        if not disparity_hr_px.is_floating_point() or disparity_hr_px.is_complex():
            raise TypeError("disparity_hr_px must be a real floating-point tensor")
        batch, candidates, height, width = disparity_hr_px.shape
        scalar_shape = (batch, candidates, height, width)
        device = disparity_hr_px.device
        for name, value in (
            ("confidence", confidence),
            ("temporal_age_frames", temporal_age_frames),
            ("z_aware_weights", z_aware_weights),
        ):
            self._check_scalar_candidates(
                name, value, scalar_shape, device=device
            )
        if not isinstance(fractional_phase_grid_px, Tensor):
            raise TypeError("fractional_phase_grid_px must be a torch.Tensor")
        expected_phase_shape = (batch, candidates, 2, height, width)
        if fractional_phase_grid_px.shape != expected_phase_shape:
            raise ValueError(
                "fractional_phase_grid_px must have shape "
                f"{expected_phase_shape}, got {tuple(fractional_phase_grid_px.shape)}"
            )
        if (
            not fractional_phase_grid_px.is_floating_point()
            or fractional_phase_grid_px.is_complex()
        ):
            raise TypeError(
                "fractional_phase_grid_px must be a real floating-point tensor"
            )
        if fractional_phase_grid_px.device != device:
            raise ValueError(f"fractional_phase_grid_px must be on device {device}")
        if not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have bool dtype")
        if valid_mask.shape != scalar_shape:
            raise ValueError(
                f"valid_mask must have shape {scalar_shape}, got {tuple(valid_mask.shape)}"
            )
        if valid_mask.device != device:
            raise ValueError(f"valid_mask must be on device {device}")

        target_dtype = disparity_hr_px.dtype
        safe_disparity = self._sanitize(
            disparity_hr_px,
            minimum=0.0,
            maximum=self.maximum_disparity_hr_px,
        ).to(dtype=target_dtype)
        safe_confidence = self._sanitize(
            confidence, minimum=0.0, maximum=1.0
        ).to(dtype=target_dtype)
        safe_phase = self._sanitize(
            fractional_phase_grid_px, minimum=-1.0, maximum=1.0
        ).to(dtype=target_dtype)
        safe_age = self._sanitize(
            temporal_age_frames,
            minimum=0.0,
            maximum=self.maximum_age_frames,
        ).to(dtype=target_dtype)
        safe_weight = self._sanitize(
            z_aware_weights, minimum=0.0, maximum=1.0
        ).to(dtype=target_dtype)
        valid_float = valid_mask.to(dtype=target_dtype)
        safe_weight = safe_weight * valid_float
        weight_sum = safe_weight.sum(dim=1, keepdim=True)
        aggregate_valid = torch.isfinite(weight_sum) & (weight_sum > 0)
        effective_weights = torch.where(
            aggregate_valid,
            safe_weight / weight_sum.clamp_min(torch.finfo(target_dtype).tiny),
            torch.zeros_like(safe_weight),
        )

        candidate_input = torch.cat(
            (
                safe_disparity.unsqueeze(2),
                safe_confidence.unsqueeze(2),
                safe_phase,
                safe_age.unsqueeze(2),
                safe_weight.unsqueeze(2),
                valid_float.unsqueeze(2),
            ),
            dim=2,
        ).to(dtype=target_dtype)
        if candidate_input.shape[2] != self.input_channels:
            raise RuntimeError("internal top-K candidate channel contract changed")
        candidate_input = candidate_input.reshape(
            batch * candidates, self.input_channels, height, width
        )
        encoded = self.candidate_encoder(candidate_input).reshape(
            batch, candidates, self.output_channels, height, width
        )
        # Mask after learned affine layers so invalid candidate output stays
        # exactly zero even when GroupNorm beta becomes non-zero during training.
        candidate_feature = torch.where(
            valid_mask.unsqueeze(2), encoded, torch.zeros_like(encoded)
        )
        aggregate_feature = (
            effective_weights.unsqueeze(2) * candidate_feature
        ).sum(dim=1)
        aggregate_feature = torch.where(
            aggregate_valid,
            aggregate_feature,
            torch.zeros_like(aggregate_feature),
        )
        return TopKHistoryEncoding(
            candidate_feature=candidate_feature,
            aggregate_feature=aggregate_feature,
            effective_weights=effective_weights,
            aggregate_valid_mask=aggregate_valid,
        )


__all__ = ["TopKHistoryEncoder", "TopKHistoryEncoding"]
