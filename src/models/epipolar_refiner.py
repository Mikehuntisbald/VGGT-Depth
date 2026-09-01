"""Local high-resolution epipolar correction for rectified stereo pairs.

The module in this file is deliberately independent from :class:`FFSOmegaTSR`.
It is the Stage-C refinement block and can be attached after a trained spatial
or temporal disparity predictor without changing that predictor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .rgb_encoder import ConvNormAct


@dataclass(frozen=True)
class EpipolarRefinementOutput:
    """Output of local HR stereo refinement.

    Attributes:
        corrected_disparity_hr_px: Refined left-view disparity ``[B,1,H,W]``
            in HR pixels.
        correction_hr_px: Bounded additive correction ``[B,1,H,W]`` in HR
            pixels. It is exactly zero where every search candidate is out of
            bounds.
        correlation: Groupwise candidate correlation ``[B,G,K,H,W]``. ``G``
            is the number of channel groups and candidates are ordered as
            :attr:`HREpipolarRefiner.candidate_offsets_hr_px`.
        candidate_valid_mask: Boolean mask ``[B,K,H,W]``. A candidate is valid
            iff its right-image sample center lies in ``[0,W-1]``.
        confidence: Peak probability ``[B,1,H,W]`` from a masked softmax over
            the group-averaged correlations. This is a local matching
            concentration score, not calibrated metric confidence.
        right_row_scale: Per-sample rectified vertical affine scale ``[B]``.
        right_row_offset_hr_px: Per-sample rectified vertical affine offset
            ``[B]`` in HR pixels, such that ``v_right = scale*v_left+offset``.
    """

    corrected_disparity_hr_px: Tensor
    correction_hr_px: Tensor
    correlation: Tensor
    candidate_valid_mask: Tensor
    confidence: Tensor
    right_row_scale: Tensor
    right_row_offset_hr_px: Tensor


def _batch_scalar(
    value: float | Tensor,
    *,
    name: str,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    positive: bool,
) -> Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim == 0:
        result = result.expand(batch)
    elif result.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape [B], got {result.shape}")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not bool((result > 0).all().item()):
        raise ValueError(f"{name} must be strictly positive")
    return result


def rectified_vertical_affine_from_intrinsics(
    intrinsics_left_hr: Tensor,
    intrinsics_right_hr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``v_right = scale*v_left + offset`` for rectified cameras.

    Inputs are cropped HR intrinsics ``[B,3,3]``. Rectification aligns camera
    rotations but does not require equal ``fy`` or ``cy``. From
    ``v=fy*Y/Z+cy``, the calibrated correspondence is

    ``v_right=(fy_right/fy_left)*(v_left-cy_left)+cy_right``.
    """

    if intrinsics_left_hr.ndim != 3 or intrinsics_left_hr.shape[-2:] != (3, 3):
        raise ValueError("intrinsics_left_hr must have shape [B,3,3]")
    if intrinsics_right_hr.shape != intrinsics_left_hr.shape:
        raise ValueError("intrinsics_right_hr must match intrinsics_left_hr")
    if not intrinsics_left_hr.is_floating_point() or not intrinsics_right_hr.is_floating_point():
        raise TypeError("left and right intrinsics must be floating point")
    if intrinsics_left_hr.device != intrinsics_right_hr.device:
        raise ValueError("left and right intrinsics must share a device")
    if not bool(torch.isfinite(intrinsics_left_hr).all().item()) or not bool(
        torch.isfinite(intrinsics_right_hr).all().item()
    ):
        raise ValueError("left and right intrinsics must be finite")
    fy_left = intrinsics_left_hr[:, 1, 1]
    fy_right = intrinsics_right_hr[:, 1, 1]
    if not bool((fy_left > 0).all().item()) or not bool((fy_right > 0).all().item()):
        raise ValueError("left and right fy must be strictly positive")
    row_scale = fy_right / fy_left
    row_offset_hr_px = (
        intrinsics_right_hr[:, 1, 2]
        - row_scale * intrinsics_left_hr[:, 1, 2]
    )
    return row_scale, row_offset_hr_px


def _validate_feature_inputs(
    feature_left_hr: Tensor,
    feature_right_hr: Tensor,
    predicted_disparity_hr_px: Tensor,
    *,
    num_groups: int,
) -> tuple[int, int, int, int]:
    if feature_left_hr.ndim != 4:
        raise ValueError(
            "feature_left_hr must have shape [B,C,H,W], got "
            f"{feature_left_hr.shape}"
        )
    if feature_right_hr.shape != feature_left_hr.shape:
        raise ValueError(
            "feature_right_hr must match feature_left_hr, got "
            f"{feature_right_hr.shape} and {feature_left_hr.shape}"
        )
    if not feature_left_hr.is_floating_point() or not feature_right_hr.is_floating_point():
        raise TypeError("left and right features must be floating-point tensors")
    if feature_left_hr.device != feature_right_hr.device:
        raise ValueError("left and right features must be on the same device")

    batch, channels, height, width = feature_left_hr.shape
    expected_disparity_shape = (batch, 1, height, width)
    if predicted_disparity_hr_px.shape != expected_disparity_shape:
        raise ValueError(
            "predicted_disparity_hr_px must have shape "
            f"{expected_disparity_shape}, got {predicted_disparity_hr_px.shape}"
        )
    if not predicted_disparity_hr_px.is_floating_point():
        raise TypeError("predicted_disparity_hr_px must be floating point")
    if predicted_disparity_hr_px.device != feature_left_hr.device:
        raise ValueError("features and predicted disparity must be on the same device")
    if num_groups <= 0 or channels % num_groups != 0:
        raise ValueError(
            f"feature channels ({channels}) must be divisible by num_groups "
            f"({num_groups})"
        )
    return batch, channels, height, width


def groupwise_epipolar_correlation(
    feature_left_hr: Tensor,
    feature_right_hr: Tensor,
    predicted_disparity_hr_px: Tensor,
    *,
    candidate_offsets_hr_px: Sequence[float] | Tensor = (-2.0, -1.0, 0.0, 1.0, 2.0),
    num_groups: int = 8,
    right_row_scale: float | Tensor = 1.0,
    right_row_offset_hr_px: float | Tensor = 0.0,
) -> tuple[Tensor, Tensor]:
    """Build a differentiable horizontal cost curve around a disparity.

    All feature maps live on the HR pixel grid. For left pixel ``(u,v)`` and
    candidate correction ``delta`` the right feature is sampled at

    ``u_right = u - predicted_disparity_hr_px - delta`` and
    ``v_right = right_row_scale*v + right_row_offset_hr_px``. The vertical
    affine comes from the cropped left/right rectified intrinsics; equal
    intrinsics reduce to same-row sampling.

    Bilinear :func:`torch.nn.functional.grid_sample` preserves fractional
    disparity phase. Channels are divided into ``num_groups`` contiguous,
    equal-sized groups; the product is averaged over channels inside each
    group. The returned correlation therefore has shape ``[B,G,K,H,W]`` and
    does *not* reduce the group dimension.

    Invalid candidates are set to zero in the returned correlation and are
    separately identified by the boolean ``[B,K,H,W]`` mask. Low-precision
    feature tensors use float32 coordinates and sampling so BF16 does not
    quantize a large disparity to integer-pixel precision.
    """

    batch, channels, height, width = _validate_feature_inputs(
        feature_left_hr,
        feature_right_hr,
        predicted_disparity_hr_px,
        num_groups=num_groups,
    )
    sampling_dtype = (
        torch.float32
        if feature_left_hr.dtype in (torch.float16, torch.bfloat16)
        else feature_left_hr.dtype
    )
    offsets = torch.as_tensor(
        candidate_offsets_hr_px,
        dtype=sampling_dtype,
        device=feature_left_hr.device,
    )
    if offsets.ndim != 1 or offsets.numel() == 0:
        raise ValueError("candidate_offsets_hr_px must be a non-empty 1D sequence")
    # The model validates its constant candidates at construction. Preserve a
    # useful error for direct CPU helper calls without adding a device sync to
    # every CUDA forward.
    if offsets.device.type == "cpu" and not bool(torch.isfinite(offsets).all()):
        raise ValueError("candidate_offsets_hr_px must contain only finite values")
    candidate_count = int(offsets.numel())
    row_scale = _batch_scalar(
        right_row_scale,
        name="right_row_scale",
        batch=batch,
        dtype=sampling_dtype,
        device=feature_left_hr.device,
        positive=True,
    )
    row_offset = _batch_scalar(
        right_row_offset_hr_px,
        name="right_row_offset_hr_px",
        batch=batch,
        dtype=sampling_dtype,
        device=feature_left_hr.device,
        positive=False,
    )

    disparity = predicted_disparity_hr_px.to(dtype=sampling_dtype)[:, 0]
    column = torch.arange(width, dtype=sampling_dtype, device=feature_left_hr.device)
    row = torch.arange(height, dtype=sampling_dtype, device=feature_left_hr.device)
    source_u = (
        column.view(1, 1, 1, width)
        - disparity.unsqueeze(1)
        - offsets.view(1, candidate_count, 1, 1)
    )
    source_v = (
        row.view(1, 1, height, 1) * row_scale.view(batch, 1, 1, 1)
        + row_offset.view(batch, 1, 1, 1)
    ).expand(batch, candidate_count, height, width)
    finite_coordinate = torch.isfinite(source_u) & torch.isfinite(source_v)
    candidate_valid_mask = (
        finite_coordinate
        & (source_u >= 0.0)
        & (source_u <= float(width - 1))
        & (source_v >= 0.0)
        & (source_v <= float(height - 1))
    )

    # NaN/Inf coordinates are made harmless before grid_sample. Their outputs
    # are ignored by candidate_valid_mask below.
    safe_source_u = torch.where(finite_coordinate, source_u, torch.zeros_like(source_u))
    safe_source_v = torch.where(finite_coordinate, source_v, torch.zeros_like(source_v))
    grid_x = 2.0 * (safe_source_u + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (safe_source_v + 0.5) / float(height) - 1.0
    candidate_grid = torch.stack((grid_x, grid_y), dim=-1)
    # Pack K horizontal candidate rows into one grid_sample call. Reshaping the
    # result restores explicit candidate order without repeating right features.
    packed_grid = candidate_grid.permute(0, 2, 1, 3, 4).reshape(
        batch, height, candidate_count * width, 2
    )
    sampled_right = functional.grid_sample(
        feature_right_hr.to(dtype=sampling_dtype),
        packed_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled_right = sampled_right.reshape(
        batch, channels, height, candidate_count, width
    ).permute(0, 1, 3, 2, 4)

    channels_per_group = channels // num_groups
    grouped_left = feature_left_hr.to(dtype=sampling_dtype).reshape(
        batch, num_groups, channels_per_group, height, width
    )
    grouped_right = sampled_right.reshape(
        batch, num_groups, channels_per_group, candidate_count, height, width
    )
    correlation = (grouped_left.unsqueeze(3) * grouped_right).mean(dim=2)
    correlation = correlation.masked_fill(~candidate_valid_mask.unsqueeze(1), 0.0)
    return correlation, candidate_valid_mask


def _masked_match_confidence(
    correlation: Tensor,
    candidate_valid_mask: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Return a finite peak-probability confidence for each HR pixel."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    group_mean = correlation.mean(dim=1)
    any_valid = candidate_valid_mask.any(dim=1, keepdim=True)
    masked_logits = group_mean.masked_fill(~candidate_valid_mask, -torch.inf)
    # Softmax(all -inf) is NaN. A deterministic zero-logit fallback is used
    # only for all-invalid pixels, then removed from the output probability.
    safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits / temperature, dim=1)
    probabilities = probabilities * candidate_valid_mask.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    probabilities = torch.where(
        any_valid,
        probabilities / normalizer,
        torch.zeros_like(probabilities),
    )
    return probabilities.amax(dim=1, keepdim=True)


class HREpipolarRefiner(nn.Module):
    """Lightweight local stereo correction on the HR grid.

    ``rgb_left_hr`` and ``rgb_right_hr`` are a rectified stereo pair shaped
    ``[B,3,H,W]``. ``predicted_disparity_hr_px`` is left-view disparity shaped
    ``[B,1,H,W]`` and measured in HR pixels. A single feature encoder is shared
    between the two images. The default five-candidate search estimates an
    additive correction in ``[-2,2]`` HR pixels.
    """

    def __init__(
        self,
        *,
        feature_channels: int = 32,
        correlation_groups: int = 8,
        candidate_offsets_hr_px: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
        correction_limit_hr_px: float = 2.0,
        confidence_temperature: float = 1.0,
        head_channels: int = 48,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or feature_channels % correlation_groups != 0:
            raise ValueError(
                "feature_channels must be positive and divisible by correlation_groups"
            )
        if len(candidate_offsets_hr_px) == 0:
            raise ValueError("candidate_offsets_hr_px cannot be empty")
        if any(not math.isfinite(value) for value in candidate_offsets_hr_px):
            raise ValueError("candidate_offsets_hr_px must contain finite values")
        if correction_limit_hr_px <= 0:
            raise ValueError("correction_limit_hr_px must be positive")
        if confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive")
        if head_channels <= 0:
            raise ValueError("head_channels must be positive")

        self.feature_channels = int(feature_channels)
        self.correlation_groups = int(correlation_groups)
        self.correction_limit_hr_px = float(correction_limit_hr_px)
        self.confidence_temperature = float(confidence_temperature)
        self.register_buffer(
            "candidate_offsets_hr_px",
            torch.tensor(candidate_offsets_hr_px, dtype=torch.float32),
            persistent=True,
        )

        stem_channels = max(16, feature_channels // 2)
        self.feature_encoder = nn.Sequential(
            ConvNormAct(3, stem_channels),
            ConvNormAct(stem_channels, feature_channels),
            ConvNormAct(feature_channels, feature_channels),
        )
        candidate_count = len(candidate_offsets_hr_px)
        # Groups-major correlation, K validity indicators, normalized current
        # disparity, local confidence, and the current left feature map.
        correction_input_channels = (
            feature_channels
            + correlation_groups * candidate_count
            + candidate_count
            + 2
        )
        self.correction_head = nn.Sequential(
            ConvNormAct(correction_input_channels, head_channels),
            ConvNormAct(head_channels, head_channels),
            nn.Conv2d(head_channels, 1, kernel_size=3, padding=1),
        )

        # Attaching Stage C to a trained Stage-B model starts as an exact no-op.
        final_layer = self.correction_head[-1]
        assert isinstance(final_layer, nn.Conv2d)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    @property
    def trainable_parameter_count(self) -> int:
        """Number of trainable scalar parameters in this refinement block."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        rgb_left_hr: Tensor,
        rgb_right_hr: Tensor,
        predicted_disparity_hr_px: Tensor,
        intrinsics_left_hr: Tensor | None = None,
        intrinsics_right_hr: Tensor | None = None,
        right_row_scale: float | Tensor | None = None,
        right_row_offset_hr_px: float | Tensor | None = None,
    ) -> EpipolarRefinementOutput:
        """Refine left-view HR disparity using real stereo correspondence.

        Args:
            rgb_left_hr: Rectified left RGB ``[B,3,H,W]``.
            rgb_right_hr: Rectified right RGB ``[B,3,H,W]``.
            predicted_disparity_hr_px: Current left disparity ``[B,1,H,W]`` in
                HR pixels, where corresponding right ``x`` is ``x_left-d``.
            intrinsics_left_hr: Optional cropped left intrinsics ``[B,3,3]``.
            intrinsics_right_hr: Optional cropped right intrinsics ``[B,3,3]``.
                Supply both or neither. Both are required by the formal
                Stage-C wrapper so a nonzero rectified vertical offset is not
                silently discarded.
            right_row_scale: Optional explicit rectified-pixel row scale,
                scalar or ``[B]``. Supply together with
                ``right_row_offset_hr_px`` and not with intrinsics.
            right_row_offset_hr_px: Optional explicit rectified-pixel row
                offset in HR pixels, scalar or ``[B]``.
        """

        if rgb_left_hr.ndim != 4 or rgb_left_hr.shape[1] != 3:
            raise ValueError(
                f"rgb_left_hr must have shape [B,3,H,W], got {rgb_left_hr.shape}"
            )
        if rgb_right_hr.shape != rgb_left_hr.shape:
            raise ValueError(
                "rgb_right_hr must match rgb_left_hr, got "
                f"{rgb_right_hr.shape} and {rgb_left_hr.shape}"
            )
        if not rgb_left_hr.is_floating_point() or not rgb_right_hr.is_floating_point():
            raise TypeError("left and right RGB inputs must be floating point")
        if rgb_left_hr.device != rgb_right_hr.device:
            raise ValueError("left and right RGB inputs must be on the same device")
        batch, _, height, width = rgb_left_hr.shape
        expected_disparity_shape = (batch, 1, height, width)
        if predicted_disparity_hr_px.shape != expected_disparity_shape:
            raise ValueError(
                "predicted_disparity_hr_px must have shape "
                f"{expected_disparity_shape}, got {predicted_disparity_hr_px.shape}"
            )
        if not predicted_disparity_hr_px.is_floating_point():
            raise TypeError("predicted_disparity_hr_px must be floating point")
        if predicted_disparity_hr_px.device != rgb_left_hr.device:
            raise ValueError("RGB inputs and predicted disparity must share a device")

        if (intrinsics_left_hr is None) != (intrinsics_right_hr is None):
            raise ValueError("left and right intrinsics must be supplied together")
        if (right_row_scale is None) != (right_row_offset_hr_px is None):
            raise ValueError("explicit row scale and offset must be supplied together")
        if intrinsics_left_hr is not None and right_row_scale is not None:
            raise ValueError("use intrinsics or an explicit row affine, not both")
        if right_row_scale is not None:
            assert right_row_offset_hr_px is not None
            resolved_row_scale = _batch_scalar(
                right_row_scale,
                name="right_row_scale",
                batch=batch,
                dtype=torch.float32,
                device=rgb_left_hr.device,
                positive=True,
            )
            resolved_row_offset_hr_px = _batch_scalar(
                right_row_offset_hr_px,
                name="right_row_offset_hr_px",
                batch=batch,
                dtype=torch.float32,
                device=rgb_left_hr.device,
                positive=False,
            )
        elif intrinsics_left_hr is None:
            resolved_row_scale = torch.ones(
                batch, dtype=torch.float32, device=rgb_left_hr.device
            )
            resolved_row_offset_hr_px = torch.zeros_like(resolved_row_scale)
        else:
            assert intrinsics_right_hr is not None
            if intrinsics_left_hr.device != rgb_left_hr.device or (
                intrinsics_right_hr.device != rgb_left_hr.device
            ):
                raise ValueError("RGB and intrinsics must share a device")
            resolved_row_scale, resolved_row_offset_hr_px = (
                rectified_vertical_affine_from_intrinsics(
                    intrinsics_left_hr.float(), intrinsics_right_hr.float()
                )
            )

        feature_left_hr = self.feature_encoder(rgb_left_hr)
        feature_right_hr = self.feature_encoder(rgb_right_hr)
        correlation, candidate_valid_mask = groupwise_epipolar_correlation(
            feature_left_hr,
            feature_right_hr,
            predicted_disparity_hr_px,
            candidate_offsets_hr_px=self.candidate_offsets_hr_px,
            num_groups=self.correlation_groups,
            right_row_scale=resolved_row_scale,
            right_row_offset_hr_px=resolved_row_offset_hr_px,
        )
        confidence = _masked_match_confidence(
            correlation,
            candidate_valid_mask,
            temperature=self.confidence_temperature,
        )

        candidate_count = int(self.candidate_offsets_hr_px.numel())
        correlation_channels = correlation.reshape(
            batch, self.correlation_groups * candidate_count, height, width
        )
        valid_channels = candidate_valid_mask.to(correlation.dtype)
        normalized_disparity = torch.nan_to_num(
            predicted_disparity_hr_px.to(correlation.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ) / float(max(width - 1, 1))
        correction_features = torch.cat(
            (
                feature_left_hr.to(correlation.dtype),
                correlation_channels,
                valid_channels,
                normalized_disparity,
                confidence,
            ),
            dim=1,
        )
        raw_correction = self.correction_head(correction_features)
        any_valid = candidate_valid_mask.any(dim=1, keepdim=True)
        correction_hr_px = (
            self.correction_limit_hr_px
            * torch.tanh(raw_correction)
            * any_valid.to(raw_correction.dtype)
        )
        corrected_disparity_hr_px = predicted_disparity_hr_px + correction_hr_px.to(
            predicted_disparity_hr_px.dtype
        )
        return EpipolarRefinementOutput(
            corrected_disparity_hr_px=corrected_disparity_hr_px,
            correction_hr_px=correction_hr_px,
            correlation=correlation,
            candidate_valid_mask=candidate_valid_mask,
            confidence=confidence,
            right_row_scale=resolved_row_scale,
            right_row_offset_hr_px=resolved_row_offset_hr_px,
        )
