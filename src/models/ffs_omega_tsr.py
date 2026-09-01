"""Trainable causal x2 super-resolution head for cached FFS/VGGT geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .convex_upsampler import ConvexUpsampler
from .rgb_encoder import ConvNormAct, RGBPyramidEncoder
from .source_gating import SourceGatingHead
from .temporal_gru import StackedConvGRU
from .topk_history_encoder import TopKHistoryEncoder


@dataclass(frozen=True)
class ModelOutput:
    """Output of one causal TSR step.

    Attributes:
        disparity_hr_px: Anchored final disparity ``[B,1,2H,2W]`` in HR pixels.
        disparity_raw_hr_px: Pre-anchor prediction with the same shape/unit.
        source_weights: LR-grid weights ``[B,3,H,W]`` ordered FFS/VGGT/history.
        log_variance: ``log(sigma^2)`` at the HR grid, shaped ``[B,1,2H,2W]``.
        uncertainty: Variance ``sigma^2 = exp(log_variance)`` at the HR grid.
        hidden_state: Two causal ConvGRU states, each ``[B,96,H,W]``.
        anchor_gate: Applied FFS correction gate ``[B,1,2H,2W]``.
        source_valid_mask: Effective pre-fallback source mask ``[B,3,H,W]``.
        disparity_source_mix_hr_px_lr_grid: Three-source mixture before the LR
            residual, shaped ``[B,1,H,W]`` and expressed in HR pixels.
        disparity_post_lr_residual_hr_px_lr_grid: Source mixture after the
            bounded LR residual, shaped ``[B,1,H,W]`` in HR pixels.
        disparity_post_convex_hr_px: Convex-upsampled disparity before the
            shallow HR residual, shaped ``[B,1,2H,2W]`` in HR pixels.
        disparity_pre_lower_bound_hr_px_lr_grid: LR disparity before an
            opt-in physical lower bound, if that ablation is enabled.
        disparity_pre_lower_bound_raw_hr_px: Raw HR disparity before an
            opt-in physical lower bound, if that ablation is enabled.
        valid_probability: Opt-in calibrated physical-valid probability
            ``[B,1,2H,2W]``.  It is exactly zero where every geometric source
            is invalid. ``None`` for legacy models.
        completion_probability: Opt-in probability that a current FFS hole was
            completed by another supported source.  It is exactly zero outside
            FFS holes and on all-source-invalid pixels. ``None`` for legacy.
        valid_logits, completion_logits: Unmasked logits used to supervise the
            two explicit heads. ``None`` for legacy models.
        output_valid_mask: Opt-in physical output-valid decision.  A zero
            disparity is always invalid; invalid disparity is represented by
            exact zero rather than epsilon. ``None`` for legacy models.
        completion_mask: Opt-in hard hole-completion decision. ``None`` for
            legacy models.
        history_topk_effective_weights: Sanitized top-K candidate weights
            ``[B,K,H,W]`` for the opt-in history V2 path.
        history_topk_valid_mask: Effective candidate mask with the same shape.

    The final five fields are diagnostic tensor taps only.  They do not add
    modules, parameters, buffers, or checkpoint state.  Their optional defaults
    keep manually constructed legacy ``ModelOutput`` fixtures source-compatible;
    every real :class:`FFSOmegaTSR` forward populates them.
    """

    disparity_hr_px: Tensor
    disparity_raw_hr_px: Tensor
    source_weights: Tensor
    log_variance: Tensor
    uncertainty: Tensor
    hidden_state: tuple[Tensor, ...]
    anchor_gate: Tensor
    source_valid_mask: Tensor
    disparity_source_mix_hr_px_lr_grid: Tensor | None = None
    disparity_post_lr_residual_hr_px_lr_grid: Tensor | None = None
    disparity_post_convex_hr_px: Tensor | None = None
    disparity_pre_lower_bound_hr_px_lr_grid: Tensor | None = None
    disparity_pre_lower_bound_raw_hr_px: Tensor | None = None
    valid_probability: Tensor | None = None
    completion_probability: Tensor | None = None
    valid_logits: Tensor | None = None
    completion_logits: Tensor | None = None
    output_valid_mask: Tensor | None = None
    completion_mask: Tensor | None = None
    history_topk_effective_weights: Tensor | None = None
    history_topk_valid_mask: Tensor | None = None


def count_trainable_parameters(module: nn.Module) -> int:
    """Count scalar parameters with ``requires_grad=True``."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class GeometryEncoder(nn.Module):
    """Encode the ten explicit LR-grid geometry channels into 64 features."""

    input_channels = 10

    def __init__(self, output_channels: int = 64) -> None:
        super().__init__()
        if output_channels != 64:
            raise ValueError(
                f"the fixed MVP geometry feature width is 64, got {output_channels}"
            )
        self.encoder = nn.Sequential(
            ConvNormAct(self.input_channels, output_channels),
            ConvNormAct(output_channels, output_channels),
            ConvNormAct(output_channels, output_channels),
        )

    def forward(self, geometry_lr: Tensor) -> Tensor:
        if geometry_lr.ndim != 4 or geometry_lr.shape[1] != self.input_channels:
            raise ValueError(
                f"geometry_lr must have shape [B,{self.input_channels},H,W], "
                f"got {geometry_lr.shape}"
            )
        return self.encoder(geometry_lr)


class FFSOmegaTSR(nn.Module):
    """Lightweight causal disparity super-resolution model for the x2 MVP.

    Every disparity input is sampled on the LR grid but expressed in **HR
    pixels**. The module never performs another disparity scale conversion.
    Optional VGGT/history arguments permit the exact same module to run Stage A
    (T=1) and Stage B (T=3).
    """

    def __init__(
        self,
        *,
        rgb_channels: tuple[int, int, int] = (32, 64, 96),
        geometry_channels: int = 64,
        hidden_channels: int = 96,
        gru_layers: int = 2,
        scale: int = 2,
        residual_limit_hr_px: float = 8.0,
        log_variance_bounds: tuple[float, float] = (-10.0, 10.0),
        sanitize_invalid_source_disparities: bool = False,
        positivity_floor_hr_px: float | None = None,
        physical_output_v2: bool = False,
        physical_valid_threshold: float = 0.5,
        completion_threshold: float = 0.5,
        trusted_ffs_confidence_threshold: float = 0.8,
        temporal_history_top_k: int | None = None,
        temporal_history_feature_channels: int = 32,
    ) -> None:
        super().__init__()
        if rgb_channels != (32, 64, 96):
            raise ValueError(f"the fixed MVP RGB channels are (32,64,96), got {rgb_channels}")
        if hidden_channels != 96:
            raise ValueError(f"the fixed MVP hidden width is 96, got {hidden_channels}")
        if gru_layers != 2:
            raise ValueError(f"the fixed MVP uses two ConvGRU layers, got {gru_layers}")
        if residual_limit_hr_px <= 0:
            raise ValueError("residual_limit_hr_px must be positive")
        minimum_log_variance, maximum_log_variance = log_variance_bounds
        if minimum_log_variance >= maximum_log_variance:
            raise ValueError("log_variance_bounds must be strictly increasing")
        if not isinstance(sanitize_invalid_source_disparities, bool):
            raise TypeError("sanitize_invalid_source_disparities must be a bool")
        if positivity_floor_hr_px is not None and (
            not isinstance(positivity_floor_hr_px, (int, float))
            or isinstance(positivity_floor_hr_px, bool)
            or not torch.isfinite(torch.tensor(positivity_floor_hr_px))
            or positivity_floor_hr_px < 0
        ):
            raise ValueError("positivity_floor_hr_px must be finite and non-negative")
        if not isinstance(physical_output_v2, bool):
            raise TypeError("physical_output_v2 must be a bool")
        for name, value in (
            ("physical_valid_threshold", physical_valid_threshold),
            ("completion_threshold", completion_threshold),
            ("trusted_ffs_confidence_threshold", trusted_ffs_confidence_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0,1)")
        if temporal_history_top_k is not None and (
            isinstance(temporal_history_top_k, bool)
            or not isinstance(temporal_history_top_k, int)
            or temporal_history_top_k < 2
        ):
            raise ValueError("temporal_history_top_k must be None or an integer >= 2")
        if (
            isinstance(temporal_history_feature_channels, bool)
            or not isinstance(temporal_history_feature_channels, int)
            or temporal_history_feature_channels <= 0
        ):
            raise ValueError("temporal_history_feature_channels must be positive")

        self.scale = scale
        self.residual_limit_hr_px = float(residual_limit_hr_px)
        self.log_variance_bounds = (
            float(minimum_log_variance),
            float(maximum_log_variance),
        )
        # These are behavior-only opt-ins: no parameters or buffers are added,
        # so a baseline checkpoint remains strictly state-dict compatible.
        self.sanitize_invalid_source_disparities = sanitize_invalid_source_disparities
        self.positivity_floor_hr_px = (
            None if positivity_floor_hr_px is None else float(positivity_floor_hr_px)
        )
        self.physical_output_v2 = physical_output_v2
        self.physical_valid_threshold = float(physical_valid_threshold)
        self.completion_threshold = float(completion_threshold)
        self.trusted_ffs_confidence_threshold = float(
            trusted_ffs_confidence_threshold
        )
        self.temporal_history_top_k = temporal_history_top_k
        self.rgb_encoder = RGBPyramidEncoder(rgb_channels)
        self.geometry_encoder = GeometryEncoder(geometry_channels)
        self.topk_history_encoder: TopKHistoryEncoder | None = None
        if temporal_history_top_k is not None:
            self.topk_history_encoder = TopKHistoryEncoder(
                output_channels=temporal_history_feature_channels
            )
        recurrent_input_channels = rgb_channels[-1] + geometry_channels
        if self.topk_history_encoder is not None:
            recurrent_input_channels += temporal_history_feature_channels
        self.temporal_fusion = StackedConvGRU(
            input_channels=recurrent_input_channels,
            hidden_channels=hidden_channels,
            num_layers=gru_layers,
        )
        self.source_gating = SourceGatingHead(hidden_channels)
        self.disparity_residual_head = nn.Conv2d(
            hidden_channels, 1, kernel_size=3, padding=1
        )
        self.convex_upsampler = ConvexUpsampler(scale)
        self.convex_mask_head = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(
                hidden_channels,
                self.convex_upsampler.mask_channels,
                kernel_size=1,
            ),
        )
        hr_decoder_channels = 32
        self.hidden_to_hr = nn.Conv2d(
            hidden_channels, hr_decoder_channels, kernel_size=3, padding=1
        )
        self.hr_output_head = nn.Sequential(
            ConvNormAct(hr_decoder_channels + rgb_channels[0], 64),
            ConvNormAct(64, hr_decoder_channels),
            nn.Conv2d(hr_decoder_channels, 2, kernel_size=3, padding=1),
        )
        # Opt-in only.  The legacy/default module graph and state_dict remain
        # byte-for-key compatible with all existing checkpoints.
        self.validity_completion_head: nn.Module | None = None
        if self.physical_output_v2:
            self.validity_completion_head = nn.Sequential(
                ConvNormAct(hr_decoder_channels + rgb_channels[0], 32),
                nn.Conv2d(32, 2, kernel_size=3, padding=1),
            )

        # The initial network is a conservative geometry interpolator: no
        # residual change, uniform convex neighborhoods, and unit variance.
        nn.init.zeros_(self.disparity_residual_head.weight)
        nn.init.zeros_(self.disparity_residual_head.bias)
        final_mask_layer = self.convex_mask_head[-1]
        assert isinstance(final_mask_layer, nn.Conv2d)
        nn.init.zeros_(final_mask_layer.weight)
        nn.init.zeros_(final_mask_layer.bias)
        final_hr_layer = self.hr_output_head[-1]
        assert isinstance(final_hr_layer, nn.Conv2d)
        nn.init.zeros_(final_hr_layer.weight)
        nn.init.zeros_(final_hr_layer.bias)
        if self.validity_completion_head is not None:
            final_validity_layer = self.validity_completion_head[-1]
            assert isinstance(final_validity_layer, nn.Conv2d)
            nn.init.zeros_(final_validity_layer.weight)
            nn.init.zeros_(final_validity_layer.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self)

    @staticmethod
    def _check_scalar_lr(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tensor.shape}")

    @staticmethod
    def _normalize_mask(
        name: str,
        mask: Tensor | None,
        *,
        expected_shape: tuple[int, int, int, int],
        default: Tensor,
    ) -> Tensor:
        if mask is None:
            return default
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
        if mask.device != default.device:
            raise ValueError(f"{name} must be on device {default.device}, got {mask.device}")
        return mask.to(dtype=torch.bool)

    def forward(
        self,
        rgb_hr: Tensor,
        disparity_ffs_hr_px: Tensor,
        confidence_ffs: Tensor,
        *,
        disparity_vggt_hr_px: Tensor | None = None,
        confidence_vggt: Tensor | None = None,
        disparity_history_hr_px: Tensor | None = None,
        confidence_history: Tensor | None = None,
        history_visibility: Tensor | None = None,
        photometric_residual: Tensor | None = None,
        fractional_offset_px: Tensor | None = None,
        history_topk_disparity_hr_px: Tensor | None = None,
        history_topk_confidence: Tensor | None = None,
        history_topk_fractional_offset_px: Tensor | None = None,
        history_topk_age_frames: Tensor | None = None,
        history_topk_weights: Tensor | None = None,
        history_topk_valid_mask: Tensor | None = None,
        valid_ffs: Tensor | None = None,
        valid_vggt: Tensor | None = None,
        valid_history: Tensor | None = None,
        hidden_state: Sequence[Tensor] | None = None,
    ) -> ModelOutput:
        """Run one causal time step.

        Args:
            rgb_hr: Current left RGB ``[B,3,2H,2W]``.
            disparity_ffs_hr_px: Current FFS disparity ``[B,1,H,W]`` in HR px.
            confidence_ffs: Current FFS confidence ``[B,1,H,W]``.
            disparity_vggt_hr_px: Aligned VGGT prior ``[B,1,H,W]`` in HR px.
            confidence_vggt: VGGT confidence ``[B,1,H,W]``.
            disparity_history_hr_px: Warped history ``[B,1,H,W]`` in HR px.
            confidence_history: Warped history confidence ``[B,1,H,W]``.
            history_visibility: Z-buffer visibility ``[B,1,H,W]``.
            photometric_residual: Current/history residual ``[B,1,H,W]``.
            fractional_offset_px: Reprojection phase ``[B,2,H,W]`` in pixels.
            history_topk_*: Opt-in candidate tensors. Scalar fields are
                ``[B,K,H,W]``, phase is ``[B,K,2,H,W]``, and validity is bool.
                They are already z-aware splatted into the current LR grid.
            valid_ffs, valid_vggt, valid_history: Optional source masks.
            hidden_state: Previous two-layer causal state, or ``None`` at reset.
        """

        if rgb_hr.ndim != 4 or rgb_hr.shape[1] != 3:
            raise ValueError(f"rgb_hr must have shape [B,3,2H,2W], got {rgb_hr.shape}")
        if not rgb_hr.is_floating_point():
            raise TypeError("rgb_hr must be floating point and normalized by the data pipeline")
        if disparity_ffs_hr_px.ndim != 4 or disparity_ffs_hr_px.shape[1] != 1:
            raise ValueError(
                "disparity_ffs_hr_px must have shape [B,1,H,W], got "
                f"{disparity_ffs_hr_px.shape}"
            )
        if not disparity_ffs_hr_px.is_floating_point():
            raise TypeError("disparity_ffs_hr_px must be floating point")
        batch, _, height_lr, width_lr = disparity_ffs_hr_px.shape
        scalar_shape = (batch, 1, height_lr, width_lr)
        self._check_scalar_lr("confidence_ffs", confidence_ffs, scalar_shape)
        expected_rgb_shape = (batch, 3, height_lr * self.scale, width_lr * self.scale)
        if rgb_hr.shape != expected_rgb_shape:
            raise ValueError(f"rgb_hr must have shape {expected_rgb_shape}, got {rgb_hr.shape}")
        if rgb_hr.device != disparity_ffs_hr_px.device or rgb_hr.device != confidence_ffs.device:
            raise ValueError("rgb_hr, disparity_ffs_hr_px, and confidence_ffs need one device")

        def optional_scalar_pair(
            disparity_hr_px_name: str,
            disparity_hr_px_tensor: Tensor | None,
            confidence_name: str,
            confidence: Tensor | None,
        ) -> tuple[Tensor, Tensor, bool]:
            if disparity_hr_px_tensor is None and confidence is None:
                zeros = disparity_ffs_hr_px.new_zeros(scalar_shape)
                return zeros, zeros.clone(), False
            if disparity_hr_px_tensor is None or confidence is None:
                raise ValueError(
                    f"{disparity_hr_px_name} and {confidence_name} must be supplied together"
                )
            self._check_scalar_lr(
                disparity_hr_px_name, disparity_hr_px_tensor, scalar_shape
            )
            self._check_scalar_lr(confidence_name, confidence, scalar_shape)
            if (
                disparity_hr_px_tensor.device != rgb_hr.device
                or confidence.device != rgb_hr.device
            ):
                raise ValueError(
                    f"{disparity_hr_px_name} and {confidence_name} need rgb_hr device"
                )
            return disparity_hr_px_tensor, confidence, True

        disparity_vggt_hr_px, confidence_vggt, has_vggt = optional_scalar_pair(
            "disparity_vggt_hr_px",
            disparity_vggt_hr_px,
            "confidence_vggt",
            confidence_vggt,
        )
        disparity_history_hr_px, confidence_history, has_history = optional_scalar_pair(
            "disparity_history_hr_px",
            disparity_history_hr_px,
            "confidence_history",
            confidence_history,
        )

        if history_visibility is None:
            history_visibility = disparity_ffs_hr_px.new_zeros(scalar_shape)
        self._check_scalar_lr("history_visibility", history_visibility, scalar_shape)
        if photometric_residual is None:
            photometric_residual = disparity_ffs_hr_px.new_zeros(scalar_shape)
        self._check_scalar_lr("photometric_residual", photometric_residual, scalar_shape)
        if fractional_offset_px is None:
            fractional_offset_px = disparity_ffs_hr_px.new_zeros(
                (batch, 2, height_lr, width_lr)
            )
        if fractional_offset_px.shape != (batch, 2, height_lr, width_lr):
            raise ValueError(
                "fractional_offset_px must have shape "
                f"{(batch, 2, height_lr, width_lr)}, got {fractional_offset_px.shape}"
            )
        for name, tensor in (
            ("history_visibility", history_visibility),
            ("photometric_residual", photometric_residual),
            ("fractional_offset_px", fractional_offset_px),
        ):
            if tensor.device != rgb_hr.device:
                raise ValueError(f"{name} must be on device {rgb_hr.device}")

        raw_disparities_hr_px = (
            disparity_ffs_hr_px,
            disparity_vggt_hr_px,
            disparity_history_hr_px,
        )
        raw_confidences = (confidence_ffs, confidence_vggt, confidence_history)
        finite_source_masks = tuple(
            torch.isfinite(disparity_hr_px_tensor)
            & (disparity_hr_px_tensor > 0)
            & torch.isfinite(confidence)
            for disparity_hr_px_tensor, confidence in zip(
                raw_disparities_hr_px, raw_confidences, strict=True
            )
        )
        source_presence_masks = (
            finite_source_masks[0],
            finite_source_masks[1] & has_vggt,
            finite_source_masks[2]
            & has_history
            & torch.isfinite(history_visibility)
            & (history_visibility > 0),
        )
        source_valid_masks = (
            self._normalize_mask(
                "valid_ffs", valid_ffs, expected_shape=scalar_shape, default=source_presence_masks[0]
            )
            & source_presence_masks[0],
            self._normalize_mask(
                "valid_vggt",
                valid_vggt,
                expected_shape=scalar_shape,
                default=source_presence_masks[1],
            )
            & source_presence_masks[1],
            self._normalize_mask(
                "valid_history",
                valid_history,
                expected_shape=scalar_shape,
                default=source_presence_masks[2],
            )
            & source_presence_masks[2],
        )
        source_valid_mask = torch.cat(source_valid_masks, dim=1)

        target_dtype = rgb_hr.dtype

        def sanitize(tensor: Tensor) -> Tensor:
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).to(
                dtype=target_dtype
            )

        safe_disparities_hr_px = tuple(
            sanitize(tensor) for tensor in raw_disparities_hr_px
        )
        if self.sanitize_invalid_source_disparities or self.physical_output_v2:
            # A source which failed the validity contract must not contribute a
            # finite negative value through the deterministic all-invalid FFS
            # fallback or through the FFS anchor.  Invalid pixels become zero,
            # not epsilon-filled positive pseudo-measurements.
            safe_disparities_hr_px = tuple(
                torch.where(valid, disparity, torch.zeros_like(disparity))
                for valid, disparity in zip(
                    source_valid_masks, safe_disparities_hr_px, strict=True
                )
            )
        safe_confidences = tuple(
            sanitize(tensor).clamp(0.0, 1.0) for tensor in raw_confidences
        )
        safe_visibility = sanitize(history_visibility).clamp(0.0, 1.0)
        geometry_lr = torch.cat(
            (
                safe_disparities_hr_px[0],
                safe_confidences[0],
                safe_disparities_hr_px[1],
                safe_confidences[1],
                safe_disparities_hr_px[2],
                safe_confidences[2],
                safe_visibility,
                sanitize(photometric_residual),
                sanitize(fractional_offset_px),
            ),
            dim=1,
        )
        assert geometry_lr.shape[1] == GeometryEncoder.input_channels

        rgb_features = self.rgb_encoder(rgb_hr)
        if rgb_features.feature_lr.shape[-2:] != (height_lr, width_lr):
            raise RuntimeError(
                "RGB encoder LR grid does not align with geometry: "
                f"{rgb_features.feature_lr.shape[-2:]} vs {(height_lr, width_lr)}"
            )
        geometry_feature_lr = self.geometry_encoder(geometry_lr)
        recurrent_features = [rgb_features.feature_lr, geometry_feature_lr]
        topk_effective_weights: Tensor | None = None
        topk_effective_valid_mask: Tensor | None = None
        supplied_topk = (
            history_topk_disparity_hr_px,
            history_topk_confidence,
            history_topk_fractional_offset_px,
            history_topk_age_frames,
            history_topk_weights,
            history_topk_valid_mask,
        )
        if self.topk_history_encoder is None:
            if any(value is not None for value in supplied_topk):
                raise ValueError(
                    "top-K history inputs require temporal_history_top_k at construction"
                )
        else:
            assert self.temporal_history_top_k is not None
            if all(value is None for value in supplied_topk):
                topk_shape = (
                    batch,
                    self.temporal_history_top_k,
                    height_lr,
                    width_lr,
                )
                topk_disparity = disparity_ffs_hr_px.new_zeros(topk_shape)
                topk_confidence = disparity_ffs_hr_px.new_zeros(topk_shape)
                topk_phase = disparity_ffs_hr_px.new_zeros(
                    (batch, self.temporal_history_top_k, 2, height_lr, width_lr)
                )
                topk_age = disparity_ffs_hr_px.new_zeros(topk_shape)
                topk_weights = disparity_ffs_hr_px.new_zeros(topk_shape)
                topk_valid = torch.zeros(
                    topk_shape, dtype=torch.bool, device=rgb_hr.device
                )
            elif any(value is None for value in supplied_topk):
                raise ValueError("all six top-K history tensors must be supplied together")
            else:
                assert history_topk_disparity_hr_px is not None
                assert history_topk_confidence is not None
                assert history_topk_fractional_offset_px is not None
                assert history_topk_age_frames is not None
                assert history_topk_weights is not None
                assert history_topk_valid_mask is not None
                topk_shape = (
                    batch,
                    self.temporal_history_top_k,
                    height_lr,
                    width_lr,
                )
                if history_topk_disparity_hr_px.shape != topk_shape:
                    raise ValueError(
                        "history_topk_disparity_hr_px must have shape "
                        f"{topk_shape}, got {tuple(history_topk_disparity_hr_px.shape)}"
                    )
                for name, value in (
                    ("history_topk_confidence", history_topk_confidence),
                    ("history_topk_age_frames", history_topk_age_frames),
                    ("history_topk_weights", history_topk_weights),
                ):
                    if value.shape != topk_shape:
                        raise ValueError(f"{name} must have shape {topk_shape}")
                expected_phase_shape = (
                    batch,
                    self.temporal_history_top_k,
                    2,
                    height_lr,
                    width_lr,
                )
                if history_topk_fractional_offset_px.shape != expected_phase_shape:
                    raise ValueError(
                        "history_topk_fractional_offset_px must have shape "
                        f"{expected_phase_shape}"
                    )
                if history_topk_valid_mask.shape != topk_shape:
                    raise ValueError(
                        f"history_topk_valid_mask must have shape {topk_shape}"
                    )
                topk_disparity = history_topk_disparity_hr_px
                topk_confidence = history_topk_confidence
                topk_phase = history_topk_fractional_offset_px
                topk_age = history_topk_age_frames
                topk_weights = history_topk_weights
                topk_valid = history_topk_valid_mask
            encoding = self.topk_history_encoder(
                topk_disparity.to(dtype=target_dtype),
                topk_confidence.to(dtype=target_dtype),
                topk_phase.to(dtype=target_dtype),
                topk_age.to(dtype=target_dtype),
                topk_weights.to(dtype=target_dtype),
                topk_valid,
            )
            recurrent_features.append(encoding.aggregate_feature)
            topk_effective_weights = encoding.effective_weights
            topk_effective_valid_mask = (
                topk_valid & (encoding.effective_weights > 0)
            )
        recurrent_input_lr = torch.cat(recurrent_features, dim=1)
        fused_feature_lr, next_hidden_state = self.temporal_fusion(
            recurrent_input_lr, hidden_state
        )

        source_weights = self.source_gating(fused_feature_lr, source_valid_mask)
        source_disparities_hr_px = torch.cat(safe_disparities_hr_px, dim=1)
        disparity_mix_hr_px = torch.sum(
            source_weights * source_disparities_hr_px, dim=1, keepdim=True
        )
        residual_hr_px_lr_grid = self.residual_limit_hr_px * torch.tanh(
            self.disparity_residual_head(fused_feature_lr)
        )
        disparity_pre_lower_bound_lr_grid = (
            disparity_mix_hr_px + residual_hr_px_lr_grid
        )
        disparity_refined_hr_px_lr_grid = disparity_pre_lower_bound_lr_grid
        if self.positivity_floor_hr_px is not None:
            disparity_refined_hr_px_lr_grid = disparity_refined_hr_px_lr_grid.clamp_min(
                self.positivity_floor_hr_px
            )
        convex_mask_logits = self.convex_mask_head(fused_feature_lr)
        disparity_convex_hr_px = self.convex_upsampler(
            disparity_refined_hr_px_lr_grid, convex_mask_logits
        )

        hidden_feature_hr = functional.interpolate(
            self.hidden_to_hr(fused_feature_lr),
            size=rgb_hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hr_decoder_input = torch.cat(
            (hidden_feature_hr, rgb_features.feature_hr), dim=1
        )
        hr_outputs = self.hr_output_head(hr_decoder_input)
        residual_hr_px = hr_outputs[:, :1]
        log_variance = hr_outputs[:, 1:2].clamp(*self.log_variance_bounds)
        disparity_pre_lower_bound_raw_hr_px = disparity_convex_hr_px + residual_hr_px
        disparity_raw_hr_px = disparity_pre_lower_bound_raw_hr_px
        if self.positivity_floor_hr_px is not None:
            disparity_raw_hr_px = disparity_raw_hr_px.clamp_min(self.positivity_floor_hr_px)

        disparity_ffs_bilinear_hr_px = functional.interpolate(
            safe_disparities_hr_px[0],
            size=rgb_hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence_ffs_hr = functional.interpolate(
            safe_confidences[0],
            size=rgb_hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        anchor_gate = (1.0 - confidence_ffs_hr + 0.1).clamp(0.1, 1.0)
        disparity_anchored_hr_px = disparity_ffs_bilinear_hr_px + anchor_gate * (
            disparity_raw_hr_px - disparity_ffs_bilinear_hr_px
        )
        valid_probability: Tensor | None = None
        completion_probability: Tensor | None = None
        valid_logits: Tensor | None = None
        completion_logits: Tensor | None = None
        output_valid_mask: Tensor | None = None
        completion_mask: Tensor | None = None
        if self.validity_completion_head is None:
            disparity_hr_px = disparity_anchored_hr_px
        else:
            # V2 represents disparity as a non-negative magnitude plus an
            # explicit physical-valid gate. Softplus avoids mirroring a large
            # negative raw proposal into a large positive disparity. Invalid
            # pixels are hard-masked to exact zero below, so the magnitude
            # parameterization never creates epsilon-valid fake points.
            validity_outputs = self.validity_completion_head(hr_decoder_input)
            valid_logits = validity_outputs[:, :1].float()
            completion_logits = validity_outputs[:, 1:2].float()
            source_support_hr = functional.interpolate(
                source_valid_mask.any(dim=1, keepdim=True).to(dtype=torch.float32),
                size=rgb_hr.shape[-2:],
                mode="nearest",
            ).to(dtype=torch.bool)
            valid_ffs_hr = functional.interpolate(
                source_valid_masks[0].to(dtype=torch.float32),
                size=rgb_hr.shape[-2:],
                mode="nearest",
            ).to(dtype=torch.bool)
            trusted_ffs_hr = valid_ffs_hr & (
                confidence_ffs_hr.float()
                >= self.trusted_ffs_confidence_threshold
            )
            hole_hr = ~valid_ffs_hr
            raw_valid_probability = torch.sigmoid(valid_logits)
            raw_completion_probability = torch.sigmoid(completion_logits)
            valid_probability = torch.where(
                source_support_hr,
                raw_valid_probability,
                torch.zeros_like(raw_valid_probability),
            )
            completion_probability = torch.where(
                source_support_hr & hole_hr,
                raw_completion_probability,
                torch.zeros_like(raw_completion_probability),
            )
            predicted_valid = (
                valid_probability >= self.physical_valid_threshold
            )
            completion_mask = (
                hole_hr
                & source_support_hr
                & (completion_probability >= self.completion_threshold)
            )
            requested_valid = (
                trusted_ffs_hr
                | (valid_ffs_hr & predicted_valid)
                | completion_mask
            )
            raw_magnitude_hr_px = functional.softplus(
                disparity_raw_hr_px.float(), beta=4.0
            )
            disparity_raw_hr_px = torch.where(
                source_support_hr,
                raw_magnitude_hr_px,
                torch.zeros_like(raw_magnitude_hr_px),
            )
            anchored_magnitude_hr_px = (
                disparity_ffs_bilinear_hr_px.float()
                + anchor_gate.float()
                * (
                    disparity_raw_hr_px
                    - disparity_ffs_bilinear_hr_px.float()
                )
            )
            # High-confidence FFS ownership is an exact conservation rule in
            # V2, not merely a small correction gate.
            anchored_magnitude_hr_px = torch.where(
                trusted_ffs_hr,
                disparity_ffs_bilinear_hr_px.float(),
                anchored_magnitude_hr_px,
            )
            output_valid_mask = requested_valid & torch.isfinite(
                anchored_magnitude_hr_px
            ) & (anchored_magnitude_hr_px > 0)
            completion_mask = completion_mask & output_valid_mask
            disparity_hr_px = torch.where(
                output_valid_mask,
                anchored_magnitude_hr_px,
                torch.zeros_like(anchored_magnitude_hr_px),
            )
        uncertainty = torch.exp(log_variance)

        return ModelOutput(
            disparity_hr_px=disparity_hr_px,
            disparity_raw_hr_px=disparity_raw_hr_px,
            source_weights=source_weights,
            log_variance=log_variance,
            uncertainty=uncertainty,
            hidden_state=next_hidden_state,
            anchor_gate=anchor_gate,
            source_valid_mask=source_valid_mask,
            disparity_source_mix_hr_px_lr_grid=disparity_mix_hr_px,
            disparity_post_lr_residual_hr_px_lr_grid=(
                disparity_refined_hr_px_lr_grid
            ),
            disparity_post_convex_hr_px=disparity_convex_hr_px,
            disparity_pre_lower_bound_hr_px_lr_grid=(
                disparity_pre_lower_bound_lr_grid
                if self.positivity_floor_hr_px is not None
                else None
            ),
            disparity_pre_lower_bound_raw_hr_px=(
                disparity_pre_lower_bound_raw_hr_px
                if self.positivity_floor_hr_px is not None
                else None
            ),
            valid_probability=valid_probability,
            completion_probability=completion_probability,
            valid_logits=valid_logits,
            completion_logits=completion_logits,
            output_valid_mask=output_valid_mask,
            completion_mask=completion_mask,
            history_topk_effective_weights=topk_effective_weights,
            history_topk_valid_mask=topk_effective_valid_mask,
        )
