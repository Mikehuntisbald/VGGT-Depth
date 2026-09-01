"""Frozen Stage-B base plus the trainable Stage-C epipolar refiner."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from geometry.epipolar import EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION
from losses.disparity import disparity_loss, finite_masked_mean

from .epipolar_refiner import (
    EpipolarRefinementOutput,
    HREpipolarRefiner,
)


BaseEndpointPredictor = Callable[[nn.Module, Mapping[str, Any]], Tensor]


@dataclass(frozen=True, slots=True)
class EpipolarStageOutput:
    """Stage-C endpoint output, with every disparity in HR pixel units.

    Shapes are ``[B,1,H,W]`` for all disparity/correction fields.
    ``refinement`` also carries the ``[B,G,K,H,W]`` correlation and
    ``[B,K,H,W]`` candidate-valid mask.
    """

    base_disparity_hr_px: Tensor
    refined_disparity_hr_px: Tensor
    correction_hr_px: Tensor
    confidence: Tensor
    refinement: EpipolarRefinementOutput


@dataclass(frozen=True, slots=True)
class EpipolarStageLoss:
    """Finite Stage-C supervision; empty masks produce differentiable zero."""

    total: Tensor
    disparity: Tensor
    correction_regularizer: Tensor
    valid_pixel_count: int

    def detached_scalars(self) -> dict[str, float | int]:
        return {
            "total": float(self.total.detach().float().cpu().item()),
            "disparity": float(self.disparity.detach().float().cpu().item()),
            "correction_regularizer": float(
                self.correction_regularizer.detach().float().cpu().item()
            ),
            "valid_pixel_count": self.valid_pixel_count,
        }


class FrozenTemporalEpipolarStage(nn.Module):
    """Run a frozen/no-grad Stage-B endpoint then refine with real right RGB.

    ``base_endpoint_predictor`` owns the exact causal T=3 unroll and must return
    HR-pixel left disparity ``[B,1,H,W]``. The base is always frozen and kept
    in evaluation mode, even when this wrapper is put in training mode. Only
    :attr:`refiner` can create optimizer parameters.
    """

    def __init__(
        self,
        base_model: nn.Module,
        refiner: HREpipolarRefiner,
        base_endpoint_predictor: BaseEndpointPredictor,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be torch.nn.Module")
        if not isinstance(refiner, HREpipolarRefiner):
            raise TypeError("refiner must be HREpipolarRefiner")
        if not callable(base_endpoint_predictor):
            raise TypeError("base_endpoint_predictor must be callable")
        self.base_model = base_model
        self.refiner = refiner
        self.base_endpoint_predictor = base_endpoint_predictor
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "FrozenTemporalEpipolarStage":
        super().train(mode)
        self.base_model.eval()
        self.refiner.train(mode)
        return self

    def forward(
        self,
        batch: Mapping[str, Any],
        rgb_right_hr: Tensor | None = None,
    ) -> EpipolarStageOutput:
        """Refine the current endpoint of a causal temporal batch.

        Args:
            batch: Collated Stage-B fields, including left RGB sequence
                ``[B,3,3,H,W]``. If ``rgb_right_hr`` is omitted, the mapping
                must contain endpoint right RGB ``[B,3,H,W]``.
            rgb_right_hr: Optional explicit endpoint right rectified RGB.
        """

        rgb_left_sequence = batch.get("rgb_hr_sequence")
        if not isinstance(rgb_left_sequence, Tensor) or (
            rgb_left_sequence.ndim != 5
            or rgb_left_sequence.shape[1] != 3
            or rgb_left_sequence.shape[2] != 3
        ):
            raise ValueError("batch rgb_hr_sequence must have shape [B,3,3,H,W]")
        rgb_left_hr = rgb_left_sequence[:, -1]
        if rgb_right_hr is None:
            candidate = batch.get("rgb_right_hr")
            if not isinstance(candidate, Tensor):
                raise ValueError("batch rgb_right_hr must have shape [B,3,H,W]")
            rgb_right_hr = candidate
        if rgb_right_hr.shape != rgb_left_hr.shape:
            raise ValueError(
                f"right RGB shape {tuple(rgb_right_hr.shape)} does not match "
                f"endpoint left RGB {tuple(rgb_left_hr.shape)}"
            )
        intrinsics_left_sequence = batch.get("K_hr_sequence")
        intrinsics_right_hr = batch.get("K_right_hr")
        if not isinstance(intrinsics_left_sequence, Tensor) or (
            intrinsics_left_sequence.shape
            != (rgb_left_hr.shape[0], 3, 3, 3)
        ):
            raise ValueError("batch K_hr_sequence must have shape [B,3,3,3]")
        if not isinstance(intrinsics_right_hr, Tensor) or (
            intrinsics_right_hr.shape != (rgb_left_hr.shape[0], 3, 3)
        ):
            raise ValueError("batch K_right_hr must have shape [B,3,3]")
        if intrinsics_left_sequence.device != rgb_left_hr.device or (
            intrinsics_right_hr.device != rgb_left_hr.device
        ):
            raise ValueError("RGB and stereo intrinsics must share a device")
        right_row_scale = batch.get("epipolar_right_row_scale")
        right_row_offset_hr_px = batch.get("epipolar_right_row_offset_hr_px")
        row_mapping_source = batch.get("epipolar_right_row_mapping_source")
        if not isinstance(right_row_scale, Tensor) or (
            right_row_scale.shape != (rgb_left_hr.shape[0],)
        ):
            raise ValueError("batch epipolar_right_row_scale must have shape [B]")
        if not isinstance(right_row_offset_hr_px, Tensor) or (
            right_row_offset_hr_px.shape != (rgb_left_hr.shape[0],)
        ):
            raise ValueError(
                "batch epipolar_right_row_offset_hr_px must have shape [B]"
            )
        if right_row_scale.device != rgb_left_hr.device or (
            right_row_offset_hr_px.device != rgb_left_hr.device
        ):
            raise ValueError("RGB and explicit epipolar row mapping must share a device")
        if not isinstance(row_mapping_source, list) or len(row_mapping_source) != (
            rgb_left_hr.shape[0]
        ):
            raise ValueError("batch epipolar row-mapping provenance is malformed")
        if any(
            source != EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION
            for source in row_mapping_source
        ):
            raise ValueError("batch epipolar row-mapping contract mismatch")
        if not torch.equal(right_row_scale, torch.ones_like(right_row_scale)) or (
            not torch.equal(
                right_row_offset_hr_px,
                torch.zeros_like(right_row_offset_hr_px),
            )
        ):
            raise ValueError(
                "formal Stage-C requires exact same-row scale=1 and offset=0"
            )

        # The no-grad island is unconditional, not caller-dependent. This
        # prevents Stage-B activation graphs or parameter gradients even if a
        # caller forgets to detach the returned base disparity.
        self.base_model.eval()
        with torch.no_grad():
            base_disparity_hr_px = self.base_endpoint_predictor(
                self.base_model, batch
            )
        if not isinstance(base_disparity_hr_px, Tensor):
            raise TypeError("base endpoint predictor must return a torch.Tensor")
        expected_shape = (rgb_left_hr.shape[0], 1, *rgb_left_hr.shape[-2:])
        if base_disparity_hr_px.shape != expected_shape:
            raise ValueError(
                "base disparity must have shape "
                f"{expected_shape}, got {tuple(base_disparity_hr_px.shape)}"
            )
        if not base_disparity_hr_px.is_floating_point():
            raise TypeError("base disparity must be floating point in HR pixels")
        base_disparity_hr_px = base_disparity_hr_px.detach()
        refinement = self.refiner(
            rgb_left_hr,
            rgb_right_hr,
            base_disparity_hr_px,
            right_row_scale=right_row_scale,
            right_row_offset_hr_px=right_row_offset_hr_px,
        )
        return EpipolarStageOutput(
            base_disparity_hr_px=base_disparity_hr_px,
            refined_disparity_hr_px=refinement.corrected_disparity_hr_px,
            correction_hr_px=refinement.correction_hr_px,
            confidence=refinement.confidence,
            refinement=refinement,
        )


def compute_epipolar_stage_loss(
    output: EpipolarStageOutput,
    target_disparity_hr_px: Tensor,
    target_trusted_mask: Tensor,
    *,
    target_confidence: Tensor | None = None,
    correction_regularizer_weight: float = 0.01,
) -> EpipolarStageLoss:
    """Supervise refined HR disparity only where teacher and search are valid."""

    if not math.isfinite(correction_regularizer_weight) or (
        correction_regularizer_weight < 0
    ):
        raise ValueError("correction_regularizer_weight must be finite and >= 0")
    expected_shape = output.refined_disparity_hr_px.shape
    for name, value in (
        ("target_disparity_hr_px", target_disparity_hr_px),
        ("target_trusted_mask", target_trusted_mask),
    ):
        if not isinstance(value, Tensor) or value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if target_confidence is not None and target_confidence.shape != expected_shape:
        raise ValueError(
            f"target_confidence must have shape {tuple(expected_shape)}"
        )
    search_valid = output.refinement.candidate_valid_mask.any(
        dim=1, keepdim=True
    )
    usable = (
        target_trusted_mask.to(dtype=torch.bool)
        & search_valid
        & torch.isfinite(target_disparity_hr_px)
        & (target_disparity_hr_px > 0)
    )
    disparity = disparity_loss(
        output.refined_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=usable,
        weights=target_confidence,
    )
    correction_regularizer = finite_masked_mean(
        output.correction_hr_px.abs(), usable
    )
    total = disparity + correction_regularizer_weight * correction_regularizer
    if not bool(torch.isfinite(total.detach()).all().item()):
        raise FloatingPointError("Stage-C epipolar loss is non-finite")
    return EpipolarStageLoss(
        total=total,
        disparity=disparity,
        correction_regularizer=correction_regularizer,
        valid_pixel_count=int(usable.sum().item()),
    )


__all__ = [
    "BaseEndpointPredictor",
    "EpipolarStageLoss",
    "EpipolarStageOutput",
    "FrozenTemporalEpipolarStage",
    "compute_epipolar_stage_loss",
]
