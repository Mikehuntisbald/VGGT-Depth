"""Explicit physical-validity and FFS-hole-completion supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .disparity import finite_masked_mean


@dataclass(frozen=True, slots=True)
class ValidityCompletionLoss:
    """Empty-safe V2 classification and calibration terms."""

    valid_bce: Tensor
    completion_bce: Tensor
    calibration: Tensor
    valid_pixel_count: int
    completion_pixel_count: int


def _check_shape(reference: Tensor, value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def _masked_bce_with_logits(logits: Tensor, target: Tensor, domain: Tensor) -> Tensor:
    safe_logits = torch.where(domain, logits, torch.zeros_like(logits))
    safe_target = torch.where(domain, target, torch.zeros_like(target))
    per_pixel = functional.binary_cross_entropy_with_logits(
        safe_logits, safe_target, reduction="none"
    )
    return finite_masked_mean(per_pixel, domain)


def validity_completion_loss(
    *,
    valid_logits: Tensor,
    completion_logits: Tensor,
    valid_probability: Tensor,
    completion_probability: Tensor,
    teacher_valid_mask: Tensor,
    teacher_confidence: Tensor,
    observation_valid_mask_hr: Tensor,
) -> ValidityCompletionLoss:
    """Supervise physical validity separately from FFS-hole completion.

    All tensors are ``[B,1,H,W]``. The soft target is
    ``teacher_valid * teacher_confidence``. Completion is supervised only in
    current-FFS holes. Calibration is a Brier term over the declared domains.
    Empty domains return differentiable zero and no positive epsilon is made.
    """

    if not isinstance(valid_logits, Tensor) or valid_logits.ndim != 4:
        raise ValueError("valid_logits must have shape [B,1,H,W]")
    if valid_logits.shape[1] != 1 or not valid_logits.is_floating_point():
        raise ValueError("valid_logits must be floating [B,1,H,W]")
    for name, value in (
        ("completion_logits", completion_logits),
        ("valid_probability", valid_probability),
        ("completion_probability", completion_probability),
        ("teacher_valid_mask", teacher_valid_mask),
        ("teacher_confidence", teacher_confidence),
        ("observation_valid_mask_hr", observation_valid_mask_hr),
    ):
        _check_shape(valid_logits, value, name)
        if value.device != valid_logits.device:
            raise ValueError(f"{name} must share valid_logits device")
    finite_target = torch.isfinite(teacher_confidence)
    confidence = torch.nan_to_num(
        teacher_confidence.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    target_valid_soft = teacher_valid_mask.to(dtype=torch.bool).float() * confidence
    valid_domain = finite_target & torch.isfinite(valid_logits)
    hole_domain = (
        ~observation_valid_mask_hr.to(dtype=torch.bool)
        & valid_domain
        & torch.isfinite(completion_logits)
    )
    valid_bce = _masked_bce_with_logits(
        valid_logits.float(), target_valid_soft, valid_domain
    )
    completion_bce = _masked_bce_with_logits(
        completion_logits.float(), target_valid_soft, hole_domain
    )
    calibration = finite_masked_mean(
        (valid_probability.float() - target_valid_soft).square(), valid_domain
    ) + finite_masked_mean(
        (completion_probability.float() - target_valid_soft).square(), hole_domain
    )
    return ValidityCompletionLoss(
        valid_bce=valid_bce,
        completion_bce=completion_bce,
        calibration=calibration,
        valid_pixel_count=int(valid_domain.sum().item()),
        completion_pixel_count=int(hole_domain.sum().item()),
    )


__all__ = ["ValidityCompletionLoss", "validity_completion_loss"]
