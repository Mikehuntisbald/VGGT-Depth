"""Named, auditable composition of the MVP training objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Fixed first-round loss coefficients from the experiment contract."""

    disparity: float = 1.00
    measurement: float = 0.50
    gradient: float = 0.20
    temporal: float = 0.10
    epipolar: float = 0.05
    uncertainty_nll: float = 0.01
    gate_regularizer: float = 0.02


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Unweighted named terms and their weighted scalar total."""

    total: Tensor
    disparity: Tensor
    measurement: Tensor
    gradient: Tensor
    temporal: Tensor
    epipolar: Tensor
    uncertainty_nll: Tensor
    gate_regularizer: Tensor
    # Populated only by the separate D-025 ablation.  Keeping this optional
    # preserves the baseline loss log schema and computation path exactly.
    positivity_penalty: Tensor | None = None
    # V2-only explicit classification/calibration terms. Their absence keeps
    # the canonical loss computation and JSON log schema unchanged.
    valid_bce: Tensor | None = None
    completion_bce: Tensor | None = None
    validity_calibration: Tensor | None = None

    def detached_scalars(self) -> dict[str, float]:
        """Return logging values without retaining an autograd graph."""

        values = {
            name: float(value.detach().cpu().item())
            for name, value in (
                ("total", self.total),
                ("disparity", self.disparity),
                ("measurement", self.measurement),
                ("gradient", self.gradient),
                ("temporal", self.temporal),
                ("epipolar", self.epipolar),
                ("uncertainty_nll", self.uncertainty_nll),
                ("gate_regularizer", self.gate_regularizer),
            )
        }
        if self.positivity_penalty is not None:
            values["positivity_penalty"] = float(
                self.positivity_penalty.detach().cpu().item()
            )
        for name in ("valid_bce", "completion_bce", "validity_calibration"):
            value = getattr(self, name)
            if value is not None:
                values[name] = float(value.detach().cpu().item())
        return values


def combine_loss_terms(
    *,
    disparity: Tensor,
    measurement: Tensor,
    gradient: Tensor,
    temporal: Tensor,
    epipolar: Tensor,
    uncertainty_nll: Tensor,
    gate_regularizer: Tensor,
    weights: LossWeights = LossWeights(),
) -> LossBreakdown:
    """Combine finite scalar terms using the declared MVP coefficients."""

    terms = {
        "disparity": disparity,
        "measurement": measurement,
        "gradient": gradient,
        "temporal": temporal,
        "epipolar": epipolar,
        "uncertainty_nll": uncertainty_nll,
        "gate_regularizer": gate_regularizer,
    }
    for name, value in terms.items():
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
    # CPU unit tests and diagnostics retain precise term-level errors without
    # imposing seven device synchronizations on every CUDA micro-batch.  On
    # CUDA the trainer checks the combined scalar once and uses
    # clip_grad_norm_(error_if_nonfinite=True) at the optimizer boundary.
    if all(value.device.type == "cpu" for value in terms.values()):
        finite = torch.stack(
            [torch.isfinite(value.detach()).reshape(()) for value in terms.values()]
        )
        if not bool(finite.all().item()):
            bad_names = [
                name
                for name, value in terms.items()
                if not bool(torch.isfinite(value.detach()).item())
            ]
            raise ValueError(f"non-finite loss terms: {bad_names}")
    total = sum(
        value * float(getattr(weights, name)) for name, value in terms.items()
    )
    return LossBreakdown(total=total, **terms)


__all__ = ["LossBreakdown", "LossWeights", "combine_loss_terms"]
