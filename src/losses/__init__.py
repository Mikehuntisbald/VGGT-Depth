"""Numerically safe supervised and consistency losses."""

from .disparity import (
    charbonnier,
    disparity_loss,
    epipolar_disparity_loss,
    finite_masked_mean,
    gradient_loss,
    lower_bound_penalty,
)
from .composite import LossBreakdown, LossWeights, combine_loss_terms
from .measurement import measurement_consistency_loss, sample_hr_at_lr_centers
from .temporal import temporal_consistency_loss
from .uncertainty import ffs_gate_regularizer, laplace_uncertainty_nll

__all__ = [
    "charbonnier",
    "disparity_loss",
    "epipolar_disparity_loss",
    "ffs_gate_regularizer",
    "finite_masked_mean",
    "gradient_loss",
    "lower_bound_penalty",
    "LossBreakdown",
    "LossWeights",
    "laplace_uncertainty_nll",
    "measurement_consistency_loss",
    "sample_hr_at_lr_centers",
    "temporal_consistency_loss",
    "combine_loss_terms",
]
