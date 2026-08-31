"""Frozen, cache-only backbone adapters."""

from .ffs_adapter import FFSAdapter, FFSOutput
from .vggt_omega_adapter import (
    ImagePreprocessMetadata,
    PreprocessedVGGTOmegaInput,
    VGGTOmegaAdapter,
    VGGTOmegaOutput,
    preprocess_vggt_omega_images,
)

__all__ = [
    "FFSAdapter",
    "FFSOutput",
    "ImagePreprocessMetadata",
    "PreprocessedVGGTOmegaInput",
    "VGGTOmegaAdapter",
    "VGGTOmegaOutput",
    "preprocess_vggt_omega_images",
]
