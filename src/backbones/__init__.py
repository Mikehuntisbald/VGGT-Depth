"""Frozen, cache-only backbone adapters."""

from .ffs_adapter import FFSAdapter, FFSOutput
from .foundationstereo_adapter import (
    FoundationStereoAdapter,
    FoundationStereoDependencyError,
    FoundationStereoLoadError,
    FoundationStereoLoadInfo,
    infer_foundation_stereo,
    infer_foundation_stereo_vit_size,
    load_foundation_stereo,
    load_foundation_stereo_adapter,
)
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
    "FoundationStereoAdapter",
    "FoundationStereoDependencyError",
    "FoundationStereoLoadError",
    "FoundationStereoLoadInfo",
    "infer_foundation_stereo",
    "infer_foundation_stereo_vit_size",
    "load_foundation_stereo",
    "load_foundation_stereo_adapter",
    "ImagePreprocessMetadata",
    "PreprocessedVGGTOmegaInput",
    "VGGTOmegaAdapter",
    "VGGTOmegaOutput",
    "preprocess_vggt_omega_images",
]
