"""Validated manifest and scale-aligned crop interfaces."""

from .cache_dataset import (
    CACHE_SCHEMA_VERSION,
    CacheIdentity,
    CacheMismatchError,
    load_cache_record,
    save_cache_record,
)
from .crop import CropWindow, sample_aligned_crop, validate_crop_origin
from .manifest import (
    ManifestRecord,
    ManifestValidationError,
    iter_manifest,
    load_manifest,
    write_manifest,
)
from .collate import collate_temporal_training_samples, collate_training_samples
from .temporal_training_dataset import (
    CachedTemporalTrainingDataset,
    TemporalTrainingSample,
)
from .training_dataset import (
    CachedFFSTrainingDataset,
    CausalWindow,
    FFSTrainingSample,
    build_causal_windows,
)

__all__ = [
    "CropWindow",
    "CACHE_SCHEMA_VERSION",
    "CacheIdentity",
    "CacheMismatchError",
    "CachedFFSTrainingDataset",
    "CachedTemporalTrainingDataset",
    "CausalWindow",
    "ManifestRecord",
    "ManifestValidationError",
    "FFSTrainingSample",
    "TemporalTrainingSample",
    "build_causal_windows",
    "collate_training_samples",
    "collate_temporal_training_samples",
    "iter_manifest",
    "load_manifest",
    "load_cache_record",
    "sample_aligned_crop",
    "save_cache_record",
    "validate_crop_origin",
    "write_manifest",
]
