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
from .spring import (
    SPRING_BASELINE_M,
    SPRING_DISPARITY_CONVENTION,
    SPRING_DISPARITY_SIZE_HW,
    SPRING_IMAGE_SIZE_HW,
    SPRING_POSE_CONVENTION,
    SpringFormatError,
    SpringFrame,
    SpringSequence,
    build_spring_manifest,
    iter_spring_sequences,
    load_spring_disparity,
    load_spring_sequence,
    relative_spring_pose,
    resolve_spring_split_root,
    spring_gt_pose_from_manifest,
    spring_manifest_records,
)
from .stereo_calibration import (
    RectifiedCalibrationIndex,
    RectifiedCalibrationRecord,
    SPRING_NATIVE_DERIVATION,
    SPRING_NATIVE_METADATA_CONTRACT,
    load_rectified_calibration_sidecar,
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
    "RectifiedCalibrationIndex",
    "RectifiedCalibrationRecord",
    "SPRING_NATIVE_DERIVATION",
    "SPRING_NATIVE_METADATA_CONTRACT",
    "FFSTrainingSample",
    "TemporalTrainingSample",
    "build_causal_windows",
    "collate_training_samples",
    "collate_temporal_training_samples",
    "iter_manifest",
    "load_manifest",
    "load_rectified_calibration_sidecar",
    "load_cache_record",
    "sample_aligned_crop",
    "save_cache_record",
    "validate_crop_origin",
    "write_manifest",
    "SPRING_BASELINE_M",
    "SPRING_DISPARITY_CONVENTION",
    "SPRING_DISPARITY_SIZE_HW",
    "SPRING_IMAGE_SIZE_HW",
    "SPRING_POSE_CONVENTION",
    "SpringFormatError",
    "SpringFrame",
    "SpringSequence",
    "build_spring_manifest",
    "iter_spring_sequences",
    "load_spring_disparity",
    "load_spring_sequence",
    "relative_spring_pose",
    "resolve_spring_split_root",
    "spring_gt_pose_from_manifest",
    "spring_manifest_records",
]
