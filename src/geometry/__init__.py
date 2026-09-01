"""Explicit metric disparity and calibrated camera geometry."""

from .camera import (
    PinholeIntrinsics,
    crop_intrinsics,
    resize_intrinsics,
    resize_intrinsics_align_corners_false,
    validate_intrinsics,
)
from .calibration_context import (
    rectified_stereo_transform_4x4,
    temporal_conditioning_transforms,
)
from .disparity import (
    depth_from_disparity,
    disparity_from_depth,
    disparity_hr_to_lr_units,
    disparity_lr_to_hr_units,
    hr_disparity_to_lr_pixels,
    lr_disparity_to_hr_pixels,
    valid_depth_mask,
    valid_disparity_mask,
)
from .align_vggt import (
    ScaleOnlyEstimate,
    VGGTAlignmentResult,
    align_vggt_depth_to_ffs_disparity,
    ffs_trusted_mask,
    robust_scale_only_irls,
)
from .pose_scale import (
    BaselineScaleEstimate,
    MetricScaledVGGTGeometry,
    PoseQuality,
    assess_pose_quality,
    camera_centers_from_extrinsics,
    estimate_baseline_metric_scale,
    metric_scale_vggt_geometry,
)
from .pose_quality import (
    CompletePoseQuality,
    DepthDisparityConsistency,
    PhotometricReprojectionDiagnostic,
    adjacent_left_photometric_reprojection,
    combine_pose_quality,
    depth_disparity_consistency,
    validate_raw_cache_pair,
)
from .zbuffer_reproject import (
    WarpResult,
    relative_camera_transform,
    zbuffer_reproject,
)
from .history_confidence import HistoryConfidenceResult, history_confidence
from .topk_splat import (
    TOPK_DIVERSITY_V31_CONTRACT,
    TopKDiversityDiagnostics,
    TopKSplatResult,
    merge_topk_splat_results,
    merge_topk_splat_results_v31,
    topk_diversity_diagnostics,
    topk_z_aware_splat,
    topk_z_aware_splat_v2,
)

__all__ = [
    "PinholeIntrinsics",
    "BaselineScaleEstimate",
    "CompletePoseQuality",
    "DepthDisparityConsistency",
    "MetricScaledVGGTGeometry",
    "HistoryConfidenceResult",
    "PoseQuality",
    "PhotometricReprojectionDiagnostic",
    "ScaleOnlyEstimate",
    "VGGTAlignmentResult",
    "WarpResult",
    "TopKSplatResult",
    "TopKDiversityDiagnostics",
    "TOPK_DIVERSITY_V31_CONTRACT",
    "align_vggt_depth_to_ffs_disparity",
    "adjacent_left_photometric_reprojection",
    "assess_pose_quality",
    "camera_centers_from_extrinsics",
    "rectified_stereo_transform_4x4",
    "combine_pose_quality",
    "crop_intrinsics",
    "depth_from_disparity",
    "disparity_from_depth",
    "disparity_hr_to_lr_units",
    "disparity_lr_to_hr_units",
    "estimate_baseline_metric_scale",
    "depth_disparity_consistency",
    "ffs_trusted_mask",
    "hr_disparity_to_lr_pixels",
    "history_confidence",
    "merge_topk_splat_results",
    "merge_topk_splat_results_v31",
    "lr_disparity_to_hr_pixels",
    "metric_scale_vggt_geometry",
    "relative_camera_transform",
    "resize_intrinsics",
    "resize_intrinsics_align_corners_false",
    "robust_scale_only_irls",
    "valid_depth_mask",
    "valid_disparity_mask",
    "validate_intrinsics",
    "validate_raw_cache_pair",
    "topk_z_aware_splat",
    "topk_z_aware_splat_v2",
    "topk_diversity_diagnostics",
    "temporal_conditioning_transforms",
    "zbuffer_reproject",
]
