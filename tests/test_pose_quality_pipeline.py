from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from data.cache_dataset import CacheMismatchError, sha256_file
from geometry.align_vggt import align_vggt_depth_to_ffs_disparity, ffs_trusted_mask
from geometry.pose_quality import (
    DepthDisparityConsistency,
    PhotometricReprojectionDiagnostic,
    adjacent_left_photometric_reprojection,
    combine_pose_quality,
    depth_disparity_consistency,
    validate_raw_cache_pair,
)
from geometry.pose_scale import (
    estimate_baseline_metric_scale,
    scale_vggt_translations_and_depth,
)
from tools.derive_geometry_cache import GeometryThresholds, derive_geometry


def _five_pair_extrinsics(predicted_baseline: float = 0.2) -> torch.Tensor:
    extrinsics = []
    for _ in range(5):
        left = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1)
        # With R=I and camera-from-world E=[I|t], centre C=-t.
        right = torch.cat(
            (torch.eye(3), torch.tensor([[-predicted_baseline], [0.0], [0.0]])),
            dim=1,
        )
        extrinsics.extend((left, right))
    return torch.stack(extrinsics)


def _write_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((32, 48, 3), value, dtype=np.uint8)).save(path)


def _raw_payloads(
    tmp_path: Path, *, previous_value: int, current_value: int
) -> tuple[dict, dict]:
    previous = tmp_path / "previous.png"
    current = tmp_path / "current.png"
    _write_rgb(previous, previous_value)
    _write_rgb(current, current_value)
    previous_sha = sha256_file(previous)
    current_sha = sha256_file(current)
    sequence_id = "synthetic_sequence"
    records = [
        {
            "sequence_id": sequence_id,
            "frame_id": index,
            "timestamp": index * 0.2,
            "left_path": str(previous if index < 4 else current),
            "right_path": str(previous if index < 4 else current),
            "K": [[40.0, 0.0, 24.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
            "baseline_m": 0.1,
            "gt_disparity_path": None,
        }
        for index in range(5)
    ]
    ordered_images = []
    for index in range(10):
        path = current if index >= 8 else previous
        ordered_images.append(
            {
                "view_index": index,
                "path": str(path),
                "sha256": current_sha if index >= 8 else previous_sha,
            }
        )
    transform_original_to_model = torch.tensor(
        [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    transform_model_to_original = torch.linalg.inv(transform_original_to_model)
    k_original = torch.tensor(records[-1]["K"])
    k_model = transform_original_to_model @ k_original
    depth_arbitrary = torch.full((1, 16, 24), 2.0)
    disparity_hr_px = torch.full((1, 1, 16, 24), 10.0)
    confidence = torch.full_like(disparity_hr_px, 0.95)
    lr_error = torch.full_like(disparity_hr_px, 0.2)
    trusted = torch.ones_like(disparity_hr_px, dtype=torch.bool)
    vggt_payload = {
        "identity": {
            "component": "vggt-omega",
            "upstream_commit": "vggt-commit",
            "checkpoint_sha256": "vggt-checkpoint",
            "torch_version": "test",
            "cuda_version": None,
            "config_sha256": "vggt-config",
        },
        "metadata": {
            "source": {
                "manifest_records": records,
                "target_sequence_id": sequence_id,
                "target_frame_id": 4,
                "target_timestamp": 0.8,
                "ordered_images": ordered_images,
            }
        },
        "tensors": {
            "vggt_depth_current_left_arbitrary": depth_arbitrary,
            "vggt_depth_conf_current_left_unbounded": torch.full_like(
                depth_arbitrary, 2.0
            ),
            "vggt_extrinsics_camera_from_world": _five_pair_extrinsics(),
            "calibrated_intrinsics_original_px": k_original.repeat(10, 1, 1),
            "calibrated_intrinsics_model_px": k_model.repeat(10, 1, 1),
            "original_to_model_transform": transform_original_to_model.repeat(
                10, 1, 1
            ),
            "model_to_original_transform": transform_model_to_original.repeat(
                10, 1, 1
            ),
            "stereo_baseline_m_by_pair": torch.full((5,), 0.1),
        },
    }
    ffs_payload = {
        "identity": {
            "component": "ffs-observation",
            "upstream_commit": "ffs-commit",
            "checkpoint_sha256": "ffs-checkpoint",
            "torch_version": "test",
            "cuda_version": None,
            "config_sha256": "ffs-config",
        },
        "metadata": {
            "source": {
                "manifest_record": records[-1],
                "left_sha256": current_sha,
                "right_sha256": current_sha,
                "ffs_input_shape_bchw": [1, 3, 16, 24],
            },
            "config": {"scale": 2, "right_left_check": True},
        },
        "tensors": {
            "observation_disparity_hr_px": disparity_hr_px,
            "observation_confidence": confidence,
            "observation_left_right_error_lr_px": lr_error,
            "observation_trusted_mask": trusted,
        },
    }
    return vggt_payload, ffs_payload


def test_numeric_geometry_and_all_quality_gates_pass(tmp_path: Path) -> None:
    extrinsics = _five_pair_extrinsics()
    estimate = estimate_baseline_metric_scale(extrinsics, 0.1)
    assert estimate.valid
    assert estimate.alpha_m_per_vggt_unit == pytest.approx(0.5)
    extrinsics_metric, depth_m = scale_vggt_translations_and_depth(
        extrinsics, torch.full((1, 8, 12), 2.0), estimate.alpha_m_per_vggt_unit
    )
    assert torch.allclose(depth_m, torch.ones_like(depth_m))
    assert torch.allclose(extrinsics_metric[1, :, 3], torch.tensor([-0.1, 0.0, 0.0]))

    disparity = torch.full_like(depth_m, 10.0)
    confidence = torch.full_like(depth_m, 0.95)
    lr_error = torch.full_like(depth_m, 0.2)
    trusted = ffs_trusted_mask(disparity, confidence, lr_error)
    alignment = align_vggt_depth_to_ffs_disparity(
        disparity,
        depth_m,
        reliable_ffs_mask=trusted,
        weights=confidence,
        min_reliable_pixels=20,
    )
    assert alignment.valid
    consistency = depth_disparity_consistency(
        disparity,
        alignment.disparity_vggt_aligned_hr_px,
        trusted_mask=trusted,
        weights=confidence,
        min_samples=20,
    )
    assert consistency.valid
    assert consistency.weighted_mae_hr_px == pytest.approx(0.0, abs=1e-5)

    previous = tmp_path / "previous.png"
    current = tmp_path / "current.png"
    _write_rgb(previous, 96)
    _write_rgb(current, 96)
    k = torch.tensor([[20.0, 0.0, 6.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]])
    photo = adjacent_left_photometric_reprojection(
        depth_m,
        extrinsic_previous_camera_from_world_metric=extrinsics_metric[6],
        extrinsic_current_camera_from_world_metric=extrinsics_metric[8],
        intrinsics_previous_model_px=k,
        intrinsics_current_model_px=k,
        previous_model_to_original_transform=torch.eye(3),
        current_model_to_original_transform=torch.eye(3),
        previous_rgb_path=previous,
        current_rgb_path=current,
        min_samples=20,
        min_valid_fraction=0.9,
    )
    assert photo.available and photo.valid
    assert photo.median_absolute_rgb_residual == pytest.approx(0.0)
    combined = combine_pose_quality(estimate, photo, consistency)
    assert combined.pose_valid
    assert not combined.failure_reasons


def test_derived_cache_masks_failed_temporal_pose_but_keeps_static_prior(
    tmp_path: Path,
) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=0, current_value=255
    )
    tensors, metadata = derive_geometry(
        vggt_payload,
        ffs_payload,
        thresholds=GeometryThresholds(
            min_alignment_pixels=20,
            min_photometric_samples=20,
            min_photometric_valid_fraction=0.9,
            max_photometric_median_absolute_rgb=0.1,
        ),
    )
    assert not bool(tensors["temporal_pose_valid"].item())
    assert torch.count_nonzero(
        tensors["vggt_extrinsics_camera_from_world_metric_temporal"]
    ) == 0
    assert bool(tensors["static_prior_valid"].item())
    assert torch.count_nonzero(
        tensors["vggt_disparity_current_left_aligned_hr_px"]
    ) > 0
    assert not metadata["pose_quality"]["pose_valid"]
    assert "photometric_residual_exceeds_threshold" in str(
        metadata["pose_quality"]["failure_reasons"]
    )


def test_derived_cache_exposes_temporal_pose_only_after_all_gates(
    tmp_path: Path,
) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=64, current_value=64
    )
    tensors, metadata = derive_geometry(
        vggt_payload,
        ffs_payload,
        thresholds=GeometryThresholds(
            min_alignment_pixels=20,
            min_photometric_samples=20,
            min_photometric_valid_fraction=0.9,
        ),
    )
    assert bool(tensors["temporal_pose_valid"].item())
    assert torch.count_nonzero(
        tensors["vggt_extrinsics_camera_from_world_metric_temporal"]
    ) > 0
    assert bool(tensors["static_prior_valid"].item())
    assert metadata["pose_quality"]["pose_valid"]


def test_missing_required_diagnostic_is_never_an_implicit_pass() -> None:
    estimate = estimate_baseline_metric_scale(_five_pair_extrinsics(), 0.1)
    missing_photo = PhotometricReprojectionDiagnostic(
        available=False,
        valid=False,
        depth_sample_count=0,
        projected_sample_count=0,
        valid_fraction=None,
        median_absolute_rgb_residual=None,
        failure_reasons=("missing_photometric_inputs",),
    )
    good_depth = DepthDisparityConsistency(
        available=True,
        valid=True,
        sample_count=100,
        weighted_mae_hr_px=0.1,
        median_absolute_error_hr_px=0.1,
        median_relative_error=0.01,
        failure_reasons=(),
    )
    combined = combine_pose_quality(estimate, missing_photo, good_depth)
    assert not combined.pose_valid
    assert "photometric:missing_or_insufficient" in combined.failure_reasons


def test_near_zero_relative_error_is_diagnostic_unless_explicitly_gated() -> None:
    target = torch.full((64,), 0.02)
    prediction = torch.full((64,), 0.12)
    trusted = torch.ones(64, dtype=torch.bool)
    default = depth_disparity_consistency(
        target,
        prediction,
        trusted_mask=trusted,
        min_samples=32,
        max_weighted_mae_hr_px=0.2,
        max_median_absolute_error_hr_px=0.2,
    )
    assert default.valid
    assert default.median_relative_error == pytest.approx(5.0)
    assert not default.relative_error_gate_enabled

    strict = depth_disparity_consistency(
        target,
        prediction,
        trusted_mask=trusted,
        min_samples=32,
        max_weighted_mae_hr_px=0.2,
        max_median_absolute_error_hr_px=0.2,
        max_median_relative_error=0.1,
    )
    assert not strict.valid
    assert strict.relative_error_gate_enabled
    assert "depth_median_relative_error_exceeds_threshold" in strict.failure_reasons


def test_raw_cache_source_mismatch_is_rejected(tmp_path: Path) -> None:
    vggt_payload, ffs_payload = _raw_payloads(
        tmp_path, previous_value=64, current_value=64
    )
    validate_raw_cache_pair(vggt_payload, ffs_payload)
    vggt_payload["metadata"]["source"]["target_frame_id"] = 999
    with pytest.raises(CacheMismatchError, match="source mismatch"):
        validate_raw_cache_pair(vggt_payload, ffs_payload)
