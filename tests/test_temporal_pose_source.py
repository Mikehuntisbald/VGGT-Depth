from __future__ import annotations

import pytest
import torch

import train
from data.manifest import ManifestRecord
from data.temporal_training_dataset import CachedTemporalTrainingDataset


def _pose_batch() -> dict[str, torch.Tensor]:
    gt = torch.zeros(1, 3, 10, 3, 4, dtype=torch.float32)
    vggt = torch.zeros_like(gt)
    gt[..., :3, :3] = torch.eye(3)
    vggt[..., :3, :3] = 2.0 * torch.eye(3)
    return {
        "gt_pose_sequence": gt,
        "vggt_extrinsics_camera_from_world_metric_sequence": vggt,
        "temporal_pose_valid_sequence": torch.tensor([[False, True, True]]),
        "temporal_pose_quality_score_sequence": torch.tensor([[0.0, 0.5, 0.75]]),
    }


def test_gt_pose_source_selects_manifest_tensor_and_trusts_valid_pose() -> None:
    batch = _pose_batch()
    config = {
        "data": {"temporal_pose_source": "gt"},
        "model": {},
    }
    poses, valid, quality = train.temporal_pose_inputs_from_batch(batch, config)
    torch.testing.assert_close(poses, batch["gt_pose_sequence"])
    assert valid.tolist() == [[True, True, True]]
    torch.testing.assert_close(quality, torch.ones_like(quality))


def test_vggt_pose_source_preserves_cache_validity_and_quality() -> None:
    batch = _pose_batch()
    config = {"data": {"temporal_pose_source": "vggt"}, "model": {}}
    poses, valid, quality = train.temporal_pose_inputs_from_batch(batch, config)
    torch.testing.assert_close(
        poses, batch["vggt_extrinsics_camera_from_world_metric_sequence"]
    )
    assert valid.tolist() == [[False, True, True]]
    torch.testing.assert_close(
        quality, batch["temporal_pose_quality_score_sequence"]
    )


def test_pose_source_alias_must_agree_and_unknown_values_fail() -> None:
    with pytest.raises(ValueError, match="disagree"):
        train.temporal_pose_source_from_config(
            {
                "data": {"temporal_pose_source": "gt"},
                "model": {"pose_source": "vggt"},
            }
        )
    with pytest.raises(ValueError, match="one of"):
        train.temporal_pose_source_from_config(
            {"data": {"temporal_pose_source": "predicted"}, "model": {}}
        )


def test_missing_gt_tensor_is_rejected_when_gt_source_selected() -> None:
    batch = _pose_batch()
    batch.pop("gt_pose_sequence")
    with pytest.raises(ValueError, match="requires"):
        train.temporal_pose_inputs_from_batch(
            batch, {"data": {"temporal_pose_source": "gt"}, "model": {}}
        )


def test_stage_b_validation_allows_gt_pose_without_vggt_pose_conditioning() -> None:
    config = train.resolve_config(
        "configs/temporal_x2.yaml",
        ["data.temporal_pose_source=gt", "model.use_vggt_pose=false"],
    )
    assert train.validate_stage_b_config(config) is None


def test_temporal_dataset_gt_pose_context_is_causal_and_pads_only_early_history() -> None:
    records = []
    for index in range(5):
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 3] = -float(index)
        records.append(
            ManifestRecord(
                sequence_id="spring",
                frame_id=index + 1,
                timestamp=float(index),
                left_path=f"/tmp/left_{index}.png",
                right_path=f"/tmp/right_{index}.png",
                K=((100.0, 0.0, 2.0), (0.0, 100.0, 2.0), (0.0, 0.0, 1.0)),
                baseline_m=0.065,
                gt_disparity_path=None,
                extras={
                    "dataset": "spring",
                    "gt_extrinsics_camera_from_world": pose.tolist(),
                },
            )
        )
    dataset = object.__new__(CachedTemporalTrainingDataset)
    dataset.records = records
    first_context = dataset._gt_pose_context_for_endpoint(2)
    assert first_context is not None
    assert tuple(first_context.shape) == (10, 3, 4)
    # endpoint frame 3 is view pair index 8; age-1 frame 2 is index 6.
    assert first_context[6, 0, 3].item() == pytest.approx(-1.0)
    assert first_context[8, 0, 3].item() == pytest.approx(-2.0)
    # The unavailable oldest context is duplicated, never taken from future.
    assert first_context[0, 0, 3].item() == pytest.approx(0.0)
