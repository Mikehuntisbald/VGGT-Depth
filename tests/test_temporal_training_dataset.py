from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    save_cache_record,
    sha256_file,
)
from data.collate import collate_temporal_training_samples
from data.manifest import ManifestRecord, write_manifest
from data.temporal_training_dataset import CachedTemporalTrainingDataset
from data.training_dataset import cache_path_for_record


def _identity(component: str, suffix: str = "a") -> CacheIdentity:
    return CacheIdentity(
        component=component,
        upstream_commit=suffix * 40,
        checkpoint_sha256=suffix * 64,
        torch_version="2.10.0+cu128",
        cuda_version="12.8",
        config_sha256=(suffix + "0") * 32,
    )


def _write_rgb(path: Path, height: int, width: int, offset: int) -> None:
    y, x = np.mgrid[:height, :width]
    array = np.stack(
        (
            (x + offset) % 256,
            (3 * y + offset) % 256,
            (x + y + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _make_temporal_cache(
    tmp_path: Path,
    *,
    include_teacher: bool = True,
    invalid_pose_is_nonzero: bool = False,
) -> tuple[
    Path,
    Path,
    Path | None,
    Path,
    dict[tuple[str, int], CacheIdentity],
]:
    height, width = 8, 12
    sequence_id = "sequence-A"
    records: list[ManifestRecord] = []
    for index in range(7):
        left = tmp_path / "images" / f"left-{index}.png"
        right = tmp_path / "images" / f"right-{index}.png"
        _write_rgb(left, height, width, offset=index)
        _write_rgb(right, height, width, offset=index + 20)
        records.append(
            ManifestRecord(
                sequence_id=sequence_id,
                frame_id=index,
                timestamp=0.2 * index,
                left_path=str(left),
                right_path=str(right),
                K=(
                    (100.0, 0.0, width / 2),
                    (0.0, 101.0, height / 2),
                    (0.0, 0.0, 1.0),
                ),
                baseline_m=0.12,
                gt_disparity_path=None,
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)
    observation_root = tmp_path / "cache" / "observation"
    teacher_root = tmp_path / "cache" / "teacher" if include_teacher else None
    observation_identity = _identity("ffs-observation", "a")
    teacher_identity = _identity("ffs-teacher", "b")
    height_lr, width_lr = height // 2, width // 2
    for index, record in enumerate(records):
        left = Path(record.left_path)
        right = Path(record.right_path)
        source = {
            "manifest_record": record.to_dict(),
            "left_sha256": sha256_file(left),
            "right_sha256": sha256_file(right),
        }
        disparity_lr_px = torch.full(
            (1, 1, height_lr, width_lr), 2.0 + index, dtype=torch.float32
        )
        observation_path = cache_path_for_record(observation_root, record)
        save_cache_record(
            observation_path,
            tensors={
                "observation_disparity_lr_px": disparity_lr_px,
                "observation_disparity_hr_px": 2.0 * disparity_lr_px,
                "observation_confidence": torch.full_like(disparity_lr_px, 0.9),
                "observation_valid_mask": torch.ones_like(
                    disparity_lr_px, dtype=torch.bool
                ),
                "observation_trusted_mask": torch.ones_like(
                    disparity_lr_px, dtype=torch.bool
                ),
            },
            metadata={"source": source},
            identity=observation_identity,
        )
        if teacher_root is not None:
            teacher = torch.full(
                (1, 1, height, width), 4.0 + 2.0 * index, dtype=torch.float32
            )
            save_cache_record(
                cache_path_for_record(teacher_root, record),
                tensors={
                    "teacher_disparity_hr_px": teacher,
                    "teacher_confidence": torch.full_like(teacher, 0.95),
                    "teacher_valid_mask": torch.ones_like(teacher, dtype=torch.bool),
                    "teacher_trusted_mask": torch.ones_like(
                        teacher, dtype=torch.bool
                    ),
                },
                metadata={"source": source},
                identity=teacher_identity,
            )

    derived_root = tmp_path / "cache" / "derived"
    derived_identity = _identity("vggt-ffs-derived-geometry", "c")
    derived_identities: dict[tuple[str, int], CacheIdentity] = {}
    rows: list[dict[str, object]] = []
    # Derived geometry starts at original index four.  Requiring all T=3
    # records therefore leaves exactly the endpoint-six window.
    validity = {
        4: (True, True),
        5: (False, True),  # photometric pose failure; static prior survives
        6: (False, False),
    }
    for selection_index, manifest_index in enumerate((4, 5, 6)):
        record = records[manifest_index]
        pose_valid, static_prior_valid = validity[manifest_index]
        if static_prior_valid:
            prior = torch.full(
                (1, height_lr, width_lr),
                20.0 + manifest_index,
                dtype=torch.float32,
            )
            prior_confidence = torch.full_like(prior, 0.7)
            prior_valid = torch.ones_like(prior, dtype=torch.bool)
        else:
            prior = torch.zeros((1, height_lr, width_lr), dtype=torch.float32)
            prior_confidence = torch.zeros_like(prior)
            prior_valid = torch.zeros_like(prior, dtype=torch.bool)
        extrinsics = torch.zeros(10, 3, 4, dtype=torch.float32)
        if pose_valid or invalid_pose_is_nonzero and manifest_index == 5:
            extrinsics[:, :3, :3] = torch.eye(3)
            extrinsics[:, 0, 3] = torch.arange(10, dtype=torch.float32) * -0.01
        observation_path = cache_path_for_record(observation_root, record)
        derived_path = cache_path_for_record(derived_root, record)
        save_cache_record(
            derived_path,
            tensors={
                "vggt_extrinsics_camera_from_world_metric_diagnostic_only": (
                    extrinsics.clone()
                ),
                "vggt_extrinsics_camera_from_world_metric_temporal": extrinsics,
                "vggt_depth_current_left_metric_m": torch.ones_like(prior),
                "vggt_disparity_current_left_aligned_hr_px": prior,
                "vggt_aligned_confidence": prior_confidence,
                "vggt_depth_metric_valid_mask": torch.ones_like(
                    prior, dtype=torch.bool
                ),
                "vggt_aligned_valid_mask": prior_valid,
                "ffs_trusted_mask": torch.ones_like(prior, dtype=torch.bool),
                "temporal_pose_valid": torch.tensor(pose_valid),
                "static_prior_valid": torch.tensor(static_prior_valid),
            },
            metadata={
                "config": {
                    "schema_version": 1,
                    "algorithm": (
                        "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
                    ),
                    "extrinsics_convention": "camera-from-world",
                    "previous_left_view_index": 6,
                    "current_left_view_index": 8,
                    "invalid_temporal_pose_policy": (
                        "zero-filled with false validity tensor"
                    ),
                },
                "source": {
                    "ffs_cache_path": str(observation_path.resolve()),
                    "ffs_cache_sha256": sha256_file(observation_path),
                    "linkage": {
                        "target_sequence_id": record.sequence_id,
                        "target_frame_id": record.frame_id,
                        "target_timestamp": record.timestamp,
                        "target_manifest_record": record.to_dict(),
                    },
                },
                "target": {
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "timestamp": record.timestamp,
                },
                "pose_quality": {
                    "pose_valid": pose_valid,
                    "alignment": {"static_prior_valid": static_prior_valid},
                },
            },
            identity=derived_identity,
        )
        rows.append(
            {
                "selection_index": selection_index,
                "target_manifest_index": manifest_index,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "cache_path": str(derived_path.resolve()),
                "cache_sha256": sha256_file(derived_path),
            }
        )
        derived_identities[(record.sequence_id, record.frame_id)] = derived_identity
    derived_manifest = derived_root / "cache_manifest.jsonl"
    derived_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "component": "vggt-ffs-derived-geometry-batch",
        "config": {
            "algorithm": (
                "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
            ),
            "extrinsics_convention": "camera-from-world",
            "previous_left_view_index": 6,
            "current_left_view_index": 8,
            "invalid_temporal_pose_policy": (
                "zero-filled with false validity tensor"
            ),
        },
        "counts": {
            "selected": 3,
            "written": 3,
            "reused": 0,
            "pose_valid": 1,
            "pose_rejected": 2,
            "static_prior_valid": 2,
            "static_prior_rejected": 1,
        },
        "selection": {"selected_windows": 3},
        "output": {"cache_manifest_sha256": sha256_file(derived_manifest)},
    }
    (derived_root / "run_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, observation_root, teacher_root, derived_root, derived_identities


def test_t3_window_is_causal_filtered_and_uses_one_crop(tmp_path: Path) -> None:
    manifest, observation, teacher, derived, identities = _make_temporal_cache(
        tmp_path
    )
    dataset = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        derived_identities=identities,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
        fixed_crop_origin_hr_xy=(2, 2),
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.sequence_id == "sequence-A"
    assert sample.frame_ids == (4, 5, 6)
    assert sample.manifest_indices == (4, 5, 6)
    assert sample.timestamps == pytest.approx((0.8, 1.0, 1.2))
    assert all(timestamp <= sample.timestamps[-1] for timestamp in sample.timestamps)
    assert sample.rgb_hr_sequence.shape == (3, 3, 4, 8)
    assert sample.observation_disparity_hr_px_sequence.shape == (3, 1, 2, 4)
    assert sample.teacher_disparity_hr_px_sequence is not None
    assert sample.teacher_disparity_hr_px_sequence.shape == (3, 1, 4, 8)
    assert sample.vggt_disparity_hr_px_sequence.shape == (3, 1, 2, 4)
    assert sample.vggt_extrinsics_camera_from_world_metric_sequence.shape == (
        3,
        10,
        3,
        4,
    )
    assert sample.temporal_pose_valid_sequence.tolist() == [True, False, False]
    assert sample.static_prior_valid_sequence.tolist() == [True, True, False]
    assert torch.count_nonzero(
        sample.vggt_extrinsics_camera_from_world_metric_sequence[1:]
    ) == 0
    assert torch.count_nonzero(sample.vggt_valid_mask_sequence[-1]) == 0
    torch.testing.assert_close(
        sample.observation_disparity_hr_px_sequence,
        2.0 * sample.observation_disparity_lr_px_sequence,
    )
    torch.testing.assert_close(
        sample.K_hr_sequence,
        torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 101.0, 2.0], [0.0, 0.0, 1.0]]]
        ).expand(3, -1, -1),
    )
    crop = sample.identity_metadata["crop_hr_px"]
    assert crop == {"x": 2, "y": 2, "width": 8, "height": 4, "spatial_scale": 2}
    assert all(
        metadata["crop_hr_px"] == crop
        for metadata in sample.identity_metadata["per_time_ffs"]
    )


def test_random_crop_and_epoch_are_deterministic_for_whole_window(
    tmp_path: Path,
) -> None:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(tmp_path)
    kwargs = dict(
        manifest_path=manifest,
        observation_cache_root=observation,
        teacher_cache_root=teacher,
        derived_cache_root=derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="random",
        seed=42,
    )
    first = CachedTemporalTrainingDataset(**kwargs)
    second = CachedTemporalTrainingDataset(**kwargs)
    crop0 = first[0].identity_metadata["crop_hr_px"]
    assert crop0 == second[0].identity_metadata["crop_hr_px"]
    first.set_epoch(3)
    crop3 = first[0].identity_metadata["crop_hr_px"]
    assert crop3 == first[0].identity_metadata["crop_hr_px"]
    assert crop3 != crop0
    assert crop3["x"] % 2 == 0 and crop3["y"] % 2 == 0


def test_temporal_collate_retains_time_and_source_aliases(tmp_path: Path) -> None:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(tmp_path)
    dataset = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    sample = dataset[0]
    batch = collate_temporal_training_samples([sample, sample])

    assert batch["rgb_hr_sequence"].shape == (2, 3, 3, 4, 8)
    assert batch["observation_disparity_hr_px_sequence"].shape == (2, 3, 1, 2, 4)
    assert batch["vggt_extrinsics_camera_from_world_metric_sequence"].shape == (
        2,
        3,
        10,
        3,
        4,
    )
    assert batch["frame_ids"].shape == (2, 3)
    assert batch["timestamps"].shape == (2, 3)
    assert batch["disparity_ffs_hr_px_sequence"] is batch[
        "observation_disparity_hr_px_sequence"
    ]
    assert batch["history_pose_valid_sequence"] is batch[
        "temporal_pose_valid_sequence"
    ]
    assert batch["target_disparity_hr_px_sequence"] is batch[
        "teacher_disparity_hr_px_sequence"
    ]


def test_teacher_can_be_absent_for_entire_temporal_batch(tmp_path: Path) -> None:
    manifest, observation, _, derived, _ = _make_temporal_cache(
        tmp_path, include_teacher=False
    )
    dataset = CachedTemporalTrainingDataset(
        manifest,
        observation,
        None,
        derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    batch = collate_temporal_training_samples([dataset[0]])
    assert batch["teacher_disparity_hr_px_sequence"] is None
    assert batch["target_disparity_hr_px_sequence"] is None


def test_expected_per_record_derived_identity_is_enforced(tmp_path: Path) -> None:
    manifest, observation, teacher, derived, identities = _make_temporal_cache(
        tmp_path
    )
    identities[("sequence-A", 5)] = _identity(
        "vggt-ffs-derived-geometry", "d"
    )
    dataset = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        derived_identities=identities,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    with pytest.raises(CacheMismatchError, match="cache identity mismatch"):
        _ = dataset[0]


def test_invalid_pose_with_nonzero_temporal_extrinsics_is_rejected(
    tmp_path: Path,
) -> None:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(
        tmp_path, invalid_pose_is_nonzero=True
    )
    dataset = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    with pytest.raises(CacheMismatchError, match="zero-filled"):
        _ = dataset[0]


def test_batch_receipt_must_bind_manifest_hash_and_coverage(tmp_path: Path) -> None:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(tmp_path)
    receipt_path = derived / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["cache_manifest_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(CacheMismatchError, match="receipt/manifest SHA-256"):
        CachedTemporalTrainingDataset(
            manifest,
            observation,
            teacher,
            derived,
            crop_size_hr_hw=(4, 8),
            crop_mode="fixed",
        )
