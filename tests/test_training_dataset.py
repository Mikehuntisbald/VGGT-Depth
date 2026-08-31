from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    save_cache_record,
    sha256_file,
)
from data.collate import collate_training_samples
from data.manifest import ManifestRecord, write_manifest
from data.training_dataset import (
    CachedFFSTrainingDataset,
    build_causal_windows,
    cache_path_for_record,
)


def _identity(component: str, config: str = "config-a") -> CacheIdentity:
    return CacheIdentity(
        component=component,
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="2.10.0+cu128",
        cuda_version="12.8",
        config_sha256=config,
    )


def _write_rgb(path: Path, height: int, width: int, offset: int = 0) -> None:
    y, x = np.mgrid[:height, :width]
    array = np.stack(
        (
            (x + offset) % 256,
            (10 * y + offset) % 256,
            (x + y + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _make_cached_example(
    tmp_path: Path,
    *,
    height: int = 8,
    width: int = 12,
    include_teacher: bool = True,
) -> tuple[Path, Path, Path | None, ManifestRecord, torch.Tensor, torch.Tensor]:
    left = tmp_path / "images" / "left.png"
    right = tmp_path / "images" / "right.png"
    _write_rgb(left, height, width)
    _write_rgb(right, height, width, offset=7)
    record = ManifestRecord(
        sequence_id="seq/with spaces",
        frame_id=12,
        timestamp=0.4,
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
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    source = {
        "manifest_record": record.to_dict(),
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
    }

    height_lr, width_lr = height // 2, width // 2
    disparity_lr_px = (
        torch.arange(height_lr * width_lr, dtype=torch.float32)
        .reshape(1, 1, height_lr, width_lr)
        .add_(1.0)
    )
    disparity_hr_px_lr_grid = 2.0 * disparity_lr_px
    observation_root = tmp_path / "cache" / "observation"
    save_cache_record(
        cache_path_for_record(observation_root, record),
        tensors={
            "observation_disparity_lr_px": disparity_lr_px,
            "observation_disparity_hr_px": disparity_hr_px_lr_grid,
            "observation_confidence": torch.full_like(disparity_lr_px, 0.9),
            "observation_valid_mask": torch.ones_like(disparity_lr_px, dtype=torch.bool),
            "observation_trusted_mask": torch.ones_like(disparity_lr_px, dtype=torch.bool),
        },
        metadata={"source": source},
        identity=_identity("ffs-observation"),
    )

    teacher_disparity_hr_px = (
        torch.arange(height * width, dtype=torch.float32)
        .reshape(1, 1, height, width)
        .add_(1.0)
    )
    teacher_root: Path | None = None
    if include_teacher:
        teacher_root = tmp_path / "cache" / "teacher"
        save_cache_record(
            cache_path_for_record(teacher_root, record),
            tensors={
                "teacher_disparity_hr_px": teacher_disparity_hr_px,
                "teacher_confidence": torch.full_like(teacher_disparity_hr_px, 0.95),
                "teacher_valid_mask": torch.ones_like(
                    teacher_disparity_hr_px, dtype=torch.bool
                ),
                "teacher_trusted_mask": torch.ones_like(
                    teacher_disparity_hr_px, dtype=torch.bool
                ),
            },
            metadata={"source": source},
            identity=_identity("ffs-teacher"),
        )
    return (
        manifest,
        observation_root,
        teacher_root,
        record,
        disparity_hr_px_lr_grid,
        teacher_disparity_hr_px,
    )


def test_fixed_crop_keeps_hr_lr_alignment_and_updates_intrinsics(tmp_path: Path) -> None:
    (
        manifest,
        observation_root,
        teacher_root,
        _,
        observation_hr_px,
        teacher_hr_px,
    ) = _make_cached_example(tmp_path)
    dataset = CachedFFSTrainingDataset(
        manifest,
        observation_root,
        teacher_root,
        observation_identity=_identity("ffs-observation"),
        teacher_identity=_identity("ffs-teacher"),
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
        fixed_crop_origin_hr_xy=(2, 2),
        spatial_scale=2,
    )

    sample = dataset[0]

    assert sample.rgb_hr.shape == (3, 4, 8)
    assert sample.rgb_hr.dtype == torch.float32
    assert 0.0 <= sample.rgb_hr.min() <= sample.rgb_hr.max() <= 1.0
    torch.testing.assert_close(
        sample.observation_disparity_hr_px,
        observation_hr_px[0, :, 1:3, 1:5],
    )
    torch.testing.assert_close(
        sample.observation_disparity_lr_px * 2.0,
        sample.observation_disparity_hr_px,
    )
    assert sample.teacher_disparity_hr_px is not None
    torch.testing.assert_close(
        sample.teacher_disparity_hr_px,
        teacher_hr_px[0, :, 2:6, 2:10],
    )
    torch.testing.assert_close(
        sample.K_hr,
        torch.tensor([[100.0, 0.0, 4.0], [0.0, 101.0, 2.0], [0.0, 0.0, 1.0]]),
    )
    assert sample.baseline_m.item() == pytest.approx(0.12)
    assert sample.identity_metadata["crop_hr_px"] == {
        "x": 2,
        "y": 2,
        "width": 8,
        "height": 4,
        "spatial_scale": 2,
    }


def test_random_crop_is_deterministic_by_seed_index_and_epoch(tmp_path: Path) -> None:
    manifest, observation_root, teacher_root, *_ = _make_cached_example(
        tmp_path, height=20, width=30
    )
    kwargs = dict(
        manifest_path=manifest,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        crop_size_hr_hw=(8, 12),
        crop_mode="random",
        seed=42,
    )
    first_dataset = CachedFFSTrainingDataset(**kwargs)
    second_dataset = CachedFFSTrainingDataset(**kwargs)

    first_crop = first_dataset[0].identity_metadata["crop_hr_px"]
    repeated_crop = first_dataset[0].identity_metadata["crop_hr_px"]
    independent_crop = second_dataset[0].identity_metadata["crop_hr_px"]
    assert first_crop == repeated_crop == independent_crop
    assert first_crop["x"] % 2 == 0
    assert first_crop["y"] % 2 == 0

    first_dataset.set_epoch(3)
    epoch_crop = first_dataset[0].identity_metadata["crop_hr_px"]
    assert epoch_crop == first_dataset[0].identity_metadata["crop_hr_px"]
    assert epoch_crop != first_crop


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_epoch_update_reaches_persistent_dataloader_worker(tmp_path: Path) -> None:
    manifest, observation_root, teacher_root, *_ = _make_cached_example(
        tmp_path, height=20, width=30
    )
    dataset = CachedFFSTrainingDataset(
        manifest,
        observation_root,
        teacher_root,
        crop_size_hr_hw=(8, 12),
        crop_mode="random",
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        collate_fn=collate_training_samples,
    )
    first_crop = next(iter(loader))["identity_metadata"][0]["crop_hr_px"]
    dataset.set_epoch(3)
    epoch_crop = next(iter(loader))["identity_metadata"][0]["crop_hr_px"]
    assert epoch_crop != first_crop
    assert epoch_crop["x"] % 2 == 0 and epoch_crop["y"] % 2 == 0
    assert loader._iterator is not None
    loader._iterator._shutdown_workers()


def test_cache_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, observation_root, teacher_root, record, *_ = _make_cached_example(tmp_path)
    left_path = Path(record.left_path)
    _write_rgb(left_path, 8, 12, offset=99)
    # Be explicit even on filesystems with coarse timestamp behavior.
    stat = left_path.stat()
    os.utime(left_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    dataset = CachedFFSTrainingDataset(
        manifest,
        observation_root,
        teacher_root,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )

    with pytest.raises(CacheMismatchError, match="left_sha256"):
        _ = dataset[0]


def test_optional_identity_is_enforced_and_teacher_can_be_absent(tmp_path: Path) -> None:
    manifest, observation_root, _, *_ = _make_cached_example(
        tmp_path, include_teacher=False
    )
    mismatched_dataset = CachedFFSTrainingDataset(
        manifest,
        observation_root,
        None,
        observation_identity=_identity("ffs-observation", config="different"),
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    with pytest.raises(CacheMismatchError, match="config_sha256"):
        _ = mismatched_dataset[0]

    dataset = CachedFFSTrainingDataset(
        manifest,
        observation_root,
        None,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
    )
    sample = dataset[0]
    assert sample.teacher_disparity_hr_px is None
    batch = collate_training_samples([sample, sample])
    assert batch["rgb_hr"].shape == (2, 3, 4, 8)
    assert batch["disparity_ffs_hr_px"] is batch["observation_disparity_hr_px"]
    assert batch["confidence_ffs"] is batch["observation_confidence"]
    assert batch["teacher_disparity_hr_px"] is None
    assert batch["target_disparity_hr_px"] is None
    assert len(batch["identity_metadata"]) == 2


def _record(sequence_id: str, frame_id: int, timestamp: float) -> ManifestRecord:
    return ManifestRecord(
        sequence_id=sequence_id,
        frame_id=frame_id,
        timestamp=timestamp,
        left_path=f"/{sequence_id}/left-{frame_id}.png",
        right_path=f"/{sequence_id}/right-{frame_id}.png",
        K=((100.0, 0.0, 5.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0)),
        baseline_m=0.1,
        gt_disparity_path=None,
    )


def test_causal_windows_never_cross_sequence_or_use_future_frames() -> None:
    records: list[ManifestRecord] = []
    for frame_id in range(6):
        records.append(_record("A", frame_id, frame_id * 0.2))
        if frame_id < 5:
            records.append(_record("B", frame_id, frame_id * 0.2 + 0.01))

    windows = build_causal_windows(records)

    assert len(windows) == 3
    for window in windows:
        assert len(window.student_indices) == 3
        assert len(window.vggt_indices) == 5
        assert window.student_indices[-1] == window.endpoint_index
        assert window.vggt_indices[-1] == window.endpoint_index
        assert all(index <= window.endpoint_index for index in window.student_indices)
        assert all(index <= window.endpoint_index for index in window.vggt_indices)
        assert all(
            records[index].sequence_id == window.sequence_id
            for index in window.student_indices
        )
        assert all(
            records[index].sequence_id == window.sequence_id
            for index in window.vggt_indices
        )
        endpoint_timestamp = records[window.endpoint_index].timestamp
        assert all(
            records[index].timestamp <= endpoint_timestamp
            for index in window.vggt_indices
        )


def test_causal_windows_reject_non_monotonic_sequence_timestamps() -> None:
    records = [_record("A", 0, 0.2), _record("A", 1, 0.2)]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_causal_windows(records)
