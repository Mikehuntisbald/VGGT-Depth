"""Stage-C stereo inputs layered over the strict causal temporal dataset.

The wrapper never resamples a crop. It loads the endpoint right image and
applies the exact HR crop already selected by :class:`TemporalTrainingSample`,
so left/right/base-disparity pixels remain in one rectified coordinate frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .cache_dataset import CacheMismatchError, sha256_file
from .collate import collate_temporal_training_samples
from .manifest import ManifestRecord
from .temporal_training_dataset import (
    CachedTemporalTrainingDataset,
    TemporalTrainingSample,
)
from geometry.camera import crop_intrinsics
from geometry.epipolar import EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class EpipolarTrainingSample:
    """One causal T=3 sample plus the cropped endpoint right RGB.

    ``temporal`` contains left RGB/geometry in oldest-to-current order.
    ``rgb_right_hr`` is the current rectified right image ``[3,H,W]`` in
    ``[0,1]`` using exactly ``temporal.identity_metadata['crop_hr_px']``.
    """

    temporal: TemporalTrainingSample
    rgb_right_hr: Tensor
    right_path: str
    right_sha256: str
    crop_hr_px: Mapping[str, int]
    K_right_hr: Tensor
    right_intrinsics_source: str
    right_row_scale: float
    right_row_offset_hr_px: float
    right_row_mapping_source: str


def _resolve_source_path(path_text: str, manifest_directory: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"right rectified source image is missing: {path}")
    return path


def _load_cropped_rgb(path: Path, crop: Mapping[str, int]) -> Tensor:
    required = {"x", "y", "width", "height", "spatial_scale"}
    if set(crop) != required:
        raise CacheMismatchError(
            f"temporal crop fields must be exactly {sorted(required)}"
        )
    values = {name: crop[name] for name in required}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise CacheMismatchError("temporal crop values must be integers")
    x_px = values["x"]
    y_px = values["y"]
    width_px = values["width"]
    height_px = values["height"]
    scale = values["spatial_scale"]
    if (
        x_px < 0
        or y_px < 0
        or width_px <= 0
        or height_px <= 0
        or scale <= 0
        or x_px % scale
        or y_px % scale
        or width_px % scale
        or height_px % scale
    ):
        raise CacheMismatchError("temporal crop is not positive and scale-aligned")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        source_width, source_height = rgb.size
        if x_px + width_px > source_width or y_px + height_px > source_height:
            raise CacheMismatchError(
                f"temporal crop exceeds right image {path}: crop={dict(crop)}, "
                f"image={(source_width, source_height)}"
            )
        array = np.asarray(
            rgb.crop((x_px, y_px, x_px + width_px, y_px + height_px)),
            dtype=np.uint8,
        ).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _cropped_right_intrinsics(
    record: ManifestRecord,
    crop: Mapping[str, int],
) -> tuple[Tensor, str]:
    """Return calibrated right intrinsics in the shared HR crop frame."""

    source = "manifest.K_right" if "K_right" in record.extras else "manifest.K"
    value = record.extras.get("K_right", record.K)
    try:
        matrix = np.asarray(value, dtype=np.float64)
        cropped = crop_intrinsics(matrix, crop["x"], crop["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheMismatchError(
            f"invalid right rectified intrinsics for {record.sequence_id}/"
            f"{record.frame_id}: {exc}"
        ) from exc
    return torch.as_tensor(cropped, dtype=torch.float32), source


class EpipolarTrainingDataset(Dataset[EpipolarTrainingSample]):
    """Attach exact-crop endpoint right RGB to a validated temporal dataset."""

    def __init__(self, temporal_dataset: CachedTemporalTrainingDataset) -> None:
        if not isinstance(temporal_dataset, CachedTemporalTrainingDataset):
            raise TypeError("temporal_dataset must be CachedTemporalTrainingDataset")
        if temporal_dataset.sequence_length != 3:
            raise ValueError("Stage C requires a causal T=3 temporal dataset")
        self.temporal_dataset = temporal_dataset

    def __len__(self) -> int:
        return len(self.temporal_dataset)

    @property
    def cache_lineage_summary(self) -> Mapping[str, Any]:
        return self.temporal_dataset.cache_lineage_summary

    def set_epoch(self, epoch: int) -> None:
        self.temporal_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> EpipolarTrainingSample:
        temporal = self.temporal_dataset[index]
        endpoint_manifest_index = temporal.manifest_indices[-1]
        record = self.temporal_dataset.records[endpoint_manifest_index]
        if (
            record.sequence_id != temporal.sequence_id
            or record.frame_id != temporal.frame_ids[-1]
            or record.timestamp != temporal.timestamps[-1]
        ):
            raise CacheMismatchError(
                "temporal endpoint metadata does not match its manifest record"
            )
        right_path = _resolve_source_path(
            record.right_path,
            self.temporal_dataset.spatial_dataset.manifest_directory,
        )
        expected_source = temporal.identity_metadata["per_time_ffs"][-1].get(
            "source_sha256"
        )
        expected_right_sha256 = (
            expected_source.get("right")
            if isinstance(expected_source, Mapping)
            else None
        )
        actual_right_sha256 = sha256_file(right_path)
        if expected_right_sha256 != actual_right_sha256:
            raise CacheMismatchError(
                "right source SHA-256 differs from the endpoint FFS lineage"
            )
        crop = temporal.identity_metadata.get("crop_hr_px")
        if not isinstance(crop, Mapping):
            raise CacheMismatchError("temporal sample crop metadata is missing")
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            for name, value in crop.items()
        ):
            raise CacheMismatchError("temporal crop metadata must contain integers")
        crop_dict = dict(crop)
        rgb_right_hr = _load_cropped_rgb(right_path, crop_dict)
        K_right_hr, right_intrinsics_source = _cropped_right_intrinsics(
            record, crop_dict
        )
        expected_shape = temporal.rgb_hr_sequence[-1].shape
        if rgb_right_hr.shape != expected_shape:
            raise CacheMismatchError(
                f"right crop shape {tuple(rgb_right_hr.shape)} does not match "
                f"endpoint left RGB {tuple(expected_shape)}"
            )
        return EpipolarTrainingSample(
            temporal=temporal,
            rgb_right_hr=rgb_right_hr,
            right_path=str(right_path),
            right_sha256=actual_right_sha256,
            crop_hr_px=crop_dict,
            K_right_hr=K_right_hr,
            right_intrinsics_source=right_intrinsics_source,
            # The saved rectified JPEG coordinate contract is audited from
            # actual correspondences. K_right remains diagnostic because its
            # cy is inconsistent with the stored pixel rows in this dataset.
            right_row_scale=1.0,
            right_row_offset_hr_px=0.0,
            right_row_mapping_source=EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION,
        )


def collate_epipolar_training_samples(
    samples: Sequence[EpipolarTrainingSample],
) -> dict[str, Any]:
    """Collate temporal tensors and endpoint right RGB with provenance."""

    if not samples:
        raise ValueError("cannot collate an empty epipolar batch")
    if not all(isinstance(sample, EpipolarTrainingSample) for sample in samples):
        raise TypeError("all samples must be EpipolarTrainingSample")
    batch = collate_temporal_training_samples(
        [sample.temporal for sample in samples]
    )
    try:
        batch["rgb_right_hr"] = torch.stack(
            [sample.rgb_right_hr for sample in samples]
        )
    except RuntimeError as exc:
        raise ValueError(f"right HR crops cannot be stacked: {exc}") from exc
    batch["right_path"] = [sample.right_path for sample in samples]
    batch["right_sha256"] = [sample.right_sha256 for sample in samples]
    batch["epipolar_crop_hr_px"] = [dict(sample.crop_hr_px) for sample in samples]
    batch["K_right_hr"] = torch.stack([sample.K_right_hr for sample in samples])
    batch["right_intrinsics_source"] = [
        sample.right_intrinsics_source for sample in samples
    ]
    batch["epipolar_right_row_scale"] = torch.tensor(
        [sample.right_row_scale for sample in samples], dtype=torch.float32
    )
    batch["epipolar_right_row_offset_hr_px"] = torch.tensor(
        [sample.right_row_offset_hr_px for sample in samples], dtype=torch.float32
    )
    batch["epipolar_right_row_mapping_source"] = [
        sample.right_row_mapping_source for sample in samples
    ]
    return batch


__all__ = [
    "EpipolarTrainingDataset",
    "EpipolarTrainingSample",
    "EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION",
    "collate_epipolar_training_samples",
]
