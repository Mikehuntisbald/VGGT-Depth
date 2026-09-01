"""Strict Spring v2 stereo dataset adapters.

Spring RGB images live on a 1920x1080 grid while disparity ground truth is
stored at 3840x2160.  The ground-truth *values* are nevertheless expressed in
1920x1080 image pixels.  Therefore image-grid supervision uses ``[::2, ::2]``
without multiplying or dividing disparity values.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .cache_dataset import sha256_file
from .manifest import ManifestRecord


SPRING_BASELINE_M = 0.065
SPRING_IMAGE_SIZE_WH = (1920, 1080)
SPRING_GT_SCALE = 2
SPRING_GT_COMPONENT = "spring-ground-truth"
SPRING_GT_TARGET_TYPE = "spring_v2_disp1_ground_truth"
SPRING_FLOW_LIBRARY_COMMIT = "8454aed75172b230304ea9942b95626b99106534"
SPRING_INTRINSICS_FORMAT = "spring_intrinsics_fx_fy_cx_cy_v1"

_FRAME_PATTERN = re.compile(r"^frame_(left|right)_(\d{4})\.png$")
_DISPARITY_PATTERN = re.compile(r"^disp1_left_(\d{4})\.dsp5$")


class SpringDatasetError(ValueError):
    """Raised when a Spring layout or numeric convention is malformed."""


@dataclass(frozen=True, slots=True)
class SpringDisparity:
    """Image-grid disparity and validity in Full-HD pixel units."""

    disparity_hr_px: np.ndarray
    valid_mask: np.ndarray
    source_size_hw: tuple[int, int]
    target_size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SpringSequenceSummary:
    """Audited coverage for one Spring sequence."""

    split: str
    sequence: str
    frames: int
    first_frame_id: int
    last_frame_id: int
    image_size_wh: tuple[int, int]
    intrinsics_path: Path
    has_ground_truth: bool


def read_spring_intrinsics(path: str | Path) -> np.ndarray:
    """Return per-frame ``[fx,fy,cx,cy]`` intrinsics as finite float64."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Spring intrinsics file is missing: {source}")
    try:
        values = np.loadtxt(source, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise SpringDatasetError(f"cannot parse Spring intrinsics: {source}") from exc
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise SpringDatasetError(
            f"Spring intrinsics must have shape [N,4], got {values.shape}: {source}"
        )
    if not np.isfinite(values).all():
        raise SpringDatasetError(f"Spring intrinsics contain NaN/Inf: {source}")
    if bool((values[:, :2] <= 0).any()):
        raise SpringDatasetError(f"Spring focal lengths must be positive: {source}")
    return np.ascontiguousarray(values)


def spring_intrinsics_matrix(row: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Convert one official ``[fx,fy,cx,cy]`` row to a pinhole matrix."""

    values = np.asarray(row, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise SpringDatasetError("Spring intrinsics row must contain four finite values")
    fx, fy, cx, cy = (float(value) for value in values)
    if fx <= 0 or fy <= 0:
        raise SpringDatasetError("Spring focal lengths must be positive")
    return ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0))


def read_spring_disparity(
    path: str | Path,
    *,
    image_size_hw: tuple[int, int] = (
        SPRING_IMAGE_SIZE_WH[1],
        SPRING_IMAGE_SIZE_WH[0],
    ),
) -> SpringDisparity:
    """Read ``disp1_left`` and sample it onto the RGB image grid.

    Stored Spring disparity is a positive magnitude.  This project uses
    ``d = x_left - x_right`` and therefore consumes that stored magnitude
    directly.  Zero-valued sky, NaN, and infinity are invalid.
    """

    source = Path(path).expanduser().resolve()
    if source.suffix != ".dsp5" or not source.is_file():
        raise FileNotFoundError(f"Spring .dsp5 file is missing: {source}")
    try:
        with h5py.File(source, "r") as handle:
            if "disparity" not in handle:
                raise SpringDatasetError(
                    f"Spring dsp5 lacks the 'disparity' dataset: {source}"
                )
            value = np.asarray(handle["disparity"][()])
    except OSError as exc:
        raise SpringDatasetError(f"cannot read Spring dsp5: {source}") from exc
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.floating):
        raise SpringDatasetError(
            f"Spring disparity must be a floating [H,W] array, got "
            f"shape={value.shape} dtype={value.dtype}: {source}"
        )
    target_height, target_width = image_size_hw
    if target_height <= 0 or target_width <= 0:
        raise SpringDatasetError("image_size_hw must be positive")
    source_size = (int(value.shape[0]), int(value.shape[1]))
    if source_size == (target_height, target_width):
        sampled = value
    elif source_size == (
        SPRING_GT_SCALE * target_height,
        SPRING_GT_SCALE * target_width,
    ):
        sampled = value[::SPRING_GT_SCALE, ::SPRING_GT_SCALE]
    else:
        raise SpringDatasetError(
            "Spring disparity/image shape mismatch: "
            f"GT={source_size}, image={(target_height, target_width)}"
        )
    sampled = np.asarray(sampled, dtype=np.float32)
    valid = np.isfinite(sampled) & (sampled > 0.0)
    disparity = np.where(valid, sampled, 0.0).astype(np.float32, copy=False)
    return SpringDisparity(
        disparity_hr_px=np.ascontiguousarray(disparity),
        valid_mask=np.ascontiguousarray(valid),
        source_size_hw=source_size,
        target_size_hw=(target_height, target_width),
    )


def _indexed_files(directory: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Spring directory is missing: {directory}")
    result: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        frame_id = int(match.group(match.lastindex or 1))
        if frame_id in result:
            raise SpringDatasetError(f"duplicate Spring frame {frame_id}: {directory}")
        result[frame_id] = path.resolve()
    if not result:
        raise SpringDatasetError(f"Spring directory contains no expected files: {directory}")
    return result


def _frame_files(directory: Path, camera: str) -> dict[int, Path]:
    pattern = re.compile(rf"^frame_{re.escape(camera)}_(\d{{4}})\.png$")
    return _indexed_files(directory, pattern)


def discover_spring_sequences(dataset_root: str | Path, split: str) -> tuple[str, ...]:
    """List numeric Spring sequence directories for ``train`` or ``test``."""

    if split not in {"train", "test"}:
        raise SpringDatasetError("Spring split must be 'train' or 'test'")
    root = Path(dataset_root).expanduser().resolve() / split
    if not root.is_dir():
        raise FileNotFoundError(f"Spring split directory is missing: {root}")
    sequences = tuple(
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and re.fullmatch(r"\d{4}", path.name)
    )
    if not sequences:
        raise SpringDatasetError(f"Spring split contains no sequence directories: {root}")
    return sequences


def build_spring_manifest_records(
    dataset_root: str | Path,
    *,
    split: str,
    include_sequences: Iterable[str] = (),
    exclude_sequences: Iterable[str] = (),
    validate_all_image_sizes: bool = True,
) -> tuple[list[ManifestRecord], tuple[SpringSequenceSummary, ...]]:
    """Scan Spring into left-reference rectified stereo manifest records."""

    root = Path(dataset_root).expanduser().resolve()
    available = discover_spring_sequences(root, split)
    included = set(include_sequences)
    excluded = set(exclude_sequences)
    if included & excluded:
        raise SpringDatasetError(
            f"Spring include/exclude sets overlap: {sorted(included & excluded)}"
        )
    unknown = (included | excluded) - set(available)
    if unknown:
        raise SpringDatasetError(f"unknown Spring sequences: {sorted(unknown)}")
    selected = tuple(
        sequence
        for sequence in available
        if (not included or sequence in included) and sequence not in excluded
    )
    if not selected:
        raise SpringDatasetError("Spring sequence selection is empty")

    records: list[ManifestRecord] = []
    summaries: list[SpringSequenceSummary] = []
    expected_size = SPRING_IMAGE_SIZE_WH
    for sequence in selected:
        sequence_root = root / split / sequence
        left = _frame_files(sequence_root / "frame_left", "left")
        right = _frame_files(sequence_root / "frame_right", "right")
        if set(left) != set(right):
            raise SpringDatasetError(
                f"Spring left/right frame coverage differs for {split}/{sequence}"
            )
        frame_ids = tuple(sorted(left))
        if frame_ids != tuple(range(1, len(frame_ids) + 1)):
            raise SpringDatasetError(
                f"Spring frames must be contiguous and 1-based: {split}/{sequence}"
            )
        intrinsics_path = (sequence_root / "cam_data" / "intrinsics.txt").resolve()
        intrinsics = read_spring_intrinsics(intrinsics_path)
        if intrinsics.shape[0] != len(frame_ids):
            raise SpringDatasetError(
                f"Spring intrinsics/frame count mismatch for {split}/{sequence}: "
                f"{intrinsics.shape[0]} vs {len(frame_ids)}"
            )
        disparities: dict[int, Path] = {}
        if split == "train":
            disparities = _indexed_files(
                sequence_root / "disp1_left", _DISPARITY_PATTERN
            )
            if set(disparities) != set(frame_ids):
                raise SpringDatasetError(
                    f"Spring disparity/frame coverage differs for {split}/{sequence}"
                )

        metadata_sha256 = sha256_file(intrinsics_path)
        image_paths = [
            path for frame_id in frame_ids for path in (left[frame_id], right[frame_id])
        ]
        paths_to_check = image_paths if validate_all_image_sizes else image_paths[:2]
        for image_path in paths_to_check:
            with Image.open(image_path) as image:
                if image.size != expected_size:
                    raise SpringDatasetError(
                        f"Spring image size mismatch {image.size}: {image_path}"
                    )
        for row_index, frame_id in enumerate(frame_ids):
            k = spring_intrinsics_matrix(intrinsics[row_index])
            k_array = np.asarray(k, dtype=np.float64)
            p_left = np.concatenate((k_array, np.zeros((3, 1))), axis=1)
            p_right = p_left.copy()
            p_right[0, 3] = -k_array[0, 0] * SPRING_BASELINE_M
            records.append(
                ManifestRecord(
                    sequence_id=f"spring_{split}_{sequence}",
                    frame_id=frame_id,
                    timestamp=float(frame_id - 1),
                    left_path=str(left[frame_id]),
                    right_path=str(right[frame_id]),
                    K=k,
                    baseline_m=SPRING_BASELINE_M,
                    gt_disparity_path=(
                        str(disparities[frame_id]) if split == "train" else None
                    ),
                    rectified=True,
                    extras={
                        "dataset": "Spring",
                        "dataset_version": "2.0",
                        "dataset_split": split,
                        "spring_sequence": sequence,
                        "spring_frame_id": frame_id,
                        "timestamp_unit": "frame_index",
                        "image_size_wh": list(expected_size),
                        "ground_truth_size_wh": (
                            [
                                SPRING_GT_SCALE * expected_size[0],
                                SPRING_GT_SCALE * expected_size[1],
                            ]
                            if split == "train"
                            else None
                        ),
                        "ground_truth_disparity_unit": "Full-HD image pixels",
                        "ground_truth_image_grid_sampling": "dsp5[::2,::2]",
                        "K_right": [list(row) for row in k],
                        "P_left": p_left.tolist(),
                        "P_right": p_right.tolist(),
                        "baseline_from_projection_m": SPRING_BASELINE_M,
                        "metadata_path": str(intrinsics_path),
                        "metadata_sha256": metadata_sha256,
                        "calibration_metadata_format": SPRING_INTRINSICS_FORMAT,
                        "calibration_metadata_row": row_index,
                    },
                )
            )
        summaries.append(
            SpringSequenceSummary(
                split=split,
                sequence=sequence,
                frames=len(frame_ids),
                first_frame_id=frame_ids[0],
                last_frame_id=frame_ids[-1],
                image_size_wh=expected_size,
                intrinsics_path=intrinsics_path,
                has_ground_truth=split == "train",
            )
        )
    return records, tuple(summaries)


__all__ = [
    "SPRING_BASELINE_M",
    "SPRING_FLOW_LIBRARY_COMMIT",
    "SPRING_GT_COMPONENT",
    "SPRING_GT_SCALE",
    "SPRING_GT_TARGET_TYPE",
    "SPRING_IMAGE_SIZE_WH",
    "SPRING_INTRINSICS_FORMAT",
    "SpringDatasetError",
    "SpringDisparity",
    "SpringSequenceSummary",
    "build_spring_manifest_records",
    "discover_spring_sequences",
    "read_spring_disparity",
    "read_spring_intrinsics",
    "spring_intrinsics_matrix",
]
