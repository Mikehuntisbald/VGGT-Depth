"""Spring stereo dataset adapter.

The public Spring archives use one directory per rendered scene and store
camera metadata in ``cam_data``::

    <root>/<split>/<sequence>/
        frame_left/frame_left_0001.png
        frame_right/frame_right_0001.png
        disp1_left/disp1_left_0001.dsp5
        disp1_right/disp1_right_0001.dsp5
        cam_data/{intrinsics,extrinsics,focaldistance}.txt

``intrinsics.txt`` contains ``fx fy cx cy`` per frame.  ``extrinsics.txt``
contains a row-major homogeneous world-to-camera OpenCV matrix per frame and
is the pose of the left camera (``Camera_L`` in the official Blender helper).
The relative pose therefore follows the project's convention:

``T_target_from_source = E_target @ inverse(E_source)``.

Spring's 4K disparity files encode disparity in full-HD pixel units.  The
adapter keeps that fact explicit in manifest metadata and never silently
divides values by two when a caller requests image-resolution output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np

from .manifest import ManifestRecord, write_manifest


SPRING_BASELINE_M = 0.065
SPRING_IMAGE_SIZE_HW = (1080, 1920)
SPRING_DISPARITY_SIZE_HW = (2160, 3840)
SPRING_POSE_CONVENTION = "world_to_camera_opencv"
SPRING_DISPARITY_CONVENTION = "positive_left_reference_magnitude"


class SpringFormatError(ValueError):
    """Raised when a Spring directory or camera sidecar is malformed."""


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SpringFormatError(f"{name} must be finite, got {value!r}")
    return result


def _read_rows(path: Path, width: int, *, name: str) -> np.ndarray:
    """Read strict whitespace-delimited rows from a Spring sidecar.

    ``numpy.loadtxt`` silently accepts malformed/blank rows and changes shape
    for one-row files.  Explicit parsing gives deterministic failures and
    keeps frame alignment auditable.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Spring {name} file does not exist: {path}")
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            raise SpringFormatError(f"blank row in {name}: {path}:{line_number}")
        tokens = line.split()
        if len(tokens) != width:
            raise SpringFormatError(
                f"{name} row {line_number} must contain {width} values, "
                f"got {len(tokens)}"
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise SpringFormatError(
                f"non-numeric value in {name}: {path}:{line_number}"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise SpringFormatError(
                f"non-finite value in {name}: {path}:{line_number}"
            )
        rows.append(values)
    if not rows:
        raise SpringFormatError(f"Spring {name} file is empty: {path}")
    return np.asarray(rows, dtype=np.float64)


def _as_tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _validate_pose(matrix: np.ndarray, *, path: Path, row_number: int) -> None:
    rotation = matrix[:3, :3]
    # Text serialization introduces tiny round-off, so use a practical but
    # strict tolerance.  A malformed pose must never be propagated to warping.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise SpringFormatError(
            f"extrinsics row {row_number} has a non-orthonormal rotation: {path}"
        )
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-5):
        raise SpringFormatError(
            f"extrinsics row {row_number} rotation determinant is not +1: {path}"
        )
    if not np.allclose(
        matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8, rtol=0.0
    ):
        raise SpringFormatError(
            f"extrinsics row {row_number} has malformed homogeneous row: {path}"
        )


@dataclass(frozen=True, slots=True)
class SpringFrame:
    """One left-reference Spring stereo timepoint with calibrated GT pose."""

    sequence_id: str
    frame_id: int
    K: tuple[tuple[float, float, float], ...]
    focal_distance_m: float
    extrinsics_camera_from_world: tuple[tuple[float, ...], ...]
    left_path: Path
    right_path: Path
    gt_disparity_left_path: Path
    gt_disparity_right_path: Path

    @property
    def timestamp(self) -> float:
        """Monotonic frame index used when Spring has no timestamp sidecar."""

        return float(self.frame_id - 1)

    @property
    def pose(self) -> np.ndarray:
        """Return the left-camera world→camera matrix as float64 [4,4]."""

        return np.asarray(self.extrinsics_camera_from_world, dtype=np.float64)

    @property
    def camera_center_world(self) -> np.ndarray:
        """Return ``C_world = -R.T @ t`` for the left camera."""

        matrix = self.pose
        return -matrix[:3, :3].T @ matrix[:3, 3]

    def right_pose(self, baseline_m: float = SPRING_BASELINE_M) -> np.ndarray:
        """Derive the rectified right-camera pose from the left pose.

        Spring's orthoparallel right camera is translated by ``+baseline`` in
        the left camera's x direction.  In camera-coordinate form this is
        ``X_right = X_left + [-baseline, 0, 0]``.
        """

        baseline = _finite(baseline_m, name="baseline_m")
        if baseline <= 0:
            raise ValueError("baseline_m must be positive")
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = -baseline
        return transform @ self.pose


@dataclass(frozen=True, slots=True)
class SpringSequence:
    """Validated sidecars and frame rows for one Spring sequence."""

    sequence_id: str
    sequence_root: Path
    frames: tuple[SpringFrame, ...]

    def frame(self, frame_id: int) -> SpringFrame:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise TypeError("frame_id must be an integer")
        if frame_id < 1 or frame_id > len(self.frames):
            raise KeyError(f"unknown Spring frame {self.sequence_id}/{frame_id}")
        return self.frames[frame_id - 1]


def resolve_spring_split_root(root: str | Path, split: str = "train") -> Path:
    """Resolve either ``.../spring`` or ``.../spring_dataset`` roots."""

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    base = Path(root).expanduser().resolve()
    candidates = (base / split, base / "spring" / split)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Spring split directory not found; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _sequence_directories(split_root: Path, sequences: Sequence[str] | None) -> list[Path]:
    requested = None if sequences is None else {str(item) for item in sequences}
    discovered = [
        path
        for path in split_root.iterdir()
        if path.is_dir() and (path / "cam_data").is_dir()
    ]
    discovered.sort(key=lambda path: path.name)
    if requested is not None:
        unknown = requested - {path.name for path in discovered}
        if unknown:
            raise SpringFormatError(
                f"requested Spring sequences are missing: {sorted(unknown)}"
            )
        discovered = [path for path in discovered if path.name in requested]
    if not discovered:
        raise SpringFormatError(f"no Spring sequence directories found in {split_root}")
    return discovered


def load_spring_sequence(
    sequence_root: str | Path,
    *,
    require_images: bool = True,
    require_disparity: bool = True,
    baseline_m: float = SPRING_BASELINE_M,
) -> SpringSequence:
    """Parse one sequence's camera sidecars and construct frame paths.

    The file-presence switches are useful while the large image/disparity
    archives are still downloading: camera metadata can be validated and a
    manifest can be staged without pretending missing pixels are available.
    """

    root = Path(sequence_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Spring sequence directory does not exist: {root}")
    sequence_id = root.name
    cam_root = root / "cam_data"
    intrinsics_path = cam_root / "intrinsics.txt"
    extrinsics_path = cam_root / "extrinsics.txt"
    focal_path = cam_root / "focaldistance.txt"
    intrinsics = _read_rows(intrinsics_path, 4, name="intrinsics")
    extrinsics_flat = _read_rows(extrinsics_path, 16, name="extrinsics")
    focal_distances = _read_rows(focal_path, 1, name="focaldistance").reshape(-1)
    row_count = intrinsics.shape[0]
    if extrinsics_flat.shape[0] != row_count or focal_distances.shape[0] != row_count:
        raise SpringFormatError(
            f"camera sidecar row counts disagree for sequence {sequence_id}: "
            f"intrinsics={row_count}, extrinsics={extrinsics_flat.shape[0]}, "
            f"focaldistance={focal_distances.shape[0]}"
        )
    baseline = _finite(baseline_m, name="baseline_m")
    if baseline <= 0:
        raise ValueError("baseline_m must be positive")

    frames: list[SpringFrame] = []
    for index in range(row_count):
        frame_id = index + 1  # Spring filenames are 1-based.
        fx, fy, cx, cy = intrinsics[index].tolist()
        if fx <= 0 or fy <= 0:
            raise SpringFormatError(f"non-positive focal length at frame {frame_id}")
        K = ((float(fx), 0.0, float(cx)), (0.0, float(fy), float(cy)), (0.0, 0.0, 1.0))
        pose = extrinsics_flat[index].reshape(4, 4)
        _validate_pose(pose, path=extrinsics_path, row_number=frame_id)
        left = root / "frame_left" / f"frame_left_{frame_id:04d}.png"
        right = root / "frame_right" / f"frame_right_{frame_id:04d}.png"
        disp_left = root / "disp1_left" / f"disp1_left_{frame_id:04d}.dsp5"
        disp_right = root / "disp1_right" / f"disp1_right_{frame_id:04d}.dsp5"
        if require_images:
            missing = [str(path) for path in (left, right) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"missing Spring image(s) for {sequence_id}/{frame_id}: {missing}"
                )
        if require_disparity:
            missing = [str(path) for path in (disp_left, disp_right) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"missing Spring disparity file(s) for {sequence_id}/{frame_id}: {missing}"
                )
        frames.append(
            SpringFrame(
                sequence_id=sequence_id,
                frame_id=frame_id,
                K=K,
                focal_distance_m=float(focal_distances[index]),
                extrinsics_camera_from_world=_as_tuple_matrix(pose),
                left_path=left,
                right_path=right,
                gt_disparity_left_path=disp_left,
                gt_disparity_right_path=disp_right,
            )
        )
    return SpringSequence(sequence_id, root, tuple(frames))


def iter_spring_sequences(
    root: str | Path,
    *,
    split: str = "train",
    sequences: Sequence[str] | None = None,
    require_images: bool = True,
    require_disparity: bool = True,
    baseline_m: float = SPRING_BASELINE_M,
) -> Iterator[SpringSequence]:
    """Yield validated Spring sequences in deterministic lexical order."""

    split_root = resolve_spring_split_root(root, split)
    for sequence_root in _sequence_directories(split_root, sequences):
        yield load_spring_sequence(
            sequence_root,
            require_images=require_images,
            require_disparity=require_disparity,
            baseline_m=baseline_m,
        )


def spring_manifest_records(
    root: str | Path,
    *,
    split: str = "train",
    sequences: Sequence[str] | None = None,
    require_images: bool = True,
    require_disparity: bool = True,
    baseline_m: float = SPRING_BASELINE_M,
    timestamp_fps: float | None = None,
) -> list[ManifestRecord]:
    """Build one left-reference manifest row per Spring frame.

    ``timestamp_fps`` is optional because Spring does not ship timestamps. If
    omitted, timestamps are monotonic frame indices (0,1,...); when supplied,
    timestamps are seconds from the first frame. Both choices preserve causal
    ordering, while the metadata makes the choice explicit.
    """

    if timestamp_fps is not None:
        fps = _finite(timestamp_fps, name="timestamp_fps")
        if fps <= 0:
            raise ValueError("timestamp_fps must be positive")
    else:
        fps = None
    records: list[ManifestRecord] = []
    for sequence in iter_spring_sequences(
        root,
        split=split,
        sequences=sequences,
        require_images=require_images,
        require_disparity=require_disparity,
        baseline_m=baseline_m,
    ):
        for frame in sequence.frames:
            timestamp = (
                float(frame.frame_id - 1) / fps if fps is not None else frame.timestamp
            )
            pose = [list(row) for row in frame.extrinsics_camera_from_world]
            records.append(
                ManifestRecord(
                    sequence_id=frame.sequence_id,
                    frame_id=frame.frame_id,
                    timestamp=timestamp,
                    left_path=str(frame.left_path),
                    right_path=str(frame.right_path),
                    K=frame.K,
                    baseline_m=float(baseline_m),
                    gt_disparity_path=str(frame.gt_disparity_left_path),
                    rectified=True,
                    extras={
                        "dataset": "spring",
                        "split": split,
                        "left_camera": "left",
                        "right_camera": "right",
                        "gt_disparity_right_path": str(frame.gt_disparity_right_path),
                        "gt_disparity_encoding": SPRING_DISPARITY_CONVENTION,
                        "gt_disparity_unit": "full_hd_pixels",
                        "gt_disparity_shape_hw": list(SPRING_DISPARITY_SIZE_HW),
                        "image_shape_hw": list(SPRING_IMAGE_SIZE_HW),
                        "gt_disparity_is_4x_resolution": True,
                        "gt_disparity_downsample_rule": "[::2,::2], values unchanged",
                        "gt_extrinsics_camera_from_world": pose,
                        "gt_pose_convention": SPRING_POSE_CONVENTION,
                        "gt_camera_center_world_m": [
                            float(value) for value in frame.camera_center_world
                        ],
                        "focal_distance_m": frame.focal_distance_m,
                        "focal_distance_semantics": (
                            "metric_focus_distance; 100 may denote infinity"
                        ),
                        "cam_data_root": str((sequence.sequence_root / "cam_data").resolve()),
                        "intrinsics_row_index": frame.frame_id - 1,
                        "frame_index_base": 1,
                        "timestamp_source": (
                            "frame_index/fps" if fps is not None else "frame_index"
                        ),
                        "timestamp_fps": fps,
                    },
                )
            )
    if not records:
        raise SpringFormatError("Spring manifest selection is empty")
    return records


def build_spring_manifest(
    root: str | Path,
    output: str | Path,
    *,
    split: str = "train",
    sequences: Sequence[str] | None = None,
    require_images: bool = True,
    require_disparity: bool = True,
    baseline_m: float = SPRING_BASELINE_M,
    timestamp_fps: float | None = None,
) -> list[ManifestRecord]:
    """Build and atomically write a validated Spring JSONL manifest."""

    records = spring_manifest_records(
        root,
        split=split,
        sequences=sequences,
        require_images=require_images,
        require_disparity=require_disparity,
        baseline_m=baseline_m,
        timestamp_fps=timestamp_fps,
    )
    write_manifest(output, records)
    return records


def relative_spring_pose(
    source: SpringFrame | np.ndarray,
    target: SpringFrame | np.ndarray,
) -> np.ndarray:
    """Return ``T_target_from_source`` for Spring camera-from-world poses."""

    def matrix(value: SpringFrame | np.ndarray, name: str) -> np.ndarray:
        result = (
            value.pose
            if isinstance(value, SpringFrame)
            else np.asarray(value, dtype=np.float64)
        )
        if result.shape != (4, 4) or not np.isfinite(result).all():
            raise ValueError(f"{name} pose must be finite [4,4]")
        _validate_pose(result, path=Path(f"<{name}>"), row_number=1)
        return result

    source_matrix = matrix(source, "source")
    target_matrix = matrix(target, "target")
    return target_matrix @ np.linalg.inv(source_matrix)


def spring_gt_pose_from_manifest(record: ManifestRecord) -> np.ndarray:
    """Extract and validate the Spring GT left-camera pose from a manifest row.

    Keeping this accessor explicit prevents an evaluator from accidentally
    substituting a predicted VGGT pose for the ground-truth pose.  The latter
    should live under a separate runtime field/cache key.
    """

    if not isinstance(record, ManifestRecord):
        raise TypeError("record must be a ManifestRecord")
    if str(record.extras.get("dataset", "")).strip().lower() != "spring":
        raise SpringFormatError("manifest record is not a Spring record")
    value = record.extras.get("gt_extrinsics_camera_from_world")
    if value is None:
        value = record.extras.get("gt_pose_camera_from_world")
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpringFormatError("Spring GT pose metadata is not numeric") from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise SpringFormatError("Spring GT pose metadata must be finite [4,4]")
    _validate_pose(pose, path=Path("<manifest:gt_extrinsics>"), row_number=1)
    return pose


def load_spring_disparity(
    path: str | Path,
    *,
    resolution: Literal["full", "image"] = "full",
    sign: Literal["positive", "left_to_right"] = "positive",
) -> np.ndarray:
    """Read one Spring ``.dsp5`` HDF5 disparity map.

    Spring stores 3840×2160 values while the vectors are expressed in 1920×
    1080 (full-HD) pixel units.  ``resolution='image'`` performs only the
    spatial ``[::2,::2]`` sampling used by the official loader; values are not
    divided by two.  ``sign='left_to_right'`` negates left-reference maps to
    match the historical ``spring_utils`` flow-vector convention.  The
    project default, ``sign='positive'``, is the positive magnitude expected by
    the FFS geometry path.
    """

    if resolution not in {"full", "image"}:
        raise ValueError("resolution must be 'full' or 'image'")
    if sign not in {"positive", "left_to_right"}:
        raise ValueError("sign must be 'positive' or 'left_to_right'")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Spring disparity file does not exist: {source}")
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on runtime profile.
        raise ImportError(
            "reading Spring .dsp5 files requires h5py; install it in the "
            "evaluation environment"
        ) from exc
    with h5py.File(source, "r") as handle:
        if "disparity" not in handle:
            raise SpringFormatError(f"Spring .dsp5 lacks dataset 'disparity': {source}")
        disparity = np.asarray(handle["disparity"], dtype=np.float32)
    if disparity.ndim != 2:
        raise SpringFormatError(
            f"Spring disparity must be 2-D, got {disparity.shape}: {source}"
        )
    if resolution == "image":
        disparity = disparity[::2, ::2]
    if sign == "left_to_right" and "disp1_left" in source.parent.name:
        disparity = -disparity
    return disparity


__all__ = [
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
