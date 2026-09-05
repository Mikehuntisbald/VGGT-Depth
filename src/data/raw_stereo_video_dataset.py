"""Raw causal clips for joint metric stereo-video geometry training.

This loader is intentionally independent from the cache-backed FFS-omega-TSR
lineage.  It reads the original calibrated stereo pixels and ground truth from
the existing :class:`~data.manifest.ManifestRecord` schema, including Spring's
right-disparity and camera-pose extras.

Each sample is oldest-to-current and has this stable contract (``T`` may vary):

* ``rgb``: ``[T,2,3,H,W]`` float32 in ``[0,1]``; view order is left, right.
* ``K``: ``[T,2,3,3]`` float32 in the final pixel grid.
* ``T_right_from_left``: ``[T,4,4]`` metric right-from-left transforms.
* ``T_current_from_previous``: ``[T,4,4]`` left-camera transforms.  Element
  zero is identity and its matching validity flag is always false.
* left/right disparity and validity: ``[T,1,H,W]`` in final-grid pixels.
  With the default ``target_mode="last"``, history slots are safe zero/invalid
  and ``target_time_mask`` selects only the endpoint; their GT files are never
  opened.  This permits a globally attentive causal prefix while supervising
  only its current frame.
* endpoint-relative temporal supervision: previous-frame left disparity and
  current-frame dynamic mask are each ``[1,H,W]``.  The latter uses Spring's
  backward rigid-map convention with **white/True meaning dynamic/excluded**.

Crop coordinates are shared by every time/view/target.  If a resize follows,
RGB uses PyTorch's ``align_corners=False`` convention, intrinsics receive its
matching half-pixel update, and horizontal disparity magnitudes are multiplied
by the width scale.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from geometry.camera import (
    crop_intrinsics,
    resize_intrinsics_align_corners_false,
    validate_intrinsics,
)

from .crop import CropWindow, sample_aligned_crop
from .manifest import ManifestRecord, load_manifest
from .spring import load_spring_disparity
from .training_dataset import build_causal_windows


@dataclass(frozen=True, slots=True)
class RawStereoVideoSample:
    """One raw, calibrated, causal stereo clip in oldest-to-current order."""

    rgb: Tensor
    K: Tensor
    T_right_from_left: Tensor
    T_current_from_previous: Tensor
    temporal_transform_valid: Tensor
    T_left_camera_from_world: Tensor
    camera_pose_valid: Tensor
    baseline_m: Tensor
    disparity_gt_left_px: Tensor
    disparity_gt_right_px: Tensor
    valid_gt_left: Tensor
    valid_gt_right: Tensor
    target_time_mask: Tensor
    previous_disparity_gt_left_px: Tensor
    previous_valid_gt_left: Tensor
    previous_disparity_gt_available: Tensor
    dynamic_mask_current: Tensor
    dynamic_mask_available: Tensor
    sequence_id: str
    frame_ids: tuple[int, ...]
    timestamps: tuple[float, ...]
    manifest_indices: tuple[int, ...]
    identity_metadata: Mapping[str, Any]

    @property
    def clip_length(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def disparity_gt_px(self) -> Tensor:
        """Return both targets as ``[T,2,1,H,W]`` in left/right order."""

        return torch.stack(
            (self.disparity_gt_left_px, self.disparity_gt_right_px), dim=1
        )

    @property
    def valid_gt(self) -> Tensor:
        """Return both target masks as ``[T,2,1,H,W]``."""

        return torch.stack((self.valid_gt_left, self.valid_gt_right), dim=1)

    @property
    def previous_disparity_gt_left_valid(self) -> Tensor:
        """Compatibility alias with validity attached to the target name."""

        return self.previous_valid_gt_left


@dataclass(frozen=True, slots=True)
class _ClipCandidate:
    sequence_id: str
    endpoint_index: int
    available_indices: tuple[int, ...]


def _plain_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _size_hw(value: tuple[int, int] | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (height,width) tuple or None")
    return (
        _plain_positive_integer(value[0], f"{name} height"),
        _plain_positive_integer(value[1], f"{name} width"),
    )


def _clip_length_bounds(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        length = _plain_positive_integer(value, "clip_length")
        return length, length
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("clip_length must be an integer or (minimum,maximum)")
    minimum = _plain_positive_integer(value[0], "minimum clip length")
    maximum = _plain_positive_integer(value[1], "maximum clip length")
    if minimum > maximum:
        raise ValueError("minimum clip length cannot exceed maximum clip length")
    return minimum, maximum


def _resolve_file(value: Any, *, manifest_directory: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest field {field_name!r} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest {field_name} does not exist: {path}")
    return path


def _image_size_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if height <= 0 or width <= 0:
        raise ValueError(f"RGB image is empty: {path}")
    return height, width


def _load_rgb_crop(
    path: Path, crop: CropWindow, expected_hw: tuple[int, int]
) -> Tensor:
    with Image.open(path) as source:
        if (source.height, source.width) != expected_hw:
            raise ValueError(
                f"RGB shape mismatch for {path}: expected {expected_hw}, "
                f"got {(source.height, source.width)}"
            )
        image = source.convert("RGB").crop(
            (crop.x_px, crop.y_px, crop.x_stop_px, crop.y_stop_px)
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _tensor_from_serialized_disparity(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "disparity" in archive:
                return np.asarray(archive["disparity"])
            if len(archive.files) != 1:
                raise ValueError(
                    f"NPZ disparity must contain 'disparity' or one array: {path}"
                )
            return np.asarray(archive[archive.files[0]])
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping):
            for key in ("disparity", "disp", "data"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, (Tensor, np.ndarray)):
            raise ValueError(f"serialized disparity is not a tensor/array: {path}")
        return np.asarray(
            payload.detach().cpu() if isinstance(payload, Tensor) else payload
        )
    with Image.open(path) as image:
        return np.asarray(image)


def _load_disparity(
    path: Path,
    *,
    expected_hw: tuple[int, int],
    crop: CropWindow,
) -> tuple[Tensor, Tensor]:
    if path.suffix.lower() == ".dsp5":
        array = load_spring_disparity(path, resolution="image", sign="positive")
    else:
        array = _tensor_from_serialized_disparity(path)
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"disparity must resolve to [H,W], got {array.shape}: {path}")
    if tuple(array.shape) != expected_hw:
        raise ValueError(
            f"disparity shape mismatch for {path}: expected {expected_hw}, "
            f"got {tuple(array.shape)}"
        )
    height_slice, width_slice = crop.slices_hw
    disparity = torch.from_numpy(
        np.asarray(array[height_slice, width_slice], dtype=np.float32).copy()
    ).unsqueeze(0)
    valid = torch.isfinite(disparity) & (disparity > 0.0)
    disparity = torch.where(valid, disparity, torch.zeros_like(disparity))
    return disparity.contiguous(), valid.contiguous()


def _path_from_manifest_value(value: str, manifest_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def _spring_dynamic_mask_path(
    record: ManifestRecord, manifest_directory: Path
) -> Path | None:
    if str(record.extras.get("dataset", "")).strip().lower() != "spring":
        return None
    if not isinstance(record.gt_disparity_path, str) or not record.gt_disparity_path:
        raise ValueError("Spring endpoint has no gt_disparity_path for map discovery")
    disparity_path = _path_from_manifest_value(
        record.gt_disparity_path, manifest_directory
    )
    name = "rigidmap_BW_left"
    return (
        disparity_path.parent.parent
        / "maps"
        / name
        / f"{name}_{record.frame_id:04d}.png"
    )


def _load_current_dynamic_mask(
    record: ManifestRecord,
    *,
    manifest_directory: Path,
    expected_image_hw: tuple[int, int],
    crop: CropWindow,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Load only the endpoint Spring BW rigid map under an explicit contract."""

    path = _spring_dynamic_mask_path(record, manifest_directory)
    target_shape = (1, crop.height_px, crop.width_px)
    contract: dict[str, Any] = {
        "available": False,
        "path": None if path is None else str(path),
        "source": "Spring/maps/rigidmap_BW_left",
        "frame_binding": "endpoint_frame_id_backward_to_previous",
        "semantic": "white_or_true_is_dynamic_and_excluded",
        "expected_file_format": "single_channel_binary_png",
        "expected_source_scale": "2x_width_and_2x_height_of_rgb",
        "image_grid_reduction": "2x2_majority_true_when_at_least_2_of_4",
        "post_reduction_transform": "shared_crop_then_nearest_exact_resize",
        "observed_pillow_mode": None,
        "observed_size_hw": None,
    }
    if path is None or not path.is_file():
        return (
            torch.zeros(target_shape, dtype=torch.bool),
            torch.tensor(False),
            contract,
        )
    if path.suffix.lower() != ".png":
        raise ValueError(f"Spring dynamic map must be PNG: {path}")

    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image).copy()
        observed_hw = (image.height, image.width)
    contract["observed_pillow_mode"] = mode
    contract["observed_size_hw"] = list(observed_hw)
    if mode not in {"1", "L"} or array.ndim != 2:
        raise ValueError(
            "Spring rigidmap_BW_left must be a single-channel binary PNG in "
            f"mode '1' or 'L', got mode={mode!r}, shape={array.shape}: {path}"
        )
    if array.dtype == np.bool_:
        dynamic_2x = array
    else:
        unique = np.unique(array)
        if not set(int(value) for value in unique).issubset({0, 255}):
            raise ValueError(
                "Spring rigidmap_BW_left must contain only black=0 and white=255: "
                f"{path}"
            )
        dynamic_2x = array == 255
    expected_2x_hw = (2 * expected_image_hw[0], 2 * expected_image_hw[1])
    if tuple(dynamic_2x.shape) != expected_2x_hw:
        raise ValueError(
            f"Spring rigidmap_BW_left shape must be {expected_2x_hw}, got "
            f"{tuple(dynamic_2x.shape)}: {path}"
        )
    height, width = expected_image_hw
    dynamic_image = dynamic_2x.reshape(height, 2, width, 2).sum(axis=(1, 3)) >= 2
    height_slice, width_slice = crop.slices_hw
    dynamic_crop = torch.from_numpy(
        dynamic_image[height_slice, width_slice].copy()
    ).unsqueeze(0)
    contract["available"] = True
    return dynamic_crop.contiguous(), torch.tensor(True), contract


def _matrix_4x4(value: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric [4,4] transform") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4,4] transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} has a malformed homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def _right_intrinsics(record: ManifestRecord) -> np.ndarray:
    value = record.extras.get("K_right", record.K)
    try:
        matrix = np.asarray(value, dtype=np.float64)
        validate_intrinsics(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid right intrinsics for {record.sequence_id}/{record.frame_id}: {exc}"
        ) from exc
    return matrix


def _right_from_left(record: ManifestRecord) -> np.ndarray:
    extras = record.extras
    explicit = None
    explicit_name = ""
    for name in (
        "T_right_from_left",
        "T_right_from_left_m",
        "T_right_rectified_from_left_rectified_m",
        "T_lr",
    ):
        if name in extras:
            explicit = extras[name]
            explicit_name = name
            break
    if explicit is not None:
        transform = _matrix_4x4(explicit, name=explicit_name)
    elif "P_left" in extras and "P_right" in extras:
        K_left = np.asarray(record.K, dtype=np.float64)
        K_right = _right_intrinsics(record)
        P_left = np.asarray(extras["P_left"], dtype=np.float64)
        P_right = np.asarray(extras["P_right"], dtype=np.float64)
        if P_left.shape != (3, 4) or P_right.shape != (3, 4):
            raise ValueError("P_left and P_right must both have shape [3,4]")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.linalg.solve(K_right, P_right[:, 3]) - np.linalg.solve(
            K_left, P_left[:, 3]
        )
    else:
        # ManifestRecord guarantees rectified=true and a positive metric
        # baseline, so this is the canonical orthoparallel fallback.
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = -float(record.baseline_m)

    transform = _matrix_4x4(transform, name="T_right_from_left")
    translation_norm = float(np.linalg.norm(transform[:3, 3]))
    tolerance = max(1e-6, 1e-3 * float(record.baseline_m))
    if not math.isclose(
        translation_norm, float(record.baseline_m), abs_tol=tolerance, rel_tol=0.0
    ):
        raise ValueError(
            "T_right_from_left translation norm disagrees with baseline_m for "
            f"{record.sequence_id}/{record.frame_id}: "
            f"{translation_norm} vs {record.baseline_m}"
        )
    return transform


def _absolute_pose(record: ManifestRecord) -> np.ndarray | None:
    for name in (
        "gt_extrinsics_camera_from_world",
        "gt_pose_camera_from_world",
        "T_left_camera_from_world",
        "T_camera_from_world",
    ):
        if name in record.extras:
            convention = record.extras.get("gt_pose_convention")
            if convention is not None and str(convention) not in {
                "world_to_camera_opencv",
                "camera_from_world",
                "camera-from-world",
            }:
                raise ValueError(
                    f"unsupported GT pose convention {convention!r} for "
                    f"{record.sequence_id}/{record.frame_id}"
                )
            return _matrix_4x4(record.extras[name], name=name)
    return None


def _direct_temporal_transform(record: ManifestRecord) -> np.ndarray | None:
    for name in (
        "T_current_from_previous",
        "T_current_from_previous_m",
        "T_temporal",
    ):
        if name in record.extras:
            if name == "T_temporal":
                convention = record.extras.get("T_temporal_convention")
                if convention not in {
                    "current_camera_from_previous_camera",
                    "current-from-previous",
                }:
                    raise ValueError(
                        "T_temporal requires T_temporal_convention="
                        "'current_camera_from_previous_camera'"
                    )
            return _matrix_4x4(record.extras[name], name=name)
    return None


def _temporal_geometry(
    records: Sequence[ManifestRecord],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    absolute = [_absolute_pose(record) for record in records]
    if any(pose is not None for pose in absolute) and not all(
        pose is not None for pose in absolute
    ):
        raise ValueError("a clip cannot mix present and missing absolute GT poses")

    identity = np.eye(4, dtype=np.float64)
    relative: list[np.ndarray] = [identity]
    if all(pose is not None for pose in absolute):
        poses = [pose for pose in absolute if pose is not None]
        for index in range(1, len(poses)):
            computed = poses[index] @ np.linalg.inv(poses[index - 1])
            direct = _direct_temporal_transform(records[index])
            if direct is not None and not np.allclose(
                direct, computed, atol=2e-4, rtol=0.0
            ):
                raise ValueError(
                    "direct temporal transform disagrees with absolute GT poses at "
                    f"{records[index].sequence_id}/{records[index].frame_id}"
                )
            relative.append(computed)
        pose_tensor = torch.as_tensor(np.stack(poses), dtype=torch.float32)
        pose_valid = torch.ones(len(records), dtype=torch.bool)
    else:
        for index in range(1, len(records)):
            direct = _direct_temporal_transform(records[index])
            if direct is None:
                raise ValueError(
                    "manifest clip requires either absolute camera-from-world poses "
                    "for every frame or direct T_current_from_previous transforms; "
                    f"missing at {records[index].sequence_id}/{records[index].frame_id}"
                )
            relative.append(direct)
        pose_tensor = (
            torch.eye(4, dtype=torch.float32).expand(len(records), -1, -1).clone()
        )
        pose_valid = torch.zeros(len(records), dtype=torch.bool)

    relative_tensor = torch.as_tensor(np.stack(relative), dtype=torch.float32)
    relative_valid = torch.ones(len(records), dtype=torch.bool)
    relative_valid[0] = False
    return relative_tensor, relative_valid, pose_tensor, pose_valid


def _build_clip_candidates(
    records: Sequence[ManifestRecord], minimum: int, maximum: int
) -> list[_ClipCandidate]:
    # Reuse the established causal validator/endpoint selection, then retain up
    # to ``maximum`` prior rows for variable-length clips.
    minimum_windows = build_causal_windows(
        records,
        student_sequence_length=minimum,
        vggt_context_pairs=minimum,
    )
    endpoints = {window.endpoint_index for window in minimum_windows}
    histories: dict[str, list[int]] = {}
    candidates: list[_ClipCandidate] = []
    for index, record in enumerate(records):
        history = histories.setdefault(record.sequence_id, [])
        history.append(index)
        if index in endpoints:
            candidates.append(
                _ClipCandidate(
                    sequence_id=record.sequence_id,
                    endpoint_index=index,
                    available_indices=tuple(history[-maximum:]),
                )
            )
    return candidates


class RawStereoVideoClipDataset(Dataset[RawStereoVideoSample]):
    """Load raw metric stereo clips without touching frozen-cache lineages.

    ``clip_length=8`` is the first joint-training configuration.  Passing a
    ``(minimum, maximum)`` pair samples a deterministic length per epoch and
    endpoint; :func:`collate_raw_stereo_video_samples` pads such samples.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        clip_length: int | tuple[int, int] = 8,
        crop_size_hw: tuple[int, int] | None = (512, 768),
        resize_size_hw: tuple[int, int] | None = None,
        crop_mode: Literal["random", "fixed"] = "random",
        fixed_crop_origin_xy: tuple[int, int] | None = None,
        crop_alignment: int = 1,
        target_mode: Literal["last", "all"] = "last",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_directory = self.manifest_path.parent
        self.records = load_manifest(self.manifest_path)
        self.minimum_clip_length, self.maximum_clip_length = _clip_length_bounds(
            clip_length
        )
        self.clip_length = clip_length
        self.crop_size_hw = _size_hw(crop_size_hw, "crop_size_hw")
        self.resize_size_hw = _size_hw(resize_size_hw, "resize_size_hw")
        self.crop_alignment = _plain_positive_integer(crop_alignment, "crop_alignment")
        if self.crop_size_hw is not None and any(
            dimension % self.crop_alignment for dimension in self.crop_size_hw
        ):
            raise ValueError("crop dimensions must be multiples of crop_alignment")
        if crop_mode not in {"random", "fixed"}:
            raise ValueError("crop_mode must be 'random' or 'fixed'")
        self.crop_mode = crop_mode
        if target_mode not in {"last", "all"}:
            raise ValueError("target_mode must be 'last' or 'all'")
        self.target_mode = target_mode
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        if fixed_crop_origin_xy is not None:
            if (
                not isinstance(fixed_crop_origin_xy, tuple)
                or len(fixed_crop_origin_xy) != 2
            ):
                raise ValueError("fixed_crop_origin_xy must be an (x,y) tuple")
            x_px, y_px = fixed_crop_origin_xy
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (x_px, y_px)
            ):
                raise ValueError(
                    "fixed crop origin values must be non-negative integers"
                )
            if x_px % self.crop_alignment or y_px % self.crop_alignment:
                raise ValueError("fixed crop origin must be crop_alignment aligned")
            self.fixed_crop_origin_xy = fixed_crop_origin_xy
        else:
            self.fixed_crop_origin_xy = None
        if crop_mode == "random" and self.fixed_crop_origin_xy is not None:
            raise ValueError("fixed_crop_origin_xy is only valid in fixed crop mode")
        self.candidates = _build_clip_candidates(
            self.records, self.minimum_clip_length, self.maximum_clip_length
        )
        if not self.candidates:
            raise ValueError(
                "manifest has no causal sequence long enough for clip_length="
                f"{clip_length}"
            )
        self._shared_epoch = mp.Value("q", 0, lock=True)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def epoch(self) -> int:
        with self._shared_epoch.get_lock():
            return int(self._shared_epoch.value)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = epoch

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence((self.seed, self.epoch, index, stream))
        )

    def _indices_for_item(self, index: int) -> tuple[int, ...]:
        available = self.candidates[index].available_indices
        upper = min(self.maximum_clip_length, len(available))
        if self.minimum_clip_length == upper:
            length = upper
        else:
            length = int(
                self._rng(index, 0).integers(self.minimum_clip_length, upper + 1)
            )
        return available[-length:]

    def _crop_for_item(
        self, index: int, source_height: int, source_width: int
    ) -> CropWindow:
        if self.crop_size_hw is None:
            if (
                source_height % self.crop_alignment
                or source_width % self.crop_alignment
            ):
                raise ValueError(
                    "full image dimensions must be multiples of crop_alignment"
                )
            return CropWindow(
                x_px=0,
                y_px=0,
                width_px=source_width,
                height_px=source_height,
                spatial_scale=self.crop_alignment,
            )
        crop_height, crop_width = self.crop_size_hw
        if self.crop_mode == "random":
            return sample_aligned_crop(
                source_height,
                source_width,
                crop_height,
                crop_width,
                self.crop_alignment,
                generator=self._rng(index, 1),
            )
        if self.fixed_crop_origin_xy is None:
            maximum_x = source_width - crop_width
            maximum_y = source_height - crop_height
            if maximum_x < 0 or maximum_y < 0:
                raise ValueError("crop dimensions exceed source image dimensions")
            x_px = (maximum_x // 2 // self.crop_alignment) * self.crop_alignment
            y_px = (maximum_y // 2 // self.crop_alignment) * self.crop_alignment
        else:
            x_px, y_px = self.fixed_crop_origin_xy
        crop = CropWindow(
            x_px=x_px,
            y_px=y_px,
            width_px=crop_width,
            height_px=crop_height,
            spatial_scale=self.crop_alignment,
        )
        crop.validate_within(source_height, source_width)
        return crop

    def __getitem__(self, index: int) -> RawStereoVideoSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        manifest_indices = self._indices_for_item(index)
        records = [self.records[item] for item in manifest_indices]
        first_left = _resolve_file(
            records[0].left_path,
            manifest_directory=self.manifest_directory,
            field_name="left_path",
        )
        source_hw = _image_size_hw(first_left)
        crop = self._crop_for_item(index, *source_hw)

        rgb_times: list[Tensor] = []
        disparity_left: list[Tensor] = []
        disparity_right: list[Tensor] = []
        valid_left: list[Tensor] = []
        valid_right: list[Tensor] = []
        intrinsics: list[Tensor] = []
        stereo_transforms: list[Tensor] = []
        baselines: list[float] = []
        source_paths: list[dict[str, str | None]] = []
        target_time_mask = torch.zeros(len(records), dtype=torch.bool)
        if self.target_mode == "all":
            target_time_mask[:] = True
        else:
            target_time_mask[-1] = True
        for time_index, record in enumerate(records):
            left_path = _resolve_file(
                record.left_path,
                manifest_directory=self.manifest_directory,
                field_name="left_path",
            )
            right_path = _resolve_file(
                record.right_path,
                manifest_directory=self.manifest_directory,
                field_name="right_path",
            )
            rgb_times.append(
                torch.stack(
                    (
                        _load_rgb_crop(left_path, crop, source_hw),
                        _load_rgb_crop(right_path, crop, source_hw),
                    )
                )
            )
            gt_left_path: Path | None = None
            gt_right_path: Path | None = None
            if bool(target_time_mask[time_index]):
                gt_left_path = _resolve_file(
                    record.gt_disparity_path,
                    manifest_directory=self.manifest_directory,
                    field_name="gt_disparity_path",
                )
                gt_right_path = _resolve_file(
                    record.extras.get("gt_disparity_right_path"),
                    manifest_directory=self.manifest_directory,
                    field_name="gt_disparity_right_path",
                )
                left_disp, left_valid = _load_disparity(
                    gt_left_path, expected_hw=source_hw, crop=crop
                )
                right_disp, right_valid = _load_disparity(
                    gt_right_path, expected_hw=source_hw, crop=crop
                )
            else:
                target_shape = (1, crop.height_px, crop.width_px)
                left_disp = torch.zeros(target_shape, dtype=torch.float32)
                right_disp = torch.zeros_like(left_disp)
                left_valid = torch.zeros(target_shape, dtype=torch.bool)
                right_valid = torch.zeros_like(left_valid)
            disparity_left.append(left_disp)
            disparity_right.append(right_disp)
            valid_left.append(left_valid)
            valid_right.append(right_valid)

            K_pair = np.stack(
                tuple(
                    crop_intrinsics(matrix, crop.x_px, crop.y_px)
                    for matrix in (np.asarray(record.K), _right_intrinsics(record))
                )
            )
            intrinsics.append(torch.as_tensor(K_pair, dtype=torch.float32))
            stereo_transforms.append(
                torch.as_tensor(_right_from_left(record), dtype=torch.float32)
            )
            baselines.append(float(record.baseline_m))
            source_paths.append(
                {
                    "left": str(left_path),
                    "right": str(right_path),
                    "gt_disparity_left": (
                        None if gt_left_path is None else str(gt_left_path)
                    ),
                    "gt_disparity_right": (
                        None if gt_right_path is None else str(gt_right_path)
                    ),
                    "temporal_previous_gt_disparity_left": None,
                }
            )

        rgb = torch.stack(rgb_times)
        disparity_gt_left_px = torch.stack(disparity_left)
        disparity_gt_right_px = torch.stack(disparity_right)
        valid_gt_left = torch.stack(valid_left)
        valid_gt_right = torch.stack(valid_right)
        K = torch.stack(intrinsics)

        previous_shape = (1, crop.height_px, crop.width_px)
        previous_disparity_gt_left_px = torch.zeros(previous_shape, dtype=torch.float32)
        previous_valid_gt_left = torch.zeros(previous_shape, dtype=torch.bool)
        previous_disparity_gt_available = torch.tensor(False)
        if len(records) >= 2:
            if self.target_mode == "all":
                previous_disparity_gt_left_px = disparity_gt_left_px[-2].clone()
                previous_valid_gt_left = valid_gt_left[-2].clone()
                source_paths[-2]["temporal_previous_gt_disparity_left"] = source_paths[
                    -2
                ]["gt_disparity_left"]
            else:
                previous_path = _resolve_file(
                    records[-2].gt_disparity_path,
                    manifest_directory=self.manifest_directory,
                    field_name="previous gt_disparity_path",
                )
                (
                    previous_disparity_gt_left_px,
                    previous_valid_gt_left,
                ) = _load_disparity(
                    previous_path,
                    expected_hw=source_hw,
                    crop=crop,
                )
                source_paths[-2]["temporal_previous_gt_disparity_left"] = str(
                    previous_path
                )
            previous_disparity_gt_available = torch.tensor(True)

        (
            dynamic_mask_current,
            dynamic_mask_available,
            dynamic_mask_contract,
        ) = _load_current_dynamic_mask(
            records[-1],
            manifest_directory=self.manifest_directory,
            expected_image_hw=source_hw,
            crop=crop,
        )
        output_hw = self.resize_size_hw or (crop.height_px, crop.width_px)
        if output_hw != (crop.height_px, crop.width_px):
            output_height, output_width = output_hw
            scale_x = output_width / crop.width_px
            scale_y = output_height / crop.height_px
            time_count = len(records)
            rgb = F.interpolate(
                rgb.reshape(time_count * 2, 3, crop.height_px, crop.width_px),
                size=output_hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(time_count, 2, 3, output_height, output_width)

            def resize_target(value: Tensor) -> Tensor:
                return F.interpolate(
                    value,
                    size=output_hw,
                    mode="nearest-exact",
                )

            disparity_gt_left_px = resize_target(disparity_gt_left_px) * scale_x
            disparity_gt_right_px = resize_target(disparity_gt_right_px) * scale_x
            valid_gt_left = resize_target(valid_gt_left.float()).bool()
            valid_gt_right = resize_target(valid_gt_right.float()).bool()
            disparity_gt_left_px = torch.where(
                valid_gt_left,
                disparity_gt_left_px,
                torch.zeros_like(disparity_gt_left_px),
            )
            disparity_gt_right_px = torch.where(
                valid_gt_right,
                disparity_gt_right_px,
                torch.zeros_like(disparity_gt_right_px),
            )
            previous_disparity_gt_left_px = (
                resize_target(previous_disparity_gt_left_px.unsqueeze(0))[0] * scale_x
            )
            previous_valid_gt_left = resize_target(
                previous_valid_gt_left.unsqueeze(0).float()
            )[0].bool()
            previous_disparity_gt_left_px = torch.where(
                previous_valid_gt_left,
                previous_disparity_gt_left_px,
                torch.zeros_like(previous_disparity_gt_left_px),
            )
            dynamic_mask_current = resize_target(
                dynamic_mask_current.unsqueeze(0).float()
            )[0].bool()
            K = resize_intrinsics_align_corners_false(K, scale_x, scale_y)
        else:
            scale_x = scale_y = 1.0

        (
            T_current_from_previous,
            temporal_transform_valid,
            T_left_camera_from_world,
            camera_pose_valid,
        ) = _temporal_geometry(records)
        metadata = {
            "manifest_path": str(self.manifest_path),
            "dataset_index": index,
            "endpoint_manifest_index": manifest_indices[-1],
            "epoch": self.epoch,
            "seed": self.seed,
            "source_size_hw": list(source_hw),
            "crop_xywh": [crop.x_px, crop.y_px, crop.width_px, crop.height_px],
            "resize_size_hw": list(output_hw),
            "resize_scale_xy": [scale_x, scale_y],
            "resize_coordinate_convention": "align_corners_false_half_pixel",
            "view_order": ["left", "right"],
            "target_mode": self.target_mode,
            "previous_disparity_gt_available": bool(previous_disparity_gt_available),
            "dynamic_mask_contract": dynamic_mask_contract,
            "source_paths": source_paths,
        }
        return RawStereoVideoSample(
            rgb=rgb.contiguous(),
            K=K.to(dtype=torch.float32).contiguous(),
            T_right_from_left=torch.stack(stereo_transforms).contiguous(),
            T_current_from_previous=T_current_from_previous.contiguous(),
            temporal_transform_valid=temporal_transform_valid,
            T_left_camera_from_world=T_left_camera_from_world.contiguous(),
            camera_pose_valid=camera_pose_valid,
            baseline_m=torch.tensor(baselines, dtype=torch.float32),
            disparity_gt_left_px=disparity_gt_left_px.contiguous(),
            disparity_gt_right_px=disparity_gt_right_px.contiguous(),
            valid_gt_left=valid_gt_left.contiguous(),
            valid_gt_right=valid_gt_right.contiguous(),
            target_time_mask=target_time_mask,
            previous_disparity_gt_left_px=(previous_disparity_gt_left_px.contiguous()),
            previous_valid_gt_left=previous_valid_gt_left.contiguous(),
            previous_disparity_gt_available=previous_disparity_gt_available,
            dynamic_mask_current=dynamic_mask_current.contiguous(),
            dynamic_mask_available=dynamic_mask_available,
            sequence_id=records[0].sequence_id,
            frame_ids=tuple(record.frame_id for record in records),
            timestamps=tuple(record.timestamp for record in records),
            manifest_indices=manifest_indices,
            identity_metadata=metadata,
        )


def _sample_field(sample: RawStereoVideoSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, RawStereoVideoSample):
        return getattr(sample, name)
    if isinstance(sample, Mapping):
        if name not in sample:
            raise KeyError(f"raw stereo-video sample is missing field {name!r}")
        return sample[name]
    raise TypeError(
        "samples must be RawStereoVideoSample instances or mappings, got "
        f"{type(sample).__name__}"
    )


def _left_pad_tensor(
    value: Tensor, maximum_t: int, *, identity: bool = False
) -> Tensor:
    padding = maximum_t - int(value.shape[0])
    if padding == 0:
        return value
    shape = (padding, *value.shape[1:])
    if identity:
        if value.shape[-2:] != (4, 4):
            raise ValueError("identity padding requires [...,4,4] tensors")
        prefix = (
            torch.eye(4, dtype=value.dtype, device=value.device).expand(shape).clone()
        )
    else:
        prefix = torch.zeros(shape, dtype=value.dtype, device=value.device)
    return torch.cat((prefix, value), dim=0)


def collate_raw_stereo_video_samples(
    samples: Sequence[RawStereoVideoSample | Mapping[str, Any]],
) -> dict[str, Any]:
    """Left-pad raw clips so every endpoint stays at batch time ``-1``."""

    if not samples:
        raise ValueError("cannot collate an empty raw stereo-video batch")
    rgbs = [_sample_field(sample, "rgb") for sample in samples]
    if not all(isinstance(value, Tensor) and value.ndim == 5 for value in rgbs):
        raise TypeError("sample rgb fields must be [T,2,3,H,W] tensors")
    lengths = [int(value.shape[0]) for value in rgbs]
    if any(length <= 0 for length in lengths):
        raise ValueError("raw stereo-video clips cannot be empty")
    maximum_t = max(lengths)
    expected_tail = tuple(rgbs[0].shape[1:])
    if expected_tail[:2] != (2, 3) or any(
        tuple(value.shape[1:]) != expected_tail for value in rgbs
    ):
        raise ValueError("all rgb clips must share [2,3,H,W] with left/right views")

    tensor_shapes = {
        "K": (2, 3, 3),
        "T_right_from_left": (4, 4),
        "T_current_from_previous": (4, 4),
        "temporal_transform_valid": (),
        "T_left_camera_from_world": (4, 4),
        "camera_pose_valid": (),
        "baseline_m": (),
        "disparity_gt_left_px": (1, expected_tail[-2], expected_tail[-1]),
        "disparity_gt_right_px": (1, expected_tail[-2], expected_tail[-1]),
        "valid_gt_left": (1, expected_tail[-2], expected_tail[-1]),
        "valid_gt_right": (1, expected_tail[-2], expected_tail[-1]),
        "target_time_mask": (),
    }
    values_by_name: dict[str, list[Tensor]] = {}
    for name, tail in tensor_shapes.items():
        values = [_sample_field(sample, name) for sample in samples]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"sample field {name!r} must be a tensor")
        for length, value in zip(lengths, values, strict=True):
            if tuple(value.shape) != (length, *tail):
                raise ValueError(
                    f"sample field {name!r} must have shape {(length, *tail)}, "
                    f"got {tuple(value.shape)}"
                )
        values_by_name[name] = values  # type: ignore[assignment]

    batch: dict[str, Any] = {
        "rgb": torch.stack([_left_pad_tensor(value, maximum_t) for value in rgbs]),
    }
    identity_fields = {
        "T_right_from_left",
        "T_current_from_previous",
        "T_left_camera_from_world",
    }
    for name, values in values_by_name.items():
        batch[name] = torch.stack(
            [
                _left_pad_tensor(value, maximum_t, identity=name in identity_fields)
                for value in values
            ]
        )

    endpoint_shapes = {
        "previous_disparity_gt_left_px": (
            1,
            expected_tail[-2],
            expected_tail[-1],
        ),
        "previous_valid_gt_left": (1, expected_tail[-2], expected_tail[-1]),
        "previous_disparity_gt_available": (),
        "dynamic_mask_current": (1, expected_tail[-2], expected_tail[-1]),
        "dynamic_mask_available": (),
    }
    for name, shape in endpoint_shapes.items():
        values = [_sample_field(sample, name) for sample in samples]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"sample endpoint field {name!r} must be a tensor")
        if any(tuple(value.shape) != shape for value in values):
            actual = [tuple(value.shape) for value in values]
            raise ValueError(
                f"sample endpoint field {name!r} must have shape {shape}, got {actual}"
            )
        batch[name] = torch.stack(values)  # type: ignore[arg-type]

    # K padding is deliberately zero and is never a usable calibration; the
    # authoritative time mask prevents padded slots from entering geometry.
    time_valid_mask = torch.zeros(len(samples), maximum_t, dtype=torch.bool)
    frame_ids = torch.full((len(samples), maximum_t), -1, dtype=torch.long)
    timestamps = torch.zeros(len(samples), maximum_t, dtype=torch.float64)
    manifest_indices = torch.full((len(samples), maximum_t), -1, dtype=torch.long)
    for batch_index, (sample, length) in enumerate(zip(samples, lengths, strict=True)):
        start = maximum_t - length
        time_valid_mask[batch_index, start:] = True
        sample_frame_ids = tuple(_sample_field(sample, "frame_ids"))
        sample_timestamps = tuple(_sample_field(sample, "timestamps"))
        sample_manifest_indices = tuple(_sample_field(sample, "manifest_indices"))
        if not all(
            len(value) == length
            for value in (sample_frame_ids, sample_timestamps, sample_manifest_indices)
        ):
            raise ValueError("sample provenance lengths must match its temporal length")
        frame_ids[batch_index, start:] = torch.tensor(
            sample_frame_ids, dtype=torch.long
        )
        timestamps[batch_index, start:] = torch.tensor(
            sample_timestamps, dtype=torch.float64
        )
        manifest_indices[batch_index, start:] = torch.tensor(
            sample_manifest_indices, dtype=torch.long
        )

    batch["time_valid_mask"] = time_valid_mask
    batch["clip_lengths"] = torch.tensor(lengths, dtype=torch.long)
    batch["frame_ids"] = frame_ids
    batch["timestamps"] = timestamps
    batch["manifest_indices"] = manifest_indices
    batch["sequence_id"] = [_sample_field(sample, "sequence_id") for sample in samples]
    batch["identity_metadata"] = [
        _sample_field(sample, "identity_metadata") for sample in samples
    ]
    batch["T_current_from_previous_valid"] = batch["temporal_transform_valid"]
    batch["previous_disparity_gt_left_valid"] = batch["previous_valid_gt_left"]
    batch["disparity_gt_px"] = torch.stack(
        (batch["disparity_gt_left_px"], batch["disparity_gt_right_px"]), dim=2
    )
    batch["valid_gt"] = torch.stack(
        (batch["valid_gt_left"], batch["valid_gt_right"]), dim=2
    )
    return batch


__all__ = [
    "RawStereoVideoClipDataset",
    "RawStereoVideoSample",
    "collate_raw_stereo_video_samples",
]
