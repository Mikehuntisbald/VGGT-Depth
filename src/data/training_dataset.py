"""Validated cached inputs for spatial and causal TSR training.

The trainable model consumes geometry saved by the isolated FFS environment;
it must never silently pair that geometry with a different manifest record or
source image.  This module therefore keeps cache provenance validation on the
data-loading boundary and makes every disparity unit explicit in public names.
"""

from __future__ import annotations

import re
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from geometry.camera import crop_intrinsics

from .cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    load_cache_record,
    sha256_file,
)
from .crop import CropWindow, sample_aligned_crop
from .manifest import ManifestRecord, load_manifest


@dataclass(frozen=True, slots=True)
class CausalWindow:
    """Indices for one endpoint using only frames from its causal past.

    ``student_indices`` contains the last ``T`` stereo times and
    ``vggt_indices`` contains the longer geometry context.  Both tuples end at
    ``endpoint_index`` and index the original input record sequence.
    """

    sequence_id: str
    endpoint_index: int
    student_indices: tuple[int, ...]
    vggt_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FFSTrainingSample:
    """One cropped endpoint loaded from frozen-backbone caches.

    Spatial shapes use ``C,H,W``.  Observation tensors live on the LR grid;
    despite that sampling grid, ``observation_disparity_hr_px`` is expressed
    in HR pixels and can be passed directly to :class:`FFSOmegaTSR`.
    ``observation_disparity_lr_px`` remains in LR pixels solely for the
    measurement-consistency loss.  Teacher tensors live on the HR grid.
    """

    rgb_hr: Tensor
    observation_disparity_hr_px: Tensor
    observation_disparity_lr_px: Tensor
    observation_confidence: Tensor
    observation_valid_mask: Tensor
    observation_trusted_mask: Tensor
    teacher_disparity_hr_px: Tensor | None
    teacher_confidence: Tensor | None
    teacher_valid_mask: Tensor | None
    teacher_trusted_mask: Tensor | None
    K_hr: Tensor
    baseline_m: Tensor
    sequence_id: str
    frame_id: int
    timestamp: float
    identity_metadata: Mapping[str, Any]

    @property
    def disparity_ffs_hr_px(self) -> Tensor:
        """Alias matching the trainable model's FFS input name."""

        return self.observation_disparity_hr_px

    @property
    def confidence_ffs(self) -> Tensor:
        """Alias matching the trainable model's FFS confidence name."""

        return self.observation_confidence

    @property
    def valid_ffs(self) -> Tensor:
        """Alias matching the trainable model's FFS validity-mask name."""

        return self.observation_valid_mask


# A shorter public spelling for callers that do not need to name the backbone.
TrainingSample = FFSTrainingSample


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_causal_windows(
    records: Sequence[ManifestRecord],
    *,
    student_sequence_length: int = 3,
    vggt_context_pairs: int = 5,
) -> list[CausalWindow]:
    """Build pure causal endpoint indices without crossing sequences.

    Timestamps must be strictly increasing within each sequence as encountered
    in ``records``.  Interleaved sequence IDs are supported: a window is built
    from prior entries of the same sequence only.  An endpoint is emitted only
    after enough same-sequence history exists for *both* requested contexts.

    Args:
        records: Validated manifest records in dataset order.
        student_sequence_length: Number of current/past stereo times used by
            the student, normally ``3``.
        vggt_context_pairs: Number of current/past stereo pairs used by VGGT,
            normally ``5``.
    """

    student_sequence_length = _positive_integer(
        student_sequence_length, "student_sequence_length"
    )
    vggt_context_pairs = _positive_integer(vggt_context_pairs, "vggt_context_pairs")
    by_sequence: dict[str, list[tuple[int, ManifestRecord]]] = defaultdict(list)
    last_timestamp: dict[str, float] = {}
    for index, record in enumerate(records):
        if not isinstance(record, ManifestRecord):
            raise TypeError(
                f"records[{index}] must be ManifestRecord, got {type(record).__name__}"
            )
        previous_timestamp = last_timestamp.get(record.sequence_id)
        if previous_timestamp is not None and record.timestamp <= previous_timestamp:
            raise ValueError(
                "timestamps must be strictly increasing within sequence "
                f"{record.sequence_id!r}: {record.timestamp} follows "
                f"{previous_timestamp} at input index {index}"
            )
        last_timestamp[record.sequence_id] = record.timestamp
        by_sequence[record.sequence_id].append((index, record))

    history_needed = max(student_sequence_length, vggt_context_pairs)
    windows: list[CausalWindow] = []
    for sequence_id, indexed_records in by_sequence.items():
        for endpoint_position in range(history_needed - 1, len(indexed_records)):
            student_start = endpoint_position - student_sequence_length + 1
            vggt_start = endpoint_position - vggt_context_pairs + 1
            endpoint_index = indexed_records[endpoint_position][0]
            student_indices = tuple(
                item[0]
                for item in indexed_records[student_start : endpoint_position + 1]
            )
            vggt_indices = tuple(
                item[0] for item in indexed_records[vggt_start : endpoint_position + 1]
            )
            if (
                student_indices[-1] != endpoint_index
                or vggt_indices[-1] != endpoint_index
            ):
                raise AssertionError("internal causal-window endpoint mismatch")
            windows.append(
                CausalWindow(
                    sequence_id=sequence_id,
                    endpoint_index=endpoint_index,
                    student_indices=student_indices,
                    vggt_indices=vggt_indices,
                )
            )
    return sorted(windows, key=lambda window: window.endpoint_index)


def _safe_cache_component(value: object) -> str:
    """Match the path-component normalization used by cache_ffs.py."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"cannot create a safe cache path component from {value!r}")
    return normalized


def cache_path_for_record(cache_root: Path, record: ManifestRecord) -> Path:
    """Return ``root/sequence_id/frame_id.pt`` using cache-writer naming."""

    return (
        cache_root
        / _safe_cache_component(record.sequence_id)
        / f"{_safe_cache_component(record.frame_id)}.pt"
    )


@lru_cache(maxsize=16_384)
def _sha256_for_unchanged_stat(path: str, size_bytes: int, mtime_ns: int) -> str:
    # ``size_bytes`` and ``mtime_ns`` intentionally participate in the cache
    # key.  Merely touching or replacing an input forces it to be hashed again.
    del size_bytes, mtime_ns
    return sha256_file(Path(path))


def _current_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_for_unchanged_stat(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def _resolve_source_path(path_text: str, manifest_directory: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest source image does not exist: {path}")
    return path


def _load_rgb_hr(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _scalar_chw(tensor: object, name: str, *, boolean: bool = False) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise CacheMismatchError(f"cache tensor {name!r} is missing or malformed")
    value = tensor
    if value.ndim == 4:
        if value.shape[:2] != (1, 1):
            raise CacheMismatchError(
                f"cache tensor {name!r} must have singleton B,C, got {tuple(value.shape)}"
            )
        value = value[0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim != 3 or value.shape[0] != 1:
        raise CacheMismatchError(
            f"cache tensor {name!r} must resolve to [1,H,W], got {tuple(value.shape)}"
        )
    if value.shape[0] != 1:
        raise CacheMismatchError(
            f"cache tensor {name!r} must resolve to [1,H,W], got {tuple(value.shape)}"
        )
    dtype = torch.bool if boolean else torch.float32
    return value.to(dtype=dtype).contiguous()


def _validate_cache_source(
    payload: Mapping[str, Any],
    *,
    record: ManifestRecord,
    left_path: Path,
    right_path: Path,
    left_sha256: str,
    right_sha256: str,
    cache_path: Path,
) -> None:
    metadata = payload.get("metadata")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping):
        raise CacheMismatchError(f"cache source metadata missing: {cache_path}")
    expected_record = record.to_dict()
    differences: dict[str, dict[str, Any]] = {}
    for name, expected, actual in (
        ("manifest_record", expected_record, source.get("manifest_record")),
        ("left_sha256", left_sha256, source.get("left_sha256")),
        ("right_sha256", right_sha256, source.get("right_sha256")),
    ):
        if actual != expected:
            differences[name] = {"expected": expected, "actual": actual}
    if differences:
        raise CacheMismatchError(
            f"cache source mismatch for {cache_path}: {differences}"
        )
    # The digest is authoritative.  Paths are included in the message to make
    # an eventual mismatch actionable without imposing an absolute-path match
    # on relocatable manifests.
    if not left_path.is_file() or not right_path.is_file():  # pragma: no cover
        raise FileNotFoundError(f"missing sources {left_path} or {right_path}")


def _crop_scalar_hr(tensor: Tensor, crop: CropWindow) -> Tensor:
    height_slice, width_slice = crop.slices_hw
    return tensor[:, height_slice, width_slice].contiguous()


def _crop_scalar_lr(tensor: Tensor, crop: CropWindow) -> Tensor:
    scale = crop.spatial_scale
    y_start = crop.y_px // scale
    x_start = crop.x_px // scale
    height_lr, width_lr = crop.lr_size_hw
    return tensor[
        :, y_start : y_start + height_lr, x_start : x_start + width_lr
    ].contiguous()


class CachedFFSTrainingDataset(Dataset[FFSTrainingSample]):
    """Join a rectified manifest to validated observation/teacher caches.

    Each item is one current-frame endpoint (Stage A, ``T=1``).  Temporal
    callers use :func:`build_causal_windows` to select T=3 endpoints without
    changing this spatial sample contract.

    Args:
        manifest_path: Validated JSONL manifest.
        observation_cache_root: Directory containing
            ``sequence_id/frame_id.pt`` observation records.
        teacher_cache_root: Equivalent teacher directory.  ``None`` supports
            measurement-only diagnostics and yields ``None`` teacher fields.
        observation_identity: If supplied, every observation cache must match
            this complete identity.
        teacher_identity: If supplied, every teacher cache must match it.
        crop_size_hr_hw: HR crop ``(height,width)``.  Defaults to the x2 MVP
            training crop ``(384,768)``; ``None`` retains the full image.
        crop_mode: ``"random"`` for deterministic epoch/index sampling or
            ``"fixed"`` for one fixed/centered crop.
        fixed_crop_origin_hr_xy: Optional ``(x,y)`` for fixed mode.  If absent,
            the largest scale-aligned centered origin is used.
        spatial_scale: HR/LR scale, fixed to ``2`` for the MVP by default.
        seed: Non-negative base seed.  Crop selection is a pure function of
            ``(seed, epoch, dataset_index)`` and is independent of access order.
    """

    sequence_length = 1

    def __init__(
        self,
        manifest_path: str | Path,
        observation_cache_root: str | Path,
        teacher_cache_root: str | Path | None,
        *,
        observation_identity: CacheIdentity | None = None,
        teacher_identity: CacheIdentity | None = None,
        crop_size_hr_hw: tuple[int, int] | None = (384, 768),
        crop_mode: Literal["random", "fixed"] = "random",
        fixed_crop_origin_hr_xy: tuple[int, int] | None = None,
        spatial_scale: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_directory = self.manifest_path.parent
        self.records = load_manifest(self.manifest_path)
        self.observation_cache_root = Path(observation_cache_root).resolve()
        self.teacher_cache_root = (
            None if teacher_cache_root is None else Path(teacher_cache_root).resolve()
        )
        self.observation_identity = observation_identity
        self.teacher_identity = teacher_identity
        self.spatial_scale = _positive_integer(spatial_scale, "spatial_scale")
        if crop_mode not in ("random", "fixed"):
            raise ValueError("crop_mode must be 'random' or 'fixed'")
        self.crop_mode = crop_mode
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        # A shared scalar keeps random crops epoch-aware even when the
        # DataLoader uses persistent forked workers.
        self._shared_epoch = mp.Value("q", 0, lock=True)
        if crop_size_hr_hw is not None:
            if len(crop_size_hr_hw) != 2:
                raise ValueError("crop_size_hr_hw must be (height,width)")
            crop_height, crop_width = crop_size_hr_hw
            _positive_integer(crop_height, "crop height")
            _positive_integer(crop_width, "crop width")
            if crop_height % self.spatial_scale or crop_width % self.spatial_scale:
                raise ValueError("HR crop dimensions must be multiples of spatial_scale")
            self.crop_size_hr_hw = (crop_height, crop_width)
        else:
            self.crop_size_hr_hw = None
        if fixed_crop_origin_hr_xy is not None:
            if len(fixed_crop_origin_hr_xy) != 2:
                raise ValueError("fixed_crop_origin_hr_xy must be (x,y)")
            x_px, y_px = fixed_crop_origin_hr_xy
            # CropWindow later checks non-negativity and alignment; validate
            # basic integer types now even before an image is loaded.
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (x_px, y_px)
            ):
                raise ValueError("fixed crop origin values must be integers")
            if x_px < 0 or y_px < 0:
                raise ValueError("fixed crop origin values must be non-negative")
            if x_px % self.spatial_scale or y_px % self.spatial_scale:
                raise ValueError("fixed crop origin must be aligned to spatial_scale")
            self.fixed_crop_origin_hr_xy = (x_px, y_px)
        else:
            self.fixed_crop_origin_hr_xy = None
        if self.crop_mode == "random" and self.fixed_crop_origin_hr_xy is not None:
            raise ValueError("fixed_crop_origin_hr_xy is only valid for fixed crop mode")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic crop epoch used by subsequent reads."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = epoch

    @property
    def epoch(self) -> int:
        """Current crop epoch, shared with persistent DataLoader workers."""

        with self._shared_epoch.get_lock():
            return int(self._shared_epoch.value)

    def _crop_for_index(self, index: int, height_hr: int, width_hr: int) -> CropWindow:
        if self.crop_size_hr_hw is None:
            if height_hr % self.spatial_scale or width_hr % self.spatial_scale:
                raise ValueError(
                    "full HR image dimensions must be multiples of spatial_scale"
                )
            return CropWindow(
                x_px=0,
                y_px=0,
                width_px=width_hr,
                height_px=height_hr,
                spatial_scale=self.spatial_scale,
            )
        crop_height, crop_width = self.crop_size_hr_hw
        if self.crop_mode == "random":
            generator = np.random.default_rng(
                np.random.SeedSequence((self.seed, self.epoch, index))
            )
            return sample_aligned_crop(
                height_hr,
                width_hr,
                crop_height,
                crop_width,
                self.spatial_scale,
                generator=generator,
            )
        if self.fixed_crop_origin_hr_xy is None:
            maximum_x = width_hr - crop_width
            maximum_y = height_hr - crop_height
            if maximum_x < 0 or maximum_y < 0:
                raise ValueError("crop dimensions exceed source image dimensions")
            x_px = (maximum_x // 2 // self.spatial_scale) * self.spatial_scale
            y_px = (maximum_y // 2 // self.spatial_scale) * self.spatial_scale
        else:
            x_px, y_px = self.fixed_crop_origin_hr_xy
        crop = CropWindow(
            x_px=x_px,
            y_px=y_px,
            width_px=crop_width,
            height_px=crop_height,
            spatial_scale=self.spatial_scale,
        )
        crop.validate_within(height_hr, width_hr)
        return crop

    def __getitem__(self, index: int) -> FFSTrainingSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        record = self.records[index]
        left_path = _resolve_source_path(record.left_path, self.manifest_directory)
        right_path = _resolve_source_path(record.right_path, self.manifest_directory)
        left_sha256 = _current_sha256(left_path)
        right_sha256 = _current_sha256(right_path)

        observation_path = cache_path_for_record(
            self.observation_cache_root, record
        )
        observation_payload = load_cache_record(
            observation_path, expected_identity=self.observation_identity
        )
        _validate_cache_source(
            observation_payload,
            record=record,
            left_path=left_path,
            right_path=right_path,
            left_sha256=left_sha256,
            right_sha256=right_sha256,
            cache_path=observation_path,
        )
        observation_tensors = observation_payload["tensors"]
        observation_disparity_hr_px = _scalar_chw(
            observation_tensors.get("observation_disparity_hr_px"),
            "observation_disparity_hr_px",
        )
        observation_disparity_lr_px = _scalar_chw(
            observation_tensors.get("observation_disparity_lr_px"),
            "observation_disparity_lr_px",
        )
        observation_confidence = _scalar_chw(
            observation_tensors.get("observation_confidence"),
            "observation_confidence",
        )
        observation_valid_mask = _scalar_chw(
            observation_tensors.get("observation_valid_mask"),
            "observation_valid_mask",
            boolean=True,
        )
        observation_trusted_mask = _scalar_chw(
            observation_tensors.get("observation_trusted_mask"),
            "observation_trusted_mask",
            boolean=True,
        )

        rgb_hr = _load_rgb_hr(left_path)
        height_hr, width_hr = rgb_hr.shape[-2:]
        if height_hr % self.spatial_scale or width_hr % self.spatial_scale:
            raise CacheMismatchError(
                f"HR image shape {(height_hr, width_hr)} is not divisible by "
                f"spatial_scale={self.spatial_scale}"
            )
        expected_observation_shape = (
            1,
            height_hr // self.spatial_scale,
            width_hr // self.spatial_scale,
        )
        for name, tensor in (
            ("observation_disparity_hr_px", observation_disparity_hr_px),
            ("observation_disparity_lr_px", observation_disparity_lr_px),
            ("observation_confidence", observation_confidence),
            ("observation_valid_mask", observation_valid_mask),
            ("observation_trusted_mask", observation_trusted_mask),
        ):
            if tensor.shape != expected_observation_shape:
                raise CacheMismatchError(
                    f"{name} shape {tuple(tensor.shape)} does not match LR grid "
                    f"{expected_observation_shape}"
                )
        finite_units = (
            torch.isfinite(observation_disparity_hr_px)
            & torch.isfinite(observation_disparity_lr_px)
        )
        if bool(finite_units.any()) and not torch.allclose(
            observation_disparity_hr_px[finite_units],
            self.spatial_scale * observation_disparity_lr_px[finite_units],
            rtol=2e-3,
            atol=2e-2,
        ):
            raise CacheMismatchError(
                "observation disparity unit mismatch: expected "
                "disparity_hr_px == spatial_scale * disparity_lr_px"
            )
        observation_valid_mask &= (
            torch.isfinite(observation_disparity_hr_px)
            & (observation_disparity_hr_px > 0)
        )
        observation_trusted_mask &= observation_valid_mask

        teacher_payload: Mapping[str, Any] | None = None
        teacher_path: Path | None = None
        teacher_disparity_hr_px: Tensor | None = None
        teacher_confidence: Tensor | None = None
        teacher_valid_mask: Tensor | None = None
        teacher_trusted_mask: Tensor | None = None
        if self.teacher_cache_root is not None:
            teacher_path = cache_path_for_record(self.teacher_cache_root, record)
            teacher_payload = load_cache_record(
                teacher_path, expected_identity=self.teacher_identity
            )
            _validate_cache_source(
                teacher_payload,
                record=record,
                left_path=left_path,
                right_path=right_path,
                left_sha256=left_sha256,
                right_sha256=right_sha256,
                cache_path=teacher_path,
            )
            teacher_tensors = teacher_payload["tensors"]
            teacher_disparity_hr_px = _scalar_chw(
                teacher_tensors.get("teacher_disparity_hr_px"),
                "teacher_disparity_hr_px",
            )
            teacher_confidence = _scalar_chw(
                teacher_tensors.get("teacher_confidence"), "teacher_confidence"
            )
            teacher_valid_mask = _scalar_chw(
                teacher_tensors.get("teacher_valid_mask"),
                "teacher_valid_mask",
                boolean=True,
            )
            teacher_trusted_mask = _scalar_chw(
                teacher_tensors.get("teacher_trusted_mask"),
                "teacher_trusted_mask",
                boolean=True,
            )
            expected_teacher_shape = (1, height_hr, width_hr)
            for name, tensor in (
                ("teacher_disparity_hr_px", teacher_disparity_hr_px),
                ("teacher_confidence", teacher_confidence),
                ("teacher_valid_mask", teacher_valid_mask),
                ("teacher_trusted_mask", teacher_trusted_mask),
            ):
                if tensor.shape != expected_teacher_shape:
                    raise CacheMismatchError(
                        f"{name} shape {tuple(tensor.shape)} does not match HR grid "
                        f"{expected_teacher_shape}"
                    )
            teacher_valid_mask &= (
                torch.isfinite(teacher_disparity_hr_px)
                & (teacher_disparity_hr_px > 0)
            )
            teacher_trusted_mask &= teacher_valid_mask

        crop = self._crop_for_index(index, height_hr, width_hr)
        height_slice, width_slice = crop.slices_hw
        rgb_hr = rgb_hr[:, height_slice, width_slice].contiguous()
        observation_disparity_hr_px = _crop_scalar_lr(
            observation_disparity_hr_px, crop
        )
        observation_disparity_lr_px = _crop_scalar_lr(
            observation_disparity_lr_px, crop
        )
        observation_confidence = _crop_scalar_lr(observation_confidence, crop)
        observation_valid_mask = _crop_scalar_lr(observation_valid_mask, crop)
        observation_trusted_mask = _crop_scalar_lr(observation_trusted_mask, crop)
        if teacher_disparity_hr_px is not None:
            assert teacher_confidence is not None
            assert teacher_valid_mask is not None
            assert teacher_trusted_mask is not None
            teacher_disparity_hr_px = _crop_scalar_hr(
                teacher_disparity_hr_px, crop
            )
            teacher_confidence = _crop_scalar_hr(teacher_confidence, crop)
            teacher_valid_mask = _crop_scalar_hr(teacher_valid_mask, crop)
            teacher_trusted_mask = _crop_scalar_hr(teacher_trusted_mask, crop)

        K_hr = torch.as_tensor(
            crop_intrinsics(np.asarray(record.K), crop.x_px, crop.y_px),
            dtype=torch.float32,
        ).contiguous()
        identity_metadata: dict[str, Any] = {
            "manifest_record": record.to_dict(),
            "manifest_path": str(self.manifest_path),
            "observation_cache_path": str(observation_path),
            "observation_cache_identity": dict(observation_payload["identity"]),
            "teacher_cache_path": None if teacher_path is None else str(teacher_path),
            "teacher_cache_identity": (
                None if teacher_payload is None else dict(teacher_payload["identity"])
            ),
            "source_sha256": {
                "left": left_sha256,
                "right": right_sha256,
            },
            "crop_hr_px": {
                "x": crop.x_px,
                "y": crop.y_px,
                "width": crop.width_px,
                "height": crop.height_px,
                "spatial_scale": crop.spatial_scale,
            },
            "dataset_index": index,
            "epoch": self.epoch,
            "seed": self.seed,
        }
        return FFSTrainingSample(
            rgb_hr=rgb_hr,
            observation_disparity_hr_px=observation_disparity_hr_px,
            observation_disparity_lr_px=observation_disparity_lr_px,
            observation_confidence=observation_confidence,
            observation_valid_mask=observation_valid_mask,
            observation_trusted_mask=observation_trusted_mask,
            teacher_disparity_hr_px=teacher_disparity_hr_px,
            teacher_confidence=teacher_confidence,
            teacher_valid_mask=teacher_valid_mask,
            teacher_trusted_mask=teacher_trusted_mask,
            K_hr=K_hr,
            baseline_m=torch.tensor(record.baseline_m, dtype=torch.float32),
            sequence_id=record.sequence_id,
            frame_id=record.frame_id,
            timestamp=record.timestamp,
            identity_metadata=identity_metadata,
        )


__all__ = [
    "CachedFFSTrainingDataset",
    "CausalWindow",
    "FFSTrainingSample",
    "TrainingSample",
    "build_causal_windows",
    "cache_path_for_record",
]
