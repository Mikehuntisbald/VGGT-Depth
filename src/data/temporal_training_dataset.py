"""Strict cached inputs for causal Stage-B temporal training.

The temporal dataset composes :class:`CachedFFSTrainingDataset` rather than
duplicating its FFS/source-lineage checks.  A sample contains exactly three
stereo times in oldest-to-current order.  All spatial tensors use one shared
scale-aligned HR crop, and every VGGT-derived tensor is tied to the matching
manifest record and FFS cache by SHA-256.

Disparities sampled on the LR grid are nevertheless expressed in **HR pixel
units**, except for the explicitly named ``observation_disparity_lr_px``
measurement tensor.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
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
from .manifest import ManifestRecord
from .stereo_calibration import RectifiedCalibrationIndex
from .training_dataset import (
    CachedFFSTrainingDataset,
    CausalWindow,
    FFSTrainingSample,
    build_causal_windows,
    cache_path_for_record,
)


DERIVED_COMPONENT = "vggt-ffs-derived-geometry"
CALIBRATED_DERIVED_COMPONENT = (
    "vggt-ffs-derived-geometry-calibrated-stereo-v2"
)
LEGACY_DERIVED_ALGORITHM = (
    "baseline_metric_scale+scale_only_alignment+strict_pose_quality"
)
CALIBRATED_DERIVED_ALGORITHM = (
    LEGACY_DERIVED_ALGORITHM + "+calibrated_stereo_constraint_v2"
)
DERIVED_CONTRACTS = {"legacy_v1", "calibrated_stereo_v2"}
STUDENT_SEQUENCE_LENGTH = 3
VGGT_CONTEXT_PAIRS = 5
VGGT_STEREO_VIEW_COUNT = 2 * VGGT_CONTEXT_PAIRS


@dataclass(frozen=True, slots=True)
class DerivedCacheEntry:
    """One trusted row from the formal derived-cache manifest."""

    target_manifest_index: int
    sequence_id: str
    frame_id: int
    timestamp: float
    cache_path: Path
    cache_sha256: str


@dataclass(frozen=True, slots=True)
class TemporalTrainingSample:
    """One oldest-to-current causal sequence for Stage B.

    Shapes use ``T=3``, HR crop ``Hh,Wh`` and LR crop ``Hl,Wl``:

    * RGB: ``[T,3,Hh,Wh]`` in ``[0,1]``.
    * FFS observation fields: ``[T,1,Hl,Wl]``.  The ``*_hr_px``
      disparity uses HR-pixel units; ``*_lr_px`` uses LR-pixel units.
    * Teacher fields: optional ``[T,1,Hh,Wh]`` in HR-pixel units.
    * VGGT prior fields: ``[T,1,Hl,Wl]`` in HR-pixel units/probability.
    * VGGT poses: ``[T,10,3,4]`` OpenCV camera-from-world metric poses.
      An entire time slice is zero when ``temporal_pose_valid_sequence`` is
      false and must not be used for history reprojection.
    * Optional Spring/GT poses: ``[T,10,3,4]`` camera-from-world metric poses
      assembled from the exact five causal manifest records.  These are kept
      separate from VGGT predictions so experiments can switch pose source
      without changing the cache lineage.
    * Intrinsics: ``[T,3,3]`` in the cropped HR coordinate system.
    """

    rgb_hr_sequence: Tensor
    observation_disparity_hr_px_sequence: Tensor
    observation_disparity_lr_px_sequence: Tensor
    observation_confidence_sequence: Tensor
    observation_valid_mask_sequence: Tensor
    observation_trusted_mask_sequence: Tensor
    teacher_disparity_hr_px_sequence: Tensor | None
    teacher_confidence_sequence: Tensor | None
    teacher_valid_mask_sequence: Tensor | None
    teacher_trusted_mask_sequence: Tensor | None
    K_hr_sequence: Tensor
    baseline_m_sequence: Tensor
    vggt_disparity_hr_px_sequence: Tensor
    vggt_confidence_sequence: Tensor
    vggt_valid_mask_sequence: Tensor
    vggt_extrinsics_camera_from_world_metric_sequence: Tensor
    temporal_pose_valid_sequence: Tensor
    temporal_pose_quality_score_sequence: Tensor
    static_prior_valid_sequence: Tensor
    sequence_id: str
    frame_ids: tuple[int, ...]
    timestamps: tuple[float, ...]
    manifest_indices: tuple[int, ...]
    identity_metadata: Mapping[str, Any]
    gt_extrinsics_camera_from_world_sequence: Tensor | None = None
    T_right_rectified_from_left_rectified_m_sequence: Tensor | None = None

    @property
    def disparity_ffs_hr_px_sequence(self) -> Tensor:
        """Model-facing alias for current-frame FFS measurements."""

        return self.observation_disparity_hr_px_sequence

    @property
    def confidence_ffs_sequence(self) -> Tensor:
        """Model-facing alias for FFS confidence."""

        return self.observation_confidence_sequence

    @property
    def valid_ffs_sequence(self) -> Tensor:
        """Model-facing alias for FFS validity."""

        return self.observation_valid_mask_sequence

    @property
    def history_pose_valid_sequence(self) -> Tensor:
        """Mask that gates all VGGT-pose history reprojection."""

        return self.temporal_pose_valid_sequence

    @property
    def gt_pose_sequence(self) -> Tensor | None:
        """Compatibility alias for the manifest GT pose context tensor."""

        return self.gt_extrinsics_camera_from_world_sequence


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _identity_from_mapping(
    value: Mapping[str, Any], *, cache_path: Path, expected_component: str
) -> CacheIdentity:
    fields = {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
    if set(value) != fields:
        raise CacheMismatchError(
            f"derived cache identity fields are malformed for {cache_path}: "
            f"expected {sorted(fields)}, got {sorted(value)}"
        )
    identity = CacheIdentity(
        component=str(value["component"]),
        upstream_commit=str(value["upstream_commit"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        torch_version=str(value["torch_version"]),
        cuda_version=(
            None if value["cuda_version"] is None else str(value["cuda_version"])
        ),
        config_sha256=str(value["config_sha256"]),
    )
    if identity.component != expected_component:
        raise CacheMismatchError(
            f"derived cache component mismatch for {cache_path}: expected "
            f"{expected_component!r}, got {identity.component!r}"
        )
    return identity


@lru_cache(maxsize=16_384)
def _sha256_for_unchanged_stat(path: str, size_bytes: int, mtime_ns: int) -> str:
    del size_bytes, mtime_ns
    return sha256_file(Path(path))


def _current_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_for_unchanged_stat(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def _load_derived_manifest(
    derived_cache_root: Path,
    records: list[Any],
) -> dict[int, DerivedCacheEntry]:
    """Load and bind the formal derived manifest to original indices."""

    manifest_path = derived_cache_root / "cache_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal derived cache manifest does not exist: {manifest_path}"
        )
    entries: dict[int, DerivedCacheEntry] = {}
    seen_targets: set[tuple[str, int]] = set()
    previous_selection_index = -1
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise CacheMismatchError(
                    f"blank row in derived cache manifest {manifest_path}:{line_number}"
                )
            try:
                row = json.loads(raw_line)
                selection_index = int(row["selection_index"])
                target_index = int(row["target_manifest_index"])
                sequence_id = str(row["sequence_id"])
                frame_id = int(row["frame_id"])
                timestamp = float(row["timestamp"])
                cache_path = Path(row["cache_path"]).expanduser().resolve()
                cache_sha256 = str(row["cache_sha256"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CacheMismatchError(
                    f"malformed derived cache manifest row "
                    f"{manifest_path}:{line_number}"
                ) from exc
            if selection_index <= previous_selection_index:
                raise CacheMismatchError(
                    "derived cache selection_index values must be strictly increasing"
                )
            previous_selection_index = selection_index
            if target_index < 0 or target_index >= len(records):
                raise CacheMismatchError(
                    f"derived target_manifest_index {target_index} is out of range"
                )
            record = records[target_index]
            if (
                sequence_id != record.sequence_id
                or frame_id != record.frame_id
                or timestamp != record.timestamp
            ):
                raise CacheMismatchError(
                    "derived cache manifest target disagrees with training manifest "
                    f"at index {target_index}"
                )
            target = (sequence_id, frame_id)
            if target in seen_targets or target_index in entries:
                raise CacheMismatchError(f"duplicate derived cache target: {target}")
            if not _is_within(cache_path, derived_cache_root):
                raise CacheMismatchError(
                    f"derived cache path escapes its declared root: {cache_path}"
                )
            canonical_path = cache_path_for_record(derived_cache_root, record).resolve()
            if cache_path != canonical_path:
                raise CacheMismatchError(
                    f"derived cache path is not canonical for {target}: {cache_path}"
                )
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"derived cache record is missing: {cache_path}"
                )
            if len(cache_sha256) != 64:
                raise CacheMismatchError(
                    f"malformed derived cache SHA-256 for {cache_path}"
                )
            entries[target_index] = DerivedCacheEntry(
                target_manifest_index=target_index,
                sequence_id=sequence_id,
                frame_id=frame_id,
                timestamp=timestamp,
                cache_path=cache_path,
                cache_sha256=cache_sha256,
            )
            seen_targets.add(target)
    if not entries:
        raise CacheMismatchError(f"derived cache manifest is empty: {manifest_path}")
    return entries


def _validate_derived_run_receipt(
    derived_cache_root: Path,
    *,
    entry_count: int,
    derived_contract: str,
) -> dict[str, Any]:
    """Validate the batch receipt that owns the derived manifest.

    Per-record derived identities intentionally differ because each binds the
    exact raw FFS/VGGT cache hashes.  This receipt validation supplies the
    complementary batch-level boundary: it binds manifest content, coverage,
    and causal geometry policy.
    """

    receipt_path = derived_cache_root / "run_receipt.json"
    manifest_path = derived_cache_root / "cache_manifest.jsonl"
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"formal derived cache receipt does not exist: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheMismatchError(
            f"cannot read derived cache receipt {receipt_path}: {exc}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise CacheMismatchError(f"derived receipt is not a mapping: {receipt_path}")
    calibrated = derived_contract == "calibrated_stereo_v2"
    expected_schema_version = 2 if calibrated else 1
    if receipt.get("schema_version") != expected_schema_version:
        raise CacheMismatchError(
            f"unsupported derived receipt schema in {receipt_path}: "
            f"{receipt.get('schema_version')!r}"
        )
    expected_component = (
        f"{CALIBRATED_DERIVED_COMPONENT}-batch"
        if calibrated
        else "vggt-ffs-derived-geometry-batch"
    )
    if receipt.get("component") != expected_component:
        raise CacheMismatchError(
            f"derived receipt component mismatch in {receipt_path}: "
            f"{receipt.get('component')!r}"
        )
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise CacheMismatchError(f"derived receipt config missing: {receipt_path}")
    required_config = {
        "algorithm": (
            CALIBRATED_DERIVED_ALGORITHM
            if calibrated
            else LEGACY_DERIVED_ALGORITHM
        ),
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    differences = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required_config.items()
        if config.get(key) != expected
    }
    if differences:
        raise CacheMismatchError(
            f"derived receipt causal contract mismatch in {receipt_path}: "
            f"{differences}"
        )
    if calibrated:
        calibration = config.get("rectified_stereo_calibration")
        if not isinstance(calibration, Mapping):
            raise CacheMismatchError(
                f"calibrated derived receipt lacks calibration lineage: {receipt_path}"
            )
        for name in ("sidecar_sha256", "receipt_sha256", "pixel_audit_sha256"):
            value = calibration.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise CacheMismatchError(
                    f"calibrated derived receipt has invalid {name}: {receipt_path}"
                )

    actual_manifest_sha256 = _current_sha256(manifest_path)
    output = receipt.get("output")
    if not isinstance(output, Mapping) or output.get(
        "cache_manifest_sha256"
    ) != actual_manifest_sha256:
        actual = (
            output.get("cache_manifest_sha256")
            if isinstance(output, Mapping)
            else None
        )
        raise CacheMismatchError(
            f"derived receipt/manifest SHA-256 mismatch in {receipt_path}: "
            f"expected {actual_manifest_sha256}, got {actual!r}"
        )

    counts = receipt.get("counts")
    selection = receipt.get("selection")
    if not isinstance(counts, Mapping) or not isinstance(selection, Mapping):
        raise CacheMismatchError(
            f"derived receipt coverage fields are missing: {receipt_path}"
        )
    selected = counts.get("selected")
    selected_windows = selection.get("selected_windows")
    written = counts.get("written")
    reused = counts.get("reused")
    count_values = (selected, selected_windows, written, reused)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in count_values
    ):
        raise CacheMismatchError(
            f"derived receipt coverage counts are malformed: {receipt_path}"
        )
    if selected != entry_count or selected_windows != entry_count:
        raise CacheMismatchError(
            f"derived receipt covers {selected}/{selected_windows} records but "
            f"manifest contains {entry_count}: {receipt_path}"
        )
    if written + reused != selected:
        raise CacheMismatchError(
            f"derived receipt written+reused does not cover selected: {receipt_path}"
        )
    for valid_name, rejected_name in (
        ("pose_valid", "pose_rejected"),
        ("static_prior_valid", "static_prior_rejected"),
    ):
        valid = counts.get(valid_name)
        rejected = counts.get(rejected_name)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (valid, rejected)
        ) or valid + rejected != selected:
            raise CacheMismatchError(
                f"derived receipt {valid_name}/{rejected_name} coverage is "
                f"malformed: {receipt_path}"
            )
    receipt_sha256 = _current_sha256(receipt_path)
    return {
        "component": receipt["component"],
        "derived_cache_root": str(derived_cache_root),
        "run_receipt_path": str(receipt_path),
        "run_receipt_sha256": receipt_sha256,
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": actual_manifest_sha256,
        "selected_records": entry_count,
        "config": dict(config),
    }


def _scalar_chw(
    tensors: Mapping[str, Any], name: str, *, boolean: bool = False
) -> Tensor:
    value = tensors.get(name)
    if not isinstance(value, Tensor):
        raise CacheMismatchError(f"derived tensor {name!r} is missing or malformed")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[:2] == (1, 1):
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 1:
        raise CacheMismatchError(
            f"derived tensor {name!r} must resolve to [1,H,W], got "
            f"{tuple(value.shape)}"
        )
    return value.to(dtype=torch.bool if boolean else torch.float32).contiguous()


def _scalar_bool(tensors: Mapping[str, Any], name: str) -> bool:
    value = tensors.get(name)
    if not isinstance(value, Tensor) or value.numel() != 1 or value.dtype != torch.bool:
        raise CacheMismatchError(
            f"derived tensor {name!r} must be one boolean scalar"
        )
    return bool(value.item())


def _continuous_pose_quality_score(
    quality: Mapping[str, Any],
    *,
    pose_valid: bool,
    derived_contract: str,
    thresholds: Mapping[str, Any] | None,
    cache_path: Path,
) -> float:
    """Map audited gate residuals to a bounded, lineage-bound quality score.

    Legacy-v1 fixture/cache metadata does not contain all diagnostic residuals,
    so it retains the historical binary valid score.  Calibrated-v2 records
    must expose every residual and the exact thresholds from their batch
    receipt.  A valid pose receives ``exp(-mean(residual/threshold))``; a
    rejected pose is exact zero.  This is a conditioning feature only and
    never changes the authoritative boolean pose gate.
    """

    if not pose_valid:
        return 0.0
    if derived_contract != "calibrated_stereo_v2":
        return 1.0
    if not isinstance(thresholds, Mapping):
        raise CacheMismatchError(
            f"calibrated pose-quality thresholds missing for {cache_path}"
        )
    baseline = quality.get("baseline")
    photometric = quality.get("photometric")
    depth = quality.get("depth_consistency")
    if not all(isinstance(value, Mapping) for value in (baseline, photometric, depth)):
        raise CacheMismatchError(
            f"calibrated pose-quality residuals missing for {cache_path}"
        )
    assert isinstance(baseline, Mapping)
    assert isinstance(photometric, Mapping)
    assert isinstance(depth, Mapping)
    specifications = (
        (baseline, "baseline_coefficient_of_variation", "max_baseline_cv"),
        (baseline, "stereo_rotation_error_max_deg", "max_stereo_rotation_error_deg"),
        (photometric, "median_absolute_rgb_residual", "max_photometric_median_absolute_rgb"),
        (depth, "weighted_mae_hr_px", "max_depth_weighted_mae_hr_px"),
        (depth, "median_absolute_error_hr_px", "max_depth_median_absolute_error_hr_px"),
    )
    ratios: list[float] = []
    for source, residual_name, threshold_name in specifications:
        residual = source.get(residual_name)
        threshold = thresholds.get(threshold_name)
        if (
            isinstance(residual, bool)
            or not isinstance(residual, (int, float))
            or not math.isfinite(float(residual))
            or float(residual) < 0
        ):
            raise CacheMismatchError(
                f"pose-quality residual {residual_name} is malformed for {cache_path}"
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) <= 0
        ):
            raise CacheMismatchError(
                f"pose-quality threshold {threshold_name} is malformed for {cache_path}"
            )
        ratios.append(float(residual) / float(threshold))
    score = math.exp(-sum(ratios) / len(ratios))
    if not math.isfinite(score) or not 0.0 < score <= 1.0:
        raise CacheMismatchError(
            f"computed pose-quality score is invalid for {cache_path}: {score}"
        )
    return score


def _validate_derived_lineage(
    payload: Mapping[str, Any],
    *,
    record: Any,
    observation_cache_path: Path,
    observation_cache_sha256: str,
    cache_path: Path,
    derived_contract: str,
    expected_calibration_lineage: Mapping[str, Any] | None = None,
    pose_quality_thresholds: Mapping[str, Any] | None = None,
) -> tuple[bool, bool, float]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CacheMismatchError(f"derived metadata missing for {cache_path}")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise CacheMismatchError(f"derived config missing for {cache_path}")
    calibrated = derived_contract == "calibrated_stereo_v2"
    required_config = {
        "algorithm": (
            CALIBRATED_DERIVED_ALGORITHM
            if calibrated
            else LEGACY_DERIVED_ALGORITHM
        ),
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    differences = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required_config.items()
        if config.get(key) != expected
    }
    if differences:
        raise CacheMismatchError(
            f"derived geometry contract mismatch for {cache_path}: {differences}"
        )
    if calibrated:
        calibration = config.get("rectified_stereo_calibration")
        if not isinstance(calibration, Mapping):
            raise CacheMismatchError(
                f"calibrated derived config lacks sidecar lineage for {cache_path}"
            )
        if not isinstance(expected_calibration_lineage, Mapping):
            raise CacheMismatchError(
                f"active calibrated sidecar lineage is unavailable for {cache_path}"
            )
        compared_fields = (
            "component",
            "contract_version",
            "sidecar_sha256",
            "receipt_sha256",
            "pixel_audit_sha256",
        )
        calibration_differences = {
            name: {
                "expected": expected_calibration_lineage.get(name),
                "actual": calibration.get(name),
            }
            for name in compared_fields
            if calibration.get(name) != expected_calibration_lineage.get(name)
        }
        if calibration_differences:
            raise CacheMismatchError(
                f"derived/active calibration lineage mismatch for {cache_path}: "
                f"{calibration_differences}"
            )
        if not isinstance(pose_quality_thresholds, Mapping) or config.get(
            "thresholds"
        ) != pose_quality_thresholds:
            raise CacheMismatchError(
                f"derived/batch pose-quality thresholds mismatch for {cache_path}"
            )

    target = metadata.get("target")
    expected_target = {
        "sequence_id": record.sequence_id,
        "frame_id": record.frame_id,
        "timestamp": record.timestamp,
    }
    if not isinstance(target, Mapping) or any(
        target.get(key) != value for key, value in expected_target.items()
    ):
        raise CacheMismatchError(
            f"derived target metadata mismatch for {cache_path}: "
            f"expected {expected_target}, got {target!r}"
        )
    source = metadata.get("source")
    linkage = source.get("linkage") if isinstance(source, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(linkage, Mapping):
        raise CacheMismatchError(f"derived source linkage missing for {cache_path}")
    if source.get("ffs_cache_sha256") != observation_cache_sha256:
        raise CacheMismatchError(
            f"derived FFS source SHA-256 mismatch for {cache_path}"
        )
    linked_record = linkage.get("target_manifest_record")
    if linked_record != record.to_dict():
        raise CacheMismatchError(
            f"derived manifest-record linkage mismatch for {cache_path}"
        )
    for name, expected in (
        ("target_sequence_id", record.sequence_id),
        ("target_frame_id", record.frame_id),
        ("target_timestamp", record.timestamp),
    ):
        if linkage.get(name) != expected:
            raise CacheMismatchError(
                f"derived linkage {name} mismatch for {cache_path}"
            )
    # The digest is authoritative and permits relocating a complete cache tree;
    # retain the current path in diagnostics without requiring old absolute paths.
    if not observation_cache_path.is_file():  # pragma: no cover
        raise FileNotFoundError(observation_cache_path)

    quality = metadata.get("pose_quality")
    alignment = quality.get("alignment") if isinstance(quality, Mapping) else None
    if not isinstance(quality, Mapping) or not isinstance(alignment, Mapping):
        raise CacheMismatchError(
            f"derived pose_quality metadata missing for {cache_path}"
        )
    pose_valid = quality.get("pose_valid")
    static_prior_valid = alignment.get("static_prior_valid")
    if not isinstance(pose_valid, bool) or not isinstance(static_prior_valid, bool):
        raise CacheMismatchError(
            f"derived pose/prior validity metadata is malformed for {cache_path}"
        )
    quality_score = _continuous_pose_quality_score(
        quality,
        pose_valid=pose_valid,
        derived_contract=derived_contract,
        thresholds=pose_quality_thresholds,
        cache_path=cache_path,
    )
    return pose_valid, static_prior_valid, quality_score


def _crop_lr(tensor: Tensor, crop: CropWindow) -> Tensor:
    x_lr = crop.x_px // crop.spatial_scale
    y_lr = crop.y_px // crop.spatial_scale
    height_lr, width_lr = crop.lr_size_hw
    return tensor[:, y_lr : y_lr + height_lr, x_lr : x_lr + width_lr].contiguous()


def _crop_hr(tensor: Tensor, crop: CropWindow) -> Tensor:
    height_slice, width_slice = crop.slices_hw
    return tensor[:, height_slice, width_slice].contiguous()


def _crop_spatial_sample(
    sample: FFSTrainingSample,
    crop: CropWindow,
    *,
    manifest_index: int,
    epoch: int,
) -> FFSTrainingSample:
    height_slice, width_slice = crop.slices_hw
    teacher_disparity = sample.teacher_disparity_hr_px
    teacher_confidence = sample.teacher_confidence
    teacher_valid = sample.teacher_valid_mask
    teacher_trusted = sample.teacher_trusted_mask
    if teacher_disparity is not None:
        assert teacher_confidence is not None
        assert teacher_valid is not None
        assert teacher_trusted is not None
        teacher_disparity = _crop_hr(teacher_disparity, crop)
        teacher_confidence = _crop_hr(teacher_confidence, crop)
        teacher_valid = _crop_hr(teacher_valid, crop)
        teacher_trusted = _crop_hr(teacher_trusted, crop)
    metadata = dict(sample.identity_metadata)
    metadata["crop_hr_px"] = {
        "x": crop.x_px,
        "y": crop.y_px,
        "width": crop.width_px,
        "height": crop.height_px,
        "spatial_scale": crop.spatial_scale,
    }
    metadata["dataset_index"] = manifest_index
    metadata["epoch"] = epoch
    return replace(
        sample,
        rgb_hr=sample.rgb_hr[:, height_slice, width_slice].contiguous(),
        observation_disparity_hr_px=_crop_lr(
            sample.observation_disparity_hr_px, crop
        ),
        observation_disparity_lr_px=_crop_lr(
            sample.observation_disparity_lr_px, crop
        ),
        observation_confidence=_crop_lr(sample.observation_confidence, crop),
        observation_valid_mask=_crop_lr(sample.observation_valid_mask, crop),
        observation_trusted_mask=_crop_lr(sample.observation_trusted_mask, crop),
        teacher_disparity_hr_px=teacher_disparity,
        teacher_confidence=teacher_confidence,
        teacher_valid_mask=teacher_valid,
        teacher_trusted_mask=teacher_trusted,
        K_hr=torch.as_tensor(
            crop_intrinsics(sample.K_hr.numpy(), crop.x_px, crop.y_px),
            dtype=torch.float32,
        ).contiguous(),
        identity_metadata=metadata,
    )


def _stack_optional(values: list[Tensor | None], name: str) -> Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise CacheMismatchError(
            f"teacher field {name!r} is present for only part of a temporal window"
        )
    return torch.stack([value for value in values if value is not None])


def _manifest_gt_pose(record: ManifestRecord) -> np.ndarray | None:
    """Read one manifest GT left-camera pose, if the producer supplied one.

    Spring manifests have used two field spellings while the adapter evolved;
    accept both so old manifests remain usable.  No VGGT/cache field is ever
    consulted here: this accessor is intentionally GT-only.
    """

    extras = record.extras
    value = None
    for key in (
        "gt_extrinsics_camera_from_world",
        "gt_pose_camera_from_world",
        "gt_pose",
    ):
        if key in extras:
            value = extras[key]
            break
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CacheMismatchError(
            "manifest GT pose metadata is not numeric for "
            f"{record.sequence_id}/{record.frame_id}"
        ) from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise CacheMismatchError(
            "manifest GT pose must be finite [4,4] for "
            f"{record.sequence_id}/{record.frame_id}"
        )
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        raise CacheMismatchError(
            "manifest GT pose homogeneous row is malformed for "
            f"{record.sequence_id}/{record.frame_id}"
        )
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0.0):
        raise CacheMismatchError(
            "manifest GT pose rotation is not orthonormal for "
            f"{record.sequence_id}/{record.frame_id}"
        )
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-5):
        raise CacheMismatchError(
            "manifest GT pose rotation determinant is not +1 for "
            f"{record.sequence_id}/{record.frame_id}"
        )
    return pose


def _manifest_gt_right_pose(record: ManifestRecord, left_pose: np.ndarray) -> np.ndarray:
    """Resolve a right-camera GT pose, deriving Spring's fixed rig if needed."""

    extras = record.extras
    value = None
    for key in (
        "gt_extrinsics_right_camera_from_world",
        "gt_pose_right_camera_from_world",
    ):
        if key in extras:
            value = extras[key]
            break
    if value is not None:
        try:
            right = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CacheMismatchError(
                "manifest right GT pose metadata is not numeric for "
                f"{record.sequence_id}/{record.frame_id}"
            ) from exc
        if right.shape != (4, 4) or not np.isfinite(right).all():
            raise CacheMismatchError(
                "manifest right GT pose must be finite [4,4] for "
                f"{record.sequence_id}/{record.frame_id}"
            )
        return right

    # Rectified Spring cameras satisfy X_right = X_left - [B,0,0].  For a
    # generic manifest this is still the least surprising fallback because
    # ManifestRecord owns the physical baseline and all existing calibration
    # code uses the same right-from-left convention.
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -float(record.baseline_m)
    return transform @ left_pose


class CachedTemporalTrainingDataset(Dataset[TemporalTrainingSample]):
    """Join T=3 cached FFS frames with per-time VGGT-derived geometry.

    Only windows whose three student records all have formal derived entries
    are emitted.  Every student and VGGT context index is at or before the
    endpoint; future frames and sequence-boundary crossings are rejected.

    ``derived_identities`` is optional because formal identities differ per
    record (they bind hashes of the exact raw FFS and VGGT caches).  When
    provided, it maps ``(sequence_id, frame_id)`` to the full expected
    :class:`CacheIdentity`.  Without it, the complete identity is still parsed
    and its component is strictly checked on every record.
    """

    sequence_length = STUDENT_SEQUENCE_LENGTH
    vggt_context_pairs = VGGT_CONTEXT_PAIRS

    def __init__(
        self,
        manifest_path: str | Path,
        observation_cache_root: str | Path,
        teacher_cache_root: str | Path | None,
        derived_cache_root: str | Path,
        *,
        observation_identity: CacheIdentity | None = None,
        teacher_identity: CacheIdentity | None = None,
        derived_identities: Mapping[tuple[str, int], CacheIdentity] | None = None,
        rectified_calibration_index: RectifiedCalibrationIndex | None = None,
        derived_contract: str = "legacy_v1",
        crop_size_hr_hw: tuple[int, int] | None = (384, 768),
        crop_mode: Literal["random", "fixed"] = "random",
        fixed_crop_origin_hr_xy: tuple[int, int] | None = None,
        spatial_scale: int = 2,
        student_sequence_length: int = STUDENT_SEQUENCE_LENGTH,
        vggt_context_pairs: int = VGGT_CONTEXT_PAIRS,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if derived_contract not in DERIVED_CONTRACTS:
            raise ValueError(
                f"derived_contract must be one of {sorted(DERIVED_CONTRACTS)}"
            )
        if (derived_contract == "calibrated_stereo_v2") != (
            rectified_calibration_index is not None
        ):
            raise ValueError(
                "calibrated_stereo_v2 requires a calibration sidecar index, and "
                "legacy_v1 forbids one"
            )
        self.derived_contract = derived_contract
        if student_sequence_length != STUDENT_SEQUENCE_LENGTH:
            raise ValueError("Stage B is fixed to a causal student sequence length T=3")
        if vggt_context_pairs != VGGT_CONTEXT_PAIRS:
            raise ValueError("Stage B is fixed to five causal VGGT stereo pairs")
        self.spatial_scale = _positive_integer(spatial_scale, "spatial_scale")
        if self.spatial_scale != 2:
            raise ValueError("the first-round temporal dataset is fixed to x2")
        if crop_mode not in ("random", "fixed"):
            raise ValueError("crop_mode must be 'random' or 'fixed'")
        self.crop_mode = crop_mode
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._shared_epoch = mp.Value("q", 0, lock=True)
        if crop_size_hr_hw is not None:
            if len(crop_size_hr_hw) != 2:
                raise ValueError("crop_size_hr_hw must be (height,width)")
            crop_height, crop_width = crop_size_hr_hw
            _positive_integer(crop_height, "crop height")
            _positive_integer(crop_width, "crop width")
            if crop_height % self.spatial_scale or crop_width % self.spatial_scale:
                raise ValueError(
                    "HR crop dimensions must be multiples of spatial_scale"
                )
            self.crop_size_hr_hw = (crop_height, crop_width)
        else:
            self.crop_size_hr_hw = None
        if fixed_crop_origin_hr_xy is not None:
            if len(fixed_crop_origin_hr_xy) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in fixed_crop_origin_hr_xy
            ):
                raise ValueError("fixed_crop_origin_hr_xy must be integer (x,y)")
            x_px, y_px = fixed_crop_origin_hr_xy
            if x_px < 0 or y_px < 0:
                raise ValueError("fixed crop origin must be non-negative")
            if x_px % self.spatial_scale or y_px % self.spatial_scale:
                raise ValueError("fixed crop origin must be aligned to spatial_scale")
            self.fixed_crop_origin_hr_xy = (x_px, y_px)
        else:
            self.fixed_crop_origin_hr_xy = None
        if crop_mode == "random" and fixed_crop_origin_hr_xy is not None:
            raise ValueError("fixed_crop_origin_hr_xy is only valid in fixed mode")

        self.spatial_dataset = CachedFFSTrainingDataset(
            manifest_path,
            observation_cache_root,
            teacher_cache_root,
            observation_identity=observation_identity,
            teacher_identity=teacher_identity,
            rectified_calibration_index=rectified_calibration_index,
            crop_size_hr_hw=None,
            crop_mode="fixed",
            spatial_scale=self.spatial_scale,
            seed=seed,
        )
        self.records = self.spatial_dataset.records
        self.manifest_path = self.spatial_dataset.manifest_path
        self.observation_cache_root = self.spatial_dataset.observation_cache_root
        self.derived_cache_root = Path(derived_cache_root).expanduser().resolve()
        if not self.derived_cache_root.is_dir():
            raise FileNotFoundError(
                f"derived cache root does not exist: {self.derived_cache_root}"
            )
        self.derived_entries = _load_derived_manifest(
            self.derived_cache_root, self.records
        )
        self.cache_lineage_summary = _validate_derived_run_receipt(
            self.derived_cache_root,
            entry_count=len(self.derived_entries),
            derived_contract=self.derived_contract,
        )
        if self.derived_contract == "calibrated_stereo_v2":
            assert rectified_calibration_index is not None
            active_calibration_lineage = {
                "component": CALIBRATED_DERIVED_COMPONENT,
                "contract_version": "stored_rectified_virtual_cameras_v1",
                "sidecar_sha256": rectified_calibration_index.sidecar_sha256,
                "receipt_sha256": rectified_calibration_index.receipt_sha256,
                "pixel_audit_sha256": rectified_calibration_index.pixel_audit_sha256,
            }
            receipt_calibration = self.cache_lineage_summary["config"].get(
                "rectified_stereo_calibration"
            )
            if not isinstance(receipt_calibration, Mapping):
                raise CacheMismatchError(
                    "calibrated derived receipt lacks active sidecar lineage"
                )
            calibration_differences = {
                name: {"expected": expected, "actual": receipt_calibration.get(name)}
                for name, expected in active_calibration_lineage.items()
                if receipt_calibration.get(name) != expected
            }
            if calibration_differences:
                raise CacheMismatchError(
                    "derived receipt/active calibration lineage mismatch: "
                    f"{calibration_differences}"
                )
        self.derived_identities = (
            None if derived_identities is None else dict(derived_identities)
        )
        candidates = build_causal_windows(
            self.records,
            student_sequence_length=STUDENT_SEQUENCE_LENGTH,
            vggt_context_pairs=VGGT_CONTEXT_PAIRS,
        )
        self.windows = [
            window
            for window in candidates
            if all(index in self.derived_entries for index in window.student_indices)
        ]
        if not self.windows:
            raise ValueError(
                "no causal T=3 window has derived geometry for all three times"
            )
        for window in self.windows:
            self._validate_window_causality(window)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def epoch(self) -> int:
        with self._shared_epoch.get_lock():
            return int(self._shared_epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """Set deterministic shared-crop epoch, including persistent workers."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = epoch
        self.spatial_dataset.set_epoch(epoch)

    def _validate_window_causality(self, window: CausalWindow) -> None:
        endpoint = self.records[window.endpoint_index]
        if len(window.student_indices) != STUDENT_SEQUENCE_LENGTH:
            raise CacheMismatchError("temporal window does not contain exactly T=3")
        if len(window.vggt_indices) != VGGT_CONTEXT_PAIRS:
            raise CacheMismatchError("VGGT window does not contain exactly five pairs")
        if window.student_indices[-1] != window.endpoint_index:
            raise CacheMismatchError("student window does not end at its endpoint")
        for index in (*window.student_indices, *window.vggt_indices):
            record = self.records[index]
            if record.sequence_id != endpoint.sequence_id:
                raise CacheMismatchError("causal window crosses a sequence boundary")
            if index > window.endpoint_index or record.timestamp > endpoint.timestamp:
                raise CacheMismatchError("causal window contains a future frame")

    def _crop_for_window(
        self, dataset_index: int, *, height_hr: int, width_hr: int
    ) -> CropWindow:
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
                np.random.SeedSequence((self.seed, self.epoch, dataset_index))
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

    def _gt_pose_context_for_endpoint(self, endpoint_index: int) -> Tensor | None:
        """Build ``[10,3,4]`` GT stereo poses for one causal endpoint.

        A Spring manifest stores one left-camera pose per frame.  The right
        camera is the fixed rectified partner and is derived with the physical
        baseline.  Early T=3 slices are left-padded with the earliest causal
        pose because no future frame may be borrowed.  Returning ``None`` for
        a manifest without GT pose metadata keeps legacy XD manifests
        backward-compatible; a partially populated context is rejected by
        :meth:`__getitem__` below.
        """

        if endpoint_index < 0 or endpoint_index >= len(self.records):
            return None
        endpoint_record = self.records[endpoint_index]
        causal_indices = [
            index
            for index, record in enumerate(self.records)
            if index <= endpoint_index
            and record.sequence_id == endpoint_record.sequence_id
        ]
        if not causal_indices:
            return None
        # Early student frames (the first two entries of a T=3 sample) do not
        # themselves have a complete five-pair VGGT context.  Pad only the
        # oldest side with its earliest causal pose; the transport/calibration
        # age gates never consume those padded slots, while indices 6/8 remain
        # the exact age-1/current poses required by T=3.
        context_indices = causal_indices[-VGGT_CONTEXT_PAIRS:]
        if len(context_indices) < VGGT_CONTEXT_PAIRS:
            context_indices = [context_indices[0]] * (
                VGGT_CONTEXT_PAIRS - len(context_indices)
            ) + context_indices
        poses: list[Tensor] = []
        for context_index in context_indices:
            record = self.records[context_index]
            left_pose = _manifest_gt_pose(record)
            if left_pose is None:
                return None
            right_pose = _manifest_gt_right_pose(record, left_pose)
            poses.extend(
                (
                    torch.as_tensor(left_pose[:3, :4], dtype=torch.float32),
                    torch.as_tensor(right_pose[:3, :4], dtype=torch.float32),
                )
            )
        result = torch.stack(poses, dim=0).contiguous()
        if result.shape != (VGGT_STEREO_VIEW_COUNT, 3, 4):
            raise CacheMismatchError(
                "manifest GT pose context must have shape [10,3,4], got "
                f"{tuple(result.shape)}"
            )
        if not bool(torch.isfinite(result).all().item()):
            raise CacheMismatchError(
                f"manifest GT pose context contains non-finite values at index "
                f"{endpoint_index}"
            )
        return result

    def _load_derived(
        self,
        manifest_index: int,
        spatial_sample: FFSTrainingSample,
        crop: CropWindow,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool, bool, dict[str, Any]]:
        record = self.records[manifest_index]
        entry = self.derived_entries[manifest_index]
        actual_cache_sha256 = _current_sha256(entry.cache_path)
        if actual_cache_sha256 != entry.cache_sha256:
            raise CacheMismatchError(
                f"derived cache SHA-256 mismatch for {entry.cache_path}: expected "
                f"{entry.cache_sha256}, got {actual_cache_sha256}"
            )
        expected_identity = None
        if self.derived_identities is not None:
            key = (record.sequence_id, record.frame_id)
            if key not in self.derived_identities:
                raise CacheMismatchError(
                    f"expected derived CacheIdentity is missing for {key}"
                )
            expected_identity = self.derived_identities[key]
        payload = load_cache_record(
            entry.cache_path, expected_identity=expected_identity
        )
        actual_identity = _identity_from_mapping(
            payload["identity"],
            cache_path=entry.cache_path,
            expected_component=(
                CALIBRATED_DERIVED_COMPONENT
                if self.derived_contract == "calibrated_stereo_v2"
                else DERIVED_COMPONENT
            ),
        )
        observation_path = cache_path_for_record(
            self.observation_cache_root, record
        ).resolve()
        (
            pose_valid_metadata,
            static_prior_valid_metadata,
            pose_quality_score,
        ) = _validate_derived_lineage(
            payload,
            record=record,
            observation_cache_path=observation_path,
            observation_cache_sha256=_current_sha256(observation_path),
            cache_path=entry.cache_path,
            derived_contract=self.derived_contract,
            expected_calibration_lineage=(
                self.cache_lineage_summary["config"].get(
                    "rectified_stereo_calibration"
                )
                if self.derived_contract == "calibrated_stereo_v2"
                else None
            ),
            pose_quality_thresholds=(
                self.cache_lineage_summary["config"].get("thresholds")
                if self.derived_contract == "calibrated_stereo_v2"
                else None
            ),
        )
        tensors = payload["tensors"]
        disparity_vggt_hr_px = _scalar_chw(
            tensors, "vggt_disparity_current_left_aligned_hr_px"
        )
        confidence_vggt = _scalar_chw(tensors, "vggt_aligned_confidence")
        valid_vggt = _scalar_chw(
            tensors, "vggt_aligned_valid_mask", boolean=True
        )
        pose_valid = _scalar_bool(tensors, "temporal_pose_valid")
        static_prior_valid = _scalar_bool(tensors, "static_prior_valid")
        if pose_valid != pose_valid_metadata:
            raise CacheMismatchError(
                f"temporal_pose_valid tensor/metadata mismatch for {entry.cache_path}"
            )
        if static_prior_valid != static_prior_valid_metadata:
            raise CacheMismatchError(
                f"static_prior_valid tensor/metadata mismatch for {entry.cache_path}"
            )
        expected_shape = spatial_sample.observation_disparity_hr_px.shape
        for name, tensor in (
            ("vggt_disparity_current_left_aligned_hr_px", disparity_vggt_hr_px),
            ("vggt_aligned_confidence", confidence_vggt),
            ("vggt_aligned_valid_mask", valid_vggt),
        ):
            if tensor.shape != expected_shape:
                raise CacheMismatchError(
                    f"derived tensor {name!r} shape {tuple(tensor.shape)} does not "
                    f"match FFS LR grid {tuple(expected_shape)}"
                )
        if not bool(torch.isfinite(disparity_vggt_hr_px).all()):
            raise CacheMismatchError(
                f"non-finite aligned disparity in {entry.cache_path}"
            )
        if not bool(torch.isfinite(confidence_vggt).all()) or bool(
            ((confidence_vggt < 0) | (confidence_vggt > 1)).any()
        ):
            raise CacheMismatchError(
                "aligned confidence is non-finite or outside [0,1] in "
                f"{entry.cache_path}"
            )
        if static_prior_valid:
            valid_vggt &= disparity_vggt_hr_px > 0
        else:
            if bool(valid_vggt.any()) or bool(
                (disparity_vggt_hr_px != 0).any()
            ) or bool((confidence_vggt != 0).any()):
                raise CacheMismatchError(
                    "invalid static VGGT prior must be zero-filled with an empty mask: "
                    f"{entry.cache_path}"
                )

        calibrated = self.derived_contract == "calibrated_stereo_v2"
        extrinsics_name = (
            "vggt_extrinsics_camera_from_world_metric_temporal_stereo_constrained"
            if calibrated
            else "vggt_extrinsics_camera_from_world_metric_temporal"
        )
        extrinsics = tensors.get(extrinsics_name)
        if not isinstance(extrinsics, Tensor) or extrinsics.shape != (
            VGGT_STEREO_VIEW_COUNT,
            3,
            4,
        ):
            shape = (
                None
                if not isinstance(extrinsics, Tensor)
                else tuple(extrinsics.shape)
            )
            raise CacheMismatchError(
                f"temporal extrinsics must have shape [10,3,4], got {shape}"
            )
        extrinsics = extrinsics.to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(extrinsics).all()):
            raise CacheMismatchError(f"non-finite temporal poses in {entry.cache_path}")
        if not pose_valid and bool((extrinsics != 0).any()):
            raise CacheMismatchError(
                f"rejected temporal pose must be zero-filled: {entry.cache_path}"
            )
        # Defense in depth: downstream sees no pose whenever the strict gate
        # rejects it, even if tensor representation changes in a later schema.
        if not pose_valid:
            extrinsics = torch.zeros_like(extrinsics)

        if calibrated:
            calibration_valid = _scalar_bool(tensors, "stereo_calibration_valid")
            cached_transform = tensors.get(
                "T_right_rectified_from_left_rectified_m"
            )
            expected_transform = (
                spatial_sample.T_right_rectified_from_left_rectified_m
            )
            if not calibration_valid:
                raise CacheMismatchError(
                    f"calibrated derived cache rejected stereo calibration: {entry.cache_path}"
                )
            if (
                not isinstance(cached_transform, Tensor)
                or tuple(cached_transform.shape) != (4, 4)
                or expected_transform is None
            ):
                raise CacheMismatchError(
                    f"calibrated stereo transform is missing or malformed: {entry.cache_path}"
                )
            if not torch.allclose(
                cached_transform.float(), expected_transform.float(), atol=1e-6, rtol=0.0
            ):
                raise CacheMismatchError(
                    f"derived/sidecar stereo transforms disagree: {entry.cache_path}"
                )

        return (
            _crop_lr(disparity_vggt_hr_px, crop),
            _crop_lr(confidence_vggt, crop),
            _crop_lr(valid_vggt, crop),
            extrinsics,
            pose_valid,
            static_prior_valid,
            {
                "cache_path": str(entry.cache_path),
                "cache_sha256": actual_cache_sha256,
                "cache_identity": actual_identity.to_dict(),
                "pose_valid": pose_valid,
                "pose_quality_score": pose_quality_score,
                "static_prior_valid": static_prior_valid,
            },
        )

    def __getitem__(self, index: int) -> TemporalTrainingSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.windows)
        if index < 0 or index >= len(self.windows):
            raise IndexError(index)
        window = self.windows[index]
        self._validate_window_causality(window)
        full_samples = [
            self.spatial_dataset[manifest_index]
            for manifest_index in window.student_indices
        ]
        spatial_shapes = {tuple(sample.rgb_hr.shape[-2:]) for sample in full_samples}
        if len(spatial_shapes) != 1:
            raise CacheMismatchError(
                "temporal RGB shapes differ within one window: "
                f"{sorted(spatial_shapes)}"
            )
        height_hr, width_hr = next(iter(spatial_shapes))
        crop = self._crop_for_window(
            index, height_hr=height_hr, width_hr=width_hr
        )
        cropped_samples = [
            _crop_spatial_sample(
                sample,
                crop,
                manifest_index=manifest_index,
                epoch=self.epoch,
            )
            for manifest_index, sample in zip(
                window.student_indices, full_samples, strict=True
            )
        ]
        derived = [
            self._load_derived(manifest_index, full_sample, crop)
            for manifest_index, full_sample in zip(
                window.student_indices, full_samples, strict=True
            )
        ]
        records = [self.records[item] for item in window.student_indices]
        gt_pose_contexts = [
            self._gt_pose_context_for_endpoint(manifest_index)
            for manifest_index in window.student_indices
        ]
        if all(value is None for value in gt_pose_contexts):
            gt_pose_sequence = None
        elif any(value is None for value in gt_pose_contexts):
            raise CacheMismatchError(
                "manifest GT pose metadata is only partially available in a "
                f"temporal window for sequence {window.sequence_id!r}"
            )
        else:
            gt_pose_sequence = torch.stack(
                [value for value in gt_pose_contexts if value is not None], dim=0
            )
        return TemporalTrainingSample(
            rgb_hr_sequence=torch.stack([sample.rgb_hr for sample in cropped_samples]),
            observation_disparity_hr_px_sequence=torch.stack(
                [sample.observation_disparity_hr_px for sample in cropped_samples]
            ),
            observation_disparity_lr_px_sequence=torch.stack(
                [sample.observation_disparity_lr_px for sample in cropped_samples]
            ),
            observation_confidence_sequence=torch.stack(
                [sample.observation_confidence for sample in cropped_samples]
            ),
            observation_valid_mask_sequence=torch.stack(
                [sample.observation_valid_mask for sample in cropped_samples]
            ),
            observation_trusted_mask_sequence=torch.stack(
                [sample.observation_trusted_mask for sample in cropped_samples]
            ),
            teacher_disparity_hr_px_sequence=_stack_optional(
                [sample.teacher_disparity_hr_px for sample in cropped_samples],
                "teacher_disparity_hr_px",
            ),
            teacher_confidence_sequence=_stack_optional(
                [sample.teacher_confidence for sample in cropped_samples],
                "teacher_confidence",
            ),
            teacher_valid_mask_sequence=_stack_optional(
                [sample.teacher_valid_mask for sample in cropped_samples],
                "teacher_valid_mask",
            ),
            teacher_trusted_mask_sequence=_stack_optional(
                [sample.teacher_trusted_mask for sample in cropped_samples],
                "teacher_trusted_mask",
            ),
            K_hr_sequence=torch.stack([sample.K_hr for sample in cropped_samples]),
            baseline_m_sequence=torch.stack(
                [sample.baseline_m for sample in cropped_samples]
            ),
            vggt_disparity_hr_px_sequence=torch.stack([item[0] for item in derived]),
            vggt_confidence_sequence=torch.stack([item[1] for item in derived]),
            vggt_valid_mask_sequence=torch.stack([item[2] for item in derived]),
            vggt_extrinsics_camera_from_world_metric_sequence=torch.stack(
                [item[3] for item in derived]
            ),
            temporal_pose_valid_sequence=torch.tensor(
                [item[4] for item in derived], dtype=torch.bool
            ),
            temporal_pose_quality_score_sequence=torch.tensor(
                [item[6]["pose_quality_score"] for item in derived],
                dtype=torch.float32,
            ),
            static_prior_valid_sequence=torch.tensor(
                [item[5] for item in derived], dtype=torch.bool
            ),
            sequence_id=window.sequence_id,
            frame_ids=tuple(record.frame_id for record in records),
            timestamps=tuple(record.timestamp for record in records),
            manifest_indices=window.student_indices,
            identity_metadata={
                "manifest_path": str(self.manifest_path),
                "derived_cache_lineage": dict(self.cache_lineage_summary),
                "sequence_id": window.sequence_id,
                "endpoint_manifest_index": window.endpoint_index,
                "student_manifest_indices": list(window.student_indices),
                "vggt_context_manifest_indices": list(window.vggt_indices),
                "crop_hr_px": {
                    "x": crop.x_px,
                    "y": crop.y_px,
                    "width": crop.width_px,
                    "height": crop.height_px,
                    "spatial_scale": crop.spatial_scale,
                },
                "epoch": self.epoch,
                "seed": self.seed,
                "per_time_ffs": [
                    dict(sample.identity_metadata) for sample in cropped_samples
                ],
                "per_time_derived": [item[6] for item in derived],
                "gt_pose_available": gt_pose_sequence is not None,
            },
            gt_extrinsics_camera_from_world_sequence=gt_pose_sequence,
            T_right_rectified_from_left_rectified_m_sequence=_stack_optional(
                [
                    sample.T_right_rectified_from_left_rectified_m
                    for sample in cropped_samples
                ],
                "T_right_rectified_from_left_rectified_m",
            ),
        )


__all__ = [
    "CachedTemporalTrainingDataset",
    "DerivedCacheEntry",
    "TemporalTrainingSample",
]
