#!/usr/bin/env python3
"""Deterministic pixel-level audit of same-row stereo rectification.

The manifest's ``rectified`` flag is not accepted as evidence.  This tool
samples each train and validation sequence independently, matches SIFT features
from left to right, applies a positive-horizontal-disparity plausibility gate,
and estimates a fundamental matrix with RANSAC.  The versioned contract is
published only when every declared finite threshold passes.

OpenCV is imported lazily so pure aggregation and threshold tests run in the
lightweight ``env-tsr`` environment.  Formal image matching runs in ``env-ffs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


AUDIT_SCHEMA_VERSION = 1
RECTIFICATION_CONTRACT = "audited_same_row_rectified_pixels_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EpipolarAuditError(RuntimeError):
    """Raised when audit inputs or image evidence are malformed."""


@dataclass(frozen=True, slots=True)
class RectificationThresholds:
    """Finite gates required to publish the pixel rectification contract."""

    min_ratio_matches_per_frame: int = 64
    min_plausible_matches_per_frame: int = 48
    min_ransac_inliers_per_frame: int = 32
    min_frame_coverage_fraction: float = 0.95
    max_abs_median_dy_px: float = 1.25
    max_p95_abs_dy_px: float = 3.0

    def validate(self) -> None:
        for name in (
            "min_ratio_matches_per_frame",
            "min_plausible_matches_per_frame",
            "min_ransac_inliers_per_frame",
        ):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 8, f"{name} must be an integer >= 8")
        _require(
            self.min_plausible_matches_per_frame
            <= self.min_ratio_matches_per_frame,
            "min_plausible_matches_per_frame cannot exceed min_ratio_matches_per_frame",
        )
        _require(
            self.min_ransac_inliers_per_frame
            <= self.min_plausible_matches_per_frame,
            "min_ransac_inliers_per_frame cannot exceed min_plausible_matches_per_frame",
        )
        for name in (
            "min_frame_coverage_fraction",
            "max_abs_median_dy_px",
            "max_p95_abs_dy_px",
        ):
            value = float(getattr(self, name))
            _require(math.isfinite(value), f"{name} must be finite")
        _require(
            0.0 < self.min_frame_coverage_fraction <= 1.0,
            "min_frame_coverage_fraction must be in (0,1]",
        )
        _require(self.max_abs_median_dy_px >= 0.0, "max_abs_median_dy_px must be non-negative")
        _require(self.max_p95_abs_dy_px > 0.0, "max_p95_abs_dy_px must be positive")


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    samples_per_sequence: int = 32
    seed: int = 42
    sift_nfeatures: int = 4096
    sift_contrast_threshold: float = 0.02
    ratio_threshold: float = 0.75
    min_horizontal_disparity_px: float = 0.25
    max_horizontal_disparity_px: float = 512.0
    broad_vertical_prefilter_px: float = 32.0
    ransac_reprojection_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iterations: int = 10_000

    def validate(self) -> None:
        for name in ("samples_per_sequence", "sift_nfeatures", "ransac_max_iterations"):
            value = getattr(self, name)
            _require(_is_int(value) and value > 0, f"{name} must be a positive integer")
        _require(_is_int(self.seed) and 0 <= self.seed <= 2**31 - 1, "seed must fit signed int32")
        for name in (
            "sift_contrast_threshold",
            "ratio_threshold",
            "min_horizontal_disparity_px",
            "max_horizontal_disparity_px",
            "broad_vertical_prefilter_px",
            "ransac_reprojection_threshold_px",
            "ransac_confidence",
        ):
            _require(math.isfinite(float(getattr(self, name))), f"{name} must be finite")
        _require(0.0 < self.ratio_threshold < 1.0, "ratio_threshold must be in (0,1)")
        _require(0.0 <= self.min_horizontal_disparity_px < self.max_horizontal_disparity_px, "horizontal disparity bounds are invalid")
        _require(self.broad_vertical_prefilter_px > 0.0, "broad vertical prefilter must be positive")
        _require(self.ransac_reprojection_threshold_px > 0.0, "RANSAC threshold must be positive")
        _require(0.0 < self.ransac_confidence < 1.0, "RANSAC confidence must be in (0,1)")


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    split: str
    manifest_index: int
    sequence_id: str
    frame_id: int
    timestamp: float
    left_path: Path
    right_path: Path
    metadata_path: Path | None
    metadata_sha256: str | None
    image_size_wh: tuple[int, int] | None
    left_cy_px: float
    right_cy_px: float
    projection_left_cy_px: float | None
    projection_right_cy_px: float | None

    @property
    def metadata_right_minus_left_cy_px(self) -> float:
        return self.right_cy_px - self.left_cy_px


@dataclass(frozen=True, slots=True)
class FrameMeasurement:
    """One frame's compact evidence plus inlier values used for aggregation."""

    split: str
    manifest_index: int
    sequence_id: str
    frame_id: int
    timestamp: float
    metadata_right_minus_left_cy_px: float
    left_path: str
    right_path: str
    left_sha256: str
    right_sha256: str
    metadata_path: str | None
    metadata_sha256_verified: str | None
    image_size_wh: tuple[int, int]
    left_keypoints: int
    right_keypoints: int
    knn_pairs: int
    ratio_matches: int
    positive_horizontal_matches: int
    plausible_prefilter_matches: int
    ransac_inliers: int
    covered: bool
    failure_reasons: tuple[str, ...]
    dy_right_minus_left_inliers_px: tuple[float, ...]
    horizontal_disparity_left_minus_right_inliers_px: tuple[float, ...]
    fundamental_matrix: tuple[tuple[float, float, float], ...] | None

    def to_report(self) -> dict[str, Any]:
        dy_summary = summarize_displacements(self.dy_right_minus_left_inliers_px)
        disparity_summary = summarize_scalar_values(
            self.horizontal_disparity_left_minus_right_inliers_px
        )
        return {
            "split": self.split,
            "manifest_index": self.manifest_index,
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source": {
                "left_path": self.left_path,
                "right_path": self.right_path,
                "left_sha256": self.left_sha256,
                "right_sha256": self.right_sha256,
                "metadata_path": self.metadata_path,
                "metadata_sha256_verified": self.metadata_sha256_verified,
                "image_size_wh": list(self.image_size_wh),
            },
            "metadata_right_minus_left_cy_px": self.metadata_right_minus_left_cy_px,
            "counts": {
                "left_keypoints": self.left_keypoints,
                "right_keypoints": self.right_keypoints,
                "knn_pairs": self.knn_pairs,
                "ratio_matches": self.ratio_matches,
                "positive_horizontal_matches": self.positive_horizontal_matches,
                "plausible_prefilter_matches": self.plausible_prefilter_matches,
                "ransac_inliers": self.ransac_inliers,
            },
            "covered": self.covered,
            "failure_reasons": list(self.failure_reasons),
            "dy_right_minus_left_px": dy_summary,
            "horizontal_disparity_left_minus_right_px": disparity_summary,
            "observed_minus_metadata_cy_delta_px": (
                None
                if dy_summary is None
                else dy_summary["signed"]["p50"]
                - self.metadata_right_minus_left_cy_px
            ),
            "fundamental_matrix": (
                None
                if self.fundamental_matrix is None
                else [list(row) for row in self.fundamental_matrix]
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpipolarAuditError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_constant(value: str) -> None:
    raise EpipolarAuditError(f"strict JSON contains {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, name: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_strict_constant,
            object_pairs_hook=_strict_object,
        )
    except EpipolarAuditError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise EpipolarAuditError(f"cannot parse strict JSON {name}: {exc}") from exc


def _finite_tree(value: Any, name: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains non-finite data")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
        return
    raise EpipolarAuditError(f"{name} contains unsupported {type(value).__name__}")


def deterministic_balanced_positions(count: int, sample_count: int) -> tuple[int, ...]:
    """Return unique, endpoint-inclusive, evenly spaced integer positions."""

    _require(_is_int(count) and count > 0, "count must be a positive integer")
    _require(_is_int(sample_count) and sample_count > 0, "sample_count must be positive")
    if sample_count >= count:
        return tuple(range(count))
    if sample_count == 1:
        return (count // 2,)
    denominator = sample_count - 1
    positions = tuple(
        (index * (count - 1) + denominator // 2) // denominator
        for index in range(sample_count)
    )
    _require(len(set(positions)) == sample_count, "balanced sampling produced duplicates")
    _require(positions[0] == 0 and positions[-1] == count - 1, "balanced sampling lost endpoints")
    return positions


def _matrix_cy(value: Any, name: str, columns: int) -> float:
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == columns for row in value),
        f"{name} must be 3x{columns}",
    )
    _finite_tree(value, name)
    return _finite(value[1][2], f"{name}[1][2]")


def _load_manifest(path: Path, split: str) -> tuple[list[ManifestFrame], str, int]:
    _require(path.is_file(), f"{split} manifest is missing: {path}")
    payload = path.read_bytes()
    _require(payload.endswith(b"\n"), f"{split} manifest has an unterminated final line")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpipolarAuditError(f"{split} manifest is not UTF-8: {exc}") from exc
    records: list[ManifestFrame] = []
    seen: set[tuple[str, int]] = set()
    last_timestamp: dict[str, float] = {}
    for index, line in enumerate(text.splitlines()):
        _require(bool(line.strip()), f"{split} manifest line {index + 1} is blank")
        row = _strict_json_loads(line, f"{split} manifest:{index + 1}")
        _require(isinstance(row, dict), f"{split} manifest row {index} is not an object")
        _finite_tree(row, f"{split} manifest[{index}]")
        sequence_id = row.get("sequence_id")
        frame_id = row.get("frame_id")
        timestamp = _finite(row.get("timestamp"), f"{split}[{index}].timestamp")
        _require(isinstance(sequence_id, str) and sequence_id, f"{split} row {index} sequence ID is invalid")
        _require(_is_int(frame_id), f"{split} row {index} frame ID is invalid")
        _require(row.get("rectified") is True, f"{split} row {index} is not declared rectified")
        target = (sequence_id, int(frame_id))
        _require(target not in seen, f"{split} manifest has duplicate {target}")
        seen.add(target)
        if sequence_id in last_timestamp:
            _require(timestamp > last_timestamp[sequence_id], f"{split} timestamps do not increase in {sequence_id}")
        last_timestamp[sequence_id] = timestamp
        left = Path(str(row.get("left_path", ""))).expanduser().resolve()
        right = Path(str(row.get("right_path", ""))).expanduser().resolve()
        _require(left.is_file() and right.is_file(), f"{split} stereo files are missing at row {index}")
        _require(left != right, f"{split} left/right paths are identical at row {index}")
        left_cy = _matrix_cy(row.get("K"), f"{split}[{index}].K", 3)
        # Spring's rectified training manifest stores one calibrated K for
        # the stereo pair; the right camera uses the same pixel intrinsics
        # (the baseline is carried separately).  Preserve strict behaviour
        # for every other dataset while accepting this explicit Spring
        # convention in the pixel audit.
        right_matrix = row.get("K_right")
        if right_matrix is None and str(row.get("dataset", "")).lower() == "spring":
            right_matrix = row.get("K")
        right_cy = _matrix_cy(right_matrix, f"{split}[{index}].K_right", 3)
        projection_left = row.get("P_left")
        projection_right = row.get("P_right")
        projection_left_cy = (
            None
            if projection_left is None
            else _matrix_cy(projection_left, f"{split}[{index}].P_left", 4)
        )
        projection_right_cy = (
            None
            if projection_right is None
            else _matrix_cy(projection_right, f"{split}[{index}].P_right", 4)
        )
        if projection_left_cy is not None:
            _require(projection_left_cy == left_cy, f"{split} K/P left cy mismatch at row {index}")
        if projection_right_cy is not None:
            _require(projection_right_cy == right_cy, f"{split} K/P right cy mismatch at row {index}")
        metadata_path_value = row.get("metadata_path")
        metadata_hash = row.get("metadata_sha256")
        metadata_path = (
            None
            if metadata_path_value is None
            else Path(str(metadata_path_value)).expanduser().resolve()
        )
        if metadata_path is not None:
            _require(metadata_path.is_file(), f"{split} metadata file is missing at row {index}")
            _require(
                isinstance(metadata_hash, str)
                and SHA256_PATTERN.fullmatch(metadata_hash) is not None,
                f"{split} metadata SHA is invalid at row {index}",
            )
        image_size = row.get("image_size_wh")
        if image_size is not None:
            _require(
                isinstance(image_size, list)
                and len(image_size) == 2
                and all(_is_int(item) and item > 0 for item in image_size),
                f"{split} image_size_wh is invalid at row {index}",
            )
            image_size_tuple: tuple[int, int] | None = (image_size[0], image_size[1])
        else:
            image_size_tuple = None
        records.append(
            ManifestFrame(
                split=split,
                manifest_index=index,
                sequence_id=sequence_id,
                frame_id=int(frame_id),
                timestamp=timestamp,
                left_path=left,
                right_path=right,
                metadata_path=metadata_path,
                metadata_sha256=metadata_hash,
                image_size_wh=image_size_tuple,
                left_cy_px=left_cy,
                right_cy_px=right_cy,
                projection_left_cy_px=projection_left_cy,
                projection_right_cy_px=projection_right_cy,
            )
        )
    _require(bool(records), f"{split} manifest is empty")
    return records, hashlib.sha256(payload).hexdigest(), len(payload)


def _balanced_sample(records: Sequence[ManifestFrame], samples_per_sequence: int) -> list[ManifestFrame]:
    grouped: dict[str, list[ManifestFrame]] = defaultdict(list)
    sequence_order: list[str] = []
    for record in records:
        if record.sequence_id not in grouped:
            sequence_order.append(record.sequence_id)
        grouped[record.sequence_id].append(record)
    selected: list[ManifestFrame] = []
    for sequence_id in sequence_order:
        candidates = grouped[sequence_id]
        for position in deterministic_balanced_positions(len(candidates), samples_per_sequence):
            selected.append(candidates[position])
    return selected


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values), "cannot compute a percentile of no values")
    _require(0.0 <= fraction <= 1.0, "percentile fraction is out of range")
    ordered = sorted(float(value) for value in values)
    _require(all(math.isfinite(value) for value in ordered), "percentile input is non-finite")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_scalar_values(values: Sequence[float]) -> dict[str, Any] | None:
    """Pure strict-JSON summary used by unit tests and the OpenCV runner."""

    if not values:
        return None
    numeric = tuple(float(value) for value in values)
    _require(all(math.isfinite(value) for value in numeric), "summary input is non-finite")
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "p05": _percentile(numeric, 0.05),
        "p25": _percentile(numeric, 0.25),
        "p50": _percentile(numeric, 0.50),
        "p75": _percentile(numeric, 0.75),
        "p95": _percentile(numeric, 0.95),
        "maximum": max(numeric),
    }


def summarize_displacements(values: Sequence[float]) -> dict[str, Any] | None:
    """Return signed and absolute dy summaries without emitting NaN."""

    if not values:
        return None
    signed = summarize_scalar_values(values)
    absolute = summarize_scalar_values([abs(value) for value in values])
    assert signed is not None and absolute is not None
    return {"signed": signed, "absolute": absolute}


def aggregate_measurements(measurements: Sequence[FrameMeasurement]) -> dict[str, Any]:
    """Aggregate one sequence or the global frame set without OpenCV."""

    _require(bool(measurements), "cannot aggregate no frame measurements")
    covered = [item for item in measurements if item.covered]
    dy = [value for item in covered for value in item.dy_right_minus_left_inliers_px]
    disparities = [
        value
        for item in covered
        for value in item.horizontal_disparity_left_minus_right_inliers_px
    ]
    metadata_deltas = [item.metadata_right_minus_left_cy_px for item in measurements]
    frame_medians = [
        _percentile(item.dy_right_minus_left_inliers_px, 0.5)
        for item in covered
        if item.dy_right_minus_left_inliers_px
    ]
    return {
        "sampled_frames": len(measurements),
        "covered_frames": len(covered),
        "uncovered_frames": len(measurements) - len(covered),
        "coverage_fraction": len(covered) / len(measurements),
        "counts": {
            "ratio_matches": sum(item.ratio_matches for item in measurements),
            "plausible_prefilter_matches": sum(
                item.plausible_prefilter_matches for item in measurements
            ),
            "ransac_inliers": sum(item.ransac_inliers for item in measurements),
        },
        "dy_right_minus_left_px": summarize_displacements(dy),
        "horizontal_disparity_left_minus_right_px": summarize_scalar_values(disparities),
        "frame_median_dy_right_minus_left_px": summarize_scalar_values(frame_medians),
        "metadata_right_minus_left_cy_px": summarize_scalar_values(metadata_deltas),
        "observed_minus_metadata_cy_delta_px": (
            None
            if not dy
            else _percentile(dy, 0.5) - _percentile(metadata_deltas, 0.5)
        ),
    }


def evaluate_rectification_contract(
    *,
    sequence_aggregates: Mapping[str, Mapping[str, Any]],
    global_aggregate: Mapping[str, Any],
    thresholds: RectificationThresholds,
) -> tuple[bool, list[dict[str, Any]]]:
    """Pure fail-closed threshold evaluation for all sequences and global data."""

    thresholds.validate()
    _require(bool(sequence_aggregates), "no sequence aggregates were supplied")
    checks: list[dict[str, Any]] = []
    scopes: list[tuple[str, Mapping[str, Any]]] = [
        *[(f"sequence:{name}", value) for name, value in sequence_aggregates.items()],
        ("global", global_aggregate),
    ]
    for scope, aggregate in scopes:
        coverage = _finite(aggregate.get("coverage_fraction"), f"{scope}.coverage")
        dy = aggregate.get("dy_right_minus_left_px")
        signed_median = None
        p95_absolute = None
        if isinstance(dy, Mapping):
            signed = dy.get("signed")
            absolute = dy.get("absolute")
            if isinstance(signed, Mapping) and isinstance(absolute, Mapping):
                signed_median = _finite(signed.get("p50"), f"{scope}.median_dy")
                p95_absolute = _finite(absolute.get("p95"), f"{scope}.p95_abs_dy")
        coverage_pass = coverage >= thresholds.min_frame_coverage_fraction
        median_pass = (
            signed_median is not None
            and abs(signed_median) <= thresholds.max_abs_median_dy_px
        )
        p95_pass = (
            p95_absolute is not None
            and p95_absolute <= thresholds.max_p95_abs_dy_px
        )
        checks.extend(
            (
                {
                    "scope": scope,
                    "metric": "frame_coverage_fraction",
                    "actual": coverage,
                    "operator": ">=",
                    "threshold": thresholds.min_frame_coverage_fraction,
                    "passed": coverage_pass,
                },
                {
                    "scope": scope,
                    "metric": "abs_median_dy_right_minus_left_px",
                    "actual": None if signed_median is None else abs(signed_median),
                    "operator": "<=",
                    "threshold": thresholds.max_abs_median_dy_px,
                    "passed": median_pass,
                },
                {
                    "scope": scope,
                    "metric": "p95_abs_dy_right_minus_left_px",
                    "actual": p95_absolute,
                    "operator": "<=",
                    "threshold": thresholds.max_p95_abs_dy_px,
                    "passed": p95_pass,
                },
            )
        )
    return all(bool(check["passed"]) for check in checks), checks


def _match_frame(
    frame: ManifestFrame,
    *,
    config: MatchingConfig,
    thresholds: RectificationThresholds,
    cv2: Any,
) -> FrameMeasurement:
    """Run deterministic SIFT, plausibility gates, and F-RANSAC for one pair."""

    import numpy as np  # lazy alongside OpenCV; unavailable in neither formal env

    left_sha = _sha256_file(frame.left_path)
    right_sha = _sha256_file(frame.right_path)
    metadata_verified: str | None = None
    if frame.metadata_path is not None:
        actual_metadata_sha = _sha256_file(frame.metadata_path)
        _require(
            actual_metadata_sha == frame.metadata_sha256,
            f"metadata SHA mismatch for {frame.metadata_path}",
        )
        metadata_verified = actual_metadata_sha
    left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
    _require(left is not None and right is not None, f"OpenCV cannot decode frame {frame.frame_id}")
    _require(left.ndim == 2 and right.ndim == 2, f"decoded stereo pair is not grayscale at {frame.frame_id}")
    _require(left.shape == right.shape, f"left/right image shapes differ at {frame.frame_id}")
    height, width = left.shape
    if frame.image_size_wh is not None:
        _require(frame.image_size_wh == (width, height), f"manifest image size mismatch at {frame.frame_id}")
    image_size = (width, height)

    cv2.setRNGSeed((config.seed + frame.manifest_index) & 0x7FFFFFFF)
    sift = cv2.SIFT_create(
        nfeatures=config.sift_nfeatures,
        contrastThreshold=config.sift_contrast_threshold,
    )
    left_keypoints, left_descriptors = sift.detectAndCompute(left, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(right, None)
    left_count = len(left_keypoints)
    right_count = len(right_keypoints)
    failure_reasons: list[str] = []
    knn_pairs: list[Any] = []
    ratio_matches: list[Any] = []
    positive_count = 0
    plausible_left = np.empty((0, 2), dtype=np.float32)
    plausible_right = np.empty((0, 2), dtype=np.float32)
    if left_descriptors is None or right_descriptors is None:
        failure_reasons.append("missing_sift_descriptors")
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        knn_pairs = [pair for pair in raw_pairs if len(pair) == 2]
        ratio_matches = [
            pair[0]
            for pair in knn_pairs
            if pair[0].distance < config.ratio_threshold * pair[1].distance
        ]
        if ratio_matches:
            points_left = np.float32(
                [left_keypoints[match.queryIdx].pt for match in ratio_matches]
            )
            points_right = np.float32(
                [right_keypoints[match.trainIdx].pt for match in ratio_matches]
            )
            disparity = points_left[:, 0] - points_right[:, 0]
            dy = points_right[:, 1] - points_left[:, 1]
            positive = (
                (disparity >= config.min_horizontal_disparity_px)
                & (disparity <= config.max_horizontal_disparity_px)
            )
            positive_count = int(positive.sum())
            plausible = positive & (np.abs(dy) <= config.broad_vertical_prefilter_px)
            plausible_left = points_left[plausible]
            plausible_right = points_right[plausible]

    if len(ratio_matches) < thresholds.min_ratio_matches_per_frame:
        failure_reasons.append("insufficient_ratio_matches")
    if len(plausible_left) < thresholds.min_plausible_matches_per_frame:
        failure_reasons.append("insufficient_plausible_matches")
    inlier_mask = np.zeros(len(plausible_left), dtype=bool)
    fundamental: tuple[tuple[float, float, float], ...] | None = None
    if len(plausible_left) >= 8:
        cv2.setRNGSeed((config.seed + frame.manifest_index) & 0x7FFFFFFF)
        matrix, mask = cv2.findFundamentalMat(
            plausible_left,
            plausible_right,
            cv2.FM_RANSAC,
            config.ransac_reprojection_threshold_px,
            config.ransac_confidence,
            config.ransac_max_iterations,
        )
        if matrix is None or mask is None or matrix.shape != (3, 3):
            failure_reasons.append("fundamental_matrix_estimation_failed")
        else:
            matrix64 = np.asarray(matrix, dtype=np.float64)
            _require(bool(np.isfinite(matrix64).all()), "fundamental matrix is non-finite")
            inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
            _require(len(inlier_mask) == len(plausible_left), "RANSAC mask length mismatch")
            fundamental = tuple(
                tuple(float(value) for value in row) for row in matrix64.tolist()
            )
    else:
        failure_reasons.append("too_few_points_for_fundamental_matrix")
    inliers = int(inlier_mask.sum())
    if inliers < thresholds.min_ransac_inliers_per_frame:
        failure_reasons.append("insufficient_ransac_inliers")
    covered = (
        len(ratio_matches) >= thresholds.min_ratio_matches_per_frame
        and len(plausible_left) >= thresholds.min_plausible_matches_per_frame
        and inliers >= thresholds.min_ransac_inliers_per_frame
        and fundamental is not None
    )
    inlier_left = plausible_left[inlier_mask]
    inlier_right = plausible_right[inlier_mask]
    dy_inliers = inlier_right[:, 1] - inlier_left[:, 1]
    disparity_inliers = inlier_left[:, 0] - inlier_right[:, 0]
    return FrameMeasurement(
        split=frame.split,
        manifest_index=frame.manifest_index,
        sequence_id=frame.sequence_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        metadata_right_minus_left_cy_px=frame.metadata_right_minus_left_cy_px,
        left_path=str(frame.left_path),
        right_path=str(frame.right_path),
        left_sha256=left_sha,
        right_sha256=right_sha,
        metadata_path=None if frame.metadata_path is None else str(frame.metadata_path),
        metadata_sha256_verified=metadata_verified,
        image_size_wh=image_size,
        left_keypoints=left_count,
        right_keypoints=right_count,
        knn_pairs=len(knn_pairs),
        ratio_matches=len(ratio_matches),
        positive_horizontal_matches=positive_count,
        plausible_prefilter_matches=len(plausible_left),
        ransac_inliers=inliers,
        covered=covered,
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
        dy_right_minus_left_inliers_px=tuple(float(value) for value in dy_inliers),
        horizontal_disparity_left_minus_right_inliers_px=tuple(
            float(value) for value in disparity_inliers
        ),
        fundamental_matrix=fundamental,
    )


def run_epipolar_audit(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    config: MatchingConfig,
    thresholds: RectificationThresholds,
    command: str,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Run the formal audit; imports OpenCV only when this function executes."""

    config.validate()
    thresholds.validate()
    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - formal env requirement
            raise EpipolarAuditError(
                "OpenCV is required for image matching; run this tool in env-ffs"
            ) from exc
    cv2 = cv2_module
    _require(hasattr(cv2, "SIFT_create"), "OpenCV build does not provide SIFT")
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    train_path = Path(train_manifest).expanduser().resolve()
    validation_path = Path(validation_manifest).expanduser().resolve()
    train_records, train_sha, train_bytes = _load_manifest(train_path, "train")
    validation_records, validation_sha, validation_bytes = _load_manifest(
        validation_path, "validation"
    )
    train_sequences = {record.sequence_id for record in train_records}
    validation_sequences = {record.sequence_id for record in validation_records}
    overlap = sorted(train_sequences.intersection(validation_sequences))
    _require(not overlap, f"train/validation sequences overlap: {overlap}")
    sampled = [
        *_balanced_sample(train_records, config.samples_per_sequence),
        *_balanced_sample(validation_records, config.samples_per_sequence),
    ]
    measurements = [
        _match_frame(frame, config=config, thresholds=thresholds, cv2=cv2)
        for frame in sampled
    ]
    grouped: dict[str, list[FrameMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[f"{measurement.split}:{measurement.sequence_id}"].append(measurement)
    sequence_aggregates = {
        key: aggregate_measurements(values) for key, values in grouped.items()
    }
    global_aggregate = aggregate_measurements(measurements)
    passed, checks = evaluate_rectification_contract(
        sequence_aggregates=sequence_aggregates,
        global_aggregate=global_aggregate,
        thresholds=thresholds,
    )
    sample_identity = [
        {
            "split": item.split,
            "manifest_index": item.manifest_index,
            "sequence_id": item.sequence_id,
            "frame_id": item.frame_id,
            "left_sha256": item.left_sha256,
            "right_sha256": item.right_sha256,
        }
        for item in measurements
    ]
    sample_digest = hashlib.sha256(
        json.dumps(sample_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    global_dy = global_aggregate["dy_right_minus_left_px"]
    global_metadata = global_aggregate["metadata_right_minus_left_cy_px"]
    _require(
        isinstance(global_dy, Mapping)
        and isinstance(global_dy.get("signed"), Mapping)
        and isinstance(global_metadata, Mapping),
        "global pixel/metadata displacement evidence is missing",
    )
    observed_median_dy = _finite(
        global_dy["signed"].get("p50"), "global observed median dy"
    )
    metadata_median_delta = _finite(
        global_metadata.get("p50"), "global metadata median cy delta"
    )
    metadata_disagreement = observed_median_dy - metadata_median_delta
    metadata_consistent = (
        abs(metadata_disagreement) <= thresholds.max_abs_median_dy_px
    )
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS" if passed else "FAIL",
        "published_contract": RECTIFICATION_CONTRACT if passed else None,
        "claim_scope": "engineering input-coordinate contract only; not a paper result or accuracy claim",
        "read_only": True,
        "algorithm": {
            "feature": "SIFT",
            "matching": "L2 KNN k=2 plus Lowe ratio test",
            "plausibility": "positive bounded left_x-right_x disparity plus broad absolute vertical prefilter",
            "robust_geometry": "fundamental matrix cv2.FM_RANSAC",
            "dy_definition": "right_y-left_y pixels",
            "horizontal_disparity_definition": "left_x-right_x pixels",
            "opencv_version": str(cv2.__version__),
            "opencv_threads": 1,
            "opencl_enabled": False,
            "gpu_used": False,
        },
        "config": asdict(config),
        "thresholds": asdict(thresholds),
        "manifests": {
            "train": {
                "path": str(train_path),
                "sha256": train_sha,
                "byte_size": train_bytes,
                "record_count": len(train_records),
                "sequence_count": len(train_sequences),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": validation_sha,
                "byte_size": validation_bytes,
                "record_count": len(validation_records),
                "sequence_count": len(validation_sequences),
            },
            "train_validation_sequence_disjoint": True,
        },
        "sampling": {
            "policy": "deterministic endpoint-inclusive balanced positions independently per split/sequence",
            "sampled_frames": len(measurements),
            "sample_identity_sha256": sample_digest,
            "sampled_manifest_indices": {
                key: [item.manifest_index for item in values]
                for key, values in grouped.items()
            },
        },
        "per_sequence": sequence_aggregates,
        "global": global_aggregate,
        "metadata_vs_pixels": {
            "conclusion": (
                "CONSISTENT_WITH_AUDITED_PIXEL_COORDINATES"
                if metadata_consistent
                else "INCONSISTENT_WITH_AUDITED_PIXEL_COORDINATES"
            ),
            "metadata_median_right_minus_left_cy_px": metadata_median_delta,
            "observed_median_right_minus_left_y_px": observed_median_dy,
            "observed_minus_metadata_px": metadata_disagreement,
            "absolute_disagreement_px": abs(metadata_disagreement),
            "metadata_consistent_with_pixels_within_contract_median_tolerance": metadata_consistent,
        },
        "threshold_checks": checks,
        "frames": [item.to_report() for item in measurements],
        "metadata_interpretation": {
            "metadata_delta": "K_right.cy-K_left.cy (and matching P_right/P_left cy)",
            "observed_delta": "RANSAC-inlier median of right_y-left_y",
            "policy": "pixel correspondence evidence owns the same-row contract; inconsistent calibration cy is not applied as an image shift",
        },
        "reproduction": {
            "command": command,
            "deterministic_seed": config.seed,
        },
    }
    _finite_tree(report, "audit report")
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sift-nfeatures", type=int, default=4096)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.02)
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--min-horizontal-disparity-px", type=float, default=0.25)
    parser.add_argument("--max-horizontal-disparity-px", type=float, default=512.0)
    parser.add_argument("--broad-vertical-prefilter-px", type=float, default=32.0)
    parser.add_argument("--ransac-reprojection-threshold-px", type=float, default=1.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iterations", type=int, default=10_000)
    parser.add_argument("--min-ratio-matches-per-frame", type=int, default=64)
    parser.add_argument("--min-plausible-matches-per-frame", type=int, default=48)
    parser.add_argument("--min-ransac-inliers-per-frame", type=int, default=32)
    parser.add_argument("--min-frame-coverage-fraction", type=float, default=0.95)
    parser.add_argument("--max-abs-median-dy-px", type=float, default=1.25)
    parser.add_argument("--max-p95-abs-dy-px", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_path = args.train_manifest.expanduser().resolve()
    validation_path = args.validation_manifest.expanduser().resolve()
    output = args.json_out.expanduser().resolve()
    command_argv = list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv])
    command = shlex.join([sys.executable, *command_argv])
    try:
        _require(
            output.parent != train_path.parent and output.parent != validation_path.parent,
            "--json-out must be outside the manifest/cache directory",
        )
        report = run_epipolar_audit(
            train_manifest=train_path,
            validation_manifest=validation_path,
            config=MatchingConfig(
                samples_per_sequence=args.samples_per_sequence,
                seed=args.seed,
                sift_nfeatures=args.sift_nfeatures,
                sift_contrast_threshold=args.sift_contrast_threshold,
                ratio_threshold=args.ratio_threshold,
                min_horizontal_disparity_px=args.min_horizontal_disparity_px,
                max_horizontal_disparity_px=args.max_horizontal_disparity_px,
                broad_vertical_prefilter_px=args.broad_vertical_prefilter_px,
                ransac_reprojection_threshold_px=args.ransac_reprojection_threshold_px,
                ransac_confidence=args.ransac_confidence,
                ransac_max_iterations=args.ransac_max_iterations,
            ),
            thresholds=RectificationThresholds(
                min_ratio_matches_per_frame=args.min_ratio_matches_per_frame,
                min_plausible_matches_per_frame=args.min_plausible_matches_per_frame,
                min_ransac_inliers_per_frame=args.min_ransac_inliers_per_frame,
                min_frame_coverage_fraction=args.min_frame_coverage_fraction,
                max_abs_median_dy_px=args.max_abs_median_dy_px,
                max_p95_abs_dy_px=args.max_p95_abs_dy_px,
            ),
            command=command,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"audit:epipolar-rectification: {report['status']}")
        print(f"contract: {report['published_contract']}")
        print(f"receipt: {output}")
        return 0 if report["status"] == "PASS" else 2
    except (OSError, EpipolarAuditError) as exc:
        failure = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "component": "pixel-level-epipolar-rectification-audit",
            "status": "FAIL",
            "published_contract": None,
            "error": str(exc),
        }
        sys.stderr.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
