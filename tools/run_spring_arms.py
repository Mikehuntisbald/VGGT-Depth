#!/usr/bin/env python3
"""Run the seven Spring screening arms with explicit pose/depth lineage.

This runner is deliberately conservative.  It prepares sequence-disjoint
manifests and the frozen-backbone caches, then launches the existing training
and evaluation entry points in the requested order.  A missing archive,
checkpoint, cache, or Stage-C audit produces a machine-readable ``BLOCKED``
row; no placeholder metric is written and no missing result is reported as a
pass.

The seven arms are:

``S0`` LR FFS bilinear, ``S1`` T1 spatial, ``S2`` T3 history (GT pose, no
VGGT depth), ``S3`` T3 top-K (GT pose, no depth), ``S4`` S3 plus actual VGGT
depth (GT pose), ``S5`` S4 with VGGT pose, and ``S6`` S5 plus HR epipolar
refinement.  S4 is materialized by :mod:`tools.override_spring_pose`, which
copies S5's depth/alignment tensors and replaces only temporal pose.  S6 is
run through the dedicated bounded Spring Stage-C adapters; the canonical
Stage-C evaluator remains bound to its separate 244/240/238 holdout.

The default seed is intentionally fixed at 42, as required by the screening
protocol.  ``--limit`` creates bounded manifests (rather than silently
claiming full-corpus coverage), and ``--steps`` overrides the optimizer steps
for all trainable arms.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SEED = 42
ARM_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
REQUIRED_METRICS = (
    "overall_epe",
    "overall_1px",
    "high_detail_epe",
    "high_detail_1px",
    "low_detail_epe",
    "matched_epe",
    "unmatched_completion_1px",
    "unmatched_completion_2px",
    "rigid_temporal_residual_error",
    "non_rigid_temporal_residual_error",
    "boundary_epe",
    "ffs_trusted_measurement_error",
    "negative_rate",
    "zero_rate",
    "invalid_rate",
)
TOPK_DIAGNOSTICS = (
    "age_2_survival_rate",
    "unique_age_fraction",
    "phase_variance",
    "candidate_depth_spread",
    "attention_entropy",
    "gain_by_fractional_phase_bucket",
    "gain_by_camera_motion_bucket",
)


class SpringRunnerError(RuntimeError):
    """A malformed runner input or unsafe lineage condition."""


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    config: str
    stage: str
    pose_source: str
    use_vggt_depth: bool
    derived_kind: str | None
    init_arm: str | None
    requires_vggt: bool


ARM_SPECS: dict[str, ArmSpec] = {
    "S0": ArmSpec("S0", "configs/spring/S0.yaml", "baseline", "none", False, None, None, False),
    "S1": ArmSpec("S1", "configs/spring/S1.yaml", "spatial", "gt", False, None, None, False),
    "S2": ArmSpec("S2", "configs/spring/S2.yaml", "temporal", "gt", False, "gt_no_depth", "S1", False),
    "S3": ArmSpec("S3", "configs/spring/S3.yaml", "temporal", "gt", False, "gt_no_depth", "spatial_v2_base", False),
    "S4": ArmSpec("S4", "configs/spring/S4.yaml", "temporal", "gt", True, "gt_pose_vggt_depth", "spatial_v2_base", True),
    "S5": ArmSpec("S5", "configs/spring/S5.yaml", "temporal", "vggt", True, "vggt_pose_vggt_depth", "spatial_v2_base", True),
    "S6": ArmSpec("S6", "configs/spring/S6.yaml", "epipolar", "vggt", True, "vggt_pose_vggt_depth", "S5", True),
}


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpringRunnerError(f"cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise SpringRunnerError(f"JSON file must contain an object: {path}")
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise SpringRunnerError(f"blank manifest row: {path}:{line_number}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SpringRunnerError(f"invalid manifest JSON: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise SpringRunnerError(f"manifest row is not an object: {path}:{line_number}")
            for key in ("sequence_id", "frame_id", "timestamp", "left_path", "right_path"):
                if key not in value:
                    raise SpringRunnerError(f"manifest row lacks {key!r}: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise SpringRunnerError(f"manifest is empty: {path}")
    return rows


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SpringRunnerError(f"refusing to write an empty manifest: {path}")
    _atomic_jsonl(path, rows)


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _safe(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("._")
    if not text:
        raise SpringRunnerError(f"invalid path component: {value!r}")
    return text


def _slice_rows(rows: Sequence[Mapping[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return [dict(row) for row in rows]
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = [dict(row) for row in rows[:limit]]
    if not selected:
        raise SpringRunnerError("limit selected no manifest rows")
    return selected


def _sequence_split(
    rows: Sequence[Mapping[str, Any]], *, seed: int, val_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    sequence_ids = sorted({str(row["sequence_id"]) for row in rows})
    if len(sequence_ids) < 2:
        raise SpringRunnerError("Spring train/validation sequence split needs at least two sequences")
    rng = random.Random(seed)
    shuffled = list(sequence_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(sequence_ids) * val_fraction)))
    val_count = min(val_count, len(sequence_ids) - 1)
    val_set = set(shuffled[:val_count])
    train_set = set(sequence_ids) - val_set
    train_rows = [dict(row) for row in rows if str(row["sequence_id"]) in train_set]
    val_rows = [dict(row) for row in rows if str(row["sequence_id"]) in val_set]
    if not train_rows or not val_rows:
        raise SpringRunnerError("sequence split produced an empty train or validation manifest")
    return train_rows, val_rows, sorted(train_set), sorted(val_set)


def _check_disjoint(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> None:
    train_sequences = {str(row["sequence_id"]) for row in train_rows}
    val_sequences = {str(row["sequence_id"]) for row in val_rows}
    overlap = train_sequences & val_sequences
    if overlap:
        raise SpringRunnerError(f"train/validation sequence overlap: {sorted(overlap)}")


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_command(
    command: Sequence[str], *, cwd: Path, log_path: Path, dry_run: bool = False
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "command": [str(value) for value in command],
        "command_text": _command_text(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "status": "PLANNED" if dry_run else "RUNNING",
        "started_at_utc": _utc_now(),
    }
    if dry_run:
        return record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                list(command),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record["returncode"] = int(process.returncode)
        record["status"] = "COMPLETE" if process.returncode == 0 else "FAILED"
    except OSError as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nrunner exception: {type(exc).__name__}: {exc}\n")
        record["returncode"] = None
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.perf_counter() - started
    record["finished_at_utc"] = _utc_now()
    return record


def _file_requirement(path: Path, label: str) -> dict[str, Any]:
    item: dict[str, Any] = {"label": label, "path": str(path), "exists": path.is_file()}
    if path.is_file():
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = _sha256(path)
    else:
        item["reason"] = "missing_file"
    return item


def _dir_requirement(path: Path, label: str) -> dict[str, Any]:
    receipt = path / "run_receipt.json"
    item: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.is_dir(),
        "receipt": str(receipt),
        "receipt_exists": receipt.is_file(),
    }
    if receipt.is_file():
        try:
            item["receipt_sha256"] = _sha256(receipt)
        except OSError as exc:
            item["receipt_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _manifest_missing_paths(rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for key in ("left_path", "right_path"):
            raw = Path(str(row[key])).expanduser()
            path = raw if raw.is_absolute() else manifest_path.parent / raw
            if not path.is_file():
                missing.append(str(path.resolve()))
                if len(missing) >= 20:
                    return missing
    return missing


def _manifest_missing_gt(rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> list[str]:
    missing: list[str] = []
    for row in rows:
        raw_value = row.get("gt_disparity_path")
        if not raw_value:
            missing.append(f"{row.get('sequence_id')}/{row.get('frame_id')}:gt_disparity_path")
            continue
        raw = Path(str(raw_value)).expanduser()
        path = raw if raw.is_absolute() else manifest_path.parent / raw
        if not path.is_file():
            missing.append(str(path.resolve()))
            if len(missing) >= 20:
                return missing
    return missing


def _receipt_manifest_matches(root: Path, manifest_path: Path) -> bool:
    receipt_path = root / "run_receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
    except SpringRunnerError:
        return False
    expected = _sha256(manifest_path)
    for key in ("manifest_sha256",):
        if receipt.get(key) == expected:
            return True
    for container_name in ("inputs", "source"):
        container = receipt.get(container_name)
        if isinstance(container, Mapping) and container.get("manifest_sha256") == expected:
            return True
    return False


def _expected_vggt_targets(
    rows: Sequence[Mapping[str, Any]], *, context_pairs: int = 5
) -> list[tuple[int, str, int, float]]:
    """Return the exact raw-VGGT endpoint inventory for a frame manifest.

    ``cache_vggt.py`` emits one record for each endpoint that has a complete
    causal context (five stereo pairs).  A plain manifest-SHA check is not
    sufficient here: a crashed/subset producer can leave a receipt bound to
    the right manifest while containing only a prefix of the endpoints.  Keep
    this calculation local and dependency-free so the runner can fail closed
    before launching an expensive train/eval subprocess.
    """

    if context_pairs <= 0:
        raise ValueError("context_pairs must be positive")
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for manifest_index, row in enumerate(rows):
        sequence_id = str(row.get("sequence_id", ""))
        if not sequence_id:
            raise ValueError("manifest row has an empty sequence_id")
        grouped.setdefault(sequence_id, []).append((manifest_index, row))
    targets: list[tuple[int, str, int, float]] = []
    for sequence_id, indexed in grouped.items():
        # The manifest builder preserves sequence order.  Do not sort here:
        # an out-of-order input must be rejected by the producer, rather than
        # silently changing the causal contract in the status checker.
        for position, (manifest_index, row) in enumerate(indexed):
            if position < context_pairs - 1:
                continue
            try:
                frame_id = int(row["frame_id"])
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed manifest endpoint row at index {manifest_index}"
                ) from exc
            targets.append((manifest_index, sequence_id, frame_id, timestamp))
    targets.sort(key=lambda value: value[0])
    return targets


def _expected_vggt_output_grid(
    rows: Sequence[Mapping[str, Any]], *, scale: int = 2
) -> tuple[int, int] | None:
    """Return the Spring x2 dense grid expected by downstream geometry."""

    if scale <= 0 or not rows:
        return None
    shapes: set[tuple[int, int]] = set()
    for row in rows:
        value = row.get("image_shape_hw")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            height, width = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
        if height <= 0 or width <= 0 or height % scale or width % scale:
            return None
        shapes.add((height // scale, width // scale))
    if len(shapes) != 1:
        return None
    return next(iter(shapes))


def _cache_row_target(
    row: Mapping[str, Any], *, row_index: int
) -> tuple[int, str, int, float, int, Path]:
    """Parse the identity fields shared by raw/derived cache manifests."""

    try:
        selection_index = int(row["selection_index"])
        target_manifest_index = int(row["target_manifest_index"])
        sequence_id = str(row["sequence_id"])
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        raw_cache_path = row.get("cache_path", row.get("vggt_cache_path"))
        if not isinstance(raw_cache_path, (str, Path)) or not str(raw_cache_path).strip():
            raise ValueError("cache path is empty")
        cache_path = Path(str(raw_cache_path)).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed cache-manifest row {row_index}") from exc
    if selection_index < 0 or target_manifest_index < 0 or not sequence_id:
        raise ValueError(f"invalid cache-manifest indices at row {row_index}")
    return (
        selection_index,
        sequence_id,
        frame_id,
        timestamp,
        target_manifest_index,
        cache_path,
    )


def _strict_vggt_cache_matches(
    root: Path,
    manifest_path: Path,
    *,
    expected_targets: Sequence[tuple[int, str, int, float]] | None = None,
    expected_output_grid: tuple[int, int] | None = None,
) -> bool:
    """Check complete raw VGGT endpoint coverage, not only manifest SHA."""

    receipt_path = root / "run_receipt.json"
    manifest_cache_path = root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not manifest_cache_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
        expected_sha = _sha256(manifest_path)
        if receipt.get("manifest_sha256") != expected_sha:
            return False
        receipt_manifest = receipt.get("manifest")
        if (
            isinstance(receipt_manifest, str)
            and Path(receipt_manifest).expanduser().resolve() != manifest_path.resolve()
        ):
            return False
        if expected_targets is None:
            expected_targets = _expected_vggt_targets(_read_manifest(manifest_path))
        if expected_output_grid is not None:
            config = receipt.get("config")
            recorded_grid = config.get("output_grid_hw") if isinstance(config, Mapping) else None
            if recorded_grid != [int(expected_output_grid[0]), int(expected_output_grid[1])]:
                return False
        expected_count = len(expected_targets)
        if expected_count <= 0:
            return False
        for key in ("available_windows", "selected_windows"):
            if int(receipt.get(key, -1)) != expected_count:
                return False
        if (
            int(receipt.get("written_records", -1))
            + int(receipt.get("reused_records", -1))
            != expected_count
        ):
            return False
        parsed: list[tuple[int, str, int, float, int, Path]] = []
        with manifest_cache_path.open("r", encoding="utf-8") as handle:
            for row_index, raw in enumerate(handle):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    return False
                parsed.append(_cache_row_target(value, row_index=row_index))
        if len(parsed) != expected_count:
            return False
        if [item[0] for item in parsed] != list(range(expected_count)):
            return False
        actual_targets = {
            (item[4], item[1], item[2], item[3])
            for item in parsed
        }
        expected_set = set(expected_targets)
        if actual_targets != expected_set:
            return False
        for _selection, _sequence, _frame, _timestamp, _target, cache_path in parsed:
            try:
                cache_path.relative_to(root.resolve())
            except ValueError:
                return False
            if not cache_path.is_file():
                return False
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _strict_derived_cache_matches(
    root: Path,
    manifest_path: Path,
    *,
    expected_targets: Sequence[tuple[int, str, int, float]],
    pose_source: str | None = None,
    depth_source: str | None = None,
) -> bool:
    """Check sparse derived coverage and optional Spring pose/depth selectors."""

    receipt_path = root / "run_receipt.json"
    manifest_cache_path = root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not manifest_cache_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
        expected_sha = _sha256(manifest_path)
        if receipt.get("manifest_sha256") != expected_sha:
            return False
        receipt_manifest = receipt.get("manifest")
        if (
            isinstance(receipt_manifest, str)
            and Path(receipt_manifest).expanduser().resolve() != manifest_path.resolve()
        ):
            return False
        config = receipt.get("config")
        if not isinstance(config, Mapping):
            return False
        if pose_source is not None and config.get("pose_source") != pose_source:
            return False
        if depth_source is not None and config.get("depth_source") != depth_source:
            return False
        counts = receipt.get("counts")
        selection = receipt.get("selection")
        if not expected_targets:
            return False
        if (
            not isinstance(counts, Mapping)
            or int(counts.get("selected", -1)) != len(expected_targets)
        ):
            return False
        if (
            int(counts.get("written", -1)) + int(counts.get("reused", -1))
            != len(expected_targets)
        ):
            return False
        if pose_source == "Spring_GT_pose" and (
            int(counts.get("pose_valid", -1)) != len(expected_targets)
            or int(counts.get("pose_rejected", -1)) != 0
        ):
            return False
        if (
            isinstance(selection, Mapping)
            and int(selection.get("selected_windows", -1)) != len(expected_targets)
        ):
            return False
        parsed: list[tuple[int, str, int, float, int, Path]] = []
        with manifest_cache_path.open("r", encoding="utf-8") as handle:
            for row_index, raw in enumerate(handle):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    return False
                parsed.append(_cache_row_target(value, row_index=row_index))
        if len(parsed) != len(expected_targets):
            return False
        if [item[0] for item in parsed] != list(range(len(parsed))):
            return False
        actual_targets = {
            (item[4], item[1], item[2], item[3])
            for item in parsed
        }
        if actual_targets != set(expected_targets):
            return False
        for _selection, _sequence, _frame, _timestamp, _target, cache_path in parsed:
            try:
                cache_path.relative_to(root.resolve())
            except ValueError:
                return False
            if not cache_path.is_file():
                return False
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _receipt_manifest_matches_with_lineage(
    root: Path,
    manifest_path: Path,
    *,
    component: str,
    upstream_commit: str,
    identity_version: int | None = None,
) -> bool:
    """Check a manifest-bound receipt and its producer lineage.

    A generic manifest hash check is sufficient for frozen FFS/VGGT caches,
    whose cache identity already names the exact checkpoint.  Spring's dense
    GT teacher is dataset-level and historically used a manifest hash as its
    ``checkpoint_sha256``; after making that identity split-independent, an
    old receipt would otherwise look reusable and later fail checkpoint
    lineage validation.  This stricter helper lets the runner invalidate such
    receipts automatically without requiring ``--overwrite``.
    """

    receipt_path = root / "run_receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
    except SpringRunnerError:
        return False
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        return False
    if identity_version is not None and receipt.get("identity_version") != identity_version:
        return False
    identity = receipt.get("identity")
    if not isinstance(identity, Mapping):
        return False
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        return False
    if (
        config.get("identity_version") != identity_version
        or config.get("dataset") != "Spring"
        or config.get("teacher_source") != "Spring_GT"
    ):
        return False
    return (
        identity.get("component") == component
        and identity.get("upstream_commit") == upstream_commit
    )


def _default_paths(project_root: Path, args: argparse.Namespace) -> dict[str, Path]:
    output_root = _resolve_path(args.output_root)
    spring_root = _resolve_path(args.spring_root)
    # The official archives unpack under ``spring/train``.  Our download
    # workflow keeps camera sidecars in the small source tree and unpacks the
    # large image/disparity archives under ``data/spring``; prefer that merged
    # tree automatically when the user leaves the historical default path.
    candidates = (
        spring_root,
        spring_root / "spring",
        spring_root.parent / "data" / "spring",
        spring_root.parent / "spring",
    )
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("train/*/frame_left"), None) is not None:
            spring_root = candidate.resolve()
            break
    cache_root = _resolve_path(args.cache_root) if args.cache_root else output_root / "cache"
    return {
        "project_root": project_root,
        "spring_root": spring_root,
        "output_root": output_root,
        "cache_root": cache_root,
        "manifest_all": _resolve_path(args.manifest) if args.manifest else output_root / "manifests" / "all.jsonl",
        "manifest_train": _resolve_path(args.train_manifest) if args.train_manifest else output_root / "manifests" / "train.jsonl",
        "manifest_val": _resolve_path(args.validation_manifest) if args.validation_manifest else output_root / "manifests" / "validation.jsonl",
        "train_obs": _resolve_path(args.train_observation_cache_root) if args.train_observation_cache_root else cache_root / "train" / "observation",
        "train_teacher": _resolve_path(args.train_teacher_cache_root) if args.train_teacher_cache_root else cache_root / "train" / "teacher",
        "val_obs": _resolve_path(args.validation_observation_cache_root) if args.validation_observation_cache_root else cache_root / "validation" / "observation",
        "val_teacher": _resolve_path(args.validation_teacher_cache_root) if args.validation_teacher_cache_root else cache_root / "validation" / "teacher",
        "train_vggt": cache_root / "train" / "vggt",
        "val_vggt": cache_root / "validation" / "vggt",
        "train_gt_geom": cache_root / "train" / "derived_gt_no_depth",
        "val_gt_geom": cache_root / "validation" / "derived_gt_no_depth",
        "train_vggt_geom": cache_root / "train" / "derived_vggt_pose_depth",
        "val_vggt_geom": cache_root / "validation" / "derived_vggt_pose_depth",
        "train_gt_pose_depth": cache_root / "train" / "derived_gt_pose_vggt_depth",
        "val_gt_pose_depth": cache_root / "validation" / "derived_gt_pose_vggt_depth",
    }


def _resolve_checkpoint(project_root: Path, value: str | None, default_relative: str) -> Path:
    return _resolve_path(value if value else project_root / default_relative)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--spring-root",
        type=Path,
        default=project_root.parent / "spring_dataset" / "data" / "spring",
        help="directory containing extracted Spring data (spring/train/<sequence>)",
    )
    parser.add_argument("--output-root", type=Path, default=project_root / "runs" / "spring_seed42")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="prebuilt all-record manifest")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", "--val-manifest", dest="validation_manifest", type=Path)
    parser.add_argument("--train-observation-cache-root", type=Path)
    parser.add_argument("--train-teacher-cache-root", type=Path)
    parser.add_argument("--validation-observation-cache-root", "--val-observation-cache-root", dest="validation_observation_cache_root", type=Path)
    parser.add_argument("--validation-teacher-cache-root", "--val-teacher-cache-root", dest="validation_teacher_cache_root", type=Path)
    parser.add_argument("--foundation-checkpoint", "--ffs-checkpoint", dest="foundation_checkpoint", type=Path)
    parser.add_argument("--foundation-repo", type=Path)
    parser.add_argument("--vggt-checkpoint", type=Path)
    parser.add_argument("--vggt-repo", type=Path)
    parser.add_argument(
        "--rectification-audit",
        type=Path,
        default=project_root / "reports" / "spring_epipolar_rectification.json",
        help="Stage-C pixel-rectification audit (missing file keeps S6 BLOCKED)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, help="bounded records per split; coverage is marked non-formal")
    parser.add_argument("--steps", type=int, help="bounded optimizer steps for each trainable arm")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--arm", "--arms", dest="arms", action="append", help="arm(s) to run; default all seven")
    parser.add_argument("--skip-cache-build", action="store_true", help="require existing caches; do not launch cache producers")
    parser.add_argument("--skip-manifest-build", action="store_true", help="require supplied/pre-existing manifests")
    parser.add_argument("--overwrite", action="store_true", help="allow cache producers to replace matching records")
    parser.add_argument("--no-resume", action="store_true", help="ignore prior arm completion state")
    parser.add_argument("--dry-run", action="store_true", help="write the full plan without launching subprocesses")
    parser.add_argument(
        "--status",
        action="store_true",
        help="read-only prerequisite/checkpoint status (equivalent to --dry-run)",
    )
    parser.add_argument("--keep-going", action="store_true", help="continue independent arms after a failure")
    return parser


def _normalise_arms(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(ARM_ORDER)
    result: list[str] = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            name = item.strip().upper()
            if name == "ALL":
                result.extend(ARM_ORDER)
            elif name not in ARM_SPECS:
                raise ValueError(f"unknown Spring arm: {item!r}")
            else:
                result.append(name)
    deduped: list[str] = []
    for name in ARM_ORDER:
        if name in result and name not in deduped:
            deduped.append(name)
    return deduped


def _prepare_manifests(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    *,
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    project_root = paths["project_root"]
    all_manifest = paths["manifest_all"]
    train_manifest = paths["manifest_train"]
    val_manifest = paths["manifest_val"]
    if args.train_manifest and args.validation_manifest:
        # Keep caller-owned manifests immutable.  In bounded screening mode
        # the downstream cache receipts must bind to the sliced rows, so
        # materialize those rows under this run's output directory rather than
        # truncating the explicit input files in place.
        source_train_manifest = train_manifest
        source_val_manifest = val_manifest
        train_rows = _read_manifest(source_train_manifest)
        val_rows = _read_manifest(source_val_manifest)
        _check_disjoint(train_rows, val_rows)
        # If the requested bound already covers each caller-owned manifest,
        # retain those exact paths.  Rewriting an identical seven-frame file
        # would unnecessarily invalidate its cache receipts by path lineage.
        needs_bounded_copy = args.limit is not None and (
            int(args.limit) < len(train_rows) or int(args.limit) < len(val_rows)
        )
        if args.limit is not None:
            train_rows = _slice_rows(train_rows, args.limit)
            val_rows = _slice_rows(val_rows, args.limit)
        if needs_bounded_copy:
            bounded_dir = paths["output_root"] / "manifests"
            bounded_dir.mkdir(parents=True, exist_ok=True)
            train_manifest = bounded_dir / f"train_bounded_limit{int(args.limit)}.jsonl"
            val_manifest = bounded_dir / f"validation_bounded_limit{int(args.limit)}.jsonl"
            # Avoid the (unlikely) case where output_root is the same
            # directory as a caller-supplied source manifest.
            if train_manifest.resolve() == source_train_manifest.resolve():
                train_manifest = bounded_dir / f"train_bounded_limit{int(args.limit)}_copy.jsonl"
            if val_manifest.resolve() == source_val_manifest.resolve():
                val_manifest = bounded_dir / f"validation_bounded_limit{int(args.limit)}_copy.jsonl"
            _write_manifest(train_manifest, train_rows)
            _write_manifest(val_manifest, val_rows)
            # All later cache/geometry/training commands consume the bounded
            # copies, while the original explicit manifests remain untouched.
            if isinstance(paths, dict):
                paths["manifest_train"] = train_manifest
                paths["manifest_val"] = val_manifest
        state["manifests"]["source"] = "explicit_train_validation"
        state["manifests"].update(
            {
                "train": str(train_manifest),
                "validation": str(val_manifest),
                "source_train": str(source_train_manifest),
                "source_validation": str(source_val_manifest),
                "train_sha256": _sha256(train_manifest),
                "validation_sha256": _sha256(val_manifest),
                "train_records": len(train_rows),
                "validation_records": len(val_rows),
                "train_sequences": sorted({str(row["sequence_id"]) for row in train_rows}),
                "validation_sequences": sorted({str(row["sequence_id"]) for row in val_rows}),
                "sequence_disjoint": True,
                "bounded_limit": args.limit,
                "formal_coverage": args.limit is None,
            }
        )
        return train_rows, val_rows

    if args.manifest:
        all_rows = _read_manifest(all_manifest)
        state["manifests"]["source"] = "explicit_all"
    elif all_manifest.is_file():
        all_rows = _read_manifest(all_manifest)
        state["manifests"]["source"] = "existing_all"
    else:
        if args.skip_manifest_build:
            raise FileNotFoundError(
                f"all manifest is missing and --skip-manifest-build was supplied: {all_manifest}"
            )
        # Use the validated Spring adapter through its CLI.  This command is
        # intentionally run only when the caller did not provide manifests;
        # malformed/missing image files therefore become an explicit blocked
        # prerequisite rather than a partial manifest.
        command = [
            str(args.python_executable),
            str(project_root / "tools" / "build_spring_manifest.py"),
            "--spring-root",
            str(paths["spring_root"]),
            "--split",
            "train",
            "--output",
            str(all_manifest),
        ]
        result = _run_command(
            command,
            cwd=project_root,
            log_path=paths["output_root"] / "logs" / "build_all_manifest.log",
            dry_run=bool(args.dry_run),
        )
        state["manifest_build"] = result
        if args.dry_run:
            # A dry-run may not have a readable manifest.  Keep a plan-only
            # placeholder; the caller still receives all cache/arm commands.
            return [], []
        if result.get("status") != "COMPLETE" or not all_manifest.is_file():
            raise SpringRunnerError("Spring manifest generation failed")
        all_rows = _read_manifest(all_manifest)
        state["manifests"]["source"] = "built_from_spring_adapter"

    train_rows, val_rows, _train_sequences, _val_sequences = _sequence_split(
        all_rows, seed=SEED, val_fraction=float(args.val_fraction)
    )
    train_rows = _slice_rows(train_rows, args.limit)
    val_rows = _slice_rows(val_rows, args.limit)
    train_sequences = sorted({str(row["sequence_id"]) for row in train_rows})
    val_sequences = sorted({str(row["sequence_id"]) for row in val_rows})
    _check_disjoint(train_rows, val_rows)
    _write_manifest(train_manifest, train_rows)
    _write_manifest(val_manifest, val_rows)
    state["manifests"].update(
        {
            "all": str(all_manifest),
            "all_sha256": _sha256(all_manifest) if all_manifest.is_file() else None,
            "train": str(train_manifest),
            "validation": str(val_manifest),
            "train_sha256": _sha256(train_manifest),
            "validation_sha256": _sha256(val_manifest),
            "train_records": len(train_rows),
            "validation_records": len(val_rows),
            "train_sequences": train_sequences,
            "validation_sequences": val_sequences,
            "sequence_disjoint": True,
            "bounded_limit": args.limit,
            "formal_coverage": args.limit is None,
        }
    )
    return train_rows, val_rows


def _cache_commands(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    *,
    need_vggt: bool,
    need_teacher: bool = True,
) -> list[tuple[str, list[str], Path, str]]:
    """Return cache-producer commands as (name, argv, log, output-root)."""

    project_root = paths["project_root"]
    foundation_checkpoint = _resolve_checkpoint(
        project_root, args.foundation_checkpoint, "checkpoints/foundationstereo/11-33-40/model_best_bp2.pth"
    )
    foundation_repo = _resolve_path(
        args.foundation_repo or project_root / "third_party" / "FoundationStereo"
    )
    # cache_spring_ffs.py passes this value directly to ``torch.device``;
    # unlike train/eval it does not interpret the convenience string ``auto``.
    cache_device = "cuda" if str(args.device).lower() == "auto" else str(args.device)
    commands: list[tuple[str, list[str], Path, str]] = []
    for split, manifest_key, obs_key, teacher_key in (
        ("train", "manifest_train", "train_obs", "train_teacher"),
        ("validation", "manifest_val", "val_obs", "val_teacher"),
    ):
        manifest = paths[manifest_key]
        obs_root = paths[obs_key]
        teacher_root = paths[teacher_key]
        if not _receipt_manifest_matches(obs_root, manifest):
            command = [
                str(args.python_executable),
                str(project_root / "tools" / "cache_spring_ffs.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(obs_root.parent),
                "--checkpoint",
                str(foundation_checkpoint),
                "--repo",
                str(foundation_repo),
                "--role",
                "observation",
                "--device",
                cache_device,
                "--right-left-check",
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(
                (
                    f"{split}_observation_cache",
                    command,
                    paths["output_root"] / "logs" / f"{split}_observation_cache.log",
                    str(obs_root),
                )
            )
        # Spring's dense GT is the teacher target.  It is independent of the
        # FoundationStereo checkpoint and must be built for every split.
        if need_teacher and not _receipt_manifest_matches_with_lineage(
            teacher_root,
            manifest,
            component="ffs-teacher",
            upstream_commit="Spring_GT:cam_data+disp1_left",
            identity_version=2,
        ):
            command = [
                str(args.python_executable),
                str(project_root / "tools" / "cache_spring_gt.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(teacher_root.parent),
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(
                (
                    f"{split}_spring_gt_teacher_cache",
                    command,
                    paths["output_root"] / "logs" / f"{split}_spring_gt_teacher_cache.log",
                    str(teacher_root),
                )
            )
        if need_vggt:
            raw_root = paths[f"{('train' if split == 'train' else 'val')}_vggt"]
            vggt_checkpoint = _resolve_checkpoint(
                project_root, args.vggt_checkpoint, "checkpoints/vggt/vggt_omega_1b_512.pt"
            )
            vggt_repo = _resolve_path(args.vggt_repo or project_root / "third_party" / "vggt-omega")
            # Raw VGGT is sparse (one endpoint per complete five-pair causal
            # context).  Require the exact endpoint set, not merely a matching
            # source-manifest hash, so a partial/crashed cache cannot slip
            # through to training.
            try:
                manifest_rows_for_vggt = _read_manifest(manifest)
                expected_raw_targets = _expected_vggt_targets(manifest_rows_for_vggt)
                expected_output_grid = _expected_vggt_output_grid(manifest_rows_for_vggt)
            except (OSError, SpringRunnerError, ValueError):
                expected_raw_targets = []
                expected_output_grid = None
            if not _strict_vggt_cache_matches(
                raw_root,
                manifest,
                expected_targets=expected_raw_targets,
                expected_output_grid=expected_output_grid,
            ):
                command = [
                    str(args.python_executable),
                    str(project_root / "tools" / "cache_vggt.py"),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(raw_root),
                    "--checkpoint",
                    str(vggt_checkpoint),
                    "--repo",
                    str(vggt_repo),
                    "--context-pairs",
                    "5",
                    "--causal",
                    "--input-mode",
                    "balanced",
                    "--image-resolution",
                    "512",
                    "--device",
                    cache_device,
                ]
                if expected_output_grid is not None:
                    command.extend(
                        [
                            "--output-grid",
                            str(expected_output_grid[0]),
                            str(expected_output_grid[1]),
                        ]
                    )
                if args.overwrite:
                    command.append("--overwrite")
                commands.append(
                    (
                        f"{split}_vggt_cache",
                        command,
                        paths["output_root"] / "logs" / f"{split}_vggt_cache.log",
                        str(raw_root),
                    )
                )
    return commands


def _geometry_commands(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    *,
    selected_arms: Sequence[str],
) -> list[tuple[str, list[str], Path, str]]:
    project_root = paths["project_root"]
    commands: list[tuple[str, list[str], Path, str]] = []
    need_gt_no_depth = any(ARM_SPECS[name].derived_kind == "gt_no_depth" for name in selected_arms)
    need_vggt_depth = any(
        ARM_SPECS[name].derived_kind in {"vggt_pose_vggt_depth", "gt_pose_vggt_depth"}
        for name in selected_arms
    )
    for split, manifest_key, obs_key, raw_key, gt_key, vggt_key, gt_pose_depth_key in (
        ("train", "manifest_train", "train_obs", "train_vggt", "train_gt_geom", "train_vggt_geom", "train_gt_pose_depth"),
        ("validation", "manifest_val", "val_obs", "val_vggt", "val_gt_geom", "val_vggt_geom", "val_gt_pose_depth"),
    ):
        manifest = paths[manifest_key]
        try:
            manifest_rows = _read_manifest(manifest)
            raw_targets = _expected_vggt_targets(manifest_rows)
            all_targets = [
                (
                    index,
                    str(row["sequence_id"]),
                    int(row["frame_id"]),
                    float(row["timestamp"]),
                )
                for index, row in enumerate(manifest_rows)
            ]
        except (OSError, SpringRunnerError, KeyError, TypeError, ValueError):
            raw_targets = []
            all_targets = []
        if need_gt_no_depth:
            target = paths[gt_key]
            if not _strict_derived_cache_matches(
                target, manifest, expected_targets=all_targets, pose_source="Spring_GT_pose"
            ):
                command = [
                    str(args.python_executable),
                    str(project_root / "tools" / "build_spring_gt_geometry.py"),
                    "--manifest",
                    str(manifest),
                    "--observation-root",
                    str(paths[obs_key]),
                    "--output",
                    str(target),
                ]
                if args.overwrite:
                    command.append("--overwrite")
                commands.append(
                    (
                        f"{split}_gt_pose_no_depth_geometry",
                        command,
                        paths["output_root"] / "logs" / f"{split}_gt_pose_no_depth_geometry.log",
                        str(target),
                    )
                )
        if need_vggt_depth:
            raw = paths[raw_key]
            target = paths[vggt_key]
            if not _strict_derived_cache_matches(
                target, manifest, expected_targets=raw_targets
            ):
                command = [
                    str(args.python_executable),
                    str(project_root / "tools" / "derive_geometry_manifest.py"),
                    "--vggt-root",
                    str(raw),
                    "--ffs-root",
                    str(paths[obs_key]),
                    "--output",
                    str(target),
                    "--cache-dtype",
                    "float32",
                ]
                if args.overwrite:
                    command.append("--overwrite")
                commands.append(
                    (
                        f"{split}_vggt_pose_depth_geometry",
                        command,
                        paths["output_root"] / "logs" / f"{split}_vggt_pose_depth_geometry.log",
                        str(target),
                    )
                )
            if any(ARM_SPECS[name].derived_kind == "gt_pose_vggt_depth" for name in selected_arms):
                gt_target = paths[gt_pose_depth_key]
                if not _strict_derived_cache_matches(
                    gt_target,
                    manifest,
                    expected_targets=raw_targets,
                    pose_source="Spring_GT_pose",
                    depth_source="copied_from_vggt_derived",
                ):
                    command = [
                        str(args.python_executable),
                        str(project_root / "tools" / "override_spring_pose.py"),
                        "--manifest",
                        str(manifest),
                        "--source-root",
                        str(target),
                        "--output",
                        str(gt_target),
                    ]
                    if args.overwrite:
                        command.append("--overwrite")
                    commands.append(
                        (
                            f"{split}_gt_pose_vggt_depth_geometry",
                            command,
                            paths["output_root"] / "logs" / f"{split}_gt_pose_vggt_depth_geometry.log",
                            str(gt_target),
                        )
                    )
    return commands


def _spatial_v2_base_command(
    args: argparse.Namespace, paths: Mapping[str, Path], output: Path
) -> list[str]:
    """Build the dedicated v2 Stage-A initializer for S3--S5."""

    command = [
        str(args.python_executable),
        str(paths["project_root"] / "train.py"),
        "--config",
        str(paths["project_root"] / "configs" / "mvp_x2_v2.yaml"),
        "--manifest",
        str(paths["manifest_train"]),
        "--observation-cache-root",
        str(paths["train_obs"]),
        "--teacher-cache-root",
        str(paths["train_teacher"]),
        "--output-dir",
        str(output),
        "--device",
        str(args.device),
    ]
    if args.steps is not None:
        command.append(f"train.steps_spatial={int(args.steps)}")
    return command


def _derived_root(paths: Mapping[str, Path], split: str, kind: str | None) -> Path | None:
    if kind is None:
        return None
    prefix = "train" if split == "train" else "val"
    return paths[f"{prefix}_{'gt_geom' if kind == 'gt_no_depth' else 'gt_pose_depth' if kind == 'gt_pose_vggt_depth' else 'vggt_geom'}"]


def _arm_train_command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    spec: ArmSpec,
    *,
    train_output: Path,
    init_checkpoint: Path | None,
) -> list[str]:
    project_root = paths["project_root"]
    # Spring Stage-C has its own bounded corpus contract.  Keep the canonical
    # Stage-C trainer untouched (it remains bound to the published holdout)
    # and route only S6 through the explicit Spring adapter.
    train_entrypoint = (
        project_root / "tools" / "train_spring_epipolar.py"
        if spec.stage == "epipolar"
        else project_root / "train.py"
    )
    command = [
        str(args.python_executable),
        str(train_entrypoint),
        "--config",
        str(project_root / spec.config),
        "--manifest",
        str(paths["manifest_train"]),
        "--observation-cache-root",
        str(paths["train_obs"]),
        "--teacher-cache-root",
        str(paths["train_teacher"]),
        "--output" if spec.stage == "epipolar" else "--output-dir",
        str(train_output),
        "--device",
        str(args.device),
    ]
    if spec.stage in {"temporal", "epipolar"}:
        derived = _derived_root(paths, "train", spec.derived_kind)
        if derived is None:
            raise SpringRunnerError(f"{spec.name} requires a derived geometry root")
        command.extend(["--derived-cache-root", str(derived)])
        if init_checkpoint is None:
            raise SpringRunnerError(f"{spec.name} requires an initializer checkpoint")
        command.extend(["--init-from" if spec.stage != "epipolar" else "--init-from", str(init_checkpoint)])
    if spec.stage == "epipolar":
        command.append("--spring-screening")
        if str(args.device).lower() == "cpu":
            command.append("--allow-cpu-smoke")
        # In a dry-run there may be no Spring-specific rectification receipt
        # yet.  Keep the command explicit with a deterministic placeholder;
        # the real run is blocked by _arm_blockers until the caller supplies
        # an actual audit file.
        audit = (
            _resolve_path(args.rectification_audit)
            if args.rectification_audit is not None
            else paths["output_root"] / "inputs" / "spring_rectification_audit.json"
        )
        command.extend(["--rectification-audit", str(audit)])
    if args.steps is not None:
        if spec.stage == "epipolar":
            # Without --run-steps Stage-C treats the invocation as formal and
            # requires a completed 15k-step canonical Stage-B base.  Spring
            # screening is bounded, so make that mode explicit.
            command.extend(["--run-steps", str(args.steps)])
        command.append(
            ("train.steps_epipolar=" if spec.stage == "epipolar" else "train.steps=" if spec.stage == "temporal" else "train.steps_spatial=")
            + str(args.steps)
        )
    elif spec.stage == "epipolar":
        # S6.yaml declares 500 screening updates; keep it bounded even when
        # the caller omits --steps.
        command.extend(["--run-steps", "500"])
    return command


def _arm_eval_command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    spec: ArmSpec,
    *,
    checkpoint: Path | None,
    eval_output: Path,
    spatial_checkpoint: Path | None,
) -> list[str]:
    project_root = paths["project_root"]
    if spec.stage == "baseline":
        return [
            str(args.python_executable),
            str(project_root / "tools" / "eval_spring_baseline.py"),
            "--manifest",
            str(paths["manifest_val"]),
            "--observation-cache-root",
            str(paths["val_obs"]),
            "--output",
            str(eval_output),
            "--seed",
            str(SEED),
            *(["--limit", str(args.limit)] if args.limit is not None else []),
        ]
    if checkpoint is None:
        raise SpringRunnerError(f"{spec.name} evaluation requires a checkpoint")
    if spec.stage == "epipolar":
        derived = _derived_root(paths, "validation", spec.derived_kind)
        if derived is None:
            raise SpringRunnerError("S6 requires validation derived geometry")
        audit = (
            _resolve_path(args.rectification_audit)
            if args.rectification_audit is not None
            else paths["output_root"] / "inputs" / "spring_rectification_audit.json"
        )
        command = [
            str(args.python_executable),
            str(project_root / "tools" / "eval_spring_epipolar.py"),
            "--config",
            str(project_root / spec.config),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(paths["manifest_val"]),
            "--observation-cache-root",
            str(paths["val_obs"]),
            "--teacher-cache-root",
            str(paths["val_teacher"]),
            "--derived-cache-root",
            str(derived),
            "--rectification-audit",
            str(audit),
            "--output",
            str(eval_output),
            "--device",
            str(args.device),
            "--spring-screening",
        ]
        if spatial_checkpoint is not None:
            command.extend(["--base-checkpoint", str(spatial_checkpoint)])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        return command
    command = [
        str(args.python_executable),
        str(project_root / "eval.py"),
        "--config",
        str(project_root / spec.config),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(paths["manifest_val"]),
        "--observation-cache-root",
        str(paths["val_obs"]),
        "--teacher-cache-root",
        str(paths["val_teacher"]),
        "--output",
        str(eval_output),
        "--device",
        str(args.device),
        "--crop-mode",
        "full",
    ]
    if spec.stage == "temporal":
        derived = _derived_root(paths, "validation", spec.derived_kind)
        if derived is None:
            raise SpringRunnerError(f"{spec.name} requires validation derived geometry")
        command.extend(["--derived-cache-root", str(derived)])
        if spatial_checkpoint is not None:
            command.extend(["--spatial-checkpoint", str(spatial_checkpoint)])
        # A bounded Spring screening split is deliberately not the project's
        # canonical holdout.  Use the evaluator's explicit non-holdout smoke
        # mode so it skips the full endpoint-coverage assertion while still
        # auditing cache/checkpoint lineage and reporting LIMITED status.
        if args.limit is not None:
            command.append("--allow-non-holdout-smoke")
        # Spring's native detail/match/rigid metrics are an explicit side
        # channel.  Keep the canonical pseudo-GT report intact while making
        # every bounded Spring temporal arm carry the requested native fields.
        command.append("--spring-native-metrics")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def _arm_audit_command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    spec: ArmSpec,
    *,
    checkpoint: Path,
    eval_output: Path,
) -> list[str] | None:
    """Build the read-only Spring Stage-C report audit command for S6."""

    if spec.name != "S6":
        return None
    return [
        str(args.python_executable),
        str(paths["project_root"] / "tools" / "audit_spring_epipolar.py"),
        "--metrics",
        str(eval_output / "metrics.json"),
        "--validation-manifest",
        str(paths["manifest_val"]),
        "--checkpoint",
        str(checkpoint),
        "--train-adapter",
        str(paths["project_root"] / "tools" / "train_spring_epipolar.py"),
        "--output",
        str(eval_output / "audit.json"),
    ]


def _empty_metric_contract() -> dict[str, Any]:
    return {name: None for name in (*REQUIRED_METRICS, *TOPK_DIAGNOSTICS)}


def _extract_metrics(
    report: Mapping[str, Any], *, preferred_method: str | None = None
) -> dict[str, Any]:
    """Map known evaluator fields without inventing unavailable values.

    ``eval.py`` always emits a ``bilinear`` comparator before the learned
    method.  Selecting the first mapping therefore silently attributed the
    baseline result to S1--S5.  Callers pass the arm-specific raw method name
    (``T1``, ``T3``, or ``T3_VGGT``); a deterministic fallback is retained for
    older reports that do not expose that key.
    """

    result = _empty_metric_contract()
    direct = report.get("metrics")
    if isinstance(direct, Mapping):
        for name in result:
            if name in direct:
                value = direct[name]
                if isinstance(value, Mapping):
                    value = value.get("value")
                # Keep structured bucket gains structured, but never expose a
                # metric contract scalar as an opaque evaluator mapping.
                if isinstance(value, (int, float, str, list, tuple, dict)) or value is None:
                    result[name] = value
    # S0 already uses the Spring metric names.  Existing eval.py reports use
    # method-specific names; map only exact, semantically equivalent fields.
    methods = report.get("methods")
    if isinstance(methods, Mapping):
        candidate: Mapping[str, Any] | None = None
        if preferred_method:
            preferred = methods.get(preferred_method)
            if isinstance(preferred, Mapping):
                candidate = preferred
        if candidate is None:
            # Keep a stable fallback for legacy reports.  Prefer a learned
            # endpoint over the comparator whenever possible, and only then
            # fall back to the first mapping for backwards compatibility.
            for method_name in ("T3_VGGT", "T3", "T1", "bilinear"):
                value = methods.get(method_name)
                if isinstance(value, Mapping):
                    candidate = value
                    break
        if candidate is None:
            for value in methods.values():
                if isinstance(value, Mapping):
                    candidate = value
                    break
        if candidate is not None:
            aliases = {
                # These are exact evaluator fields.  Do not substitute a
                # pseudo-GT or low-confidence domain for Spring detail/match
                # domains; those remain null unless a Spring-aware evaluator
                # emits them explicitly.
                "overall_epe": ("epe_px", "epe_hr_px", "overall_epe"),
                "overall_1px": ("bad_1", "bad_1px", "overall_1px"),
                "boundary_epe": ("boundary_epe_px", "boundary_epe_hr_px", "boundary_epe"),
                "rigid_temporal_residual_error": (
                    "rigid_temporal_residual_error",
                    "temporal_residual_error_native_px",
                ),
                "negative_rate": (
                    "negative_rate",
                    "output_negative_rate",
                ),
                "zero_rate": ("zero_rate", "output_zero_rate"),
                "invalid_rate": (
                    "invalid_rate",
                    "output_invalid_rate",
                ),
            }
            for target, names in aliases.items():
                if result[target] is None:
                    for source in names:
                        value = candidate.get(source)
                        if isinstance(value, Mapping):
                            value = value.get("value")
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            result[target] = value
                            break
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        for section in diagnostics.values():
            if not isinstance(section, Mapping):
                continue
            aliases = {
                "age_2_survival_rate": (
                    "age_2_survival_rate",
                    "age2_survival_rate",
                ),
                "unique_age_fraction": ("unique_age_fraction",),
                "phase_variance": (
                    "phase_variance",
                    "fractional_phase_variance",
                ),
                "candidate_depth_spread": (
                    "candidate_depth_spread",
                    "candidate_depth_spread_m",
                ),
                "attention_entropy": (
                    "attention_entropy",
                    "metric_attention_weight_entropy",
                    "topk_weight_entropy",
                ),
            }
            for name, source_names in aliases.items():
                if result[name] is not None:
                    continue
                for source_name in source_names:
                    value = section.get(source_name)
                    if isinstance(value, Mapping):
                        value = value.get("value")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        result[name] = value
                        break
    return result


def _read_result_metrics(
    eval_output: Path, *, preferred_method: str | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [eval_output / "metrics.json", eval_output / "metrics_summary.json"]
    for path in candidates:
        if path.is_file():
            try:
                report = _read_json(path)
            except SpringRunnerError as exc:
                return None, str(exc)
            return {
                "path": str(path),
                "sha256": _sha256(path),
                "report": report,
                "metrics": _extract_metrics(
                    report, preferred_method=preferred_method
                ),
            }, None
    return None, "evaluation metrics.json was not produced"


def _preferred_eval_method(arm_name: str) -> str | None:
    """Return the raw evaluator method owned by a Spring arm.

    The temporal evaluator emits several paired ablations in one report.  A
    Spring arm must consume only its configured branch: S2/S3 are the no-VGGT
    ``T3`` branch, while S4/S5 use the VGGT-prior ``T3_VGGT`` branch.  S0 has
    a dedicated evaluator and S6 is fail-closed before canonical Stage-C
    evaluation.
    """

    return {
        "S1": "T1",
        "S2": "T3",
        "S3": "T3",
        "S4": "T3_VGGT",
        "S5": "T3_VGGT",
        "S6": "T3_VGGT_epipolar",
    }.get(str(arm_name).upper())


def _preflight(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    selected_arms: Sequence[str],
) -> dict[str, Any]:
    project_root = paths["project_root"]
    need_vggt = any(ARM_SPECS[name].requires_vggt for name in selected_arms)
    foundation_checkpoint = _resolve_checkpoint(project_root, args.foundation_checkpoint, "checkpoints/foundationstereo/11-33-40/model_best_bp2.pth")
    vggt_checkpoint = _resolve_checkpoint(project_root, args.vggt_checkpoint, "checkpoints/vggt/vggt_omega_1b_512.pt")
    foundation_repo = _resolve_path(args.foundation_repo or project_root / "third_party" / "FoundationStereo")
    vggt_repo = _resolve_path(args.vggt_repo or project_root / "third_party" / "vggt-omega")
    observation_build_required = any(
        not _receipt_manifest_matches(paths[key], paths[manifest_key])
        for key, manifest_key in (("train_obs", "manifest_train"), ("val_obs", "manifest_val"))
    )
    vggt_build_required = False
    if need_vggt:
        for key, manifest_key in (("train_vggt", "manifest_train"), ("val_vggt", "manifest_val")):
            manifest = paths[manifest_key]
            try:
                manifest_rows_for_vggt = _read_manifest(manifest)
                expected_targets = _expected_vggt_targets(manifest_rows_for_vggt)
                expected_output_grid = _expected_vggt_output_grid(manifest_rows_for_vggt)
            except (OSError, SpringRunnerError, ValueError):
                expected_targets = []
                expected_output_grid = None
            if not _strict_vggt_cache_matches(
                paths[key],
                manifest,
                expected_targets=expected_targets,
                expected_output_grid=expected_output_grid,
            ):
                vggt_build_required = True
    image_inputs_required = observation_build_required or vggt_build_required or any(
        ARM_SPECS[name].stage != "baseline" for name in selected_arms
    )
    prerequisites: dict[str, Any] = {
        "foundation_checkpoint": _file_requirement(foundation_checkpoint, "FoundationStereo checkpoint"),
        "foundation_repo": {"label": "FoundationStereo repository", "path": str(foundation_repo), "exists": foundation_repo.is_dir()},
        "observation_cache_build_required": observation_build_required,
        "vggt_required": need_vggt,
        "vggt_cache_build_required": vggt_build_required,
        "vggt_checkpoint": _file_requirement(vggt_checkpoint, "VGGT-Omega checkpoint") if need_vggt else None,
        "vggt_repo": {"label": "VGGT-Omega repository", "path": str(vggt_repo), "exists": vggt_repo.is_dir()} if need_vggt else None,
        "image_inputs_required": image_inputs_required,
        "train_image_missing": (
            _manifest_missing_paths(train_rows, paths["manifest_train"])
            if image_inputs_required and train_rows
            else (["manifest not available (dry-run)"] if not train_rows else [])
        ),
        "validation_image_missing": (
            _manifest_missing_paths(val_rows, paths["manifest_val"])
            if image_inputs_required and val_rows
            else (["manifest not available (dry-run)"] if not val_rows else [])
        ),
        "train_gt_missing": _manifest_missing_gt(train_rows, paths["manifest_train"]) if train_rows else ["manifest not available (dry-run)"],
        "validation_gt_missing": _manifest_missing_gt(val_rows, paths["manifest_val"]) if val_rows else ["manifest not available (dry-run)"],
        "rectification_audit": _file_requirement(_resolve_path(args.rectification_audit), "Stage-C rectification audit") if args.rectification_audit else {"label": "Stage-C rectification audit", "exists": False, "reason": "--rectification-audit not supplied"},
    }
    # eval_epipolar.py owns a deliberately fixed canonical holdout (244
    # manifest records / 240 derived endpoints / 238 T3 windows plus a bound
    # SHA-256).  Spring uses the dedicated screening adapters instead.  Keep
    # canonical eligibility false even when the Spring path is runnable.
    prerequisites["formal_stage_c_eligible"] = False
    spring_trainer = project_root / "tools" / "train_spring_epipolar.py"
    spring_evaluator = project_root / "tools" / "eval_spring_epipolar.py"
    spring_auditor = project_root / "tools" / "audit_spring_epipolar.py"
    prerequisites["spring_stage_c"] = {
        "protocol": "spring_stage_c_sequence_screening_v1",
        "canonical": False,
        "trainer": _file_requirement(spring_trainer, "Spring Stage-C trainer"),
        "evaluator": _file_requirement(spring_evaluator, "Spring Stage-C evaluator"),
        "auditor": _file_requirement(spring_auditor, "Spring Stage-C evaluator audit"),
        "available": spring_trainer.is_file()
        and spring_evaluator.is_file()
        and spring_auditor.is_file(),
    }
    cache_requirements: dict[str, Any] = {}
    for key, label in (("train_obs", "train observation"), ("train_teacher", "train teacher"), ("val_obs", "validation observation"), ("val_teacher", "validation teacher")):
        cache_requirements[key] = _dir_requirement(paths[key], label)
    if need_vggt:
        for key, label in (("train_vggt", "train raw VGGT"), ("val_vggt", "validation raw VGGT"), ("train_vggt_geom", "train VGGT pose/depth"), ("val_vggt_geom", "validation VGGT pose/depth")):
            cache_requirements[key] = _dir_requirement(paths[key], label)
    if any(ARM_SPECS[name].derived_kind == "gt_no_depth" for name in selected_arms):
        for key, label in (("train_gt_geom", "train GT pose/no depth"), ("val_gt_geom", "validation GT pose/no depth")):
            cache_requirements[key] = _dir_requirement(paths[key], label)
    if any(ARM_SPECS[name].derived_kind == "gt_pose_vggt_depth" for name in selected_arms):
        for key, label in (("train_gt_pose_depth", "train GT pose/VGGT depth"), ("val_gt_pose_depth", "validation GT pose/VGGT depth")):
            cache_requirements[key] = _dir_requirement(paths[key], label)
    prerequisites["caches"] = cache_requirements
    blockers: list[str] = []
    if observation_build_required and not prerequisites["foundation_checkpoint"]["exists"]:
        blockers.append(f"missing FoundationStereo checkpoint: {foundation_checkpoint}")
    if observation_build_required and not prerequisites["foundation_repo"]["exists"]:
        blockers.append(f"missing FoundationStereo repository: {foundation_repo}")
    if vggt_build_required and not prerequisites["vggt_checkpoint"]["exists"]:
        blockers.append(f"missing VGGT-Omega checkpoint: {vggt_checkpoint}")
    if vggt_build_required and not prerequisites["vggt_repo"]["exists"]:
        blockers.append(f"missing VGGT-Omega repository: {vggt_repo}")
    # Stage-C audit availability is an S6-only prerequisite.  Keep it in the
    # structured preflight report, but do not make it a global cache blocker
    # that would unnecessarily stop S0--S5.
    for name in ("train_image_missing", "validation_image_missing", "train_gt_missing", "validation_gt_missing"):
        values = prerequisites[name]
        if values:
            blockers.append(f"{name}: {values[:3]}")
    prerequisites["blockers"] = blockers
    prerequisites["status"] = "BLOCKED" if blockers else "READY"
    return prerequisites


def _arm_blockers(
    name: str,
    spec: ArmSpec,
    *,
    prerequisites: Mapping[str, Any],
    paths: Mapping[str, Path],
    completed: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    blockers: list[str] = []
    if spec.stage in {"temporal", "epipolar"}:
        # ``build_causal_windows`` needs five records for a raw VGGT endpoint,
        # and a T=3 student window needs three consecutive *derived* endpoints.
        # Thus VGGT-dependent S4--S6 need at least seven records per sequence;
        # GT-pose S2/S3 need only the five-pair warm-up.  A global bound is not
        # sufficient for interleaved/custom manifests, so inspect each split.
        minimum_records = 7 if spec.requires_vggt else 5
        if args.limit is not None and args.limit < minimum_records:
            blockers.append(
                f"{name} requires --limit >= {minimum_records} records per selected sequence; "
                f"got {args.limit}"
            )
        for split, manifest_key in (("train", "manifest_train"), ("validation", "manifest_val")):
            manifest = paths[manifest_key]
            if not manifest.is_file():
                continue
            try:
                rows = _read_manifest(manifest)
            except (OSError, SpringRunnerError):
                continue
            counts: dict[str, int] = {}
            for row in rows:
                sequence_id = str(row.get("sequence_id", ""))
                counts[sequence_id] = counts.get(sequence_id, 0) + 1
            insufficient = sorted(
                sequence_id
                for sequence_id, count in counts.items()
                if count < minimum_records
            )
            if insufficient:
                blockers.append(
                    f"{name} requires at least {minimum_records} causal records per sequence; "
                    f"{split} sequences below minimum: {insufficient[:8]}"
                )
    if prerequisites.get("blockers"):
        # Keep global blockers relevant to this arm.  Missing Stage-C audit is
        # only an S6 blocker; all other data/weight blockers apply broadly.
        for reason in prerequisites["blockers"]:
            if "Stage-C rectification audit" in reason and name != "S6":
                continue
            blockers.append(str(reason))
    if spec.init_arm:
        init = completed.get(spec.init_arm)
        if not init or init.get("status") != "COMPLETE":
            blockers.append(f"initializer {spec.init_arm} did not complete")
        else:
            checkpoint = init.get("train", {}).get("checkpoint")
            if not checkpoint or not Path(str(checkpoint)).is_file():
                blockers.append(f"initializer {spec.init_arm} checkpoint is missing")
    if name == "S6" and args.rectification_audit is None:
        blockers.append(
            "S6 requires --rectification-audit; Spring Stage-C is fail-closed"
        )
    if name == "S6" and args.rectification_audit is not None:
        audit_path = _resolve_path(args.rectification_audit)
        if not audit_path.is_file():
            blockers.append(
                f"missing Spring Stage-C rectification audit: {audit_path}"
            )
    if name == "S6" and args.limit is None:
        blockers.append(
            "S6 Spring Stage-C screening requires an explicit --limit; "
            "unbounded evaluation is reserved for canonical Stage-C"
        )
    if name == "S6" and not prerequisites.get("spring_stage_c", {}).get(
        "available", False
    ):
        blockers.append(
            "Spring Stage-C trainer/evaluator adapters are missing; "
            "S6 cannot run"
        )
    # Cache producers are separate subprocesses.  In particular,
    # ``--skip-cache-build`` marks their planned tasks BLOCKED, but that task
    # status is not part of ``prerequisites[\"blockers\"]``.  Check the
    # manifest-bound receipts here as well, otherwise an arm could proceed to
    # training with an empty/stale cache (and fail much later with a confusing
    # dataloader error).  The checks apply after producers have run too, so a
    # failed producer remains an explicit arm blocker.
    manifest_keys = {"train": "manifest_train", "validation": "manifest_val"}

    def require_receipt(root: Path, split: str, label: str) -> None:
        manifest = paths[manifest_keys[split]]
        if label == "teacher":
            matches = _receipt_manifest_matches_with_lineage(
                root,
                manifest,
                component="ffs-teacher",
                upstream_commit="Spring_GT:cam_data+disp1_left",
                identity_version=2,
            )
        elif label == "raw VGGT":
            try:
                manifest_rows_for_vggt = _read_manifest(manifest)
                expected_targets = _expected_vggt_targets(manifest_rows_for_vggt)
                expected_output_grid = _expected_vggt_output_grid(manifest_rows_for_vggt)
            except (OSError, SpringRunnerError, ValueError):
                expected_targets = []
                expected_output_grid = None
            matches = _strict_vggt_cache_matches(
                root,
                manifest,
                expected_targets=expected_targets,
                expected_output_grid=expected_output_grid,
            )
        elif label in {"vggt_pose_vggt_depth", "gt_pose_vggt_depth"}:
            try:
                rows = _read_manifest(manifest)
                expected_targets = _expected_vggt_targets(rows)
            except (OSError, SpringRunnerError, ValueError):
                expected_targets = []
            matches = _strict_derived_cache_matches(
                root,
                manifest,
                expected_targets=expected_targets,
                **(
                    {
                        "pose_source": "Spring_GT_pose",
                        "depth_source": "copied_from_vggt_derived",
                    }
                    if label == "gt_pose_vggt_depth"
                    else {}
                ),
            )
        elif label == "gt_no_depth":
            try:
                rows = _read_manifest(manifest)
                expected_targets = [
                    (
                        index,
                        str(row["sequence_id"]),
                        int(row["frame_id"]),
                        float(row["timestamp"]),
                    )
                    for index, row in enumerate(rows)
                ]
            except (OSError, SpringRunnerError, KeyError, TypeError, ValueError):
                expected_targets = []
            matches = _strict_derived_cache_matches(
                root,
                manifest,
                expected_targets=expected_targets,
                pose_source="Spring_GT_pose",
            )
        else:
            matches = _receipt_manifest_matches(root, manifest)
        if not matches:
            blockers.append(
                f"missing or stale {split} {label} cache receipt for {manifest}: "
                f"{root / 'run_receipt.json'}"
            )

    if spec.stage == "baseline":
        # S0 only consumes the validation FFS observation cache; it does not
        # need teacher or geometry/VGGT artifacts.
        require_receipt(paths["val_obs"], "validation", "observation")
    else:
        for split in ("train", "validation"):
            require_receipt(paths[f"{'train' if split == 'train' else 'val'}_obs"], split, "observation")
            require_receipt(paths[f"{'train' if split == 'train' else 'val'}_teacher"], split, "teacher")

        if spec.requires_vggt:
            for split in ("train", "validation"):
                require_receipt(paths[f"{'train' if split == 'train' else 'val'}_vggt"], split, "raw VGGT")

        if spec.stage in {"temporal", "epipolar"} and spec.derived_kind is not None:
            for split in ("train", "validation"):
                root = _derived_root(paths, split, spec.derived_kind)
                if root is None:
                    blockers.append(f"{split} derived geometry root is unresolved")
                else:
                    require_receipt(root, split, spec.derived_kind)
    return blockers


def run(args: argparse.Namespace) -> int:
    # ``--status`` is intentionally an alias for the non-mutating planning
    # path.  It is accepted separately because operators commonly poll a
    # long-running Spring download with a status command.
    if args.status:
        args.dry_run = True
    if args.seed != SEED:
        raise ValueError("Spring screening is fixed to seed 42; other seeds require a separate protocol")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not 0.0 < float(args.val_fraction) < 1.0:
        raise ValueError("--val-fraction must be in (0,1)")
    selected_arms = _normalise_arms(args.arms)
    project_root = _resolve_path(args.project_root)
    paths = _default_paths(project_root, args)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "spring_seed42_state.json"
    summary_path = output_root / "spring_seed42_summary.json"
    prior = _read_json(state_path) if state_path.is_file() and not args.no_resume else {}
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "component": "spring-seven-arm-runner",
        "status": "RUNNING",
        "seed": SEED,
        "started_at_utc": prior.get("started_at_utc", _utc_now()),
        "updated_at_utc": _utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "selected_arms": selected_arms,
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "manifests": {},
        "prerequisites": {},
        "cache_tasks": [],
        "geometry_tasks": [],
        "arms": [],
        "metrics_contract": {
            "required_metrics": list(REQUIRED_METRICS),
            "topk_diagnostics": list(TOPK_DIAGNOSTICS),
            "missing_value_semantics": "null means unavailable/not evaluated; never imputed",
        },
    }

    def persist() -> None:
        state["updated_at_utc"] = _utc_now()
        _atomic_json(state_path, state)
        # Summary is intentionally the same auditable object.  Keeping both
        # names makes it convenient for batch systems while avoiding a second
        # source of truth.
        _atomic_json(summary_path, state)

    persist()
    try:
        train_rows, val_rows = _prepare_manifests(args, paths, state=state)
    except (OSError, SpringRunnerError, ValueError) as exc:
        state["status"] = "BLOCKED"
        state["blocker"] = f"manifest preparation: {type(exc).__name__}: {exc}"
        persist()
        return 2
    if args.dry_run and (not train_rows or not val_rows):
        # Continue with a plan even when the source images are not downloaded.
        # Preflight will retain the missing-manifest reason and every command
        # remains visible in the receipt.
        state["manifests"].update({"train": str(paths["manifest_train"]), "validation": str(paths["manifest_val"]), "formal_coverage": False, "bounded_limit": args.limit})
    persist()
    prerequisites = _preflight(args, paths, train_rows, val_rows, selected_arms)
    state["prerequisites"] = prerequisites
    persist()

    need_vggt = any(ARM_SPECS[name].requires_vggt for name in selected_arms)
    cache_tasks = _cache_commands(
        args,
        paths,
        need_vggt=need_vggt,
        need_teacher=any(ARM_SPECS[name].stage != "baseline" for name in selected_arms),
    )
    geometry_tasks = _geometry_commands(args, paths, selected_arms=selected_arms)
    state["cache_tasks"] = [{"name": name, "output_root": output, "command": command, "command_text": _command_text(command), "status": "PLANNED"} for name, command, _log, output in cache_tasks]
    state["geometry_tasks"] = [{"name": name, "output_root": output, "command": command, "command_text": _command_text(command), "status": "PLANNED"} for name, command, _log, output in geometry_tasks]
    persist()

    # In strict existing-cache mode, missing receipts are blockers.  In build
    # mode, execute producers unless a global preflight blocker already proves
    # that doing so cannot succeed (for example, no images/checkpoint).
    # A missing Stage-C rectification audit blocks only S6; it must not prevent
    # S0--S5 cache production.  Other global blockers (missing images, weights,
    # or required VGGT runtime) are safe to apply to all producers.  This
    # initial value is refreshed after preparation below because --skip-cache-
    # build and failed producer tasks add blockers during that phase.
    global_blockers = [
        str(reason)
        for reason in prerequisites.get("blockers", [])
        if "Stage-C rectification audit" not in str(reason)
    ]
    prep_blocked = bool(global_blockers)
    all_tasks = [("cache", item) for item in cache_tasks] + [("geometry", item) for item in geometry_tasks]
    for task_index, (kind, item) in enumerate(all_tasks):
        name, command, log_path, output = item
        task_list = state["cache_tasks"] if kind == "cache" else state["geometry_tasks"]
        task_record = task_list[next(i for i, row in enumerate(task_list) if row["name"] == name)]
        if args.skip_cache_build:
            task_record.update({"status": "BLOCKED", "reason": "--skip-cache-build and required receipt is absent"})
            persist()
            continue
        if prep_blocked and not args.dry_run:
            task_record.update({"status": "BLOCKED", "reason": f"global preflight blocker; producer not launched: {global_blockers[:3]}"})
            persist()
            continue
        result = _run_command(command, cwd=project_root, log_path=log_path, dry_run=bool(args.dry_run))
        task_record.update(result)
        persist()
        if result.get("status") == "FAILED" and not args.keep_going:
            # Remaining tasks are not attempted, but remain explicit.
            for _kind, (_name, _command, _log, _output) in all_tasks[task_index + 1 :]:
                target = state["cache_tasks"] if _kind == "cache" else state["geometry_tasks"]
                for row in target:
                    if row["name"] == _name and row["status"] == "PLANNED":
                        row.update({"status": "BLOCKED", "reason": f"prior {_name} task failed"})
            persist()
            break

    # Recompute preparation blockers after the producer phase.  A cache task
    # may have been planned/failed even though the source-data preflight was
    # otherwise healthy; downstream arms must never launch against a partial
    # cache.  In ``--dry-run`` mode planned tasks are reported as pending
    # blockers so the receipt cannot be mistaken for an executable result.
    preparation_blockers: list[str] = []
    for kind, task in (("cache", row) for row in state["cache_tasks"]):
        status = str(task.get("status"))
        if status != "COMPLETE":
            reason = task.get("reason") or (
                "producer not launched in dry-run" if args.dry_run else "producer did not complete"
            )
            preparation_blockers.append(f"{kind} task {task.get('name')}: {reason}")
    for kind, task in (("geometry", row) for row in state["geometry_tasks"]):
        status = str(task.get("status"))
        if status != "COMPLETE":
            reason = task.get("reason") or (
                "producer not launched in dry-run" if args.dry_run else "producer did not complete"
            )
            preparation_blockers.append(f"{kind} task {task.get('name')}: {reason}")
    if preparation_blockers:
        prerequisites["preparation_blockers"] = preparation_blockers
        # Keep producer failures separate from source-data blockers.  Cache
        # tasks are intentionally shared by several arms, but their absence
        # must not globally block an independent arm: for example S0 only
        # needs the validation observation cache, while S4--S6 may be waiting
        # on CUDA VGGT caches.  ``_arm_blockers`` performs dependency-aware
        # receipt checks for each arm and the state summary still records all
        # preparation failures in this dedicated field.
        prerequisites["status"] = "BLOCKED"
        state["prerequisites"] = prerequisites
        persist()

    global_blockers = [
        str(reason)
        for reason in prerequisites.get("blockers", [])
        if "Stage-C rectification audit" not in str(reason)
    ]

    completed: dict[str, dict[str, Any]] = {}
    # S3--S5 share a separately trained v2 Stage-A graph.  It cannot be
    # initialized from S1 because the top-K/history modules change the state
    # dict.  Materialize (or explicitly plan) this hidden prerequisite before
    # evaluating the public arms.
    need_v2_base = any(name in selected_arms for name in ("S3", "S4", "S5", "S6"))
    base_output = output_root / "spatial_v2_base" / "train"
    base_checkpoint = base_output / "final.pt"
    if need_v2_base:
        base_command = _spatial_v2_base_command(args, paths, base_output)
        base_record: dict[str, Any] = {
            "arm": "spatial_v2_base",
            "status": "PLANNED",
            "stage": "spatial",
            "config": str(project_root / "configs" / "mvp_x2_v2.yaml"),
            "train_output": str(base_output),
            "commands": [],
        }
        # Compute this before the dry-run branch as well.  A status receipt
        # should make it obvious that the hidden initializer is not runnable
        # until its train caches exist, while still recording the command that
        # would be launched once they are ready.
        base_cache_blockers: list[str] = []
        for root_key, label in (("train_obs", "observation"), ("train_teacher", "teacher")):
            root = paths[root_key]
            matches = (
                _receipt_manifest_matches_with_lineage(
                    root,
                    paths["manifest_train"],
                    component="ffs-teacher",
                    upstream_commit="Spring_GT:cam_data+disp1_left",
                    identity_version=2,
                )
                if label == "teacher"
                else _receipt_manifest_matches(root, paths["manifest_train"])
            )
            if not matches:
                base_cache_blockers.append(
                    f"missing or stale train {label} cache receipt for "
                    f"{paths['manifest_train']}: {root / 'run_receipt.json'}"
                )
        if base_checkpoint.is_file() and not args.no_resume:
            base_record.update(
                {
                    "status": "COMPLETE",
                    "train": {
                        "status": "REUSED",
                        "checkpoint": str(base_checkpoint.resolve()),
                        "checkpoint_exists": True,
                    },
                }
            )
        elif args.dry_run:
            base_record["commands"].append(
                _run_command(
                    base_command,
                    cwd=project_root,
                    log_path=output_root / "logs" / "spatial_v2_base_train.log",
                    dry_run=True,
                )
            )
            if global_blockers or base_cache_blockers:
                base_record.update(
                    {
                        "status": "BLOCKED",
                        "blockers": [*global_blockers, *base_cache_blockers],
                    }
                )
        else:
            # The v2 Stage-A initializer consumes the train observation and
            # teacher caches directly.  Treat missing/stale receipts as a
            # first-class blocker (including ``--skip-cache-build``), rather
            # than launching a job that will fail inside the dataloader.
            if global_blockers or base_cache_blockers:
                base_record.update(
                    {
                        "status": "BLOCKED",
                        "blockers": [*global_blockers, *base_cache_blockers],
                    }
                )
            else:
                result = _run_command(
                    base_command,
                    cwd=project_root,
                    log_path=output_root / "logs" / "spatial_v2_base_train.log",
                )
                base_record["commands"].append(result)
                if result.get("status") == "COMPLETE" and base_checkpoint.is_file():
                    base_record.update(
                        {
                            "status": "COMPLETE",
                            "train": {
                                "status": "COMPLETE",
                                "checkpoint": str(base_checkpoint.resolve()),
                                "checkpoint_exists": True,
                            },
                        }
                    )
                else:
                    base_record.update({"status": "FAILED", "blockers": ["v2 spatial base did not produce final.pt"]})
        completed["spatial_v2_base"] = base_record
        state["spatial_v2_base"] = base_record
        persist()
    prior_arms = {str(row.get("arm")): row for row in prior.get("arms", []) if isinstance(row, Mapping)}
    for name in selected_arms:
        spec = ARM_SPECS[name]
        arm_dir = output_root / "arms" / name
        train_output = arm_dir / "train"
        eval_output = arm_dir / "eval"
        row: dict[str, Any] = {
            "arm": name,
            "status": "PLANNED",
            "config": str(project_root / spec.config),
            "stage": spec.stage,
            "pose_source": spec.pose_source,
            "use_vggt_depth": spec.use_vggt_depth,
            "derived_kind": spec.derived_kind,
            "derived_train_root": str(_derived_root(paths, "train", spec.derived_kind)) if spec.derived_kind else None,
            "derived_validation_root": str(_derived_root(paths, "validation", spec.derived_kind)) if spec.derived_kind else None,
            "train_output": str(train_output),
            "eval_output": str(eval_output),
            "metrics": _empty_metric_contract(),
            "metrics_status": "UNAVAILABLE",
            "commands": [],
        }
        previous = prior_arms.get(name)
        if previous and previous.get("status") == "COMPLETE" and not args.no_resume:
            metrics_info, metrics_error = _read_result_metrics(
                eval_output, preferred_method=_preferred_eval_method(name)
            )
            checkpoint = Path(str(previous.get("train", {}).get("checkpoint", ""))) if previous.get("train", {}).get("checkpoint") else None
            spring_audit_ok = True
            if name == "S6":
                audit_path = eval_output / "audit.json"
                try:
                    spring_audit_ok = (
                        audit_path.is_file()
                        and _read_json(audit_path).get("status") == "PASS"
                    )
                except SpringRunnerError:
                    spring_audit_ok = False
            if spring_audit_ok and (name == "S0" or (checkpoint is not None and checkpoint.is_file())):
                row = copy.deepcopy(previous)
                row["status"] = "COMPLETE"
                if metrics_info is not None:
                    row["metrics"] = metrics_info["metrics"]
                    row["metrics_status"] = "AVAILABLE"
                completed[name] = row
                state["arms"].append(row)
                persist()
                continue
        blockers = _arm_blockers(name, spec, prerequisites=prerequisites, paths=paths, completed=completed, args=args)
        # A dry-run intentionally keeps missing source files visible but still
        # emits commands for every arm.  The per-arm status is PLANNED rather
        # than COMPLETE; no metrics are fabricated.
        if blockers and not args.dry_run:
            row.update({"status": "BLOCKED", "blockers": blockers})
            state["arms"].append(row)
            completed[name] = row
            persist()
            continue
        if args.dry_run:
            if spec.stage != "baseline":
                init_path = None
                if spec.init_arm:
                    init_path = (
                        base_checkpoint
                        if spec.init_arm == "spatial_v2_base"
                        else output_root / "arms" / spec.init_arm / "train" / "final.pt"
                    )
                train_command = _arm_train_command(args, paths, spec, train_output=train_output, init_checkpoint=init_path)
                row["commands"].append(_run_command(train_command, cwd=project_root, log_path=output_root / "logs" / f"{name}_train.log", dry_run=True))
                checkpoint_path = train_output / "final.pt"
            else:
                checkpoint_path = None
            spatial_path = (
                output_root / "arms" / "S5" / "train" / "final.pt"
                if name == "S6"
                else base_checkpoint
                if name in {"S3", "S4", "S5"}
                else output_root / "arms" / "S1" / "train" / "final.pt"
                if name == "S2"
                else None
            )
            eval_command = _arm_eval_command(args, paths, spec, checkpoint=checkpoint_path, eval_output=eval_output, spatial_checkpoint=spatial_path)
            row["commands"].append(_run_command(eval_command, cwd=project_root, log_path=output_root / "logs" / f"{name}_eval.log", dry_run=True))
            audit_command = (
                _arm_audit_command(
                    args,
                    paths,
                    spec,
                    checkpoint=checkpoint_path,
                    eval_output=eval_output,
                )
                if checkpoint_path is not None
                else None
            )
            if audit_command is not None:
                row["commands"].append(
                    _run_command(
                        audit_command,
                        cwd=project_root,
                        log_path=output_root / "logs" / f"{name}_audit.log",
                        dry_run=True,
                    )
                )
            # A dry-run must still make missing artifacts explicit.  Keep the
            # commands in the receipt, but classify an arm with unresolved
            # inputs as BLOCKED rather than implying it is runnable.
            row["status"] = "BLOCKED" if blockers else "PLANNED"
            row["blockers"] = blockers
            state["arms"].append(row)
            completed[name] = row
            persist()
            continue

        # S0 is checkpoint-free and can run as soon as the observation cache
        # exists.  The other arms train then evaluate in fixed order.
        if spec.stage == "baseline":
            if not (paths["val_obs"] / "run_receipt.json").is_file():
                row.update({"status": "BLOCKED", "blockers": [f"missing validation observation cache receipt: {paths['val_obs'] / 'run_receipt.json'}"]})
                state["arms"].append(row)
                completed[name] = row
                persist()
                continue
            eval_command = _arm_eval_command(args, paths, spec, checkpoint=None, eval_output=eval_output, spatial_checkpoint=None)
            eval_record = _run_command(eval_command, cwd=project_root, log_path=output_root / "logs" / f"{name}_eval.log")
            row["commands"].append(eval_record)
            metrics_info, metrics_error = _read_result_metrics(eval_output)
            if eval_record.get("status") == "COMPLETE" and metrics_info is not None:
                row.update({"status": "COMPLETE", "metrics": metrics_info["metrics"], "metrics_status": "AVAILABLE", "metrics_artifact": metrics_info})
            else:
                row.update({"status": "FAILED" if eval_record.get("status") == "FAILED" else "BLOCKED", "blockers": [metrics_error or "S0 metrics unavailable"]})
            state["arms"].append(row)
            completed[name] = row
            persist()
            continue

        init_checkpoint: Path | None = None
        if spec.init_arm:
            init_row = completed.get(spec.init_arm)
            if init_row:
                candidate = init_row.get("train", {}).get("checkpoint")
                if candidate:
                    init_checkpoint = Path(str(candidate)).expanduser().resolve()
            if init_checkpoint is None:
                init_checkpoint = (
                    base_checkpoint
                    if spec.init_arm == "spatial_v2_base"
                    else output_root / "arms" / spec.init_arm / "train" / "final.pt"
                )
        train_command = _arm_train_command(args, paths, spec, train_output=train_output, init_checkpoint=init_checkpoint)
        train_record = _run_command(train_command, cwd=project_root, log_path=output_root / "logs" / f"{name}_train.log")
        row["commands"].append(train_record)
        checkpoint_path = train_output / "final.pt"
        row["train"] = {"status": train_record.get("status"), "checkpoint": str(checkpoint_path), "checkpoint_exists": checkpoint_path.is_file()}
        if train_record.get("status") != "COMPLETE" or not checkpoint_path.is_file():
            row.update({"status": "FAILED" if train_record.get("status") == "FAILED" else "BLOCKED", "blockers": ["training did not produce final.pt"]})
            state["arms"].append(row)
            completed[name] = row
            persist()
            if not args.keep_going:
                # Dependent arms will be marked blocked by _arm_blockers.
                continue
            continue
        spatial_path = (
            output_root / "arms" / "S5" / "train" / "final.pt"
            if name == "S6"
            else base_checkpoint
            if name in {"S3", "S4", "S5"}
            else output_root / "arms" / "S1" / "train" / "final.pt"
            if name == "S2"
            else None
        )
        eval_command = _arm_eval_command(args, paths, spec, checkpoint=checkpoint_path, eval_output=eval_output, spatial_checkpoint=spatial_path)
        eval_record = _run_command(eval_command, cwd=project_root, log_path=output_root / "logs" / f"{name}_eval.log")
        row["commands"].append(eval_record)
        metrics_info, metrics_error = _read_result_metrics(
            eval_output, preferred_method=_preferred_eval_method(name)
        )
        audit_record: dict[str, Any] | None = None
        audit_error: str | None = None
        if eval_record.get("status") == "COMPLETE" and metrics_info is not None:
            audit_command = _arm_audit_command(
                args,
                paths,
                spec,
                checkpoint=checkpoint_path,
                eval_output=eval_output,
            )
            if audit_command is not None:
                audit_record = _run_command(
                    audit_command,
                    cwd=project_root,
                    log_path=output_root / "logs" / f"{name}_audit.log",
                )
                row["commands"].append(audit_record)
                audit_path = eval_output / "audit.json"
                audit_payload: dict[str, Any] | None = None
                try:
                    audit_payload = _read_json(audit_path)
                    if audit_record.get("status") != "COMPLETE" or audit_payload.get(
                        "status"
                    ) != "PASS":
                        audit_error = (
                            f"Spring Stage-C audit did not PASS: {audit_payload.get('error', audit_payload.get('status'))}"
                        )
                    auditor = audit_payload.get("auditor")
                    expected_auditor = paths["project_root"] / "tools" / "audit_spring_epipolar.py"
                    if (
                        not isinstance(auditor, Mapping)
                        or auditor.get("path") != str(expected_auditor.resolve())
                        or auditor.get("sha256") != _sha256(expected_auditor)
                    ):
                        audit_error = "Spring Stage-C audit source identity is missing or stale"
                except SpringRunnerError as exc:
                    audit_error = str(exc)
                row["audit"] = {
                    "path": str(audit_path),
                    "status": audit_record.get("status"),
                    "payload": audit_payload,
                }
            if audit_error is None:
                row.update({"status": "COMPLETE", "metrics": metrics_info["metrics"], "metrics_status": "AVAILABLE", "metrics_artifact": metrics_info})
            else:
                row.update({"status": "BLOCKED", "metrics": metrics_info["metrics"], "metrics_status": "AVAILABLE", "metrics_artifact": metrics_info, "blockers": [audit_error]})
        else:
            row.update({"status": "FAILED" if eval_record.get("status") == "FAILED" else "BLOCKED", "blockers": [metrics_error or "evaluation metrics unavailable"]})
        state["arms"].append(row)
        completed[name] = row
        persist()

    statuses = [str(row.get("status")) for row in state["arms"]]
    if args.dry_run:
        state["status"] = "DRY_RUN"
    elif any(status == "FAILED" for status in statuses):
        state["status"] = "PARTIAL_FAILED"
    elif any(status == "BLOCKED" for status in statuses) or any(
        task.get("status") == "BLOCKED" for task in (*state["cache_tasks"], *state["geometry_tasks"])
    ):
        state["status"] = "BLOCKED"
    elif statuses and all(status == "COMPLETE" for status in statuses):
        state["status"] = "COMPLETE"
    else:
        state["status"] = "PARTIAL"
    state["finished_at_utc"] = _utc_now()
    state["summary"] = {
        "arms_requested": selected_arms,
        "arms_complete": [row["arm"] for row in state["arms"] if row.get("status") == "COMPLETE"],
        "arms_blocked": [row["arm"] for row in state["arms"] if row.get("status") == "BLOCKED"],
        "arms_failed": [row["arm"] for row in state["arms"] if row.get("status") == "FAILED"],
        "formal_coverage": bool(state["manifests"].get("formal_coverage", False)),
        "seed": SEED,
    }
    persist()
    print(json.dumps({"status": state["status"], "summary": str(summary_path), "state": str(state_path)}, sort_keys=True))
    return 0 if state["status"] in {"COMPLETE", "DRY_RUN"} else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
