#!/usr/bin/env python3
"""Resumable Spring v3.1 + FastFS common-domain runner (seed 42 only).

This runner keeps the experiment lineage explicit instead of hiding it in a
large shell script.  F0/F1 are frozen FastFS baselines; F2 is the v3.1 T1
Stage-A owner; F3 is an independent v2/K=2 GT-pose control; F4--F6 are
v3.1 T3 arms with GT/VGGT pose and optional VGGT depth; F7 is an optional
Stage-C placeholder and is never silently treated as complete.

The raw Spring manifests and raw VGGT caches remain immutable.  FastFS
observation caches are generated in a new tree, and all derived geometry is
rebuilt against that identity.  Every command, input SHA, config SHA and
result path is recorded in ``run_receipt.json`` under the selected run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
SEED = 42
ARM_ORDER = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
GT_POSE_QUALITY_SCORE_OVERRIDE = "authoritative_gt_pose"
# Canonical common-domain evaluation window.  Spring validation frames are
# 1080x1920, so this is the center-aligned x2-compatible origin.  Keep the
# origin explicit in every receipt rather than relying on evaluator defaults.
COMMON_CROP_SIZE_HR = (384, 768)
COMMON_CROP_ORIGIN_HR_XY = (576, 348)
COMMON_EVAL_DIR_NAME = "eval_common_fixed384"


def _common_eval_dir(run_root: Path, arm: str) -> Path:
    """Return the immutable fixed-crop evaluation directory for an arm."""

    return run_root / "arms" / str(arm).upper() / COMMON_EVAL_DIR_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _collect_command_records(value: Any) -> list[dict[str, Any]]:
    """Collect nested command receipts for the final run receipt.

    The runner stores command results under ``state['results']`` so each
    operation can also carry arm/split-specific metadata.  Flattening them at
    finalization keeps the receipt useful to auditors without duplicating
    every call site or losing the exact argv/log path.
    """

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("command"), list) and isinstance(
            value.get("command_text"), str
        ):
            records.append(dict(value))
        for child in value.values():
            records.extend(_collect_command_records(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            records.extend(_collect_command_records(child))
    return records


def _parse_arms(value: str) -> list[str]:
    raw = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("--arms cannot be empty")
    unknown = sorted(set(raw) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    return [arm for arm in ARM_ORDER if arm in raw]


def _run_command(
    command: Sequence[str], *, log_path: Path, cwd: Path, dry_run: bool = False
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "command": [str(item) for item in command],
        "command_text": shlex.join(str(item) for item in command),
        "log_path": str(log_path.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        record.update({"status": "DRY_RUN", "returncode": 0})
        _atomic_json(log_path.with_suffix(".json"), record)
        return record
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    record.update(
        {
            "status": "COMPLETE" if process.returncode == 0 else "FAILED",
            "returncode": int(process.returncode),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(log_path.with_suffix(".json"), record)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}); see {log_path}: {record['command_text']}"
        )
    return record


def _cache_complete(
    root: Path,
    manifest: Path,
    *,
    component: str,
    expected_config: Mapping[str, Any] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a cache is safe to reuse for this exact recipe.

    A receipt bound only to a manifest and component is insufficient for the
    F0/F1 matrix: scale, max-disp, right/left checking and checkpoint can all
    change while retaining the same output directory.  Keep this check
    intentionally local to the runner so a stale cache is rebuilt instead of
    being silently attributed to a different arm.
    """
    receipt_path = root / "run_receipt.json"
    manifest_cache_path = root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not manifest_cache_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
        identity = receipt.get("identity")
        if not isinstance(identity, Mapping) or identity.get("component") != component:
            return False
        expected_manifest_sha = _sha256(manifest)
        expected_count = sum(
            1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if (
            receipt.get("manifest_sha256") != expected_manifest_sha
            or int(receipt.get("selected_records", -1)) != expected_count
            or int(receipt.get("written_records", -1))
            + int(receipt.get("reused_records", -1))
            != expected_count
        ):
            return False
        if expected_config is not None:
            config = receipt.get("config")
            if not isinstance(config, Mapping):
                return False
            for key, value in expected_config.items():
                if config.get(key) != value:
                    return False
        if expected_identity is not None:
            for key, value in expected_identity.items():
                if identity.get(key) != value:
                    return False
        # Verify that the index itself is complete and points inside this
        # cache root.  This catches a receipt left behind by a truncated run.
        rows = [line for line in manifest_cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected_count:
            return False
        for raw in rows:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                return False
            cache_path = Path(str(value.get("cache_path", ""))).expanduser().resolve()
            try:
                cache_path.relative_to(root.resolve())
            except ValueError:
                return False
            if not cache_path.is_file():
                return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _derived_complete(
    root: Path,
    manifest: Path,
    *,
    calibrated: bool,
    expected_ffs_root: Path | None = None,
    expected_vggt_root: Path | None = None,
    sidecar: Path | None = None,
) -> bool:
    receipt_path = root / "run_receipt.json"
    cache_manifest = root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not cache_manifest.is_file():
        return False
    try:
        receipt = _read_json(receipt_path)
        counts = receipt.get("counts")
        config = receipt.get("config")
        expected_component = (
            "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
            if calibrated
            else "vggt-ffs-derived-geometry-batch"
        )
        expected_schema = 2 if calibrated else 1
        if (
            receipt.get("component") != expected_component
            or receipt.get("manifest_sha256") != _sha256(manifest)
            or not isinstance(counts, Mapping)
            or int(counts.get("selected", -1)) <= 0
            or int(counts.get("selected", -1))
            != int(counts.get("written", 0)) + int(counts.get("reused", 0))
            or not isinstance(config, Mapping)
            or config.get("schema_version") != expected_schema
        ):
            return False
        inputs = receipt.get("inputs")
        if not isinstance(inputs, Mapping):
            return False
        if expected_ffs_root is not None and Path(
            str(inputs.get("ffs_root", ""))
        ).expanduser().resolve() != expected_ffs_root.expanduser().resolve():
            return False
        if expected_vggt_root is not None and Path(
            str(inputs.get("vggt_root", ""))
        ).expanduser().resolve() != expected_vggt_root.expanduser().resolve():
            return False
        if calibrated:
            if sidecar is None or not sidecar.is_file():
                return False
            calibration = config.get("rectified_stereo_calibration")
            sidecar_receipt = sidecar.with_suffix(".receipt.json")
            if not isinstance(calibration, Mapping) or (
                calibration.get("sidecar_path") != str(sidecar.resolve())
                or calibration.get("sidecar_sha256") != _sha256(sidecar)
                or calibration.get("receipt_path") != str(sidecar_receipt.resolve())
                or calibration.get("receipt_sha256") != _sha256(sidecar_receipt)
            ):
                return False
        rows = [line for line in cache_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != int(counts.get("selected", -1)):
            return False
        for raw in rows:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                return False
            cache_path = Path(str(value.get("cache_path", ""))).expanduser().resolve()
            cache_path.relative_to(root.resolve())
            if not cache_path.is_file():
                return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _override_complete(
    root: Path,
    manifest: Path,
    source_root: Path,
    *,
    calibrated: bool,
    sidecar: Path | None = None,
) -> bool:
    """Validate a GT-pose override before reusing it.

    Pose overrides copy tensors from a source derived cache.  Reusing an old
    receipt merely because it exists can therefore mix a different FastFS
    observation, VGGT run, or calibration sidecar into the current arm.
    """

    receipt_path = root / "run_receipt.json"
    manifest_cache = root / "cache_manifest.jsonl"
    source_receipt_path = source_root / "run_receipt.json"
    source_manifest = source_root / "cache_manifest.jsonl"
    if not all(path.is_file() for path in (receipt_path, manifest_cache, source_receipt_path, source_manifest)):
        return False
    try:
        receipt = _read_json(receipt_path)
        source_receipt = _read_json(source_receipt_path)
        expected_component = (
            "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
            if calibrated
            else "vggt-ffs-derived-geometry-batch"
        )
        expected_schema = 2 if calibrated else 1
        if (
            source_receipt.get("component") != expected_component
            or source_receipt.get("schema_version") != expected_schema
        ):
            return False
        if (
            receipt.get("component") != expected_component
            or receipt.get("schema_version") != expected_schema
            or receipt.get("manifest_sha256") != _sha256(manifest)
        ):
            return False
        config = receipt.get("config")
        if not isinstance(config, Mapping):
            return False
        if config.get("pose_source") != "Spring_GT_pose":
            return False
        if config.get("depth_source") != "copied_from_vggt_derived":
            return False
        if config.get("source_derived_manifest_sha256") != _sha256(source_manifest):
            return False
        if config.get("source_derived_receipt_sha256") != _sha256(source_receipt_path):
            return False
        source_config = source_receipt.get("config")
        if not isinstance(source_config, Mapping):
            return False
        if source_receipt.get("manifest_sha256") != _sha256(manifest):
            return False
        source_counts = source_receipt.get("counts")
        if calibrated:
            # A calibrated GT override intentionally exposes an authoritative
            # Spring pose, even when the producer VGGT pose was rejected.  The
            # producer gate decision must nevertheless remain bound in the
            # override receipt, so an old/hand-edited output cannot be reused
            # merely because its files happen to exist.
            if not isinstance(source_counts, Mapping):
                return False
            source_selected = source_counts.get("selected")
            if (
                type(source_selected) is not int
                or source_selected <= 0
                or source_selected != int(source_counts.get("written", -1))
                + int(source_counts.get("reused", -1))
            ):
                return False
            for valid_name, rejected_name in (
                ("pose_valid", "pose_rejected"),
                ("static_prior_valid", "static_prior_rejected"),
            ):
                valid = source_counts.get(valid_name)
                rejected = source_counts.get(rejected_name)
                if (
                    type(valid) is not int
                    or type(rejected) is not int
                    or valid < 0
                    or rejected < 0
                    or valid + rejected != source_selected
                ):
                    return False
        counts = receipt.get("counts")
        if not isinstance(counts, Mapping):
            return False
        if int(counts.get("selected", -1)) != int(counts.get("written", 0)) + int(counts.get("reused", 0)):
            return False
        if calibrated:
            if config.get("quality_score_override") != GT_POSE_QUALITY_SCORE_OVERRIDE:
                return False
            override_summary = receipt.get("pose_override")
            if (
                not isinstance(override_summary, Mapping)
                or override_summary.get("quality_score_override")
                != GT_POSE_QUALITY_SCORE_OVERRIDE
            ):
                return False
            selected = counts.get("selected")
            if type(selected) is not int or selected <= 0:
                return False
            # GT pose is authoritative for the override view.  Source VGGT
            # rejection counts are checked separately below and must match the
            # producer receipt exactly.
            if counts.get("pose_valid") != selected or counts.get("pose_rejected") != 0:
                return False
            for name in (
                "source_pose_valid",
                "source_pose_rejected",
                "source_static_prior_valid",
                "source_static_prior_rejected",
            ):
                value = counts.get(name)
                if type(value) is not int or value < 0:
                    return False
            if (
                counts["source_pose_valid"] != source_counts["pose_valid"]
                or counts["source_pose_rejected"] != source_counts["pose_rejected"]
                or counts["source_static_prior_valid"]
                != source_counts["static_prior_valid"]
                or counts["source_static_prior_rejected"]
                != source_counts["static_prior_rejected"]
            ):
                return False
            if (
                counts["source_pose_valid"] + counts["source_pose_rejected"] != selected
                or counts["source_static_prior_valid"]
                + counts["source_static_prior_rejected"]
                != selected
            ):
                return False
        if calibrated:
            if sidecar is None or not sidecar.is_file():
                return False
            calibration = config.get("rectified_stereo_calibration")
            if not isinstance(calibration, Mapping):
                return False
            sidecar_receipt = sidecar.with_suffix(".receipt.json")
            if (
                calibration.get("sidecar_path") != str(sidecar.resolve())
                or calibration.get("sidecar_sha256") != _sha256(sidecar)
                or calibration.get("receipt_path") != str(sidecar_receipt.resolve())
                or calibration.get("receipt_sha256") != _sha256(sidecar_receipt)
            ):
                return False
        source_rows_by_target: dict[int, Mapping[str, Any]] = {}
        if calibrated:
            for raw_source in source_manifest.read_text(encoding="utf-8").splitlines():
                if not raw_source.strip():
                    return False
                source_value = json.loads(raw_source)
                if not isinstance(source_value, Mapping):
                    return False
                target = source_value.get("target_manifest_index")
                if type(target) is not int or target in source_rows_by_target:
                    return False
                source_rows_by_target[target] = source_value
            if len(source_rows_by_target) != int(source_counts["selected"]):
                return False
        rows = [line for line in manifest_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = int(counts.get("selected", -1))
        if selected <= 0 or len(rows) != selected:
            return False
        source_pose_valid_count = source_static_prior_valid_count = 0
        source_pose_rejected_count = source_static_prior_rejected_count = 0
        output_targets: set[int] = set()
        for raw in rows:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                return False
            path = Path(str(value.get("cache_path", ""))).expanduser().resolve()
            path.relative_to(root.resolve())
            if not path.is_file():
                return False
            if calibrated:
                if value.get("quality_score_override") != GT_POSE_QUALITY_SCORE_OVERRIDE:
                    return False
                if value.get("pose_valid") is not True or value.get("failure_reasons") != []:
                    return False
                source_pose_valid = value.get("source_pose_valid")
                source_static_valid = value.get("source_static_prior_valid")
                source_reasons = value.get("source_failure_reasons")
                if type(source_pose_valid) is not bool or type(source_static_valid) is not bool:
                    return False
                if (
                    not isinstance(source_reasons, list)
                    or not all(isinstance(reason, str) for reason in source_reasons)
                ):
                    return False
                target = value.get("target_manifest_index")
                if (
                    type(target) is not int
                    or target not in source_rows_by_target
                    or target in output_targets
                ):
                    return False
                source_value = source_rows_by_target[target]
                if any(
                    value.get(name) != source_value.get(name)
                    for name in ("sequence_id", "frame_id", "timestamp")
                ):
                    return False
                if (
                    source_pose_valid != source_value.get("pose_valid")
                    or source_static_valid != source_value.get("static_prior_valid")
                    or source_reasons != source_value.get("failure_reasons")
                ):
                    return False
                source_pose_valid_count += int(source_pose_valid)
                source_pose_rejected_count += int(not source_pose_valid)
                source_static_prior_valid_count += int(source_static_valid)
                source_static_prior_rejected_count += int(not source_static_valid)
                output_targets.add(target)
        if calibrated and output_targets != set(source_rows_by_target):
            return False
        if calibrated and (
            source_pose_valid_count != counts["source_pose_valid"]
            or source_pose_rejected_count != counts["source_pose_rejected"]
            or source_static_prior_valid_count != counts["source_static_prior_valid"]
            or source_static_prior_rejected_count != counts["source_static_prior_rejected"]
        ):
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _identity(root: Path) -> dict[str, Any] | None:
    path = root / "run_receipt.json"
    if not path.is_file():
        return None
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    identity = value.get("identity")
    return dict(identity) if isinstance(identity, Mapping) else None


def _config_sha(path: Path) -> str:
    return _sha256(path)


def _base_paths(run_root: Path) -> dict[str, Path]:
    old = PROJECT_ROOT / "runs" / "spring_seed42_primary"
    return {
        "train_manifest": old / "manifests" / "train.jsonl",
        "val_manifest": old / "manifests" / "validation.jsonl",
        "train_teacher": old / "cache" / "train" / "teacher",
        "val_teacher": old / "cache" / "validation" / "teacher",
        "train_vggt": old / "cache" / "train" / "vggt",
        "val_vggt": old / "cache" / "validation" / "vggt",
        "train_ffs_half": run_root / "cache" / "train" / "ffs_half" / "observation",
        "val_ffs_half": run_root / "cache" / "validation" / "ffs_half" / "observation",
        "val_ffs_full": run_root / "cache" / "validation" / "ffs_full416" / "observation",
        "train_derived_legacy": run_root / "cache" / "train" / "derived_vggt_pose_depth",
        "val_derived_legacy": run_root / "cache" / "validation" / "derived_vggt_pose_depth",
        "train_derived_cal": run_root / "cache" / "train" / "derived_calibrated_vggt_pose_depth",
        "val_derived_cal": run_root / "cache" / "validation" / "derived_calibrated_vggt_pose_depth",
        "train_gt_legacy": run_root / "cache" / "train" / "derived_gt_pose_no_depth",
        "val_gt_legacy": run_root / "cache" / "validation" / "derived_gt_pose_no_depth",
        "train_gt_cal": run_root / "cache" / "train" / "derived_calibrated_gt_pose_depth",
        "val_gt_cal": run_root / "cache" / "validation" / "derived_calibrated_gt_pose_depth",
        "train_sidecar": run_root / "sidecars" / "train_calibration.jsonl",
        "val_sidecar": run_root / "sidecars" / "validation_calibration.jsonl",
        "endpoint_list": run_root / "manifests" / "common_endpoints.json",
        "logs": run_root / "logs",
        "arms": run_root / "arms",
    }


def _sidecar_complete(
    sidecar: Path,
    receipt: Path,
    manifest: Path,
    audit: Path,
) -> bool:
    """Check the immutable calibration sidecar before reusing it."""

    if not all(path.is_file() for path in (sidecar, receipt, manifest, audit)):
        return False
    try:
        payload = _read_json(receipt)
        source = payload.get("source")
        output = payload.get("output")
        counts = payload.get("counts")
        derivation = payload.get("derivation")
        expected_manifest_sha = _sha256(manifest)
        expected_audit_sha = _sha256(audit)
        expected_records = sum(
            1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if (
            payload.get("component") != "rectified-stereo-calibration-receipt"
            or payload.get("contract_version") != "stored_rectified_virtual_cameras_v1"
            or payload.get("status") != "PASS"
            or not isinstance(source, Mapping)
            or not isinstance(output, Mapping)
            or not isinstance(counts, Mapping)
            or not isinstance(derivation, Mapping)
            or source.get("manifest_path") != str(manifest.resolve())
            or source.get("manifest_sha256") != expected_manifest_sha
            or source.get("pixel_audit_path") != str(audit.resolve())
            or source.get("pixel_audit_sha256") != expected_audit_sha
            or output.get("sidecar_path") != str(sidecar.resolve())
            or output.get("sidecar_sha256") != _sha256(sidecar)
            or int(counts.get("records", -1)) != expected_records
            or derivation.get("method") != "spring_rectified_native_v1"
        ):
            return False
        rows = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
        return len(rows) == expected_records
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _ensure_sidecars(paths: Mapping[str, Path], *, dry_run: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    audit = PROJECT_ROOT / "reports" / "spring_epipolar_rectification_primary.json"
    for split, manifest_key, output_key in (
        ("train", "train_manifest", "train_sidecar"),
        ("validation", "val_manifest", "val_sidecar"),
    ):
        sidecar = paths[output_key]
        receipt = sidecar.with_suffix(".receipt.json")
        if _sidecar_complete(
            sidecar,
            receipt,
            paths[manifest_key],
            audit,
        ):
            result[split] = {"status": "REUSED", "sidecar": str(sidecar)}
            continue
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "build_stereo_calibration.py"),
            "--manifest",
            str(paths[manifest_key]),
            "--pixel-audit",
            str(audit),
            "--output",
            str(sidecar),
            "--receipt",
            str(receipt),
            "--spring-native",
        ]
        log = paths["logs"] / f"sidecar_{split}.log"
        result[split] = _run_command(command, log_path=log, cwd=PROJECT_ROOT, dry_run=dry_run)
    return result


def _ensure_cache(
    paths: Mapping[str, Path], *, full: bool, split: str, device: str, dry_run: bool
) -> dict[str, Any]:
    split_key = "train" if split == "train" else "val"
    manifest = paths[f"{split_key}_manifest"]
    root = paths["val_ffs_full" if full else f"{split_key}_ffs_half"]
    parent = root.parent
    checkpoint = PROJECT_ROOT / "checkpoints" / "ffs" / "20-30-48" / "model_best_bp2_serialize.pth"
    expected_config = {
        "role": "observation",
        "scale": 1 if full else 2,
        "resolution_mode": "full_resolution_observation" if full else "mvp",
        "iterations": 4,
        "max_disp": 416 if full else 192,
        "volume_backend": "pytorch1",
        "right_left_check": True,
        "checkpoint_label": "20-30-48",
        "expected_checkpoint_label": "20-30-48",
        "provisional_checkpoint_role": False,
        "missing_normalize": "error",
    }
    expected_identity: dict[str, Any] = {"component": "ffs-observation"}
    cache_reusable = checkpoint.is_file()
    if cache_reusable:
        expected_identity["checkpoint_sha256"] = _sha256(checkpoint)
    if not dry_run and cache_reusable and _cache_complete(
        root,
        manifest,
        component="ffs-observation",
        expected_config=expected_config,
        expected_identity=expected_identity,
    ):
        return {"status": "REUSED", "root": str(root), "identity": _identity(root)}
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "tools" / "cache_ffs.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(parent),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-label",
        "20-30-48",
        "--role",
        "observation",
        "--scale",
        "1" if full else "2",
        "--iterations",
        "4",
        "--max-disp",
        "416" if full else "192",
        "--volume-backend",
        "pytorch1",
        "--right-left-check",
        "--device",
        device,
    ]
    if full:
        command.append("--allow-full-resolution-observation")
    log = paths["logs"] / f"cache_{split}_{'full' if full else 'half'}.log"
    result = _run_command(command, log_path=log, cwd=PROJECT_ROOT, dry_run=dry_run)
    result.update({"root": str(root), "identity": _identity(root)})
    return result


def _ensure_derived(
    paths: Mapping[str, Path], *, calibrated: bool, split: str, dry_run: bool
) -> dict[str, Any]:
    split_key = "train" if split == "train" else "val"
    manifest = paths[f"{split_key}_manifest"]
    vggt = paths[f"{split_key}_vggt"]
    ffs = paths[f"{split_key}_ffs_half"]
    root = paths[f"{split_key}_derived_cal" if calibrated else f"{split_key}_derived_legacy"]
    if not dry_run and _derived_complete(
        root,
        manifest,
        calibrated=calibrated,
        expected_ffs_root=ffs,
        expected_vggt_root=vggt,
        sidecar=paths[f"{split_key}_sidecar"] if calibrated else None,
    ):
        return {"status": "REUSED", "root": str(root)}
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "tools" / "derive_geometry_manifest.py"),
        "--vggt-root",
        str(vggt),
        "--ffs-root",
        str(ffs),
        "--output",
        str(root),
        "--cache-dtype",
        "float32",
    ]
    if calibrated:
        command.extend(
            (
                "--rectified-calibration-sidecar",
                str(paths[f"{split_key}_sidecar"]),
                "--rectified-calibration-receipt",
                str(paths[f"{split_key}_sidecar"].with_suffix(".receipt.json")),
            )
        )
    log = paths["logs"] / f"derive_{split}_{'calibrated' if calibrated else 'legacy'}.log"
    result = _run_command(command, log_path=log, cwd=PROJECT_ROOT, dry_run=dry_run)
    result["root"] = str(root)
    return result


def _ensure_gt_override(
    paths: Mapping[str, Path], *, calibrated: bool, split: str, dry_run: bool
) -> dict[str, Any]:
    split_key = "train" if split == "train" else "val"
    source = paths[f"{split_key}_derived_cal" if calibrated else f"{split_key}_derived_legacy"]
    output = paths[f"{split_key}_gt_cal" if calibrated else f"{split_key}_gt_legacy"]
    if not dry_run and _override_complete(
        output,
        paths[f"{split_key}_manifest"],
        source,
        calibrated=calibrated,
        sidecar=paths[f"{split_key}_sidecar"] if calibrated else None,
    ):
        return {"status": "REUSED", "root": str(output)}
    if calibrated:
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "override_spring_pose_calibrated.py"),
            "--manifest",
            str(paths[f"{split_key}_manifest"]),
            "--source-root",
            str(source),
            "--calibration-sidecar",
            str(paths[f"{split_key}_sidecar"]),
            "--output",
            str(output),
        ]
    else:
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "override_spring_pose.py"),
            "--manifest",
            str(paths[f"{split_key}_manifest"]),
            "--source-root",
            str(source),
            "--output",
            str(output),
        ]
    # A failed reuse audit means the existing tree belongs to an older
    # override recipe (for example, before the calibrated quality-audit
    # marker was added).  Ask the producer to atomically rewrite those known
    # output records; otherwise its per-record identity check would reject
    # the stale files before it had a chance to repair them.
    if output.exists():
        command.append("--overwrite")
    log = paths["logs"] / f"override_{split}_{'calibrated' if calibrated else 'legacy'}.log"
    result = _run_command(command, log_path=log, cwd=PROJECT_ROOT, dry_run=dry_run)
    result["root"] = str(output)
    return result


def _train_command(
    paths: Mapping[str, Path],
    *,
    arm: str,
    config: Path,
    output: Path,
    device: str,
    init: Path | None,
    derived: Path | None,
    sidecar: Path | None,
    steps: int | None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(config),
        "--manifest",
        str(paths["train_manifest"]),
        "--observation-cache-root",
        str(paths["train_ffs_half"]),
        "--teacher-cache-root",
        str(paths["train_teacher"]),
        "--output-dir",
        str(output),
        "--device",
        device,
    ]
    if derived is not None:
        command.extend(("--derived-cache-root", str(derived)))
    if sidecar is not None:
        command.extend(("--calibration-sidecar", str(sidecar)))
    if init is not None:
        command.extend(("--init-from", str(init)))
    if steps is not None:
        key = "train.steps_spatial" if arm in {"F2", "F3_init"} else "train.steps"
        command.append(f"{key}={steps}")
    return command


def _eval_command(
    paths: Mapping[str, Path],
    *,
    arm: str,
    config: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    derived: Path | None,
    sidecar: Path | None,
    limit: int,
) -> list[str]:
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "eval.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(paths["val_manifest"]),
        "--observation-cache-root",
        str(paths["val_ffs_half"]),
        "--teacher-cache-root",
        str(paths["val_teacher"]),
        "--output",
        str(output),
        "--device",
        device,
        "--crop-mode",
        # All trainable arms share the canonical 384x768 fixed evaluation
        # window.  Full-image temporal evaluation exceeds the 24-GiB budget
        # once the top-K hidden warp is materialized.
        "fixed",
        "--crop-origin",
        str(COMMON_CROP_ORIGIN_HR_XY[0]),
        str(COMMON_CROP_ORIGIN_HR_XY[1]),
        "--spring-endpoint-index-list",
        str(paths["endpoint_list"]),
        "--spring-native-metrics",
        "--visualization-samples",
        "0",
    ]
    if derived is not None:
        command.extend(("--derived-cache-root", str(derived)))
    if sidecar is not None:
        command.extend(("--calibration-sidecar", str(sidecar)))
    if arm not in {"F2"}:
        # F3--F6 are trained on the disjoint train sequences and evaluated on
        # the validation sequences.  They are formal holdout runs even when
        # the common endpoint file is a bounded 1302-window domain.  The
        # non-holdout smoke override would incorrectly downgrade their
        # lineage to NON_HOLDOUT_SMOKE, so keep only the explicit endpoint
        # limit here.  The endpoint index itself remains the authoritative
        # common-domain selection.  Keep the limit last for stable command
        # receipts and simple audit tooling.
        command.extend(("--limit", str(limit)))
    return command


def _write_lineage(paths: Mapping[str, Path], run_root: Path, arms: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "spring_v31_ffs_common_domain_v1",
        "seed": SEED,
        "arms": list(arms),
        "manifests": {},
        "sidecars": {},
        "caches": {},
    }
    for key in ("train_manifest", "val_manifest", "endpoint_list", "train_sidecar", "val_sidecar"):
        path = paths[key]
        if path.is_file():
            payload["manifests" if "manifest" in key or "endpoint" in key else "sidecars"][key] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
    for key in (
        "train_ffs_half",
        "val_ffs_half",
        "val_ffs_full",
        "train_derived_legacy",
        "val_derived_legacy",
        "train_derived_cal",
        "val_derived_cal",
        "train_gt_legacy",
        "val_gt_legacy",
        "train_gt_cal",
        "val_gt_cal",
    ):
        identity = _identity(paths[key])
        if identity is not None:
            payload["caches"][key] = {
                "root": str(paths[key].resolve()),
                "receipt_sha256": _sha256(paths[key] / "run_receipt.json"),
                "identity": identity,
            }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(run_root / "lineage.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "spring_v31_ffs")
    parser.add_argument("--arms", default=",".join(ARM_ORDER[:-1]))
    parser.add_argument("--device-f2", default="cuda:0")
    parser.add_argument("--device-f3", default="cuda:1")
    parser.add_argument("--device-temporal", default="cuda:0")
    parser.add_argument("--steps", type=int, help="bounded pilot steps for trainable arms")
    parser.add_argument("--endpoint-limit", type=int, default=1302)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-f7", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    global PROJECT_ROOT
    PROJECT_ROOT = args.project_root.expanduser().resolve()
    if args.run_root.is_absolute():
        run_root = args.run_root.expanduser().resolve()
    else:
        run_root = (PROJECT_ROOT / args.run_root).resolve()
    arms = _parse_arms(args.arms)
    if "F7" in arms and not args.run_f7:
        raise ValueError("F7 is optional; pass --run-f7 explicitly")
    if args.endpoint_limit <= 0:
        raise ValueError("--endpoint-limit must be positive")
    paths = _base_paths(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "spring_v31_ffs_common_domain_v1",
        "seed": SEED,
        "arms": arms,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "commands": [],
        "results": {},
        "protocol_contract": {
            "evaluation_crop_mode": "fixed",
            "evaluation_crop_size_hr_hw": list(COMMON_CROP_SIZE_HR),
            "evaluation_crop_origin_hr_xy": list(COMMON_CROP_ORIGIN_HR_XY),
            "endpoint_limit": int(args.endpoint_limit),
            "endpoint_selection_kind": "spring_common_endpoint_index",
        },
        "input_lineage": {
            "train_manifest": {
                "path": str(paths["train_manifest"].resolve()),
                "sha256": _sha256(paths["train_manifest"])
                if paths["train_manifest"].is_file()
                else None,
            },
            "validation_manifest": {
                "path": str(paths["val_manifest"].resolve()),
                "sha256": _sha256(paths["val_manifest"])
                if paths["val_manifest"].is_file()
                else None,
            },
            "endpoint_list": {
                "path": str(paths["endpoint_list"].resolve()),
                "sha256": _sha256(paths["endpoint_list"])
                if paths["endpoint_list"].is_file()
                else None,
            },
        },
    }
    _atomic_json(run_root / "run_receipt.json", state)

    # Immutable sidecars and the new FastFS observations.
    state["results"]["sidecars"] = _ensure_sidecars(paths, dry_run=args.dry_run)
    if not args.skip_cache:
        if any(arm in arms for arm in ("F0", "F1", "F2", "F3", "F4", "F5", "F6")):
            state["results"]["cache_train_half"] = _ensure_cache(
                paths, full=False, split="train", device=args.device_f2, dry_run=args.dry_run
            )
            state["results"]["cache_val_half"] = _ensure_cache(
                paths, full=False, split="validation", device=args.device_temporal, dry_run=args.dry_run
            )
        if "F0" in arms:
            state["results"]["cache_val_full"] = _ensure_cache(
                paths, full=True, split="validation", device=args.device_temporal, dry_run=args.dry_run
            )

    # Geometry needed by F3--F6.  The legacy and calibrated trees share the
    # same new FastFS identity but remain separate cache contracts.
    temporal_arms = set(arms).intersection({"F3", "F4", "F5", "F6"})
    if temporal_arms and not args.skip_derived:
        state["results"]["derive_train_legacy"] = _ensure_derived(
            paths, calibrated=False, split="train", dry_run=args.dry_run
        )
        state["results"]["derive_val_legacy"] = _ensure_derived(
            paths, calibrated=False, split="validation", dry_run=args.dry_run
        )
        state["results"]["override_train_legacy"] = _ensure_gt_override(
            paths, calibrated=False, split="train", dry_run=args.dry_run
        )
        state["results"]["override_val_legacy"] = _ensure_gt_override(
            paths, calibrated=False, split="validation", dry_run=args.dry_run
        )
        state["results"]["derive_train_calibrated"] = _ensure_derived(
            paths, calibrated=True, split="train", dry_run=args.dry_run
        )
        state["results"]["derive_val_calibrated"] = _ensure_derived(
            paths, calibrated=True, split="validation", dry_run=args.dry_run
        )
        state["results"]["override_train_calibrated"] = _ensure_gt_override(
            paths, calibrated=True, split="train", dry_run=args.dry_run
        )
        state["results"]["override_val_calibrated"] = _ensure_gt_override(
            paths, calibrated=True, split="validation", dry_run=args.dry_run
        )

    # Build the final common endpoint list after derived coverage is known.
    # Recompute on every non-dry run.  A stale endpoint file from a previous
    # screening pass (for example the 1318 causal endpoints before derived
    # student coverage is applied) must never silently become the F0--F6
    # comparison domain.
    if temporal_arms and not args.dry_run:
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "build_spring_endpoint_index.py"),
            "--manifest",
            str(paths["val_manifest"]),
            "--output",
            str(paths["endpoint_list"]),
            "--derived-cache-root",
            str(paths["val_derived_cal"]),
            # Keep the fixed common-domain cardinality explicit.  Without
            # this bound the causal/derived intersection is 1318 endpoints,
            # which would silently invalidate the existing 1302-domain F0--F6
            # reports and fixed-384 protocol.
            "--limit",
            str(args.endpoint_limit),
        ]
        state["results"]["endpoint_list"] = _run_command(
            command, log_path=paths["logs"] / "endpoint_list.log", cwd=PROJECT_ROOT
        )

    # Frozen baselines.
    if "F0" in arms and not args.dry_run:
        output = _common_eval_dir(run_root, "F0")
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "eval_spring_baseline.py"),
            "--manifest",
            str(paths["val_manifest"]),
            "--observation-cache-root",
            str(paths["val_ffs_full"]),
            "--output",
            str(output),
            "--cache-role",
            "observation",
            "--arm",
            "F0",
            "--spring-endpoint-index-list",
            str(paths["endpoint_list"]),
            "--crop-mode",
            "fixed",
            "--crop-size",
            str(COMMON_CROP_SIZE_HR[0]),
            str(COMMON_CROP_SIZE_HR[1]),
            "--crop-origin",
            str(COMMON_CROP_ORIGIN_HR_XY[0]),
            str(COMMON_CROP_ORIGIN_HR_XY[1]),
        ]
        state["results"]["F0"] = _run_command(
            command, log_path=paths["logs"] / "F0_eval.log", cwd=PROJECT_ROOT
        )
    if "F1" in arms and not args.dry_run:
        output = _common_eval_dir(run_root, "F1")
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools" / "eval_spring_baseline.py"),
            "--manifest",
            str(paths["val_manifest"]),
            "--observation-cache-root",
            str(paths["val_ffs_half"]),
            "--output",
            str(output),
            "--cache-role",
            "observation",
            "--arm",
            "F1",
            "--spring-endpoint-index-list",
            str(paths["endpoint_list"]),
            "--crop-mode",
            "fixed",
            "--crop-size",
            str(COMMON_CROP_SIZE_HR[0]),
            str(COMMON_CROP_SIZE_HR[1]),
            "--crop-origin",
            str(COMMON_CROP_ORIGIN_HR_XY[0]),
            str(COMMON_CROP_ORIGIN_HR_XY[1]),
        ]
        state["results"]["F1"] = _run_command(
            command, log_path=paths["logs"] / "F1_eval.log", cwd=PROJECT_ROOT
        )

    # F2 v3.1 Stage-A and the separate v2 Stage-A owner for F3.
    f2_config = PROJECT_ROOT / "configs" / "spring_v31_ffs" / "F2.yaml"
    f3_init_config = PROJECT_ROOT / "configs" / "spring_v31_ffs" / "F3_init.yaml"
    if "F2" in arms or "F4" in arms or "F5" in arms or "F6" in arms:
        f2_out = paths["arms"] / "F2" / "train"
        f2_command = _train_command(
            paths,
            arm="F2",
            config=f2_config,
            output=f2_out,
            device=args.device_f2,
            init=None,
            derived=None,
            sidecar=paths["train_sidecar"],
            steps=args.steps,
        )
        if not args.dry_run and not (f2_out / "final.pt").is_file():
            state["results"]["F2_train"] = _run_command(
                f2_command, log_path=paths["logs"] / "F2_train.log", cwd=PROJECT_ROOT
            )
        f2_ckpt = f2_out / "final.pt"
        if "F2" in arms and not args.dry_run and f2_ckpt.is_file():
            eval_out = _common_eval_dir(run_root, "F2")
            state["results"]["F2_eval"] = _run_command(
                _eval_command(
                    paths,
                    arm="F2",
                    config=f2_config,
                    checkpoint=f2_ckpt,
                    output=eval_out,
                    device=args.device_f2,
                    derived=None,
                    sidecar=paths["val_sidecar"],
                    limit=args.endpoint_limit,
                ),
                log_path=paths["logs"] / "F2_eval.log",
                cwd=PROJECT_ROOT,
            )

    if "F3" in arms:
        f3_init_out = paths["arms"] / "F3_init" / "train"
        f3_init_ckpt = f3_init_out / "final.pt"
        if not args.dry_run and not f3_init_ckpt.is_file():
            state["results"]["F3_init_train"] = _run_command(
                _train_command(
                    paths,
                    arm="F3_init",
                    config=f3_init_config,
                    output=f3_init_out,
                    device=args.device_f3,
                    init=None,
                    derived=None,
                    sidecar=None,
                    steps=args.steps,
                ),
                log_path=paths["logs"] / "F3_init_train.log",
                cwd=PROJECT_ROOT,
            )
        f3_out = paths["arms"] / "F3" / "train"
        f3_config = PROJECT_ROOT / "configs" / "spring_v31_ffs" / "F3.yaml"
        f3_ckpt = f3_out / "final.pt"
        if not args.dry_run and not f3_ckpt.is_file():
            state["results"]["F3_train"] = _run_command(
                _train_command(
                    paths,
                    arm="F3",
                    config=f3_config,
                    output=f3_out,
                    device=args.device_f3,
                    init=f3_init_ckpt,
                    derived=paths["train_gt_legacy"],
                    sidecar=None,
                    steps=args.steps,
                ),
                log_path=paths["logs"] / "F3_train.log",
                cwd=PROJECT_ROOT,
            )
        if not args.dry_run and f3_ckpt.is_file():
            state["results"]["F3_eval"] = _run_command(
                _eval_command(
                    paths,
                    arm="F3",
                    config=f3_config,
                    checkpoint=f3_ckpt,
                    output=_common_eval_dir(run_root, "F3"),
                    device=args.device_f3,
                    derived=paths["val_gt_legacy"],
                    sidecar=None,
                    limit=args.endpoint_limit,
                ),
                log_path=paths["logs"] / "F3_eval.log",
                cwd=PROJECT_ROOT,
            )

    # F4/F5/F6 all use the same v3.1 F2 Stage-A initialization.  This keeps
    # the pose/depth effect attributable to the declared arm switch and obeys
    # train.py's strict Stage-B-from-Stage-A checkpoint contract.
    f2_ckpt = paths["arms"] / "F2" / "train" / "final.pt"
    for arm, derived_train_key, derived_val_key, config_name in (
        ("F4", "train_gt_cal", "val_gt_cal", "F4.yaml"),
        ("F5", "train_gt_cal", "val_gt_cal", "F5.yaml"),
        ("F6", "train_derived_cal", "val_derived_cal", "F6.yaml"),
    ):
        if arm not in arms:
            continue
        config = PROJECT_ROOT / "configs" / "spring_v31_ffs" / config_name
        train_out = paths["arms"] / arm / "train"
        ckpt = train_out / "final.pt"
        if not args.dry_run and not ckpt.is_file():
            state["results"][f"{arm}_train"] = _run_command(
                _train_command(
                    paths,
                    arm=arm,
                    config=config,
                    output=train_out,
                    device=args.device_temporal,
                    init=f2_ckpt,
                    derived=paths[derived_train_key],
                    sidecar=paths["train_sidecar"],
                    steps=args.steps,
                ),
                log_path=paths["logs"] / f"{arm}_train.log",
                cwd=PROJECT_ROOT,
            )
        if not args.dry_run and ckpt.is_file():
            state["results"][f"{arm}_eval"] = _run_command(
                _eval_command(
                    paths,
                    arm=arm,
                    config=config,
                    checkpoint=ckpt,
                    output=_common_eval_dir(run_root, arm),
                    device=args.device_temporal,
                    derived=paths[derived_val_key],
                    sidecar=paths["val_sidecar"],
                    limit=args.endpoint_limit,
                ),
                log_path=paths["logs"] / f"{arm}_eval.log",
                cwd=PROJECT_ROOT,
            )

    if "F7" in arms:
        state["results"]["F7"] = {
            "status": "OPTIONAL_NOT_IMPLEMENTED",
            "reason": "Stage-C v3.1 Spring adapter is not part of the common-domain baseline queue",
        }
    state["status"] = "COMPLETE"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    # Preserve every exact argv/log receipt, including commands nested under
    # split/arm result dictionaries.  This used to remain an empty list even
    # after a successful run, which made a later audit unable to reconstruct
    # how an artifact was produced.
    state["commands"] = _collect_command_records(state["results"])
    outputs: dict[str, Any] = {}
    for arm in arms:
        if arm == "F7":
            continue
        evaluation_dir = _common_eval_dir(run_root, arm)
        checkpoint_path = paths["arms"] / arm / "train" / "final.pt"
        metrics_path = evaluation_dir / "metrics.json"
        outputs[arm] = {
            "evaluation_dir": str(evaluation_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "metrics_exists": metrics_path.is_file(),
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "exists": checkpoint_path.is_file(),
                "sha256": _sha256(checkpoint_path)
                if checkpoint_path.is_file()
                else None,
            },
        }
    state["outputs"] = outputs
    state["lineage"] = _write_lineage(paths, run_root, arms)
    _atomic_json(run_root / "run_receipt.json", state)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        # Keep a durable failure receipt even when a cache/evaluation command
        # fails before ``run()`` reaches its normal finalization block.
        project_root = args.project_root.expanduser().resolve()
        run_root = (
            args.run_root.expanduser().resolve()
            if args.run_root.is_absolute()
            else (project_root / args.run_root).resolve()
        )
        receipt_path = run_root / "run_receipt.json"
        try:
            state = _read_json(receipt_path) if receipt_path.is_file() else {}
            state.update(
                {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            if isinstance(state.get("results"), Mapping):
                state["commands"] = _collect_command_records(state["results"])
            _atomic_json(receipt_path, state)
        except Exception:
            # Preserve the original exception/traceback if the failure receipt
            # itself cannot be written (for example, a read-only run root).
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
