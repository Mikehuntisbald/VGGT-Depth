#!/usr/bin/env python3
"""Compose a read-only Spring v3.1 + Fast-FoundationStereo arm report.

This utility deliberately only *reads* arm receipts and metrics.  It is safe
to run while a train process is active: no checkpoint, cache, log, or process
state under ``runs/`` is modified.  The two requested report files are the
only outputs written.

The evaluator has two metric layouts in the wild:

* F0/F1 frozen observations expose the exact Spring fields under a top-level
  ``metrics`` object.
* F2--F7 trainable arms expose method rows (and, for recent evaluations, an
  exact ``spring_native_metrics`` side channel).

The report keeps a flat ``metrics`` mapping for convenient table consumers and
also stores ``metric_receipts`` with value/numerator/count/valid metadata.  A
missing value is represented by JSON ``null``; it is never imputed from a
different crop or pseudo-GT domain.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "spring_v31_ffs"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "spring_v31_ffs_common_domain_report.json"
DEFAULT_MD = PROJECT_ROOT / "reports" / "spring_v31_ffs_common_domain_report.md"

ARMS: tuple[str, ...] = tuple(f"F{i}" for i in range(8))
EXPECTED_ENDPOINT_COUNT = 1302
EXPECTED_CROP_SIZE = [384, 768]
EXPECTED_CROP_ORIGIN = [576, 348]

REQUIRED_METRICS: tuple[str, ...] = (
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

TOPK_METRICS: tuple[str, ...] = (
    "age_2_survival_rate",
    "unique_age_fraction",
    "phase_variance",
    "candidate_depth_spread",
    "attention_entropy",
    "gain_by_fractional_phase_bucket",
    "gain_by_camera_motion_bucket",
)

ARM_PURPOSES: dict[str, str] = {
    "F0": "Full-resolution FFS",
    "F1": "Half-resolution FFS + bilinear",
    "F2": "Half-resolution FFS + v3.1 T1",
    "F3": "Half-resolution FFS + v2/K=2 T3 control, GT pose",
    "F4": "Half-resolution FFS + v3.1 T3, GT pose",
    "F5": "F4 + VGGT depth prior, GT pose",
    "F6": "F5 + VGGT pose",
    "F7": "F6 + optional Stage C",
}

PRIMARY_METHODS: dict[str, tuple[str, ...]] = {
    "F2": ("T1",),
    "F3": ("T3",),
    "F4": ("T3",),
    "F5": ("T3_VGGT", "T3"),
    "F6": ("T3_VGGT", "T3_VGGT_epipolar", "T3"),
    "F7": ("T3_VGGT_epipolar", "T3_VGGT", "T3"),
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "overall_epe": ("overall_epe", "epe_px"),
    "overall_1px": ("overall_1px", "bad_1"),
    "high_detail_epe": ("high_detail_epe",),
    "high_detail_1px": ("high_detail_1px",),
    "low_detail_epe": ("low_detail_epe",),
    "matched_epe": ("matched_epe",),
    "unmatched_completion_1px": ("unmatched_completion_1px",),
    "unmatched_completion_2px": ("unmatched_completion_2px",),
    "rigid_temporal_residual_error": ("rigid_temporal_residual_error",),
    "non_rigid_temporal_residual_error": ("non_rigid_temporal_residual_error",),
    "boundary_epe": ("boundary_epe", "boundary_epe_px"),
    # ``trusted_region_epe_px`` is an older evaluator spelling.  It is only a
    # fallback when an exact native field is absent, and the source alias is
    # retained in the receipt metadata so it cannot be mistaken for a new
    # measurement.
    "ffs_trusted_measurement_error": (
        "ffs_trusted_measurement_error",
        "trusted_region_epe_px",
    ),
    "negative_rate": ("negative_rate", "output_negative_rate"),
    "zero_rate": ("zero_rate", "output_zero_rate"),
    "invalid_rate": ("invalid_rate", "output_invalid_rate"),
}

TOPK_ALIASES: dict[str, tuple[str, ...]] = {
    "age_2_survival_rate": ("age_2_survival_rate", "age2_survival_rate"),
    "unique_age_fraction": ("unique_age_fraction",),
    "phase_variance": ("phase_variance", "fractional_phase_variance"),
    "candidate_depth_spread": (
        "candidate_depth_spread",
        "candidate_depth_spread_m",
    ),
    "attention_entropy": (
        "attention_entropy",
        "metric_attention_weight_entropy",
        "topk_weight_entropy",
    ),
    "gain_by_fractional_phase_bucket": ("gain_by_fractional_phase_bucket",),
    "gain_by_camera_motion_bucket": ("gain_by_camera_motion_bucket",),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _resolve_path(value: Any, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value)).expanduser()
    except (TypeError, ValueError):
        return None
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_present(*values: Any) -> Any:
    """Return the first value that is not ``None`` (false is meaningful)."""

    for value in values:
        if value is not None:
            return value
    return None


def _nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _finite(value: Any) -> Any:
    # JSON disallows NaN/Inf in the emitted report.  Avoid importing numpy just
    # to check scalar values; bools are intentionally left untouched.
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def _unwrap_metric(raw: Any) -> tuple[Any, Any, Any, bool | None]:
    """Return value, numerator, count, valid from either metric layout."""

    if isinstance(raw, Mapping) and "value" in raw:
        value = _finite(raw.get("value"))
        numerator = _finite(raw.get("numerator"))
        count = raw.get("count")
        valid = _as_bool(raw.get("valid"))
        return value, numerator, count, valid
    return _finite(raw), None, None, None


def _extract_metric(container: Mapping[str, Any] | None, aliases: Sequence[str]) -> dict[str, Any]:
    if not isinstance(container, Mapping):
        return {"value": None, "numerator": None, "count": None, "valid": False, "source_key": None}
    for key in aliases:
        if key not in container:
            continue
        value, numerator, count, valid = _unwrap_metric(container.get(key))
        if numerator is None and not isinstance(container.get(key), Mapping):
            numerator = _finite(container.get(f"{key}_numerator"))
        if count is None and not isinstance(container.get(key), Mapping):
            count = container.get(f"{key}_count")
        if valid is None:
            valid = value is not None and (count is None or count != 0)
        return {
            "value": value,
            "numerator": numerator,
            "count": count,
            "valid": bool(valid),
            "source_key": key,
        }
    return {"value": None, "numerator": None, "count": None, "valid": False, "source_key": None}


def _extract_topk(container: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    if not isinstance(container, Mapping):
        return values, sources
    for target, aliases in TOPK_ALIASES.items():
        for key in aliases:
            if key not in container:
                continue
            raw = container[key]
            # Bucket gains are mappings and must remain mappings.  Scalar
            # MetricResult-style wrappers are unwrapped for consistency.
            if isinstance(raw, Mapping) and "value" in raw:
                raw = raw.get("value")
            values[target] = _finite(raw)
            sources[target] = key
            break
    return values, sources


def _simple_config(path: Path) -> dict[str, Any]:
    """Read the small scalar subset needed when an arm has no metrics yet."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, Any] = {}
    scalar_keys = (
        "experiment",
        "arm",
        "protocol",
        "stage",
        "backbone",
        "resolution_mode",
        "cache_role",
        "cache_scale",
        "checkpoint_label",
        "max_disp",
        "description",
        "base_arm",
        "status",
        "defaults_from",
    )
    for key in scalar_keys:
        match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$", text, re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).split(" #", 1)[0].strip()
        if raw.lower() in {"true", "false"}:
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = raw.strip("'\"")
    for key in ("use_vggt_depth", "use_vggt_pose", "use_history", "epipolar_refinement"):
        matches = re.findall(rf"^\s*{re.escape(key)}\s*:\s*(true|false)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if matches:
            out[key] = matches[-1].lower() == "true"
    match = re.findall(r"^\s*temporal_pose_source\s*:\s*(\S+)\s*$", text, re.MULTILINE)
    if match:
        out["temporal_pose_source"] = match[-1].strip("'\"")
    return out


def _records_evaluated(data: Mapping[str, Any]) -> int | None:
    direct = _as_int(data.get("records_evaluated"))
    if direct is not None:
        return direct
    selection = data.get("selection")
    if isinstance(selection, Mapping):
        direct = _as_int(selection.get("records"))
        if direct is not None:
            return direct
    metrics = data.get("metrics")
    if isinstance(metrics, Mapping):
        direct = _as_int(metrics.get("frames"))
        if direct is not None:
            return direct
    native = data.get("spring_native_metrics")
    if isinstance(native, Mapping):
        direct = _as_int(native.get("records"))
        if direct is not None:
            return direct
    return None


def _checkpoint_info(data: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        checkpoint = {}
    completion = data.get("checkpoint_training_completion")
    if not isinstance(completion, Mapping):
        completion = {}
    return {
        "path": checkpoint.get("path"),
        "sha256": checkpoint.get("checkpoint_sha256") or checkpoint.get("sha256"),
        "step": _as_int(checkpoint.get("step")),
        "git_hash": checkpoint.get("git_hash"),
        "parameter_count": _as_int(checkpoint.get("parameter_count") or data.get("parameter_count")),
        "training_completion": dict(completion) if completion else None,
    }


def _candidate_summary(path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    crop_mode = data.get("crop_mode")
    hr_crop = _as_list(data.get("hr_crop"))
    crop_contract = data.get("crop_contract")
    if hr_crop is None and isinstance(crop_contract, Mapping):
        hr_crop = _as_list(crop_contract.get("size_hr_hw"))
    origin = data.get("fixed_crop_origin_hr_xy")
    if isinstance(origin, list) and origin and isinstance(origin[0], list):
        origin = origin[0]
    if origin is None and isinstance(crop_contract, Mapping):
        resolved = crop_contract.get("resolved_origins_hr_xy")
        if isinstance(resolved, list) and resolved:
            origin = resolved[0]
        if origin is None:
            origin = crop_contract.get("requested_origin_hr_xy")
    endpoint = data.get("common_domain_endpoint_count")
    if endpoint is None:
        endpoint = _nested(data, "endpoint_selection", "endpoint_count")
    endpoint = _as_int(endpoint)
    records = _records_evaluated(data)
    common_complete = _first_present(
        _as_bool(data.get("common_domain_complete")),
        _as_bool(_nested(data, "claims", "common_domain_complete")),
    )
    coverage = _first_present(
        _as_bool(data.get("coverage_eligible")),
        _as_bool(_nested(data, "claims", "coverage_eligible")),
        _as_bool(data.get("common_domain_coverage_eligible")),
    )
    final = _first_present(
        _as_bool(data.get("final_acceptance_eligible")),
        _as_bool(_nested(data, "claims", "final_acceptance_eligible")),
    )
    checkpoint = _checkpoint_info(data)
    final_ckpt = _first_present(
        _as_bool(_nested(data, "final_training_checkpoint")),
        _as_bool(_nested(data, "claims", "final_training_checkpoint")),
        _as_bool(_nested(data, "checkpoint_training_completion", "final_training_checkpoint")),
    )
    fixed = crop_mode == "fixed" and hr_crop == EXPECTED_CROP_SIZE
    origin_matches = origin == EXPECTED_CROP_ORIGIN
    common_dir = path.parent.name == "eval_common_fixed384"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "path": str(path.resolve()),
        "status": data.get("status"),
        "crop_mode": crop_mode,
        "hr_crop": hr_crop,
        "crop_origin_hr_xy": origin,
        "fixed384": fixed,
        "origin_matches_protocol": origin_matches,
        "common_domain_complete": common_complete,
        "coverage_eligible": coverage,
        "final_acceptance_eligible": final,
        "final_training_checkpoint": final_ckpt,
        "final_evaluation": bool(
            str(data.get("status") or "").startswith("FINAL")
            or (common_complete is True and final_ckpt is True)
        ),
        "records_evaluated": records,
        "endpoint_count": endpoint,
        "checkpoint_step": checkpoint.get("step"),
        "exact_common_dir": common_dir,
        "mtime": mtime,
    }


def _candidate_score(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    # Fixed-crop contract and common-domain completeness dominate selection;
    # then prefer final/eligible checkpoints and larger evaluated populations.
    return (
        bool(summary.get("fixed384")),
        bool(summary.get("origin_matches_protocol")),
        bool(summary.get("common_domain_complete")),
        bool(summary.get("coverage_eligible")),
        bool(summary.get("final_evaluation")),
        bool(summary.get("final_training_checkpoint")),
        int(summary.get("endpoint_count") or summary.get("records_evaluated") or 0),
        int(summary.get("records_evaluated") or 0),
        int(summary.get("checkpoint_step") or 0),
        bool(summary.get("exact_common_dir")),
        float(summary.get("mtime") or 0.0),
    )


def _select_candidate(arm_dir: Path) -> tuple[Path | None, dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    errors: list[str] = []
    if not arm_dir.is_dir():
        return None, None, [], ["arm directory missing"]
    for path in sorted(arm_dir.rglob("metrics.json")):
        data = _read_json(path)
        if data is None:
            errors.append(f"malformed metrics: {path}")
            continue
        summary = _candidate_summary(path, data)
        candidates.append((path, data, summary))
    if not candidates:
        return None, None, [], errors or ["no metrics.json"]
    fixed = [item for item in candidates if item[2].get("fixed384")]
    pool = fixed or candidates
    selected_path, selected_data, selected_summary = max(pool, key=lambda item: _candidate_score(item[2]))
    all_summaries = [item[2] for item in candidates]
    return selected_path, selected_data, all_summaries, errors


def _config_for_arm(project_root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    path = project_root / "configs" / "spring_v31_ffs" / f"{arm}.yaml"
    return path, _simple_config(path)


def _manifest_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "sha256": _sha256(path), "exists": path.is_file()}


def _find_shared_manifests(run_root: Path, project_root: Path, endpoint_data: Mapping[str, Any] | None) -> dict[str, dict[str, Any] | None]:
    def resolve_candidates(values: Sequence[Any]) -> Path | None:
        for value in values:
            path = _resolve_path(value, base=project_root)
            if path is not None and path.is_file():
                return path
        return None

    validation = resolve_candidates(
        [
            _nested(endpoint_data, "manifest_path") if endpoint_data else None,
            project_root / "runs" / "spring_seed42_primary" / "manifests" / "validation.jsonl",
            run_root.parent / "spring_seed42_primary" / "manifests" / "validation.jsonl",
        ]
    )
    train = resolve_candidates(
        [
            project_root / "runs" / "spring_seed42_primary" / "manifests" / "train.jsonl",
            run_root.parent / "spring_seed42_primary" / "manifests" / "train.jsonl",
        ]
    )
    return {"validation": _manifest_info(validation), "train": _manifest_info(train)}


def _summarize_identity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    keys = (
        "component",
        "checkpoint_sha256",
        "config_sha256",
        "upstream_commit",
        "torch_version",
        "cuda_version",
    )
    result = {key: value.get(key) for key in keys if value.get(key) is not None}
    return result or None


def _summarize_cache(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in (
        "component",
        "derived_contract",
        "derived_cache_root",
        "cache_manifest_path",
        "cache_manifest_sha256",
        "run_receipt_path",
        "run_receipt_sha256",
        "selected_records",
        "manifest_path",
        "manifest_sha256",
        "receipt_path",
        "receipt_sha256",
    ):
        if key in value:
            result[key] = value[key]
    config = value.get("config")
    if isinstance(config, Mapping):
        selected = {
            key: config.get(key)
            for key in (
                "algorithm",
                "schema_version",
                "pose_source",
                "depth_source",
                "temporal_pose_source",
                "current_left_view_index",
                "previous_left_view_index",
            )
            if config.get(key) is not None
        }
        if selected:
            result["config"] = selected
    return result or None


def _extract_lineage(
    arm: str,
    data: Mapping[str, Any] | None,
    config_path: Path,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    data = data or {}
    resolved = data.get("resolved_config")
    if not isinstance(resolved, Mapping):
        resolved = {}
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        checkpoint = {}
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        training_config = {}
    model_cfg = _first_present(resolved.get("model"), training_config.get("model"), {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    data_cfg = _first_present(resolved.get("data"), training_config.get("data"), {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    checkpoint_lineage = data.get("checkpoint_lineage")
    if not isinstance(checkpoint_lineage, Mapping):
        checkpoint_lineage = {}
    cache_ids = data.get("cache_identities")
    if not isinstance(cache_ids, Mapping):
        cache_ids = {}
    cache_lineage = data.get("cache_lineage")
    if not isinstance(cache_lineage, Mapping):
        cache_lineage = {}
    observation_identity = _summarize_identity(
        _first_present(cache_ids.get("observation"), cache_lineage.get("identity"))
    )
    teacher_identity = _summarize_identity(cache_ids.get("teacher"))
    derived = _summarize_cache(
        _first_present(data.get("derived_cache_lineage"), data_cfg.get("derived_cache_lineage"))
    )
    holdout = data.get("holdout_and_raw_lineage")
    if not isinstance(holdout, Mapping):
        holdout = {}
    raw_vggt = _summarize_cache(
        _first_present(holdout.get("evaluation_raw_vggt"), holdout.get("raw_vggt"))
    )
    manifest_path = _resolve_path(
        _first_present(data.get("manifest_path"), _nested(data, "manifest", "path")),
        base=project_root,
    )
    if manifest_path is not None:
        manifest = _manifest_info(manifest_path)
    else:
        manifest = None
    pose_source = _first_present(
        data_cfg.get("temporal_pose_source"),
        checkpoint_lineage.get("temporal_pose_source"),
        holdout.get("pose_source"),
        config.get("temporal_pose_source"),
    )
    if pose_source is None:
        pose_source = {"F0": "none", "F1": "none", "F2": "gt", "F3": "gt", "F4": "gt", "F5": "gt", "F6": "vggt", "F7": "vggt"}.get(arm)
    use_depth = _first_present(model_cfg.get("use_vggt_depth"), config.get("use_vggt_depth"))
    use_pose = _first_present(model_cfg.get("use_vggt_pose"), config.get("use_vggt_pose"))
    if use_depth is None:
        use_depth = arm in {"F5", "F6", "F7"}
    if use_pose is None:
        use_pose = arm in {"F6", "F7"}
    crop = data.get("crop_contract")
    if not isinstance(crop, Mapping):
        crop = {}
    origin = data.get("fixed_crop_origin_hr_xy")
    if isinstance(origin, list) and origin and isinstance(origin[0], list):
        origin = origin[0]
    if origin is None:
        origin = _first_present(crop.get("resolved_origins_hr_xy"), crop.get("requested_origin_hr_xy"))
        if isinstance(origin, list) and origin and isinstance(origin[0], list):
            origin = origin[0]
    consistency: dict[str, Any] = {}
    protocol_manifest = protocol.get("validation_manifest")
    protocol_sha = protocol_manifest.get("sha256") if isinstance(protocol_manifest, Mapping) else None
    arm_sha = manifest.get("sha256") if isinstance(manifest, Mapping) else None
    consistency["manifest_matches_protocol"] = (
        None if protocol_sha is None or arm_sha is None else protocol_sha == arm_sha
    )
    endpoint = data.get("endpoint_selection")
    if not isinstance(endpoint, Mapping):
        # Frozen baseline receipts nest the same endpoint receipt under the
        # generic selection object; preserve it rather than treating F0/F1 as
        # lacking endpoint lineage.
        endpoint = _nested(data, "selection", "endpoint_index_list")
    endpoint_hash = _first_present(
        _nested(endpoint, "endpoint_id_sha256"),
        _nested(endpoint, "evaluated_endpoint_id_sha256"),
        _nested(data, "common_endpoints", "endpoint_id_sha256"),
    )
    expected_hash = protocol.get("endpoint_id_sha256")
    consistency["endpoint_id_matches_protocol"] = (
        None if endpoint_hash is None or expected_hash is None else endpoint_hash == expected_hash
    )
    consistency["crop_size_matches_protocol"] = (
        None
        if not isinstance(data.get("hr_crop"), list)
        else data.get("hr_crop") == EXPECTED_CROP_SIZE
    )
    consistency["crop_origin_matches_protocol"] = None if origin is None else origin == EXPECTED_CROP_ORIGIN
    expected_obs = protocol.get("observation_checkpoint")
    expected_obs_sha = expected_obs.get("checkpoint_sha256") if isinstance(expected_obs, Mapping) else None
    obs_sha = observation_identity.get("checkpoint_sha256") if observation_identity else None
    consistency["observation_checkpoint_matches_protocol"] = (
        None if expected_obs_sha is None or obs_sha is None else expected_obs_sha == obs_sha
    )
    known_checks = [value for value in consistency.values() if isinstance(value, bool)]
    consistency["all_available_checks_pass"] = bool(known_checks) and all(known_checks)
    return {
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
            "declared": dict(config),
        },
        "manifest": manifest,
        "endpoint_selection": dict(endpoint) if isinstance(endpoint, Mapping) else None,
        "observation_cache_identity": observation_identity,
        "teacher_cache_identity": teacher_identity,
        "derived_geometry_cache": derived,
        "raw_vggt_cache": raw_vggt,
        "pose_source": pose_source,
        "use_vggt_depth": bool(use_depth),
        "use_vggt_pose": bool(use_pose),
        "stage": _first_present(resolved.get("train", {}).get("stage") if isinstance(resolved.get("train"), Mapping) else None, data.get("stage"), config.get("stage")),
        "derived_contract": _first_present(data_cfg.get("derived_contract"), derived.get("derived_contract") if isinstance(derived, Mapping) else None),
        "calibration_conditioning": _first_present(resolved.get("calibration_conditioning_v3"), training_config.get("calibration_conditioning_v3")),
        "temporal_history": _first_present(resolved.get("temporal_history_v2"), training_config.get("temporal_history_v2")),
        "temporal_candidate_fusion": _first_present(resolved.get("temporal_candidate_fusion_v3_1"), training_config.get("temporal_candidate_fusion_v3_1")),
        "checkpoint_lineage": {
            key: checkpoint_lineage.get(key)
            for key in (
                "stage",
                "source_sequence_length",
                "temporal_pose_source",
                "stage_a_initialization_path",
                "stage_a_initialization_sha256",
            )
            if checkpoint_lineage.get(key) is not None
        },
        "holdout": {
            key: holdout.get(key)
            for key in (
                "formal_holdout",
                "same_manifest",
                "sequence_overlap",
                "evaluation_sequences",
                "training_sequences",
                "pose_source",
            )
            if holdout.get(key) is not None
        },
        "consistency": consistency,
    }


def _eligibility(data: Mapping[str, Any] | None, summary: Mapping[str, Any], arm: str) -> dict[str, Any]:
    is_missing = data is None
    source = data or {}
    claims = source.get("claims") if isinstance(source.get("claims"), Mapping) else {}
    holdout = source.get("holdout_and_raw_lineage") if isinstance(source.get("holdout_and_raw_lineage"), Mapping) else {}
    completion = source.get("checkpoint_training_completion") if isinstance(source.get("checkpoint_training_completion"), Mapping) else {}
    common = _first_present(_as_bool(source.get("common_domain_complete")), _as_bool(claims.get("common_domain_complete")))
    common_cov = _first_present(_as_bool(source.get("common_domain_coverage_eligible")), _as_bool(claims.get("common_domain_coverage_eligible")))
    coverage = _first_present(_as_bool(source.get("coverage_eligible")), _as_bool(claims.get("coverage_eligible")), common_cov)
    formal = _first_present(_as_bool(claims.get("formal_holdout")), _as_bool(holdout.get("formal_holdout")), _as_bool(source.get("formal_holdout")))
    final_ckpt = _first_present(_as_bool(source.get("final_training_checkpoint")), _as_bool(claims.get("final_training_checkpoint")), _as_bool(completion.get("final_training_checkpoint")))
    final = _first_present(_as_bool(source.get("final_acceptance_eligible")), _as_bool(claims.get("final_acceptance_eligible")))
    records = _as_int(summary.get("records_evaluated"))
    endpoint = _as_int(summary.get("endpoint_count"))
    status = str(source.get("status") or summary.get("status") or "")
    path = str(summary.get("path") or "").lower()
    canary = bool(
        records is not None
        and records < EXPECTED_ENDPOINT_COUNT
    ) or any(token in path for token in ("canary", "smoke", "step2000_64", "_64_fixed384"))
    if is_missing:
        canary = False
    fixed = bool(summary.get("fixed384"))
    common_complete = (
        common is True
        if common is not None
        else fixed
        and endpoint == EXPECTED_ENDPOINT_COUNT
        and records == EXPECTED_ENDPOINT_COUNT
    )
    if common is None and common_complete:
        common = True
    common_eligible = common_cov if common_cov is not None else common_complete
    cross_arm = bool(fixed and common_complete and common_eligible and not canary)
    # F0/F1 are frozen observations and do not carry a trainable final
    # checkpoint; they remain common-domain comparable by design.
    if arm in {"F0", "F1"} and fixed and common_complete:
        cross_arm = True
    return {
        "common_domain_complete": common,
        "common_domain_coverage_eligible": common_cov,
        "coverage_eligible": coverage,
        "formal_holdout": formal,
        "final_training_checkpoint": final_ckpt,
        "final_acceptance_eligible": final,
        "common_domain_endpoint_count": endpoint,
        "records_evaluated": records,
        "is_missing": is_missing,
        "is_canonical_fixed384": fixed,
        "is_canary": canary,
        "is_limited_evaluation": bool(records is not None and records < EXPECTED_ENDPOINT_COUNT),
        "is_optional": arm == "F7",
        "cross_arm_common_domain_eligible": cross_arm,
        "status": status or ("OPTIONAL_NOT_RUN" if arm == "F7" else "MISSING"),
    }


def _extract_arm(
    arm: str,
    run_root: Path,
    project_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    config_path, config = _config_for_arm(project_root, arm)
    path, data, candidates, errors = _select_candidate(run_root / "arms" / arm)
    selected_summary = (
        next(
            (item for item in candidates if item.get("path") == str(path.resolve())),
            None,
        )
        if path
        else None
    )
    if selected_summary is None:
        selected_summary = {
            "path": str(path.resolve()) if path else None,
            "status": None,
            "fixed384": False,
            "origin_matches_protocol": False,
            "records_evaluated": None,
            "endpoint_count": None,
            "checkpoint_step": None,
            "common_domain_complete": None,
        }
    checkpoint = _checkpoint_info(data or {})
    if data is not None and arm in {"F0", "F1"}:
        metric_container: Mapping[str, Any] | None = data.get("metrics") if isinstance(data.get("metrics"), Mapping) else None
        metric_source = "metrics"
        primary_method = None
    elif data is not None:
        native = data.get("spring_native_metrics")
        native_methods = native.get("methods") if isinstance(native, Mapping) and isinstance(native.get("methods"), Mapping) else {}
        methods = data.get("methods") if isinstance(data.get("methods"), Mapping) else {}
        primary_method = None
        metric_container = None
        metric_source = None
        for method in PRIMARY_METHODS.get(arm, ("T3_VGGT_epipolar", "T3_VGGT", "T3")):
            if isinstance(native_methods, Mapping) and isinstance(native_methods.get(method), Mapping):
                metric_container = native_methods[method]
                primary_method = method
                metric_source = f"spring_native_metrics.methods.{method}"
                break
            if isinstance(methods.get(method), Mapping):
                metric_container = methods[method]
                primary_method = method
                metric_source = f"methods.{method}"
                break
    else:
        metric_container = None
        metric_source = None
        primary_method = None
    metric_receipts: dict[str, dict[str, Any]] = {}
    flat_metrics: dict[str, Any] = {}
    metric_sources: dict[str, str | None] = {}
    for target in REQUIRED_METRICS:
        receipt = _extract_metric(metric_container, METRIC_ALIASES[target])
        metric_receipts[target] = receipt
        flat_metrics[target] = receipt["value"]
        flat_metrics[f"{target}_numerator"] = receipt["numerator"]
        flat_metrics[f"{target}_count"] = receipt["count"]
        metric_sources[target] = receipt.get("source_key")
    topk_values: dict[str, Any] = {}
    topk_sources: dict[str, str] = {}
    if data is not None:
        native = data.get("spring_native_metrics")
        native_topk = native.get("topk_diagnostics") if isinstance(native, Mapping) else None
        diagnostics = data.get("diagnostics")
        legacy_topk = diagnostics.get("topk_candidate_complementarity_v3_1") if isinstance(diagnostics, Mapping) else None
        topk_values, topk_sources = _extract_topk(native_topk if isinstance(native_topk, Mapping) else legacy_topk)
    lineage = _extract_lineage(
        arm, data, config_path, config, protocol, project_root=project_root
    )
    eligibility = _eligibility(data, selected_summary, arm)
    status = str((data or {}).get("status") or eligibility["status"])
    declared_config_status = config.get("status")
    if data is None and arm == "F7":
        status = "OPTIONAL_NOT_RUN"
    train_completion = checkpoint.get("training_completion")
    if isinstance(train_completion, Mapping):
        checkpoint["training_completion"] = {
            key: train_completion.get(key)
            for key in (
                "actual_step",
                "configured_steps",
                "declared_schedule_steps",
                "canonical_steps",
                "canonical_schedule",
                "execution_complete",
                "final_training_checkpoint",
                "stage",
            )
            if train_completion.get(key) is not None
        }
    return {
        "arm": arm,
        "purpose": ARM_PURPOSES[arm],
        "status": status,
        "declared_config_status": declared_config_status,
        "primary_method": primary_method,
        "metric_source": metric_source,
        "selected_metrics_path": selected_summary.get("path"),
        "candidate_files": candidates,
        "selection_errors": errors,
        "records_evaluated": selected_summary.get("records_evaluated"),
        "endpoint_count": selected_summary.get("endpoint_count"),
        "crop": {
            "mode": selected_summary.get("crop_mode"),
            "size_hr_hw": selected_summary.get("hr_crop"),
            "origin_hr_xy": selected_summary.get("crop_origin_hr_xy"),
            "fixed384": selected_summary.get("fixed384"),
            "origin_matches_protocol": selected_summary.get("origin_matches_protocol"),
        },
        "checkpoint": checkpoint,
        "eligibility": eligibility,
        "flags": {
            "is_missing": eligibility["is_missing"],
            "is_canonical_fixed384": eligibility["is_canonical_fixed384"],
            "is_final_evaluation": bool(selected_summary.get("final_evaluation")),
            "is_canary": eligibility["is_canary"],
            "is_limited_evaluation": eligibility["is_limited_evaluation"],
            "is_formal_holdout": eligibility["formal_holdout"],
            "is_final_checkpoint": eligibility["final_training_checkpoint"],
            "is_optional": eligibility["is_optional"],
            "cross_arm_common_domain_eligible": eligibility["cross_arm_common_domain_eligible"],
        },
        "lineage": lineage,
        "metrics": flat_metrics,
        "metric_receipts": metric_receipts,
        "metric_sources": metric_sources,
        "topk_diagnostics": topk_values,
        "topk_sources": topk_sources,
        "raw_status": (data or {}).get("status"),
        "selection_class": (
            "missing"
            if eligibility["is_missing"]
            else "canonical_fixed384_canary"
            if eligibility["is_canary"] and eligibility["is_canonical_fixed384"]
            else "canonical_fixed384_full"
            if eligibility["cross_arm_common_domain_eligible"]
            else "fixed384_limited"
            if eligibility["is_canonical_fixed384"]
            else "noncanonical"
        ),
    }


def _read_endpoint_data(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "manifests" / "common_endpoints.json"
    data = _read_json(path)
    if data is None:
        return None
    result = dict(data)
    result["path"] = str(path.resolve())
    result["file_sha256"] = _sha256(path)
    return result


def _protocol(run_root: Path, project_root: Path, endpoint_data: Mapping[str, Any] | None, arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    manifest_info = _find_shared_manifests(run_root, project_root, endpoint_data)
    endpoint_data = endpoint_data or {}
    endpoint_count = _as_int(endpoint_data.get("endpoint_count")) or EXPECTED_ENDPOINT_COUNT
    endpoint_hash = endpoint_data.get("endpoint_id_sha256")
    endpoint_path = endpoint_data.get("path")
    endpoint_file_sha = endpoint_data.get("file_sha256")
    shared_observation: dict[str, Any] | None = None
    shared_commit: str | None = None
    for arm in arms:
        identity = _nested(arm, "lineage", "observation_cache_identity")
        if isinstance(identity, Mapping):
            if shared_observation is None:
                shared_observation = dict(identity)
            if shared_commit is None:
                shared_commit = identity.get("upstream_commit")
    protocol_name = "spring_v31_ffs_common_domain_v1"
    for arm in arms:
        declared = _nested(arm, "lineage", "config", "declared")
        if isinstance(declared, Mapping) and declared.get("protocol"):
            protocol_name = str(declared["protocol"])
            break
    return {
        "name": protocol_name,
        "seed": 42,
        "validation_manifest": manifest_info.get("validation"),
        "train_manifest": manifest_info.get("train"),
        "endpoint_selection": {
            "path": endpoint_path,
            "file_sha256": endpoint_file_sha,
            "endpoint_count": endpoint_count,
            "endpoint_id_sha256": endpoint_hash,
            "kind": endpoint_data.get("kind"),
            "schema_version": endpoint_data.get("schema_version"),
            "endpoint_id_hash_algorithm": endpoint_data.get("endpoint_id_hash_algorithm"),
        },
        "endpoint_count": endpoint_count,
        "endpoint_id_sha256": endpoint_hash,
        "crop": {
            "mode": "fixed",
            "size_hr_hw": list(EXPECTED_CROP_SIZE),
            "origin_hr_xy": list(EXPECTED_CROP_ORIGIN),
            "spatial_scale": 2,
        },
        "observation_checkpoint": shared_observation,
        "observation_upstream_commit": shared_commit,
        "sequence_disjoint_expected": True,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _relative(path: Any, root: Path) -> str:
    if not path:
        return "—"
    try:
        return str(Path(str(path)).resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(path)


def _markdown(payload: Mapping[str, Any], *, json_path: Path, run_root: Path) -> str:
    protocol = payload["protocol"]
    arms = payload["arms"]
    status = payload.get("status", "UNKNOWN")
    lines = [
        "# Spring v3.1 + Fast-FoundationStereo common-domain report",
        "",
        f"**Report status: {status}** — seed 42; this file is generated from read-only arm receipts.",
        "",
        "The canonical comparison contract is fixed HR `384×768`, origin `(x,y)=(576,348)`, over the shared Spring endpoint list. A row marked `canary` or `limited` is not a full common-domain ranking result; its raw numerator/count pairs remain in the JSON.",
        "",
        "## Protocol and shared lineage",
        "",
        f"- protocol: `{protocol.get('name')}`; seed `{protocol.get('seed')}`",
        f"- validation manifest: `{_relative(_nested(protocol, 'validation_manifest', 'path'), run_root.parent)}` (SHA256 `{_nested(protocol, 'validation_manifest', 'sha256') or '—'}`)",
        f"- train manifest: `{_relative(_nested(protocol, 'train_manifest', 'path'), run_root.parent)}` (SHA256 `{_nested(protocol, 'train_manifest', 'sha256') or '—'}`)",
        f"- endpoint file: `{_relative(_nested(protocol, 'endpoint_selection', 'path'), run_root)}` (file SHA256 `{_nested(protocol, 'endpoint_selection', 'file_sha256') or '—'}`)",
        f"- endpoints: `{protocol.get('endpoint_count')}`; endpoint-ID SHA256 `{protocol.get('endpoint_id_sha256') or '—'}`",
        f"- crop: `{_nested(protocol, 'crop', 'mode')}` HR `{_nested(protocol, 'crop', 'size_hr_hw')}` origin `{_nested(protocol, 'crop', 'origin_hr_xy')}`",
        f"- shared FFS observation checkpoint: `{_nested(protocol, 'observation_checkpoint', 'checkpoint_sha256') or '—'}`; upstream commit `{protocol.get('observation_upstream_commit') or '—'}`",
        "",
        "## Arm matrix",
        "",
        "| Arm | Primary row | Selected output | Status | Records | Fixed384 | Canary | Common eligible | Final ckpt | Overall EPE | 1px | HD EPE | LD EPE | Matched | Unmatched @1/@2 | Boundary | Rigid | Non-rigid |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for arm in arms:
        metrics = arm.get("metrics", {})
        flags = arm.get("flags", {})
        lines.append(
            "| {arm} | {method} | `{path}` | {status} | {records} | {fixed} | {canary} | {eligible} | {final} | {epe} | {one} | {hd} | {ld} | {matched} | {u1}/{u2} | {boundary} | {rigid} | {nonrigid} |".format(
                arm=arm.get("arm"),
                method=arm.get("primary_method") or "—",
                path=_relative(arm.get("selected_metrics_path"), run_root),
                status=arm.get("status") or "—",
                records=_fmt(arm.get("records_evaluated")),
                fixed=_fmt(flags.get("is_canonical_fixed384")),
                canary=_fmt(flags.get("is_canary")),
                eligible=_fmt(flags.get("cross_arm_common_domain_eligible")),
                final=_fmt(flags.get("is_final_checkpoint")),
                epe=_fmt(metrics.get("overall_epe")),
                one=_fmt(metrics.get("overall_1px")),
                hd=_fmt(metrics.get("high_detail_epe")),
                ld=_fmt(metrics.get("low_detail_epe")),
                matched=_fmt(metrics.get("matched_epe")),
                u1=_fmt(metrics.get("unmatched_completion_1px")),
                u2=_fmt(metrics.get("unmatched_completion_2px")),
                boundary=_fmt(metrics.get("boundary_epe")),
                rigid=_fmt(metrics.get("rigid_temporal_residual_error")),
                nonrigid=_fmt(metrics.get("non_rigid_temporal_residual_error")),
            )
        )
    lines.extend(
        [
            "",
            "## Output health and FFS trusted measurement",
            "",
            "| Arm | FFS trusted error | Negative rate | Zero rate | Invalid rate | Numerator/count details |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for arm in arms:
        metrics = arm.get("metrics", {})
        lines.append(
            f"| {arm.get('arm')} | {_fmt(metrics.get('ffs_trusted_measurement_error'))} | {_fmt(metrics.get('negative_rate'))} | {_fmt(metrics.get('zero_rate'))} | {_fmt(metrics.get('invalid_rate'))} | `{_relative(json_path, run_root.parent)}` |"
        )
    topk_arms = [arm for arm in arms if arm.get("topk_diagnostics")]
    if topk_arms:
        lines.extend(
            [
                "",
                "## Top-K temporal complementarity diagnostics",
                "",
                "| Arm | Age-2 survival | Unique-age | Phase variance | Depth spread | Attention entropy | Fractional-phase gain | Camera-motion gain |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for arm in topk_arms:
            topk = arm.get("topk_diagnostics", {})
            lines.append(
                f"| {arm.get('arm')} | {_fmt(topk.get('age_2_survival_rate'))} | {_fmt(topk.get('unique_age_fraction'))} | {_fmt(topk.get('phase_variance'))} | {_fmt(topk.get('candidate_depth_spread'))} | {_fmt(topk.get('attention_entropy'))} | {_fmt(topk.get('gain_by_fractional_phase_bucket'))} | {_fmt(topk.get('gain_by_camera_motion_bucket'))} |"
            )
    lines.extend(
        [
            "",
            "## Lineage and eligibility notes",
            "",
            "- F0/F1 use the top-level frozen-observation `metrics` object; F2–F7 use the declared primary method row, preferring the exact `spring_native_metrics` side channel when present.",
            "- GT pose and VGGT pose are reported independently per arm. VGGT depth and pose switches are read from the resolved checkpoint config (or the arm YAML when no checkpoint exists).",
            "- `cross_arm_common_domain_eligible` requires the fixed crop, the complete 1302-endpoint domain, and coverage eligibility. Canary rows stay visible but are not silently promoted to full validation.",
            "- F7 is optional; a missing F7 receipt is represented as `OPTIONAL_NOT_RUN` with null metrics rather than inferred from F6.",
            "- Metric numerator/count pairs, candidate paths, checkpoint hashes, cache identities, and consistency checks are in the JSON report.",
            "",
            f"Machine-readable report: `{json_path.resolve()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(*, run_root: Path = DEFAULT_RUN_ROOT, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    endpoint_data = _read_endpoint_data(run_root)
    # Build arm rows once, then derive shared protocol facts from their cache
    # identities.  This does not open checkpoints or instantiate CUDA.
    provisional_protocol: dict[str, Any] = {
        "validation_manifest": None,
        "endpoint_id_sha256": (endpoint_data or {}).get("endpoint_id_sha256"),
        "observation_checkpoint": None,
    }
    arms = [_extract_arm(arm, run_root, project_root, provisional_protocol) for arm in ARMS]
    protocol = _protocol(run_root, project_root, endpoint_data, arms)
    # Recompute per-arm consistency against the now-complete protocol.  This is
    # still read-only and avoids embedding a stale provisional manifest hash.
    for arm in arms:
        config_path = _resolve_path(_nested(arm, "lineage", "config", "path"))
        config = _nested(arm, "lineage", "config", "declared")
        data_path = arm.get("selected_metrics_path")
        data = _read_json(Path(data_path)) if data_path else None
        if config_path is not None and isinstance(config, Mapping):
            arm["lineage"] = _extract_lineage(
                arm["arm"], data, config_path, config, protocol, project_root=project_root
            )
    non_optional = [arm for arm in arms if arm.get("arm") != "F7"]
    complete = all(
        bool(_nested(arm, "flags", "cross_arm_common_domain_eligible"))
        for arm in non_optional
    )
    # F7 is explicitly optional in the protocol. Keep it visible as
    # OPTIONAL_NOT_RUN, but do not downgrade an otherwise complete F0--F6
    # matrix merely because the optional Stage-C adapter is absent.
    missing = [
        arm["arm"]
        for arm in non_optional
        if _nested(arm, "flags", "is_missing")
    ]
    optional_missing = [
        arm["arm"]
        for arm in arms
        if arm.get("arm") == "F7" and _nested(arm, "flags", "is_missing")
    ]
    canaries = [arm["arm"] for arm in arms if _nested(arm, "flags", "is_canary")]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "report": "spring_v31_ffs_common_domain",
        "status": "COMPLETE" if complete and not missing else "PARTIAL",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(run_root),
        "protocol": protocol,
        "required_metrics": list(REQUIRED_METRICS),
        "topk_diagnostics": list(TOPK_METRICS),
        "selection_policy": {
            "canonical_fixed384": "prefer fixed crop 384x768; rank complete common-domain/final candidates ahead of limited canaries",
            "frozen_baselines": "F0/F1 read top-level metrics",
            "trainable_arms": "F2-F7 read primary methods, preferring spring_native_metrics.methods when available",
            "missing_policy": "retain arm with null metrics and explicit missing/optional status",
        },
        "summary": {
            "arm_count": len(arms),
            "complete_common_domain_arm_count": sum(bool(_nested(arm, "flags", "cross_arm_common_domain_eligible")) for arm in arms),
            "canary_arm_count": len(canaries),
            "missing_arm_count": len(missing),
            "missing_arms": missing,
            "optional_arms_not_run": optional_missing,
            "canary_arms": canaries,
            "formal_arms": [
                arm["arm"]
                for arm in arms
                if _nested(arm, "flags", "is_formal_holdout") is True
            ],
            "full_domain_arms": [
                arm["arm"]
                for arm in arms
                if _nested(arm, "flags", "cross_arm_common_domain_eligible") is True
            ],
            "all_non_optional_arms_present": not any(_nested(arm, "flags", "is_missing") for arm in arms if arm.get("arm") != "F7"),
        },
        "arms": arms,
        "notes": [
            "This report is read-only with respect to runs/spring_v31_ffs; training and evaluation processes are not started or stopped.",
            "Canary and limited-subset metric values are included for evidence continuity but are not full-domain ranking claims.",
            "Null means unavailable or not evaluated; no metric is imputed across crop, pose, or GT lineage boundaries.",
        ],
    }
    return payload


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    payload = build_report(run_root=run_root, project_root=project_root)
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    _atomic_write(output_json, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(output_md, _markdown(payload, json_path=output_json, run_root=run_root))
    print(json.dumps({
        "status": payload["status"],
        "json": str(output_json),
        "markdown": str(output_md),
        "arms": {arm["arm"]: arm["status"] for arm in payload["arms"]},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
