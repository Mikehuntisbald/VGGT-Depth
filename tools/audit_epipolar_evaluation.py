#!/usr/bin/env python3
"""Read-only final-gate audit for a Stage-C ``metrics.json`` evaluation.

This tool never runs a model.  It cross-checks the evaluator report against the
Stage-C training audit and completion summary plus the canonical Stage-B final
report.  Only the raw ``T3_VGGT_epipolar`` output owns engineering gates;
``clamp0`` remains a declared diagnostic and the same-family FFS pseudo-target
is never promoted to paper ground truth.

Exit codes are 0 for a valid final engineering-gate pass, 1 for a structurally
valid gate failure/ineligible intermediate result, and 2 for an invalid or
tampered artifact set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA_VERSION = 1
AUDIT_COMPONENT = "ffs-omega-tsr-epipolar-evaluation-audit"
METRICS_STAGE = "STAGE_C_EPIPOLAR_HELD_OUT"
TRAINING_AUDIT_COMPONENT = "ffs-omega-tsr-epipolar-training-audit"
TRAINING_SUMMARY_COMPONENT = "ffs-omega-tsr-epipolar-training-run"
STAGE_B_REPORT_COMPONENT = "stage-b-final-holdout-evaluation-with-sign-health"
PSEUDO_GT_TARGET = "trusted_hr_ffs_teacher_pseudo_gt"
POINT_TO_PLANE_NOT_AVAILABLE = {
    "status": "NOT_AVAILABLE",
    "reason": "target point normals and explicit correspondences are unavailable",
}
CLAMP0_OPERATION = (
    "torch.where(isfinite(disparity_hr_px) & "
    "(disparity_hr_px < 0), 0, disparity_hr_px)"
)
FORMAL_STAGE_C_STEPS = 5_000
FORMAL_STAGE_B_STEPS = 15_000
EXPECTED_REFINER_PARAMETERS = 69_905
FORMAL_MANIFEST_RECORDS = 244
FORMAL_DERIVED_RECORDS = 240
FORMAL_EVALUABLE_WINDOWS = 238
FORMAL_TRAIN_DERIVED_RECORDS = 2_779
FORMAL_TRAIN_WINDOWS = 2_775
FORMAL_VALIDATION_MANIFEST_SHA256 = (
    "014bd75de8ffbf74530c64eac76394a30bfc62d65b2da02397de2fb5c984760c"
)
FORMAL_TRAIN_MANIFEST_SHA256 = (
    "596702933688f695d9aac480d9f01e5764f40a4d7b28d72d73c550eb209b301c"
)
FORMAL_STAGE_B_REPORT_SHA256 = (
    "942c231787654602a66394fd8ab124839ffc0f12d09953885ab4d6f08b567220"
)
FORMAL_STAGE_B_CHECKPOINT_SHA256 = (
    "1b9a35ebc77784c7ef62c6d03af4fc956b2f28406dc94789056d14fa35ae8637"
)
FORMAL_STAGE_B_METRICS_SHA256 = (
    "2ba036ad502cc26dab531a40ea6d4cbe361f179d8b035374c43a6aecda922fd8"
)
FORMAL_STAGE_B_TRAINING_AUDIT_SHA256 = (
    "c3d6ddc1514b08dab6076cd4773e73020ad83f3fd57acf12f692975f80aa3e13"
)
FORMAL_STAGE_C_TRAINING_GIT_HASH = (
    "4e6b7eb488201227e46b30e2ac90d34991466f2c"
)
FORMAL_STAGE_C_CONFIG_SHA256 = (
    "b02473a3d1df02ef527769d34d62729389ff205cdcdfc82feb8b5787415364f7"
)
FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256 = (
    "6a2cb37fb56dd79661c75d16c2128bc5355b857ea371d2b2c5de488e63acc4e5"
)
FORMAL_STAGE_C_EVALUATOR_SHA256 = (
    "d88f69bc49ff8410c3628f5d6db3c9595a4ac791e07d0e72870ee3914468204e"
)
FORMAL_STAGE_C_TRAINING_AUDITOR_SHA256 = (
    "2cfec90af44a77bdf6e4876d42306956ba9fcc253209b3733f82f2a0f73df7ad"
)
FORMAL_RECTIFICATION_AUDIT_SHA256 = (
    "3eb3e8853e4723b9e0703aeaffd36b9ef482b311ff5b9cff5a79e28e60e84429"
)
FORMAL_VALIDATION_DERIVED_MANIFEST_SHA256 = (
    "f9e34fa730e43013aa7b1e19b57b132bc929f9b6ffe72c4d7796cd67fa1cd594"
)
FORMAL_VALIDATION_DERIVED_RECEIPT_SHA256 = (
    "1e05c4e081620b9f634e9b3020779c6e4fdb7e21bf1df55365c3d4cde3d50620"
)
FORMAL_VALIDATION_RAW_VGGT_MANIFEST_SHA256 = (
    "12e6d39c765d377699595a560c3f85f1d29e04cb53243198e350539900a3a0d3"
)
FORMAL_TRAIN_RIGHT_SOURCE_DIGEST_SHA256 = (
    "ae283814edd4e51f3c05187dd2694a4f6cf7bf7e7df09db3fe4b7f1055fd4384"
)
FORMAL_VALIDATION_RIGHT_SOURCE_DIGEST_SHA256 = (
    "363daad4a7e96208de9fe25be4611ecc2cbfe18b29c9cc45087525e9207b4d1f"
)
FORMAL_VALIDATION_RAW_PAYLOAD_DIGEST_SHA256 = (
    "6ad38fdf067b4b7b0d7442bce7bb339d9aa959c0c8808ffd6923fc986c108bc5"
)
FORMAL_TRAIN_RAW_VGGT_RECEIPT_SHA256 = (
    "37ea38ef0a35ec816e180091d323d95cbe8b895b90a440c2e8aaf3ec1c64590d"
)
FORMAL_TRAIN_DERIVED_RECEIPT_SHA256 = (
    "3cd6edf37636690c613d268d6c5bcbfaa004c1a330e1970ad6cd8e34fe989db7"
)
FORMAL_TRAIN_DERIVED_MANIFEST_SHA256 = (
    "5dba94d14a2fe27bfd92e524751ee849cbac8f0ed48be0c4a994ba99c373af64"
)
FORMAL_VALIDATION_RAW_VGGT_RECEIPT_SHA256 = (
    "6309d056760c1f36c32fb0695c88664f3d057b2d155e90f5c9960965edc67bd2"
)
RAW_HEALTH_THRESHOLD = 0.005
TRUSTED_DEGRADATION_LIMIT_PERCENT = 2.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RAW_BASE = "T3_VGGT_base"
RAW_REFINED = "T3_VGGT_epipolar"
CLAMP_BASE = "T3_VGGT_base_clamp0"
CLAMP_REFINED = "T3_VGGT_epipolar_clamp0"
METHOD_NAMES = {RAW_BASE, RAW_REFINED, CLAMP_BASE, CLAMP_REFINED}
EXPECTED_METRICS_TOP_LEVEL = {
    "schema_version",
    "stage",
    "status",
    "claims",
    "target",
    "postprocess_contract",
    "methods",
    "comparisons",
    "refinement_statistics",
    "runtime_geometry_statistics",
    "windows_evaluated",
    "full_evaluable_windows",
    "formal_coverage",
    "canonical_coverage",
    "fixed_hr_crop",
    "crop_contract",
    "visualizations_written",
    "elapsed_seconds",
    "device",
    "checkpoint_completion",
    "execution_contract",
    "stage_c_checkpoint",
    "stage_b_base_checkpoint",
    "lineage",
    "source_hashes",
    "resolved_config",
}
ACCURACY_METRICS = (
    "boundary_epe_px",
    "bad_1",
    "epe_px",
    "low_confidence_epe_px",
)
PRIMARY_ACCURACY_GATES = ("boundary_epe_px", "bad_1")
HEALTH_METRICS = (
    "output_invalid_rate",
    "output_negative_rate",
    "output_nan_rate",
)
REQUIRED_METRICS = (
    *ACCURACY_METRICS,
    "bad_2",
    "trusted_region_epe_px",
    "invalid_region_completeness",
    *HEALTH_METRICS,
    "output_infinite_rate",
    "output_zero_rate",
)
PAIRED_DIAGNOSTIC_METRICS = (
    "paired_epe_improvement_hr_px",
    "paired_refined_better_rate",
    "paired_refined_worse_rate",
    "paired_unchanged_rate",
    "paired_finite_coverage_rate",
    "paired_nonfinite_rate",
)


class EpipolarEvaluationAuditError(RuntimeError):
    """Raised when an evaluation artifact violates the audit contract."""


@dataclass(frozen=True, slots=True)
class JSONArtifact:
    path: Path
    sha256: str
    byte_size: int
    value: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpipolarEvaluationAuditError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str) -> int:
    _require(_is_int(value) and int(value) > 0, f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    _require(
        _is_int(value) and int(value) >= 0,
        f"{name} must be a non-negative integer",
    )
    return int(value)


def _finite_float(value: object, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _require_sha256(value: object, name: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return str(value)


def _reject_json_constant(value: str) -> None:
    raise EpipolarEvaluationAuditError(
        f"strict JSON contains non-finite constant {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, name: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except EpipolarEvaluationAuditError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EpipolarEvaluationAuditError(
            f"cannot parse strict JSON {name}: {exc}"
        ) from exc


def _finite_json_tree(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_json_tree(child, f"{name}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json_tree(child, f"{name}[{index}]")
        return
    raise EpipolarEvaluationAuditError(
        f"{name} contains unsupported JSON value {type(value).__name__}"
    )


def _read_regular_file(path: Path, label: str) -> bytes:
    _require(path.exists(), f"{label} is missing: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EpipolarEvaluationAuditError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size)
            == (after.st_dev, after.st_ino, after.st_size),
            f"{label} changed while it was read",
        )
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"{label} was read incompletely")
        return payload
    finally:
        os.close(descriptor)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_recorded_path(value: object, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} path is malformed")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path: str | Path, label: str) -> JSONArtifact:
    resolved = Path(path).expanduser().resolve()
    payload = _read_regular_file(resolved, label)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpipolarEvaluationAuditError(f"{label} is not UTF-8") from exc
    value = _strict_json_loads(text, label)
    _require(isinstance(value, Mapping), f"{label} is not a JSON object")
    _finite_json_tree(value, label)
    return JSONArtifact(resolved, _sha256_bytes(payload), len(payload), value)


def _verify_file_identity(
    identity: object,
    *,
    expected_path: Path | None,
    name: str,
) -> dict[str, Any]:
    _require(isinstance(identity, Mapping), f"{name} identity is malformed")
    path = _resolve_recorded_path(identity.get("path"), name)
    if expected_path is not None:
        _require(path == expected_path.resolve(), f"{name} path differs")
    expected_sha = _require_sha256(identity.get("sha256"), f"{name} SHA-256")
    payload = _read_regular_file(path, name)
    actual_sha = _sha256_bytes(payload)
    _require(actual_sha == expected_sha, f"{name} SHA-256 mismatch")
    return {"path": str(path), "sha256": actual_sha, "byte_size": len(payload)}


def _canonical_config_sha256(config: object) -> str:
    _require(isinstance(config, Mapping), "Stage-C checkpoint config is malformed")
    try:
        encoded = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EpipolarEvaluationAuditError(
            f"Stage-C checkpoint config is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_training_summary(
    summary: JSONArtifact,
    *,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    value = summary.value
    _require(
        value.get("schema_version") == 1
        and value.get("component") == TRAINING_SUMMARY_COMPONENT
        and value.get("status") == "TRAINING_COMPLETE"
        and value.get("stage") == "epipolar",
        "Stage-C training summary component/schema/status differs",
    )
    _require(
        value.get("steps") == FORMAL_STAGE_C_STEPS
        and value.get("configured_steps") == FORMAL_STAGE_C_STEPS
        and value.get("formal_training_complete") is True,
        "Stage-C training summary is not formal step 5000 completion",
    )
    _require(
        value.get("git_hash") == FORMAL_STAGE_C_TRAINING_GIT_HASH
        and value.get("config_sha256") == FORMAL_STAGE_C_CONFIG_SHA256
        and value.get("runtime_source_bundle_sha256")
        == FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256,
        "Stage-C training summary Git/config/runtime identity is not canonical",
    )
    files = audit.get("files")
    _require(isinstance(files, Mapping), "Stage-C training audit files are missing")
    audit_summary = files.get("run_summary")
    _require(isinstance(audit_summary, Mapping), "training audit run-summary identity is missing")
    _require(
        _resolve_recorded_path(audit_summary.get("path"), "training audit summary")
        == summary.path
        and audit_summary.get("sha256") == summary.sha256,
        "Stage-C training audit does not bind the supplied run summary",
    )
    identities: dict[str, dict[str, Any]] = {}
    for summary_name, audit_name, canonical_name in (
        ("final_checkpoint", "final_checkpoint", "Stage-C final checkpoint"),
        ("latest_checkpoint", "latest_checkpoint", "Stage-C latest checkpoint"),
        ("training_log", "training_log", "Stage-C training log"),
    ):
        summary_identity = value.get(summary_name)
        audit_identity = files.get(audit_name)
        _require(
            isinstance(summary_identity, Mapping)
            and isinstance(audit_identity, Mapping)
            and summary_identity.get("sha256") == audit_identity.get("sha256")
            and _resolve_recorded_path(summary_identity.get("path"), canonical_name)
            == _resolve_recorded_path(audit_identity.get("path"), canonical_name),
            f"Stage-C summary/audit {summary_name} identity differs",
        )
        identities[summary_name] = _verify_file_identity(
            summary_identity,
            expected_path=None,
            name=canonical_name,
        )
    final_identity = files["final_checkpoint"]
    _require(
        final_identity.get("step") == FORMAL_STAGE_C_STEPS
        and final_identity.get("parameter_count") == EXPECTED_REFINER_PARAMETERS,
        "Stage-C training audit final checkpoint identity is malformed",
    )
    checkpoint_validation = audit.get("checkpoint_validation")
    audited_base = (
        checkpoint_validation.get("base_checkpoint")
        if isinstance(checkpoint_validation, Mapping)
        else None
    )
    _require(
        isinstance(audited_base, Mapping)
        and value.get("base_checkpoint")
        == {
            "path": audited_base.get("path"),
            "sha256": audited_base.get("sha256"),
            "step": audited_base.get("step"),
        },
        "Stage-C training summary base identity differs from training audit",
    )
    return {
        "path": str(summary.path),
        "sha256": summary.sha256,
        "git_hash": value.get("git_hash"),
        "config_sha256": value.get("config_sha256"),
        "base_checkpoint": value.get("base_checkpoint"),
        "runtime_source_bundle_sha256": value.get(
            "runtime_source_bundle_sha256"
        ),
        "artifacts": identities,
        "formal_training_complete": True,
    }


def _validate_training_audit(
    artifact: JSONArtifact,
    *,
    summary: JSONArtifact | None,
) -> dict[str, Any]:
    value = artifact.value
    training_auditor_path = PROJECT_ROOT / "tools" / "audit_epipolar_training_run.py"
    training_auditor_payload = _read_regular_file(
        training_auditor_path, "Stage-C training auditor source"
    )
    _require(
        _sha256_bytes(training_auditor_payload)
        == FORMAL_STAGE_C_TRAINING_AUDITOR_SHA256,
        "Stage-C training auditor source SHA-256 is not canonical",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("component") == TRAINING_AUDIT_COMPONENT
        and value.get("read_only") is True,
        "Stage-C training audit component/schema/read-only contract differs",
    )
    safe_load = value.get("safe_load")
    _require(
        isinstance(safe_load, Mapping)
        and safe_load.get("torch_weights_only") is True
        and safe_load.get("arbitrary_pickle_globals_enabled") is False
        and safe_load.get("symlink_artifacts_allowed") is False,
        "Stage-C training audit did not use the strict safe-load contract",
    )
    output_dir_value = value.get("output_dir")
    _require(
        isinstance(output_dir_value, str) and bool(output_dir_value),
        "Stage-C training audit output_dir is missing",
    )
    training_output_dir = Path(output_dir_value).expanduser().resolve()
    completion = value.get("completion")
    files = value.get("files")
    checkpoint_validation = value.get("checkpoint_validation")
    _require(isinstance(completion, Mapping), "Stage-C training audit completion is missing")
    _require(isinstance(files, Mapping), "Stage-C training audit files are missing")
    _require(
        isinstance(checkpoint_validation, Mapping),
        "Stage-C training checkpoint validation is missing",
    )
    checkpoint_completion = checkpoint_validation.get("completion")
    _require(
        isinstance(checkpoint_completion, Mapping),
        "Stage-C checkpoint completion receipt is missing",
    )
    formally_complete = bool(
        value.get("status") == "PASS"
        and value.get("training_status") == "TRAINING_COMPLETE"
        and completion.get("receipt_present") is True
        and completion.get("receipt_valid") is True
        and completion.get("formal_training_complete") is True
        and isinstance(completion.get("summary"), Mapping)
        and completion["summary"].get("valid") is True
        and checkpoint_completion.get("formal_training_complete") is True
        and checkpoint_completion.get("actual_step") == FORMAL_STAGE_C_STEPS
    )
    declared_in_progress = bool(
        value.get("status") == "IN_PROGRESS"
        and value.get("training_status") == "IN_PROGRESS"
        and completion.get("receipt_present") is False
        and completion.get("receipt_valid") is False
        and completion.get("formal_training_complete") is False
        and checkpoint_completion.get("formal_training_complete") is False
    )
    _require(
        formally_complete or declared_in_progress,
        "Stage-C training audit is neither valid completion nor valid in-progress",
    )
    if formally_complete:
        _require(summary is not None, "formal Stage-C audit requires its run summary")
        selected = files.get("final_checkpoint")
        summary_receipt = _validate_training_summary(summary, audit=value)
        latest_identity = files.get("latest_checkpoint")
        log_validation = value.get("log_validation")
        _require(
            isinstance(latest_identity, Mapping)
            and latest_identity.get("step") == FORMAL_STAGE_C_STEPS
            and isinstance(log_validation, Mapping)
            and log_validation.get("records") == FORMAL_STAGE_C_STEPS
            and log_validation.get("last_step") == FORMAL_STAGE_C_STEPS
            and log_validation.get("latest_checkpoint_lag_steps") == 0
            and log_validation.get("steps_continuous") is True
            and log_validation.get("learning_rate_schedule_exact") is True
            and log_validation.get("finite") is True,
            "formal Stage-C training audit log/latest validation is incomplete",
        )
    else:
        _require(summary is None, "in-progress Stage-C audit must not supply a final summary")
        _require(
            files.get("final_checkpoint") is None
            and files.get("run_summary") is None,
            "in-progress Stage-C audit unexpectedly contains final artifacts",
        )
        selected = files.get("latest_checkpoint")
        summary_receipt = None
    _require(isinstance(selected, Mapping), "Stage-C selected checkpoint identity is missing")
    selected_path = _resolve_recorded_path(selected.get("path"), "Stage-C selected checkpoint")
    selected_sha = _require_sha256(
        selected.get("sha256"), "Stage-C selected checkpoint SHA-256"
    )
    selected_payload = _read_regular_file(selected_path, "Stage-C selected checkpoint")
    _require(
        _sha256_bytes(selected_payload) == selected_sha,
        "Stage-C selected checkpoint SHA-256 mismatch",
    )
    selected_step = _positive_int(selected.get("step"), "Stage-C selected checkpoint step")
    _require(
        selected.get("parameter_count") == EXPECTED_REFINER_PARAMETERS,
        "Stage-C parameter count differs",
    )
    _require(
        selected.get("git_hash") == FORMAL_STAGE_C_TRAINING_GIT_HASH
        and selected.get("config_sha256") == FORMAL_STAGE_C_CONFIG_SHA256,
        "Stage-C selected checkpoint Git/config identity is not canonical",
    )
    if formally_complete:
        _require(selected_step == FORMAL_STAGE_C_STEPS, "formal Stage-C audit is not step 5000")
    else:
        _require(
            selected_step < FORMAL_STAGE_C_STEPS,
            "in-progress Stage-C checkpoint reached final step",
        )
    base = checkpoint_validation.get("base_checkpoint")
    runtime = checkpoint_validation.get("runtime_source_bundle")
    rectification = checkpoint_validation.get("rectification_audit")
    training_runtime = checkpoint_validation.get("training_runtime")
    _require(
        isinstance(base, Mapping)
        and base.get("step") == FORMAL_STAGE_B_STEPS
        and isinstance(base.get("completion"), Mapping)
        and base["completion"].get("complete") is True,
        "Stage-C training audit canonical Stage-B base receipt differs",
    )
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("file_count") == 52
        and runtime.get("git_hash") == FORMAL_STAGE_C_TRAINING_GIT_HASH
        and runtime.get("bundle_sha256")
        == FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256
        and runtime.get("all_files_match_checkpoint_git_tree") is True,
        "Stage-C training audit 52-file runtime-source receipt differs",
    )
    _require(
        isinstance(rectification, Mapping)
        and rectification.get("status") == "PASS"
        and rectification.get("sha256") == FORMAL_RECTIFICATION_AUDIT_SHA256,
        "Stage-C training audit rectification receipt differs",
    )
    _require(
        isinstance(training_runtime, Mapping)
        and training_runtime.get("formal_cuda_bf16_eligible") is True
        and training_runtime.get("native_cuda_bf16") is True
        and training_runtime.get("strict_determinism") is True
        and isinstance(training_runtime.get("device_name"), str)
        and "5090" in training_runtime["device_name"]
        and training_runtime.get("device_capability") == [12, 0]
        and training_runtime.get("cuda_version") == "12.8",
        "Stage-C training audit CUDA/BF16/determinism receipt differs",
    )
    return {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "output_dir": str(training_output_dir),
        "formally_complete": formally_complete,
        "in_progress": declared_in_progress,
        "checkpoint": {
            "path": str(selected_path),
            "sha256": selected_sha,
            "step": selected_step,
            "git_hash": selected.get("git_hash"),
            "config_sha256": selected.get("config_sha256"),
            "parameter_count": selected.get("parameter_count"),
        },
        "base_checkpoint": {
            "path": base.get("path"),
            "sha256": base.get("sha256"),
            "step": base.get("step"),
        },
        "runtime_source_bundle": dict(runtime),
        "rectification_audit": dict(rectification),
        "training_runtime": dict(training_runtime),
        "summary": summary_receipt,
    }


def _validate_stage_b_report(artifact: JSONArtifact) -> dict[str, Any]:
    value = artifact.value
    _require(
        artifact.sha256 == FORMAL_STAGE_B_REPORT_SHA256,
        "Stage-B final report SHA-256 is not the canonical receipt",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("component") == STAGE_B_REPORT_COMPONENT,
        "Stage-B final report component/schema differs",
    )
    claim = value.get("claim_boundary")
    checkpoint = value.get("checkpoint")
    _require(isinstance(claim, Mapping), "Stage-B claim boundary is missing")
    _require(isinstance(checkpoint, Mapping), "Stage-B checkpoint receipt is missing")
    _require(
        claim.get("final_acceptance_eligible") is True
        and claim.get("final_training_checkpoint") is True
        and claim.get("formal_holdout") is True
        and claim.get("coverage_eligible") is True
        and claim.get("paper_accuracy") is False
        and claim.get("paper_ground_truth") is False,
        "Stage-B final claim boundary is not canonical engineering-only evidence",
    )
    _require(
        checkpoint.get("step") == FORMAL_STAGE_B_STEPS
        and checkpoint.get("configured_steps") == FORMAL_STAGE_B_STEPS,
        "Stage-B final report is not canonical step 15000",
    )
    checkpoint_identity = _verify_file_identity(
        checkpoint,
        expected_path=None,
        name="Stage-B final checkpoint",
    )
    _require(
        checkpoint_identity["sha256"] == FORMAL_STAGE_B_CHECKPOINT_SHA256,
        "Stage-B final checkpoint SHA-256 is not canonical",
    )
    training_audit = value.get("training_audit")
    _require(
        isinstance(training_audit, Mapping)
        and training_audit.get("status") == "PASS"
        and training_audit.get("training_status") == "TRAINING_COMPLETE"
        and training_audit.get("completion_receipt_valid") is True
        and training_audit.get("latest_checkpoint_step") == FORMAL_STAGE_B_STEPS,
        "Stage-B embedded final training audit is invalid",
    )
    coverage = value.get("coverage")
    _require(
        isinstance(coverage, Mapping)
        and coverage.get("manifest_records") == FORMAL_MANIFEST_RECORDS
        and coverage.get("derived_endpoint_records") == FORMAL_DERIVED_RECORDS
        and coverage.get("evaluable_t3_windows") == FORMAL_EVALUABLE_WINDOWS
        and coverage.get("evaluated_t3_windows") == FORMAL_EVALUABLE_WINDOWS
        and coverage.get("future_frames") is False,
        "Stage-B final report coverage differs from 244/240/238 causal holdout",
    )
    artifacts = value.get("artifacts")
    verified_artifacts: dict[str, Any] = {}
    _require(isinstance(artifacts, Mapping), "Stage-B final artifacts are missing")
    for path_key, sha_key, label in (
        ("metrics_json", "metrics_json_sha256", "Stage-B final metrics.json"),
        ("training_audit", "training_audit_sha256", "Stage-B final training audit"),
    ):
        identity = {"path": artifacts.get(path_key), "sha256": artifacts.get(sha_key)}
        verified_artifacts[path_key] = _verify_file_identity(
            identity, expected_path=None, name=label
        )
    _require(
        verified_artifacts["metrics_json"]["sha256"]
        == FORMAL_STAGE_B_METRICS_SHA256
        and verified_artifacts["training_audit"]["sha256"]
        == FORMAL_STAGE_B_TRAINING_AUDIT_SHA256,
        "Stage-B final metrics/training-audit SHA is not canonical",
    )
    stage_b_metrics = _load_json(
        verified_artifacts["metrics_json"]["path"], "Stage-B final metrics.json"
    )
    stage_b_methods = stage_b_metrics.value.get("methods")
    _require(isinstance(stage_b_methods, Mapping), "Stage-B final methods are missing")
    base_methods: dict[str, dict[str, Any]] = {}
    for source_name, output_name in (
        ("T3_VGGT", RAW_BASE),
        ("T3_VGGT_clamp0", CLAMP_BASE),
    ):
        method = stage_b_methods.get(source_name)
        _require(isinstance(method, Mapping), f"Stage-B method {source_name} is missing")
        base_methods[output_name] = {
            metric_name: _metric(
                method.get(metric_name),
                name=f"Stage-B {source_name}.{metric_name}",
            )
            for metric_name in REQUIRED_METRICS
        }
    return {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "checkpoint": {
            **checkpoint_identity,
            "step": FORMAL_STAGE_B_STEPS,
            "git_hash": checkpoint.get("training_git_hash"),
        },
        "coverage": dict(coverage),
        "artifacts": verified_artifacts,
        "base_methods": base_methods,
        "paper_ground_truth": False,
        "final_acceptance_eligible": True,
    }


def _metric(value: object, *, name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"metric {name} is missing")
    _require(
        set(value) == {"value", "numerator", "count", "valid"},
        f"metric {name} schema differs",
    )
    _require(type(value.get("valid")) is bool, f"metric {name} valid flag is malformed")
    valid = bool(value["valid"])
    count = _nonnegative_int(value.get("count"), f"metric {name} count")
    if not valid:
        _require(
            value.get("value") is None and value.get("numerator") is None,
            f"invalid metric {name} must have null value/numerator",
        )
        return {"value": None, "numerator": None, "count": count, "valid": False}
    _require(count > 0, f"valid metric {name} must have a positive count")
    metric_value = _finite_float(value.get("value"), f"metric {name} value")
    numerator = _finite_float(value.get("numerator"), f"metric {name} numerator")
    _require(
        math.isclose(metric_value, numerator / count, rel_tol=1e-10, abs_tol=1e-12),
        f"metric {name} value differs from numerator/count",
    )
    return {
        "value": metric_value,
        "numerator": numerator,
        "count": count,
        "valid": True,
    }


def _metric_change(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    valid = bool(baseline["valid"] and candidate["valid"])
    if not valid:
        return {
            "baseline": dict(baseline),
            "candidate": dict(candidate),
            "absolute_change": None,
            "relative_change_percent": None,
            "valid": False,
            "relative_valid": False,
        }
    absolute = float(candidate["value"]) - float(baseline["value"])
    relative_valid = float(baseline["value"]) != 0.0
    relative = (
        100.0 * absolute / float(baseline["value"])
        if relative_valid
        else None
    )
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "absolute_change": absolute,
        "relative_change_percent": relative,
        "valid": valid,
        "relative_valid": relative_valid,
    }


def _same_number(left: object, right: float, name: str) -> None:
    value = _finite_float(left, name)
    _require(
        math.isclose(value, right, rel_tol=1e-10, abs_tol=1e-12),
        f"{name} differs from recomputed value",
    )


def _validate_recorded_change(
    value: object,
    *,
    metric_name: str,
    recomputed: Mapping[str, Any],
) -> None:
    _require(isinstance(value, Mapping), f"recorded change {metric_name} is missing")
    _require(
        value.get("metric") == metric_name
        and value.get("valid") is recomputed["valid"]
        and value.get("relative_valid") is recomputed["relative_valid"],
        f"recorded change {metric_name} validity/schema differs",
    )
    _require(
        value.get("baseline") == recomputed["baseline"]
        and value.get("candidate") == recomputed["candidate"],
        f"recorded change {metric_name} baseline/candidate differs",
    )
    if recomputed["valid"]:
        _same_number(
            value.get("absolute_change"),
            float(recomputed["absolute_change"]),
            f"recorded change {metric_name} absolute",
        )
    else:
        _require(
            value.get("absolute_change") is None,
            f"recorded change {metric_name} absolute must be null",
        )
    if recomputed["relative_valid"]:
        _same_number(
            value.get("relative_change_percent"),
            float(recomputed["relative_change_percent"]),
            f"recorded change {metric_name} relative",
        )
    else:
        _require(
            value.get("relative_change_percent") is None,
            f"recorded change {metric_name} relative must be null",
        )


def _validate_methods(
    metrics: Mapping[str, Any], *, stage_b: Mapping[str, Any]
) -> dict[str, Any]:
    methods = metrics.get("methods")
    _require(
        isinstance(methods, Mapping) and set(methods) == METHOD_NAMES,
        "Stage-C method set differs",
    )
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for method_name in METHOD_NAMES:
        method = methods[method_name]
        _require(isinstance(method, Mapping), f"method {method_name} is malformed")
        variant = method.get("output_variant")
        expected_type = (
            "PHYSICAL_CLAMP_MIN_ZERO"
            if method_name in {CLAMP_BASE, CLAMP_REFINED}
            else "RAW_MODEL_OUTPUT"
        )
        _require(
            isinstance(variant, Mapping)
            and variant.get("type") == expected_type
            and variant.get("epsilon_fill") is False,
            f"method {method_name} output variant differs",
        )
        parsed[method_name] = {
            name: _metric(method.get(name), name=f"{method_name}.{name}")
            for name in REQUIRED_METRICS
        }
        _require(
            method.get("point_to_plane_error_m")
            == POINT_TO_PLANE_NOT_AVAILABLE,
            f"method {method_name} invents or alters point-to-plane evidence",
        )
    claims = metrics.get("claims")
    primary_health = (
        claims.get("primary_raw_output_health")
        if isinstance(claims, Mapping)
        else None
    )
    _require(
        isinstance(primary_health, Mapping)
        and primary_health
        == {
            name: parsed[RAW_REFINED][name]
            for name in (
                "output_invalid_rate",
                "output_negative_rate",
                "output_nan_rate",
                "output_infinite_rate",
                "output_zero_rate",
            )
        },
        "Stage-C primary raw output-health claim differs from raw refined metrics",
    )
    evaluated_windows = _positive_int(
        metrics.get("windows_evaluated"), "windows_evaluated"
    )
    expected_pixels = evaluated_windows * 384 * 768
    for method_name in METHOD_NAMES:
        for name in (*HEALTH_METRICS, "output_infinite_rate", "output_zero_rate"):
            _require(
                parsed[method_name][name]["valid"] is True
                and parsed[method_name][name]["count"] == expected_pixels,
                f"{method_name}.{name} denominator differs from evaluated HR pixels",
            )
    for name in (
        *ACCURACY_METRICS,
        "bad_2",
        "trusted_region_epe_px",
        "invalid_region_completeness",
    ):
        _require(
            parsed[RAW_BASE][name]["count"] == parsed[RAW_REFINED][name]["count"],
            f"raw base/refined {name} domains differ",
        )
    base_reproduction_checked = evaluated_windows == FORMAL_EVALUABLE_WINDOWS
    if base_reproduction_checked:
        stage_b_base_methods = stage_b.get("base_methods")
        _require(
            isinstance(stage_b_base_methods, Mapping),
            "audited Stage-B base methods are missing",
        )
        for method_name in (RAW_BASE, CLAMP_BASE):
            _require(
                parsed[method_name] == stage_b_base_methods.get(method_name),
                f"Stage-C {method_name} does not exactly reproduce Stage-B final metrics",
            )
    changes = {
        name: _metric_change(parsed[RAW_BASE][name], parsed[RAW_REFINED][name])
        for name in REQUIRED_METRICS
    }
    clamp_changes = {
        name: _metric_change(
            parsed[CLAMP_BASE][name], parsed[CLAMP_REFINED][name]
        )
        for name in REQUIRED_METRICS
    }
    comparisons = metrics.get("comparisons")
    _require(isinstance(comparisons, Mapping), "Stage-C comparisons are missing")
    raw_all = comparisons.get("raw_all_metric_changes")
    _require(isinstance(raw_all, Mapping), "raw_all_metric_changes is missing")
    paired = comparisons.get("paired_pixel_changes")
    _require(
        isinstance(paired, Mapping)
        and set(paired) == set(PAIRED_DIAGNOSTIC_METRICS),
        "paired raw pixel diagnostics are missing or malformed",
    )
    parsed_paired = {
        name: _metric(paired.get(name), name=f"paired_pixel_changes.{name}")
        for name in PAIRED_DIAGNOSTIC_METRICS
    }
    for name in REQUIRED_METRICS:
        _validate_recorded_change(raw_all.get(name), metric_name=name, recomputed=changes[name])
    _validate_recorded_change(
        comparisons.get("raw_epe_change"),
        metric_name="epe_px",
        recomputed=changes["epe_px"],
    )
    primary = comparisons.get("raw_refined_vs_base")
    _require(isinstance(primary, Mapping), "raw_refined_vs_base comparison is missing")
    for key, metric_name in (
        ("trusted_region_degradation", "trusted_region_epe_px"),
        ("low_confidence_epe_change", "low_confidence_epe_px"),
        ("invalid_region_completeness_change", "invalid_region_completeness"),
    ):
        _validate_recorded_change(
            primary.get(key), metric_name=metric_name, recomputed=changes[metric_name]
        )
    clamp_all = comparisons.get("clamp0_all_metric_changes")
    _require(isinstance(clamp_all, Mapping), "clamp0_all_metric_changes is missing")
    for name in REQUIRED_METRICS:
        _validate_recorded_change(
            clamp_all.get(name), metric_name=name, recomputed=clamp_changes[name]
        )
    clamp_primary = comparisons.get("clamp0_refined_vs_base")
    _require(
        isinstance(clamp_primary, Mapping),
        "clamp0_refined_vs_base comparison is missing",
    )
    for key, metric_name in (
        ("trusted_region_degradation", "trusted_region_epe_px"),
        ("low_confidence_epe_change", "low_confidence_epe_px"),
        ("invalid_region_completeness_change", "invalid_region_completeness"),
    ):
        _validate_recorded_change(
            clamp_primary.get(key),
            metric_name=metric_name,
            recomputed=clamp_changes[metric_name],
        )
    for raw_name, clamp_name in (
        (RAW_BASE, CLAMP_BASE),
        (RAW_REFINED, CLAMP_REFINED),
    ):
        raw = parsed[raw_name]
        clamp = parsed[clamp_name]
        for metric_name in (
            "output_invalid_rate",
            "output_nan_rate",
            "output_infinite_rate",
        ):
            _require(
                clamp[metric_name] == raw[metric_name],
                f"{clamp_name}.{metric_name} violates clamp0 conservation",
            )
        raw_negative = float(raw["output_negative_rate"]["value"])
        raw_zero = float(raw["output_zero_rate"]["value"])
        clamp_negative = float(clamp["output_negative_rate"]["value"])
        clamp_zero = float(clamp["output_zero_rate"]["value"])
        _require(
            clamp_negative == 0.0
            and float(clamp["output_negative_rate"]["numerator"]) == 0.0
            and clamp_zero >= raw_zero
            and math.isclose(
                clamp_negative + clamp_zero,
                raw_negative + raw_zero,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
            f"{clamp_name} violates negative-to-zero clamp0 conservation",
        )
        _require(
            math.isclose(
                float(clamp["output_zero_rate"]["numerator"]),
                float(raw["output_zero_rate"]["numerator"])
                + float(raw["output_negative_rate"]["numerator"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            f"{clamp_name} zero numerator does not equal raw zero plus negative",
        )
    return {
        "methods": parsed,
        "raw_changes": changes,
        "clamp0_changes": clamp_changes,
        "paired_pixel_changes": parsed_paired,
        "stage_b_base_reproduction_checked": base_reproduction_checked,
    }


def _validate_coverage(metrics: Mapping[str, Any]) -> dict[str, Any]:
    formal = metrics.get("formal_coverage")
    canonical = metrics.get("canonical_coverage")
    _require(isinstance(formal, Mapping), "Stage-C formal coverage is missing")
    expected = {
        "manifest_records": FORMAL_MANIFEST_RECORDS,
        "derived_endpoint_records": FORMAL_DERIVED_RECORDS,
        "evaluable_t3_windows": FORMAL_EVALUABLE_WINDOWS,
    }
    _require(canonical == expected, "Stage-C canonical coverage differs from 244/240/238")
    for name, expected_value in expected.items():
        _require(formal.get(name) == expected_value, f"Stage-C formal coverage {name} differs")
    verified: dict[str, Any] = {}
    for path_key, sha_key, label in (
        (
            "derived_cache_manifest_path",
            "derived_cache_manifest_sha256",
            "validation derived manifest",
        ),
        (
            "derived_run_receipt_path",
            "derived_run_receipt_sha256",
            "validation derived receipt",
        ),
        (
            "raw_vggt_cache_manifest_path",
            "raw_vggt_cache_manifest_sha256",
            "validation raw VGGT manifest",
        ),
    ):
        verified[path_key] = _verify_file_identity(
            {"path": formal.get(path_key), "sha256": formal.get(sha_key)},
            expected_path=None,
            name=label,
        )
    _require(
        verified["derived_cache_manifest_path"]["sha256"]
        == FORMAL_VALIDATION_DERIVED_MANIFEST_SHA256
        and verified["derived_run_receipt_path"]["sha256"]
        == FORMAL_VALIDATION_DERIVED_RECEIPT_SHA256
        and verified["raw_vggt_cache_manifest_path"]["sha256"]
        == FORMAL_VALIDATION_RAW_VGGT_MANIFEST_SHA256,
        "Stage-C validation cache artifacts are not the canonical receipts",
    )
    windows = _positive_int(metrics.get("windows_evaluated"), "windows_evaluated")
    full_windows = _positive_int(
        metrics.get("full_evaluable_windows"), "full_evaluable_windows"
    )
    _require(full_windows == FORMAL_EVALUABLE_WINDOWS, "full evaluable windows differs from 238")
    _require(windows <= full_windows, "evaluated windows exceeds formal corpus")
    _require(
        metrics.get("fixed_hr_crop") == [384, 768],
        "Stage-C evaluation crop is not fixed 384x768",
    )
    source_hashes = metrics.get("source_hashes")
    resolved_config = metrics.get("resolved_config")
    data = resolved_config.get("data") if isinstance(resolved_config, Mapping) else None
    _require(isinstance(source_hashes, Mapping), "Stage-C source hashes are missing")
    _require(isinstance(data, Mapping), "Stage-C resolved evaluation data config is missing")
    validation_manifest = _resolve_recorded_path(
        data.get("manifest_path"), "validation manifest"
    )
    manifest_payload = _read_regular_file(validation_manifest, "validation manifest")
    manifest_sha = _sha256_bytes(manifest_payload)
    _require(
        manifest_sha == FORMAL_VALIDATION_MANIFEST_SHA256
        and source_hashes.get("validation_manifest_sha256") == manifest_sha,
        "Stage-C validation manifest SHA is not canonical",
    )
    return {
        **expected,
        "windows_evaluated": windows,
        "full_selection": windows == full_windows,
        "validation_manifest_path": str(validation_manifest),
        "validation_manifest_sha256": manifest_sha,
        "verified_coverage_artifacts": verified,
    }


def _validate_claims(
    metrics: Mapping[str, Any],
    *,
    producer_eligible: bool,
    full_selection: bool,
) -> dict[str, Any]:
    claims = metrics.get("claims")
    target = metrics.get("target")
    postprocess = metrics.get("postprocess_contract")
    _require(isinstance(claims, Mapping), "Stage-C claims are missing")
    _require(isinstance(target, Mapping), "Stage-C target declaration is missing")
    _require(isinstance(postprocess, Mapping), "Stage-C postprocess contract is missing")
    _require(
        claims.get("paper_ground_truth") is False
        and claims.get("paper_accuracy") is False
        and claims.get("pseudo_gt_engineering_only") is True
        and claims.get("future_frames") is False
        and claims.get("point_to_plane") == "NOT_AVAILABLE"
        and claims.get("performance_acceptance_claimed") is False,
        "Stage-C evaluator claim boundary is not engineering-only",
    )
    _require(
        claims.get("primary_claim_method") == RAW_REFINED
        and claims.get("primary_claim_variant") == "RAW_MODEL_OUTPUT"
        and claims.get("primary_comparison") == "raw_refined_vs_base"
        and claims.get("clamp0_acceptance_owner") is False,
        "Stage-C primary claim is not raw refined-vs-base",
    )
    _require(
        target.get("type") == PSEUDO_GT_TARGET,
        "Stage-C target is not the declared FFS pseudo-GT",
    )
    _require(
        postprocess.get("role") == "DECLARED_PHYSICAL_POSTPROCESS_DIAGNOSTIC"
        and postprocess.get("operation") == CLAMP0_OPERATION
        and postprocess.get("epsilon_fill") is False
        and postprocess.get("zero_remains_invalid") is True
        and postprocess.get("nan_and_positive_negative_infinity_preserved") is True
        and postprocess.get("completeness_is_not_fabricated") is True
        and postprocess.get("raw_rows_are_retained") is True,
        "Stage-C clamp0 diagnostic boundary differs",
    )
    _require(
        claims.get("acceptance_eligible") is producer_eligible,
        "Stage-C evaluator acceptance_eligible flag is inconsistent",
    )
    expected_status = (
        "EVALUATION_COMPLETE"
        if producer_eligible
        else (
            "LIMITED_SMOKE_ONLY"
            if not full_selection
            else "INTERMEDIATE_CHECKPOINT_EVALUATION"
        )
    )
    _require(metrics.get("status") == expected_status, "Stage-C evaluator status is inconsistent")
    return {
        "producer_acceptance_eligible": producer_eligible,
        "paper_ground_truth": False,
        "paper_accuracy": False,
        "pseudo_gt_engineering_only": True,
        "primary_method": RAW_REFINED,
        "primary_variant": "RAW_MODEL_OUTPUT",
        "clamp0_acceptance_owner": False,
    }


def _validate_crop_execution(
    metrics: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    crop = metrics.get("crop_contract")
    execution = metrics.get("execution_contract")
    _require(isinstance(crop, Mapping), "Stage-C crop contract is missing")
    _require(isinstance(execution, Mapping), "Stage-C execution contract is missing")
    crop_eligible = bool(
        crop.get("trained_hr_crop") == [384, 768]
        and crop.get("evaluation_hr_crop") == [384, 768]
        and crop.get("canonical_hr_crop") == [384, 768]
        and crop.get("exact_training_crop") is True
        and crop.get("canonical_crop") is True
        and crop.get("training_crop_mode") == "random"
        and crop.get("evaluation_crop_mode") == "fixed"
        and crop.get("canonical_modes") is True
    )
    _require(
        crop.get("eligible") is crop_eligible,
        "Stage-C crop eligibility flag is inconsistent",
    )
    evaluation_runtime = execution.get("evaluation_runtime")
    stage_c_metadata = metrics.get("stage_c_checkpoint")
    checkpoint_runtime = (
        stage_c_metadata.get("training_runtime_receipt")
        if isinstance(stage_c_metadata, Mapping)
        else None
    )
    recorded_runtime_receipt = execution.get("recorded_training_runtime")
    _require(
        isinstance(checkpoint_runtime, Mapping)
        and checkpoint_runtime == recorded_runtime_receipt,
        "Stage-C recorded training runtime differs between checkpoint/evaluation",
    )
    recorded_runtime = checkpoint_runtime.get("recorded")
    _require(
        isinstance(recorded_runtime, Mapping),
        "Stage-C producer runtime values are missing",
    )
    deterministic_training = bool(
        recorded_runtime.get("deterministic_algorithms_enabled") is True
        and recorded_runtime.get("deterministic_algorithms_warn_only") is False
        and recorded_runtime.get("cublas_workspace_config") == ":4096:8"
        and recorded_runtime.get("cudnn_deterministic") is True
        and recorded_runtime.get("cudnn_benchmark") is False
    )
    formal_training_runtime = bool(
        recorded_runtime.get("device_type") == "cuda"
        and isinstance(recorded_runtime.get("device_name"), str)
        and "5090" in recorded_runtime["device_name"]
        and recorded_runtime.get("device_capability") == [12, 0]
        and recorded_runtime.get("cuda_version") == "12.8"
        and recorded_runtime.get("cuda_available") is True
        and recorded_runtime.get("bf16_supported") is True
        and recorded_runtime.get("autocast_enabled") is True
        and recorded_runtime.get("autocast_dtype") == "torch.bfloat16"
        and deterministic_training
        and recorded_runtime.get("strict_determinism_eligible") is True
        and recorded_runtime.get("formal_cuda_bf16_eligible") is True
        and checkpoint_runtime.get("eligible") is True
        and checkpoint_runtime.get("producer_cuda_bf16_eligible") is True
        and checkpoint_runtime.get("strict_determinism_eligible") is True
        and checkpoint_runtime.get("cuda_12_8_or_newer") is True
        and checkpoint_runtime.get("blackwell_capability") is True
        and checkpoint_runtime.get("rtx_5090") is True
    )
    audit_runtime = training.get("training_runtime")
    _require(
        isinstance(audit_runtime, Mapping)
        and audit_runtime.get("device") == recorded_runtime.get("device")
        and audit_runtime.get("device_name") == recorded_runtime.get("device_name")
        and audit_runtime.get("device_capability")
        == recorded_runtime.get("device_capability")
        and audit_runtime.get("torch_version")
        == recorded_runtime.get("torch_version")
        and audit_runtime.get("cuda_version") == recorded_runtime.get("cuda_version")
        and audit_runtime.get("native_cuda_bf16") is True
        and audit_runtime.get("strict_determinism") is True
        and audit_runtime.get("formal_cuda_bf16_eligible") is True,
        "Stage-C training audit runtime differs from evaluator checkpoint receipt",
    )
    deterministic_evaluation = bool(
        isinstance(evaluation_runtime, Mapping)
        and evaluation_runtime.get("deterministic_algorithms_enabled") is True
        and evaluation_runtime.get("deterministic_algorithms_warn_only") is False
        and evaluation_runtime.get("cublas_workspace_config") == ":4096:8"
        and evaluation_runtime.get("cudnn_deterministic") is True
        and evaluation_runtime.get("cudnn_benchmark") is False
        and evaluation_runtime.get("strict_determinism_eligible") is True
    )
    formal_evaluation_runtime = bool(
        isinstance(evaluation_runtime, Mapping)
        and evaluation_runtime.get("device_name") == recorded_runtime.get("device_name")
        and evaluation_runtime.get("device_capability")
        == recorded_runtime.get("device_capability")
        and evaluation_runtime.get("torch_version")
        == recorded_runtime.get("torch_version")
        and evaluation_runtime.get("cuda_version") == recorded_runtime.get("cuda_version")
        and evaluation_runtime.get("cuda_bf16_supported") is True
        and evaluation_runtime.get("autocast_dtype") == "torch.bfloat16"
        and evaluation_runtime.get("cuda_12_8_or_newer") is True
        and evaluation_runtime.get("blackwell_capability") is True
        and evaluation_runtime.get("rtx_5090") is True
        and evaluation_runtime.get("versions_and_device_match_training") is True
        and deterministic_evaluation
        and evaluation_runtime.get("eligible") is True
    )
    execution_eligible = bool(
        execution.get("saved_precision") == "bf16"
        and execution.get("evaluation_precision") == "bf16"
        and execution.get("saved_optimizer") == "adamw"
        and execution.get("canonical_training_values") is True
        and execution.get("canonical_batch_schedule") is True
        and execution.get("recorded_training_eligible") is formal_training_runtime
        and formal_training_runtime
        and formal_evaluation_runtime
    )
    _require(
        execution.get("eligible") is execution_eligible,
        "Stage-C execution eligibility flag is inconsistent",
    )
    return {
        "crop_eligible": crop_eligible,
        "execution_eligible": execution_eligible,
        "training_runtime_eligible": formal_training_runtime,
        "evaluation_runtime_eligible": formal_evaluation_runtime,
        "trained_hr_crop": crop.get("trained_hr_crop"),
        "evaluation_hr_crop": crop.get("evaluation_hr_crop"),
        "training_crop_mode": crop.get("training_crop_mode"),
        "evaluation_crop_mode": crop.get("evaluation_crop_mode"),
    }


def _validate_runtime_geometry(
    metrics: Mapping[str, Any], *, windows_evaluated: int
) -> dict[str, Any]:
    geometry = metrics.get("runtime_geometry_statistics")
    _require(isinstance(geometry, Mapping), "runtime geometry statistics are missing")
    contract = geometry.get("contract")
    _require(
        isinstance(contract, Mapping)
        and contract.get("version") == "audited_same_row_rectified_pixels_v1"
        and contract.get("runtime_right_row_scale") == 1.0
        and contract.get("runtime_right_row_offset_hr_px") == 0.0
        and contract.get("vertical_correspondence") == "v_right=v_left"
        and contract.get("horizontal_correspondence")
        == "u_right=u_left-disparity-delta",
        "runtime same-row epipolar geometry contract differs",
    )

    def exact_stat(value: object, expected: float, name: str) -> dict[str, Any]:
        _require(isinstance(value, Mapping), f"runtime geometry {name} is missing")
        _require(
            value.get("valid") is True
            and value.get("count") == windows_evaluated,
            f"runtime geometry {name} coverage differs",
        )
        for field in ("mean", "minimum", "maximum"):
            _require(
                _finite_float(value.get(field), f"runtime geometry {name}.{field}")
                == expected,
                f"runtime geometry {name}.{field} differs from {expected}",
            )
        return dict(value)

    row_scale = exact_stat(geometry.get("right_row_scale"), 1.0, "right_row_scale")
    row_offset = exact_stat(
        geometry.get("right_row_offset_hr_px"),
        0.0,
        "right_row_offset_hr_px",
    )
    health = geometry.get("horizontal_correspondence_health")
    _require(
        geometry.get("right_intrinsics_source") == "manifest.K_right"
        and geometry.get("metadata_runtime_mismatch_is_expected") is True
        and isinstance(health, Mapping)
        and health.get("role") == "DIAGNOSTIC_ONLY"
        and health.get("changes_training_mask") is False
        and health.get("changes_accuracy_metrics") is False,
        "runtime geometry diagnostic/metric ownership differs",
    )
    return {
        "contract_version": contract["version"],
        "right_row_scale": row_scale,
        "right_row_offset_hr_px": row_offset,
        "horizontal_correspondence_role": "DIAGNOSTIC_ONLY",
        "changes_accuracy_metrics": False,
    }


def _validate_lineage(
    metrics: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    stage_c = metrics.get("stage_c_checkpoint")
    stage_b_metrics = metrics.get("stage_b_base_checkpoint")
    source = metrics.get("source_hashes")
    completion = metrics.get("checkpoint_completion")
    lineage = metrics.get("lineage")
    _require(isinstance(stage_c, Mapping), "metrics Stage-C checkpoint metadata is missing")
    _require(isinstance(stage_b_metrics, Mapping), "metrics Stage-B base metadata is missing")
    _require(isinstance(source, Mapping), "metrics source hashes are missing")
    _require(isinstance(completion, Mapping), "metrics checkpoint completion is missing")
    _require(isinstance(lineage, Mapping), "metrics lineage is missing")
    selected = training["checkpoint"]
    _require(
        stage_c.get("checkpoint_sha256") == selected["sha256"]
        and source.get("stage_c_checkpoint_sha256") == selected["sha256"]
        and stage_c.get("step") == selected["step"]
        and stage_c.get("git_hash") == selected["git_hash"]
        and stage_c.get("parameter_count") == EXPECTED_REFINER_PARAMETERS
        and _resolve_recorded_path(stage_c.get("path"), "metrics Stage-C checkpoint")
        == Path(selected["path"]).resolve(),
        "metrics Stage-C checkpoint differs from training audit",
    )
    config_sha = _canonical_config_sha256(stage_c.get("config"))
    _require(config_sha == selected["config_sha256"], "metrics Stage-C config SHA differs")
    base = stage_b["checkpoint"]
    _require(
        stage_b_metrics.get("sha256") == base["sha256"]
        and source.get("stage_b_checkpoint_sha256") == base["sha256"]
        and stage_b_metrics.get("step") == FORMAL_STAGE_B_STEPS
        and _resolve_recorded_path(stage_b_metrics.get("path"), "metrics Stage-B checkpoint")
        == Path(base["path"]).resolve(),
        "metrics Stage-B checkpoint differs from final Stage-B report",
    )
    recorded_base = stage_c.get("base_checkpoint")
    _require(
        isinstance(recorded_base, Mapping)
        and recorded_base.get("sha256") == base["sha256"]
        and recorded_base.get("step") == FORMAL_STAGE_B_STEPS
        and training["base_checkpoint"].get("sha256") == base["sha256"],
        "Stage-C frozen-base lineage differs from final Stage-B report",
    )
    stage_c_completion = completion.get("stage_c")
    stage_b_completion = completion.get("stage_b_base")
    _require(isinstance(stage_c_completion, Mapping), "Stage-C completion block is malformed")
    _require(isinstance(stage_b_completion, Mapping), "Stage-B completion block is malformed")
    stage_c_complete = bool(
        stage_c_completion.get("actual_step") == FORMAL_STAGE_C_STEPS
        and stage_c_completion.get("configured_steps") == FORMAL_STAGE_C_STEPS
        and stage_c_completion.get("execution_complete") is True
        and stage_c_completion.get("canonical_schedule") is True
        and stage_c_completion.get("complete") is True
    )
    stage_b_complete = bool(
        stage_b_completion.get("actual_step") == FORMAL_STAGE_B_STEPS
        and stage_b_completion.get("configured_steps") == FORMAL_STAGE_B_STEPS
        and stage_b_completion.get("execution_complete") is True
        and stage_b_completion.get("canonical_schedule") is True
        and stage_b_completion.get("complete") is True
    )
    _require(
        completion.get("all_complete") is (stage_c_complete and stage_b_complete),
        "metrics all_complete flag is inconsistent",
    )
    _require(stage_b_complete, "metrics Stage-B base is not complete")
    _require(
        stage_c_complete is bool(training["formally_complete"]),
        "metrics Stage-C completion differs from training audit",
    )
    runtime_bundle = stage_c.get("runtime_source_bundle")
    runtime_receipt = source.get("runtime_source_bundle")
    training_runtime = training["runtime_source_bundle"]
    bundle_records = (
        runtime_bundle.get("files") if isinstance(runtime_bundle, Mapping) else None
    )
    receipt_files = (
        runtime_receipt.get("files")
        if isinstance(runtime_receipt, Mapping)
        else None
    )
    _require(
        isinstance(runtime_bundle, Mapping)
        and isinstance(runtime_receipt, Mapping)
        and runtime_bundle.get("git_head") == selected["git_hash"]
        and runtime_bundle.get("bundle_sha256") == training_runtime.get("bundle_sha256")
        and runtime_receipt.get("checkpoint_bundle_sha256")
        == training_runtime.get("bundle_sha256")
        and runtime_receipt.get("checkpoint_git_hash") == selected["git_hash"]
        and runtime_receipt.get("all_byte_identical") is True
        and isinstance(bundle_records, list)
        and len(bundle_records) == 52
        and isinstance(receipt_files, Mapping)
        and len(receipt_files) == 52,
        "metrics Stage-C 52-file runtime lineage differs from training audit",
    )
    bundle_paths = [
        record.get("path") if isinstance(record, Mapping) else None
        for record in bundle_records
    ]
    _require(
        all(isinstance(path, str) and bool(path) for path in bundle_paths)
        and len(set(bundle_paths)) == 52
        and set(bundle_paths) == set(receipt_files),
        "Stage-C runtime bundle paths are duplicated or differ from receipt",
    )
    encoded_bundle = json.dumps(
        {"git_head": selected["git_hash"], "files": bundle_records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _require(
        _sha256_bytes(encoded_bundle) == runtime_bundle.get("bundle_sha256"),
        "Stage-C runtime bundle digest does not match its file records",
    )
    for record in bundle_records:
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and bool(record.get("path")),
            "Stage-C runtime bundle file record is malformed",
        )
        relative = str(record["path"])
        recorded_sha = _require_sha256(
            record.get("sha256"), f"runtime bundle source {relative}"
        )
        pair = receipt_files.get(relative)
        _require(
            isinstance(pair, Mapping)
            and pair.get("current_sha256") == recorded_sha
            and pair.get("checkpoint_commit_sha256") == recorded_sha,
            f"runtime source receipt differs for {relative}",
        )
    recomputed = lineage.get("recomputed_stage_c_training")
    held_out = lineage.get("held_out_validation")
    stage_lineage = lineage.get("stage_c_and_base")
    right_sources = lineage.get("validation_endpoint_right_sources")
    raw_payloads = lineage.get("validation_raw_payload_audit")
    rectification = lineage.get("rectification_audit")
    _require(isinstance(recomputed, Mapping), "recomputed Stage-C training lineage is missing")
    _require(isinstance(held_out, Mapping), "held-out Stage-C lineage is missing")
    _require(isinstance(stage_lineage, Mapping), "Stage-C/base lineage is missing")
    _require(isinstance(right_sources, Mapping), "validation right-source lineage is missing")
    _require(isinstance(raw_payloads, Mapping), "validation raw-payload lineage is missing")
    _require(isinstance(rectification, Mapping), "rectification lineage is missing")
    _require(
        recomputed.get("manifest_sha256") == FORMAL_TRAIN_MANIFEST_SHA256
        and recomputed.get("derived_endpoint_records") == FORMAL_TRAIN_DERIVED_RECORDS
        and recomputed.get("evaluable_t3_windows") == FORMAL_TRAIN_WINDOWS
        and stage_c.get("base_lineage") == recomputed.get("base_lineage")
        and stage_c.get("raw_lineage") == recomputed.get("raw_lineage"),
        "recomputed Stage-C training lineage differs",
    )
    training_right_sources = recomputed.get("audited_endpoint_right_source_digest")
    training_raw_lineage = recomputed.get("raw_lineage")
    training_derived = (
        training_raw_lineage.get("derived_cache_lineage")
        if isinstance(training_raw_lineage, Mapping)
        else None
    )
    _require(
        isinstance(training_right_sources, Mapping)
        and training_right_sources.get("records") == FORMAL_TRAIN_WINDOWS
        and training_right_sources.get("sha256")
        == FORMAL_TRAIN_RIGHT_SOURCE_DIGEST_SHA256,
        "Stage-C training endpoint right-source digest differs",
    )
    _require(
        isinstance(training_raw_lineage, Mapping)
        and isinstance(training_derived, Mapping)
        and training_raw_lineage.get("raw_vggt_receipt_sha256")
        == FORMAL_TRAIN_RAW_VGGT_RECEIPT_SHA256
        and training_derived.get("run_receipt_sha256")
        == FORMAL_TRAIN_DERIVED_RECEIPT_SHA256
        and training_derived.get("cache_manifest_sha256")
        == FORMAL_TRAIN_DERIVED_MANIFEST_SHA256
        and training_derived.get("selected_records")
        == FORMAL_TRAIN_DERIVED_RECORDS,
        "Stage-C training raw/derived cache receipt lineage differs",
    )
    _require(
        held_out.get("formal_holdout") is True
        and held_out.get("non_holdout_smoke_override") is False
        and held_out.get("same_manifest") is False
        and held_out.get("sequence_overlap") == []
        and held_out.get("training_manifest_sha256") == FORMAL_TRAIN_MANIFEST_SHA256
        and held_out.get("evaluation_manifest_sha256")
        == FORMAL_VALIDATION_MANIFEST_SHA256,
        "Stage-C held-out video-isolation lineage differs",
    )
    evaluation_raw_vggt = held_out.get("evaluation_raw_vggt")
    _require(
        isinstance(evaluation_raw_vggt, Mapping)
        and evaluation_raw_vggt.get("receipt_sha256")
        == FORMAL_VALIDATION_RAW_VGGT_RECEIPT_SHA256
        and evaluation_raw_vggt.get("manifest_sha256")
        == FORMAL_VALIDATION_MANIFEST_SHA256,
        "Stage-C held-out raw VGGT receipt lineage differs",
    )
    _require(
        stage_lineage.get("sequence_overlap") == []
        and stage_lineage.get("base_checkpoint_sha256") == base["sha256"]
        and stage_lineage.get("base_checkpoint_step") == FORMAL_STAGE_B_STEPS
        and stage_lineage.get("stage_c_training_manifest_sha256")
        == FORMAL_TRAIN_MANIFEST_SHA256,
        "Stage-C/base lineage cross-check differs",
    )
    _require(
        right_sources.get("records") == coverage["windows_evaluated"]
        and right_sources.get("all_source_sha256_match") is True,
        "held-out endpoint right-source SHA audit differs",
    )
    right_source_sha = _require_sha256(
        right_sources.get("sha256"), "held-out right-source digest"
    )
    if coverage["full_selection"]:
        _require(
            right_source_sha == FORMAL_VALIDATION_RIGHT_SOURCE_DIGEST_SHA256,
            "full held-out right-source digest is not canonical",
        )
    _require(
        raw_payloads.get("derived_records") == FORMAL_DERIVED_RECORDS
        and raw_payloads.get("vggt_payloads_hashed") == FORMAL_DERIVED_RECORDS
        and raw_payloads.get("ffs_payloads_hashed") == FORMAL_DERIVED_RECORDS
        and raw_payloads.get("all_payload_sha256_match") is True,
        "held-out raw payload SHA audit differs",
    )
    _require_sha256(
        raw_payloads.get("canonical_reference_digest_sha256"),
        "held-out raw payload digest",
    )
    _require(
        raw_payloads.get("canonical_reference_digest_sha256")
        == FORMAL_VALIDATION_RAW_PAYLOAD_DIGEST_SHA256,
        "held-out raw-payload digest is not canonical",
    )
    training_audit_rect = metrics.get("stage_c_checkpoint", {}).get(
        "rectification_audit"
    )
    _require(
        isinstance(training_audit_rect, Mapping)
        and training_audit_rect.get("sha256")
        == training["rectification_audit"].get("sha256")
        and rectification.get("sha256") == training_audit_rect.get("sha256")
        and rectification.get("sha256") == FORMAL_RECTIFICATION_AUDIT_SHA256
        and rectification.get("status") == "PASS",
        "metrics rectification-audit lineage differs",
    )
    evaluator_path = _resolve_recorded_path(source.get("evaluator_path"), "Stage-C evaluator")
    evaluator_payload = _read_regular_file(evaluator_path, "Stage-C evaluator")
    evaluator_sha = _sha256_bytes(evaluator_payload)
    _require(
        source.get("evaluator_sha256") == evaluator_sha
        and evaluator_sha == FORMAL_STAGE_C_EVALUATOR_SHA256,
        "Stage-C evaluator source SHA-256 mismatch or non-canonical evaluator",
    )
    repository_hash = source.get("repository_git_hash")
    _require(
        isinstance(repository_hash, str)
        and GIT_HASH_PATTERN.fullmatch(repository_hash) is not None,
        "Stage-C evaluator repository git hash is malformed",
    )
    _require(
        repository_hash == selected["git_hash"] == FORMAL_STAGE_C_TRAINING_GIT_HASH,
        "Stage-C evaluator repository Git hash differs from training checkpoint",
    )
    return {
        "stage_c_checkpoint": dict(selected),
        "stage_b_checkpoint": dict(base),
        "stage_c_complete": stage_c_complete,
        "stage_b_complete": stage_b_complete,
        "all_complete": stage_c_complete and stage_b_complete,
        "training_manifest_sha256": FORMAL_TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": coverage["validation_manifest_sha256"],
        "video_disjoint": True,
        "runtime_source_files": 52,
        "validation_right_sources_hashed": FORMAL_EVALUABLE_WINDOWS,
        "validation_raw_vggt_payloads_hashed": FORMAL_DERIVED_RECORDS,
        "validation_raw_ffs_payloads_hashed": FORMAL_DERIVED_RECORDS,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_sha,
    }


def _gate_report(parsed: Mapping[str, Any]) -> dict[str, Any]:
    methods = parsed["methods"]
    changes = parsed["raw_changes"]
    gates: dict[str, dict[str, Any]] = {}
    for name in ACCURACY_METRICS:
        change = changes[name]
        passed = bool(
            change["valid"]
            and float(change["candidate"]["value"])
            < float(change["baseline"]["value"])
        )
        gates[f"raw_{name}_improves"] = {
            **change,
            "criterion": "candidate < baseline on identical raw domain",
            "required_for_final_gate": name in PRIMARY_ACCURACY_GATES,
            "passed": passed,
        }
    trusted = changes["trusted_region_epe_px"]
    trusted_relative = (
        float(trusted["relative_change_percent"])
        if trusted["relative_valid"]
        else None
    )
    gates["raw_trusted_region_degradation_at_most_2_percent"] = {
        **trusted,
        "criterion": "relative_change_percent <= 2.0",
        "limit_percent": TRUSTED_DEGRADATION_LIMIT_PERCENT,
        "required_for_final_gate": True,
        "passed": bool(
            trusted_relative is not None
            and trusted_relative <= TRUSTED_DEGRADATION_LIMIT_PERCENT
        ),
    }
    for name in HEALTH_METRICS:
        change = changes[name]
        candidate = float(methods[RAW_REFINED][name]["value"])
        baseline = float(methods[RAW_BASE][name]["value"])
        gates[f"raw_{name}_below_0_5_percent"] = {
            **change,
            "candidate_rate": candidate,
            "candidate_percent": candidate * 100.0,
            "threshold_rate": RAW_HEALTH_THRESHOLD,
            "threshold_percent": RAW_HEALTH_THRESHOLD * 100.0,
            "criterion": "candidate < 0.005",
            "required_for_final_gate": True,
            "not_worse_than_base_diagnostic": candidate <= baseline,
            "passed": candidate < RAW_HEALTH_THRESHOLD,
        }
    candidate_sign_rate = sum(
        float(methods[RAW_REFINED][name]["value"])
        for name in (
            "output_negative_rate",
            "output_nan_rate",
            "output_infinite_rate",
        )
    )
    baseline_sign_rate = sum(
        float(methods[RAW_BASE][name]["value"])
        for name in (
            "output_negative_rate",
            "output_nan_rate",
            "output_infinite_rate",
        )
    )
    gates["raw_sign_rate_below_0_5_percent"] = {
        "baseline_rate": baseline_sign_rate,
        "candidate_rate": candidate_sign_rate,
        "candidate_percent": candidate_sign_rate * 100.0,
        "threshold_rate": RAW_HEALTH_THRESHOLD,
        "threshold_percent": RAW_HEALTH_THRESHOLD * 100.0,
        "formula": "negative_rate + nan_rate + infinite_rate",
        "criterion": "candidate < 0.005",
        "required_for_final_gate": True,
        "not_worse_than_base_diagnostic": (
            candidate_sign_rate <= baseline_sign_rate
        ),
        "passed": candidate_sign_rate < RAW_HEALTH_THRESHOLD,
    }
    required_gates = [
        gate
        for gate in gates.values()
        if gate.get("required_for_final_gate") is True
    ]
    return {
        "primary_comparison": "raw_refined_vs_base",
        "clamp0_used_for_any_gate": False,
        "target": PSEUDO_GT_TARGET,
        "paper_ground_truth": False,
        "paper_accuracy_claim": False,
        "gates": gates,
        "all_required_gates_pass": all(gate["passed"] for gate in required_gates),
        "all_reported_checks_pass": all(gate["passed"] for gate in gates.values()),
        "raw_invalid_region_completeness_change_diagnostic": changes[
            "invalid_region_completeness"
        ],
        "raw_output_infinite_rate_diagnostic": methods[RAW_REFINED][
            "output_infinite_rate"
        ],
    }


def audit_epipolar_evaluation(
    evaluation_dir: str | Path,
    *,
    stage_c_training_audit_path: str | Path,
    stage_b_final_report_path: str | Path,
    stage_c_training_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Cross-audit Stage-C evaluation/training/base receipts without mutation."""

    original_eval = Path(evaluation_dir).expanduser()
    _require(original_eval.exists(), f"Stage-C evaluation directory is missing: {original_eval}")
    _require(not original_eval.is_symlink(), "Stage-C evaluation directory must not be a symlink")
    eval_root = original_eval.resolve()
    _require(eval_root.is_dir(), "Stage-C evaluation path is not a directory")
    metrics = _load_json(eval_root / "metrics.json", "Stage-C metrics.json")
    training_audit = _load_json(
        stage_c_training_audit_path, "Stage-C training final/in-progress audit"
    )
    training_summary = (
        None
        if stage_c_training_summary_path is None
        else _load_json(stage_c_training_summary_path, "Stage-C training run summary")
    )
    stage_b_report = _load_json(stage_b_final_report_path, "Stage-B final report")
    _require(
        metrics.value.get("schema_version") == 1
        and metrics.value.get("stage") == METRICS_STAGE
        and set(metrics.value) == EXPECTED_METRICS_TOP_LEVEL,
        "Stage-C metrics component/schema differs",
    )
    training = _validate_training_audit(training_audit, summary=training_summary)
    stage_b = _validate_stage_b_report(stage_b_report)
    coverage = _validate_coverage(metrics.value)
    lineage = _validate_lineage(
        metrics.value,
        training=training,
        stage_b=stage_b,
        coverage=coverage,
    )
    numerical_contract = _validate_crop_execution(
        metrics.value, training=training
    )
    runtime_geometry = _validate_runtime_geometry(
        metrics.value,
        windows_evaluated=int(coverage["windows_evaluated"]),
    )
    producer_eligible = bool(
        coverage["full_selection"]
        and lineage["all_complete"]
        and training["formally_complete"]
        and numerical_contract["crop_eligible"]
        and numerical_contract["execution_eligible"]
    )
    claims = _validate_claims(
        metrics.value,
        producer_eligible=producer_eligible,
        full_selection=bool(coverage["full_selection"]),
    )
    parsed = _validate_methods(metrics.value, stage_b=stage_b)
    gate = _gate_report(parsed)
    if not producer_eligible:
        status = "INELIGIBLE_FOR_FINAL_GATE"
        final_gate_result = "INELIGIBLE"
    elif gate["all_required_gates_pass"]:
        status = "STAGE_C_M5_GATE_PASS"
        final_gate_result = "PASS"
    else:
        status = "STAGE_C_M5_GATE_FAIL"
        final_gate_result = "FAIL"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": AUDIT_COMPONENT,
        "status": status,
        "read_only": True,
        "final_gate": {
            "eligible": producer_eligible,
            "result": final_gate_result,
            "all_required_gates_pass": gate["all_required_gates_pass"],
            "all_reported_checks_pass": gate["all_reported_checks_pass"],
            "limited_or_intermediate_cannot_pass": True,
        },
        "claims": {
            **claims,
            "engineering_evidence_only": True,
            "paper_claim_eligible": False,
            "clamp0_or_pseudo_gt_promoted": False,
            "refined_temporal_metric_available": False,
            "refined_temporal_improvement_claimed": False,
        },
        "gates": gate,
        "coverage": coverage,
        "numerical_contract": numerical_contract,
        "runtime_geometry": runtime_geometry,
        "lineage": lineage,
        "training": training,
        "stage_b_final": stage_b,
        "artifacts": {
            "metrics": metrics.identity(),
            "stage_c_training_audit": training_audit.identity(),
            "stage_c_training_summary": (
                None if training_summary is None else training_summary.identity()
            ),
            "stage_b_final_report": stage_b_report.identity(),
        },
    }


def _safe_json_output(
    path: Path,
    *,
    protected_files: Sequence[Path],
    protected_roots: Sequence[Path],
) -> Path:
    output = path.expanduser().resolve()
    _require(
        all(output != item.expanduser().resolve() for item in protected_files),
        "--json-out must not overwrite any audited input artifact",
    )
    _require(
        all(
            output != root.expanduser().resolve()
            and root.expanduser().resolve() not in output.parents
            for root in protected_roots
        ),
        "--json-out must not write inside an audited evaluation/training directory",
    )
    return output


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Stage-C held-out metrics against final training/base receipts."
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--stage-c-training-audit", type=Path, required=True)
    parser.add_argument("--stage-c-training-summary", type=Path)
    parser.add_argument("--stage-b-final-report", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_epipolar_evaluation(
            args.evaluation_dir,
            stage_c_training_audit_path=args.stage_c_training_audit,
            stage_c_training_summary_path=args.stage_c_training_summary,
            stage_b_final_report_path=args.stage_b_final_report,
        )
        protected_files = [
            args.stage_c_training_audit,
            args.stage_b_final_report,
        ]
        if args.stage_c_training_summary is not None:
            protected_files.append(args.stage_c_training_summary)
        json_out = (
            None
            if args.json_out is None
            else _safe_json_output(
                args.json_out,
                protected_files=protected_files,
                protected_roots=[
                    args.evaluation_dir,
                    Path(report["training"]["output_dir"]),
                ],
            )
        )
        if json_out is not None:
            _write_json_atomic(json_out, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["status"] == "STAGE_C_M5_GATE_PASS" else 1
    except (EpipolarEvaluationAuditError, OSError) as exc:
        error = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "component": AUDIT_COMPONENT,
            "status": "INVALID_ARTIFACT_SET",
            "error": str(exc),
        }
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
