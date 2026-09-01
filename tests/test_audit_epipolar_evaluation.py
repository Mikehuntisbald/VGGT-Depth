from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import tools.audit_epipolar_evaluation as auditor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_metrics_csv(path: Path, methods: dict) -> None:
    metric_names = sorted(auditor.REQUIRED_METRICS)
    fields = ["method", "target_type", "paper_ground_truth", "point_to_plane"]
    for name in metric_names:
        fields.extend((name, f"{name}_valid", f"{name}_count", f"{name}_numerator"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name in (
            auditor.RAW_BASE,
            auditor.CLAMP_BASE,
            auditor.RAW_REFINED,
            auditor.CLAMP_REFINED,
        ):
            method = methods[method_name]
            row = {
                "method": method_name,
                "target_type": auditor.PSEUDO_GT_TARGET,
                "paper_ground_truth": False,
                "point_to_plane": "NOT_AVAILABLE",
            }
            for name in metric_names:
                metric = method[name]
                row[name] = metric["value"]
                row[f"{name}_valid"] = metric["valid"]
                row[f"{name}_count"] = metric["count"]
                row[f"{name}_numerator"] = metric["numerator"]
            writer.writerow(row)


def _metric(value: float, count: int) -> dict:
    return {
        "value": value,
        "numerator": value * count,
        "count": count,
        "valid": True,
    }


def _invalid_metric(count: int = 0) -> dict:
    return {
        "value": None,
        "numerator": None,
        "count": count,
        "valid": False,
    }


def _horizontal_record(domain: int, *, left_oob: int) -> dict:
    finite = domain
    in_bounds = finite - left_oob
    return {
        "domain_pixel_count": domain,
        "finite_count": finite,
        "nonfinite_count": 0,
        "in_bounds_count": in_bounds,
        "oob_count": left_oob,
        "left_oob_count": left_oob,
        "right_oob_count": 0,
        "finite_rate": finite / domain if domain else None,
        "nonfinite_rate": 0.0 if domain else None,
        "in_bounds_rate": in_bounds / domain if domain else None,
        "oob_rate": left_oob / domain if domain else None,
        "left_oob_rate": left_oob / domain if domain else None,
        "right_oob_rate": 0.0 if domain else None,
        "valid": domain > 0,
    }


def _method(
    *,
    windows: int,
    refined: bool,
    clamp: bool,
    boundary_regression: bool = False,
) -> dict:
    accuracy_count = 10_000
    pixels = windows * 384 * 768
    if refined:
        values = {
            "boundary_epe_px": 1.1 if boundary_regression else 0.9,
            "bad_1": 0.09,
            "bad_2": 0.025,
            "epe_px": 0.18,
            "low_confidence_epe_px": 0.27,
            "trusted_region_epe_px": 0.101,
            "invalid_region_completeness": 0.55,
            "output_invalid_rate": 0.003,
            "output_negative_rate": 0.002,
            "output_nan_rate": 0.0,
            "output_infinite_rate": 0.0,
            "output_zero_rate": 0.001,
        }
    else:
        values = {
            "boundary_epe_px": 1.0,
            "bad_1": 0.1,
            "bad_2": 0.03,
            "epe_px": 0.2,
            "low_confidence_epe_px": 0.3,
            "trusted_region_epe_px": 0.1,
            "invalid_region_completeness": 0.5,
            "output_invalid_rate": 0.004,
            "output_negative_rate": 0.003,
            "output_nan_rate": 0.0,
            "output_infinite_rate": 0.0,
            "output_zero_rate": 0.001,
        }
    if clamp:
        values["output_negative_rate"] = 0.0
        values["output_zero_rate"] += 0.002 if refined else 0.003
    method = {
        name: _metric(
            value,
            pixels if name.startswith("output_") else accuracy_count,
        )
        for name, value in values.items()
    }
    method["output_variant"] = {
        "type": "PHYSICAL_CLAMP_MIN_ZERO" if clamp else "RAW_MODEL_OUTPUT",
        "epsilon_fill": False,
    }
    method["point_to_plane_error_m"] = dict(
        auditor.POINT_TO_PLANE_NOT_AVAILABLE
    )
    return method


def _change(metric_name: str, baseline: dict, candidate: dict) -> dict:
    base = baseline[metric_name]
    refined = candidate[metric_name]
    absolute = refined["value"] - base["value"]
    relative_valid = base["value"] != 0.0
    relative = 100.0 * absolute / base["value"] if relative_valid else None
    return {
        "metric": metric_name,
        "baseline": base,
        "candidate": refined,
        "absolute_change": absolute,
        "relative_change_percent": relative,
        "valid": True,
        "relative_valid": relative_valid,
    }


def _methods_and_comparisons(
    *, windows: int, boundary_regression: bool
) -> tuple[dict, dict]:
    methods = {
        auditor.RAW_BASE: _method(windows=windows, refined=False, clamp=False),
        auditor.CLAMP_BASE: _method(windows=windows, refined=False, clamp=True),
        auditor.RAW_REFINED: _method(
            windows=windows,
            refined=True,
            clamp=False,
            boundary_regression=boundary_regression,
        ),
        auditor.CLAMP_REFINED: _method(
            windows=windows,
            refined=True,
            clamp=True,
            boundary_regression=boundary_regression,
        ),
    }
    raw_changes = {
        name: _change(name, methods[auditor.RAW_BASE], methods[auditor.RAW_REFINED])
        for name in auditor.REQUIRED_METRICS
    }
    clamp_changes = {
        name: _change(
            name,
            methods[auditor.CLAMP_BASE],
            methods[auditor.CLAMP_REFINED],
        )
        for name in auditor.REQUIRED_METRICS
    }
    return methods, {
        "raw_refined_vs_base": {
            "trusted_region_degradation": raw_changes["trusted_region_epe_px"],
            "low_confidence_epe_change": raw_changes["low_confidence_epe_px"],
            "invalid_region_completeness_change": raw_changes[
                "invalid_region_completeness"
            ],
        },
        "clamp0_refined_vs_base": {
            "trusted_region_degradation": clamp_changes[
                "trusted_region_epe_px"
            ],
            "low_confidence_epe_change": clamp_changes[
                "low_confidence_epe_px"
            ],
            "invalid_region_completeness_change": clamp_changes[
                "invalid_region_completeness"
            ],
        },
        "raw_epe_change": raw_changes["epe_px"],
        "raw_all_metric_changes": raw_changes,
        "clamp0_all_metric_changes": clamp_changes,
        "paired_pixel_changes": {
            "paired_epe_improvement_hr_px": _metric(0.02, 10_000),
            "paired_refined_better_rate": _metric(0.6, 10_000),
            "paired_refined_worse_rate": _metric(0.3, 10_000),
            "paired_unchanged_rate": _metric(0.1, 10_000),
            "paired_finite_coverage_rate": _metric(1.0, 10_000),
            "paired_nonfinite_rate": _metric(0.0, 10_000),
        },
    }


def _patch_formal_hashes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_sha: str,
    evaluator_sha: str,
    validation_manifest_sha: str,
    derived_manifest_sha: str,
    derived_receipt_sha: str,
    raw_manifest_sha: str,
    stage_b_checkpoint_sha: str,
    stage_b_metrics_sha: str,
    stage_b_audit_sha: str,
    stage_b_report_sha: str,
    rectification_sha: str,
) -> None:
    values = {
        "FORMAL_STAGE_C_CONFIG_SHA256": config_sha,
        "FORMAL_STAGE_C_EVALUATOR_SHA256": evaluator_sha,
        "FORMAL_VALIDATION_MANIFEST_SHA256": validation_manifest_sha,
        "FORMAL_VALIDATION_DERIVED_MANIFEST_SHA256": derived_manifest_sha,
        "FORMAL_VALIDATION_DERIVED_RECEIPT_SHA256": derived_receipt_sha,
        "FORMAL_VALIDATION_RAW_VGGT_MANIFEST_SHA256": raw_manifest_sha,
        "FORMAL_STAGE_B_CHECKPOINT_SHA256": stage_b_checkpoint_sha,
        "FORMAL_STAGE_B_METRICS_SHA256": stage_b_metrics_sha,
        "FORMAL_STAGE_B_TRAINING_AUDIT_SHA256": stage_b_audit_sha,
        "FORMAL_STAGE_B_REPORT_SHA256": stage_b_report_sha,
        "FORMAL_RECTIFICATION_AUDIT_SHA256": rectification_sha,
    }
    for name, value in values.items():
        monkeypatch.setattr(auditor, name, value)


def _build_artifacts(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool,
    limited: bool = False,
    boundary_regression: bool = False,
) -> dict[str, Path]:
    inputs = root / "inputs"
    evaluation = root / "evaluation"
    inputs.mkdir(parents=True)
    evaluation.mkdir()
    windows = 10 if limited else 238

    validation_manifest = inputs / "val.jsonl"
    validation_manifest.write_text('{"frame": 1}\n', encoding="utf-8")
    training_manifest = inputs / "train.jsonl"
    training_manifest.write_text('{"frame": 2}\n', encoding="utf-8")
    train_manifest_sha = _sha(training_manifest)
    monkeypatch.setattr(auditor, "FORMAL_TRAIN_MANIFEST_SHA256", train_manifest_sha)

    coverage_files = {}
    for name in ("derived_manifest", "derived_receipt", "raw_manifest"):
        path = inputs / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        coverage_files[name] = path

    train_derived_manifest = inputs / "train_derived_manifest.jsonl"
    train_derived_manifest.write_text("train-derived-manifest\n", encoding="utf-8")
    train_derived_receipt = inputs / "train_derived_receipt.json"
    train_derived_receipt.write_text("train-derived-receipt\n", encoding="utf-8")
    monkeypatch.setattr(
        auditor,
        "FORMAL_TRAIN_DERIVED_MANIFEST_SHA256",
        _sha(train_derived_manifest),
    )
    monkeypatch.setattr(
        auditor,
        "FORMAL_TRAIN_DERIVED_RECEIPT_SHA256",
        _sha(train_derived_receipt),
    )

    raw_identity = {"component": "vggt-omega", "checkpoint_sha256": "d" * 64}
    raw_roots: dict[str, Path] = {}
    raw_receipts: dict[str, Path] = {}
    for prefix, manifest_sha, selected in (
        ("training", train_manifest_sha, 2_779),
        ("validation", _sha(validation_manifest), 240),
    ):
        raw_root = inputs / f"{prefix}_raw_vggt"
        raw_root.mkdir()
        raw_manifest = raw_root / "cache_manifest.jsonl"
        raw_manifest.write_text(f"{prefix}-raw-manifest\n", encoding="utf-8")
        raw_receipt = raw_root / "run_receipt.json"
        _write_json(
            raw_receipt,
            {
                "schema_version": 1,
                "manifest_sha256": manifest_sha,
                "selected_windows": selected,
                "identity": raw_identity,
            },
        )
        raw_roots[prefix] = raw_root
        raw_receipts[prefix] = raw_receipt
    monkeypatch.setattr(
        auditor,
        "FORMAL_TRAIN_RAW_VGGT_RECEIPT_SHA256",
        _sha(raw_receipts["training"]),
    )
    monkeypatch.setattr(
        auditor,
        "FORMAL_TRAIN_RAW_VGGT_MANIFEST_SHA256",
        _sha(raw_roots["training"] / "cache_manifest.jsonl"),
    )
    monkeypatch.setattr(
        auditor,
        "FORMAL_VALIDATION_RAW_VGGT_RECEIPT_SHA256",
        _sha(raw_receipts["validation"]),
    )
    monkeypatch.setattr(
        auditor,
        "FORMAL_VALIDATION_RAW_VGGT_MANIFEST_SHA256",
        _sha(raw_roots["validation"] / "cache_manifest.jsonl"),
    )
    coverage_files["raw_manifest"] = (
        raw_roots["validation"] / "cache_manifest.jsonl"
    )

    evaluator = inputs / "eval_epipolar_snapshot.py"
    evaluator.write_text("# formal evaluator snapshot\n", encoding="utf-8")
    rectification_sha = "b" * 64

    stage_b_checkpoint = inputs / "stage_b_final.pt"
    stage_b_checkpoint.write_bytes(b"stage-b-final")
    stage_b_audit_file = inputs / "stage_b_training_audit.json"
    _write_json(stage_b_audit_file, {"status": "PASS"})
    stage_b_methods = {
        "T3_VGGT": _method(windows=238, refined=False, clamp=False),
        "T3_VGGT_clamp0": _method(windows=238, refined=False, clamp=True),
    }
    stage_b_metrics = inputs / "stage_b_metrics.json"
    _write_json(stage_b_metrics, {"methods": stage_b_methods})

    stage_b_report_path = inputs / "stage_b_report.json"
    stage_b_report = {
        "schema_version": 1,
        "component": auditor.STAGE_B_REPORT_COMPONENT,
        "status": "FINAL_ACCURACY_GATES_PASS_OUTPUT_HEALTH_FAIL",
        "claim_boundary": {
            "final_acceptance_eligible": True,
            "final_training_checkpoint": True,
            "formal_holdout": True,
            "coverage_eligible": True,
            "paper_accuracy": False,
            "paper_ground_truth": False,
        },
        "checkpoint": {
            "path": str(stage_b_checkpoint),
            "sha256": _sha(stage_b_checkpoint),
            "step": 15_000,
            "configured_steps": 15_000,
            "training_git_hash": "c" * 40,
        },
        "training_audit": {
            "status": "PASS",
            "training_status": "TRAINING_COMPLETE",
            "completion_receipt_valid": True,
            "latest_checkpoint_step": 15_000,
        },
        "coverage": {
            "manifest_records": 244,
            "derived_endpoint_records": 240,
            "evaluable_t3_windows": 238,
            "evaluated_t3_windows": 238,
            "future_frames": False,
        },
        "artifacts": {
            "metrics_json": str(stage_b_metrics),
            "metrics_json_sha256": _sha(stage_b_metrics),
            "training_audit": str(stage_b_audit_file),
            "training_audit_sha256": _sha(stage_b_audit_file),
        },
    }
    _write_json(stage_b_report_path, stage_b_report)

    stage_c_checkpoint = inputs / ("stage_c_final.pt" if complete else "stage_c_latest.pt")
    stage_c_checkpoint.write_bytes(
        b"stage-c-final" if complete else b"stage-c-intermediate"
    )
    stage_c_latest = inputs / "stage_c_latest.pt"
    if complete:
        stage_c_latest.write_bytes(b"stage-c-latest")
    else:
        stage_c_latest = stage_c_checkpoint
    training_log = inputs / "stage_c_train.jsonl"
    training_log.write_text("{}\n", encoding="utf-8")
    base_identity = {
        "path": str(stage_b_checkpoint),
        "sha256": _sha(stage_b_checkpoint),
        "step": 15_000,
    }
    rectification = {"status": "PASS", "sha256": rectification_sha}
    config = {
        "train": {"stage": "epipolar", "steps_epipolar": 5_000},
        "data": {"manifest_path": str(inputs / "train.jsonl")},
        "model": {"epipolar_refinement": True},
    }
    config_sha = auditor._canonical_config_sha256(config)
    step = 5_000 if complete else 500
    runtime_records = [
        {"path": f"source_{index}.py", "sha256": f"{index:064x}"}
        for index in range(52)
    ]
    runtime_bundle_payload = json.dumps(
        {
            "git_head": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
            "files": runtime_records,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(
        auditor,
        "FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256",
        hashlib.sha256(runtime_bundle_payload).hexdigest(),
    )
    run_summary_path = inputs / "stage_c_run_summary.json"
    if complete:
        run_summary = {
            "schema_version": 1,
            "component": auditor.TRAINING_SUMMARY_COMPONENT,
            "status": "TRAINING_COMPLETE",
            "stage": "epipolar",
            "steps": 5_000,
            "configured_steps": 5_000,
            "formal_training_complete": True,
            "git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
            "config_sha256": config_sha,
            "runtime_source_bundle_sha256": auditor.FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256,
            "base_checkpoint": base_identity,
            "final_checkpoint": {
                "path": str(stage_c_checkpoint),
                "sha256": _sha(stage_c_checkpoint),
            },
            "latest_checkpoint": {
                "path": str(stage_c_latest),
                "sha256": _sha(stage_c_latest),
            },
            "training_log": {
                "path": str(training_log),
                "sha256": _sha(training_log),
            },
        }
        _write_json(run_summary_path, run_summary)

    checkpoint_identity = {
        "path": str(stage_c_checkpoint),
        "sha256": _sha(stage_c_checkpoint),
        "step": step,
        "parameter_count": 69_905,
        "git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
        "config_sha256": config_sha,
    }
    training_audit_path = inputs / "stage_c_training_audit.json"
    training_audit = {
        "schema_version": 1,
        "component": auditor.TRAINING_AUDIT_COMPONENT,
        "read_only": True,
        "output_dir": str(inputs),
        "safe_load": {
            "torch_weights_only": True,
            "arbitrary_pickle_globals_enabled": False,
            "symlink_artifacts_allowed": False,
        },
        "status": "PASS" if complete else "IN_PROGRESS",
        "training_status": "TRAINING_COMPLETE" if complete else "IN_PROGRESS",
        "completion": {
            "receipt_present": complete,
            "receipt_valid": complete,
            "formal_training_complete": complete,
            "summary": {"valid": True} if complete else None,
        },
        "files": {
            "final_checkpoint": checkpoint_identity if complete else None,
            "latest_checkpoint": (
                {
                    **checkpoint_identity,
                    "path": str(stage_c_latest),
                    "sha256": _sha(stage_c_latest),
                }
                if complete
                else checkpoint_identity
            ),
            "training_log": {
                "path": str(training_log),
                "sha256": _sha(training_log),
            },
            "run_summary": (
                {
                    "path": str(run_summary_path),
                    "sha256": _sha(run_summary_path),
                }
                if complete
                else None
            ),
        },
        "checkpoint_validation": {
            "completion": {
                "actual_step": step,
                "formal_training_complete": complete,
            },
            "base_checkpoint": {
                **base_identity,
                "completion": {"complete": True},
            },
            "runtime_source_bundle": {
                "file_count": 52,
                "git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
                "bundle_sha256": auditor.FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256,
                "all_files_match_checkpoint_git_tree": True,
            },
            "rectification_audit": rectification,
            "training_runtime": {
                "formal_cuda_bf16_eligible": True,
                "native_cuda_bf16": True,
                "strict_determinism": True,
                "device_name": "NVIDIA GeForce RTX 5090",
                "device_capability": [12, 0],
                "cuda_version": "12.8",
                "torch_version": "2.10.0+cu128",
                "device": "cuda:0",
            },
        },
        "log_validation": {
            "records": 5_000 if complete else 500,
            "last_step": 5_000 if complete else 500,
            "latest_checkpoint_lag_steps": 0,
            "steps_continuous": True,
            "learning_rate_schedule_exact": True,
            "finite": True,
        },
    }
    _write_json(training_audit_path, training_audit)

    methods, comparisons = _methods_and_comparisons(
        windows=windows, boundary_regression=boundary_regression
    )
    primary_health = {
        name: methods[auditor.RAW_REFINED][name]
        for name in (
            "output_invalid_rate",
            "output_negative_rate",
            "output_nan_rate",
            "output_infinite_rate",
            "output_zero_rate",
        )
    }
    full_complete = complete
    acceptance = bool(full_complete and not limited)
    status = (
        "EVALUATION_COMPLETE"
        if acceptance
        else ("LIMITED_SMOKE_ONLY" if limited else "INTERMEDIATE_CHECKPOINT_EVALUATION")
    )
    raw_lineage = {
        "raw_vggt_identity": raw_identity,
        "raw_vggt_receipt_sha256": auditor.FORMAL_TRAIN_RAW_VGGT_RECEIPT_SHA256,
        "derived_cache_lineage": {
            "run_receipt_sha256": auditor.FORMAL_TRAIN_DERIVED_RECEIPT_SHA256,
            "cache_manifest_sha256": auditor.FORMAL_TRAIN_DERIVED_MANIFEST_SHA256,
            "run_receipt_path": str(train_derived_receipt),
            "cache_manifest_path": str(train_derived_manifest),
            "selected_records": 2_779,
        },
    }
    base_lineage = {"stage": "temporal"}
    source_files = {
        record["path"]: {
            "current_sha256": record["sha256"],
            "checkpoint_commit_sha256": record["sha256"],
        }
        for record in runtime_records
    }
    recorded_runtime = {
        "device": "cuda:0",
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_capability": [12, 0],
        "torch_version": "2.10.0+cu128",
        "cuda_version": "12.8",
        "cuda_available": True,
        "bf16_supported": True,
        "autocast_enabled": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "strict_determinism_eligible": True,
        "formal_cuda_bf16_eligible": True,
    }
    training_runtime_receipt = {
        "recorded": recorded_runtime,
        "producer_cuda_bf16_eligible": True,
        "strict_determinism_eligible": True,
        "cuda_12_8_or_newer": True,
        "blackwell_capability": True,
        "rtx_5090": True,
        "eligible": True,
    }
    evaluation_runtime = {
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_capability": [12, 0],
        "torch_version": "2.10.0+cu128",
        "cuda_version": "12.8",
        "cuda_bf16_supported": True,
        "autocast_dtype": "torch.bfloat16",
        "cuda_12_8_or_newer": True,
        "blackwell_capability": True,
        "rtx_5090": True,
        "versions_and_device_match_training": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "strict_determinism_eligible": True,
        "eligible": True,
    }
    metrics = {
        "schema_version": 1,
        "stage": auditor.METRICS_STAGE,
        "status": status,
        "claims": {
            "acceptance_eligible": acceptance,
            "paper_ground_truth": False,
            "paper_accuracy": False,
            "pseudo_gt_engineering_only": True,
            "future_frames": False,
            "point_to_plane": "NOT_AVAILABLE",
            "performance_acceptance_claimed": False,
            "primary_claim_method": auditor.RAW_REFINED,
            "primary_claim_variant": "RAW_MODEL_OUTPUT",
            "primary_comparison": "raw_refined_vs_base",
            "clamp0_acceptance_owner": False,
            "primary_raw_output_health": primary_health,
        },
        "target": {"type": auditor.PSEUDO_GT_TARGET},
        "postprocess_contract": {
            "role": "DECLARED_PHYSICAL_POSTPROCESS_DIAGNOSTIC",
            "operation": auditor.CLAMP0_OPERATION,
            "epsilon_fill": False,
            "zero_remains_invalid": True,
            "nan_and_positive_negative_infinity_preserved": True,
            "completeness_is_not_fabricated": True,
            "raw_rows_are_retained": True,
        },
        "methods": methods,
        "comparisons": comparisons,
        "refinement_statistics": {
            "candidate_coverage_rate": {
                "value": (windows * 384 * 768 - 100)
                / (windows * 384 * 768),
                "numerator": float(windows * 384 * 768 - 100),
                "count": windows * 384 * 768,
                "valid": True,
            },
            "correction_signed_hr_px": {
                "count": windows * 384 * 768 - 100,
                "mean": -0.1,
                "minimum": -0.5,
                "maximum": 0.25,
                "valid": True,
            },
            "correction_absolute_hr_px": {
                "count": windows * 384 * 768 - 100,
                "mean": 0.2,
                "minimum": 0.0,
                "maximum": 0.5,
                "valid": True,
            },
            "confidence": {
                "count": windows * 384 * 768 - 100,
                "mean": 0.7,
                "minimum": 0.1,
                "maximum": 1.0,
                "valid": True,
            },
            "correction_nonzero_rate": {
                "value": (windows * 384 * 768 - 200)
                / (windows * 384 * 768 - 100),
                "numerator": float(windows * 384 * 768 - 200),
                "count": windows * 384 * 768 - 100,
                "valid": True,
            },
            "correction_saturated_rate": {
                "value": 10.0 / (windows * 384 * 768 - 100),
                "numerator": 10.0,
                "count": windows * 384 * 768 - 100,
                "valid": True,
            },
        },
        "runtime_geometry_statistics": {
            "contract": {
                "version": "audited_same_row_rectified_pixels_v1",
                "runtime_right_row_scale": 1.0,
                "runtime_right_row_offset_hr_px": 0.0,
                "vertical_correspondence": "v_right=v_left",
                "horizontal_correspondence": (
                    "u_right=u_left-disparity-delta"
                ),
            },
            "right_intrinsics_source": "manifest.K_right",
            "right_row_scale": {
                "count": windows,
                "mean": 1.0,
                "minimum": 1.0,
                "maximum": 1.0,
                "valid": True,
            },
            "right_row_offset_hr_px": {
                "count": windows,
                "mean": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "valid": True,
            },
            "metadata_runtime_mismatch_is_expected": True,
            "horizontal_correspondence_health": {
                "role": "DIAGNOSTIC_ONLY",
                "changes_training_mask": False,
                "changes_accuracy_metrics": False,
                "methods": {
                    method_name: {
                        "all_pixels": _horizontal_record(
                            windows * 384 * 768, left_oob=100
                        ),
                        "candidate_any_valid": _horizontal_record(
                            windows * 384 * 768 - 100, left_oob=50
                        ),
                        "teacher_trusted": _horizontal_record(
                            10_000, left_oob=10
                        ),
                        "candidate_boundary_band": _horizontal_record(
                            200, left_oob=50
                        ),
                    }
                    for method_name in (auditor.RAW_BASE, auditor.RAW_REFINED)
                },
            },
        },
        "windows_evaluated": windows,
        "full_evaluable_windows": 238,
        "fixed_hr_crop": [384, 768],
        "visualizations_written": 4,
        "elapsed_seconds": 1.0,
        "device": "cuda:0",
        "formal_coverage": {
            "manifest_records": 244,
            "derived_endpoint_records": 240,
            "evaluable_t3_windows": 238,
            "derived_cache_manifest_path": str(coverage_files["derived_manifest"]),
            "derived_cache_manifest_sha256": _sha(coverage_files["derived_manifest"]),
            "derived_run_receipt_path": str(coverage_files["derived_receipt"]),
            "derived_run_receipt_sha256": _sha(coverage_files["derived_receipt"]),
            "raw_vggt_cache_manifest_path": str(coverage_files["raw_manifest"]),
            "raw_vggt_cache_manifest_sha256": _sha(coverage_files["raw_manifest"]),
        },
        "canonical_coverage": {
            "manifest_records": 244,
            "derived_endpoint_records": 240,
            "evaluable_t3_windows": 238,
        },
        "crop_contract": {
            "trained_hr_crop": [384, 768],
            "evaluation_hr_crop": [384, 768],
            "canonical_hr_crop": [384, 768],
            "exact_training_crop": True,
            "canonical_crop": True,
            "training_crop_mode": "random",
            "evaluation_crop_mode": "fixed",
            "canonical_modes": True,
            "eligible": True,
        },
        "execution_contract": {
            "saved_precision": "bf16",
            "evaluation_precision": "bf16",
            "saved_optimizer": "adamw",
            "canonical_training_values": True,
            "canonical_batch_schedule": True,
            "recorded_training_eligible": True,
            "recorded_training_runtime": training_runtime_receipt,
            "evaluation_runtime": evaluation_runtime,
            "eligible": True,
        },
        "checkpoint_completion": {
            "stage_c": {
                "actual_step": step,
                "configured_steps": 5_000,
                "execution_complete": complete,
                "canonical_schedule": True,
                "complete": complete,
            },
            "stage_b_base": {
                "actual_step": 15_000,
                "configured_steps": 15_000,
                "execution_complete": True,
                "canonical_schedule": True,
                "complete": True,
            },
            "all_complete": complete,
        },
        "stage_c_checkpoint": {
            "path": str(stage_c_checkpoint),
            "checkpoint_sha256": _sha(stage_c_checkpoint),
            "step": step,
            "git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
            "parameter_count": 69_905,
            "config": config,
            "base_checkpoint": base_identity,
            "base_lineage": base_lineage,
            "raw_lineage": raw_lineage,
            "runtime_source_bundle": {
                "git_head": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
                "bundle_sha256": auditor.FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256,
                "files": runtime_records,
            },
            "rectification_audit": rectification,
            "training_runtime_receipt": training_runtime_receipt,
        },
        "stage_b_base_checkpoint": base_identity,
        "lineage": {
            "recomputed_stage_c_training": {
                "manifest_path": str(training_manifest),
                "manifest_sha256": train_manifest_sha,
                "derived_endpoint_records": 2_779,
                "evaluable_t3_windows": 2_775,
                "base_lineage": base_lineage,
                "raw_lineage": raw_lineage,
                "audited_endpoint_right_source_digest": {
                    "records": 2_775,
                    "sha256": auditor.FORMAL_TRAIN_RIGHT_SOURCE_DIGEST_SHA256,
                },
            },
            "held_out_validation": {
                "formal_holdout": True,
                "non_holdout_smoke_override": False,
                "same_manifest": False,
                "sequence_overlap": [],
                "training_manifest_sha256": train_manifest_sha,
                "training_manifest_path": str(training_manifest),
                "evaluation_manifest_sha256": _sha(validation_manifest),
                "evaluation_raw_vggt": {
                    "receipt_sha256": (
                        auditor.FORMAL_VALIDATION_RAW_VGGT_RECEIPT_SHA256
                    ),
                    "manifest_sha256": _sha(validation_manifest),
                    "receipt_path": str(raw_receipts["validation"]),
                    "root": str(raw_roots["validation"]),
                    "identity": raw_identity,
                },
                "training_raw_vggt": {
                    "receipt_sha256": (
                        auditor.FORMAL_TRAIN_RAW_VGGT_RECEIPT_SHA256
                    ),
                    "manifest_sha256": train_manifest_sha,
                    "receipt_path": str(raw_receipts["training"]),
                    "root": str(raw_roots["training"]),
                    "identity": raw_identity,
                },
            },
            "stage_c_and_base": {
                "sequence_overlap": [],
                "base_checkpoint_sha256": _sha(stage_b_checkpoint),
                "base_checkpoint_step": 15_000,
                "stage_c_training_manifest_sha256": train_manifest_sha,
            },
            "validation_endpoint_right_sources": {
                "records": windows,
                "sha256": (
                    auditor.FORMAL_VALIDATION_RIGHT_SOURCE_DIGEST_SHA256
                    if not limited
                    else "e" * 64
                ),
                "all_source_sha256_match": True,
            },
            "validation_raw_payload_audit": {
                "derived_records": 240,
                "vggt_payloads_hashed": 240,
                "ffs_payloads_hashed": 240,
                "canonical_reference_digest_sha256": (
                    auditor.FORMAL_VALIDATION_RAW_PAYLOAD_DIGEST_SHA256
                ),
                "all_payload_sha256_match": True,
            },
            "rectification_audit": rectification,
        },
        "source_hashes": {
            "evaluator_path": str(evaluator),
            "evaluator_sha256": _sha(evaluator),
            "repository_git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
            "validation_manifest_sha256": _sha(validation_manifest),
            "stage_c_checkpoint_sha256": _sha(stage_c_checkpoint),
            "stage_b_checkpoint_sha256": _sha(stage_b_checkpoint),
            "runtime_source_bundle": {
                "checkpoint_bundle_sha256": auditor.FORMAL_STAGE_C_RUNTIME_BUNDLE_SHA256,
                "checkpoint_git_hash": auditor.FORMAL_STAGE_C_TRAINING_GIT_HASH,
                "all_byte_identical": True,
                "files": source_files,
            },
        },
        "resolved_config": {"data": {"manifest_path": str(validation_manifest)}},
    }
    _write_json(evaluation / "metrics.json", metrics)
    _write_metrics_csv(evaluation / "metrics.csv", methods)

    _patch_formal_hashes(
        monkeypatch,
        config_sha=config_sha,
        evaluator_sha=_sha(evaluator),
        validation_manifest_sha=_sha(validation_manifest),
        derived_manifest_sha=_sha(coverage_files["derived_manifest"]),
        derived_receipt_sha=_sha(coverage_files["derived_receipt"]),
        raw_manifest_sha=_sha(coverage_files["raw_manifest"]),
        stage_b_checkpoint_sha=_sha(stage_b_checkpoint),
        stage_b_metrics_sha=_sha(stage_b_metrics),
        stage_b_audit_sha=_sha(stage_b_audit_file),
        stage_b_report_sha=_sha(stage_b_report_path),
        rectification_sha=rectification_sha,
    )
    return {
        "evaluation": evaluation,
        "training_audit": training_audit_path,
        "training_summary": run_summary_path,
        "stage_b_report": stage_b_report_path,
    }


def _audit(paths: dict[str, Path], *, complete: bool) -> dict:
    return auditor.audit_epipolar_evaluation(
        paths["evaluation"],
        stage_c_training_audit_path=paths["training_audit"],
        stage_c_training_summary_path=(
            paths["training_summary"] if complete else None
        ),
        stage_b_final_report_path=paths["stage_b_report"],
    )


def test_formal_raw_improvements_pass_engineering_gate_without_paper_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)

    report = _audit(paths, complete=True)

    assert report["status"] == "STAGE_C_M5_GATE_PASS"
    assert report["final_gate"] == {
        "eligible": True,
        "result": "PASS",
        "all_required_gates_pass": True,
        "all_reported_checks_pass": True,
        "limited_or_intermediate_cannot_pass": True,
    }
    assert report["claims"]["paper_claim_eligible"] is False
    assert report["claims"]["clamp0_acceptance_owner"] is False
    assert report["claims"]["refined_temporal_metric_available"] is False
    assert report["gates"]["clamp0_used_for_any_gate"] is False
    assert report["lineage"]["runtime_source_files"] == 52


def test_formal_boundary_regression_is_valid_artifact_but_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(
        tmp_path,
        monkeypatch,
        complete=True,
        boundary_regression=True,
    )

    report = _audit(paths, complete=True)

    assert report["status"] == "STAGE_C_M5_GATE_FAIL"
    assert report["final_gate"]["eligible"] is True
    assert report["final_gate"]["result"] == "FAIL"
    assert report["gates"]["gates"]["raw_boundary_epe_px_improves"][
        "passed"
    ] is False


def test_epe_regression_is_reported_but_does_not_override_m5_primary_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    refined = metrics["methods"][auditor.RAW_REFINED]
    refined["epe_px"] = _metric(0.21, refined["epe_px"]["count"])
    recomputed = _change(
        "epe_px", metrics["methods"][auditor.RAW_BASE], refined
    )
    metrics["comparisons"]["raw_epe_change"] = recomputed
    metrics["comparisons"]["raw_all_metric_changes"]["epe_px"] = recomputed
    metrics["comparisons"]["paired_pixel_changes"][
        "paired_epe_improvement_hr_px"
    ] = _metric(-0.01, refined["epe_px"]["count"])
    _write_json(metrics_path, metrics)
    _write_metrics_csv(paths["evaluation"] / "metrics.csv", metrics["methods"])

    report = _audit(paths, complete=True)

    assert report["status"] == "STAGE_C_M5_GATE_PASS"
    assert report["final_gate"]["all_required_gates_pass"] is True
    assert report["final_gate"]["all_reported_checks_pass"] is False
    epe = report["gates"]["gates"]["raw_epe_px_improves"]
    assert epe["required_for_final_gate"] is False
    assert epe["passed"] is False


def test_in_progress_full_evaluation_is_never_final_gate_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(
        tmp_path,
        monkeypatch,
        complete=False,
    )

    report = _audit(paths, complete=False)

    assert report["status"] == "INELIGIBLE_FOR_FINAL_GATE"
    assert report["final_gate"]["eligible"] is False
    assert report["final_gate"]["result"] == "INELIGIBLE"
    assert report["training"]["in_progress"] is True
    assert report["coverage"]["windows_evaluated"] == 238


def test_limited_final_checkpoint_evaluation_is_never_final_gate_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(
        tmp_path,
        monkeypatch,
        complete=True,
        limited=True,
    )

    report = _audit(paths, complete=True)

    assert report["status"] == "INELIGIBLE_FOR_FINAL_GATE"
    assert report["final_gate"]["eligible"] is False
    assert report["final_gate"]["result"] == "INELIGIBLE"
    assert report["training"]["formally_complete"] is True
    assert report["coverage"]["windows_evaluated"] == 10


def test_limited_empty_boundary_domain_is_valid_report_but_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(
        tmp_path,
        monkeypatch,
        complete=False,
        limited=True,
    )
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    invalid_metric = {"value": None, "numerator": None, "count": 0, "valid": False}
    for name in auditor.METHOD_NAMES:
        metrics["methods"][name]["boundary_epe_px"] = invalid_metric
    def invalid_change(base_name: str, candidate_name: str) -> dict:
        return {
            "metric": "boundary_epe_px",
            "baseline": metrics["methods"][base_name]["boundary_epe_px"],
            "candidate": metrics["methods"][candidate_name]["boundary_epe_px"],
            "absolute_change": None,
            "relative_change_percent": None,
            "valid": False,
            "relative_valid": False,
        }

    raw_change = invalid_change(auditor.RAW_BASE, auditor.RAW_REFINED)
    clamp_change = invalid_change(auditor.CLAMP_BASE, auditor.CLAMP_REFINED)
    metrics["comparisons"]["raw_all_metric_changes"][
        "boundary_epe_px"
    ] = raw_change
    metrics["comparisons"]["clamp0_all_metric_changes"][
        "boundary_epe_px"
    ] = clamp_change
    _write_json(metrics_path, metrics)
    _write_metrics_csv(paths["evaluation"] / "metrics.csv", metrics["methods"])

    report = _audit(paths, complete=False)

    assert report["status"] == "INELIGIBLE_FOR_FINAL_GATE"
    boundary = report["gates"]["gates"]["raw_boundary_epe_px_improves"]
    assert boundary["valid"] is False
    assert boundary["passed"] is False


def test_tampered_stage_c_checkpoint_sha_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["source_hashes"]["stage_c_checkpoint_sha256"] = "0" * 64
    _write_json(metrics_path, metrics)

    with pytest.raises(
        auditor.EpipolarEvaluationAuditError,
        match="Stage-C checkpoint differs from training audit",
    ):
        _audit(paths, complete=True)


def test_clamp0_cannot_be_promoted_to_acceptance_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["claims"]["clamp0_acceptance_owner"] = True
    metrics["claims"]["primary_claim_variant"] = "PHYSICAL_CLAMP_MIN_ZERO"
    _write_json(metrics_path, metrics)

    with pytest.raises(
        auditor.EpipolarEvaluationAuditError,
        match="primary claim is not raw refined-vs-base",
    ):
        _audit(paths, complete=True)


@pytest.mark.parametrize("tamper", ["clamp", "source_git", "runtime_device"])
def test_clamp_source_and_runtime_tampering_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if tamper == "clamp":
        clamp = metrics["methods"][auditor.CLAMP_REFINED]
        clamp["output_negative_rate"] = _metric(
            0.001, clamp["output_negative_rate"]["count"]
        )
    elif tamper == "source_git":
        metrics["source_hashes"]["repository_git_hash"] = "0" * 40
    else:
        metrics["execution_contract"]["evaluation_runtime"][
            "device_name"
        ] = "Different GPU"
    _write_json(metrics_path, metrics)

    with pytest.raises(auditor.EpipolarEvaluationAuditError):
        _audit(paths, complete=True)


@pytest.mark.parametrize(
    "tamper",
    [
        "refinement",
        "paired",
        "paired_improvement",
        "horizontal",
        "lineage_bytes",
        "metrics_csv",
    ],
)
def test_archival_statistics_lineage_and_csv_tampering_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics_path = paths["evaluation"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if tamper == "refinement":
        metrics["refinement_statistics"]["confidence"]["count"] += 1
        _write_json(metrics_path, metrics)
    elif tamper == "paired":
        record = metrics["comparisons"]["paired_pixel_changes"][
            "paired_unchanged_rate"
        ]
        record["numerator"] += 1.0
        record["value"] = record["numerator"] / record["count"]
        _write_json(metrics_path, metrics)
    elif tamper == "paired_improvement":
        record = metrics["comparisons"]["paired_pixel_changes"][
            "paired_epe_improvement_hr_px"
        ]
        record["numerator"] += 1.0
        record["value"] = record["numerator"] / record["count"]
        _write_json(metrics_path, metrics)
    elif tamper == "horizontal":
        metrics["runtime_geometry_statistics"][
            "horizontal_correspondence_health"
        ]["methods"][auditor.RAW_REFINED]["all_pixels"]["oob_count"] += 1
        _write_json(metrics_path, metrics)
    elif tamper == "lineage_bytes":
        receipt = Path(
            metrics["lineage"]["held_out_validation"]["training_raw_vggt"][
                "receipt_path"
            ]
        )
        receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        csv_path = paths["evaluation"] / "metrics.csv"
        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
        rows[0]["epe_px"] = "999.0"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(auditor.EpipolarEvaluationAuditError):
        _audit(paths, complete=True)


def test_refinement_audit_accepts_real_mixed_paired_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics = json.loads(
        (paths["evaluation"] / "metrics.json").read_text(encoding="utf-8")
    )
    candidate_count = int(
        metrics["refinement_statistics"]["candidate_coverage_rate"]["numerator"]
    )
    metrics["refinement_statistics"]["correction_signed_hr_px"][
        "count"
    ] = candidate_count - 1
    metrics["refinement_statistics"]["correction_absolute_hr_px"][
        "count"
    ] = candidate_count - 1
    paired_count = 10_000
    mixed_paired = {
        "paired_epe_improvement_hr_px": _invalid_metric(paired_count),
        "paired_refined_better_rate": _invalid_metric(paired_count),
        "paired_refined_worse_rate": _invalid_metric(paired_count),
        "paired_unchanged_rate": _invalid_metric(paired_count),
        "paired_finite_coverage_rate": {
            "value": 9_999 / paired_count,
            "numerator": 9_999.0,
            "count": paired_count,
            "valid": True,
        },
        "paired_nonfinite_rate": {
            "value": 1 / paired_count,
            "numerator": 1.0,
            "count": paired_count,
            "valid": True,
        },
    }
    parsed_methods = {
        "methods": {
            auditor.RAW_REFINED: {"epe_px": _invalid_metric(paired_count)},
        },
        "paired_pixel_changes": mixed_paired,
    }

    receipt = auditor._validate_refinement_statistics(
        metrics,
        parsed_methods=parsed_methods,
        windows_evaluated=238,
    )

    assert receipt["paired"]["status"] == "AUDITED"
    assert receipt["paired"]["finite_count"] == 9_999
    assert receipt["paired"]["nonfinite_count"] == 1
    assert receipt["paired"]["outcome_partition"]["status"] == "NOT_AUDITABLE"
    assert receipt["paired"]["mean_improvement"]["status"] == "NOT_AUDITABLE"
    assert receipt["correction_domain"]["nonfinite_correction_count"] == 1

    mixed_paired["paired_finite_coverage_rate"] = {
        "value": 9_998 / paired_count,
        "numerator": 9_998.0,
        "count": paired_count,
        "valid": True,
    }
    with pytest.raises(
        auditor.EpipolarEvaluationAuditError,
        match="finite/nonfinite counts do not partition",
    ):
        auditor._validate_refinement_statistics(
            metrics,
            parsed_methods=parsed_methods,
            windows_evaluated=238,
        )


def test_refinement_audit_accepts_empty_candidate_domain_as_not_auditable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    metrics = json.loads(
        (paths["evaluation"] / "metrics.json").read_text(encoding="utf-8")
    )
    statistics = metrics["refinement_statistics"]
    expected_pixels = 238 * 384 * 768
    statistics["candidate_coverage_rate"] = _metric(0.0, expected_pixels)
    empty_statistic = {
        "count": 0,
        "mean": None,
        "minimum": None,
        "maximum": None,
        "valid": False,
    }
    for name in (
        "correction_signed_hr_px",
        "correction_absolute_hr_px",
        "confidence",
    ):
        statistics[name] = dict(empty_statistic)
    statistics["correction_nonzero_rate"] = _invalid_metric()
    statistics["correction_saturated_rate"] = _invalid_metric()
    parsed_methods = {
        "methods": metrics["methods"],
        "paired_pixel_changes": metrics["comparisons"]["paired_pixel_changes"],
    }

    receipt = auditor._validate_refinement_statistics(
        metrics,
        parsed_methods=parsed_methods,
        windows_evaluated=238,
    )

    assert receipt["candidate_valid_pixels"] == 0
    assert receipt["correction_domain"] == {
        "status": "NOT_AUDITABLE",
        "reason": "empty candidate-valid domain",
    }


def test_cli_refuses_to_write_report_inside_evaluation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    output = paths["evaluation"] / "final_audit.json"

    exit_code = auditor.main(
        [
            "--evaluation-dir",
            str(paths["evaluation"]),
            "--stage-c-training-audit",
            str(paths["training_audit"]),
            "--stage-c-training-summary",
            str(paths["training_summary"]),
            "--stage-b-final-report",
            str(paths["stage_b_report"]),
            "--json-out",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()


def test_cli_cannot_overwrite_input_or_write_inside_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_artifacts(tmp_path, monkeypatch, complete=True)
    common = [
        "--evaluation-dir",
        str(paths["evaluation"]),
        "--stage-c-training-audit",
        str(paths["training_audit"]),
        "--stage-c-training-summary",
        str(paths["training_summary"]),
        "--stage-b-final-report",
        str(paths["stage_b_report"]),
    ]
    summary_sha = _sha(paths["training_summary"])

    overwrite_exit = auditor.main(
        [*common, "--json-out", str(paths["training_summary"])]
    )
    training_subtree_output = paths["training_audit"].parent / "new_audit.json"
    subtree_exit = auditor.main(
        [*common, "--json-out", str(training_subtree_output)]
    )

    assert overwrite_exit == 2
    assert subtree_exit == 2
    assert _sha(paths["training_summary"]) == summary_sha
    assert not training_subtree_output.exists()
