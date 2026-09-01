from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import tools.audit_d025_evaluation as auditor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(value: float, *, count: int = 100) -> dict:
    return {"value": value, "numerator": value * count, "count": count, "valid": True}


def _identities() -> tuple[dict, dict, dict]:
    observation = {"component": "ffs-observation", "checkpoint_sha256": "1" * 64}
    teacher = {"component": "ffs-teacher", "checkpoint_sha256": "2" * 64}
    derived = {"component": "vggt-ffs-derived-geometry-batch", "config": {"mode": "strict"}}
    return observation, teacher, derived


def _methods(*, improved_health: bool) -> dict:
    values = {
        "output_negative_rate": 0.004 if improved_health else 0.02,
        "output_invalid_rate": 0.004 if improved_health else 0.02,
        "output_nan_rate": 0.0,
        "low_confidence_epe_px": 0.18,
        "invalid_region_completeness": 0.60,
        "trusted_region_epe_px": 0.10,
        "epe_px": 0.15,
        "bad_1": 0.01,
        "bad_2": 0.004,
        "boundary_epe_px": 0.90,
        "temporal_disparity_error_native_px": 0.25,
    }
    return {
        auditor.RAW_OWNER: {
            **{name: _metric(value) for name, value in values.items()},
            "output_variant": {"type": "RAW_MODEL_OUTPUT", "source_method": auditor.RAW_OWNER},
        }
    }


def _write_csv(path: Path, methods: dict) -> None:
    row = {"method": auditor.RAW_OWNER}
    for name in auditor.RAW_METRICS:
        metric = methods[auditor.RAW_OWNER][name]
        row[name] = str(metric["value"])
        row[f"{name}_valid"] = "True"
        row[f"{name}_count"] = str(metric["count"])
        row[f"{name}_numerator"] = str(metric["numerator"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _comparison(relative: float) -> dict:
    return {"relative_change_percent": relative, "valid": True}


def _metrics(
    *,
    checkpoint_sha256: str,
    git_hash: str,
    config: dict,
    stage_a_sha256: str,
    methods: dict,
    claims_final: bool,
) -> dict:
    observation, teacher, derived = _identities()
    return {
        "stage": "T3_CAUSAL_STAGE_B",
        "target": {"type": auditor.PSEUDO_GT_TARGET},
        "claims": {
            "paper_accuracy": False,
            "paper_gt": False,
            "final_training_checkpoint": claims_final,
            "final_acceptance_eligible": claims_final,
            "full_validation_selection": claims_final,
            "formal_holdout": claims_final,
        },
        "records_evaluated": 238,
        "formal_temporal_coverage": {
            "manifest_records": 244,
            "derived_endpoint_records": 240,
            "evaluable_t3_windows": 238,
        },
        "methods": methods,
        "checkpoint": {
            "checkpoint_sha256": checkpoint_sha256,
            "git_hash": git_hash,
            "step": 15000,
            "training_config": config,
        },
        "spatial_checkpoint": {"checkpoint_sha256": stage_a_sha256},
        "cache_identities": {"observation": observation, "teacher": teacher},
        "evaluator": {"git_hash": "7" * 40, "eval_py_sha256": "e" * 64, "evaluation_module_sha256": "f" * 64},
        "comparisons": {
            "T3_VGGT_vs_bilinear": {
                "low_confidence_epe_change": _comparison(-12.0),
                "invalid_region_completeness_change": _comparison(20.0),
                "trusted_region_degradation": _comparison(1.0),
            },
            "T3_vs_T1_temporal": _comparison(-12.0),
        },
        "derived_cache_lineage": derived,
        "holdout_and_raw_lineage": {"training_manifest_sha256": "d" * 64, "evaluation_manifest_sha256": "6" * 64},
    }


def _paths(tmp_path: Path, *, claims_final: bool = True) -> dict[str, Path]:
    stage_a_sha = "a" * 64
    checkpoint_sha = "b" * 64
    git_hash = "c" * 40
    observation, teacher, derived = _identities()
    config = {
        "data": {
            "sequence_length": 3,
            "manifest_path": str((tmp_path / "train.jsonl").resolve()),
            "observation_cache_identity": observation,
            "teacher_cache_identity": teacher,
            "derived_cache_lineage": derived,
        },
        "train": {"steps": 15000, "initialization_checkpoint_sha256": stage_a_sha},
        "positivity_ablation": {"enabled": True},
    }
    training = {
        "component": "training-run-audit",
        "status": "PASS",
        "training_status": "TRAINING_COMPLETE",
        "stage": "temporal",
        "configured_steps": 15000,
        "logged_steps": 15000,
        "latest_checkpoint_step": 15000,
        "checkpoint_lag_steps": 0,
        "git_hash": git_hash,
        "config_fingerprint": auditor._canonical_sha256(config),
        "validation": {
            "strict_json": True,
            "continuous_steps_from_one": True,
            "all_numeric_values_finite": True,
            "checkpoint_state_finite": True,
            "learning_rate_schedule_exact": True,
            "completion_receipt_valid": True,
            "loss_schema": {"positivity_ablation_enabled": True, "terms": [
                "disparity", "epipolar", "gate_regularizer", "gradient", "measurement",
                "positivity_penalty", "temporal", "total", "uncertainty_nll",
            ]},
        },
        "statistics": {"loss": {"positivity_penalty": {"minimum": 0.0}}},
        "files": {"final_checkpoint": {"sha256": checkpoint_sha}},
    }
    preflight = {
        "status": "PREFLIGHT_PASS",
        "protocol": {"name": "full_stage_b_rerun_from_final_stage_a", "required_updates": 15000},
        "stage_a_final": {"steps": 5000, "completed_step": 5000, "checkpoint_sha256": stage_a_sha},
        "formal_train_inputs": {
            "manifest_path": config["data"]["manifest_path"],
            "manifest_sha256": "d" * 64,
            "observation_identity": observation,
            "teacher_identity": teacher,
            "derived_cache_lineage": derived,
        },
    }
    d025_dir, canonical_dir = tmp_path / "d025_eval", tmp_path / "canonical_eval"
    d025_dir.mkdir(); canonical_dir.mkdir()
    d025_methods, canonical_methods = _methods(improved_health=True), _methods(improved_health=False)
    _write_json(d025_dir / "metrics.json", _metrics(checkpoint_sha256=checkpoint_sha, git_hash=git_hash, config=config, stage_a_sha256=stage_a_sha, methods=d025_methods, claims_final=claims_final))
    _write_csv(d025_dir / "metrics.csv", d025_methods)
    canonical_config = dict(config); canonical_config["positivity_ablation"] = {"enabled": False}
    canonical_metrics = _metrics(checkpoint_sha256="9" * 64, git_hash="8" * 40, config=canonical_config, stage_a_sha256=stage_a_sha, methods=canonical_methods, claims_final=True)
    _write_json(canonical_dir / "metrics.json", canonical_metrics)
    _write_csv(canonical_dir / "metrics.csv", canonical_methods)
    report = {
        "component": "stage-b-final-holdout-evaluation-with-sign-health",
        "checkpoint": {"step": 15000, "configured_steps": 15000, "sha256": "9" * 64},
        "evaluator": {"eval_py_sha256": "e" * 64, "evaluation_module_sha256": "f" * 64},
        "artifacts": {
            "metrics_json_sha256": _sha(canonical_dir / "metrics.json"),
            "metrics_csv_sha256": _sha(canonical_dir / "metrics.csv"),
        },
    }
    paths = {"training": tmp_path / "training.json", "preflight": tmp_path / "preflight.json", "canonical_report": tmp_path / "canonical_report.json", "d025": d025_dir, "canonical": canonical_dir}
    _write_json(paths["training"], training); _write_json(paths["preflight"], preflight); _write_json(paths["canonical_report"], report)
    return paths


def test_d025_final_controlled_comparison_passes_raw_owner_gates(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    report = auditor.audit_d025_evaluation(paths["training"], paths["d025"], paths["canonical_report"], paths["canonical"], paths["preflight"])
    assert report["status"] == "D025_FINAL_CONTROLLED_COMPARISON_PASS"
    assert report["final_gate"] == {"eligible": True, "result": "PASS", "limited_or_intermediate_cannot_pass": True}
    assert report["claims"]["clamp0_owner"] is False
    assert report["gates"]["gates"]["raw_output_negative_rate_below_0_5_percent"]["pass"]


def test_d025_intermediate_is_never_a_final_pass(tmp_path: Path) -> None:
    paths = _paths(tmp_path, claims_final=False)
    report = auditor.audit_d025_evaluation(paths["training"], paths["d025"], paths["canonical_report"], paths["canonical"], paths["preflight"])
    assert report["status"] == "INELIGIBLE_FOR_FINAL_GATE"
    assert report["final_gate"]["result"] == "INELIGIBLE"


def test_d025_metrics_csv_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    csv_path = paths["d025"] / "metrics.csv"
    rows = list(csv.DictReader(csv_path.open()))
    rows[0]["epe_px"] = "999"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(auditor.D025EvaluationAuditError, match="metrics.csv value differs"):
        auditor.audit_d025_evaluation(paths["training"], paths["d025"], paths["canonical_report"], paths["canonical"], paths["preflight"])


def test_d025_checkpoint_binding_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    metrics_path = paths["d025"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["checkpoint"]["checkpoint_sha256"] = "0" * 64
    _write_json(metrics_path, metrics)
    with pytest.raises(auditor.D025EvaluationAuditError, match="checkpoint SHA mismatch"):
        auditor.audit_d025_evaluation(paths["training"], paths["d025"], paths["canonical_report"], paths["canonical"], paths["preflight"])
