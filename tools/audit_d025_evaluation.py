#!/usr/bin/env python3
"""Read-only final-gate audit for the controlled D-025 positivity rerun.

The raw ``T3_VGGT`` row is the only owner of D-025 output-health and accuracy
gates.  ``*_clamp0`` is an explicitly non-owning safety diagnostic, while the
trusted FFS teacher remains pseudo-GT engineering evidence rather than paper
ground truth.  No model, GPU, cache producer, or training output is touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIT_SCHEMA_VERSION = 1
AUDIT_COMPONENT = "d025-positivity-final-evaluation-audit"
PSEUDO_GT_TARGET = "trusted_hr_ffs_teacher_pseudo_gt"
RAW_OWNER = "T3_VGGT"
FORMAL_STEPS = 15_000
FORMAL_STAGE_A_STEPS = 5_000
FORMAL_MANIFEST_RECORDS = 244
FORMAL_DERIVED_RECORDS = 240
FORMAL_WINDOWS = 238
RAW_HEALTH_LIMIT = 0.005
MAX_ERROR_REGRESSION_PERCENT = 2.0
MAX_COMPLETENESS_DROP_PERCENT = 2.0
LOW_CONFIDENCE_IMPROVEMENT_PERCENT = -10.0
COMPLETENESS_IMPROVEMENT_PERCENT = 15.0
TRUSTED_DEGRADATION_PERCENT = 2.0
TEMPORAL_IMPROVEMENT_PERCENT = -10.0

RAW_METRICS = (
    "output_negative_rate",
    "output_invalid_rate",
    "output_nan_rate",
    "low_confidence_epe_px",
    "invalid_region_completeness",
    "trusted_region_epe_px",
    "epe_px",
    "bad_1",
    "bad_2",
    "boundary_epe_px",
    "temporal_disparity_error_native_px",
)
ERROR_GUARDRAIL_METRICS = (
    "low_confidence_epe_px",
    "trusted_region_epe_px",
    "epe_px",
    "bad_1",
    "bad_2",
    "boundary_epe_px",
    "temporal_disparity_error_native_px",
)
SHA256_LENGTH = 64


class D025EvaluationAuditError(RuntimeError):
    """Raised when an input artifact is malformed, inconsistent, or tampered."""


@dataclass(frozen=True, slots=True)
class JSONArtifact:
    path: Path
    sha256: str
    value: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D025EvaluationAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise D025EvaluationAuditError(f"strict JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise D025EvaluationAuditError(f"strict JSON has non-finite constant {value}")


def _load_json(path: str | Path, label: str) -> JSONArtifact:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise D025EvaluationAuditError(f"cannot parse {label}: {exc}") from exc
    _require(isinstance(value, Mapping), f"{label} is not a JSON object")
    return JSONArtifact(resolved, hashlib.sha256(payload).hexdigest(), value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == SHA256_LENGTH, f"{name} must be SHA-256")
    _require(all(character in "0123456789abcdef" for character in value), f"{name} must be lowercase SHA-256")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise D025EvaluationAuditError(f"config is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _metric(value: Any, name: str) -> dict[str, float | int]:
    metric = _mapping(value, name)
    _require(set(metric) == {"value", "numerator", "count", "valid"}, f"{name} schema differs")
    _require(metric["valid"] is True, f"{name} is invalid")
    count = _int(metric["count"], f"{name}.count")
    _require(count > 0, f"{name}.count must be positive")
    numerator = _finite(metric["numerator"], f"{name}.numerator")
    metric_value = _finite(metric["value"], f"{name}.value")
    _require(math.isclose(metric_value, numerator / count, rel_tol=1e-9, abs_tol=1e-12), f"{name} value disagrees with numerator/count")
    return {"value": metric_value, "numerator": numerator, "count": count}


def _relative_change(candidate: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return 100.0 * (candidate - reference) / reference


def _validate_csv(eval_root: Path, methods: Mapping[str, Any]) -> dict[str, Any]:
    path = eval_root / "metrics.csv"
    _require(path.is_file() and not path.is_symlink(), "metrics.csv is missing or unsafe")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise D025EvaluationAuditError(f"cannot parse metrics.csv: {exc}") from exc
    _require(rows and rows[0].get("method") is not None, "metrics.csv header is malformed")
    by_method = {row.get("method"): row for row in rows}
    _require(set(by_method) == set(methods), "metrics.csv method set differs from metrics.json")
    row = by_method.get(RAW_OWNER)
    _require(row is not None, f"metrics.csv lacks {RAW_OWNER}")
    raw = _mapping(methods[RAW_OWNER], f"methods.{RAW_OWNER}")
    for metric_name in RAW_METRICS:
        metric = _metric(raw.get(metric_name), f"methods.{RAW_OWNER}.{metric_name}")
        _require(row.get(f"{metric_name}_valid") == "True", f"metrics.csv invalid flag differs for {metric_name}")
        _require(int(str(row.get(f"{metric_name}_count"))) == metric["count"], f"metrics.csv count differs for {metric_name}")
        _require(
            math.isclose(float(str(row.get(f"{metric_name}_numerator"))), metric["numerator"], rel_tol=1e-9, abs_tol=1e-12),
            f"metrics.csv numerator differs for {metric_name}",
        )
        _require(
            math.isclose(float(str(row.get(metric_name))), metric["value"], rel_tol=1e-9, abs_tol=1e-12),
            f"metrics.csv value differs for {metric_name}",
        )
    return {"path": str(path.resolve()), "sha256": _sha256(path), "rows": len(rows)}


def _validate_training_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("component") == "training-run-audit", "D025 training audit component differs")
    formal = (
        value.get("status") == "PASS"
        and value.get("training_status") == "TRAINING_COMPLETE"
        and value.get("stage") == "temporal"
        and value.get("configured_steps") == FORMAL_STEPS
        and value.get("logged_steps") == FORMAL_STEPS
        and value.get("latest_checkpoint_step") == FORMAL_STEPS
        and value.get("checkpoint_lag_steps") == 0
    )
    validation = _mapping(value.get("validation"), "D025 training audit validation")
    for key in (
        "strict_json", "continuous_steps_from_one", "all_numeric_values_finite",
        "checkpoint_state_finite", "learning_rate_schedule_exact", "completion_receipt_valid",
    ):
        formal = formal and validation.get(key) is True
    schema = _mapping(validation.get("loss_schema"), "D025 training audit loss schema")
    formal = formal and schema.get("positivity_ablation_enabled") is True
    terms = schema.get("terms")
    _require(isinstance(terms, list), "D025 loss schema terms must be a list")
    _require(set(terms) == {
        "disparity", "epipolar", "gate_regularizer", "gradient", "measurement",
        "positivity_penalty", "temporal", "total", "uncertainty_nll",
    }, "D025 training audit loss schema is not the strict nine-term contract")
    loss = _mapping(_mapping(value.get("statistics"), "D025 training audit statistics").get("loss"), "D025 training audit loss statistics")
    penalty = _mapping(loss.get("positivity_penalty"), "D025 positivity penalty statistics")
    _require(_finite(penalty.get("minimum"), "D025 positivity penalty minimum") >= 0.0, "D025 positivity penalty is negative")
    files = _mapping(value.get("files"), "D025 training audit files")
    final = _mapping(files.get("final_checkpoint"), "D025 final checkpoint")
    return {
        "formal": formal,
        "checkpoint_sha256": _sha(final.get("sha256"), "D025 final checkpoint SHA-256"),
        "git_hash": str(value.get("git_hash")),
        "config_fingerprint": _sha(value.get("config_fingerprint"), "D025 config fingerprint"),
    }


def _validate_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("status") == "PREFLIGHT_PASS", "D025 frozen preflight did not pass")
    protocol = _mapping(value.get("protocol"), "D025 preflight protocol")
    _require(protocol.get("name") == "full_stage_b_rerun_from_final_stage_a", "D025 preflight is not full Stage-B rerun protocol")
    _require(protocol.get("required_updates") == FORMAL_STEPS, "D025 preflight update count differs")
    stage_a = _mapping(value.get("stage_a_final"), "D025 preflight final Stage-A")
    _require(stage_a.get("steps") == FORMAL_STAGE_A_STEPS and stage_a.get("completed_step") == FORMAL_STAGE_A_STEPS, "D025 preflight Stage-A is incomplete")
    inputs = _mapping(value.get("formal_train_inputs"), "D025 preflight formal train inputs")
    return {
        "stage_a_sha256": _sha(stage_a.get("checkpoint_sha256"), "D025 Stage-A SHA-256"),
        "train_manifest_path": str(inputs.get("manifest_path")),
        "train_manifest_sha256": _sha(inputs.get("manifest_sha256"), "D025 train manifest SHA-256"),
        "observation_identity": _mapping(inputs.get("observation_identity"), "D025 observation identity"),
        "teacher_identity": _mapping(inputs.get("teacher_identity"), "D025 teacher identity"),
        "derived_cache_lineage": _mapping(inputs.get("derived_cache_lineage"), "D025 derived lineage"),
    }


def _validate_eval(
    metrics: Mapping[str, Any], *, training: Mapping[str, Any], preflight: Mapping[str, Any]
) -> tuple[dict[str, dict[str, float | int]], bool, dict[str, Any]]:
    _require(metrics.get("stage") == "T3_CAUSAL_STAGE_B", "evaluation is not Stage-B T3")
    _require(metrics.get("target", {}).get("type") == PSEUDO_GT_TARGET, "evaluation target is not declared pseudo-GT")
    claims = _mapping(metrics.get("claims"), "evaluation claims")
    _require(claims.get("paper_accuracy") is False and claims.get("paper_gt") is False, "evaluation promotes pseudo-GT to paper evidence")
    _require(metrics.get("records_evaluated") == FORMAL_WINDOWS, "evaluation does not contain 238 T3 windows")
    coverage = _mapping(metrics.get("formal_temporal_coverage"), "formal temporal coverage")
    _require(coverage.get("manifest_records") == FORMAL_MANIFEST_RECORDS, "evaluation manifest coverage differs")
    _require(coverage.get("derived_endpoint_records") == FORMAL_DERIVED_RECORDS, "evaluation derived coverage differs")
    _require(coverage.get("evaluable_t3_windows") == FORMAL_WINDOWS, "evaluation causal coverage differs")
    methods = _mapping(metrics.get("methods"), "evaluation methods")
    _require(RAW_OWNER in methods, f"evaluation lacks raw owner {RAW_OWNER}")
    raw = _mapping(methods[RAW_OWNER], f"methods.{RAW_OWNER}")
    variant = _mapping(raw.get("output_variant"), "raw owner output variant")
    _require(variant.get("type") == "RAW_MODEL_OUTPUT", "clamp0 cannot own D025 gates")
    parsed = {name: _metric(raw.get(name), f"methods.{RAW_OWNER}.{name}") for name in RAW_METRICS}
    checkpoint = _mapping(metrics.get("checkpoint"), "evaluation checkpoint")
    _require(checkpoint.get("step") == FORMAL_STEPS, "evaluation checkpoint is not final 15k")
    _require(_sha(checkpoint.get("checkpoint_sha256"), "evaluation checkpoint SHA-256") == training["checkpoint_sha256"], "evaluation/training checkpoint SHA mismatch")
    _require(str(checkpoint.get("git_hash")) == training["git_hash"], "evaluation/training Git mismatch")
    config = _mapping(checkpoint.get("training_config"), "evaluation training config")
    _require(_canonical_sha256(config) == training["config_fingerprint"], "evaluation/training resolved config fingerprint mismatch")
    positivity = _mapping(config.get("positivity_ablation"), "evaluation positivity config")
    _require(positivity.get("enabled") is True, "evaluation checkpoint is not D025 positivity")
    data = _mapping(config.get("data"), "evaluation training data config")
    train = _mapping(config.get("train"), "evaluation training config")
    _require(data.get("sequence_length") == 3 and train.get("steps") == FORMAL_STEPS, "evaluation training schedule is not full Stage-B")
    _require(str(train.get("initialization_checkpoint_sha256")) == preflight["stage_a_sha256"], "D025 Stage-A initialization SHA mismatch")
    _require(Path(str(data.get("manifest_path"))).resolve() == Path(preflight["train_manifest_path"]).resolve(), "D025 training manifest differs from frozen preflight")
    _require(_mapping(data.get("observation_cache_identity"), "training observation identity") == preflight["observation_identity"], "D025 observation cache differs from frozen preflight")
    _require(_mapping(data.get("teacher_cache_identity"), "training teacher identity") == preflight["teacher_identity"], "D025 teacher cache differs from frozen preflight")
    _require(_mapping(data.get("derived_cache_lineage"), "training derived lineage") == preflight["derived_cache_lineage"], "D025 derived cache differs from frozen preflight")
    holdout = _mapping(metrics.get("holdout_and_raw_lineage"), "evaluation holdout lineage")
    _require(_sha(holdout.get("training_manifest_sha256"), "evaluation training manifest SHA") == preflight["train_manifest_sha256"], "D025 evaluation training manifest differs from frozen preflight")
    spatial = _mapping(metrics.get("spatial_checkpoint"), "evaluation spatial checkpoint")
    _require(_sha(spatial.get("checkpoint_sha256"), "evaluation Stage-A SHA-256") == preflight["stage_a_sha256"], "evaluation Stage-A differs from frozen preflight")
    evaluator = _mapping(metrics.get("evaluator"), "evaluation evaluator receipt")
    _require(isinstance(evaluator.get("git_hash"), str) and len(str(evaluator["git_hash"])) == 40, "evaluation evaluator Git hash is invalid")
    for key in ("eval_py_sha256", "evaluation_module_sha256"):
        _sha(evaluator.get(key), f"evaluation evaluator {key}")
    eligible = (
        claims.get("final_training_checkpoint") is True
        and claims.get("final_acceptance_eligible") is True
        and claims.get("full_validation_selection") is True
        and claims.get("formal_holdout") is True
    )
    return parsed, eligible, {"methods": methods, "cache_identities": metrics.get("cache_identities"), "coverage": coverage, "holdout": holdout, "evaluator": evaluator, "comparisons": metrics.get("comparisons")}


def _validate_canonical(
    report: JSONArtifact,
    metrics_artifact: JSONArtifact,
    eval_root: Path,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    _require(report.value.get("component") == "stage-b-final-holdout-evaluation-with-sign-health", "canonical report component differs")
    checkpoint = _mapping(report.value.get("checkpoint"), "canonical report checkpoint")
    _require(checkpoint.get("step") == FORMAL_STEPS and checkpoint.get("configured_steps") == FORMAL_STEPS, "canonical report is not final 15k")
    artifacts = _mapping(report.value.get("artifacts"), "canonical report artifacts")
    _require(metrics_artifact.sha256 == _sha(artifacts.get("metrics_json_sha256"), "canonical report metrics JSON SHA"), "canonical report/metrics.json hash mismatch")
    _require(_sha256(eval_root / "metrics.csv") == _sha(artifacts.get("metrics_csv_sha256"), "canonical report metrics CSV SHA"), "canonical report/metrics.csv hash mismatch")
    metrics = metrics_artifact.value
    methods = _mapping(metrics.get("methods"), "canonical methods")
    canonical_checkpoint = _mapping(metrics.get("checkpoint"), "canonical metrics checkpoint")
    _require(
        _sha(canonical_checkpoint.get("checkpoint_sha256"), "canonical metrics checkpoint SHA")
        == _sha(checkpoint.get("sha256"), "canonical report checkpoint SHA"),
        "canonical report/metrics checkpoint SHA mismatch",
    )
    report_evaluator = _mapping(report.value.get("evaluator"), "canonical report evaluator")
    metrics_evaluator = _mapping(metrics.get("evaluator"), "canonical metrics evaluator")
    for key in ("eval_py_sha256", "evaluation_module_sha256"):
        _require(
            _sha(report_evaluator.get(key), f"canonical report {key}")
            == _sha(metrics_evaluator.get(key), f"canonical metrics {key}"),
            f"canonical report/metrics evaluator {key} mismatch",
        )
    raw = _mapping(methods.get(RAW_OWNER), "canonical raw owner")
    _require(_mapping(raw.get("output_variant"), "canonical raw owner variant").get("type") == "RAW_MODEL_OUTPUT", "canonical raw owner is not raw")
    spatial = _mapping(metrics.get("spatial_checkpoint"), "canonical spatial checkpoint")
    _require(_sha(spatial.get("checkpoint_sha256"), "canonical Stage-A SHA") == preflight["stage_a_sha256"], "canonical and D025 do not share Stage-A")
    return {name: _metric(raw.get(name), f"canonical.{RAW_OWNER}.{name}") for name in RAW_METRICS}


def _gate_report(
    candidate: Mapping[str, Mapping[str, float | int]],
    canonical: Mapping[str, Mapping[str, float | int]],
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    for name in ("output_negative_rate", "output_invalid_rate", "output_nan_rate"):
        value = float(candidate[name]["value"])
        gates[f"raw_{name}_below_0_5_percent"] = {"value": value, "limit": RAW_HEALTH_LIMIT, "pass": value < RAW_HEALTH_LIMIT}
    for name in ERROR_GUARDRAIL_METRICS:
        value, reference = float(candidate[name]["value"]), float(canonical[name]["value"])
        relative = _relative_change(value, reference)
        gates[f"{name}_no_more_than_2_percent_regression"] = {"candidate": value, "canonical": reference, "relative_change_percent": relative, "limit": MAX_ERROR_REGRESSION_PERCENT, "pass": relative is not None and relative <= MAX_ERROR_REGRESSION_PERCENT}
    value = float(candidate["invalid_region_completeness"]["value"])
    reference = float(canonical["invalid_region_completeness"]["value"])
    completeness_relative = _relative_change(value, reference)
    gates["invalid_region_completeness_no_more_than_2_percent_drop"] = {"candidate": value, "canonical": reference, "relative_change_percent": completeness_relative, "limit": -MAX_COMPLETENESS_DROP_PERCENT, "pass": completeness_relative is not None and completeness_relative >= -MAX_COMPLETENESS_DROP_PERCENT}
    comparisons = _mapping(comparisons, "D025 comparisons")
    spatial = _mapping(comparisons.get("T3_VGGT_vs_bilinear"), "D025 raw T3_VGGT comparison")
    temporal = _mapping(comparisons.get("T3_vs_T1_temporal"), "D025 temporal comparison")
    def comparison_value(container: Mapping[str, Any], name: str) -> float:
        return _finite(_mapping(container.get(name), name).get("relative_change_percent"), f"{name}.relative_change_percent")
    low = comparison_value(spatial, "low_confidence_epe_change")
    complete = comparison_value(spatial, "invalid_region_completeness_change")
    trusted = comparison_value(spatial, "trusted_region_degradation")
    tepe = _finite(temporal.get("relative_change_percent"), "T3_vs_T1_temporal.relative_change_percent")
    gates["low_confidence_epe_improves_at_least_10_percent"] = {"value": low, "limit": LOW_CONFIDENCE_IMPROVEMENT_PERCENT, "pass": low <= LOW_CONFIDENCE_IMPROVEMENT_PERCENT}
    gates["invalid_region_completeness_improves_at_least_15_percent"] = {"value": complete, "limit": COMPLETENESS_IMPROVEMENT_PERCENT, "pass": complete >= COMPLETENESS_IMPROVEMENT_PERCENT}
    gates["trusted_region_error_degradation_at_most_2_percent"] = {"value": trusted, "limit": TRUSTED_DEGRADATION_PERCENT, "pass": trusted <= TRUSTED_DEGRADATION_PERCENT}
    gates["T3_temporal_error_improves_at_least_10_percent"] = {"value": tepe, "limit": TEMPORAL_IMPROVEMENT_PERCENT, "pass": tepe <= TEMPORAL_IMPROVEMENT_PERCENT}
    return {"all_required_gates_pass": all(item["pass"] for item in gates.values()), "gates": gates}


def audit_d025_evaluation(
    d025_training_audit_path: str | Path,
    d025_evaluation_dir: str | Path,
    canonical_stage_b_report_path: str | Path,
    canonical_stage_b_evaluation_dir: str | Path,
    d025_preflight_path: str | Path,
) -> dict[str, Any]:
    """Cross-audit the final D-025 arm without modifying any input artifact."""

    training_audit = _load_json(d025_training_audit_path, "D025 training audit")
    preflight = _load_json(d025_preflight_path, "D025 frozen preflight")
    canonical_report = _load_json(canonical_stage_b_report_path, "canonical Stage-B final report")
    d025_root = Path(d025_evaluation_dir).expanduser().resolve()
    canonical_root = Path(canonical_stage_b_evaluation_dir).expanduser().resolve()
    _require(d025_root.is_dir() and not d025_root.is_symlink(), "D025 evaluation directory is invalid")
    _require(canonical_root.is_dir() and not canonical_root.is_symlink(), "canonical evaluation directory is invalid")
    d025_metrics_artifact = _load_json(d025_root / "metrics.json", "D025 metrics.json")
    canonical_metrics_artifact = _load_json(canonical_root / "metrics.json", "canonical metrics.json")
    training = _validate_training_audit(training_audit.value)
    frozen = _validate_preflight(preflight.value)
    candidate, eval_final, candidate_context = _validate_eval(d025_metrics_artifact.value, training=training, preflight=frozen)
    canonical = _validate_canonical(
        canonical_report,
        canonical_metrics_artifact,
        canonical_root,
        preflight=frozen,
    )
    _validate_csv(d025_root, _mapping(d025_metrics_artifact.value.get("methods"), "D025 methods"))
    _validate_csv(canonical_root, _mapping(canonical_metrics_artifact.value.get("methods"), "canonical methods"))
    _require(candidate_context["cache_identities"] == canonical_metrics_artifact.value.get("cache_identities"), "D025/canonical evaluation cache identities differ")
    _require(candidate_context["coverage"] == canonical_metrics_artifact.value.get("formal_temporal_coverage"), "D025/canonical evaluation coverage differs")
    canonical_holdout = _mapping(canonical_metrics_artifact.value.get("holdout_and_raw_lineage"), "canonical holdout lineage")
    _require(
        _sha(candidate_context["holdout"].get("evaluation_manifest_sha256"), "D025 evaluation manifest SHA")
        == _sha(canonical_holdout.get("evaluation_manifest_sha256"), "canonical evaluation manifest SHA"),
        "D025/canonical evaluation manifest differs",
    )
    gates = _gate_report(candidate, canonical, _mapping(candidate_context["comparisons"], "D025 comparisons"))
    producer_eligible = bool(training["formal"] and eval_final)
    if not producer_eligible:
        status, result = "INELIGIBLE_FOR_FINAL_GATE", "INELIGIBLE"
    elif gates["all_required_gates_pass"]:
        status, result = "D025_FINAL_CONTROLLED_COMPARISON_PASS", "PASS"
    else:
        status, result = "D025_FINAL_CONTROLLED_COMPARISON_FAIL", "FAIL"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "component": AUDIT_COMPONENT,
        "status": status,
        "read_only": True,
        "final_gate": {"eligible": producer_eligible, "result": result, "limited_or_intermediate_cannot_pass": True},
        "claims": {"raw_owner": RAW_OWNER, "clamp0_owner": False, "pseudo_gt_engineering_only": True, "paper_ground_truth": False, "paper_accuracy": False},
        "primary_objective": {"raw_negative_rate": {"strict_limit": RAW_HEALTH_LIMIT}, "raw_invalid_rate": {"strict_limit": RAW_HEALTH_LIMIT}},
        "raw_t3_vggt_comparison": {"d025": candidate, "canonical_stage_b": canonical},
        "gates": gates,
        "training": training,
        "artifacts": {"d025_training_audit": training_audit.identity(), "d025_preflight": preflight.identity(), "d025_metrics": d025_metrics_artifact.identity(), "d025_metrics_csv": {"path": str((d025_root / "metrics.csv").resolve()), "sha256": _sha256(d025_root / "metrics.csv")}, "canonical_stage_b_report": canonical_report.identity(), "canonical_metrics": canonical_metrics_artifact.identity(), "canonical_metrics_csv": {"path": str((canonical_root / "metrics.csv").resolve()), "sha256": _sha256(canonical_root / "metrics.csv")}},
    }


def _safe_output(path: Path, protected_roots: Sequence[Path]) -> Path:
    output = path.expanduser().resolve()
    _require(all(root not in output.parents and output != root for root in protected_roots), "--json-out must not write inside an audited evaluation directory")
    return output


def _write_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d025-training-audit", type=Path, required=True)
    parser.add_argument("--d025-evaluation-dir", type=Path, required=True)
    parser.add_argument("--canonical-stage-b-report", type=Path, required=True)
    parser.add_argument("--canonical-stage-b-evaluation-dir", type=Path, required=True)
    parser.add_argument("--d025-preflight", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_d025_evaluation(
            args.d025_training_audit,
            args.d025_evaluation_dir,
            args.canonical_stage_b_report,
            args.canonical_stage_b_evaluation_dir,
            args.d025_preflight,
        )
        if args.json_out is not None:
            _write_atomic(_safe_output(args.json_out, [args.d025_evaluation_dir, args.canonical_stage_b_evaluation_dir]), report)
    except D025EvaluationAuditError as exc:
        print(f"D025 evaluation audit failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "final_gate": report["final_gate"]}, sort_keys=True))
    return 0 if report["final_gate"]["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
