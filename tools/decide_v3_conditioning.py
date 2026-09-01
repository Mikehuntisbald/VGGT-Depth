#!/usr/bin/env python3
"""Fail-closed metric decisions for calibration-conditioning v3 ablations.

The tool consumes a manifest written by ``run_v3_experiments.py``.  Aggregate
metrics are sufficient only for the seed-42 compute-screening recommendation.
A final GO requires all seeds 42/43/44 and paired per-record JSONL for a
deterministic bootstrap confidence interval; missing records never produce a
fabricated interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COMPONENT = "v3-calibration-conditioning-decision"
EXPECTED_SEEDS = (42, 43, 44)
STAGE_A_RECORDS = 244
STAGE_B_RECORDS = 238
ARMS = ("A0", "A1", "A2", "A3", "B0", "B1")

# Predeclared engineering thresholds from the v3 plan.
PRIMARY_IMPROVEMENT_PERCENT = 1.0
MAX_EPE_DEGRADATION_PERCENT = 0.5
MAX_BOUNDARY_DEGRADATION_PERCENT = 0.5
MAX_TRUSTED_DEGRADATION_PERCENT = 2.0
MAX_OUTPUT_BAD_RATE = 0.005
MIN_T3_VS_T1_TEMPORAL_IMPROVEMENT_PERCENT = 10.0
MIN_TEMPORAL_POSE_IMPROVEMENT_PERCENT = 5.0
MIN_TEMPORAL_PAIRED_RECORDS = 30
MIN_TEMPORAL_PAIRED_FRACTION = 0.10
MAX_RUNTIME_DEGRADATION_PERCENT = 5.0
RUNTIME_METRICS = (
    "model_forward_latency_ms_mean",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
)
EXPECTED_SWITCHES = {
    "A0": (False, False, False),
    "A1": (True, False, False),
    "A2": (False, True, False),
    "A3": (True, True, False),
    "B0": (True, True, False),
    "B1": (True, True, True),
}

LOWER_IS_BETTER_SAFETY = (
    ("epe_px", MAX_EPE_DEGRADATION_PERCENT),
    ("boundary_epe_px", MAX_BOUNDARY_DEGRADATION_PERCENT),
    ("trusted_region_epe_px", MAX_TRUSTED_DEGRADATION_PERCENT),
)
OUTPUT_RATE_METRICS = (
    "output_negative_rate",
    "output_invalid_rate",
    "output_nan_rate",
    "output_infinite_rate",
)
RECORD_METRICS = (
    "low_confidence_epe_px",
    "epe_px",
    "boundary_epe_px",
    "trusted_region_epe_px",
    "invalid_region_completeness",
    *OUTPUT_RATE_METRICS,
    "temporal_residual_error_native_px",
)


class DecisionInputError(ValueError):
    """Raised for malformed decision inputs before a report can be produced."""


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    metrics_path: Path
    records_path: Path
    metrics_sha256: str
    records_sha256: str | None
    method: str
    metrics: Mapping[str, float]
    runtime: Mapping[str, float]
    runtime_contract_sha256: str
    temporal_change_percent: float | None
    shared_lineage_sha256: str
    derived_lineage_sha256: str | None


@dataclass(frozen=True, slots=True)
class PairedRecordDeltas:
    values: tuple[float, ...]
    eligible_count: int
    total_count: int


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(
    path: Path, name: str, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    if not path.is_file():
        raise DecisionInputError(f"{name} does not exist: {path}")
    try:
        raw = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise DecisionInputError(f"{name} SHA-256 differs: {path}")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecisionInputError(f"cannot parse {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise DecisionInputError(f"{name} root must be a JSON object: {path}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionInputError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionInputError(f"{name} must be finite")
    return result


def _metric_domain(value: float, metric: str, name: str) -> float:
    if metric in OUTPUT_RATE_METRICS or metric == "invalid_region_completeness":
        if not 0.0 <= value <= 1.0:
            raise DecisionInputError(f"{name} must be in [0,1]")
    elif value < 0.0:
        raise DecisionInputError(f"{name} must be non-negative")
    return value


def _canonical_sha256(value: Any, name: str) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecisionInputError(f"{name} is not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _metric_value(methods: Mapping[str, Any], method: str, metric: str) -> float:
    method_payload = methods.get(method)
    if not isinstance(method_payload, Mapping):
        raise DecisionInputError(f"method {method!r} is absent")
    metric_payload = method_payload.get(metric)
    if not isinstance(metric_payload, Mapping) or metric_payload.get("valid") is not True:
        raise DecisionInputError(f"metric {method}.{metric} is absent or invalid")
    name = f"{method}.{metric}.value"
    return _metric_domain(_finite_number(metric_payload.get("value"), name), metric, name)


def _acceptance_eligible(report: Mapping[str, Any]) -> bool:
    claims = report.get("claims")
    if not isinstance(claims, Mapping):
        return False
    # Current evaluators own final_acceptance_eligible.  The alias keeps the
    # reader compatible with older synthetic fixtures without weakening a
    # current false value.
    if "final_acceptance_eligible" in claims:
        return claims.get("final_acceptance_eligible") is True
    return claims.get("acceptance_eligible") is True


def _load_arm_evidence(
    entry: Mapping[str, Any],
    *,
    arm: str,
    arm_name: str,
    seed: int,
    expected_count: int,
    validation_manifest_path: Path,
) -> ArmEvidence:
    metrics_value = entry.get("metrics_json")
    records_value = entry.get("per_record_jsonl")
    if not isinstance(metrics_value, str) or not metrics_value:
        raise DecisionInputError(f"{arm} metrics_json is missing")
    if not isinstance(records_value, str) or not records_value:
        raise DecisionInputError(f"{arm} per_record_jsonl is missing")
    metrics_path = Path(metrics_value).expanduser().resolve()
    records_path = Path(records_value).expanduser().resolve()
    metrics_sha256 = entry.get("metrics_sha256")
    records_sha256 = entry.get("per_record_jsonl_sha256")
    if not isinstance(metrics_sha256, str) or len(metrics_sha256) != 64:
        raise DecisionInputError(f"{arm} metrics_sha256 is missing or malformed")
    if records_sha256 is not None and (
        not isinstance(records_sha256, str) or len(records_sha256) != 64
    ):
        raise DecisionInputError(f"{arm} per_record_jsonl_sha256 is malformed")
    if records_path.is_file():
        if records_sha256 is None:
            raise DecisionInputError(f"{arm} per-record file lacks a declared SHA-256")
        if hashlib.sha256(records_path.read_bytes()).hexdigest() != records_sha256:
            raise DecisionInputError(f"{arm} per-record SHA-256 differs")
    elif records_sha256 is not None:
        raise DecisionInputError(f"{arm} declares a missing per-record artifact")
    report = _load_json(
        metrics_path, f"{arm} metrics", expected_sha256=metrics_sha256
    )
    if not _acceptance_eligible(report):
        raise DecisionInputError(f"{arm} evaluation is not final-acceptance eligible")
    # Callers prefix the arm with ``seedN.`` for actionable error messages.
    stage_a = arm.rsplit(".", 1)[-1].startswith("A")
    expected_stage = "T1_SPATIAL_ONLY" if stage_a else "T3_CAUSAL_STAGE_B"
    if report.get("stage") != expected_stage:
        raise DecisionInputError(
            f"{arm} stage mismatch: expected {expected_stage!r}, got {report.get('stage')!r}"
        )
    if report.get("crop_mode") != "full":
        raise DecisionInputError(f"{arm} must be a full-resolution evaluation")
    claims = report.get("claims")
    if not isinstance(claims, Mapping) or claims.get("full_validation_selection") is not True:
        raise DecisionInputError(f"{arm} is not the full validation selection")
    report_manifest = report.get("manifest_path")
    if not isinstance(report_manifest, str) or (
        Path(report_manifest).expanduser().resolve() != validation_manifest_path
    ):
        raise DecisionInputError(f"{arm} validation manifest path differs")
    resolved = report.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise DecisionInputError(f"{arm} resolved_config is missing")
    if resolved.get("seed") != seed:
        raise DecisionInputError(f"{arm} resolved seed must equal {seed}")
    calibration = resolved.get("calibration_conditioning_v3")
    if not isinstance(calibration, Mapping):
        raise DecisionInputError(f"{arm} calibration config is missing")
    expected_switches = EXPECTED_SWITCHES[arm_name]
    actual_switches = tuple(
        calibration.get(name)
        for name in ("use_rays", "use_stereo_pose", "use_temporal_pose")
    )
    if (
        calibration.get("enabled") is not True
        or calibration.get("protocol_version") != "dense_rays_factorized_pose_v3"
        or any(type(value) is not bool for value in actual_switches)
        or actual_switches != expected_switches
    ):
        raise DecisionInputError(f"{arm} calibration treatment identity differs")
    data_config = resolved.get("data")
    if not isinstance(data_config, Mapping) or data_config.get("sequence_length") != (
        1 if stage_a else 3
    ):
        raise DecisionInputError(f"{arm} sequence-length identity differs")
    eval_config = resolved.get("eval")
    if not isinstance(eval_config, Mapping) or eval_config.get("crop_mode") != "full":
        raise DecisionInputError(f"{arm} resolved eval crop mode differs")
    # ``eval.py`` stores this field as ``records_evaluated`` for both stages.
    # Its human-facing completion line calls Stage-B samples "windows", but
    # that alias is deliberately not part of metrics.json.
    count_field = "records_evaluated"
    if report.get(count_field) != expected_count:
        raise DecisionInputError(
            f"{arm} {count_field} must equal {expected_count}, got {report.get(count_field)!r}"
        )
    methods = report.get("methods")
    if not isinstance(methods, Mapping):
        raise DecisionInputError(f"{arm} methods are missing")
    method = "T1" if stage_a else "T3_VGGT"
    metric_names = (
        "low_confidence_epe_px",
        "epe_px",
        "boundary_epe_px",
        "trusted_region_epe_px",
        "invalid_region_completeness",
        *OUTPUT_RATE_METRICS,
    )
    values = {name: _metric_value(methods, method, name) for name in metric_names}
    runtime_payload = report.get("runtime_v3")
    if not isinstance(runtime_payload, Mapping):
        raise DecisionInputError(f"{arm} runtime_v3 is missing")
    runtime: dict[str, float] = {}
    for name in RUNTIME_METRICS:
        value = _finite_number(runtime_payload.get(name), f"{arm}.runtime_v3.{name}")
        if value <= 0.0:
            raise DecisionInputError(f"{arm}.runtime_v3.{name} must be positive")
        runtime[name] = value
    runtime_contract = runtime_payload.get("contract_version")
    timing_backend = runtime_payload.get("timing_backend")
    forward_calls = runtime_payload.get("model_forward_calls")
    report_device = report.get("device")
    if (
        runtime_contract != "matched_candidate_forward_runtime_v1"
        or timing_backend != "torch.cuda.Event"
        or isinstance(forward_calls, bool)
        or not isinstance(forward_calls, int)
        or forward_calls <= 0
        or not isinstance(report_device, str)
        or not report_device.startswith("cuda")
    ):
        raise DecisionInputError(f"{arm} runtime protocol/device identity differs")
    runtime_contract_sha256 = _canonical_sha256(
        {
            "contract_version": runtime_contract,
            "timing_backend": timing_backend,
            "model_forward_calls": forward_calls,
            "device": report_device,
        },
        f"{arm} runtime contract",
    )
    temporal_change: float | None = None
    if not stage_a:
        # T3_VGGT's within-report paired T1/T3 field is intentionally invalid:
        # its safe domain differs.  B1/B0 therefore use each run's native
        # teacher-residual value, then pair identical record IDs across runs.
        values["temporal_residual_error_native_px"] = _metric_value(
            methods, method, "temporal_residual_error_native_px"
        )
        comparisons = report.get("comparisons")
        temporal = (
            comparisons.get("T3_vs_T1_temporal")
            if isinstance(comparisons, Mapping)
            else None
        )
        if not isinstance(temporal, Mapping) or temporal.get("valid") is not True:
            raise DecisionInputError(f"{arm} T3_vs_T1_temporal comparison is invalid")
        temporal_change = _finite_number(
            temporal.get("relative_change_percent"),
            f"{arm}.T3_vs_T1_temporal.relative_change_percent",
        )
    cache_identities = report.get("cache_identities")
    calibration_lineage = data_config.get("calibration_sidecar_lineage")
    if not isinstance(cache_identities, Mapping) or not isinstance(
        calibration_lineage, Mapping
    ):
        raise DecisionInputError(f"{arm} shared validation lineage is missing")
    shared_lineage_sha256 = _canonical_sha256(
        {
            "manifest_path": str(validation_manifest_path),
            "cache_identities": cache_identities,
            "calibration_sidecar_lineage": calibration_lineage,
        },
        f"{arm} shared validation lineage",
    )
    derived_lineage = report.get("derived_cache_lineage")
    if stage_a:
        derived_lineage_sha256 = None
    else:
        if not isinstance(derived_lineage, Mapping):
            raise DecisionInputError(f"{arm} derived validation lineage is missing")
        derived_lineage_sha256 = _canonical_sha256(
            derived_lineage, f"{arm} derived validation lineage"
        )
    return ArmEvidence(
        metrics_path=metrics_path,
        records_path=records_path,
        metrics_sha256=metrics_sha256,
        records_sha256=records_sha256,
        method=method,
        metrics=values,
        runtime=runtime,
        runtime_contract_sha256=runtime_contract_sha256,
        temporal_change_percent=temporal_change,
        shared_lineage_sha256=shared_lineage_sha256,
        derived_lineage_sha256=derived_lineage_sha256,
    )


def _relative_change_percent(control: float, candidate: float) -> float:
    if control == 0.0:
        # Metrics are domain-validated non-negative. Zero->zero is unchanged;
        # zero->positive is an unambiguous regression even though the usual
        # percentage is undefined.
        return 0.0 if candidate == 0.0 else 100.0
    return 100.0 * (candidate - control) / abs(control)


def _aggregate_checks(
    control: ArmEvidence,
    candidate: ArmEvidence,
    *,
    primary_metric: str,
    minimum_improvement_percent: float,
    require_t3_gate: bool,
) -> dict[str, Any]:
    primary_change = _relative_change_percent(
        control.metrics[primary_metric], candidate.metrics[primary_metric]
    )
    checks: dict[str, bool] = {
        "primary_improvement": primary_change <= -minimum_improvement_percent,
    }
    changes: dict[str, float] = {primary_metric: primary_change}
    for metric, maximum_degradation in LOWER_IS_BETTER_SAFETY:
        change = _relative_change_percent(
            control.metrics[metric], candidate.metrics[metric]
        )
        changes[metric] = change
        checks[f"{metric}_degradation"] = change <= maximum_degradation
    completeness_change = (
        candidate.metrics["invalid_region_completeness"]
        - control.metrics["invalid_region_completeness"]
    )
    checks["completeness_non_decreasing"] = completeness_change >= 0.0
    for metric in OUTPUT_RATE_METRICS:
        control_value = control.metrics[metric]
        candidate_value = candidate.metrics[metric]
        checks[f"{metric}_absolute"] = candidate_value < MAX_OUTPUT_BAD_RATE
        checks[f"{metric}_vs_control"] = candidate_value <= control_value + 1e-12
    runtime_changes: dict[str, float] = {}
    checks["runtime_protocol_and_forward_calls_match"] = (
        control.runtime_contract_sha256 == candidate.runtime_contract_sha256
    )
    for metric in RUNTIME_METRICS:
        change = _relative_change_percent(
            control.runtime[metric], candidate.runtime[metric]
        )
        runtime_changes[metric] = change
        checks[f"runtime_{metric}_degradation"] = (
            change <= MAX_RUNTIME_DEGRADATION_PERCENT
        )
    if require_t3_gate:
        assert candidate.temporal_change_percent is not None
        checks["t3_vs_t1_temporal_at_least_10_percent"] = (
            candidate.temporal_change_percent
            <= -MIN_T3_VS_T1_TEMPORAL_IMPROVEMENT_PERCENT
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes_percent": changes,
        "runtime_relative_changes_percent": runtime_changes,
        "completeness_absolute_change": completeness_change,
        "candidate_t3_vs_t1_temporal_change_percent": (
            candidate.temporal_change_percent if require_t3_gate else None
        ),
    }


def _optional_record_metric(value: Any, name: str) -> float | None:
    """Parse a per-record scalar; explicit invalid entries become ``None``."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        valid = value.get("valid")
        if valid is False:
            return None
        if valid is not True:
            raise DecisionInputError(f"{name}.valid must be boolean")
        value = value.get("value")
    return _finite_number(value, name)


def _load_record_metrics(
    path: Path, *, expected_count: int, expected_sha256: str | None
) -> dict[str, Mapping[str, float | None]]:
    if not path.is_file():
        raise DecisionInputError(f"per-record bootstrap data is unavailable: {path}")
    records: dict[str, Mapping[str, float | None]] = {}
    try:
        raw = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is None or actual_sha256 != expected_sha256:
            raise DecisionInputError(f"per-record SHA-256 differs: {path}")
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DecisionInputError(f"cannot read per-record data: {path}") from exc
    if len(lines) != expected_count:
        raise DecisionInputError(
            f"per-record data must have {expected_count} rows, got {len(lines)}: {path}"
        )
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DecisionInputError(f"blank per-record row {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionInputError(
                f"malformed per-record row {path}:{line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise DecisionInputError(f"per-record row is not an object: {path}:{line_number}")
        record_id = row.get("record_id")
        metrics = row.get("metrics")
        if not isinstance(record_id, str) or not record_id:
            raise DecisionInputError(f"record_id is missing: {path}:{line_number}")
        if record_id in records:
            raise DecisionInputError(f"duplicate record_id {record_id!r}: {path}")
        if not isinstance(metrics, Mapping):
            raise DecisionInputError(f"metrics are missing: {path}:{line_number}")
        normalized: dict[str, float | None] = {}
        for name, value in metrics.items():
            if name in RECORD_METRICS:
                normalized[name] = _optional_record_metric(
                    value, f"{path}:{line_number}.metrics.{name}"
                )
                if normalized[name] is not None:
                    normalized[name] = _metric_domain(
                        normalized[name],
                        name,
                        f"{path}:{line_number}.metrics.{name}",
                    )
        records[record_id] = normalized
    return records


def _paired_deltas(
    control_path: Path,
    candidate_path: Path,
    *,
    metric: str,
    expected_count: int,
    allow_invalid_intersection: bool,
    control_sha256: str | None,
    candidate_sha256: str | None,
) -> PairedRecordDeltas:
    control = _load_record_metrics(
        control_path,
        expected_count=expected_count,
        expected_sha256=control_sha256,
    )
    candidate = _load_record_metrics(
        candidate_path,
        expected_count=expected_count,
        expected_sha256=candidate_sha256,
    )
    if set(control) != set(candidate):
        raise DecisionInputError(
            f"paired record IDs differ: {control_path} vs {candidate_path}"
        )
    deltas: list[float] = []
    for record_id in sorted(control):
        if metric not in control[record_id] or metric not in candidate[record_id]:
            if allow_invalid_intersection:
                # The evaluator may omit an invalid metric instead of writing
                # an explicit null/valid=false payload for a safe-mask-empty
                # window. Both representations mean ineligible for this pair.
                continue
            raise DecisionInputError(
                f"paired metric {metric!r} is missing for record {record_id!r}"
            )
        control_value = control[record_id][metric]
        candidate_value = candidate[record_id][metric]
        if control_value is None or candidate_value is None:
            if allow_invalid_intersection:
                continue
            raise DecisionInputError(
                f"paired metric {metric!r} is invalid for record {record_id!r}"
            )
        deltas.append(candidate_value - control_value)
    return PairedRecordDeltas(
        values=tuple(deltas),
        eligible_count=len(deltas),
        total_count=len(control),
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise DecisionInputError("cannot compute percentile of an empty sequence")
    position = quantile * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_ci(
    deltas_by_seed: Mapping[int, Sequence[float]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, float | int]:
    if replicates < 200:
        raise DecisionInputError("bootstrap_replicates must be at least 200")
    if tuple(sorted(deltas_by_seed)) != EXPECTED_SEEDS:
        raise DecisionInputError("bootstrap requires exactly seeds 42,43,44")
    if any(not values for values in deltas_by_seed.values()):
        raise DecisionInputError("bootstrap seed has no paired records")
    rng = random.Random(random_seed)
    seeds = list(EXPECTED_SEEDS)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        seed_means: list[float] = []
        for seed in sampled_seeds:
            values = deltas_by_seed[seed]
            resampled = [rng.choice(values) for _ in values]
            seed_means.append(sum(resampled) / len(resampled))
        estimates.append(sum(seed_means) / len(seed_means))
    estimates.sort()
    observed_seed_means = [
        sum(deltas_by_seed[seed]) / len(deltas_by_seed[seed])
        for seed in EXPECTED_SEEDS
    ]
    return {
        "replicates": replicates,
        "random_seed": random_seed,
        "observed_mean_delta": sum(observed_seed_means) / len(observed_seed_means),
        "ci95_lower": _percentile(estimates, 0.025),
        "ci95_upper": _percentile(estimates, 0.975),
    }


def _comparison_decision(
    evidence_by_seed: Mapping[int, Mapping[str, ArmEvidence]],
    *,
    control_arm: str,
    candidate_arm: str,
    primary_metric: str,
    minimum_improvement_percent: float,
    expected_count: int,
    require_t3_gate: bool,
    allow_invalid_record_intersection: bool,
    bootstrap_replicates: int,
    bootstrap_random_seed: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if tuple(sorted(evidence_by_seed)) != EXPECTED_SEEDS:
        reasons.append("final decision requires exactly seeds 42,43,44")
        return {"status": "NO-GO", "reasons": reasons, "fail_closed": True}
    aggregate_by_seed: dict[str, Any] = {}
    deltas_by_seed: dict[int, Sequence[float]] = {}
    paired_coverage_by_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        arms = evidence_by_seed[seed]
        if control_arm not in arms or candidate_arm not in arms:
            reasons.append(f"seed {seed} lacks {control_arm}/{candidate_arm}")
            continue
        aggregate = _aggregate_checks(
            arms[control_arm],
            arms[candidate_arm],
            primary_metric=primary_metric,
            minimum_improvement_percent=minimum_improvement_percent,
            require_t3_gate=require_t3_gate,
        )
        aggregate_by_seed[str(seed)] = aggregate
        if not aggregate["passed"]:
            reasons.append(f"seed {seed} aggregate thresholds failed")
        try:
            paired = _paired_deltas(
                arms[control_arm].records_path,
                arms[candidate_arm].records_path,
                metric=primary_metric,
                expected_count=expected_count,
                allow_invalid_intersection=allow_invalid_record_intersection,
                control_sha256=arms[control_arm].records_sha256,
                candidate_sha256=arms[candidate_arm].records_sha256,
            )
            deltas_by_seed[seed] = paired.values
            fraction = paired.eligible_count / paired.total_count
            paired_coverage_by_seed[str(seed)] = {
                "eligible_records": paired.eligible_count,
                "total_records": paired.total_count,
                "eligible_fraction": fraction,
            }
            if allow_invalid_record_intersection and (
                paired.eligible_count < MIN_TEMPORAL_PAIRED_RECORDS
                or fraction < MIN_TEMPORAL_PAIRED_FRACTION
            ):
                reasons.append(
                    f"seed {seed} native temporal paired intersection has "
                    f"{paired.eligible_count}/{paired.total_count} eligible records; "
                    f"requires >= {MIN_TEMPORAL_PAIRED_RECORDS} and >= "
                    f"{MIN_TEMPORAL_PAIRED_FRACTION:.0%}"
                )
        except DecisionInputError as exc:
            reasons.append(str(exc))
    if reasons:
        return {
            "status": "NO-GO",
            "reasons": reasons,
            "aggregate_by_seed": aggregate_by_seed,
            "paired_coverage_by_seed": paired_coverage_by_seed,
            "fail_closed": True,
        }
    seed_mean_deltas = {
        str(seed): sum(values) / len(values) for seed, values in deltas_by_seed.items()
    }
    if any(value >= 0.0 for value in seed_mean_deltas.values()):
        reasons.append("paired primary metric does not improve in every seed")
    bootstrap = _bootstrap_ci(
        deltas_by_seed,
        replicates=bootstrap_replicates,
        random_seed=bootstrap_random_seed,
    )
    if float(bootstrap["ci95_upper"]) >= 0.0:
        reasons.append("paired bootstrap 95% CI upper bound is not below zero")
    return {
        "status": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "aggregate_by_seed": aggregate_by_seed,
        "paired_coverage_by_seed": paired_coverage_by_seed,
        "paired_seed_mean_deltas": seed_mean_deltas,
        "bootstrap": bootstrap,
        "fail_closed": bool(reasons),
    }


def _seed42_screening(
    evidence_by_seed: Mapping[int, Mapping[str, ArmEvidence]],
    *,
    unique_static_calibrations: int | None,
    temporal_pose_identifiable: bool,
) -> dict[str, Any]:
    arms = evidence_by_seed.get(42)
    if arms is None:
        return {
            "continue_additional_seeds": False,
            "reason": "seed42 evidence is unavailable",
            "final_evidence": False,
        }
    screens: dict[str, Any] = {}
    specifications = [
        (
            "rays",
            "A0",
            "A1",
            "low_confidence_epe_px",
            PRIMARY_IMPROVEMENT_PERCENT,
            False,
        ),
    ]
    if isinstance(unique_static_calibrations, int) and (
        not isinstance(unique_static_calibrations, bool)
        and unique_static_calibrations >= 2
    ):
        specifications.append(
            (
                "static_stereo_pose",
                "A1",
                "A3",
                "low_confidence_epe_px",
                PRIMARY_IMPROVEMENT_PERCENT,
                False,
            )
        )
    else:
        screens["static_stereo_pose"] = {
            "passed": False,
            "status": "NOT_IDENTIFIABLE",
            "reason": "fewer than two audited static stereo calibrations",
        }
    if temporal_pose_identifiable:
        specifications.append(
            (
                "temporal_pose",
                "B0",
                "B1",
                "temporal_residual_error_native_px",
                MIN_TEMPORAL_POSE_IMPROVEMENT_PERCENT,
                True,
            )
        )
    else:
        screens["temporal_pose"] = {
            "passed": False,
            "status": "NOT_IDENTIFIABLE",
            "reason": "audited temporal-pose variation is unavailable",
        }
    for name, control, candidate, primary, improvement, temporal in specifications:
        if control not in arms or candidate not in arms:
            screens[name] = {"passed": False, "reason": "arm evidence missing"}
            continue
        screens[name] = _aggregate_checks(
            arms[control],
            arms[candidate],
            primary_metric=primary,
            minimum_improvement_percent=improvement,
            require_t3_gate=temporal,
        )
    return {
        "continue_additional_seeds": any(
            value.get("passed") is True for value in screens.values()
        ),
        "components": screens,
        "final_evidence": False,
        "warning": (
            "aggregate seed42 screening controls compute only; it is not a GO and "
            "never substitutes for three-seed paired bootstrap evidence"
        ),
    }


def decide_manifest(
    manifest_path: str | Path,
    *,
    bootstrap_replicates: int = 2_000,
    bootstrap_random_seed: int = 20260901,
) -> dict[str, Any]:
    """Return a machine-readable GO/NO-GO/NOT_IDENTIFIABLE report."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(manifest_file, "decision manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DecisionInputError("decision manifest schema_version mismatch")
    if manifest.get("component") != "v3-experiment-decision-inputs":
        raise DecisionInputError("decision manifest component mismatch")
    expected_counts = manifest.get("expected_counts")
    if not isinstance(expected_counts, Mapping):
        raise DecisionInputError("decision manifest expected_counts are missing")
    stage_a_count = expected_counts.get("stage_a_records")
    stage_b_count = expected_counts.get("stage_b_windows")
    if stage_a_count != STAGE_A_RECORDS or stage_b_count != STAGE_B_RECORDS:
        raise DecisionInputError(
            "formal decision requires 244 Stage-A records and 238 Stage-B windows"
        )
    validation_manifest = manifest.get("validation_manifest")
    if not isinstance(validation_manifest, Mapping):
        raise DecisionInputError("decision manifest validation_manifest is missing")
    validation_manifest_value = validation_manifest.get("path")
    validation_manifest_sha256 = validation_manifest.get("sha256")
    if not isinstance(validation_manifest_value, str) or not isinstance(
        validation_manifest_sha256, str
    ):
        raise DecisionInputError("validation manifest identity is malformed")
    validation_manifest_path = Path(validation_manifest_value).expanduser().resolve()
    if not validation_manifest_path.is_file() or (
        hashlib.sha256(validation_manifest_path.read_bytes()).hexdigest()
        != validation_manifest_sha256
    ):
        raise DecisionInputError("validation manifest identity changed")
    seeds_payload = manifest.get("seeds")
    if not isinstance(seeds_payload, Mapping):
        raise DecisionInputError("decision manifest seeds are missing")

    evidence_by_seed: dict[int, dict[str, ArmEvidence]] = {}
    input_errors: list[str] = []
    for seed_text, arms_payload in seeds_payload.items():
        try:
            seed = int(seed_text)
        except (TypeError, ValueError):
            input_errors.append(f"malformed seed key {seed_text!r}")
            continue
        if seed not in EXPECTED_SEEDS or not isinstance(arms_payload, Mapping):
            input_errors.append(f"unsupported or malformed seed {seed}")
            continue
        missing_arms = sorted(set(ARMS) - set(arms_payload))
        extra_arms = sorted(set(arms_payload) - set(ARMS))
        if missing_arms:
            input_errors.append(f"seed {seed} lacks required arms {missing_arms}")
        if extra_arms:
            input_errors.append(f"seed {seed} has unsupported arms {extra_arms}")
        arms: dict[str, ArmEvidence] = {}
        for arm, entry in arms_payload.items():
            if arm not in ARMS or not isinstance(entry, Mapping):
                input_errors.append(f"seed {seed} has malformed arm {arm!r}")
                continue
            expected_count = STAGE_A_RECORDS if arm.startswith("A") else STAGE_B_RECORDS
            try:
                arms[arm] = _load_arm_evidence(
                    entry,
                    arm=f"seed{seed}.{arm}",
                    arm_name=arm,
                    seed=seed,
                    expected_count=expected_count,
                    validation_manifest_path=validation_manifest_path,
                )
            except DecisionInputError as exc:
                input_errors.append(str(exc))
        evidence_by_seed[seed] = arms

    shared_lineages = {
        evidence.shared_lineage_sha256
        for arms in evidence_by_seed.values()
        for evidence in arms.values()
    }
    if len(shared_lineages) > 1:
        input_errors.append("validation manifest/cache/calibration lineage differs across arms")
    derived_lineages = {
        evidence.derived_lineage_sha256
        for arms in evidence_by_seed.values()
        for name, evidence in arms.items()
        if name in ("B0", "B1")
    }
    if len(derived_lineages) > 1:
        input_errors.append("Stage-B derived validation lineage differs across arms")

    rays = _comparison_decision(
        evidence_by_seed,
        control_arm="A0",
        candidate_arm="A1",
        primary_metric="low_confidence_epe_px",
        minimum_improvement_percent=PRIMARY_IMPROVEMENT_PERCENT,
        expected_count=STAGE_A_RECORDS,
        require_t3_gate=False,
        allow_invalid_record_intersection=False,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )
    identifiability = manifest.get("identifiability")
    unique_static = (
        identifiability.get("unique_static_stereo_calibrations")
        if isinstance(identifiability, Mapping)
        else None
    )
    if not isinstance(unique_static, int) or unique_static < 2:
        static_pose: dict[str, Any] = {
            "status": "NOT_IDENTIFIABLE",
            "reasons": [
                "static stereo pose has fewer than two audited calibrations"
            ],
            "unique_static_stereo_calibrations": unique_static,
        }
    else:
        static_pose = _comparison_decision(
            evidence_by_seed,
            control_arm="A1",
            candidate_arm="A3",
            primary_metric="low_confidence_epe_px",
            minimum_improvement_percent=PRIMARY_IMPROVEMENT_PERCENT,
            expected_count=STAGE_A_RECORDS,
            require_t3_gate=False,
            allow_invalid_record_intersection=False,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_random_seed=bootstrap_random_seed + 1,
        )
    temporal_identifiable = (
        identifiability.get("temporal_pose_varies") is True
        if isinstance(identifiability, Mapping)
        else False
    )
    if not temporal_identifiable:
        temporal_pose: dict[str, Any] = {
            "status": "NOT_IDENTIFIABLE",
            "reasons": ["audited temporal-pose variation is unavailable"],
        }
    else:
        temporal_pose = _comparison_decision(
            evidence_by_seed,
            control_arm="B0",
            candidate_arm="B1",
            primary_metric="temporal_residual_error_native_px",
            minimum_improvement_percent=MIN_TEMPORAL_POSE_IMPROVEMENT_PERCENT,
            expected_count=STAGE_B_RECORDS,
            require_t3_gate=True,
            allow_invalid_record_intersection=True,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_random_seed=bootstrap_random_seed + 2,
        )

    components = {
        "rays": rays,
        "static_stereo_pose": static_pose,
        "temporal_pose": temporal_pose,
    }
    if rays["status"] != "GO":
        overall = "NO-GO"
        exact_arm = None
        recommended = {
            "use_rays": False,
            "use_stereo_pose": False,
            "use_temporal_pose": False,
        }
        recipe_boundary = (
            "No learned calibration conditioning is promoted. Keep only the "
            "non-learned audited sidecar, hard geometry constraints, and dual-K/B."
        )
    elif temporal_pose["status"] == "GO" and static_pose["status"] != "NO-GO":
        overall = "GO"
        exact_arm = "B1"
        recommended = {
            "use_rays": True,
            "use_stereo_pose": True,
            "use_temporal_pose": True,
        }
        recipe_boundary = (
            "Exact evaluated B1 recipe on its same-seed A3 lineage. Stereo-pose "
            "conditioning remains an unidentifiable background factor and this "
            "recipe does not establish that static extrinsic embedding is effective."
        )
    elif static_pose["status"] == "GO":
        overall = "GO"
        exact_arm = "A3"
        recommended = {
            "use_rays": True,
            "use_stereo_pose": True,
            "use_temporal_pose": False,
        }
        recipe_boundary = (
            "Exact evaluated A3 recipe. Temporal pose is not promoted because its "
            "conditional B1/B0 evidence did not pass."
        )
    else:
        overall = "GO"
        exact_arm = "A1"
        recommended = {
            "use_rays": True,
            "use_stereo_pose": False,
            "use_temporal_pose": False,
        }
        recipe_boundary = (
            "Exact evaluated A1 recipe. A stereo-off plus temporal-on combination "
            "is not recommended because it was never trained or evaluated."
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "decision": overall,
        "components": components,
        "recommended_switches": recommended,
        "promotion_recipe": {
            "exact_evaluated_arm": exact_arm,
            "switches": recommended,
            "claim_boundary": recipe_boundary,
        },
        "screening": _seed42_screening(
            evidence_by_seed,
            unique_static_calibrations=unique_static,
            temporal_pose_identifiable=temporal_identifiable,
        ),
        "input_errors": input_errors,
        "thresholds": {
            "primary_improvement_percent": PRIMARY_IMPROVEMENT_PERCENT,
            "max_epe_degradation_percent": MAX_EPE_DEGRADATION_PERCENT,
            "max_boundary_degradation_percent": MAX_BOUNDARY_DEGRADATION_PERCENT,
            "max_trusted_degradation_percent": MAX_TRUSTED_DEGRADATION_PERCENT,
            "max_output_bad_rate": MAX_OUTPUT_BAD_RATE,
            "max_runtime_degradation_percent": MAX_RUNTIME_DEGRADATION_PERCENT,
            "runtime_metrics": list(RUNTIME_METRICS),
            "min_t3_vs_t1_temporal_improvement_percent": (
                MIN_T3_VS_T1_TEMPORAL_IMPROVEMENT_PERCENT
            ),
            "min_temporal_pose_improvement_percent": (
                MIN_TEMPORAL_POSE_IMPROVEMENT_PERCENT
            ),
            "min_temporal_paired_records": MIN_TEMPORAL_PAIRED_RECORDS,
            "min_temporal_paired_fraction": MIN_TEMPORAL_PAIRED_FRACTION,
            "required_seeds": list(EXPECTED_SEEDS),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_random_seed": bootstrap_random_seed,
        },
        "claim_boundary": (
            "Same-family FFS pseudo-GT engineering evidence only. Missing seeds, "
            "ineligible evaluations, or missing paired per-record data fail closed. "
            "Bootstrap units are record IDs; overlapping temporal windows are not "
            "claimed to be independent video-level samples, and native B0/B1 pixel "
            "support may differ within an eligible record."
        ),
        "inputs": {
            "manifest_path": str(manifest_file),
            "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        },
    }
    # A partial component result remains visible, but an incomplete formal
    # experiment never becomes a recommended recipe.
    if input_errors and report["decision"] == "GO":
        report["decision"] = "NO-GO"
    if report["decision"] != "GO":
        report["recommended_switches"] = {
            "use_rays": False,
            "use_stereo_pose": False,
            "use_temporal_pose": False,
        }
        report["promotion_recipe"] = {
            "exact_evaluated_arm": None,
            "switches": report["recommended_switches"],
            "claim_boundary": (
                "Formal evidence is incomplete or rays did not pass; no learned "
                "calibration-conditioning recipe is promoted."
            ),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-random-seed", type=int, default=20260901)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = decide_manifest(
        args.manifest,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_random_seed=args.bootstrap_random_seed,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["decision"] == "GO":
        return 0
    if report["decision"] == "NOT_IDENTIFIABLE":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "COMPONENT",
    "DecisionInputError",
    "EXPECTED_SEEDS",
    "decide_manifest",
]
