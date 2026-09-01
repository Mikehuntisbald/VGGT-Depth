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
from collections import defaultdict
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

# ``component`` remains the stable decision-input schema identifier for both
# lineages.  Corrected v3.1 manifests additionally carry these identities so a
# copied original-v3 metrics bundle cannot be relabelled as v3.1 at decision
# time.  A manifest with none of the three lineage fields is the immutable
# pre-v3.1 compatibility form and is interpreted as legacy v3.
LINEAGE_MANIFEST_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    "v3": {
        "lineage_component": "v3-experiment-decision-inputs",
        "output_component": "v3-experiment-orchestrator",
    },
    "v3_1": {
        "lineage_component": "v3.1-experiment-decision-inputs",
        "output_component": "v3.1-experiment-orchestrator",
    },
}
V31_PIXEL_CENTER_CONTRACT = "align_corners_false_half_pixel_v3_1"
V31_MEASUREMENT_CONTRACT: Mapping[str, Any] = {
    "enabled": True,
    "protocol_version": "lr_center_projection_bounded_subpixel_v3_1",
    "minimum_subpixel_residual_hr_px": 1.0,
    "maximum_subpixel_residual_hr_px": 8.0,
    "boundary_relative_scale": 0.10,
}
V31_CANDIDATE_CONTRACT: Mapping[str, Any] = {
    "enabled": True,
    "protocol_version": "current_conditioned_age_phase_diverse_v3_1",
    "per_age_quota": 2,
    "surface_depth_gap_m": 0.05,
    "surface_relative_depth_gap": 0.05,
    "phase_redundancy_sigma_grid_px": 0.125,
    "phase_redundancy_penalty": 0.25,
}
V31_TOPK_DIAGNOSTICS = (
    "unique_age_fraction",
    "age2_survival_rate",
    "fractional_phase_variance",
    "attended_fractional_phase_variance",
    "topk_weight_entropy",
    "context_attention_weight_entropy",
    "metric_attention_weight_entropy",
    "candidate_depth_spread_m",
    "rank0_disparity_epe_hr_px",
    "weighted_disparity_epe_hr_px",
    "weighted_minus_rank0_epe_hr_px",
    "attention_weighted_disparity_epe_hr_px",
    "attention_weighted_minus_rank0_epe_hr_px",
)

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
    derived_lineage: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class PairedRecordDeltas:
    values: Mapping[tuple[str, int], float]
    eligible_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class ValidationRecordIdentity:
    sequence_id: str
    frame_id: int
    timestamp: float
    manifest_index: int

    @property
    def record_id(self) -> str:
        return f"{self.sequence_id}/{self.frame_id}"

    @property
    def cluster_key(self) -> tuple[str, int]:
        return (self.sequence_id, self.frame_id)


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


def _manifest_lineage(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> str:
    """Validate and return the concrete experiment lineage.

    Original v3 decision manifests predate explicit lineage fields.  That exact
    all-absent form remains supported.  Once any lineage field is present, all
    three fields are mandatory and must bind the decision-input file to its
    orchestrator output root.
    """

    lineage_fields = ("lineage", "lineage_component", "output_identity")
    present = tuple(name in manifest for name in lineage_fields)
    if not any(present):
        return "v3"
    if not all(present):
        raise DecisionInputError(
            "decision manifest lineage, lineage_component, and output_identity "
            "must be supplied together"
        )
    lineage = manifest.get("lineage")
    if not isinstance(lineage, str) or lineage not in LINEAGE_MANIFEST_CONTRACTS:
        raise DecisionInputError("decision manifest lineage is unsupported")
    expected = LINEAGE_MANIFEST_CONTRACTS[lineage]
    if manifest.get("lineage_component") != expected["lineage_component"]:
        raise DecisionInputError("decision manifest lineage_component mismatch")
    output_identity = manifest.get("output_identity")
    if not isinstance(output_identity, Mapping) or set(output_identity) != {
        "component",
        "lineage",
        "path",
    }:
        raise DecisionInputError("decision manifest output_identity is malformed")
    output_path = output_identity.get("path")
    if (
        output_identity.get("component") != expected["output_component"]
        or output_identity.get("lineage") != lineage
        or not isinstance(output_path, str)
        or Path(output_path).expanduser().resolve() != manifest_path.parent
    ):
        raise DecisionInputError("decision manifest output_identity mismatch")
    return lineage


def _contract_value_matches(actual: Any, expected: Any) -> bool:
    """Use strict bool/string identity and finite numeric equality."""

    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and float(actual) == float(expected)
        )
    return type(actual) is type(expected) and actual == expected


def _require_resolved_v31_contract(
    resolved: Mapping[str, Any], *, arm: str
) -> None:
    """Reject an arm whose resolved config is not the exact v3.1 recipe."""

    calibration = resolved.get("calibration_conditioning_v3")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("pixel_center_contract")
        != V31_PIXEL_CENTER_CONTRACT
    ):
        raise DecisionInputError(f"{arm} v3.1 pixel-center contract differs")
    for section_name, expected_contract in (
        ("measurement_ownership_v3_1", V31_MEASUREMENT_CONTRACT),
        ("temporal_candidate_fusion_v3_1", V31_CANDIDATE_CONTRACT),
    ):
        section = resolved.get(section_name)
        if not isinstance(section, Mapping) or any(
            not _contract_value_matches(section.get(name), expected)
            for name, expected in expected_contract.items()
        ):
            raise DecisionInputError(f"{arm} {section_name} contract differs")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_checkpoint_binding(
    entry: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    arm: str,
    stage_a: bool,
    lineage: str,
) -> None:
    """Bind decision evidence to the selected primary and Stage-A checkpoints."""

    explicit_primary = "checkpoint_sha256" in entry
    if lineage == "v3_1" or explicit_primary:
        checkpoint_sha256 = entry.get("checkpoint_sha256")
        checkpoint = report.get("checkpoint")
        if (
            not _is_sha256(checkpoint_sha256)
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise DecisionInputError(f"{arm} checkpoint SHA-256 binding differs")

    explicit_spatial = "spatial_checkpoint_sha256" in entry
    if stage_a:
        if (lineage == "v3_1" and not explicit_spatial) or (
            explicit_spatial and entry.get("spatial_checkpoint_sha256") is not None
        ):
            raise DecisionInputError(
                f"{arm} Stage-A spatial checkpoint binding must be null"
            )
        return
    if lineage == "v3_1" or explicit_spatial:
        spatial_sha256 = entry.get("spatial_checkpoint_sha256")
        spatial_checkpoint = report.get("spatial_checkpoint")
        if (
            not _is_sha256(spatial_sha256)
            or not isinstance(spatial_checkpoint, Mapping)
            or spatial_checkpoint.get("checkpoint_sha256") != spatial_sha256
        ):
            raise DecisionInputError(
                f"{arm} spatial checkpoint SHA-256 binding differs"
            )


def _require_v31_topk_interpretation(
    report: Mapping[str, Any],
    records_path: Path,
    *,
    arm: str,
    expected_count: int,
) -> None:
    """Require finite aggregate and per-record top-K interpretation evidence.

    These fields explain whether temporal gains used diverse candidates.  They
    are availability/audit gates only: no diagnostic value is compared against
    a promotion threshold here.
    """

    diagnostics = report.get("diagnostics")
    topk = (
        diagnostics.get("topk_candidate_complementarity_v3_1")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(topk, Mapping):
        raise DecisionInputError(f"{arm} v3.1 top-K diagnostics are missing")
    for name in V31_TOPK_DIAGNOSTICS:
        metric = topk.get(name)
        if not isinstance(metric, Mapping) or metric.get("valid") is not True:
            raise DecisionInputError(
                f"{arm} aggregate topk_{name} diagnostic is invalid"
            )
        count = metric.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise DecisionInputError(
                f"{arm} aggregate topk_{name} diagnostic count is invalid"
            )
        _finite_number(metric.get("value"), f"{arm}.diagnostics.{name}.value")
        _finite_number(
            metric.get("numerator"), f"{arm}.diagnostics.{name}.numerator"
        )

    if not records_path.is_file():
        raise DecisionInputError(f"{arm} v3.1 per-record top-K diagnostics are missing")
    try:
        lines = records_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DecisionInputError(
            f"cannot read {arm} per-record top-K diagnostics"
        ) from exc
    if len(lines) != expected_count or any(not line.strip() for line in lines):
        raise DecisionInputError(
            f"{arm} per-record top-K diagnostic coverage differs"
        )
    valid_counts = {name: 0 for name in V31_TOPK_DIAGNOSTICS}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionInputError(
                f"malformed {arm} per-record row {line_number}"
            ) from exc
        metrics = row.get("metrics") if isinstance(row, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise DecisionInputError(
                f"{arm} per-record metrics are malformed on row {line_number}"
            )
        for name in V31_TOPK_DIAGNOSTICS:
            field = f"topk_{name}"
            if field not in metrics:
                raise DecisionInputError(
                    f"{arm} per-record {field} is missing on row {line_number}"
                )
            if metrics[field] is None:
                continue
            _finite_number(
                metrics[field], f"{arm}.per_record[{line_number}].{field}"
            )
            valid_counts[name] += 1
    missing = sorted(name for name, count in valid_counts.items() if count <= 0)
    if missing:
        raise DecisionInputError(
            f"{arm} per-record top-K diagnostics have no valid values: {missing}"
        )


def _validation_record_sets(
    path: Path,
) -> tuple[
    Mapping[tuple[str, int], ValidationRecordIdentity],
    Mapping[tuple[str, int], ValidationRecordIdentity],
]:
    """Load exact Stage-A records and formal T=3/five-context endpoints."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DecisionInputError(f"cannot read validation manifest: {path}") from exc
    if len(lines) != STAGE_A_RECORDS or any(not line.strip() for line in lines):
        raise DecisionInputError(
            f"formal validation manifest must have {STAGE_A_RECORDS} nonblank rows"
        )
    stage_a: dict[tuple[str, int], ValidationRecordIdentity] = {}
    by_sequence: dict[str, list[ValidationRecordIdentity]] = defaultdict(list)
    last_timestamp: dict[str, float] = {}
    for manifest_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionInputError(
                f"malformed validation manifest row {manifest_index + 1}"
            ) from exc
        if not isinstance(row, Mapping):
            raise DecisionInputError(
                f"validation manifest row {manifest_index + 1} is not an object"
            )
        sequence_id = row.get("sequence_id")
        frame_id = row.get("frame_id")
        timestamp = row.get("timestamp")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
        ):
            raise DecisionInputError(
                f"validation identity is malformed on row {manifest_index + 1}"
            )
        timestamp_float = _finite_number(
            timestamp, f"validation[{manifest_index}].timestamp"
        )
        previous = last_timestamp.get(sequence_id)
        if previous is not None and timestamp_float <= previous:
            raise DecisionInputError(
                f"validation timestamps are not increasing for {sequence_id!r}"
            )
        identity = ValidationRecordIdentity(
            sequence_id=sequence_id,
            frame_id=frame_id,
            timestamp=timestamp_float,
            manifest_index=manifest_index,
        )
        if identity.cluster_key in stage_a:
            raise DecisionInputError(
                f"duplicate validation record {identity.cluster_key!r}"
            )
        stage_a[identity.cluster_key] = identity
        by_sequence[sequence_id].append(identity)
        last_timestamp[sequence_id] = timestamp_float

    # The calibrated derived cache covers endpoints with five-view context
    # (positions >=4). The T=3 dataset additionally requires the preceding two
    # student endpoints to have derived entries, so formal scored positions are
    # >=6 in each sequence.
    stage_b: dict[tuple[str, int], ValidationRecordIdentity] = {}
    for records in by_sequence.values():
        for identity in records[6:]:
            stage_b[identity.cluster_key] = identity
    if len(stage_b) != STAGE_B_RECORDS:
        raise DecisionInputError(
            f"formal validation manifest yields {len(stage_b)} T=3 endpoints, "
            f"expected {STAGE_B_RECORDS}"
        )
    return stage_a, stage_b


def _validate_hash_bound_temporal_audit(
    identifiability: Mapping[str, Any] | None,
    *,
    validation_manifest_path: Path,
    validation_manifest_sha256: str,
    formal_records: Mapping[tuple[str, int], ValidationRecordIdentity],
    evidence_by_seed: Mapping[int, Mapping[str, ArmEvidence]],
) -> Mapping[str, Any] | None:
    """Require a live, exact audit chain; a bare boolean is never evidence."""

    if not isinstance(identifiability, Mapping) or (
        identifiability.get("temporal_pose_varies") is not True
    ):
        return None
    identity = identifiability.get("temporal_pose_variation_audit")
    if not isinstance(identity, Mapping):
        return None
    path_value = identity.get("path")
    declared_sha256 = identity.get("sha256")
    if (
        not isinstance(path_value, str)
        or not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
    ):
        raise DecisionInputError("temporal-pose audit identity is malformed")
    audit_path = Path(path_value).expanduser().resolve()
    audit = _load_json(
        audit_path,
        "temporal-pose variation audit",
        expected_sha256=declared_sha256,
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("component") != "v3-temporal-pose-variation-audit"
        or audit.get("status") != "PASS"
        or audit.get("temporal_pose_varies") is not True
    ):
        raise DecisionInputError("temporal-pose audit is not a PASS v1 receipt")
    ages = audit.get("ages")
    if not isinstance(ages, Mapping) or any(
        not isinstance(ages.get(str(age)), Mapping)
        or ages[str(age)].get("varies") is not True
        for age in (1, 2)
    ):
        raise DecisionInputError("temporal-pose audit does not pass both ages")
    inputs = audit.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DecisionInputError("temporal-pose audit input lineage is missing")
    audit_manifest = inputs.get("validation_manifest")
    if (
        not isinstance(audit_manifest, Mapping)
        or Path(str(audit_manifest.get("path"))).expanduser().resolve()
        != validation_manifest_path
        or audit_manifest.get("sha256") != validation_manifest_sha256
        or audit_manifest.get("records") != STAGE_A_RECORDS
    ):
        raise DecisionInputError("temporal-pose audit validation manifest differs")
    derived_root = Path(str(inputs.get("derived_root"))).expanduser().resolve()
    receipt_path = derived_root / "run_receipt.json"
    cache_manifest_path = derived_root / "cache_manifest.jsonl"
    if (
        not receipt_path.is_file()
        or not cache_manifest_path.is_file()
        or inputs.get("run_receipt_sha256")
        != hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        or inputs.get("cache_manifest_sha256")
        != hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest()
    ):
        raise DecisionInputError("temporal-pose audit derived inputs changed")
    receipt = _load_json(receipt_path, "audited derived receipt")
    output = receipt.get("output")
    if not isinstance(output, Mapping) or output.get(
        "cache_manifest_sha256"
    ) != inputs.get("cache_manifest_sha256"):
        raise DecisionInputError("audited derived receipt does not bind cache manifest")

    binding = audit.get("formal_endpoint_binding")
    expected_ids = [
        identity.record_id
        for identity in sorted(
            formal_records.values(), key=lambda value: value.manifest_index
        )
    ]
    if not isinstance(binding, Mapping) or binding.get("available") is not True:
        raise DecisionInputError("temporal-pose audit lacks formal endpoint binding")
    record_ids = binding.get("record_ids")
    valid_ids = binding.get("pose_valid_record_ids")
    if (
        record_ids != expected_ids
        or binding.get("record_ids_sha256")
        != _canonical_sha256(expected_ids, "formal endpoint IDs")
        or not isinstance(valid_ids, list)
        or any(not isinstance(value, str) for value in valid_ids)
        or len(set(valid_ids)) != len(valid_ids)
        or not set(valid_ids).issubset(set(expected_ids))
        or len(valid_ids) < MIN_TEMPORAL_PAIRED_RECORDS
    ):
        raise DecisionInputError("temporal-pose audit formal endpoint identities differ")
    counts = audit.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("formal_temporal_endpoints") != STAGE_B_RECORDS
        or counts.get("formal_windows") != STAGE_B_RECORDS
        or counts.get("formal_pose_valid_windows") != len(valid_ids)
    ):
        raise DecisionInputError("temporal-pose audit formal endpoint counts differ")

    for seed, arms in evidence_by_seed.items():
        for arm_name in ("B0", "B1"):
            evidence = arms.get(arm_name)
            if evidence is None:
                continue
            lineage = evidence.derived_lineage
            if (
                not isinstance(lineage, Mapping)
                or Path(str(lineage.get("derived_cache_root"))).expanduser().resolve()
                != derived_root
                or lineage.get("run_receipt_sha256")
                != inputs.get("run_receipt_sha256")
                or lineage.get("cache_manifest_sha256")
                != inputs.get("cache_manifest_sha256")
            ):
                raise DecisionInputError(
                    f"seed {seed} {arm_name} evaluated derived lineage differs from audit"
                )
    return {
        "path": str(audit_path),
        "sha256": declared_sha256,
        "derived_root": str(derived_root),
        "run_receipt_sha256": inputs["run_receipt_sha256"],
        "cache_manifest_sha256": inputs["cache_manifest_sha256"],
        "formal_endpoint_ids_sha256": binding["record_ids_sha256"],
        "formal_pose_valid_windows": len(valid_ids),
    }


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
    lineage: str,
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
    _require_checkpoint_binding(
        entry,
        report,
        arm=arm,
        stage_a=stage_a,
        lineage=lineage,
    )
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
    if lineage == "v3_1":
        _require_resolved_v31_contract(resolved, arm=arm)
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
    per_record_receipt = report.get("per_record_metrics")
    if not isinstance(per_record_receipt, Mapping):
        raise DecisionInputError(f"{arm} metrics report lacks per-record binding")
    receipt_path_value = per_record_receipt.get("path")
    if (
        not isinstance(receipt_path_value, str)
        or Path(receipt_path_value).expanduser().resolve() != records_path
        or per_record_receipt.get("sha256") != records_sha256
        or per_record_receipt.get("records") != expected_count
        or per_record_receipt.get("paired_bootstrap_unit")
        != "sequence_id/frame_id"
    ):
        raise DecisionInputError(f"{arm} metrics/per-record binding differs")
    if lineage == "v3_1" and not stage_a:
        _require_v31_topk_interpretation(
            report,
            records_path,
            arm=arm,
            expected_count=expected_count,
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
        normalized_derived_lineage = None
    else:
        if not isinstance(derived_lineage, Mapping):
            raise DecisionInputError(f"{arm} derived validation lineage is missing")
        normalized_derived_lineage = dict(derived_lineage)
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
        derived_lineage=normalized_derived_lineage,
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
        candidate_value = candidate.metrics[metric]
        checks[f"{metric}_absolute"] = candidate_value < MAX_OUTPUT_BAD_RATE
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
    path: Path,
    *,
    expected_records: Mapping[tuple[str, int], ValidationRecordIdentity],
    expected_method: str,
    expected_sha256: str | None,
) -> dict[tuple[str, int], Mapping[str, float | None]]:
    if not path.is_file():
        raise DecisionInputError(f"per-record bootstrap data is unavailable: {path}")
    records: dict[tuple[str, int], Mapping[str, float | None]] = {}
    try:
        raw = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is None or actual_sha256 != expected_sha256:
            raise DecisionInputError(f"per-record SHA-256 differs: {path}")
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DecisionInputError(f"cannot read per-record data: {path}") from exc
    if len(lines) != len(expected_records):
        raise DecisionInputError(
            f"per-record data must have {len(expected_records)} rows, got {len(lines)}: {path}"
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
        sequence_id = row.get("sequence_id")
        frame_id = row.get("frame_id")
        timestamp = row.get("timestamp")
        manifest_index = row.get("manifest_index")
        method = row.get("method")
        metrics = row.get("metrics")
        if (
            not isinstance(sequence_id, str)
            or isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
        ):
            raise DecisionInputError(f"record identity is malformed: {path}:{line_number}")
        cluster_key = (sequence_id, frame_id)
        expected = expected_records.get(cluster_key)
        if expected is None:
            raise DecisionInputError(
                f"record is outside exact validation selection: {cluster_key!r}"
            )
        if (
            record_id != expected.record_id
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isclose(
                _finite_number(timestamp, f"{path}:{line_number}.timestamp"),
                expected.timestamp,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or manifest_index != expected.manifest_index
            or method != expected_method
        ):
            raise DecisionInputError(
                f"record metadata differs from validation manifest: {path}:{line_number}"
            )
        if cluster_key in records:
            raise DecisionInputError(f"duplicate record identity {cluster_key!r}: {path}")
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
        records[cluster_key] = normalized
    if set(records) != set(expected_records):
        raise DecisionInputError(f"per-record identities do not cover exact selection: {path}")
    return records


def _paired_deltas(
    control_path: Path,
    candidate_path: Path,
    *,
    metric: str,
    expected_records: Mapping[tuple[str, int], ValidationRecordIdentity],
    expected_method: str,
    allow_invalid_intersection: bool,
    control_sha256: str | None,
    candidate_sha256: str | None,
) -> PairedRecordDeltas:
    control = _load_record_metrics(
        control_path,
        expected_records=expected_records,
        expected_method=expected_method,
        expected_sha256=control_sha256,
    )
    candidate = _load_record_metrics(
        candidate_path,
        expected_records=expected_records,
        expected_method=expected_method,
        expected_sha256=candidate_sha256,
    )
    if set(control) != set(candidate):
        raise DecisionInputError(
            f"paired record IDs differ: {control_path} vs {candidate_path}"
        )
    deltas: dict[tuple[str, int], float] = {}
    for cluster_key in sorted(control):
        if metric not in control[cluster_key] or metric not in candidate[cluster_key]:
            if allow_invalid_intersection:
                # The evaluator may omit an invalid metric instead of writing
                # an explicit null/valid=false payload for a safe-mask-empty
                # window. Both representations mean ineligible for this pair.
                continue
            raise DecisionInputError(
                f"paired metric {metric!r} is missing for record {cluster_key!r}"
            )
        control_value = control[cluster_key][metric]
        candidate_value = candidate[cluster_key][metric]
        if control_value is None or candidate_value is None:
            if allow_invalid_intersection:
                continue
            raise DecisionInputError(
                f"paired metric {metric!r} is invalid for record {cluster_key!r}"
            )
        deltas[cluster_key] = candidate_value - control_value
    return PairedRecordDeltas(
        values=deltas,
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
    deltas_by_seed: Mapping[int, Mapping[tuple[str, int], float]],
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
    common_clusters = set.intersection(
        *(set(values) for values in deltas_by_seed.values())
    )
    if not common_clusters:
        raise DecisionInputError("bootstrap has no common sequence/source-frame clusters")
    clusters_by_sequence: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cluster in sorted(common_clusters):
        clusters_by_sequence[cluster[0]].append(cluster)
    sequence_ids = sorted(clusters_by_sequence)
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(replicates):
        # Draw one common hierarchical cluster sample for every fixed seed
        # stratum: sequences first, then source frames within each sequence.
        # The same keys across seeds preserve paired treatment effects.
        sampled_clusters: list[tuple[str, int]] = []
        for _sequence_draw in sequence_ids:
            sequence_id = rng.choice(sequence_ids)
            source_frames = clusters_by_sequence[sequence_id]
            sampled_clusters.extend(
                rng.choice(source_frames) for _ in range(len(source_frames))
            )
        seed_means = [
            sum(deltas_by_seed[seed][cluster] for cluster in sampled_clusters)
            / len(sampled_clusters)
            for seed in EXPECTED_SEEDS
        ]
        estimates.append(sum(seed_means) / len(seed_means))
    estimates.sort()
    observed_seed_means = [
        sum(deltas_by_seed[seed][cluster] for cluster in common_clusters)
        / len(common_clusters)
        for seed in EXPECTED_SEEDS
    ]
    return {
        "replicates": replicates,
        "random_seed": random_seed,
        "observed_mean_delta": sum(observed_seed_means) / len(observed_seed_means),
        "cluster_unit": "sequence_id/source_frame_id",
        "cluster_sampling": "joint_across_fixed_seed_strata; sequence_then_source_frame",
        "common_clusters": len(common_clusters),
        "sequences": len(sequence_ids),
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
    expected_records: Mapping[tuple[str, int], ValidationRecordIdentity],
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
    deltas_by_seed: dict[int, Mapping[tuple[str, int], float]] = {}
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
                expected_records=expected_records,
                expected_method=arms[control_arm].method,
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
    common_clusters = set.intersection(
        *(set(values) for values in deltas_by_seed.values())
    )
    common_fraction = len(common_clusters) / len(expected_records)
    if allow_invalid_record_intersection and (
        len(common_clusters) < MIN_TEMPORAL_PAIRED_RECORDS
        or common_fraction < MIN_TEMPORAL_PAIRED_FRACTION
    ):
        return {
            "status": "NO-GO",
            "reasons": [
                "three-seed native temporal cluster intersection has "
                f"{len(common_clusters)}/{len(expected_records)} eligible records; "
                f"requires >= {MIN_TEMPORAL_PAIRED_RECORDS} and >= "
                f"{MIN_TEMPORAL_PAIRED_FRACTION:.0%}"
            ],
            "aggregate_by_seed": aggregate_by_seed,
            "paired_coverage_by_seed": paired_coverage_by_seed,
            "common_cluster_coverage": {
                "eligible_records": len(common_clusters),
                "total_records": len(expected_records),
                "eligible_fraction": common_fraction,
            },
            "fail_closed": True,
        }
    seed_mean_deltas = {
        str(seed): (
            sum(values[cluster] for cluster in common_clusters)
            / len(common_clusters)
        )
        for seed, values in deltas_by_seed.items()
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
        "common_cluster_coverage": {
            "eligible_records": len(common_clusters),
            "total_records": len(expected_records),
            "eligible_fraction": common_fraction,
        },
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
    lineage = _manifest_lineage(manifest, manifest_path=manifest_file)
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
    stage_a_records, stage_b_records = _validation_record_sets(
        validation_manifest_path
    )
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
                    lineage=lineage,
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
        expected_records=stage_a_records,
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
            expected_records=stage_a_records,
            require_t3_gate=False,
            allow_invalid_record_intersection=False,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_random_seed=bootstrap_random_seed + 1,
        )
    temporal_audit_identity: Mapping[str, Any] | None = None
    try:
        temporal_audit_identity = _validate_hash_bound_temporal_audit(
            identifiability if isinstance(identifiability, Mapping) else None,
            validation_manifest_path=validation_manifest_path,
            validation_manifest_sha256=validation_manifest_sha256,
            formal_records=stage_b_records,
            evidence_by_seed=evidence_by_seed,
        )
    except DecisionInputError as exc:
        input_errors.append(str(exc))
    temporal_identifiable = temporal_audit_identity is not None
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
            expected_records=stage_b_records,
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
        "lineage": lineage,
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
            "The bootstrap jointly draws the same sequence/source-frame clusters "
            "across fixed seed strata. Overlapping windows remain engineering "
            "evidence rather than independent paper-level video samples, and native "
            "B0/B1 pixel support may differ within an eligible record."
        ),
        "inputs": {
            "manifest_path": str(manifest_file),
            "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
            "temporal_pose_variation_audit": temporal_audit_identity,
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
