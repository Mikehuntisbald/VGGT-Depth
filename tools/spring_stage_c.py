#!/usr/bin/env python3
"""Spring-specific Stage-C contracts.

The canonical :mod:`eval_epipolar` path is intentionally bound to the
published 244/240/238 validation corpus.  Spring is a separate, sequence-
disjoint screening protocol and must not weaken that canonical gate.  This
module contains only the *bounded Spring* contracts used by the two thin
Spring train/eval entry points:

* complete causal coverage is checked against the supplied Spring manifest;
* the exact train/validation sequence split and cache identities remain
  bound; and
* a pixel rectification receipt is bound to both supplied manifests without
  assuming the canonical frame/count/hash values.

No function here grants formal/canonical acceptance.  Every report produced
through this path is explicitly screening-only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import sha256_file
from data.manifest import load_manifest
from data.training_dataset import build_causal_windows


SPRING_STAGE_C_PROTOCOL = "spring_stage_c_sequence_screening_v1"
SPRING_STAGE_C_REPORT_STAGE = "SPRING_STAGE_C_EPIPOLAR_SCREENING"


class SpringStageCError(ValueError):
    """Raised when a Spring Stage-C contract cannot be proven."""


def _as_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpringStageCError(f"{name} must be a lowercase SHA-256")
    return value


def validate_spring_manifest(path: str | Path, *, expected_split: str | None = None) -> dict[str, Any]:
    """Validate that a manifest is genuinely Spring and sequence ordered.

    The generic manifest loader validates calibration/source fields.  This
    additional check prevents accidentally feeding an unrelated manifest to
    the Spring evaluator merely because it happens to have a convenient
    filename.
    """

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = load_manifest(manifest)
    if not records:
        raise SpringStageCError(f"Spring manifest is empty: {manifest}")
    split_values = {str(record.extras.get("split", "")) for record in records}
    dataset_values = {str(record.extras.get("dataset", "")) for record in records}
    if dataset_values != {"spring"}:
        raise SpringStageCError(
            f"manifest is not exclusively Spring (dataset values={sorted(dataset_values)})"
        )
    if expected_split is not None and split_values != {str(expected_split)}:
        raise SpringStageCError(
            f"manifest split differs from expected {expected_split!r}: {sorted(split_values)}"
        )
    seen: set[tuple[str, int]] = set()
    by_sequence: dict[str, list[tuple[float, int]]] = {}
    for record in records:
        key = (record.sequence_id, int(record.frame_id))
        if key in seen:
            raise SpringStageCError(f"duplicate Spring manifest record: {key}")
        seen.add(key)
        by_sequence.setdefault(record.sequence_id, []).append(
            (float(record.timestamp), int(record.frame_id))
        )
    for sequence_id, values in by_sequence.items():
        if any(
            current[0] <= previous[0] or current[1] <= previous[1]
            for previous, current in zip(values, values[1:])
        ):
            raise SpringStageCError(
                f"Spring sequence {sequence_id!r} is not strictly ordered"
            )
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "records": len(records),
        "sequences": sorted(by_sequence),
        "sequence_count": len(by_sequence),
        "dataset": "spring",
        "split_values": sorted(split_values),
    }


def spring_temporal_coverage(dataset: Any) -> dict[str, Any]:
    """Require complete causal coverage for the *supplied* Spring manifest.

    ``CachedTemporalTrainingDataset`` already filters windows whose three
    student endpoints have derived geometry.  We independently reconstruct
    the expected set here, so a subset-derived cache cannot silently look like
    a valid screening run.  Unlike the canonical evaluator, no fixed 244/240/
    238 count or validation-manifest SHA is assumed.
    """

    records = getattr(dataset, "records", None)
    derived_entries = getattr(dataset, "derived_entries", None)
    windows = getattr(dataset, "windows", None)
    derived_root = Path(getattr(dataset, "derived_cache_root")).expanduser().resolve()
    manifest_path = Path(getattr(dataset, "manifest_path")).expanduser().resolve()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise SpringStageCError("Spring temporal dataset has no records")
    if not isinstance(derived_entries, Mapping) or not derived_entries:
        raise SpringStageCError("Spring derived geometry entries are empty")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise SpringStageCError("Spring temporal dataset has no causal T=3 windows")
    validate_spring_manifest(manifest_path)

    candidates = build_causal_windows(
        records,
        student_sequence_length=3,
        vggt_context_pairs=5,
    )
    expected_endpoints = {int(window.endpoint_index) for window in candidates}
    actual_endpoints = {int(index) for index in derived_entries}
    if actual_endpoints != expected_endpoints:
        missing = sorted(expected_endpoints - actual_endpoints)
        extra = sorted(actual_endpoints - expected_endpoints)
        raise SpringStageCError(
            "Spring derived endpoint coverage is incomplete: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    expected_windows = [
        (int(window.endpoint_index), tuple(int(index) for index in window.student_indices))
        for window in candidates
        if all(int(index) in actual_endpoints for index in window.student_indices)
    ]
    actual_windows = [
        (int(window.endpoint_index), tuple(int(index) for index in window.student_indices))
        for window in windows
    ]
    if actual_windows != expected_windows:
        raise SpringStageCError(
            "Spring temporal window set is not the complete causal set"
        )

    receipt_path = derived_root / "run_receipt.json"
    manifest_cache_path = derived_root / "cache_manifest.jsonl"
    if not receipt_path.is_file() or not manifest_cache_path.is_file():
        raise SpringStageCError(
            f"Spring derived receipt/manifest is missing under {derived_root}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpringStageCError(f"cannot read Spring derived receipt: {receipt_path}") from exc
    if not isinstance(receipt, Mapping):
        raise SpringStageCError("Spring derived receipt is not a mapping")
    selection = receipt.get("selection")
    counts = receipt.get("counts")
    if not isinstance(selection, Mapping) or not isinstance(counts, Mapping):
        raise SpringStageCError("Spring derived receipt coverage fields are missing")
    if selection.get("start_window") != 0 or selection.get("limit") is not None:
        raise SpringStageCError(
            "Spring Stage-C refuses a subset-derived geometry receipt; rebuild all windows"
        )
    if selection.get("selected_windows") != len(actual_endpoints) or counts.get(
        "selected"
    ) != len(actual_endpoints):
        raise SpringStageCError("Spring derived receipt does not cover every endpoint")
    raw_inputs = receipt.get("inputs")
    raw_manifest_value = (
        raw_inputs.get("vggt_cache_manifest")
        if isinstance(raw_inputs, Mapping)
        else None
    )
    raw_manifest_sha = (
        raw_inputs.get("vggt_cache_manifest_sha256")
        if isinstance(raw_inputs, Mapping)
        else None
    )
    if not isinstance(raw_manifest_value, str):
        raise SpringStageCError("Spring derived receipt is not bound to raw VGGT manifest")
    raw_manifest = Path(raw_manifest_value).expanduser().resolve()
    if not raw_manifest.is_file() or sha256_file(raw_manifest) != _as_sha(
        raw_manifest_sha, "vggt_cache_manifest_sha256"
    ):
        raise SpringStageCError("Spring derived/raw VGGT manifest SHA-256 mismatch")
    return {
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "canonical": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_records": len(records),
        "sequence_count": len({record.sequence_id for record in records}),
        "derived_endpoint_records": len(actual_endpoints),
        "evaluable_t3_windows": len(actual_windows),
        "derived_run_receipt_path": str(receipt_path),
        "derived_run_receipt_sha256": sha256_file(receipt_path),
        "derived_cache_manifest_path": str(manifest_cache_path),
        "derived_cache_manifest_sha256": sha256_file(manifest_cache_path),
        "raw_vggt_cache_manifest_path": str(raw_manifest),
        "raw_vggt_cache_manifest_sha256": raw_manifest_sha,
    }


def require_spring_stage_c_coverage(
    coverage: Mapping[str, Any], *, manifest_sha256: str
) -> dict[str, Any]:
    """Validate the compact coverage object returned above.

    This separate function mirrors the canonical evaluator's two-step API and
    makes accidental substitution of canonical constants easy to test.
    """

    if coverage.get("protocol") != SPRING_STAGE_C_PROTOCOL:
        raise SpringStageCError("coverage is not the Spring Stage-C protocol")
    if coverage.get("canonical") is not False:
        raise SpringStageCError("Spring Stage-C coverage cannot be canonical")
    for name in (
        "manifest_records",
        "derived_endpoint_records",
        "evaluable_t3_windows",
    ):
        value = coverage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SpringStageCError(f"Spring coverage field is malformed: {name}")
    if coverage.get("manifest_sha256") != manifest_sha256:
        raise SpringStageCError("Spring coverage manifest SHA-256 differs")
    return dict(coverage)


def validate_spring_rectification_binding(
    *,
    stage_c_metadata: Mapping[str, Any],
    receipt_path: str | Path,
    validation_manifest_sha256: str,
    generic_validator: Any,
) -> dict[str, Any]:
    """Bind a generic pixel audit to Spring train/validation manifests.

    ``generic_validator`` is ``train_epipolar._validated_rectification_audit``;
    using it preserves the strict schema/threshold checks while this function
    deliberately omits the canonical receipt hash/count gate.
    """

    config = stage_c_metadata.get("config")
    data = config.get("data") if isinstance(config, Mapping) else None
    recorded = stage_c_metadata.get("rectification_audit")
    if not isinstance(data, Mapping) or not isinstance(recorded, Mapping):
        raise SpringStageCError("Spring rectification audit lineage is missing")
    train_manifest_value = data.get("manifest_path")
    if not isinstance(train_manifest_value, str):
        raise SpringStageCError("Stage-C training manifest path is missing")
    train_manifest = Path(train_manifest_value).expanduser().resolve()
    train_info = validate_spring_manifest(train_manifest)
    current = generic_validator(
        receipt_path,
        expected_train_manifest_sha256=train_info["sha256"],
        allow_consistent_metadata=True,
    )
    manifest_hashes = current.get("manifest_sha256")
    if not isinstance(manifest_hashes, Mapping):
        raise SpringStageCError("rectification audit manifest binding is malformed")
    _as_sha(manifest_hashes.get("train"), "rectification train manifest SHA")
    if manifest_hashes.get("validation") != validation_manifest_sha256:
        raise SpringStageCError(
            "rectification audit validation manifest SHA differs from Spring evaluation"
        )
    if manifest_hashes.get("train") != train_info["sha256"]:
        raise SpringStageCError("rectification audit train manifest SHA differs")
    recorded_content = {name: value for name, value in recorded.items() if name != "path"}
    current_content = {name: value for name, value in current.items() if name != "path"}
    # The checkpoint stores the receipt's compact content, including its own
    # file hash.  It must match exactly; otherwise a changed audit could be
    # swapped in between training and evaluation.
    if recorded_content != current_content:
        raise SpringStageCError(
            "current Spring rectification audit differs from the checkpoint"
        )
    configured = data.get("epipolar_rectification_audit")
    configured_path = data.get("epipolar_rectification_audit_path")
    if configured != recorded or not isinstance(configured_path, str):
        raise SpringStageCError("Stage-C config is not bound to the Spring audit")
    if Path(configured_path).expanduser().resolve() != Path(
        str(recorded.get("path"))
    ).expanduser().resolve():
        raise SpringStageCError("Stage-C audit path differs from its recorded path")
    return {
        **current,
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "canonical": False,
        "checkpoint_recorded_path": str(recorded.get("path")),
        "current_verified_path": str(current.get("path")),
        "train_manifest": train_info,
    }


def strict_spring_holdout_lineage(original_validator: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the shared lineage auditor with no smoke-overlap bypass."""

    result = original_validator(allow_non_holdout_smoke=False, **kwargs)
    if result.get("sequence_overlap"):
        raise SpringStageCError(
            f"Spring Stage-C train/validation sequences overlap: {result['sequence_overlap']}"
        )
    if result.get("same_manifest"):
        raise SpringStageCError("Spring Stage-C evaluation reuses the training manifest")
    result = dict(result)
    result.update(
        {
            "protocol": SPRING_STAGE_C_PROTOCOL,
            "canonical": False,
            "formal_holdout": False,
            "spring_sequence_disjoint": True,
        }
    )
    return result


def sha256_path(path: str | Path) -> str:
    """Small public helper used when rewriting the wrapper report."""

    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def validate_spring_checkpoint_marker(
    checkpoint: str | Path,
    *,
    train_adapter_path: str | Path,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require the explicit Spring adapter marker embedded by training.

    The canonical Stage-C runtime bundle does not include this wrapper (by
    design, so canonical lineage remains unchanged).  Binding the wrapper's
    byte hash into the checkpoint config closes that otherwise easy-to-miss
    provenance gap: evaluating with a modified adapter fails before any model
    forward.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SpringStageCError("Spring Stage-C checkpoint is not a mapping")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise SpringStageCError("Spring Stage-C checkpoint config is missing")
    if config.get("spring_stage_c_protocol") != SPRING_STAGE_C_PROTOCOL:
        raise SpringStageCError(
            "checkpoint is not marked as the Spring Stage-C screening protocol"
        )
    train_adapter = Path(train_adapter_path).expanduser().resolve()
    expected = sha256_file(train_adapter)
    recorded = config.get("spring_stage_c_train_adapter_sha256")
    if recorded != expected:
        raise SpringStageCError(
            "Spring Stage-C training adapter SHA-256 differs from checkpoint"
        )
    result = {
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "train_adapter_path": str(train_adapter),
        "train_adapter_sha256": expected,
    }
    if contract_path is not None:
        contract = Path(contract_path).expanduser().resolve()
        expected_contract = sha256_file(contract)
        recorded_contract = config.get("spring_stage_c_contract_sha256")
        if recorded_contract != expected_contract:
            raise SpringStageCError(
                "Spring Stage-C contract SHA-256 differs from checkpoint"
            )
        result["contract_path"] = str(contract)
        result["contract_sha256"] = expected_contract
    return result


def audit_spring_screening_report(
    report: Mapping[str, Any],
    *,
    metrics_path: str | Path | None = None,
    validation_manifest: str | Path | None = None,
    checkpoint: str | Path | None = None,
    train_adapter_path: str | Path | None = None,
    auditor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only final gate for a Spring Stage-C screening report.

    This intentionally has no acceptance path: ``status=PASS`` means only
    that the bounded report is internally self-consistent and provenance is
    reproducible.  Callers may use it to distinguish a valid screening result
    from a stale/edited metrics file.
    """

    if not isinstance(report, Mapping):
        raise SpringStageCError("Spring metrics report must be a mapping")
    if report.get("stage") != SPRING_STAGE_C_REPORT_STAGE:
        raise SpringStageCError("metrics report is not Spring Stage-C screening")
    if report.get("status") != "SPRING_STAGE_C_SCREENING":
        raise SpringStageCError("metrics report status is not Spring screening")
    claims = report.get("claims")
    if not isinstance(claims, Mapping) or claims.get("spring_screening") is not True:
        raise SpringStageCError("Spring screening claim marker is missing")
    if claims.get("acceptance_eligible") is not False or claims.get(
        "formal_holdout"
    ) is not False:
        raise SpringStageCError("Spring screening report accidentally claims formal eligibility")
    screening = report.get("spring_screening")
    if not isinstance(screening, Mapping) or screening.get("protocol") != SPRING_STAGE_C_PROTOCOL:
        raise SpringStageCError("Spring screening protocol marker is malformed")
    if screening.get("canonical") is not False:
        raise SpringStageCError("Spring screening report cannot be canonical")
    limit = screening.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise SpringStageCError("Spring screening limit is malformed")
    coverage = report.get("formal_coverage")
    if not isinstance(coverage, Mapping):
        raise SpringStageCError("Spring coverage receipt is missing")
    coverage_sha = _as_sha(
        coverage.get("manifest_sha256"), "Spring coverage manifest SHA"
    )
    require_spring_stage_c_coverage(
        coverage,
        manifest_sha256=coverage_sha,
    )
    canonical_coverage = report.get("canonical_coverage")
    if isinstance(canonical_coverage, Mapping) and canonical_coverage.get(
        "canonical"
    ) is not False:
        raise SpringStageCError("canonical coverage field is unexpectedly eligible")
    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise SpringStageCError("Spring lineage section is missing")
    protocol_lineage = lineage.get("spring_protocol")
    if not isinstance(protocol_lineage, Mapping) or protocol_lineage.get(
        "protocol"
    ) != SPRING_STAGE_C_PROTOCOL:
        raise SpringStageCError("Spring lineage protocol marker is missing")
    held_out = lineage.get("held_out_validation")
    if not isinstance(held_out, Mapping):
        raise SpringStageCError("Spring held-out lineage is missing")
    if held_out.get("same_manifest") is not False or held_out.get(
        "sequence_overlap"
    ) not in ([], None):
        raise SpringStageCError("Spring train/validation sequence isolation failed")
    sources = report.get("source_hashes")
    if not isinstance(sources, Mapping):
        raise SpringStageCError("Spring source hashes are missing")
    evaluator_path = sources.get("evaluator_path")
    evaluator_sha = sources.get("evaluator_sha256")
    if not isinstance(evaluator_path, str) or not Path(evaluator_path).is_file():
        raise SpringStageCError("Spring evaluator source path is missing")
    if evaluator_sha != sha256_file(Path(evaluator_path)):
        raise SpringStageCError("Spring evaluator source SHA-256 differs")
    contract_path_value = sources.get("spring_contract_path")
    contract_sha_value = sources.get("spring_contract_sha256")
    if contract_path_value is not None or contract_sha_value is not None:
        if not isinstance(contract_path_value, str) or not Path(
            contract_path_value
        ).is_file():
            raise SpringStageCError("Spring contract source path is missing")
        if contract_sha_value != sha256_file(Path(contract_path_value)):
            raise SpringStageCError("Spring contract source SHA-256 differs")
    if metrics_path is not None:
        metrics_file = Path(metrics_path).expanduser().resolve()
        if not metrics_file.is_file():
            raise FileNotFoundError(metrics_file)
        if sources.get("metrics_sha256") is not None and sources.get(
            "metrics_sha256"
        ) != sha256_file(metrics_file):
            raise SpringStageCError("Spring metrics source SHA-256 differs")
    validation_info: dict[str, Any] | None = None
    if validation_manifest is not None:
        validation_info = validate_spring_manifest(validation_manifest)
        if coverage.get("manifest_sha256") != validation_info["sha256"]:
            raise SpringStageCError("Spring report validation manifest SHA differs")
        if screening.get("manifest") != validation_info["path"]:
            raise SpringStageCError("Spring report validation manifest path differs")
    coverage_manifest_path = coverage.get("manifest_path")
    if coverage_manifest_path is not None and coverage_manifest_path != screening.get(
        "manifest"
    ):
        raise SpringStageCError("Spring coverage/report manifest paths differ")
    rectification = lineage.get("rectification_audit")
    if rectification is not None and (
        not isinstance(rectification, Mapping)
        or rectification.get("protocol") != SPRING_STAGE_C_PROTOCOL
        or rectification.get("canonical") is not False
    ):
        raise SpringStageCError("Spring rectification lineage marker is malformed")
    auditor_info: dict[str, Any] | None = None
    if auditor_path is not None:
        auditor = Path(auditor_path).expanduser().resolve()
        auditor_value = report.get("auditor")
        if not isinstance(auditor_value, Mapping):
            raise SpringStageCError("Spring report auditor identity is missing")
        if auditor_value.get("path") != str(auditor):
            raise SpringStageCError("Spring report auditor path differs")
        if auditor_value.get("sha256") != sha256_file(auditor):
            raise SpringStageCError("Spring report auditor SHA-256 differs")
        auditor_info = {
            "path": str(auditor),
            "sha256": auditor_value.get("sha256"),
        }
    checkpoint_info: dict[str, Any] | None = None
    if checkpoint is not None:
        if train_adapter_path is None:
            train_adapter_path = Path(__file__).resolve().parent / "train_spring_epipolar.py"
        checkpoint_info = validate_spring_checkpoint_marker(
            checkpoint,
            train_adapter_path=train_adapter_path,
            contract_path=Path(__file__).resolve(),
        )
        checkpoint_hash = sources.get("stage_c_checkpoint_sha256")
        if checkpoint_hash != sha256_file(Path(checkpoint).expanduser().resolve()):
            raise SpringStageCError("Spring checkpoint SHA-256 differs from report")
    return {
        "status": "PASS",
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "canonical": False,
        "metrics_path": None
        if metrics_path is None
        else str(Path(metrics_path).expanduser().resolve()),
        "validation_manifest": validation_info,
        "auditor": auditor_info,
        "checkpoint": checkpoint_info,
        "acceptance_eligible": False,
    }
