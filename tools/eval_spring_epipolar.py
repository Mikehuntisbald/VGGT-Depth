#!/usr/bin/env python3
"""Evaluate Stage-C HR epipolar refinement on a Spring split.

The project-level :mod:`eval_epipolar` evaluator intentionally remains bound
to its canonical 244/240/238 holdout.  This adapter keeps that file unchanged
and reuses its full model/metric/lineage implementation in a subprocess-local
patch.  Only the corpus-specific contracts are replaced:

* causal coverage is reconstructed from the supplied Spring manifest;
* the pixel rectification audit is bound to the supplied train/validation
  manifests without canonical count/hash assumptions; and
* train/validation sequence isolation is still strict (no overlap bypass).

``--spring-screening`` and an explicit ``--limit`` are mandatory.  The output
is always marked screening-only and can never become a canonical acceptance
result by changing a count or filename.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_epipolar as canonical  # noqa: E402
import train_epipolar as training_canonical  # noqa: E402
from data.cache_dataset import sha256_file  # noqa: E402
from tools.spring_stage_c import (  # noqa: E402
    SPRING_STAGE_C_PROTOCOL,
    SPRING_STAGE_C_REPORT_STAGE,
    require_spring_stage_c_coverage,
    spring_temporal_coverage,
    strict_spring_holdout_lineage,
    validate_spring_checkpoint_marker,
    validate_spring_manifest,
    validate_spring_rectification_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = canonical.build_parser()
    parser.description = (
        "Evaluate Stage-C epipolar refinement on a bounded Spring screening split."
    )
    parser.add_argument(
        "--spring-screening",
        action="store_true",
        help="required explicit opt-in; output is never canonical/formal",
    )
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mark_screening_report(
    args: argparse.Namespace, *, checkpoint_marker: dict[str, Any]
) -> None:
    output = args.output.expanduser().resolve()
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(
            f"canonical Spring Stage-C evaluation did not produce {metrics_path}"
        )
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read Spring Stage-C metrics: {metrics_path}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("Spring Stage-C metrics report is not a JSON object")
    prior_status = report.get("status")
    source_hashes = report.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    canonical_path = Path(__file__).resolve().parents[1] / "eval_epipolar.py"
    source_hashes["canonical_evaluator_path"] = str(canonical_path)
    source_hashes["canonical_evaluator_sha256"] = sha256_file(canonical_path)
    source_hashes["evaluator_path"] = str(Path(__file__).resolve())
    source_hashes["evaluator_sha256"] = sha256_file(Path(__file__).resolve())
    contract_path = PROJECT_ROOT / "tools" / "spring_stage_c.py"
    source_hashes["spring_contract_path"] = str(contract_path)
    source_hashes["spring_contract_sha256"] = sha256_file(contract_path)
    report["source_hashes"] = source_hashes
    report["stage"] = SPRING_STAGE_C_REPORT_STAGE
    report["status"] = "SPRING_STAGE_C_SCREENING"
    claims = report.get("claims")
    if not isinstance(claims, dict):
        claims = {}
    claims.update(
        {
            "acceptance_eligible": False,
            "formal_holdout": False,
            "spring_screening": True,
            "canonical_stage_c_replacement": False,
            "performance_acceptance_claimed": False,
            "canonical_status_before_spring_wrapper": prior_status,
        }
    )
    report["claims"] = claims
    report["spring_screening"] = {
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "canonical": False,
        "limit": int(args.limit),
        "manifest": str(args.manifest.expanduser().resolve()),
        "checkpoint_adapter_binding": checkpoint_marker,
        "note": (
            "This report is a bounded, sequence-disjoint Spring screening result; "
            "it is not the canonical 244/240/238 holdout."
        ),
    }
    lineage = report.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
    lineage["spring_protocol"] = {
        "protocol": SPRING_STAGE_C_PROTOCOL,
        "canonical": False,
        "sequence_disjoint_required": True,
    }
    report["lineage"] = lineage
    _atomic_json(metrics_path, report)


def run(args: argparse.Namespace) -> int:
    if not args.spring_screening:
        raise ValueError(
            "Spring Stage-C requires the explicit --spring-screening opt-in"
        )
    if args.limit is None or int(args.limit) <= 0:
        raise ValueError(
            "Spring Stage-C screening requires an explicit positive --limit"
        )
    if args.manifest is None:
        raise ValueError("Spring Stage-C requires --manifest")
    # The official Spring source labels all extracted frames ``split=train``;
    # this evaluator receives a separate sequence-disjoint validation file and
    # must not infer the protocol partition from that source tag.
    validate_spring_manifest(args.manifest)
    checkpoint_marker = validate_spring_checkpoint_marker(
        args.checkpoint,
        train_adapter_path=PROJECT_ROOT / "tools" / "train_spring_epipolar.py",
        contract_path=PROJECT_ROOT / "tools" / "spring_stage_c.py",
    )

    # Preserve an explicit protocol marker in the resolved evaluation config;
    # it is diagnostic only and never turns this run into a canonical result.
    training_canonical.STAGE_C_DEFAULTS.setdefault("spring_screening", False)
    training_canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_protocol", None)
    training_canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_train_adapter_sha256", None)
    training_canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_contract_sha256", None)
    overrides = list(args.overrides)
    for item in overrides:
        key, _, value = str(item).partition("=")
        if key == "spring_screening" and value.lower() not in {"true", "1", "yes"}:
            raise ValueError("spring_screening cannot be disabled for this entry point")
        if key == "spring_stage_c_protocol" and value != SPRING_STAGE_C_PROTOCOL:
            raise ValueError("spring_stage_c_protocol override differs from Spring protocol")
    if not any(str(item).split("=", 1)[0] == "spring_screening" for item in overrides):
        overrides.append("spring_screening=true")
    overrides = [
        item
        for item in overrides
        if str(item).split("=", 1)[0]
        not in {
            "spring_stage_c_protocol",
            "spring_stage_c_train_adapter_sha256",
            "spring_stage_c_contract_sha256",
        }
    ]
    overrides.extend(
        (
            f"spring_stage_c_protocol={SPRING_STAGE_C_PROTOCOL}",
            f"spring_stage_c_train_adapter_sha256={checkpoint_marker['train_adapter_sha256']}",
            f"spring_stage_c_contract_sha256={checkpoint_marker['contract_sha256']}",
        )
    )
    args.overrides = tuple(overrides)

    # The canonical parser/runner is reused verbatim.  Its strict checkpoint,
    # runtime-source, cache, and metric code remains active; only these
    # corpus-specific functions are replaced for this process.
    original_coverage = canonical._validate_formal_temporal_coverage
    original_require = canonical.require_formal_stage_c_coverage
    original_rectification = canonical.validate_rectification_audit_binding
    original_holdout = canonical._audit_temporal_holdout_and_raw_lineage
    canonical._validate_formal_temporal_coverage = spring_temporal_coverage
    canonical.require_formal_stage_c_coverage = require_spring_stage_c_coverage
    canonical.validate_rectification_audit_binding = (
        lambda stage_c_metadata, *, receipt_path, validation_manifest_sha256: (
            validate_spring_rectification_binding(
                stage_c_metadata=stage_c_metadata,
                receipt_path=receipt_path,
                validation_manifest_sha256=validation_manifest_sha256,
                generic_validator=canonical._validated_rectification_audit,
            )
        )
    )
    canonical._audit_temporal_holdout_and_raw_lineage = (
        lambda **kwargs: strict_spring_holdout_lineage(original_holdout, **kwargs)
    )
    try:
        result = canonical.run(args)
    finally:
        canonical._validate_formal_temporal_coverage = original_coverage
        canonical.require_formal_stage_c_coverage = original_require
        canonical.validate_rectification_audit_binding = original_rectification
        canonical._audit_temporal_holdout_and_raw_lineage = original_holdout
    if int(result) == 0:
        _mark_screening_report(args, checkpoint_marker=checkpoint_marker)
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
