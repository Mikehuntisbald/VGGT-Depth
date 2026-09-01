#!/usr/bin/env python3
"""Bounded Spring Stage-C trainer.

This is a deliberately thin adapter around the canonical ``train_epipolar``
implementation.  The canonical trainer keeps its formal temporal-coverage
gate unchanged; this entry point replaces only that gate with the supplied
Spring-manifest coverage contract and requires an explicit screening opt-in.
All optimizer/checkpoint/runtime/lineage checks remain owned by the canonical
trainer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_epipolar as canonical  # noqa: E402

from tools.spring_stage_c import (  # noqa: E402
    SPRING_STAGE_C_PROTOCOL,
    spring_temporal_coverage,
    sha256_path,
    validate_spring_checkpoint_marker,
    validate_spring_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = canonical.build_parser()
    parser.description = (
        "Train the Stage-C refiner on a sequence-disjoint Spring screening split."
    )
    parser.add_argument(
        "--spring-screening",
        action="store_true",
        help="required explicit opt-in; this run is never canonical/formal",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.spring_screening:
        raise ValueError(
            "Spring Stage-C requires the explicit --spring-screening opt-in"
        )
    if args.manifest is None:
        raise ValueError("Spring Stage-C requires --manifest")
    # The official Spring source marks every frame ``split=train``; the
    # runner's protocol train/validation partition is sequence-disjoint and is
    # represented by separate manifest files, not by mutating that source tag.
    validate_spring_manifest(args.manifest)
    if args.run_steps is None and not args.dry_run:
        raise ValueError(
            "Spring Stage-C screening requires bounded --run-steps; "
            "canonical unbounded training belongs to train_epipolar.py"
        )
    if args.resume is not None:
        validate_spring_checkpoint_marker(
            args.resume,
            train_adapter_path=Path(__file__).resolve(),
            contract_path=Path(__file__).resolve().parent / "spring_stage_c.py",
        )

    # ``resolve_epipolar_config`` is shared with the canonical trainer.  Add a
    # typed marker to the saved config so downstream evaluators can prove that
    # the checkpoint was produced by this non-formal protocol.
    canonical.STAGE_C_DEFAULTS.setdefault("spring_screening", False)
    canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_protocol", None)
    canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_train_adapter_sha256", None)
    canonical.STAGE_C_DEFAULTS.setdefault("spring_stage_c_contract_sha256", None)
    overrides = list(args.overrides)
    for item in overrides:
        key, _, value = str(item).partition("=")
        if key == "spring_screening" and value.lower() not in {"true", "1", "yes"}:
            raise ValueError("spring_screening cannot be disabled for this entry point")
    if not any(str(item).split("=", 1)[0] == "spring_screening" for item in overrides):
        overrides.append("spring_screening=true")
    # These markers are part of the checkpoint's resolved config and bind the
    # non-canonical adapter itself to the training artifact.
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
            f"spring_stage_c_train_adapter_sha256={sha256_path(__file__)}",
            f"spring_stage_c_contract_sha256={sha256_path(Path(__file__).resolve().parent / 'spring_stage_c.py')}",
        )
    )
    args.overrides = tuple(overrides)

    # The trainer's only formal-corpus dependency is this imported function;
    # patch it for the lifetime of this process.  No canonical module/file is
    # changed and the runtime source bundle embedded in the checkpoint still
    # records the exact committed canonical code.
    original_coverage = canonical._validate_formal_temporal_coverage
    original_rectification = canonical._validated_rectification_audit
    canonical._validate_formal_temporal_coverage = spring_temporal_coverage
    canonical._validated_rectification_audit = (
        lambda path, *, expected_train_manifest_sha256: original_rectification(
            path,
            expected_train_manifest_sha256=expected_train_manifest_sha256,
            allow_consistent_metadata=True,
        )
    )
    try:
        result = canonical.run(args)
    finally:
        canonical._validate_formal_temporal_coverage = original_coverage
        canonical._validated_rectification_audit = original_rectification
    # Keep a machine-readable protocol marker in stdout for batch runners;
    # canonical checkpoint metadata already contains ``spring_screening``.
    print(
        "SPRING_STAGE_C_PROTOCOL=" + SPRING_STAGE_C_PROTOCOL,
        file=sys.stderr,
    )
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
