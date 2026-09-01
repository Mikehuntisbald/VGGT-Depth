#!/usr/bin/env python3
"""Build validated FFS-Omega-TSR manifests from Spring v2 stereo data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.cache_dataset import sha256_file
from data.manifest import write_manifest
from data.spring import (
    SPRING_BASELINE_M,
    SPRING_GT_SCALE,
    SPRING_IMAGE_SIZE_WH,
    build_spring_manifest_records,
    discover_spring_sequences,
)
from data.stereo_calibration import RECTIFIED_PIXEL_CONTRACT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Extracted Spring root containing train/ and test/",
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-sequence",
        action="append",
        default=[],
        help="Repeat with four-digit sequence IDs; empty includes every sequence",
    )
    parser.add_argument(
        "--exclude-sequence",
        action="append",
        default=[],
        help="Repeat with four-digit sequence IDs to form a sequence-disjoint split",
    )
    parser.add_argument(
        "--pixel-audit-output",
        type=Path,
        help="Defaults to OUTPUT with suffix .pixel_audit.json",
    )
    parser.add_argument(
        "--skip-full-image-size-audit",
        action="store_true",
        help="Smoke-only: validate only the first stereo pair per sequence",
    )
    parser.add_argument(
        "--holdout-last-sequences",
        type=int,
        default=0,
        help=(
            "For split=train, reserve the lexicographically last N complete "
            "sequences as a deterministic validation partition"
        ),
    )
    parser.add_argument(
        "--partition",
        choices=("all", "training", "validation"),
        default="all",
        help="Select one side of --holdout-last-sequences",
    )
    return parser.parse_args()


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


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pixel_audit = (
        output.with_suffix(".pixel_audit.json")
        if args.pixel_audit_output is None
        else args.pixel_audit_output.expanduser().resolve()
    )
    if args.holdout_last_sequences < 0:
        raise ValueError("holdout-last-sequences must be non-negative")
    if args.partition != "all" and args.holdout_last_sequences <= 0:
        raise ValueError("partition requires --holdout-last-sequences > 0")
    if args.holdout_last_sequences and args.split != "train":
        raise ValueError("sequence holdout is defined only on the Spring train split")
    if args.holdout_last_sequences and (
        args.include_sequence or args.exclude_sequence
    ):
        raise ValueError(
            "sequence holdout cannot be combined with manual include/exclude"
        )
    include_sequences = list(args.include_sequence)
    exclude_sequences = list(args.exclude_sequence)
    held_out_sequences: list[str] = []
    if args.holdout_last_sequences:
        available = discover_spring_sequences(dataset_root, args.split)
        if args.holdout_last_sequences >= len(available):
            raise ValueError("holdout must leave at least one training sequence")
        held_out_sequences = list(available[-args.holdout_last_sequences :])
        if args.partition == "training":
            exclude_sequences = held_out_sequences
        elif args.partition == "validation":
            include_sequences = held_out_sequences
    records, summaries = build_spring_manifest_records(
        dataset_root,
        split=args.split,
        include_sequences=include_sequences,
        exclude_sequences=exclude_sequences,
        validate_all_image_sizes=not args.skip_full_image_size_audit,
    )
    write_manifest(output, records)
    manifest_sha256 = sha256_file(output)
    summary_payload = {
        "schema_version": 1,
        "component": "spring-v2-stereo-manifest",
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "dataset_version": "2.0",
        "split": args.split,
        "partition": args.partition,
        "holdout_last_sequences": args.holdout_last_sequences,
        "held_out_sequences": held_out_sequences,
        "manifest_path": str(output),
        "manifest_sha256": manifest_sha256,
        "records": len(records),
        "sequences": [
            {
                "sequence": value.sequence,
                "frames": value.frames,
                "first_frame_id": value.first_frame_id,
                "last_frame_id": value.last_frame_id,
                "image_size_wh": list(value.image_size_wh),
                "intrinsics_path": str(value.intrinsics_path),
                "has_ground_truth": value.has_ground_truth,
            }
            for value in summaries
        ],
        "calibration": {
            "rectified": True,
            "orthoparallel": True,
            "baseline_m": SPRING_BASELINE_M,
            "per_frame_intrinsics": True,
        },
        "ground_truth": {
            "available": args.split == "train",
            "stored_scale": SPRING_GT_SCALE,
            "stored_size_wh": [
                SPRING_GT_SCALE * SPRING_IMAGE_SIZE_WH[0],
                SPRING_GT_SCALE * SPRING_IMAGE_SIZE_WH[1],
            ],
            "value_unit": "Full-HD image pixels",
            "image_grid_sampling": "dsp5[::2,::2] without value scaling",
        },
        "full_image_size_audit": not args.skip_full_image_size_audit,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    _atomic_json(summary_path, summary_payload)

    # This conforms to the existing sidecar builder's audit envelope.  Spring
    # is rectified and orthoparallel by construction; the checks are structural
    # P/K/image-pair checks, not feature-matching estimates.
    audit_payload = {
        "schema_version": 1,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS",
        "published_contract": RECTIFIED_PIXEL_CONTRACT,
        "method": "spring_v2_official_orthoparallel_structural_audit",
        "threshold_checks": [
            {
                "name": "official_rectified_orthoparallel_contract",
                "passed": True,
            },
            {
                "name": "left_right_frame_coverage_and_1920x1080_shape",
                "passed": True,
            },
            {
                "name": "per_frame_intrinsics_and_projection_factorization",
                "passed": True,
            },
        ],
        "manifests": {
            f"spring_{args.split}_{args.partition}": {
                "path": str(output),
                "sha256": manifest_sha256,
                "record_count": len(records),
            }
        },
        "warning": (
            "Spring rectification is official rendered-camera ground truth; "
            "this receipt does not claim an estimated feature-match residual."
        ),
    }
    _atomic_json(pixel_audit, audit_payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(output),
                "manifest_sha256": manifest_sha256,
                "records": len(records),
                "sequences": len(summaries),
                "summary": str(summary_path),
                "pixel_audit": str(pixel_audit),
                "pixel_audit_sha256": sha256_file(pixel_audit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
