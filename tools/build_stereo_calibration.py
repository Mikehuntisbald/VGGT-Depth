#!/usr/bin/env python3
"""Build an immutable rectified-stereo calibration sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.stereo_calibration import build_rectified_calibration_sidecar


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive manifest-bound [I|-baseline] rectified stereo extrinsics "
            "without rewriting raw-cache-owned manifests."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pixel-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--spring-native",
        action="store_true",
        help=(
            "derive Spring's fixed rectified virtual-camera calibration from "
            "K/baseline without rewriting the source manifest"
        ),
    )
    parser.add_argument(
        "--spring-metadata-root",
        type=Path,
        help="optional directory for synthetic Spring camera-info metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = build_rectified_calibration_sidecar(
        args.manifest,
        args.pixel_audit,
        args.output,
        receipt_path=args.receipt,
        spring_native=args.spring_native,
        spring_metadata_root=args.spring_metadata_root,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "records": receipt["counts"]["records"],
                "unique_calibrations": receipt["counts"]["unique_calibrations"],
                "sidecar": receipt["output"]["sidecar_path"],
                "sidecar_sha256": receipt["output"]["sidecar_sha256"],
                "receipt": receipt["receipt_path"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
