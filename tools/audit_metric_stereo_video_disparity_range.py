#!/usr/bin/env python3
"""Audit Spring GT disparity values against the FFS 384-HR-pixel ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import load_manifest  # noqa: E402
from metrics.spring_arms import SpringNativeMapError, spring_map_bundle  # noqa: E402
from data.spring import load_spring_disparity  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=384.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_manifest(path: Path, threshold: float, *, shard: int = 0, shards: int = 1) -> dict[str, Any]:
    records = load_manifest(path)
    totals = {name: 0 for name in ("gt_valid", "over_threshold", "dynamic_valid", "dynamic_over_threshold", "detail_valid", "detail_over_threshold")}
    frames = 0
    dynamic_frames = detail_frames = 0
    for record_index, record in enumerate(records):
        if record_index % shards != shard:
            continue
        if not record.gt_disparity_path:
            continue
        gt_path = Path(record.gt_disparity_path).expanduser()
        if not gt_path.is_absolute():
            gt_path = path.parent / gt_path
        disparity = np.asarray(load_spring_disparity(gt_path, resolution="image", sign="positive"), dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 0)
        over = valid & (disparity > threshold)
        totals["gt_valid"] += int(valid.sum())
        totals["over_threshold"] += int(over.sum())
        try:
            bundle = spring_map_bundle(record.to_dict(), target_hw=disparity.shape, manifest_path=path)
        except SpringNativeMapError:
            bundle = {"detail": None, "matched": None, "rigid": None}
        if bundle.get("rigid") is not None:
            dynamic = valid & np.asarray(bundle["rigid"], dtype=bool)
            totals["dynamic_valid"] += int(dynamic.sum())
            totals["dynamic_over_threshold"] += int((dynamic & (disparity > threshold)).sum())
            dynamic_frames += 1
        if bundle.get("detail") is not None:
            detail = valid & np.asarray(bundle["detail"], dtype=bool)
            totals["detail_valid"] += int(detail.sum())
            totals["detail_over_threshold"] += int((detail & (disparity > threshold)).sum())
            detail_frames += 1
        frames += 1
    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None
    return {
        "manifest": str(path.resolve()),
        "frames": frames,
        "threshold_hr_px": threshold,
        "map_frames": {"dynamic": dynamic_frames, "detail": detail_frames},
        "counts": totals,
        "rates": {
            "overall": rate(totals["over_threshold"], totals["gt_valid"]),
            "dynamic": rate(totals["dynamic_over_threshold"], totals["dynamic_valid"]),
            "high_detail": rate(totals["detail_over_threshold"], totals["detail_valid"]),
        },
    }


def main() -> int:
    args = _args()
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")
    shard = int(__import__("os").environ.get("RANGE_AUDIT_SHARD", "0"))
    shards = int(__import__("os").environ.get("RANGE_AUDIT_SHARDS", "1"))
    if not 0 <= shard < shards:
        raise ValueError("RANGE_AUDIT_SHARD must lie in [0,RANGE_AUDIT_SHARDS)")
    report = {"schema_version": 1, "shard": shard, "shards": shards, "audits": [audit_manifest(path.expanduser().resolve(), args.threshold, shard=shard, shards=shards) for path in args.manifest]}
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
