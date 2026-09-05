#!/usr/bin/env python3
"""Merge shard outputs from audit_metric_stereo_video_disparity_range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not payloads:
        raise ValueError("no shard reports")
    merged = []
    for audit_index in range(len(payloads[0]["audits"])):
        rows = [payload["audits"][audit_index] for payload in payloads]
        counts: dict[str, int] = {}
        frames = 0
        map_frames = {"dynamic": 0, "detail": 0}
        for row in rows:
            frames += int(row["frames"])
            for name, value in row["counts"].items():
                counts[name] = counts.get(name, 0) + int(value)
            for name, value in row["map_frames"].items():
                map_frames[name] += int(value)
        def rate(num: str, den: str) -> float | None:
            return counts[num] / counts[den] if counts[den] else None
        merged.append({
            "manifest": rows[0]["manifest"],
            "frames": frames,
            "threshold_hr_px": rows[0]["threshold_hr_px"],
            "map_frames": map_frames,
            "counts": counts,
            "rates": {
                "overall": rate("over_threshold", "gt_valid"),
                "dynamic": rate("dynamic_over_threshold", "dynamic_valid"),
                "high_detail": rate("detail_over_threshold", "detail_valid"),
            },
        })
    report = {"schema_version": 1, "shards": len(payloads), "audits": merged}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
