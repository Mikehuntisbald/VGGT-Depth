#!/usr/bin/env python3
"""CLI wrapper for the validated Spring dataset adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.spring import SPRING_BASELINE_M, build_spring_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Spring JSONL manifest")
    parser.add_argument("--spring-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--timestamp-fps", type=float)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-missing-disparity", action="store_true")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="1-based first frame retained within each selected sequence",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        help="bounded deterministic screening subset per sequence",
    )
    args = parser.parse_args()
    if args.start_frame < 1:
        raise ValueError("--start-frame must be >= 1")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        raise ValueError("--max-frames-per-sequence must be positive")
    records = build_spring_manifest(
        args.spring_root,
        args.output,
        split=args.split,
        sequences=args.sequence or None,
        require_images=not args.allow_missing_images,
        require_disparity=(
            args.split == "train" and not args.allow_missing_disparity
        ),
        baseline_m=SPRING_BASELINE_M,
        timestamp_fps=args.timestamp_fps,
    )
    if args.start_frame != 1 or args.max_frames_per_sequence is not None:
        grouped: dict[str, list] = {}
        for record in records:
            grouped.setdefault(record.sequence_id, []).append(record)
        selected = []
        for sequence_id in sorted(grouped):
            sequence_records = [
                record for record in grouped[sequence_id] if record.frame_id >= args.start_frame
            ]
            if args.max_frames_per_sequence is not None:
                sequence_records = sequence_records[: args.max_frames_per_sequence]
            selected.extend(sequence_records)
        if not selected:
            raise ValueError("frame selection is empty")
        from data.manifest import write_manifest

        write_manifest(args.output, selected)
        records = selected
    summary = {
        "dataset": "spring",
        "split": args.split,
        "records": len(records),
        "sequences": sorted({record.sequence_id for record in records}),
        "baseline_m": SPRING_BASELINE_M,
        "output": str(args.output.resolve()),
        "gt_pose_source": "cam_data/extrinsics.txt",
        "gt_pose_convention": "world_to_camera_opencv",
        "start_frame": args.start_frame,
        "max_frames_per_sequence": args.max_frames_per_sequence,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
