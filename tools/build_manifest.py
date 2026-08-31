#!/usr/bin/env python3
"""Convert the XD 5-fps stereo CSV/metadata layout into validated JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from data.manifest import ManifestRecord, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-source-video-stem",
        action="append",
        default=[],
        help="Repeat to form a video-isolated split; empty includes every CSV video",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-hz", type=float, default=5.0)
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_3x3(flat: Any, field: str) -> list[list[float]]:
    if not isinstance(flat, list) or len(flat) != 9:
        raise ValueError(f"{field} must contain 9 values")
    values = [float(value) for value in flat]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} contains non-finite values")
    return [values[0:3], values[3:6], values[6:9]]


def _matrix_3x4(flat: Any, field: str) -> list[list[float]]:
    if not isinstance(flat, list) or len(flat) != 12:
        raise ValueError(f"{field} must contain 12 values")
    values = [float(value) for value in flat]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} contains non-finite values")
    return [values[0:4], values[4:8], values[8:12]]


def _resolve_source_csv(args: argparse.Namespace) -> Path:
    if args.source_csv is not None:
        return args.source_csv
    return args.data_root / "manifest.csv"


def _load_records(args: argparse.Namespace) -> list[ManifestRecord]:
    source_csv = _resolve_source_csv(args)
    if not source_csv.is_file():
        raise FileNotFoundError(f"source CSV does not exist: {source_csv}")
    included_stems = set(args.include_source_video_stem)
    records: list[ManifestRecord] = []
    seen: set[tuple[str, int]] = set()
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"token", "source_video", "source_time_sec", "source_frame_index", "left", "right"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV fields must include {sorted(required)}")
        for csv_row_number, row in enumerate(reader, start=2):
            source_video = row["source_video"]
            sequence_id = Path(source_video).stem
            if included_stems and sequence_id not in included_stems:
                continue
            original_left_path = Path(row["left"])
            sample_dir = original_left_path.parent
            left_rect_path = sample_dir / "left_rect.jpg"
            right_rect_path = sample_dir / "right_rect.jpg"
            metadata_path = sample_dir / "meta.yaml"
            for required_path in (left_rect_path, right_rect_path, metadata_path):
                if not required_path.is_file():
                    raise FileNotFoundError(f"CSV row {csv_row_number}: missing {required_path}")
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("rectified") is not True:
                raise ValueError(f"{metadata_path}: rectified must be true")
            if metadata.get("source_video") != source_video:
                raise ValueError(f"{metadata_path}: source_video disagrees with CSV")
            timestamp = float(row["source_time_sec"])
            frame_id = int(row["source_frame_index"])
            if not math.isclose(
                float(metadata["source_time_sec"]),
                timestamp,
                rel_tol=0.0,
                abs_tol=args.timestamp_tolerance,
            ):
                raise ValueError(f"{metadata_path}: timestamp disagrees with CSV")
            if int(metadata["source_frame_index"]) != frame_id:
                raise ValueError(f"{metadata_path}: frame index disagrees with CSV")
            key = (sequence_id, frame_id)
            if key in seen:
                raise ValueError(f"duplicate sequence/frame identity: {key}")
            seen.add(key)

            left_rect = metadata["left_rect_camera_info"]
            right_rect = metadata["right_rect_camera_info"]
            left_projection = _matrix_3x4(left_rect["p"], "left_rect_camera_info.p")
            right_projection = _matrix_3x4(right_rect["p"], "right_rect_camera_info.p")
            baseline_m = float(metadata["stereo_baseline_m"])
            baseline_from_projection_m = abs(
                right_projection[0][3] / right_projection[0][0]
            )
            if not math.isclose(
                baseline_m,
                baseline_from_projection_m,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{metadata_path}: baseline and right projection disagree")
            records.append(
                ManifestRecord(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    left_path=str(left_rect_path.resolve()),
                    right_path=str(right_rect_path.resolve()),
                    K=tuple(tuple(row_values) for row_values in _matrix_3x3(left_rect["k"], "left_rect_camera_info.k")),
                    baseline_m=baseline_m,
                    gt_disparity_path=None,
                    rectified=True,
                    extras={
                        "token": row["token"],
                        "source_video": source_video,
                        "source_frame_index": frame_id,
                        "source_time_sec": timestamp,
                        "source_csv": str(source_csv.resolve()),
                        "source_csv_row": csv_row_number,
                        "metadata_path": str(metadata_path.resolve()),
                        "metadata_sha256": sha256_file(metadata_path),
                        "image_size_wh": [int(left_rect["width"]), int(left_rect["height"])],
                        "K_right": _matrix_3x3(right_rect["k"], "right_rect_camera_info.k"),
                        "P_left": left_projection,
                        "P_right": right_projection,
                        "baseline_from_projection_m": baseline_from_projection_m,
                        "split_source": metadata.get("split"),
                    },
                )
            )
    return records


def _validate_contiguous(records: list[ManifestRecord], expected_hz: float, tolerance: float) -> dict[str, Any]:
    if expected_hz <= 0 or not math.isfinite(expected_hz):
        raise ValueError("expected-hz must be finite and positive")
    expected_dt = 1.0 / expected_hz
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sequence_id].append(record)
    summaries: dict[str, Any] = {}
    for sequence_id, sequence_records in grouped.items():
        sequence_records.sort(key=lambda record: record.timestamp)
        deltas = [
            current.timestamp - previous.timestamp
            for previous, current in zip(sequence_records, sequence_records[1:])
        ]
        discontinuities = [
            index
            for index, delta in enumerate(deltas, start=1)
            if not math.isclose(delta, expected_dt, rel_tol=0.0, abs_tol=tolerance)
        ]
        if discontinuities:
            raise ValueError(
                f"sequence {sequence_id!r} is not contiguous at {expected_hz} Hz; "
                f"first bad local index {discontinuities[0]}"
            )
        summaries[sequence_id] = {
            "frames": len(sequence_records),
            "first_timestamp": sequence_records[0].timestamp,
            "last_timestamp": sequence_records[-1].timestamp,
            "first_frame_id": sequence_records[0].frame_id,
            "last_frame_id": sequence_records[-1].frame_id,
            "t3_endpoints": max(0, len(sequence_records) - 2),
            "vggt5_endpoints": max(0, len(sequence_records) - 4),
        }
    return summaries


def main() -> int:
    args = parse_args()
    if args.start_index < 0 or args.limit is not None and args.limit <= 0:
        raise ValueError("start-index must be non-negative and limit must be positive")
    records = _load_records(args)
    records.sort(key=lambda record: (record.sequence_id, record.timestamp))
    selected = records[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("manifest selection is empty")
    sequence_summary = _validate_contiguous(selected, args.expected_hz, args.timestamp_tolerance)
    write_manifest(args.output, selected)
    summary = {
        "schema_version": 1,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "records": len(selected),
        "expected_hz": args.expected_hz,
        "source_csv": str(_resolve_source_csv(args).resolve()),
        "source_csv_sha256": sha256_file(_resolve_source_csv(args)),
        "included_source_video_stems": args.include_source_video_stem,
        "sequences": sequence_summary,
        "rectified_only": True,
        "gt_disparity": None,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

