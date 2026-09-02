#!/usr/bin/env python3
"""Build a deterministic, sequence-disjoint Spring train/validation split.

The screening runner's bounded ``--limit`` option is useful for smoke tests,
but it cannot express a validation set stratified by motion, calibration, and
rigid-map support.  This small producer keeps the input manifest immutable and
writes validated JSONL manifests plus a receipt that records the exact
sequence lists and source hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import load_manifest, write_manifest  # noqa: E402


PRIMARY_VALIDATION = (
    "0005",
    "0010",
    "0015",
    "0021",
    "0023",
    "0030",
    "0032",
    "0047",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sequences(value: str) -> tuple[str, ...]:
    result = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not result:
        raise ValueError("validation sequence list must not be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="all-record Spring JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-sequences",
        default=",".join(PRIMARY_VALIDATION),
        help="comma-separated sequence IDs assigned to validation",
    )
    parser.add_argument(
        "--min-frames-per-sequence",
        type=int,
        default=15,
        help="reject selected sequences shorter than this many frames",
    )
    args = parser.parse_args()
    if args.min_frames_per_sequence <= 0:
        raise ValueError("--min-frames-per-sequence must be positive")

    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = load_manifest(source)
    validation_sequences = _parse_sequences(args.validation_sequences)
    all_sequences = {record.sequence_id for record in records}
    missing = sorted(set(validation_sequences) - all_sequences)
    if missing:
        raise ValueError(f"validation sequences are absent from input: {missing}")

    counts: dict[str, int] = {}
    for record in records:
        counts[record.sequence_id] = counts.get(record.sequence_id, 0) + 1
    too_short = sorted(
        sequence
        for sequence in validation_sequences
        if counts[sequence] < args.min_frames_per_sequence
    )
    if too_short:
        raise ValueError(
            "validation sequences do not meet the minimum frame count "
            f"({args.min_frames_per_sequence}): {too_short}"
        )

    validation_set = set(validation_sequences)
    train_records = [record for record in records if record.sequence_id not in validation_set]
    validation_records = [record for record in records if record.sequence_id in validation_set]
    if not train_records or not validation_records:
        raise ValueError("split produced an empty train or validation manifest")
    train_sequences = sorted({record.sequence_id for record in train_records})
    actual_validation_sequences = sorted({record.sequence_id for record in validation_records})
    if set(train_sequences) & set(actual_validation_sequences):
        raise RuntimeError("train/validation sequence overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_manifest(train_path, train_records)
    write_manifest(validation_path, validation_records)
    receipt = {
        "schema_version": 1,
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "train_manifest": str(train_path),
        "train_manifest_sha256": _sha256(train_path),
        "validation_manifest": str(validation_path),
        "validation_manifest_sha256": _sha256(validation_path),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_sequences": train_sequences,
        "validation_sequences": actual_validation_sequences,
        "sequence_disjoint": True,
        "min_frames_per_validation_sequence": args.min_frames_per_sequence,
    }
    receipt_path = output_dir / "split_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
