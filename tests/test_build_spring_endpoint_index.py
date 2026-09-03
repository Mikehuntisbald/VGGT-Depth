from __future__ import annotations

import json
from pathlib import Path

from data.manifest import ManifestRecord, write_manifest
from tools.build_spring_endpoint_index import (
    _sequence_warmup_indices,
    build_parser,
    run,
)


def _manifest(tmp_path: Path) -> Path:
    rows = []
    for sequence in ("a", "b"):
        for frame in range(1, 9):
            rows.append(
                ManifestRecord(
                    sequence_id=sequence,
                    frame_id=frame,
                    timestamp=float(frame - 1),
                    left_path=f"{sequence}-left-{frame}.png",
                    right_path=f"{sequence}-right-{frame}.png",
                    K=((100.0, 0.0, 4.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0)),
                    baseline_m=0.1,
                    gt_disparity_path=None,
                    extras={"dataset": "spring", "timestamp_source": "frame_index"},
                )
            )
    path = tmp_path / "validation.jsonl"
    write_manifest(path, rows)
    return path


def test_sequence_warmup_is_per_sequence_and_preserves_manifest_indices(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    from data.manifest import load_manifest

    records = load_manifest(manifest)
    indices = list(range(len(records)))
    assert _sequence_warmup_indices(records, indices, 6) == [6, 7, 14, 15]


def test_builder_cli_records_warmup_and_output_lineage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "common.json"
    receipt = tmp_path / "common.receipt.json"
    args = build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--sequence-warmup",
            "6",
            "--receipt",
            str(receipt),
        ]
    )
    assert run(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["endpoint_count"] == 4
    assert receipt_payload["protocol"]["sequence_warmup"] == 6
    assert receipt_payload["selection"]["after_warmup"] == 4
    assert receipt_payload["output"]["endpoint_id_sha256"] == payload[
        "endpoint_id_sha256"
    ]
    assert receipt_payload["output"]["file_sha256"]


def test_builder_parser_keeps_zero_warmup_as_legacy_default() -> None:
    args = build_parser().parse_args(["--manifest", "m", "--output", "o"])
    assert args.sequence_warmup == 0
