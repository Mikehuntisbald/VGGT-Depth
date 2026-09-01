from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.run_spring_arms import (
    _expected_vggt_targets,
    _strict_vggt_cache_matches,
)


def _manifest_rows(count: int = 7) -> list[dict[str, object]]:
    return [
        {
            "sequence_id": "0007",
            "frame_id": index + 1,
            "timestamp": float(index),
            "left_path": f"left/{index + 1}.png",
            "right_path": f"right/{index + 1}.png",
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_expected_vggt_targets_start_after_five_pair_warmup() -> None:
    rows = _manifest_rows()
    targets = _expected_vggt_targets(rows)
    assert targets == [
        (4, "0007", 5, 4.0),
        (5, "0007", 6, 5.0),
        (6, "0007", 7, 6.0),
    ]


def test_strict_vggt_cache_rejects_manifest_bound_partial_prefix(tmp_path: Path) -> None:
    manifest = tmp_path / "validation.jsonl"
    rows = _manifest_rows()
    _write_jsonl(manifest, rows)
    root = tmp_path / "vggt"
    root.mkdir()
    cache_path = root / "0007" / "5.pt"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"placeholder")
    cache_manifest = root / "cache_manifest.jsonl"
    _write_jsonl(
        cache_manifest,
        [
            {
                "selection_index": 0,
                "target_manifest_index": 4,
                "sequence_id": "0007",
                "frame_id": 5,
                "timestamp": 4.0,
                "cache_path": str(cache_path),
            }
        ],
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "run_receipt.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha,
                "available_windows": 1,
                "selected_windows": 1,
                "written_records": 1,
                "reused_records": 0,
            }
        ),
        encoding="utf-8",
    )
    assert not _strict_vggt_cache_matches(
        root, manifest, expected_targets=_expected_vggt_targets(rows)
    )
