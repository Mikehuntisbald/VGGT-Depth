from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.endpoint_selection import (
    EndpointSelectionError,
    load_endpoint_index,
    resolve_endpoint_dataset_indices,
    write_endpoint_index,
)
from data.manifest import ManifestRecord, write_manifest


def _manifest(tmp_path: Path) -> Path:
    records = [
        ManifestRecord(
            sequence_id="spring-seq",
            frame_id=index,
            timestamp=float(index),
            left_path=f"left-{index}.png",
            right_path=f"right-{index}.png",
            K=((100.0, 0.0, 4.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0)),
            baseline_m=0.1,
            gt_disparity_path=None,
            extras={"dataset": "spring"},
        )
        for index in range(8)
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records)
    return path


def test_endpoint_index_roundtrip_binds_manifest_and_maps_positions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selection = write_endpoint_index(
        tmp_path / "common.json",
        manifest_path=manifest,
        manifest_indices=[4, 6],
    )
    loaded = load_endpoint_index(selection.path, manifest_path=manifest)
    assert loaded.manifest_indices == (4, 6)
    assert loaded.count == 2
    assert loaded.entries_sha256 == selection.entries_sha256
    assert resolve_endpoint_dataset_indices(loaded, [2, 4, 6]) == (1, 2)
    report = loaded.to_report(available_endpoint_count=3)
    assert report["endpoint_count"] == 2
    assert report["available_endpoint_count"] == 3
    assert report["endpoint_id_sha256"] == loaded.entries_sha256


def test_endpoint_index_fails_closed_for_missing_or_mismatched_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selection = write_endpoint_index(
        tmp_path / "common.json",
        manifest_path=manifest,
        manifest_indices=[4, 6],
    )
    with pytest.raises(EndpointSelectionError, match="missing required endpoint"):
        resolve_endpoint_dataset_indices(selection, [4])

    altered = tmp_path / "altered.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[4]["frame_id"] = 400
    altered.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(EndpointSelectionError, match="different manifest SHA"):
        load_endpoint_index(selection.path, manifest_path=altered)


def test_endpoint_index_rejects_unsorted_entries(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selection = write_endpoint_index(
        tmp_path / "common.json",
        manifest_path=manifest,
        manifest_indices=[4, 6],
    )
    payload = json.loads(selection.path.read_text())
    payload["entries"] = list(reversed(payload["entries"]))
    selection.path.write_text(json.dumps(payload))
    with pytest.raises(EndpointSelectionError, match="strictly increasing"):
        load_endpoint_index(selection.path, manifest_path=manifest)


def test_endpoint_index_accepts_index_only_or_sequence_frame_entries(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selection = write_endpoint_index(
        tmp_path / "common.json",
        manifest_path=manifest,
        manifest_indices=[4, 6],
    )
    payload = json.loads(selection.path.read_text())
    payload["entries"] = [
        {"manifest_index": 4},
        {"sequence_id": "spring-seq", "frame_id": 6},
    ]
    # The loader canonicalizes omitted fields against the bound manifest, so
    # recompute the declared identity digest using the generated selection.
    canonical = write_endpoint_index(
        tmp_path / "canonical.json",
        manifest_path=manifest,
        manifest_indices=[4, 6],
    )
    payload["endpoint_id_sha256"] = canonical.entries_sha256
    selection.path.write_text(json.dumps(payload))
    loaded = load_endpoint_index(selection.path, manifest_path=manifest)
    assert loaded.manifest_indices == (4, 6)
