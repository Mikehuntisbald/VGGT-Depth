from __future__ import annotations

from pathlib import Path

import yaml

from data.cache_dataset import sha256_file
from data.manifest import ManifestRecord, load_manifest, write_manifest
from tools.augment_spring_v31_manifest import augment_manifest


def _record(tmp_path: Path, frame_id: int) -> ManifestRecord:
    left = tmp_path / f"left_{frame_id}.png"
    right = tmp_path / f"right_{frame_id}.png"
    # The historical manifest already carries image_shape_hw; no image decode
    # is needed for this pure lineage test.
    left.touch()
    right.touch()
    return ManifestRecord(
        sequence_id="spring_seq",
        frame_id=frame_id,
        timestamp=float(frame_id - 1),
        left_path=str(left),
        right_path=str(right),
        K=((100.0, 0.0, 3.0), (0.0, 100.0, 2.0), (0.0, 0.0, 1.0)),
        baseline_m=0.065,
        gt_disparity_path=None,
        extras={"dataset": "spring", "image_shape_hw": [4, 6]},
    )


def test_augment_adds_v31_calibration_and_image_size(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "v31.jsonl"
    metadata_root = tmp_path / "metadata"
    write_manifest(source, [_record(tmp_path, 1), _record(tmp_path, 2)])
    source_before = source.read_bytes()

    receipt = augment_manifest(
        input_manifest=source,
        output_manifest=output,
        metadata_root=metadata_root,
    )

    assert source.read_bytes() == source_before
    assert receipt["status"] == "PASS"
    assert receipt["metadata"]["unique_calibrations"] == 1
    rows = load_manifest(output)
    extras = rows[0].extras
    assert extras["image_size_wh"] == [6, 4]
    assert extras["K_right"] == [list(row) for row in rows[0].K]
    assert extras["P_left"][0][3] == 0.0
    assert extras["P_right"][0][3] == -6.5
    metadata_path = Path(str(extras["metadata_path"]))
    assert extras["metadata_sha256"] == sha256_file(metadata_path)
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["rectified"] is True
    assert metadata["right_rect_camera_info"]["p"][3] == -6.5


def test_augment_is_immutable_and_rejects_in_place(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "v31.jsonl"
    write_manifest(source, [_record(tmp_path, 1)])
    first = augment_manifest(
        input_manifest=source,
        output_manifest=output,
        metadata_root=tmp_path / "metadata",
    )
    second = augment_manifest(
        input_manifest=source,
        output_manifest=output,
        metadata_root=tmp_path / "metadata",
    )
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert first["receipt_sha256"] == second["receipt_sha256"]

    try:
        augment_manifest(
            input_manifest=source,
            output_manifest=source,
            metadata_root=tmp_path / "metadata2",
        )
    except ValueError as exc:
        assert "differ" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("in-place augmentation unexpectedly succeeded")

