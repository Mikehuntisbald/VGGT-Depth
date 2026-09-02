#!/usr/bin/env python3
"""Create an immutable Spring manifest with explicit v3.1 calibration fields.

The historical Spring manifest contains the left-camera ``K`` and the physical
baseline, but not the right-camera projection matrices or a metadata hash.  A
v3.1 calibrated-stereo sidecar can consume a new manifest carrying those
fields.  This tool deliberately never edits the source manifest or any cache.

For the already-rectified Spring virtual rig the derived contract is::

    K_right = K_left
    P_left  = [K_left | 0]
    P_right = [K_left | (-fx * baseline_m, 0, 0)^T]

One deterministic YAML camera-info file is emitted for each unique ``(K,
baseline)`` pair.  Its path and SHA-256 are recorded in every corresponding
manifest row, allowing the normal strict calibration sidecar builder to verify
the metadata without special-casing the source manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import canonical_json_sha256, sha256_file  # noqa: E402
from data.manifest import ManifestRecord, load_manifest  # noqa: E402


COMPONENT = "spring-v3-1-manifest-augmentation"
SCHEMA_VERSION = 1
METADATA_CONTRACT = "spring_rectified_camera_info_v1"


def _atomic_text(path: Path, text: str) -> None:
    """Write an immutable text artifact, accepting only identical repeats."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing artifact differs; refusing overwrite: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _calibration_key(record: ManifestRecord) -> str:
    return canonical_json_sha256(
        {
            "K": [list(row) for row in record.K],
            "baseline_m": float(record.baseline_m),
        }
    )


def _projection_matrices(
    record: ManifestRecord,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    k = [[float(value) for value in row] for row in record.K]
    fx = float(k[0][0])
    baseline = float(record.baseline_m)
    if not (fx > 0.0 and baseline > 0.0):
        raise ValueError(
            f"Spring calibration requires positive fx/baseline at "
            f"{record.sequence_id}/{record.frame_id}"
        )
    p_left = [row + [0.0] for row in k]
    p_right = [row[:] for row in p_left]
    p_right[0][3] = -fx * baseline
    return k, p_left, p_right


def _image_size_wh(record: ManifestRecord) -> list[int]:
    """Return image size in the manifest convention ``[width, height]``."""

    shape = record.extras.get("image_shape_hw")
    if (
        isinstance(shape, list)
        and len(shape) == 2
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in shape
        )
    ):
        return [int(shape[1]), int(shape[0])]
    # Older hand-authored manifests may omit image_shape_hw.  Decode both
    # images in that case so a malformed stereo pair cannot be stamped with a
    # guessed size.
    try:
        from PIL import Image

        left_path = Path(record.left_path).expanduser().resolve()
        right_path = Path(record.right_path).expanduser().resolve()
        with Image.open(left_path) as left:
            size = tuple(int(value) for value in left.size)
        with Image.open(right_path) as right:
            if tuple(int(value) for value in right.size) != size:
                raise ValueError("left/right image dimensions differ")
        return [size[0], size[1]]
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot resolve image_size_wh at {record.sequence_id}/{record.frame_id}"
        ) from exc


def _flatten(matrix: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _metadata_payload(record: ManifestRecord) -> dict[str, Any]:
    k_right, p_left, p_right = _projection_matrices(record)
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": METADATA_CONTRACT,
        "dataset": "Spring",
        "rectified": True,
        "left_frame_id": "spring_left",
        "right_frame_id": "spring_right",
        "stereo_baseline_m": float(record.baseline_m),
        "left_rect_camera_info": {
            "r": identity,
            "k": _flatten(k_right),
            "p": _flatten(p_left),
        },
        "right_rect_camera_info": {
            "r": identity,
            "k": _flatten(k_right),
            "p": _flatten(p_right),
        },
    }


def _metadata_text(record: ManifestRecord) -> str:
    return yaml.safe_dump(
        _metadata_payload(record),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=False,
    )


def augment_manifest(
    *,
    input_manifest: str | Path,
    output_manifest: str | Path,
    metadata_root: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Augment one Spring JSONL manifest and return its receipt payload."""

    source = Path(input_manifest).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    metadata_dir = Path(metadata_root).expanduser().resolve()
    receipt = (
        output.with_suffix(output.suffix + ".augmentation.json")
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if output == source:
        raise ValueError("output manifest must differ from input manifest")
    if receipt in {source, output}:
        raise ValueError("receipt path must differ from source/output manifest")
    records = load_manifest(source)
    if not records:
        raise ValueError("input manifest is empty")
    if any(
        str(record.extras.get("dataset", "")).strip().lower() != "spring"
        for record in records
    ):
        raise ValueError("input manifest must contain only dataset=Spring records")

    source_sha = sha256_file(source)
    metadata_by_key: dict[str, tuple[Path, str]] = {}
    augmented: list[ManifestRecord] = []
    for record in records:
        key = _calibration_key(record)
        metadata = metadata_by_key.get(key)
        if metadata is None:
            metadata_path = metadata_dir / f"{key}.yaml"
            _atomic_text(metadata_path, _metadata_text(record))
            metadata = (metadata_path, sha256_file(metadata_path))
            metadata_by_key[key] = metadata
        metadata_path, metadata_sha = metadata
        k_right, p_left, p_right = _projection_matrices(record)
        image_size_wh = _image_size_wh(record)
        extras = dict(record.extras)
        expected: dict[str, Any] = {
            "K_right": k_right,
            "P_left": p_left,
            "P_right": p_right,
            "image_size_wh": image_size_wh,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha,
        }
        for name, value in expected.items():
            if name in extras and extras[name] != value:
                raise ValueError(
                    f"existing {name} differs at {record.sequence_id}/{record.frame_id}"
                )
            extras[name] = value
        extras["baseline_from_projection_m"] = float(record.baseline_m)
        extras["spring_v3_1_calibration_augmented"] = True
        extras["spring_v3_1_calibration_metadata_contract"] = METADATA_CONTRACT
        extras["spring_v3_1_calibration_key"] = key
        extras["spring_v3_1_source_manifest"] = str(source)
        extras["spring_v3_1_source_manifest_sha256"] = source_sha
        augmented.append(
            ManifestRecord(
                sequence_id=record.sequence_id,
                frame_id=record.frame_id,
                timestamp=record.timestamp,
                left_path=record.left_path,
                right_path=record.right_path,
                K=record.K,
                baseline_m=record.baseline_m,
                gt_disparity_path=record.gt_disparity_path,
                rectified=True,
                extras=extras,
            )
        )

    # Build the exact JSONL bytes before touching the output path, then accept
    # only a byte-identical repeat.
    output_bytes = b"".join(
        (
            json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for record in augmented
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        if output.read_bytes() != output_bytes:
            raise RuntimeError(f"existing augmented manifest differs; refusing overwrite: {output}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(output_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    validated = load_manifest(output)
    if len(validated) != len(records):
        raise AssertionError("augmented manifest record count changed")
    output_sha = sha256_file(output)
    receipt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "contract": METADATA_CONTRACT,
        "source": {"path": str(source), "sha256": source_sha, "records": len(records)},
        "output": {"path": str(output), "sha256": output_sha, "records": len(validated)},
        "metadata": {
            "root": str(metadata_dir),
            "files": [
                {"path": str(path), "sha256": sha}
                for path, sha in sorted(metadata_by_key.values(), key=lambda item: str(item[0]))
            ],
            "unique_calibrations": len(metadata_by_key),
        },
        "derivation": {
            "K_right": "copied from K_left for rectified Spring rig",
            "P_left": "[K_left|0]",
            "P_right": "[K_left|(-fx*baseline_m,0,0)]",
            "image_size_wh": "reverse of source image_shape_hw [height,width]",
            "source_unchanged": True,
        },
    }
    _atomic_json(receipt, receipt_payload)
    receipt_payload["receipt_path"] = str(receipt)
    receipt_payload["receipt_sha256"] = sha256_file(receipt)
    return receipt_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment a Spring manifest for strict calibrated v3.1 geometry"
    )
    parser.add_argument("--input", "--manifest", dest="input_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = augment_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output,
        metadata_root=args.metadata_root,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

