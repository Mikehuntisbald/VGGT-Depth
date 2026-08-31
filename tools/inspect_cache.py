#!/usr/bin/env python3
"""Inspect a versioned cache record and optionally render its scalar maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.cache_dataset import load_cache_record
from utils.visualization import (
    grayscale_to_rgb_uint8,
    save_rgb_uint8,
    scalar_to_rgb_uint8,
    tensor_statistics,
)


def _is_renderable_scalar_map_shape(shape: tuple[int, ...]) -> bool:
    """Match the singleton squeeze rules used by the visualizer."""

    dimensions = list(shape)
    while len(dimensions) > 2 and dimensions[0] == 1:
        dimensions.pop(0)
    if len(dimensions) > 2 and dimensions[-1] == 1:
        dimensions.pop()
    return (
        len(dimensions) == 2
        and dimensions[0] >= 16
        and dimensions[1] >= 16
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_record", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_cache_record(args.cache_record)
    summary = {
        "path": str(args.cache_record.resolve()),
        "schema_version": payload["schema_version"],
        "identity": payload["identity"],
        "metadata": payload["metadata"],
        "tensors": {
            name: tensor_statistics(tensor) for name, tensor in payload["tensors"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_dir is None:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inspection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tensors = payload["tensors"]
    valid_mask = tensors.get("observation_valid_mask")
    if valid_mask is None:
        valid_mask = tensors.get("teacher_valid_mask")
    for name, tensor in tensors.items():
        if not _is_renderable_scalar_map_shape(tuple(tensor.shape)):
            continue
        if name.endswith("valid_mask") or name.endswith("visibility_mask"):
            image = grayscale_to_rgb_uint8(tensor, minimum=0.0, maximum=1.0)
        elif "confidence" in name and "unbounded" not in name:
            image = grayscale_to_rgb_uint8(tensor, valid_mask=valid_mask, minimum=0.0, maximum=1.0)
        elif "entropy" in name:
            image = grayscale_to_rgb_uint8(tensor, valid_mask=valid_mask, minimum=0.0, maximum=1.0)
        else:
            image = scalar_to_rgb_uint8(tensor, valid_mask=valid_mask)
        save_rgb_uint8(args.output_dir / f"{name}.png", image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
