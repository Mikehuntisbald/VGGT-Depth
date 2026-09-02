#!/usr/bin/env python3
"""Audit joined observation/teacher FFS caches against one JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from data.cache_dataset import CacheIdentity, load_cache_record, sha256_file
from data.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--observation-scale",
        type=int,
        choices=(1, 2),
        default=2,
        help="HR/input scale recorded by the observation cache (default: 2)",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _receipt(root: Path) -> dict:
    with (root / "run_receipt.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_path(root: Path, sequence_id: str, frame_id: int) -> Path:
    return root / sequence_id / f"{frame_id}.pt"


def _sample_indices(record_count: int, sample_count: int) -> list[int]:
    if sample_count <= 0:
        raise ValueError("samples must be positive")
    if sample_count == 1:
        return [0]
    if record_count <= sample_count:
        return list(range(record_count))
    return sorted(
        {
            round(index * (record_count - 1) / (sample_count - 1))
            for index in range(sample_count)
        }
    )


def _assert_tensor_contract(
    observation: dict,
    teacher: dict,
    *,
    hr_height: int,
    hr_width: int,
    scale: int,
) -> dict[str, float]:
    obs = observation["tensors"]
    teach = teacher["tensors"]
    obs_shape = (1, 1, hr_height // scale, hr_width // scale)
    teacher_shape = (1, 1, hr_height, hr_width)
    for name in (
        "observation_disparity_lr_px",
        "observation_disparity_hr_px",
        "observation_confidence",
        "observation_entropy",
        "observation_last_update_magnitude_lr_px",
        "observation_left_right_error_lr_px",
        "observation_valid_mask",
        "observation_trusted_mask",
    ):
        if tuple(obs[name].shape) != obs_shape:
            raise AssertionError(f"{name} has shape {tuple(obs[name].shape)}, expected {obs_shape}")
    for name in (
        "teacher_disparity_hr_px",
        "teacher_confidence",
        "teacher_entropy",
        "teacher_last_update_magnitude_hr_px",
        "teacher_valid_mask",
        "teacher_trusted_mask",
    ):
        # ``cache_ffs.py`` stores teacher tensors without a redundant batch
        # axis ([1,H,W]), while older producers used [1,1,H,W].  Both encode
        # the same singleton-channel HR grid and are accepted explicitly.
        if tuple(teach[name].shape) not in {
            teacher_shape,
            (1, hr_height, hr_width),
        }:
            raise AssertionError(
                f"{name} has shape {tuple(teach[name].shape)}, expected {teacher_shape}"
            )

    scaling_error = (
        obs["observation_disparity_hr_px"].float()
        - scale * obs["observation_disparity_lr_px"].float()
    ).abs().max()
    # The two fields are quantized to float16 independently after the exact
    # float32 conversion. One half-precision bin at this disparity range is an
    # expected cache artifact, not a unit conversion error.
    if float(scaling_error) > 0.02:
        raise AssertionError(f"LR-to-HR disparity scaling error is {float(scaling_error)}")

    for prefix, tensors in (("observation", obs), ("teacher", teach)):
        valid = tensors[f"{prefix}_valid_mask"]
        trusted = tensors[f"{prefix}_trusted_mask"]
        disparity = tensors[f"{prefix}_disparity_hr_px"]
        confidence = tensors[f"{prefix}_confidence"]
        lr_error_name = (
            "observation_left_right_error_lr_px"
            if prefix == "observation"
            else "teacher_left_right_error_hr_px"
        )
        lr_error = tensors.get(lr_error_name)
        if not bool(torch.isfinite(disparity).all()):
            raise AssertionError(f"{prefix} disparity contains non-finite values")
        if not bool(torch.isfinite(confidence).all()):
            raise AssertionError(f"{prefix} confidence contains non-finite values")
        if lr_error is not None and not bool(torch.isfinite(lr_error[valid]).all()):
            raise AssertionError(f"{prefix} LR error is non-finite inside valid mask")
        if bool((trusted & ~valid).any()):
            raise AssertionError(f"{prefix} trusted mask is not a subset of valid")
        if lr_error is not None and bool((trusted & (lr_error >= 1.0)).any()):
            raise AssertionError(f"{prefix} trusted mask contains LR error >= 1 px")
        # Trusted was computed in float32 before confidence was cached in
        # float16, so allow one half-precision quantization bin at 0.8.
        if bool((trusted & (confidence < 0.799)).any()):
            raise AssertionError(f"{prefix} trusted mask contains low confidence")
    return {
        "scale_max_abs_error": float(scaling_error),
        "observation_valid_fraction": float(obs["observation_valid_mask"].float().mean()),
        "observation_trusted_fraction": float(obs["observation_trusted_mask"].float().mean()),
        "teacher_valid_fraction": float(teach["teacher_valid_mask"].float().mean()),
        "teacher_trusted_fraction": float(teach["teacher_trusted_mask"].float().mean()),
    }


def main() -> int:
    args = parse_args()
    records = load_manifest(args.manifest)
    if not records:
        raise ValueError("manifest is empty")
    observation_receipt = _receipt(args.observation_root)
    teacher_receipt = _receipt(args.teacher_root)
    manifest_hash = sha256_file(args.manifest)
    for name, receipt in (
        ("observation", observation_receipt),
        ("teacher", teacher_receipt),
    ):
        if receipt["manifest_sha256"] != manifest_hash:
            raise AssertionError(f"{name} receipt manifest hash mismatch")
        if receipt["selected_records"] != len(records):
            raise AssertionError(f"{name} receipt record count mismatch")
    observed_scale = int(observation_receipt.get("config", {}).get("scale", -1))
    if observed_scale != args.observation_scale:
        raise AssertionError(
            "observation receipt scale mismatch: "
            f"expected {args.observation_scale}, got {observed_scale}"
        )
    observation_identity = CacheIdentity(**observation_receipt["identity"])
    teacher_identity = CacheIdentity(**teacher_receipt["identity"])

    sample_results = []
    for index in _sample_indices(len(records), args.samples):
        record = records[index]
        observation = load_cache_record(
            _record_path(args.observation_root, record.sequence_id, record.frame_id),
            expected_identity=observation_identity,
        )
        teacher = load_cache_record(
            _record_path(args.teacher_root, record.sequence_id, record.frame_id),
            expected_identity=teacher_identity,
        )
        manifest_mapping = record.to_dict()
        for name, payload in (("observation", observation), ("teacher", teacher)):
            cached_record = payload["metadata"]["source"]["manifest_record"]
            if cached_record != manifest_mapping:
                raise AssertionError(f"{name} source record mismatch at index {index}")
        image_size = record.extras.get("image_size_wh")
        if isinstance(image_size, list) and len(image_size) == 2:
            width, height = (int(image_size[0]), int(image_size[1]))
        else:
            image_shape = record.extras.get("image_shape_hw")
            if not isinstance(image_shape, list) or len(image_shape) != 2:
                raise AssertionError(
                    "manifest record lacks image_size_wh/image_shape_hw"
                )
            height, width = (int(image_shape[0]), int(image_shape[1]))
        statistics = _assert_tensor_contract(
            observation,
            teacher,
            hr_height=height,
            hr_width=width,
            scale=args.observation_scale,
        )
        sample_results.append(
            {
                "index": index,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                **statistics,
            }
        )

    report = {
        "status": "PASS",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "record_count": len(records),
        "sample_count": len(sample_results),
        "observation_identity": observation_receipt["identity"],
        "teacher_identity": teacher_receipt["identity"],
        "observation_scale": args.observation_scale,
        "samples": sample_results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"audit:ffs-cache: PASS ({len(sample_results)}/{len(records)} sampled)")
    print(f"receipt: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
