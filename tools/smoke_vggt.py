#!/usr/bin/env python3
"""Run a real frozen VGGT-Omega causal-window CUDA smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools._m0_status import (
    Status,
    add_check,
    base_receipt,
    finalize,
    git_snapshot,
    sha256_file,
)
from tools.cache_vggt import (
    CONTEXT_PAIRS,
    VIEW_COUNT,
    VIEW_ORDER,
    build_causal_stereo_windows,
)


DEFAULT_REPO = PROJECT_ROOT / "third_party" / "vggt-omega"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the real VGGT-Omega checkpoint on 10 real images."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=os.environ.get("VGGT_OMEGA_CKPT")
    )
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        help="Exactly 10 real images ordered L[t-4],R[t-4],...,L[t],R[t]",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Alternative to --images: derive one validated causal window",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Zero-based causal-window index when --manifest is used",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--input-mode", choices=["balanced", "max_size"], default="balanced"
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--sequence-metadata",
        type=Path,
        help="Optional JSON provenance supplement for explicit --images",
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("reports/m0/smoke_vggt.json")
    )
    return parser.parse_args()


def load_lock() -> dict[str, Any]:
    return json.loads(
        (PROJECT_ROOT / "third_party" / "LOCK.json").read_text(encoding="utf-8")
    )["components"]["vggt-omega"]


def tensor_stats(tensor: Any) -> dict[str, Any]:
    import torch

    finite = torch.isfinite(tensor)
    stats: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_fraction": float(finite.float().mean().item()),
    }
    if tensor.numel() and bool(finite.any().item()):
        values = tensor[finite].float()
        stats |= {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }
    return stats


def image_record(path: Path, index: int) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "index": index,
        "view_label": VIEW_ORDER[index],
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
    }


def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[list[Path] | None, Any | None, dict[str, Any] | None, list[str]]:
    """Resolve explicit images or one manifest window without inference."""

    missing: list[str] = []
    calibrated_k = None
    provenance: dict[str, Any] | None = None
    if args.images is not None and args.manifest is not None:
        return None, None, None, ["choose only one of --images or --manifest"]
    if args.images is not None:
        if len(args.images) != VIEW_COUNT:
            missing.append(
                f"exactly {VIEW_COUNT} ordered real images (received {len(args.images)})"
            )
            return None, None, None, missing
        paths = [path.resolve() for path in args.images]
        missing.extend(f"image: {path}" for path in paths if not path.is_file())
        return paths, calibrated_k, provenance, missing
    if args.manifest is not None:
        if not args.manifest.is_file():
            return None, None, None, [f"manifest: {args.manifest}"]
        if args.window_index < 0:
            return None, None, None, ["--window-index must be non-negative"]
        try:
            from data.manifest import load_manifest

            records = load_manifest(args.manifest)
            windows = build_causal_stereo_windows(records)
            if args.window_index >= len(windows):
                return None, None, None, [
                    f"window-index {args.window_index} outside available range "
                    f"0..{len(windows) - 1}"
                ]
            window = windows[args.window_index]
            paths = list(window.ordered_image_paths(args.manifest))
            missing.extend(f"image: {path}" for path in paths if not path.is_file())
            calibrated_k = window.calibrated_intrinsics_ordered()
            provenance = {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256_file(args.manifest),
                "window_index": args.window_index,
                "manifest_indices": list(window.manifest_indices),
                "sequence_id": window.target.sequence_id,
                "target_frame_id": window.target.frame_id,
                "target_timestamp": window.target.timestamp,
                "ordered_manifest_records": [record.to_dict() for record in window.records],
            }
            return paths, calibrated_k, provenance, missing
        except Exception as exc:
            return None, None, None, [f"manifest/window: {type(exc).__name__}: {exc}"]
    return None, None, None, [
        "--images (10 ordered paths) or --manifest (at least 5 same-sequence records)"
    ]


def main() -> int:
    args = parse_args()
    receipt = base_receipt("smoke:vggt-omega")
    receipt["image_order_contract"] = ",".join(VIEW_ORDER)
    receipt["arguments"] = {
        "input_mode": args.input_mode,
        "image_resolution": args.image_resolution,
        "patch_size": args.patch_size,
        "context_pairs": CONTEXT_PAIRS,
        "causal": True,
    }

    if not args.repo.is_dir():
        add_check(
            receipt,
            "upstream_repository",
            Status.BLOCKED,
            {"missing": str(args.repo)},
        )
        return finalize(receipt, Status.BLOCKED, args.json_out)

    lock = load_lock()
    repo_state = git_snapshot(args.repo)
    receipt["upstream"] = repo_state | {"expected": lock}
    repo_ok = (
        repo_state["head"] == lock["commit"]
        and repo_state["remote"].rstrip("/") == lock["remote"].rstrip("/")
        and not repo_state["dirty"]
    )
    add_check(
        receipt,
        "upstream_identity",
        Status.PASS if repo_ok else Status.FAIL,
        repo_state,
    )
    if not repo_ok:
        return finalize(receipt, Status.FAIL, args.json_out)

    missing: list[str] = []
    if args.checkpoint is None:
        missing.append("--checkpoint or VGGT_OMEGA_CKPT (vggt_omega_1b_512.pt)")
    elif not args.checkpoint.is_file():
        missing.append(f"checkpoint: {args.checkpoint}")
    image_paths, calibrated_k, provenance, input_missing = _resolve_inputs(args)
    missing.extend(input_missing)
    if args.sequence_metadata is not None and not args.sequence_metadata.is_file():
        missing.append(f"sequence metadata: {args.sequence_metadata}")
    if missing:
        add_check(receipt, "required_inputs", Status.BLOCKED, {"missing": missing})
        receipt["access_note"] = (
            "A real approved local VGGT-Omega checkpoint and real causal images "
            "are mandatory; this smoke test never substitutes synthetic success."
        )
        return finalize(receipt, Status.BLOCKED, args.json_out)
    assert args.checkpoint is not None and image_paths is not None
    add_check(receipt, "required_inputs", Status.PASS)

    checkpoint = args.checkpoint.resolve()
    receipt["checkpoint"] = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
    }
    receipt["inputs"] = [
        image_record(path, index) for index, path in enumerate(image_paths)
    ]
    if provenance is not None:
        receipt["causal_window_provenance"] = provenance
    if args.sequence_metadata is not None:
        receipt["sequence_metadata"] = {
            "path": str(args.sequence_metadata.resolve()),
            "sha256": sha256_file(args.sequence_metadata),
            "content": json.loads(args.sequence_metadata.read_text(encoding="utf-8")),
        }

    try:
        sys.path.insert(0, str(args.repo.resolve()))
        import torch

        from backbones.vggt_omega_adapter import VGGTOmegaAdapter
        from vggt_omega.models import VGGTOmega

        add_check(
            receipt,
            "official_api_import",
            Status.PASS,
            {
                "model": "vggt_omega.models.VGGTOmega",
                "adapter": "backbones.vggt_omega_adapter.VGGTOmegaAdapter",
                "torch": torch.__version__,
            },
        )
    except Exception as exc:
        add_check(
            receipt,
            "official_api_import",
            Status.FAIL,
            f"{type(exc).__name__}: {exc}",
        )
        return finalize(receipt, Status.FAIL, args.json_out)

    try:
        if not torch.cuda.is_available():
            add_check(receipt, "cuda_available", Status.FAIL, False)
            return finalize(receipt, Status.FAIL, args.json_out)

        model = VGGTOmega()
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.requires_grad_(False).eval().cuda()
        adapter = VGGTOmegaAdapter(
            model,
            input_mode=args.input_mode,
            image_resolution=args.image_resolution,
            patch_size=args.patch_size,
            context_pairs=CONTEXT_PAIRS,
        )
        frozen = not model.training and not any(
            parameter.requires_grad for parameter in model.parameters()
        )
        add_check(
            receipt,
            "model_frozen_eval",
            Status.PASS if frozen else Status.FAIL,
            frozen,
        )
        if not frozen:
            return finalize(receipt, Status.FAIL, args.json_out)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = adapter(
            image_paths,
            intrinsics_calibrated_original=calibrated_k,
        )
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started

        outputs = {
            "depth_camera_z_arbitrary_scale": tensor_stats(output.depth),
            "depth_confidence_unbounded_gt_one": tensor_stats(output.depth_conf),
            "pose_encoding": tensor_stats(output.pose_enc),
            "extrinsics_camera_from_world_opencv": tensor_stats(output.extrinsics),
            "predicted_intrinsics_diagnostic_only": tensor_stats(output.intrinsics_pred),
            "camera_tokens": tensor_stats(output.camera_tokens),
            "register_tokens": tensor_stats(output.register_tokens),
        }
        if output.intrinsics_calibrated_original is not None:
            outputs["calibrated_intrinsics_original_geometry_owner"] = tensor_stats(
                output.intrinsics_calibrated_original
            )
            outputs["calibrated_intrinsics_model_geometry_owner"] = tensor_stats(
                output.intrinsics_calibrated_model
            )
        receipt["outputs"] = outputs
        receipt["adapter_metadata"] = dict(output.metadata)
        receipt["preprocessing"] = [
            item.as_dict() for item in output.preprocessing
        ]
        receipt["runtime"] = {
            "elapsed_seconds": elapsed_s,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        }

        all_finite = all(
            value["finite_fraction"] == 1.0 for value in outputs.values()
        )
        sequence_ok = (
            output.depth.shape[1] == VIEW_COUNT
            and output.extrinsics.shape[1] == VIEW_COUNT
        )
        tokens_ok = (
            output.camera_tokens.shape[2] == 1
            and output.register_tokens.shape[2] >= 1
        )
        transforms_ok = all(
            len(item.original_to_model_3x3) == 3
            and len(item.model_to_original_3x3) == 3
            for item in output.preprocessing
        )
        add_check(
            receipt,
            "outputs_all_finite",
            Status.PASS if all_finite else Status.FAIL,
            all_finite,
        )
        add_check(
            receipt,
            "causal_sequence_length_10",
            Status.PASS if sequence_ok else Status.FAIL,
            sequence_ok,
        )
        add_check(
            receipt,
            "camera_and_register_tokens_split",
            Status.PASS if tokens_ok else Status.FAIL,
            tokens_ok,
        )
        add_check(
            receipt,
            "preprocessing_transforms_recorded",
            Status.PASS if transforms_ok else Status.FAIL,
            transforms_ok,
        )
        passed = all_finite and sequence_ok and tokens_ok and transforms_ok
        return finalize(
            receipt,
            Status.PASS if passed else Status.FAIL,
            args.json_out,
        )
    except Exception as exc:
        add_check(receipt, "inference", Status.FAIL, f"{type(exc).__name__}: {exc}")
        return finalize(receipt, Status.FAIL, args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
