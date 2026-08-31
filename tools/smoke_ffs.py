#!/usr/bin/env python3
"""Run a real, non-interactive Fast-FoundationStereo CUDA smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools._m0_status import Status, add_check, base_receipt, finalize, git_snapshot, sha256_file
except ModuleNotFoundError:  # Direct execution: python tools/smoke_ffs.py
    from _m0_status import Status, add_check, base_receipt, finalize, git_snapshot, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "Fast-FoundationStereo"
DEFAULT_LEFT = DEFAULT_REPO / "demo_data" / "left.png"
DEFAULT_RIGHT = DEFAULT_REPO / "demo_data" / "right.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("FFS_FAST_CKPT"))
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--volume-backend", choices=["pytorch1", "triton"], default="pytorch1")
    parser.add_argument(
        "--missing-normalize",
        choices=["error", "true", "false"],
        default="error",
        help=(
            "Explicit compatibility policy when a serialized checkpoint lacks "
            "model.args.normalize; never inferred silently"
        ),
    )
    parser.add_argument(
        "--disable-dynamo",
        action="store_true",
        help="Disable upstream torch.compile wrappers for a recorded correctness fallback",
    )
    parser.add_argument("--json-out", type=Path, default=Path("reports/m0/smoke_ffs.json"))
    return parser.parse_args()


def load_lock() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "third_party" / "LOCK.json").read_text(encoding="utf-8"))[
        "components"
    ]["Fast-FoundationStereo"]


def image_tensor(path: Path, torch_module: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy()
    tensor = torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
    }


def main() -> int:
    args = parse_args()
    receipt = base_receipt("smoke:ffs")
    receipt["arguments"] = {
        "iters": args.iters,
        "max_disp": args.max_disp,
        "requested_volume_backend": args.volume_backend,
        "disable_dynamo": args.disable_dynamo,
        "missing_normalize": args.missing_normalize,
    }

    missing: list[str] = []
    if args.checkpoint is None:
        missing.append("--checkpoint or FFS_FAST_CKPT")
    for label, path in (("FFS repository", args.repo), ("left image", args.left), ("right image", args.right)):
        if not path.exists():
            missing.append(f"{label}: {path}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        missing.append(f"checkpoint: {args.checkpoint}")
    if missing:
        add_check(receipt, "required_inputs", Status.BLOCKED, {"missing": missing})
        return finalize(receipt, Status.BLOCKED, args.json_out)
    add_check(receipt, "required_inputs", Status.PASS)

    lock = load_lock()
    repo_state = git_snapshot(args.repo)
    receipt["upstream"] = repo_state | {"expected": lock}
    repo_ok = (
        repo_state["head"] == lock["commit"]
        and repo_state["remote"].rstrip("/") == lock["remote"].rstrip("/")
        and not repo_state["dirty"]
    )
    add_check(receipt, "upstream_identity", Status.PASS if repo_ok else Status.FAIL, repo_state)
    if not repo_ok:
        return finalize(receipt, Status.FAIL, args.json_out)

    checkpoint = args.checkpoint.resolve()
    receipt["checkpoint"] = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "provenance_note": "Hash recorded by this run; source-specific verification is reported separately.",
    }
    cfg_path = checkpoint.parent / "cfg.yaml"
    if cfg_path.is_file():
        receipt["checkpoint"]["cfg_path"] = str(cfg_path)
        receipt["checkpoint"]["cfg_sha256"] = sha256_file(cfg_path)

    if args.iters <= 0 or args.max_disp <= 0 or args.max_disp % 4:
        add_check(receipt, "inference_parameters", Status.FAIL, "iters > 0 and max_disp > 0 divisible by 4")
        return finalize(receipt, Status.FAIL, args.json_out)

    if args.disable_dynamo:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    try:
        sys.path.insert(0, str(args.repo.resolve()))
        import torch

        from core.utils.utils import InputPadder

        # Importing registers aliases needed by upstream full-model pickle files.
        import core.foundation_stereo  # noqa: F401
        from Utils import AMP_DTYPE

        if not torch.cuda.is_available():
            add_check(receipt, "cuda_available", Status.FAIL, False)
            return finalize(receipt, Status.FAIL, args.json_out)

        left_cpu, left_meta = image_tensor(args.left, torch)
        right_cpu, right_meta = image_tensor(args.right, torch)
        receipt["inputs"] = {"left": left_meta, "right": right_meta, "range": "RGB float32 0..255"}
        if left_cpu.shape != right_cpu.shape:
            add_check(
                receipt,
                "stereo_shape_match",
                Status.FAIL,
                {"left": list(left_cpu.shape), "right": list(right_cpu.shape)},
            )
            return finalize(receipt, Status.FAIL, args.json_out)

        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        compatibility_fallback = None
        checkpoint_normalize = model.args.get("normalize")
        if checkpoint_normalize is None:
            if args.missing_normalize == "error":
                add_check(
                    receipt,
                    "checkpoint_arg_normalize",
                    Status.FAIL,
                    "checkpoint is missing args.normalize; pass an explicit compatibility policy",
                )
                return finalize(receipt, Status.FAIL, args.json_out)
            checkpoint_normalize = args.missing_normalize == "true"
            model.args.normalize = checkpoint_normalize
            compatibility_fallback = (
                "injected missing model.args.normalize="
                f"{checkpoint_normalize}; upstream volume helper default is true"
            )
            add_check(
                receipt,
                "checkpoint_arg_normalize",
                Status.PASS_WITH_FALLBACK,
                compatibility_fallback,
            )
        else:
            add_check(
                receipt,
                "checkpoint_arg_normalize",
                Status.PASS,
                bool(checkpoint_normalize),
            )
        model.requires_grad_(False).eval()
        model.args.valid_iters = args.iters
        model.args.max_disp = args.max_disp
        frozen = not model.training and not any(parameter.requires_grad for parameter in model.parameters())
        add_check(receipt, "model_frozen_eval", Status.PASS if frozen else Status.FAIL, frozen)
        if not frozen:
            return finalize(receipt, Status.FAIL, args.json_out)

        left = left_cpu.cuda(non_blocking=False)
        right = right_cpu.cuda(non_blocking=False)
        padder = InputPadder(left.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left, right)
        padded_ok = left_padded.shape[-2] % 32 == 0 and left_padded.shape[-1] % 32 == 0
        receipt["padding"] = {
            "original_shape": list(left.shape),
            "padded_shape": list(left_padded.shape),
            "padding_lrtb": list(padder._pad),
        }
        add_check(receipt, "padding_multiple_32", Status.PASS if padded_ok else Status.FAIL, receipt["padding"])

        model = model.cuda()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        actual_backend = args.volume_backend
        fallback_error = None
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                disparity_padded = model(
                    left_padded,
                    right_padded,
                    iters=args.iters,
                    test_mode=True,
                    optimize_build_volume=actual_backend,
                )
        except Exception as exc:
            if args.volume_backend == "pytorch1":
                raise
            fallback_error = f"{type(exc).__name__}: {exc}"
            actual_backend = "pytorch1"
            torch.cuda.empty_cache()
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                disparity_padded = model(
                    left_padded,
                    right_padded,
                    iters=args.iters,
                    test_mode=True,
                    optimize_build_volume=actual_backend,
                )
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started
        disparity = padder.unpad(disparity_padded.float())

        finite_mask = torch.isfinite(disparity)
        finite_fraction = float(finite_mask.float().mean().item())
        expected_shape = (left.shape[0], 1, left.shape[-2], left.shape[-1])
        shape_ok = tuple(disparity.shape) == expected_shape
        finite_ok = finite_fraction == 1.0
        receipt["output"] = {
            "shape": list(disparity.shape),
            "dtype": str(disparity.dtype),
            "unit": "input-image pixels",
            "finite_fraction": finite_fraction,
            "negative_fraction": float((disparity < 0).float().mean().item()),
            "min": float(disparity.min().item()),
            "max": float(disparity.max().item()),
            "mean": float(disparity.mean().item()),
        }
        receipt["runtime"] = {
            "elapsed_seconds_including_first_compile": elapsed_s,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "actual_volume_backend": actual_backend,
            "fallback_error": fallback_error,
            "compatibility_fallback": compatibility_fallback,
        }
        add_check(receipt, "output_shape", Status.PASS if shape_ok else Status.FAIL, receipt["output"]["shape"])
        add_check(receipt, "output_all_finite", Status.PASS if finite_ok else Status.FAIL, finite_fraction)
        add_check(
            receipt,
            "backend",
            Status.PASS_WITH_FALLBACK if fallback_error or compatibility_fallback else Status.PASS,
            receipt["runtime"],
        )
        if not (padded_ok and shape_ok and finite_ok):
            return finalize(receipt, Status.FAIL, args.json_out)
        return finalize(
            receipt,
            Status.PASS_WITH_FALLBACK if fallback_error or compatibility_fallback else Status.PASS,
            args.json_out,
        )
    except Exception as exc:
        add_check(receipt, "inference", Status.FAIL, f"{type(exc).__name__}: {exc}")
        return finalize(receipt, Status.FAIL, args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
