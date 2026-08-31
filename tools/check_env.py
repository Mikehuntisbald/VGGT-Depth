#!/usr/bin/env python3
"""Verify one Blackwell-compatible project environment and emit a receipt."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tools._m0_status import Status, add_check, base_receipt, command_output, finalize
except ModuleNotFoundError:  # Direct execution: python tools/check_env.py
    from _m0_status import Status, add_check, base_receipt, command_output, finalize


PROFILE_MODULES = {
    "ffs": [
        "torch",
        "torchvision",
        "timm",
        "einops",
        "omegaconf",
        "scipy",
        "numpy",
        "skimage",
        "cv2",
        "imageio",
        "yaml",
        "open3d",
    ],
    "vggt": [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "einops",
        "safetensors",
        "cv2",
        "vggt_omega",
    ],
    "tsr": ["torch", "torchvision", "numpy", "PIL", "yaml", "pytest"],
}


def parse_version(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(piece) for piece in re.findall(r"\d+", value)[:3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_MODULES), required=True)
    parser.add_argument("--expect-env", help="Expected conda environment name")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.json_out or Path(f"reports/m0/env_{args.profile}.json")
    receipt = base_receipt(f"environment:{args.profile}")
    actual_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    receipt["python"] = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "version": sys.version,
        "environment": actual_env,
    }

    hard_failure = False
    blocked = False
    if args.expect_env and actual_env != args.expect_env:
        add_check(
            receipt,
            "environment_identity",
            Status.FAIL,
            {"expected": args.expect_env, "actual": actual_env},
        )
        hard_failure = True
    else:
        add_check(receipt, "environment_identity", Status.PASS, actual_env)

    missing: list[str] = []
    import_errors: dict[str, str] = {}
    for name in PROFILE_MODULES[args.profile]:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            import_errors[name] = f"{type(exc).__name__}: {exc}"
    receipt["required_modules"] = PROFILE_MODULES[args.profile]
    if missing:
        add_check(receipt, "required_modules", Status.BLOCKED, {"missing": missing})
        blocked = True
    elif import_errors:
        add_check(receipt, "required_modules", Status.FAIL, {"import_errors": import_errors})
        hard_failure = True
    else:
        add_check(receipt, "required_modules", Status.PASS)

    nvidia_smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    receipt["nvidia_smi"] = nvidia_smi
    add_check(
        receipt,
        "nvidia_smi",
        Status.PASS if nvidia_smi.get("returncode") == 0 else Status.FAIL,
        nvidia_smi.get("stdout") or nvidia_smi.get("error"),
    )
    hard_failure |= nvidia_smi.get("returncode") != 0
    receipt["nvcc"] = command_output(["nvcc", "--version"])

    if importlib.util.find_spec("torch") is None:
        return finalize(receipt, Status.FAIL if hard_failure else Status.BLOCKED, output)

    try:
        import torch

        receipt["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        }
        cuda_ok = torch.cuda.is_available()
        add_check(receipt, "cuda_available", Status.PASS if cuda_ok else Status.FAIL, cuda_ok)
        hard_failure |= not cuda_ok

        runtime_ok = parse_version(torch.version.cuda) >= (12, 8)
        add_check(
            receipt,
            "torch_cuda_at_least_12_8",
            Status.PASS if runtime_ok else Status.FAIL,
            torch.version.cuda,
        )
        hard_failure |= not runtime_ok

        if cuda_ok:
            device = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)
            receipt["gpu"] = {
                "name": device.name,
                "capability": list(capability),
                "total_memory_bytes": device.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            gpu_ok = "RTX 5090" in device.name and capability >= (12, 0)
            add_check(receipt, "rtx_5090_sm120", Status.PASS if gpu_ok else Status.FAIL, receipt["gpu"])
            hard_failure |= not gpu_ok

            for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                try:
                    lhs = torch.ones((256, 256), device="cuda", dtype=dtype)
                    result = lhs @ lhs
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(result).all().item())
                    correct = float(result[0, 0].float().item()) == 256.0
                    ok = finite and correct
                    add_check(
                        receipt,
                        f"cuda_matmul_{dtype_name}",
                        Status.PASS if ok else Status.FAIL,
                        {"finite": finite, "sample": float(result[0, 0].float().item())},
                    )
                    hard_failure |= not ok
                except Exception as exc:  # CUDA errors must be captured in the receipt.
                    add_check(
                        receipt,
                        f"cuda_matmul_{dtype_name}",
                        Status.FAIL,
                        f"{type(exc).__name__}: {exc}",
                    )
                    hard_failure = True
        torch.cuda.empty_cache() if cuda_ok else None
    except Exception as exc:
        add_check(receipt, "torch_runtime", Status.FAIL, f"{type(exc).__name__}: {exc}")
        hard_failure = True

    status = Status.FAIL if hard_failure else Status.BLOCKED if blocked else Status.PASS
    return finalize(receipt, status, output)


if __name__ == "__main__":
    raise SystemExit(main())
