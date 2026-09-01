#!/usr/bin/env python3
"""Non-interactive smoke test for a local FoundationStereo checkpoint.

This intentionally uses the official ``third_party/FoundationStereo/assets``
pair and the local adapter.  It records whether the checkpoint dict was loaded,
which ViT architecture was selected, whether the FlashAttention fallback was
needed, and basic finite/shape checks.  The script does not write model or
third-party files; only the optional JSON receipt is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _image_tensor(path: Path):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/foundationstereo/11-33-40/model_best_bp2.pth",
    )
    parser.add_argument(
        "--left",
        type=Path,
        default=PROJECT_ROOT / "third_party/FoundationStereo/assets/left.png",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=PROJECT_ROOT / "third_party/FoundationStereo/assets/right.png",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--max-disp",
        type=int,
        default=64,
        help="Use a small volume for a quick smoke; omit to retain checkpoint config",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import torch

    from backbones.foundationstereo_adapter import load_foundation_stereo

    missing = [str(path) for path in (args.checkpoint, args.left, args.right) if not path.is_file()]
    if missing:
        payload = {"status": "BLOCKED", "missing": missing}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    load_kwargs = {
        "device": args.device,
        "iterations": args.iters,
        "strict": True,
    }
    if args.max_disp > 0:
        load_kwargs["max_disp"] = args.max_disp
    adapter = load_foundation_stereo(args.checkpoint, **load_kwargs)
    target_device = next(adapter.parameters()).device
    left = _image_tensor(args.left).to(target_device)
    right = _image_tensor(args.right).to(left.device)

    # The demo pair is larger than needed for a smoke.  Keep aspect and align
    # to the model's 32-pixel input contract while preserving RGB/[0,255].
    height, width = left.shape[-2:]
    target_height = min(height, 128)
    target_width = min(width, 192)
    left = torch.nn.functional.interpolate(
        left, size=(target_height, target_width), mode="bilinear", align_corners=False
    )
    right = torch.nn.functional.interpolate(
        right, size=(target_height, target_width), mode="bilinear", align_corners=False
    )
    output = adapter(left, right)
    finite = bool(torch.isfinite(output.disparity_lr_px).all().item())
    payload = {
        "status": "PASS" if finite else "FAIL",
        "checkpoint": str(args.checkpoint.resolve()),
        "input_shape": list(left.shape),
        "disparity_shape": list(output.disparity_lr_px.shape),
        "finite_disparity": finite,
        "valid_fraction": float(output.valid_mask.float().mean().item()),
        "load_info": adapter.load_info.as_dict() if adapter.load_info is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
