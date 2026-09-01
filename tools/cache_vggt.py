#!/usr/bin/env python3
"""Build versioned offline VGGT-Omega caches from causal stereo windows.

Each output record is owned by the current (last) manifest timepoint and is
constructed from exactly five same-sequence timepoints ordered as
``L[t-4], R[t-4], ..., L[t], R[t]``.  VGGT-Omega geometry remains in its
native arbitrary scale here; metric baseline scaling and FFS alignment are
separate geometry stages.

CUDA is the default and formal inference path.  An explicit ``--device cpu``
mode is provided for bounded screening when CUDA capacity is unavailable; its
receipt records the device and the upstream-autocast compatibility shim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F

from backbones.vggt_omega_adapter import (
    ImagePreprocessMetadata,
    VGGTOmegaAdapter,
    VGGTOmegaOutput,
)
from data.cache_dataset import (
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
    sha256_file,
)
from data.manifest import ManifestRecord, load_manifest


CONTEXT_PAIRS = 5
VIEW_COUNT = 2 * CONTEXT_PAIRS
CURRENT_LEFT_VIEW_INDEX = 2 * (CONTEXT_PAIRS - 1)
VIEW_ORDER = tuple(
    side
    for offset in range(-(CONTEXT_PAIRS - 1), 1)
    for side in (
        "L[t]" if offset == 0 else f"L[t{offset:+d}]",
        "R[t]" if offset == 0 else f"R[t{offset:+d}]",
    )
)


@dataclass(frozen=True, slots=True)
class CausalStereoWindow:
    """Five calibrated stereo records ending at one current timepoint."""

    records: tuple[ManifestRecord, ...]
    manifest_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.records) != CONTEXT_PAIRS:
            raise ValueError(
                f"a VGGT-Omega window requires {CONTEXT_PAIRS} records, "
                f"got {len(self.records)}"
            )
        if len(self.manifest_indices) != CONTEXT_PAIRS:
            raise ValueError("records and manifest_indices must have equal length")
        sequence_ids = {record.sequence_id for record in self.records}
        if len(sequence_ids) != 1:
            raise ValueError("a causal window cannot cross sequence boundaries")
        timestamps = [record.timestamp for record in self.records]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("window timestamps must be strictly increasing")
        frame_ids = [record.frame_id for record in self.records]
        if any(current <= previous for previous, current in zip(frame_ids, frame_ids[1:])):
            raise ValueError("window frame_id values must be strictly increasing")

    @property
    def target(self) -> ManifestRecord:
        return self.records[-1]

    @property
    def target_manifest_index(self) -> int:
        return self.manifest_indices[-1]

    def ordered_image_paths(self, manifest_path: Path) -> tuple[Path, ...]:
        """Resolve paths in causal ``L,R`` order without using future frames."""

        paths: list[Path] = []
        for record in self.records:
            paths.append(_resolve_manifest_path(record.left_path, manifest_path))
            paths.append(_resolve_manifest_path(record.right_path, manifest_path))
        return tuple(paths)

    def calibrated_intrinsics_ordered(self) -> torch.Tensor:
        """Return calibrated K as ``[10,3,3]`` in the exact image order."""

        matrices: list[Any] = []
        for record in self.records:
            matrices.append(record.K)
            matrices.append(record.extras.get("K_right", record.K))
        intrinsics = torch.tensor(matrices, dtype=torch.float32)
        if tuple(intrinsics.shape) != (VIEW_COUNT, 3, 3):
            raise ValueError(
                f"ordered calibrated intrinsics must be [10,3,3], got {intrinsics.shape}"
            )
        if not bool(torch.isfinite(intrinsics).all().item()):
            raise ValueError("calibrated intrinsics contain NaN or infinity")
        return intrinsics


def build_causal_stereo_windows(
    records: Sequence[ManifestRecord],
    *,
    context_pairs: int = CONTEXT_PAIRS,
) -> list[CausalStereoWindow]:
    """Build all no-future windows while preserving manifest target order.

    Records for a given sequence may occupy separate manifest blocks, but
    their manifest order must be strictly increasing in both timestamp and
    frame ID. No sorting is performed, so an out-of-order source is rejected
    rather than silently rewritten.
    """

    if context_pairs != CONTEXT_PAIRS:
        raise ValueError(
            f"the fixed MVP contract requires context_pairs={CONTEXT_PAIRS}, "
            f"got {context_pairs}"
        )
    grouped: OrderedDict[str, list[tuple[int, ManifestRecord]]] = OrderedDict()
    for manifest_index, record in enumerate(records):
        grouped.setdefault(record.sequence_id, []).append((manifest_index, record))

    windows: list[CausalStereoWindow] = []
    for sequence_id, indexed_records in grouped.items():
        timestamps = [record.timestamp for _, record in indexed_records]
        frame_ids = [record.frame_id for _, record in indexed_records]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(
                f"sequence {sequence_id!r} is not strictly timestamp-ordered in manifest"
            )
        if any(current <= previous for previous, current in zip(frame_ids, frame_ids[1:])):
            raise ValueError(
                f"sequence {sequence_id!r} is not strictly frame_id-ordered in manifest"
            )
        for end in range(context_pairs - 1, len(indexed_records)):
            selected = indexed_records[end - context_pairs + 1 : end + 1]
            window = CausalStereoWindow(
                records=tuple(record for _, record in selected),
                manifest_indices=tuple(index for index, _ in selected),
            )
            # Make the no-future invariant executable, not merely documentary.
            if any(record.timestamp > window.target.timestamp for record in window.records):
                raise AssertionError("causal window contains a future timestamp")
            windows.append(window)

    windows.sort(key=lambda item: item.target_manifest_index)
    return windows


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return path.resolve()


def _safe_component(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"cannot create a safe path component from {value!r}")
    return normalized


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _require_pinned_upstream(repo: Path) -> str:
    if not repo.is_dir():
        raise FileNotFoundError(f"VGGT-Omega repository does not exist: {repo}")
    lock_path = PROJECT_ROOT / "third_party" / "LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["components"][
        "vggt-omega"
    ]
    actual = _git_head(repo)
    if actual != lock["commit"]:
        raise RuntimeError(
            f"VGGT-Omega commit mismatch: expected {lock['commit']}, got {actual}"
        )
    if _git_dirty(repo):
        raise RuntimeError(
            "pinned third_party/vggt-omega is dirty; cache generation refuses "
            "an unrecorded upstream modification"
        )
    return actual


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _load_official_model(
    checkpoint: Path,
    repo: Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(repo.resolve()))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError("VGGT-Omega checkpoint must contain a state_dict mapping")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    return model.requires_grad_(False).eval().to(device)


@contextmanager
def _inference_context(device: torch.device):
    """Provide a narrow CPU compatibility shim for the pinned upstream model.

    VGGT-Omega's released ``forward`` wraps both encoder/head passes in a
    CUDA autocast context unconditionally.  The model itself has no CUDA-only
    tensor operations, so an explicit CPU screening run can safely disable
    those contexts for the duration of cache generation.  The monkeypatch is
    process-local and restored even when inference raises; pinned upstream
    source remains untouched and the receipt records the actual device.
    """

    if device.type != "cpu":
        yield
        return

    original_autocast = torch.autocast
    original_bf16_probe = torch.cuda.is_bf16_supported

    def cpu_safe_autocast(device_type: str, *args: Any, **kwargs: Any):
        if str(device_type).lower() == "cuda":
            return nullcontext()
        return original_autocast(device_type, *args, **kwargs)

    torch.autocast = cpu_safe_autocast  # type: ignore[assignment]
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.autocast = original_autocast  # type: ignore[assignment]
        torch.cuda.is_bf16_supported = original_bf16_probe  # type: ignore[assignment]


def _left_scale_transform(
    matrix: tuple[tuple[float, float, float], ...],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[tuple[float, float, float], ...]:
    """Left-multiply a homogeneous pixel transform by an axis scale."""

    return (
        tuple(float(scale_x * value) for value in matrix[0]),
        tuple(float(scale_y * value) for value in matrix[1]),
        tuple(float(value) for value in matrix[2]),
    )


def _resample_output_grid(
    output: VGGTOmegaOutput,
    target_hw: tuple[int, int],
) -> VGGTOmegaOutput:
    """Resample dense outputs to the FFS x2 grid while preserving coordinates.

    VGGT's patch-grid dimensions must be divisible by its patch size, whereas
    Spring's x2 FFS grid is exactly 540x960.  The model is therefore run at
    its native grid and the dense depth/confidence maps are resized once at
    cache time.  The recorded pixel transforms and calibrated model
    intrinsics are composed with the same axis scale, so downstream geometry
    still operates in the target grid's coordinate system.
    """

    if len(target_hw) != 2 or any(int(value) <= 0 for value in target_hw):
        raise ValueError(f"target dense grid must be positive (H,W), got {target_hw!r}")
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    if output.depth.ndim != 5 or output.depth.shape[0] != 1:
        raise ValueError(f"VGGT depth must be [1,S,H,W,1], got {tuple(output.depth.shape)}")
    native_height, native_width = (
        int(output.depth.shape[2]),
        int(output.depth.shape[3]),
    )
    if (native_height, native_width) == (target_height, target_width):
        return output
    if output.depth_conf.ndim != 4 or tuple(output.depth_conf.shape[:2]) != tuple(output.depth.shape[:2]):
        raise ValueError("VGGT depth_conf shape does not match depth batch/sequence")
    scale_x = target_width / native_width
    scale_y = target_height / native_height
    depth_dtype = output.depth.dtype
    conf_dtype = output.depth_conf.dtype
    depth = F.interpolate(
        output.depth[..., 0].reshape(-1, 1, native_height, native_width).float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, output.depth.shape[1], target_height, target_width, 1).to(depth_dtype)
    depth_conf = F.interpolate(
        output.depth_conf.reshape(-1, 1, native_height, native_width).float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, output.depth_conf.shape[1], target_height, target_width).to(conf_dtype)

    preprocessing: list[ImagePreprocessMetadata] = []
    for item in output.preprocessing:
        original_to_target = _left_scale_transform(
            item.original_to_model_3x3,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        # The inverse of diag(scale_x, scale_y, 1) @ original_to_model is
        # model_to_original @ diag(1/scale_x, 1/scale_y, 1).
        inverse_scale = (
            (1.0 / scale_x, 0.0, 0.0),
            (0.0, 1.0 / scale_y, 0.0),
            (0.0, 0.0, 1.0),
        )
        old_inverse = item.model_to_original_3x3
        model_to_target_original = tuple(
            tuple(
                float(
                    old_inverse[row][0] * inverse_scale[0][column]
                    + old_inverse[row][1] * inverse_scale[1][column]
                    + old_inverse[row][2] * inverse_scale[2][column]
                )
                for column in range(3)
            )
            for row in range(3)
        )
        preprocessing.append(
            replace(
                item,
                model_size_hw=(target_height, target_width),
                original_to_model_3x3=original_to_target,
                model_to_original_3x3=model_to_target_original,
            )
        )

    calibrated_model = output.intrinsics_calibrated_model
    if calibrated_model is not None:
        scale = torch.eye(3, dtype=calibrated_model.dtype, device=calibrated_model.device)
        scale[0, 0] = scale_x
        scale[1, 1] = scale_y
        calibrated_model = scale.unsqueeze(0).unsqueeze(0) @ calibrated_model
    metadata = dict(output.metadata)
    metadata.update(
        {
            "native_model_grid_hw": [native_height, native_width],
            "cached_dense_grid_hw": [target_height, target_width],
            "cache_grid_resample": "bilinear_align_corners_false",
        }
    )
    return VGGTOmegaOutput(
        depth=depth,
        depth_conf=depth_conf,
        pose_enc=output.pose_enc,
        extrinsics=output.extrinsics,
        intrinsics_pred=output.intrinsics_pred,
        camera_tokens=output.camera_tokens,
        register_tokens=output.register_tokens,
        preprocessing=tuple(preprocessing),
        intrinsics_calibrated_original=output.intrinsics_calibrated_original,
        intrinsics_calibrated_model=calibrated_model,
        metadata=metadata,
    )


def _window_source_metadata(
    window: CausalStereoWindow,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    digest_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Create the exact source identity used for stale-cache rejection."""

    paths = window.ordered_image_paths(manifest_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing causal-window images: {missing}")
    image_records = [
        {
            "view_index": index,
            "view_label": VIEW_ORDER[index],
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
        for index, path in enumerate(paths)
    ]
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_indices": list(window.manifest_indices),
        "manifest_records": [record.to_dict() for record in window.records],
        "target_manifest_index": window.target_manifest_index,
        "target_sequence_id": window.target.sequence_id,
        "target_frame_id": window.target.frame_id,
        "target_timestamp": window.target.timestamp,
        "ordered_images": image_records,
        "view_order": list(VIEW_ORDER),
        "causal": True,
    }


def validate_cached_source(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> None:
    """Reject source or window changes even when model identity still matches."""

    if actual != expected:
        actual_mapping: Mapping[str, Any] = actual or {}
        keys = sorted(set(actual_mapping) | set(expected))
        differences = {
            key: {"expected": expected.get(key), "actual": actual_mapping.get(key)}
            for key in keys
            if expected.get(key) != actual_mapping.get(key)
        }
        raise CacheMismatchError(
            "cache source mismatch: "
            + json.dumps(differences, sort_keys=True, separators=(",", ":"))
        )


def cache_tensors_from_output(
    output: VGGTOmegaOutput,
    window: CausalStereoWindow,
    *,
    cache_dtype: torch.dtype,
    all_view_dense: bool,
) -> dict[str, torch.Tensor]:
    """Select safe CPU-cache tensors with explicit shape/unit names."""

    if cache_dtype not in {torch.float16, torch.float32}:
        raise ValueError("VGGT dense/token cache dtype must be float16 or float32")
    current_left = CURRENT_LEFT_VIEW_INDEX
    transforms_original_to_model = torch.tensor(
        [item.original_to_model_3x3 for item in output.preprocessing],
        dtype=torch.float32,
        device=output.extrinsics.device,
    )
    transforms_model_to_original = torch.tensor(
        [item.model_to_original_3x3 for item in output.preprocessing],
        dtype=torch.float32,
        device=output.extrinsics.device,
    )
    calibrated_original = output.intrinsics_calibrated_original
    calibrated_model = output.intrinsics_calibrated_model
    if calibrated_original is None or calibrated_model is None:
        raise ValueError("cache generation requires calibrated intrinsics for all views")

    tensors: dict[str, torch.Tensor] = {
        "vggt_depth_current_left_arbitrary": output.depth[
            0, current_left, :, :, 0
        ].unsqueeze(0).to(cache_dtype),
        "vggt_depth_conf_current_left_unbounded": output.depth_conf[
            0, current_left
        ].unsqueeze(0).to(cache_dtype),
        "vggt_extrinsics_camera_from_world": output.extrinsics[0].to(torch.float32),
        "vggt_intrinsics_pred_model_px": output.intrinsics_pred[0].to(torch.float32),
        "vggt_pose_encoding": output.pose_enc[0].to(torch.float32),
        "vggt_camera_tokens": output.camera_tokens[0].to(cache_dtype),
        "vggt_register_tokens": output.register_tokens[0].to(cache_dtype),
        "calibrated_intrinsics_original_px": calibrated_original[0].to(torch.float32),
        "calibrated_intrinsics_model_px": calibrated_model[0].to(torch.float32),
        "original_to_model_transform": transforms_original_to_model,
        "model_to_original_transform": transforms_model_to_original,
        "stereo_baseline_m_by_pair": torch.tensor(
            [record.baseline_m for record in window.records],
            dtype=torch.float32,
            device=output.extrinsics.device,
        ),
    }
    if all_view_dense:
        tensors["vggt_depth_all_views_arbitrary"] = output.depth[0].permute(
            0, 3, 1, 2
        ).to(cache_dtype)
        tensors["vggt_depth_conf_all_views_unbounded"] = output.depth_conf[
            0
        ].unsqueeze(1).to(cache_dtype)
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f"cache tensor {name!r} became non-finite at dtype {tensor.dtype}; "
                "use --cache-dtype float32 rather than writing a corrupt record"
            )
    return tensors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen VGGT-Omega outputs for exact causal 5-pair windows."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "vggt-omega",
    )
    parser.add_argument("--context-pairs", type=int, default=CONTEXT_PAIRS)
    parser.add_argument("--causal", action="store_true", help="Required explicit contract flag")
    parser.add_argument("--input-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help=(
            "inference device; CUDA is the default/formal path, while CPU is "
            "an explicit bounded-screening fallback"
        ),
    )
    parser.add_argument(
        "--output-grid",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "optional cached dense grid (H W); useful when the downstream FFS "
            "grid is not divisible by the VGGT patch size"
        ),
    )
    parser.add_argument(
        "--all-view-dense",
        action="store_true",
        help="Also cache depth/conf for all 10 views; default stores current-left only",
    )
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.context_pairs != CONTEXT_PAIRS:
        raise ValueError(f"MVP context-pairs must be exactly {CONTEXT_PAIRS}")
    if not args.causal:
        raise ValueError("MVP cache generation requires the explicit --causal flag")
    if args.start_window < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("start-window must be non-negative and limit must be positive")
    if args.image_resolution <= 0 or args.patch_size <= 0:
        raise ValueError("image-resolution and patch-size must be positive")
    if args.image_resolution % args.patch_size:
        raise ValueError("image-resolution must be divisible by patch-size")
    if args.output_grid is not None and any(int(value) <= 0 for value in args.output_grid):
        raise ValueError("output-grid dimensions must be positive")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {args.manifest}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA VGGT-Omega cache device is unavailable")

    upstream_commit = _require_pinned_upstream(args.repo)
    records = load_manifest(args.manifest)
    all_windows = build_causal_stereo_windows(records, context_pairs=args.context_pairs)
    selected = all_windows[args.start_window :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError(
            "causal-window selection is empty; each sequence needs at least five records"
        )

    checkpoint_sha256 = sha256_file(args.checkpoint)
    manifest_sha256 = sha256_file(args.manifest)
    config = {
        "schema_version": 1,
        "model": "VGGT-Omega-1B-512",
        "context_pairs": CONTEXT_PAIRS,
        "view_order": list(VIEW_ORDER),
        "causal": True,
        "input_mode": args.input_mode,
        "image_resolution": args.image_resolution,
        "patch_size": args.patch_size,
        "cache_dtype": args.cache_dtype,
        "dense_cache_scope": "all_views" if args.all_view_dense else "current_left_only",
        "current_left_view_index": CURRENT_LEFT_VIEW_INDEX,
        "inference_device": str(device),
        "cpu_upstream_autocast_compat": bool(device.type == "cpu"),
        "output_grid_hw": (
            [int(args.output_grid[0]), int(args.output_grid[1])]
            if args.output_grid is not None
            else None
        ),
        "pose_scale_stage": "not_applied; downstream pose_scale.py owns metric scaling",
        "depth_alignment_stage": "not_applied; downstream align_vggt.py owns FFS alignment",
    }
    identity = CacheIdentity(
        component="vggt-omega",
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if device.type == "cuda" else None,
        config_sha256=canonical_json_sha256(config),
    )
    canonical_receipt_path = args.output / "run_receipt.json"
    existing_canonical_receipt: dict[str, Any] | None = None
    if canonical_receipt_path.is_file():
        existing_canonical_receipt = json.loads(
            canonical_receipt_path.read_text(encoding="utf-8")
        )
        if existing_canonical_receipt.get("identity") != identity.to_dict():
            raise CacheMismatchError(
                "cache root canonical identity differs from this run; choose a new output root"
            )
        if existing_canonical_receipt.get("manifest_sha256") != manifest_sha256:
            raise CacheMismatchError(
                "cache root canonical manifest differs from this run; choose a new output root"
            )

    digest_cache: dict[Path, str] = {}

    def memoized_digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = sha256_file(path)
        return digest_cache[path]

    prepared: list[tuple[CausalStereoWindow, Path, dict[str, Any], int]] = []
    index_rows: list[dict[str, Any]] = []
    for selection_index, window in enumerate(selected, start=args.start_window):
        source = _window_source_metadata(
            window,
            manifest_path=args.manifest,
            manifest_sha256=manifest_sha256,
            digest_file=memoized_digest,
        )
        relative_path = (
            Path(_safe_component(window.target.sequence_id))
            / f"{_safe_component(window.target.frame_id)}.pt"
        )
        cache_path = args.output / relative_path
        if cache_path.exists() and not args.overwrite:
            payload = load_cache_record(cache_path, expected_identity=identity)
            validate_cached_source(payload["metadata"].get("source"), source)
            index_rows.append(
                {
                    "selection_index": selection_index,
                    "target_manifest_index": window.target_manifest_index,
                    "sequence_id": window.target.sequence_id,
                    "frame_id": window.target.frame_id,
                    "timestamp": window.target.timestamp,
                    "cache_path": str(cache_path.resolve()),
                    "status": "reused_identity_and_source_match",
                }
            )
        else:
            prepared.append((window, cache_path, source, selection_index))

    model: torch.nn.Module | None = None
    adapter: VGGTOmegaAdapter | None = None
    started = time.perf_counter()
    try:
        if prepared:
            model = _load_official_model(args.checkpoint, args.repo, device=device)
            adapter = VGGTOmegaAdapter(
                model,
                input_mode=args.input_mode,
                image_resolution=args.image_resolution,
                patch_size=args.patch_size,
                context_pairs=CONTEXT_PAIRS,
            )
            cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
            with _inference_context(device):
                for write_index, (window, cache_path, source, selection_index) in enumerate(
                    prepared, start=1
                ):
                    paths = window.ordered_image_paths(args.manifest)
                    calibrated_k = window.calibrated_intrinsics_ordered()
                    output = adapter(
                        paths,
                        intrinsics_calibrated_original=calibrated_k,
                    )
                    if args.output_grid is not None:
                        output = _resample_output_grid(
                            output,
                            (int(args.output_grid[0]), int(args.output_grid[1])),
                        )
                    tensors = cache_tensors_from_output(
                        output,
                        window,
                        cache_dtype=cache_dtype,
                        all_view_dense=args.all_view_dense,
                    )
                    metadata = {
                        "source": source,
                        "checkpoint": {
                            "path": str(args.checkpoint.resolve()),
                            "size_bytes": args.checkpoint.stat().st_size,
                            "sha256": checkpoint_sha256,
                        },
                        "config": config,
                        "adapter": dict(output.metadata),
                        "preprocessing": [item.as_dict() for item in output.preprocessing],
                        "cache_tensor_semantics": {
                            "dense_grid": "VGGT model-input grid; use recorded transforms",
                            "depth": "positive camera-z in VGGT arbitrary scale; not metric yet",
                            "depth_conf": "unbounded upstream score 1 + exp(logit); not probability",
                            "extrinsics": "OpenCV camera-from-world [R|t], arbitrary translation scale",
                            "intrinsics_pred": "diagnostic only; calibrated K remains geometry owner",
                            "current_left_view_index": CURRENT_LEFT_VIEW_INDEX,
                        },
                        "inference_device": str(device),
                        "cpu_upstream_autocast_compat": bool(device.type == "cpu"),
                    }
                    save_cache_record(
                        cache_path,
                        tensors=tensors,
                        metadata=metadata,
                        identity=identity,
                    )
                    index_rows.append(
                        {
                            "selection_index": selection_index,
                            "target_manifest_index": window.target_manifest_index,
                            "sequence_id": window.target.sequence_id,
                            "frame_id": window.target.frame_id,
                            "timestamp": window.target.timestamp,
                            "cache_path": str(cache_path.resolve()),
                            "status": "written",
                        }
                    )
                    print(f"[{write_index}/{len(prepared)}] {cache_path}")
                    del output, tensors
    finally:
        del adapter, model
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_rows.sort(key=lambda row: int(row["selection_index"]))
    elapsed_seconds = time.perf_counter() - started
    run_receipt = {
            "schema_version": 1,
            "identity": identity.to_dict(),
            "config": config,
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "size_bytes": args.checkpoint.stat().st_size,
                "sha256": checkpoint_sha256,
            },
            "upstream_repo": str(args.repo.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha256,
            "available_windows": len(all_windows),
            "selected_windows": len(selected),
            "written_records": sum(row["status"] == "written" for row in index_rows),
            "reused_records": sum(row["status"].startswith("reused") for row in index_rows),
            "elapsed_seconds": elapsed_seconds,
        }
    selection_end = args.start_window + len(selected) - 1
    selection_tag = f"windows_{args.start_window:06d}_{selection_end:06d}"
    _atomic_jsonl(args.output / "runs" / f"{selection_tag}.jsonl", index_rows)
    _atomic_json(args.output / "runs" / f"{selection_tag}.json", run_receipt)

    # Keep the most complete same-identity run as the root-level canonical
    # inventory. A quick subset reuse probe must not erase a full-cache receipt.
    existing_selected = (
        int(existing_canonical_receipt.get("selected_windows", 0))
        if existing_canonical_receipt is not None
        else 0
    )
    if len(selected) >= existing_selected:
        _atomic_jsonl(args.output / "cache_manifest.jsonl", index_rows)
        _atomic_json(canonical_receipt_path, run_receipt)
    print(
        f"VGGT cache windows={len(index_rows)} written={len(prepared)} "
        f"elapsed={elapsed_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
