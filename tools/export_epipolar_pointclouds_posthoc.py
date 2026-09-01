#!/usr/bin/env python3
"""Export Stage-C visualization point clouds through the frozen formal evaluator.

This is deliberately a **POSTHOC_DIAGNOSTIC** wrapper.  It executes the
unchanged frozen ``4e6b7eb`` evaluator and lets that evaluator retain every
runtime, source-bundle, checkpoint, cache, and lineage gate.  The only hook
captures the evaluator's already-validated endpoint calibration and appends
colored base/refined camera-frame PLY files after its own visualization call.

No accuracy metric is computed, parsed, modified, or claimed here.  In
particular, point-to-plane remains unavailable without target normals and
correspondences.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE_C_COMMIT = "4e6b7eb488201227e46b30e2ac90d34991466f2c"
FROZEN_EVALUATOR_SHA256 = (
    "d88f69bc49ff8410c3628f5d6db3c9595a4ac791e07d0e72870ee3914468204e"
)
POSTHOC_COMPONENT = "ffs-omega-tsr-stage-c-pointcloud-posthoc"


class PosthocPointCloudError(RuntimeError):
    """Raised when the immutable evaluator or calibrated callback contract fails."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one exact artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PosthocPointCloudError(
            f"cannot inspect frozen evaluator git worktree {root}: {exc}"
        ) from exc
    return completed.stdout.strip()


def verify_frozen_evaluator_source(formal_source_root: Path) -> dict[str, str]:
    """Fail closed unless the exact clean frozen Stage-C evaluator is present."""

    root = formal_source_root.expanduser().resolve()
    evaluator_path = root / "eval_epipolar.py"
    if not evaluator_path.is_file():
        raise PosthocPointCloudError(f"frozen evaluator is missing: {evaluator_path}")
    head = _git(root, "rev-parse", "HEAD")
    if head != FROZEN_STAGE_C_COMMIT:
        raise PosthocPointCloudError(
            "posthoc export requires the exact frozen Stage-C commit "
            f"{FROZEN_STAGE_C_COMMIT}, got {head}"
        )
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PosthocPointCloudError(
            "frozen Stage-C worktree must be clean; refusing runtime source changes"
        )
    evaluator_sha256 = sha256_file(evaluator_path)
    if evaluator_sha256 != FROZEN_EVALUATOR_SHA256:
        raise PosthocPointCloudError(
            "frozen eval_epipolar.py SHA-256 differs from the audited source: "
            f"{evaluator_sha256}"
        )
    return {
        "formal_source_root": str(root),
        "git_commit": head,
        "eval_epipolar_sha256": evaluator_sha256,
    }


def _load_exporter() -> Callable[..., Any]:
    """Load the main-worktree PLY exporter under a private module name.

    The frozen 4e6 source predates its visualization PLY integration.  A
    private module name avoids replacing any frozen evaluator ``metrics``
    imports while still reusing the tested calibrated exporter implementation.
    """

    metrics_root = PROJECT_ROOT / "src" / "metrics"
    package_name = "_posthoc_metrics"
    package = ModuleType(package_name)
    package.__path__ = [str(metrics_root)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    for module_name, filename in (
        (f"{package_name}.disparity", "disparity.py"),
        (f"{package_name}.pointcloud", "pointcloud.py"),
    ):
        path = metrics_root / filename
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PosthocPointCloudError(f"cannot load PLY exporter module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[f"{package_name}.pointcloud"]
    exporter = getattr(module, "export_colored_point_cloud_ply", None)
    if not callable(exporter):
        raise PosthocPointCloudError("calibrated PLY exporter is unavailable")
    return exporter


def _refuse_conflicting_project_modules(formal_source_root: Path) -> None:
    """Avoid accidentally satisfying frozen imports from the mutable worktree."""

    roots = ("eval", "train", "train_epipolar", "data", "evaluation", "metrics", "models", "utils")
    source_root = formal_source_root.resolve()
    for name in roots:
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve().relative_to(source_root)
        except ValueError as exc:
            raise PosthocPointCloudError(
                "refusing dynamic frozen-evaluator import because project module "
                f"{name!r} is already loaded from outside {source_root}: {module_file}"
            ) from exc


def load_frozen_evaluator(formal_source_root: Path) -> ModuleType:
    """Dynamically execute only the verified frozen evaluator source file."""

    root = formal_source_root.expanduser().resolve()
    _refuse_conflicting_project_modules(root)
    sys.path.insert(0, str(root))
    module_name = "_frozen_stage_c_eval_epipolar_4e6b7eb"
    spec = importlib.util.spec_from_file_location(module_name, root / "eval_epipolar.py")
    if spec is None or spec.loader is None:
        raise PosthocPointCloudError("cannot create frozen evaluator import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True, slots=True)
class CapturedEndpointCalibration:
    """One exact endpoint K/baseline captured before frozen evaluator device move."""

    K_hr_px: Tensor
    baseline_m: Tensor


@dataclass(slots=True)
class PosthocPlyCallback:
    """Stateful frozen-evaluator hook with no metric ownership."""

    exporter: Callable[..., Any]
    captured: dict[tuple[int, str, int], CapturedEndpointCalibration]
    records: list[dict[str, Any]]

    @staticmethod
    def _key_from_provenance(provenance: Mapping[str, Any]) -> tuple[int, str, int]:
        try:
            return (
                int(provenance["manifest_index"]),
                str(provenance["sequence_id"]),
                int(provenance["frame_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PosthocPointCloudError(
                "frozen visualization provenance lacks endpoint identity"
            ) from exc

    def capture_batch(self, batch: Mapping[str, Any]) -> None:
        """Capture only CPU endpoint calibration already accepted by frozen checks."""

        K_sequence = batch.get("K_hr_sequence")
        baseline_sequence = batch.get("baseline_m_sequence")
        sequence_ids = batch.get("sequence_id")
        frame_ids = batch.get("frame_ids")
        manifest_indices = batch.get("manifest_indices")
        if (
            not isinstance(K_sequence, Tensor)
            or not isinstance(baseline_sequence, Tensor)
            or not isinstance(frame_ids, Tensor)
            or not isinstance(manifest_indices, Tensor)
            or not isinstance(sequence_ids, list)
            or K_sequence.ndim != 4
            or K_sequence.shape[1:] != (3, 3, 3)
            or baseline_sequence.shape != K_sequence.shape[:2]
            or frame_ids.shape != K_sequence.shape[:2]
            or manifest_indices.shape != K_sequence.shape[:2]
            or len(sequence_ids) != K_sequence.shape[0]
        ):
            raise PosthocPointCloudError(
                "frozen evaluator batch lacks exact endpoint K/baseline tensors"
            )
        for item in range(K_sequence.shape[0]):
            K_hr_px = K_sequence[item, -1].detach().cpu().float().clone()
            baseline_m = baseline_sequence[item, -1].detach().cpu().float().clone()
            if K_hr_px.shape != (3, 3) or not bool(torch.isfinite(K_hr_px).all()):
                raise PosthocPointCloudError("captured endpoint K_hr_px is invalid")
            if not bool(torch.isfinite(baseline_m) and baseline_m > 0):
                raise PosthocPointCloudError("captured endpoint baseline_m is invalid")
            key = (
                int(manifest_indices[item, -1].item()),
                str(sequence_ids[item]),
                int(frame_ids[item, -1].item()),
            )
            if key in self.captured:
                raise PosthocPointCloudError(f"duplicate endpoint calibration key: {key}")
            self.captured[key] = CapturedEndpointCalibration(K_hr_px, baseline_m)

    def save_after_frozen_visualization(
        self,
        root: Path,
        bound_arguments: Mapping[str, Any],
    ) -> None:
        """Append PLYs using the exact captured K/baseline and frozen RGB/disparity."""

        provenance = bound_arguments.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PosthocPointCloudError("frozen visualization provenance is missing")
        key = self._key_from_provenance(provenance)
        calibration = self.captured.pop(key, None)
        if calibration is None:
            raise PosthocPointCloudError(
                f"no captured endpoint calibration for frozen visualization {key}"
            )
        sample_name = bound_arguments.get("sample_name")
        rgb_left_hr = bound_arguments.get("rgb_left_hr")
        base = bound_arguments.get("base_disparity_hr_px")
        refined = bound_arguments.get("refined_disparity_hr_px")
        if not isinstance(sample_name, str) or not all(
            isinstance(value, Tensor) for value in (rgb_left_hr, base, refined)
        ):
            raise PosthocPointCloudError("frozen visualization tensors are missing")
        directory = Path(root) / sample_name
        base_result = self.exporter(
            base,
            rgb_left_hr,
            calibration.K_hr_px,
            calibration.baseline_m,
            directory / "base_point_cloud_camera_frame.ply",
        )
        refined_result = self.exporter(
            refined,
            rgb_left_hr,
            calibration.K_hr_px,
            calibration.baseline_m,
            directory / "refined_point_cloud_camera_frame.ply",
        )
        self.records.append(
            {
                "sample_name": sample_name,
                "endpoint": {
                    "manifest_index": key[0],
                    "sequence_id": key[1],
                    "frame_id": key[2],
                    "K_hr_px": calibration.K_hr_px.tolist(),
                    "baseline_m": float(calibration.baseline_m.item()),
                },
                "coordinate_frame": "left_camera_frame",
                "coordinate_units": "meters",
                "base": {
                    "path": str(base_result.path),
                    "point_count": int(base_result.point_count),
                },
                "refined": {
                    "path": str(refined_result.path),
                    "point_count": int(refined_result.point_count),
                },
            }
        )


def install_posthoc_ply_callback(evaluator: ModuleType) -> PosthocPlyCallback:
    """Wrap only frozen calibration validation and visualization callback globals."""

    original_validate = getattr(evaluator, "validate_epipolar_batch_causality", None)
    original_save = getattr(evaluator, "_save_visualization", None)
    if not callable(original_validate) or not callable(original_save):
        raise PosthocPointCloudError("frozen evaluator callback surface is incompatible")
    callback = PosthocPlyCallback(_load_exporter(), {}, [])
    signature = __import__("inspect").signature(original_save)

    def capture_then_validate(batch: Mapping[str, Any]) -> Any:
        result = original_validate(batch)
        callback.capture_batch(batch)
        return result

    def save_then_export(root: Path, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(root, *args, **kwargs)
        original_save(root, *args, **kwargs)
        callback.save_after_frozen_visualization(root, bound.arguments)

    evaluator.validate_epipolar_batch_causality = capture_then_validate
    evaluator._save_visualization = save_then_export
    return callback


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation-cache-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--derived-cache-root", type=Path, required=True)
    parser.add_argument("--rectification-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def _frozen_evaluator_args(args: argparse.Namespace, evaluator: ModuleType) -> argparse.Namespace:
    """Use the frozen parser with no override or limit/bypass surface."""

    parser = evaluator.build_parser()
    values = [
        "--config", str(args.config.expanduser().resolve()),
        "--checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--base-checkpoint", str(args.base_checkpoint.expanduser().resolve()),
        "--manifest", str(args.manifest.expanduser().resolve()),
        "--observation-cache-root", str(args.observation_cache_root.expanduser().resolve()),
        "--teacher-cache-root", str(args.teacher_cache_root.expanduser().resolve()),
        "--derived-cache-root", str(args.derived_cache_root.expanduser().resolve()),
        "--rectification-audit", str(args.rectification_audit.expanduser().resolve()),
        "--output", str(args.output.expanduser().resolve()),
        "--device", str(args.device),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--visualization-samples", str(args.samples),
    ]
    return parser.parse_args(values)


def run(args: argparse.Namespace) -> int:
    _positive_int(args.samples, "samples")
    _positive_int(args.batch_size, "batch_size")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    source = verify_frozen_evaluator_source(args.formal_source_root)
    for name in ("checkpoint", "base_checkpoint"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluator = load_frozen_evaluator(args.formal_source_root)
    callback = install_posthoc_ply_callback(evaluator)
    output = args.output.expanduser().resolve()
    frozen_args = _frozen_evaluator_args(args, evaluator)
    return_code = evaluator.run(frozen_args)
    if return_code != 0:
        raise PosthocPointCloudError(f"frozen evaluator returned non-zero: {return_code}")
    receipt = {
        "schema_version": 1,
        "component": POSTHOC_COMPONENT,
        "status": "POSTHOC_DIAGNOSTIC_COMPLETE",
        "claim_boundary": {
            "classification": "POSTHOC_DIAGNOSTIC",
            "accuracy_metrics": "NOT_COMPUTED_OR_CLAIMED_BY_POSTHOC_WRAPPER",
            "formal_metrics_owner": "unchanged frozen eval_epipolar.py at 4e6b7eb",
            "point_to_plane": "NOT_AVAILABLE",
        },
        "frozen_evaluator": source,
        "checkpoint": {
            "stage_c_path": str(args.checkpoint.expanduser().resolve()),
            "stage_c_sha256": sha256_file(args.checkpoint.expanduser().resolve()),
            "stage_b_path": str(args.base_checkpoint.expanduser().resolve()),
            "stage_b_sha256": sha256_file(args.base_checkpoint.expanduser().resolve()),
        },
        "frozen_evaluator_visualization_samples": args.samples,
        "ply_records": callback.records,
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "posthoc_pointcloud_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
