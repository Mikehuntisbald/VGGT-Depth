#!/usr/bin/env python3
"""Evaluate bilinear FFS and the Stage-A T=1 spatial model.

The target is explicitly the trusted HR FFS teacher pseudo-GT.  This script
does not report T=3, temporal, VGGT, epipolar, or paper-accuracy claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.collate import collate_training_samples  # noqa: E402
from data.training_dataset import CachedFFSTrainingDataset  # noqa: E402
from evaluation import (  # noqa: E402
    MethodMetricAccumulator,
    POINT_TO_PLANE_NOT_AVAILABLE,
    PSEUDO_GT_LABEL,
    comparison_from_aggregates,
    compute_sample_metrics,
    load_model_for_evaluation,
    upsample_ffs_inputs_to_hr,
)
from models.ffs_omega_tsr import count_trainable_parameters  # noqa: E402
from train import DEFAULT_CONFIG, build_model, load_receipt_identity  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from utils.visualization import (  # noqa: E402
    grayscale_to_rgb_uint8,
    save_rgb_uint8,
    scalar_to_rgb_uint8,
)


EVALUATION_DEFAULTS: dict[str, Any] = {
    "eval": {
        "output_dir": None,
        "crop_mode": "fixed",
        "fixed_crop_origin_hr_xy": None,
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "precision": "bf16",
        "limit": None,
        "visualization_samples": 4,
        "low_confidence_threshold": 0.8,
        "boundary_gradient_threshold_px": 1.0,
        "boundary_radius_px": 1,
    }
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bilinear LR-FFS and Stage-A T=1 against trusted HR FFS "
            "teacher pseudo-GT."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=Path, help="Stage-A training checkpoint")
    parser.add_argument("--manifest", type=Path, help="explicit validation JSONL manifest")
    parser.add_argument(
        "--observation-cache-root",
        type=Path,
        help="validation FFS observation cache root",
    )
    parser.add_argument(
        "--teacher-cache-root",
        type=Path,
        help="validation HR FFS teacher cache root",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="directory for metrics.json, metrics.csv, and visualizations",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, help="evaluation batch size")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count")
    parser.add_argument("--limit", type=int, help="evaluate the first N records")
    parser.add_argument(
        "--visualization-samples",
        type=int,
        help="number of leading samples to visualize",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("fixed", "full"),
        help="fixed center/origin crop, or full-resolution evaluation",
    )
    parser.add_argument(
        "--crop-origin",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="scale-aligned HR origin for fixed crops",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config and construct the model without loading data/checkpoint",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. eval.limit=8 data.hr_crop=[384,768]",
    )
    return parser


def _load_yaml_with_inheritance(path: Path, seen: set[Path] | None = None) -> DictConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"config does not exist: {resolved}")
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic defaults_from chain at {resolved}")
    seen.add(resolved)
    loaded = OmegaConf.load(resolved)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"config must resolve to a mapping: {resolved}")
    inherited_name = loaded.get("defaults_from")
    if inherited_name is None:
        return loaded
    del loaded["defaults_from"]
    inherited_path = Path(str(inherited_name)).expanduser()
    if not inherited_path.is_absolute():
        project_candidate = PROJECT_ROOT / inherited_path
        inherited_path = (
            project_candidate
            if project_candidate.exists()
            else resolved.parent / inherited_path
        )
    return OmegaConf.merge(
        _load_yaml_with_inheritance(inherited_path, seen), loaded
    )


def resolve_evaluation_config(
    config_path: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> DictConfig:
    """Resolve training/model defaults plus struct-checked evaluation options."""

    config = OmegaConf.merge(
        OmegaConf.create(DEFAULT_CONFIG),
        OmegaConf.create(EVALUATION_DEFAULTS),
        _load_yaml_with_inheritance(Path(config_path)),
    )
    OmegaConf.set_struct(config, True)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def _update_cli_values(config: DictConfig, args: argparse.Namespace) -> None:
    values = {
        "data.manifest_path": args.manifest,
        "data.observation_cache_root": args.observation_cache_root,
        "data.teacher_cache_root": args.teacher_cache_root,
        "eval.output_dir": args.output_dir,
        "eval.batch_size": args.batch_size,
        "eval.num_workers": args.num_workers,
        "eval.limit": args.limit,
        "eval.visualization_samples": args.visualization_samples,
        "eval.crop_mode": args.crop_mode,
        "eval.fixed_crop_origin_hr_xy": args.crop_origin,
    }
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value.expanduser().resolve())
        elif isinstance(value, tuple):
            value = list(value)
        OmegaConf.update(config, key, value, merge=False)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_evaluation_config(config: DictConfig) -> None:
    """Validate Stage-A/x2 and deterministic evaluation settings."""

    if int(config.data.sequence_length) != 1:
        raise ValueError("Stage-A evaluation is T=1; data.sequence_length must be 1")
    if int(config.data.scale) != 2 or int(config.model.convex_scale) != 2:
        raise ValueError("Stage-A evaluation is fixed to x2")
    if list(config.model.rgb_channels) != [32, 64, 96]:
        raise ValueError("model.rgb_channels must be [32,64,96]")
    if str(config.eval.crop_mode) not in {"fixed", "full"}:
        raise ValueError("eval.crop_mode must be fixed or full")
    crop = list(config.data.hr_crop)
    if len(crop) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in crop
    ):
        raise ValueError("data.hr_crop must be [height,width] positive integers")
    if any(value % 2 for value in crop):
        raise ValueError("data.hr_crop dimensions must be divisible by x2")
    origin = config.eval.fixed_crop_origin_hr_xy
    if origin is not None:
        values = list(origin)
        if len(values) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("eval.fixed_crop_origin_hr_xy must be [x,y] non-negative")
        if any(value % 2 for value in values):
            raise ValueError("fixed evaluation crop origin must be x2-aligned")
        if str(config.eval.crop_mode) != "fixed":
            raise ValueError("a fixed crop origin requires eval.crop_mode=fixed")
    _positive_int(config.eval.batch_size, "eval.batch_size")
    _nonnegative_int(config.eval.num_workers, "eval.num_workers")
    _nonnegative_int(config.eval.visualization_samples, "eval.visualization_samples")
    if config.eval.limit is not None:
        _positive_int(config.eval.limit, "eval.limit")
    threshold = float(config.eval.low_confidence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("eval.low_confidence_threshold must be in [0,1]")
    if float(config.eval.boundary_gradient_threshold_px) < 0.0:
        raise ValueError("boundary gradient threshold must be non-negative")
    _nonnegative_int(config.eval.boundary_radius_px, "eval.boundary_radius_px")
    if str(config.eval.precision).lower() not in {"bf16", "fp32"}:
        raise ValueError("eval.precision must be bf16 or fp32")


def _required_path(config: DictConfig, key: str, *, directory: bool) -> Path:
    value = OmegaConf.select(config, key)
    if value is None or not str(value).strip():
        raise ValueError(
            f"{key} is required and must identify the validation artifact explicitly"
        )
    path = Path(str(value)).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{key} {kind} does not exist: {path}")
    return path


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return device


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _rgb_chw_to_uint8(rgb: Tensor) -> np.ndarray:
    array = (
        rgb.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return np.rint(array * 255.0).astype(np.uint8)


def _save_visualization(
    root: Path,
    *,
    sample_name: str,
    rgb_hr: Tensor,
    baseline_hr_px: Tensor,
    output_hr_px: Tensor,
    target_hr_px: Tensor,
    target_trusted_mask: Tensor,
    source_weights_lr: Tensor,
    uncertainty_hr: Tensor,
) -> None:
    sample_root = root / sample_name
    target_mask = target_trusted_mask.to(dtype=torch.bool)
    absolute_error = (output_hr_px - target_hr_px).abs()
    save_rgb_uint8(sample_root / "rgb.png", _rgb_chw_to_uint8(rgb_hr))
    for filename, value, mask in (
        ("bilinear_ffs_hr_px.png", baseline_hr_px, None),
        ("t1_disparity_hr_px.png", output_hr_px, None),
        ("teacher_pseudo_gt_hr_px.png", target_hr_px, target_mask),
        ("absolute_error_hr_px.png", absolute_error, target_mask),
        ("uncertainty_variance.png", uncertainty_hr, None),
    ):
        save_rgb_uint8(
            sample_root / filename,
            scalar_to_rgb_uint8(value, valid_mask=mask),
        )
    source_names = ("ffs", "vggt", "history")
    for source_index, source_name in enumerate(source_names):
        source_hr = functional.interpolate(
            source_weights_lr[source_index : source_index + 1].unsqueeze(0),
            size=output_hr_px.shape[-2:],
            mode="nearest",
        )[0]
        save_rgb_uint8(
            sample_root / f"source_weight_{source_name}.png",
            grayscale_to_rgb_uint8(source_hr, minimum=0.0, maximum=1.0),
        )


def _write_csv(
    path: Path,
    methods: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
) -> None:
    metric_names = sorted(
        {
            name
            for method in methods.values()
            for name, value in method.items()
            if isinstance(value, dict) and "value" in value
        }
    )
    fields = ["method", "target_type", "point_to_plane"]
    for name in metric_names:
        fields.extend((name, f"{name}_valid", f"{name}_count", f"{name}_numerator"))
    fields.extend(
        (
            "trusted_region_degradation_percent",
            "invalid_region_completeness_change_percent",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name in ("bilinear", "T1"):
            method = methods[method_name]
            row: dict[str, Any] = {
                "method": method_name,
                "target_type": PSEUDO_GT_LABEL,
                "point_to_plane": "NOT_AVAILABLE",
            }
            for name in metric_names:
                metric = method.get(name)
                if not isinstance(metric, dict):
                    continue
                row[name] = metric["value"]
                row[f"{name}_valid"] = metric["valid"]
                row[f"{name}_count"] = metric["count"]
                row[f"{name}_numerator"] = metric["numerator"]
            if method_name == "bilinear":
                trusted = methods["bilinear"]["trusted_region_epe_px"]
                row["trusted_region_degradation_percent"] = (
                    0.0 if trusted["valid"] else None
                )
                row["invalid_region_completeness_change_percent"] = (
                    0.0
                    if methods["bilinear"]["invalid_region_completeness"]["valid"]
                    else None
                )
            else:
                row["trusted_region_degradation_percent"] = comparisons[
                    "trusted_region_degradation"
                ]["relative_change_percent"]
                row["invalid_region_completeness_change_percent"] = comparisons[
                    "invalid_region_completeness_change"
                ]["relative_change_percent"]
            writer.writerow(row)


def _resolved_dict(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config is not a mapping")
    return value


def run(args: argparse.Namespace) -> int:
    config = resolve_evaluation_config(args.config, args.overrides)
    _update_cli_values(config, args)
    validate_evaluation_config(config)
    seed_everything(int(config.seed), deterministic=True)
    model = build_model(config)
    parameter_count = count_trainable_parameters(model)
    if parameter_count <= 0 or parameter_count >= 12_000_000:
        raise ValueError(f"trainable parameter count must be in (0,12M), got {parameter_count}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "stage": "T1_SPATIAL_ONLY",
                    "target_type": PSEUDO_GT_LABEL,
                    "parameter_count": parameter_count,
                    "crop_mode": str(config.eval.crop_mode),
                    "resolved_config": _resolved_dict(config),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry-run is used")
    manifest_path = _required_path(config, "data.manifest_path", directory=False)
    observation_root = _required_path(
        config, "data.observation_cache_root", directory=True
    )
    teacher_root = _required_path(config, "data.teacher_cache_root", directory=True)
    output_value = config.eval.output_dir
    output_dir = (
        PROJECT_ROOT / "outputs" / str(config.experiment) / "evaluation"
        if output_value is None or not str(output_value).strip()
        else Path(str(output_value)).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_identity = load_receipt_identity(
        observation_root,
        expected_component="ffs-observation",
        manifest_path=manifest_path,
    )
    teacher_identity = load_receipt_identity(
        teacher_root,
        expected_component="ffs-teacher",
        manifest_path=manifest_path,
    )
    full_resolution = str(config.eval.crop_mode) == "full"
    crop_size = None if full_resolution else tuple(int(v) for v in config.data.hr_crop)
    origin_value = config.eval.fixed_crop_origin_hr_xy
    fixed_origin = (
        None if origin_value is None else tuple(int(value) for value in origin_value)
    )
    dataset = CachedFFSTrainingDataset(
        manifest_path=manifest_path,
        observation_cache_root=observation_root,
        teacher_cache_root=teacher_root,
        observation_identity=observation_identity,
        teacher_identity=teacher_identity,
        crop_size_hr_hw=crop_size,
        crop_mode="fixed",
        fixed_crop_origin_hr_xy=fixed_origin,
        spatial_scale=int(config.data.scale),
        seed=int(config.seed),
    )
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    requested_limit = config.eval.limit
    sample_count = (
        len(dataset)
        if requested_limit is None
        else min(int(requested_limit), len(dataset))
    )
    if sample_count <= 0:
        raise ValueError("evaluation selects no validation records")
    selected_dataset = Subset(dataset, range(sample_count))
    device = _resolve_device(args.device)
    pin_memory = bool(config.eval.pin_memory) and device.type == "cuda"
    loader = DataLoader(
        selected_dataset,
        batch_size=int(config.eval.batch_size),
        shuffle=False,
        num_workers=int(config.eval.num_workers),
        persistent_workers=int(config.eval.num_workers) > 0,
        pin_memory=pin_memory,
        collate_fn=collate_training_samples,
    )

    checkpoint_metadata = load_model_for_evaluation(
        args.checkpoint,
        model,
        expected_parameter_count=parameter_count,
    )
    model.to(device=device).eval()
    accumulators = {
        "bilinear": MethodMetricAccumulator(),
        "T1": MethodMetricAccumulator(),
    }
    visualization_limit = min(int(config.eval.visualization_samples), sample_count)
    visualized = 0
    started = time.perf_counter()

    use_cuda_bf16 = (
        device.type == "cuda" and str(config.eval.precision).lower() == "bf16"
    )
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            target = batch["teacher_disparity_hr_px"]
            target_trusted = batch["teacher_trusted_mask"]
            if not isinstance(target, Tensor) or not isinstance(target_trusted, Tensor):
                raise ValueError("evaluation requires teacher disparity and trusted mask")
            output_size = tuple(int(value) for value in target.shape[-2:])
            baseline, confidence_hr, valid_hr, trusted_hr = upsample_ffs_inputs_to_hr(
                batch["observation_disparity_hr_px"],
                batch["observation_confidence"],
                batch["observation_valid_mask"],
                batch["observation_trusted_mask"],
                output_size_hw=output_size,
            )
            # Construct a fresh context manager per batch; this remains safe
            # for context-manager implementations that are not re-entrant.
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda_bf16
                else nullcontext()
            )
            with autocast_context:
                model_output = model(
                    batch["rgb_hr"],
                    batch["observation_disparity_hr_px"],
                    batch["observation_confidence"],
                    valid_ffs=batch["observation_valid_mask"],
                )
            predictions = {
                "bilinear": baseline.float(),
                "T1": model_output.disparity_hr_px.float(),
            }
            for method_name, prediction in predictions.items():
                sample_metrics = compute_sample_metrics(
                    prediction,
                    target.float(),
                    target_trusted_mask=target_trusted,
                    ffs_confidence_hr=confidence_hr.float(),
                    ffs_valid_mask_hr=valid_hr,
                    ffs_trusted_mask_hr=trusted_hr,
                    low_confidence_threshold=float(config.eval.low_confidence_threshold),
                    boundary_gradient_threshold_px=float(
                        config.eval.boundary_gradient_threshold_px
                    ),
                    boundary_radius_px=int(config.eval.boundary_radius_px),
                )
                accumulators[method_name].update(sample_metrics)

            batch_size = target.shape[0]
            for item_index in range(batch_size):
                if visualized >= visualization_limit:
                    break
                sequence_id = str(batch["sequence_id"][item_index]).replace("/", "_")
                frame_id = int(batch["frame_id"][item_index].item())
                _save_visualization(
                    output_dir / "visualizations",
                    sample_name=f"{visualized:04d}_{sequence_id}_{frame_id}",
                    rgb_hr=batch["rgb_hr"][item_index],
                    baseline_hr_px=baseline[item_index],
                    output_hr_px=model_output.disparity_hr_px[item_index].float(),
                    target_hr_px=target[item_index].float(),
                    target_trusted_mask=target_trusted[item_index],
                    source_weights_lr=model_output.source_weights[item_index].float(),
                    uncertainty_hr=model_output.uncertainty[item_index].float(),
                )
                visualized += 1

    elapsed_seconds = time.perf_counter() - started
    finalized = {
        method: {name: result.to_dict() for name, result in accumulator.finalize().items()}
        for method, accumulator in accumulators.items()
    }
    for method in finalized.values():
        method["point_to_plane_error_m"] = dict(POINT_TO_PLANE_NOT_AVAILABLE)
    comparisons = comparison_from_aggregates(
        accumulators["bilinear"].finalize(), accumulators["T1"].finalize()
    )
    report = {
        "schema_version": 1,
        "stage": "T1_SPATIAL_ONLY",
        "target": {
            "type": PSEUDO_GT_LABEL,
            "paper_accuracy": False,
            "warning": (
                "Metrics use trusted output from the same FFS family as pseudo-GT; "
                "they are engineering validation only."
            ),
        },
        "methods": finalized,
        "comparisons": comparisons,
        "point_to_plane": dict(POINT_TO_PLANE_NOT_AVAILABLE),
        "records_evaluated": sample_count,
        "visualizations_written": visualized,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "crop_mode": str(config.eval.crop_mode),
        "hr_crop": None if full_resolution else list(crop_size or ()),
        "parameter_count": parameter_count,
        "checkpoint": checkpoint_metadata,
        "manifest_path": str(manifest_path),
        "cache_identities": {
            "observation": asdict(observation_identity),
            "teacher": asdict(teacher_identity),
        },
        "resolved_config": _resolved_dict(config),
    }
    metrics_json = output_dir / "metrics.json"
    metrics_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", finalized, comparisons)
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "T1_SPATIAL_ONLY",
                "target_type": PSEUDO_GT_LABEL,
                "records_evaluated": sample_count,
                "parameter_count": parameter_count,
                "metrics_json": str(metrics_json),
                "metrics_csv": str(output_dir / "metrics.csv"),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
