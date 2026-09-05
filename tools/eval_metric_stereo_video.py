#!/usr/bin/env python3
"""Evaluate a trained causal metric stereo-video checkpoint.

The evaluator loads one FSDP rank shard per process and runs the validation
clips through five shared-checkpoint component variants.  The resulting JSON
and CSV explicitly identify the surgical ablations; they are not independent
retraining results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from metrics.metric_stereo_video import MetricAccumulator, MetricValue, endpoint_metric_values, scalar_metric
from geometry.metric_reprojection import stereo_reproject_right_to_left, temporal_reproject_previous_to_current
from models.metric_stereo_video_system import MetricStereoVideoSystem
from tools.train_metric_stereo_video import (
    _dataset,
    _distributed_context,
    _loader,
    _move_batch,
    _project_path,
    _read_config,
    _sha256,
    _unwrapped,
    _wrap_distributed,
    build_model,
)


VARIANTS: dict[str, dict[str, bool]] = {
    "stereo_metric_prior_only": {
        "enable_vggt_features": False,
        "enable_vggt_gauge": False,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "stereo_plus_vggt_gauge": {
        "enable_vggt_features": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "stereo_plus_temporal_memory": {
        "enable_vggt_features": False,
        "enable_vggt_gauge": False,
        "enable_temporal_memory": True,
        "visibility_aware_gating": True,
    },
    "full_model": {
        "enable_vggt_features": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": True,
        "visibility_aware_gating": True,
    },
    "full_model_no_visibility_gating": {
        "enable_vggt_features": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": True,
        "visibility_aware_gating": False,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="step_XXXXXXX directory")
    parser.add_argument("--config", type=Path, help="resolved training YAML; defaults to checkpoint run config")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/metric_stereo_video/evaluation_final"))
    parser.add_argument("--max-batches", type=int, help="limit validation batches per rank for a smoke evaluation")
    parser.add_argument("--num-workers", type=int, help="override validation DataLoader workers")
    parser.add_argument(
        "--variants",
        help="comma-separated variant names (default: all five)",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge selected variants into an existing metrics.json report",
    )
    return parser.parse_args()


def _checkpoint_paths(checkpoint: Path, rank: int, world_size: int) -> tuple[dict[str, Any], Path]:
    checkpoint = checkpoint.expanduser().resolve()
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
        raise RuntimeError("checkpoint manifest is missing or incomplete")
    if int(manifest.get("world_size", -1)) != world_size:
        raise RuntimeError(f"checkpoint world_size {manifest.get('world_size')} != evaluator {world_size}")
    rank_name = f"rank_{rank:04d}.pt"
    rank_files = manifest.get("rank_files")
    if not isinstance(rank_files, list) or rank_name not in rank_files:
        raise RuntimeError(f"checkpoint manifest does not list {rank_name}")
    rank_path = checkpoint / rank_name
    expected_sha = manifest.get("rank_file_sha256", {}).get(rank_name)
    if not rank_path.is_file() or (expected_sha and _sha256(rank_path) != expected_sha):
        raise RuntimeError(f"checkpoint shard integrity check failed for {rank_name}")
    return dict(manifest), rank_path


def _load_model(config: Mapping[str, Any], checkpoint: Path, context: Any) -> torch.nn.Module:
    _manifest, rank_path = _checkpoint_paths(checkpoint, context.rank, context.world_size)
    raw = build_model(config)
    wrapped = _wrap_distributed(raw, config, context, no_fsdp=False)
    payload = torch.load(rank_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError(f"malformed model payload: {rank_path}")
    if int(payload.get("world_size", -1)) != context.world_size:
        raise RuntimeError("rank payload world size mismatch")
    if context.world_size > 1:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardedStateDictConfig,
            StateDictType,
        )

        with FSDP.state_dict_type(
            wrapped,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            wrapped.load_state_dict(payload["model"], strict=True)
    else:
        wrapped.load_state_dict(payload["model"], strict=True)
    wrapped.eval()
    # Keep the FSDP wrapper alive for forward gathers.  Component flags are
    # mutated through the unwrapped root below, but inference must call the
    # wrapper itself.
    return wrapped


def _set_variant(model: torch.nn.Module, settings: Mapping[str, bool]) -> None:
    root = _unwrapped(model)
    root.enable_vggt_features = bool(settings["enable_vggt_features"])
    geometry = root.geometry_model
    geometry.enable_vggt_gauge = bool(settings["enable_vggt_gauge"])
    geometry.enable_temporal_memory = bool(settings["enable_temporal_memory"])
    geometry.visibility_aware_gating = bool(settings["visibility_aware_gating"])


def _endpoint_metrics(output: Any, batch: Mapping[str, Any]) -> dict[str, MetricValue]:
    gt_disp = batch["disparity_gt_left_px"][:, -1]
    gt_valid = batch["valid_gt_left"][:, -1].bool()
    endpoint = output.endpoint
    values = endpoint_metric_values(
        predicted_depth_m=endpoint.depth_m,
        predicted_disparity_px=endpoint.disparity_left_px,
        predicted_valid_mask=endpoint.valid_mask,
        predicted_valid_probability=endpoint.valid_probability,
        predicted_uncertainty=endpoint.uncertainty,
        gt_disparity_px=gt_disp,
        gt_valid_mask=gt_valid,
        intrinsics_left=batch["K"][:, -1, 0].float(),
        baseline_m=batch["baseline_m"][:, -1].float(),
        dynamic_mask=batch["dynamic_mask_current"].bool(),
        dynamic_available=batch["dynamic_mask_available"].bool(),
    )
    left_rgb = batch["rgb"][:, -1, 0]
    right_rgb = batch["rgb"][:, -1, 1]
    stereo_reprojection = stereo_reproject_right_to_left(left_rgb, right_rgb, endpoint.disparity_left_px)
    stereo_photo = (stereo_reprojection.image - left_rgb).abs().mean(dim=1, keepdim=True)
    values["stereo_reprojection_residual"] = scalar_metric(stereo_photo, stereo_reprojection.valid_mask)

    lr_error = output.stereo.left_right_error_hr_px_lr_grid[:, -1]
    lr_valid = output.stereo.valid_left_mask_lr[:, -1]
    values["left_right_consistency_residual_px"] = scalar_metric(lr_error, lr_valid)
    values["left_right_consistency_valid_fraction"] = MetricValue(float(lr_valid.sum().item()), int(lr_valid.numel()))

    temporal = endpoint.temporal
    visible = temporal.zbuffer_visible_mask
    all_pixels = torch.ones_like(visible)
    values["temporal_visibility_fraction"] = scalar_metric(visible.float(), all_pixels)
    values["temporal_valid_fraction"] = scalar_metric(temporal.valid_mask.float(), all_pixels)
    values["temporal_depth_consistent_fraction"] = scalar_metric(temporal.depth_consistent_mask.float(), all_pixels)
    values["temporal_collision_fraction"] = scalar_metric(temporal.collision_mask.float(), all_pixels)
    values["temporal_collision_conditioned_on_visible"] = scalar_metric(
        temporal.collision_mask.float(), visible
    )
    if temporal.valid_mask.any():
        target_shape = temporal.valid_mask.shape[-2:]
        current_inverse = F.interpolate(endpoint.inverse_depth_m_inv.float(), size=target_shape, mode="bilinear", align_corners=False)
        pre_warp = temporal.warped_inverse_depth_pre_consistency_m_inv.float()
        post_warp = temporal.warped_inverse_depth_m_inv.float()
        pre_valid = temporal.zbuffer_visible_mask & _finite_positive(pre_warp)
        post_valid = temporal.valid_mask & _finite_positive(post_warp)
        pre_error = (torch.log(current_inverse.clamp_min(1e-8) / pre_warp.clamp_min(1e-8))).abs()
        post_error = (torch.log(current_inverse.clamp_min(1e-8) / post_warp.clamp_min(1e-8))).abs()
        values["temporal_flicker_log_abs"] = scalar_metric(pre_error, pre_valid)
        values["temporal_warp_residual_log_abs"] = scalar_metric(post_error, post_valid)
    else:
        values["temporal_flicker_log_abs"] = MetricValue(0.0, 0)
        values["temporal_warp_residual_log_abs"] = MetricValue(0.0, 0)

    if batch["rgb"].shape[1] >= 2:
        temporal_photo = temporal_reproject_previous_to_current(
            left_rgb,
            batch["rgb"][:, -2, 0],
            endpoint.inverse_depth_m_inv,
            batch["K"][:, -1, 0].float(),
            batch["K"][:, -2, 0].float(),
            batch["T_current_from_previous"][:, -1].float(),
        )
        photo_error = (temporal_photo.image - left_rgb).abs().mean(dim=1, keepdim=True)
        values["temporal_reprojection_residual"] = scalar_metric(photo_error, temporal_photo.valid_mask)
    return values


def _sync(accumulator: MetricAccumulator, context: Any) -> None:
    if context.world_size <= 1:
        return
    names = sorted(accumulator.values)
    for name in names:
        item = accumulator.values[name]
        packed = torch.tensor([float(item[0]), float(item[1])], dtype=torch.float64, device=context.device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        item[0], item[1] = float(packed[0].item()), int(round(float(packed[1].item())))


def _write_reports(
    output_dir: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    results: Mapping[str, Any],
    context: Any,
    *,
    merge_existing: bool = False,
) -> None:
    if not context.primary:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if merge_existing and (output_dir / "metrics.json").is_file():
        loaded = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    merged_variants = dict(existing.get("variants", {}))
    merged_variants.update(results)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "shared_checkpoint_ablation": True,
        "ablation_note": "All variants use the same final trained checkpoint; component flags are changed only at evaluation time and are not independent retraining results.",
        "config": dict(config),
        "variants": merged_variants,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "metric", "value", "numerator", "count", "valid"])
        writer.writeheader()
        for variant, payload in merged_variants.items():
            for name, metric in payload["metrics"].items():
                writer.writerow({"variant": variant, "metric": name, **metric})
    (output_dir / "ablation_matrix.json").write_text(
        json.dumps({"schema_version": 1, "shared_checkpoint": True, "variants": merged_variants}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    context = _distributed_context()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else checkpoint.parents[1] / "resolved_config.yaml"
    config = _read_config(config_path)
    if args.num_workers is not None:
        config["data"]["num_workers"] = int(args.num_workers)
    dataset = _dataset(config, training=False)
    loader, sampler = _loader(dataset, config, context, training=False)
    if sampler is not None:
        sampler.set_epoch(0)
    model = _load_model(config, checkpoint, context)
    results: dict[str, Any] = {}
    selected_variants = list(VARIANTS)
    if args.variants:
        selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = sorted(set(selected_variants) - set(VARIANTS))
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        if not selected_variants:
            raise ValueError("--variants cannot be empty")
    for variant in selected_variants:
        settings = VARIANTS[variant]
        _set_variant(model, settings)
        accumulator = MetricAccumulator()
        batches_seen = 0
        samples_seen = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                batch = _move_batch(batch, context.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch)
                accumulator.update(_endpoint_metrics(output, batch))
                batches_seen += 1
                samples_seen += int(batch["rgb"].shape[0])
        _sync(accumulator, context)
        if context.primary:
            results[variant] = {
                "flags": dict(settings),
                "batches": batches_seen * context.world_size,
                "samples": samples_seen * context.world_size,
                "metrics": accumulator.finalize(),
            }
            print(json.dumps({"variant": variant, "samples": results[variant]["samples"], "metrics": results[variant]["metrics"]}, sort_keys=True), flush=True)
        if context.world_size > 1:
            dist.barrier()
    _write_reports(
        args.output_dir.expanduser().resolve(),
        config,
        checkpoint,
        results,
        context,
        merge_existing=args.merge_existing,
    )
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _finite_positive(value: Tensor) -> Tensor:
    return torch.isfinite(value) & (value > 0)


if __name__ == "__main__":
    raise SystemExit(main())
