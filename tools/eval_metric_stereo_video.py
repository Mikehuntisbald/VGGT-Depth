#!/usr/bin/env python3
"""Evaluate a trained causal metric stereo-video checkpoint.

The default evaluates exactly the configuration represented by the checkpoint.
Explicit ``--variants`` are shared-checkpoint diagnostics only and are never
reported as independently trained ablations.
"""

from __future__ import annotations

import argparse
import copy
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
from metrics.metric_stereo_video import (
    AccuracyCoverageHistogram,
    temporal_residual_metric_values,
)
from metrics.boundary import disparity_boundary_mask
from metrics.spring_arms import SpringNativeMapError, spring_map_bundle
from geometry.metric_reprojection import stereo_reproject_right_to_left, temporal_reproject_previous_to_current
from geometry.zbuffer_reproject import zbuffer_reproject
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
    "diagnostic_A0_reconstruction_only": {
        "enable_vggt_dense_features": False,
        "enable_vggt_geometry": False,
        "enable_vggt_gauge": False,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "diagnostic_A1_vggt_gauge_only": {
        "enable_vggt_dense_features": False,
        "enable_vggt_geometry": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "diagnostic_A2_vggt_dense_feature_only": {
        "enable_vggt_dense_features": True,
        "enable_vggt_geometry": False,
        "enable_vggt_gauge": False,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "diagnostic_A3_temporal_memory_only": {
        "enable_vggt_dense_features": False,
        "enable_vggt_geometry": False,
        "enable_vggt_gauge": False,
        "enable_temporal_memory": True,
        "visibility_aware_gating": True,
    },
    "diagnostic_A4_vggt_gauge_and_features": {
        "enable_vggt_dense_features": True,
        "enable_vggt_geometry": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": False,
        "visibility_aware_gating": True,
    },
    "diagnostic_A5_full_model": {
        "enable_vggt_dense_features": True,
        "enable_vggt_geometry": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": True,
        "visibility_aware_gating": True,
    },
    "diagnostic_A6_full_no_visibility_gating": {
        "enable_vggt_dense_features": True,
        "enable_vggt_geometry": True,
        "enable_vggt_gauge": True,
        "enable_temporal_memory": True,
        "visibility_aware_gating": False,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="step_XXXXXXX directory")
    parser.add_argument("--config", type=Path, help="resolved training YAML; defaults to checkpoint run config")
    parser.add_argument(
        "--evaluation-contract",
        type=Path,
        default=Path("configs/metric_stereo_video/evaluation_contract.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/metric_stereo_video/evaluation_final"))
    parser.add_argument("--max-batches", type=int, help="limit validation batches per rank for a smoke evaluation")
    parser.add_argument("--num-workers", type=int, help="override validation DataLoader workers")
    parser.add_argument(
        "--variants",
        help=(
            "explicit comma-separated shared-checkpoint diagnostic variants; "
            "default evaluates the checkpoint's trained configuration only"
        ),
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge selected variants into an existing metrics.json report",
    )
    return parser.parse_args()


def _selected_variants(requested: str | None) -> list[str]:
    if requested is None:
        return ["trained_configuration"]
    selected = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if not selected:
        raise ValueError("--variants cannot be empty")
    return selected


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
    root.enable_vggt_dense_features = bool(settings["enable_vggt_dense_features"])
    root.enable_vggt_geometry = bool(settings["enable_vggt_geometry"])
    geometry = root.geometry_model
    geometry.enable_vggt_gauge = bool(settings["enable_vggt_gauge"])
    geometry.enable_temporal_memory = bool(settings["enable_temporal_memory"])
    geometry.visibility_aware_gating = bool(settings["visibility_aware_gating"])


def _trained_settings(model: torch.nn.Module) -> dict[str, bool]:
    root = _unwrapped(model)
    geometry = root.geometry_model
    return {
        "enable_vggt_dense_features": bool(root.enable_vggt_dense_features),
        "enable_vggt_geometry": bool(root.enable_vggt_geometry),
        "enable_vggt_gauge": bool(geometry.enable_vggt_gauge),
        "enable_temporal_memory": bool(geometry.enable_temporal_memory),
        "visibility_aware_gating": bool(geometry.visibility_aware_gating),
    }


def _previous_prefix_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the current frame so a second forward predicts strictly t-1."""

    frames = int(batch["rgb"].shape[1])
    if frames < 2:
        raise ValueError("temporal residual evaluation requires at least two frames")
    temporal_fields = {
        "rgb",
        "K",
        "T_right_from_left",
        "T_current_from_previous",
        "temporal_transform_valid",
        "T_current_from_previous_valid",
        "T_left_camera_from_world",
        "camera_pose_valid",
        "baseline_m",
        "disparity_gt_left_px",
        "disparity_gt_right_px",
        "valid_gt_left",
        "valid_gt_right",
        "target_time_mask",
        "time_valid_mask",
        "frame_ids",
        "timestamps",
        "manifest_indices",
    }
    prefix = dict(batch)
    for name in temporal_fields:
        value = prefix.get(name)
        if isinstance(value, Tensor):
            prefix[name] = value[:, :-1]
    prefix["target_time_mask"] = torch.zeros_like(prefix["target_time_mask"])
    prefix["target_time_mask"][:, -1] = True
    prefix["clip_lengths"] = batch["clip_lengths"] - 1
    return prefix


def _spring_partition_masks(
    cpu_batch: Mapping[str, Any], dataset: Any
) -> dict[str, Tensor]:
    """Load official Spring masks on the exact fixed-crop output grid."""

    height, width = cpu_batch["rgb"].shape[-2:]
    details: list[Tensor] = []
    matched: list[Tensor] = []
    for metadata in cpu_batch["identity_metadata"]:
        endpoint_index = int(metadata["endpoint_manifest_index"])
        record = dataset.records[endpoint_index].to_dict()
        crop = tuple(int(value) for value in metadata["crop_xywh"])
        try:
            bundle = spring_map_bundle(
                record,
                target_hw=(height, width),
                manifest_path=dataset.manifest_path,
                crop_hr_xywh=crop,
                require_rigid=False,
            )
        except SpringNativeMapError as exc:
            raise RuntimeError(
                f"formal Spring partition maps unavailable for endpoint {endpoint_index}: {exc}"
            ) from exc
        details.append(torch.from_numpy(bundle["detail"]).unsqueeze(0))
        matched.append(torch.from_numpy(bundle["matched"]).unsqueeze(0))
    return {
        "detail": torch.stack(details).bool(),
        "matched": torch.stack(matched).bool(),
    }


def _warp_previous_prediction_to_current(
    previous_output: Any, batch: Mapping[str, Any]
) -> tuple[Any, Tensor]:
    batch_size = int(batch["rgb"].shape[0])
    identity = torch.eye(
        4, device=batch["rgb"].device, dtype=torch.float32
    ).expand(batch_size, -1, -1)
    warp = zbuffer_reproject(
        previous_output.disparity_left_px.float(),
        previous_output.depth_m.float(),
        previous_output.confidence.float(),
        batch["K"][:, -2, 0].float(),
        identity,
        batch["T_current_from_previous"][:, -1].float(),
        intrinsics_current_hr_3x3=batch["K"][:, -1, 0].float(),
        baseline_previous_m=batch["baseline_m"][:, -2].float(),
        baseline_current_m=batch["baseline_m"][:, -1].float(),
    )
    height, width = previous_output.valid_mask.shape[-2:]
    source_u = warp.source_uv[:, 0].long().clamp(0, width - 1)
    source_v = warp.source_uv[:, 1].long().clamp(0, height - 1)
    source_linear = (source_v * width + source_u).reshape(batch_size, 1, -1)
    winner_valid = torch.gather(
        previous_output.valid_mask.flatten(2), 2, source_linear
    ).reshape_as(previous_output.valid_mask)
    return warp, warp.valid_mask & winner_valid


def _warp_previous_gt_to_current(batch: Mapping[str, Any]) -> Any:
    previous_disparity = batch["previous_disparity_gt_left_px"].float()
    previous_valid = batch["previous_valid_gt_left"].bool()
    previous_disparity = torch.where(
        previous_valid, previous_disparity, torch.zeros_like(previous_disparity)
    )
    factor = (
        batch["K"][:, -2, 0, 0, 0].float()
        * batch["baseline_m"][:, -2].float()
    ).reshape(-1, 1, 1, 1)
    previous_depth = torch.where(
        previous_valid,
        factor / previous_disparity.clamp_min(1e-8),
        torch.zeros_like(previous_disparity),
    )
    identity = torch.eye(
        4, device=previous_disparity.device, dtype=torch.float32
    ).expand(previous_disparity.shape[0], -1, -1)
    return zbuffer_reproject(
        previous_disparity,
        previous_depth,
        previous_valid.float(),
        batch["K"][:, -2, 0].float(),
        identity,
        batch["T_current_from_previous"][:, -1].float(),
        intrinsics_current_hr_3x3=batch["K"][:, -1, 0].float(),
        baseline_previous_m=batch["baseline_m"][:, -2].float(),
        baseline_current_m=batch["baseline_m"][:, -1].float(),
    )


def _motion_bucket_masks(
    batch: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Tensor]:
    transform = batch["T_current_from_previous"][:, -1].float()
    translation = torch.linalg.vector_norm(transform[:, :3, 3], dim=-1)
    cosine = ((transform[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    score = translation + torch.acos(cosine)
    low = float(contract["temporal"]["small_medium_threshold"])
    high = float(contract["temporal"]["medium_large_threshold"])
    shape = (score.shape[0], 1, batch["rgb"].shape[-2], batch["rgb"].shape[-1])
    return {
        "small_motion": (score <= low).reshape(-1, 1, 1, 1).expand(shape),
        "medium_motion": ((score > low) & (score <= high)).reshape(-1, 1, 1, 1).expand(shape),
        "large_motion": (score > high).reshape(-1, 1, 1, 1).expand(shape),
    }


def _endpoint_metrics(
    output: Any,
    previous_output: Any,
    batch: Mapping[str, Any],
    *,
    spring_masks: Mapping[str, Tensor],
    contract: Mapping[str, Any],
) -> dict[str, MetricValue]:
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
        detail_mask=spring_masks["detail"],
        matched_mask=spring_masks["matched"],
        boundary_mask=disparity_boundary_mask(
            gt_disp.float(),
            gradient_threshold_px=float(
                contract["spring_partitions"]["boundary_gradient_threshold_px"]
            ),
            radius_px=int(contract["spring_partitions"]["boundary_radius_px"]),
        ),
        invalid_penalty_px=float(
            contract["validity"]["all_gt_error_cap_and_invalid_penalty_px"]
        ),
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
        prediction_warp, prediction_winner_valid = _warp_previous_prediction_to_current(
            previous_output, batch
        )
        gt_warp = _warp_previous_gt_to_current(batch)
        values.update(
            temporal_residual_metric_values(
                current_prediction_disparity_px=endpoint.disparity_left_px,
                warped_previous_prediction_disparity_px=prediction_warp.disparity_hr_px,
                current_gt_disparity_px=gt_disp,
                warped_previous_gt_disparity_px=gt_warp.disparity_hr_px,
                current_prediction_valid=endpoint.valid_mask,
                warped_prediction_valid=prediction_winner_valid,
                current_gt_valid=gt_valid,
                warped_gt_valid=gt_warp.valid_mask,
                dynamic_mask=batch["dynamic_mask_current"].bool(),
                dynamic_available=batch["dynamic_mask_available"].bool(),
                motion_bucket_masks=_motion_bucket_masks(batch, contract),
                invalid_penalty_px=float(
                    contract["validity"]["all_gt_error_cap_and_invalid_penalty_px"]
                ),
            )
        )
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
    evaluation_contract: Mapping[str, Any],
    shared_checkpoint_ablation: bool,
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
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "evaluation_contract": dict(evaluation_contract),
        "model_pose_contract": {
            "class": "known_pose_stereo_video",
            "active_pose_source": "spring_gt_camera_from_world_relative_transform",
            "pose_is_model_input": True,
            "pose_is_not_estimated_by_vggt": True,
        },
        "shared_checkpoint_ablation": shared_checkpoint_ablation,
        "ablation_note": (
            "Explicit diagnostic variants use one trained checkpoint with runtime component toggles; they are not independent retraining results."
            if shared_checkpoint_ablation
            else "The checkpoint is evaluated with the component configuration used to construct its trained model."
        ),
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
        json.dumps({"schema_version": 2, "shared_checkpoint": shared_checkpoint_ablation, "evaluation_contract": dict(evaluation_contract), "variants": merged_variants}, indent=2, sort_keys=True) + "\n",
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
    evaluation_contract = _read_config(args.evaluation_contract)
    if args.num_workers is not None:
        config["data"]["num_workers"] = int(args.num_workers)
    dataset = _dataset(config, training=False)
    loader, sampler = _loader(dataset, config, context, training=False)
    if sampler is not None:
        sampler.set_epoch(0)
    model = _load_model(config, checkpoint, context)
    results: dict[str, Any] = {}
    selected_variants = _selected_variants(args.variants)
    for variant in selected_variants:
        if variant == "trained_configuration":
            settings = _trained_settings(model)
        else:
            settings = VARIANTS[variant]
            _set_variant(model, settings)
        accumulator = MetricAccumulator()
        coverage = AccuracyCoverageHistogram(
            bins=int(
                evaluation_contract["validity"][
                    "accuracy_coverage_histogram_bins"
                ]
            ),
            device=context.device,
        )
        batches_seen = 0
        samples_seen = 0
        with torch.inference_mode():
            for batch_index, cpu_batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                spring_masks = {
                    name: value.to(context.device, non_blocking=True)
                    for name, value in _spring_partition_masks(
                        cpu_batch, dataset
                    ).items()
                }
                batch = _move_batch(cpu_batch, context.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch)
                # FSDP may recycle gathered parameter/storage-backed tensors on
                # the next forward. Snapshot the structured endpoint before
                # running the t-1 prefix, whose outputs are needed only for
                # temporal residuals.
                output_snapshot = copy.deepcopy(output)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    previous_output = model(_previous_prefix_batch(batch))
                # DistributedSampler pads two validation examples to preserve
                # FSDP collective parity. Only the rank owning the original
                # strided index contributes metrics, preventing duplicate GT.
                if int(batch["rgb"].shape[0]) != 1:
                    raise RuntimeError("formal evaluator requires micro batch size one")
                dataset_index = int(cpu_batch["identity_metadata"][0]["dataset_index"])
                unique_owner = dataset_index % context.world_size == context.rank
                if unique_owner:
                    accumulator.update(
                        _endpoint_metrics(
                            output_snapshot,
                            previous_output,
                            batch,
                            spring_masks=spring_masks,
                            contract=evaluation_contract,
                        )
                    )
                    gt_disparity = batch["disparity_gt_left_px"][:, -1].float()
                    gt_valid = batch["valid_gt_left"][:, -1].bool() & torch.isfinite(
                        gt_disparity
                    ) & (gt_disparity > 0)
                    coverage.update(
                        output_snapshot.valid_probability.float(),
                        (
                            output_snapshot.disparity_left_px.float()
                            - gt_disparity
                        ).abs(),
                        gt_valid,
                        error_cap_px=float(
                            evaluation_contract["validity"][
                                "all_gt_error_cap_and_invalid_penalty_px"
                            ]
                        ),
                    )
                batches_seen += 1
                samples_seen += int(unique_owner)
        _sync(accumulator, context)
        coverage.all_reduce_()
        global_samples_tensor = torch.tensor(
            samples_seen, dtype=torch.int64, device=context.device
        )
        if context.world_size > 1:
            dist.all_reduce(global_samples_tensor, op=dist.ReduceOp.SUM)
        global_samples = int(global_samples_tensor.item())
        if context.primary:
            finalized = accumulator.finalize()
            coverage_curve = coverage.finalize(
                evaluation_contract["validity"]["accuracy_coverage_points"]
            )
            point_99 = next(
                point
                for point in coverage_curve["points"]
                if abs(float(point["requested_coverage"]) - 0.99) < 1e-8
            )
            finalized["epe_at_99pct_coverage_px"] = {
                "value": point_99["epe_px"],
                "numerator": None,
                "count": point_99["effective_count"],
                "valid": point_99["epe_px"] is not None,
            }
            results[variant] = {
                "flags": dict(settings),
                "batches": batches_seen * context.world_size,
                "samples": global_samples,
                "metrics": finalized,
                "accuracy_coverage_curve": coverage_curve,
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
        evaluation_contract=evaluation_contract,
        shared_checkpoint_ablation=args.variants is not None,
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
