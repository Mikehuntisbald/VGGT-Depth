#!/usr/bin/env python3
"""Diagnose whether independent A5 temporal memory uses useful history.

The analysis compares independently trained A4 (VGGT gauge/features, no
temporal memory) and A5 (full model) on the same validation endpoints. It
separates an oracle opportunity -- whether the warped A5 history is closer to
GT than A4 -- from the actual A5 gain. No weights or checkpoints are changed.
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

from geometry.zbuffer_reproject import zbuffer_reproject
from metrics.spring_arms import SpringNativeMapError, spring_map_bundle
from tools.eval_metric_stereo_video import _load_model
from tools.train_metric_stereo_video import (
    _dataset,
    _distributed_context,
    _loader,
    _masked_resize_scalar,
    _move_batch,
    _read_config,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a4-checkpoint", type=Path, required=True)
    parser.add_argument("--a5-checkpoint", type=Path, required=True)
    parser.add_argument("--a4-config", type=Path, required=True)
    parser.add_argument("--a5-config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/metric_stereo_video/temporal_analysis_a4_vs_a5"),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--opportunity-threshold-px", type=float, default=0.1,
        help="minimum EPE advantage used to call a history opportunity/gain",
    )
    return parser.parse_args()


def _finite_positive(value: Tensor) -> Tensor:
    return torch.isfinite(value) & (value > 0)


def _spring_masks(cpu_batch: Mapping[str, Any], dataset: Any) -> dict[str, Tensor]:
    height, width = cpu_batch["rgb"].shape[-2:]
    detail: list[Tensor] = []
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
                f"Spring partition map unavailable for endpoint {endpoint_index}: {exc}"
            ) from exc
        detail.append(torch.from_numpy(bundle["detail"]).unsqueeze(0))
    return {"high_detail": torch.stack(detail).bool()}


def _warp_gt(batch: Mapping[str, Any]) -> Any:
    disparity = batch["previous_disparity_gt_left_px"].float()
    valid = batch["previous_valid_gt_left"].bool() & _finite_positive(disparity)
    disparity = torch.where(valid, disparity, torch.zeros_like(disparity))
    factor = (
        batch["K"][:, -2, 0, 0, 0].float() * batch["baseline_m"][:, -2].float()
    ).reshape(-1, 1, 1, 1)
    depth = torch.where(
        valid,
        factor / disparity.clamp_min(1e-8),
        torch.zeros_like(disparity),
    )
    identity = torch.eye(4, device=disparity.device, dtype=torch.float32).expand(
        disparity.shape[0], -1, -1
    )
    return zbuffer_reproject(
        disparity,
        depth,
        valid.float(),
        batch["K"][:, -2, 0].float(),
        identity,
        batch["T_current_from_previous"][:, -1].float(),
        intrinsics_current_hr_3x3=batch["K"][:, -1, 0].float(),
        baseline_previous_m=batch["baseline_m"][:, -2].float(),
        baseline_current_m=batch["baseline_m"][:, -1].float(),
    )


def _add(stats: dict[str, list[float]], name: str, value: Tensor, mask: Tensor) -> None:
    selected = value.float()[mask.bool()]
    selected = selected[torch.isfinite(selected)]
    item = stats.setdefault(name, [0.0, 0.0])
    item[0] += float(selected.double().sum().item())
    item[1] += float(selected.numel())


def _add_rate(stats: dict[str, list[float]], name: str, event: Tensor, mask: Tensor) -> None:
    _add(stats, name, event.float(), mask)


def _sync(stats: dict[str, list[float]], context: Any) -> None:
    names = sorted(stats)
    if context.world_size <= 1:
        return
    for name in names:
        packed = torch.tensor(stats[name], dtype=torch.float64, device=context.device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        stats[name] = [float(packed[0].item()), float(packed[1].item())]


def _record_top(
    top: list[dict[str, Any]],
    score: Tensor,
    mask: Tensor,
    cpu_batch: Mapping[str, Any],
    *,
    a4_error: Tensor,
    history_error: Tensor,
    a5_error: Tensor,
    gate: Tensor,
    score_name: str = "oracle_opportunity_px",
) -> None:
    selected = torch.where(mask, score, torch.full_like(score, -torch.inf))
    flat_index = int(selected.reshape(-1).argmax().item())
    best = float(selected.reshape(-1)[flat_index].item())
    if not torch.isfinite(torch.tensor(best)):
        return
    height, width = score.shape[-2:]
    pixel = flat_index % (height * width)
    y, x = divmod(pixel, width)
    metadata = cpu_batch["identity_metadata"][0]
    row = {
        "dataset_index": int(metadata["dataset_index"]),
        "sequence_id": str(cpu_batch["sequence_id"][0]),
        "frame_id": int(cpu_batch["frame_ids"][0, -1].item()),
        "x": x,
        "y": y,
        score_name: best,
        "oracle_opportunity_px": float(
            (a4_error - history_error).reshape(-1)[flat_index].item()
        ),
        "a4_error_px": float(a4_error.reshape(-1)[flat_index].item()),
        "history_error_px": float(history_error.reshape(-1)[flat_index].item()),
        "a5_error_px": float(a5_error.reshape(-1)[flat_index].item()),
        "gate": float(gate.reshape(-1)[flat_index].item()),
    }
    top.append(row)
    top.sort(key=lambda item: item.get(score_name, -float("inf")), reverse=True)
    del top[10:]


def _finalize(stats: Mapping[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    return {
        name: {
            "value": numerator / count if count else None,
            "numerator": numerator,
            "count": int(round(count)),
            "valid": bool(count),
        }
        for name, (numerator, count) in sorted(stats.items())
    }


def main() -> int:
    args = _args()
    if args.opportunity_threshold_px <= 0:
        raise ValueError("--opportunity-threshold-px must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    context = _distributed_context()
    a4_config = _read_config(args.a4_config)
    a5_config = _read_config(args.a5_config)
    for config in (a4_config, a5_config):
        config.setdefault("data", {})["num_workers"] = int(args.num_workers)
    dataset = _dataset(a5_config, training=False)
    loader, sampler = _loader(dataset, a5_config, context, training=False)
    if sampler is not None:
        sampler.set_epoch(0)
    model_a4 = _load_model(a4_config, args.a4_checkpoint, context)
    model_a5 = _load_model(a5_config, args.a5_checkpoint, context)
    stats: dict[str, list[float]] = {}
    top_opportunities: list[dict[str, Any]] = []
    top_failures: list[dict[str, Any]] = []
    batches_seen = 0
    samples_seen = 0
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if int(cpu_batch["rgb"].shape[0]) != 1:
                raise RuntimeError("temporal analysis requires micro batch size one")
            dataset_index = int(cpu_batch["identity_metadata"][0]["dataset_index"])
            owner = dataset_index % context.world_size == context.rank
            batch = _move_batch(cpu_batch, context.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a4 = model_a4(batch)
                a5 = model_a5(batch)
            if owner:
                gt = batch["disparity_gt_left_px"][:, -1].float()
                gt_valid = batch["valid_gt_left"][:, -1].bool() & _finite_positive(gt)
                a4_pred = a4.endpoint.disparity_left_px.float()
                a5_pred = a5.endpoint.disparity_left_px.float()
                a4_valid = a4.endpoint.valid_mask.bool() & _finite_positive(a4.endpoint.depth_m)
                a5_valid = a5.endpoint.valid_mask.bool() & _finite_positive(a5.endpoint.depth_m)
                a4_error = (a4_pred - gt).abs()
                a5_error = (a5_pred - gt).abs()
                temporal = a5.endpoint.temporal
                current_factor = (
                    batch["K"][:, -1, 0, 0, 0].float()
                    * batch["baseline_m"][:, -1].float()
                ).reshape(-1, 1, 1, 1)
                history_inverse = temporal.warped_inverse_depth_pre_consistency_m_inv.float()
                history_lr_valid = temporal.zbuffer_visible_mask.bool() & _finite_positive(history_inverse)
                history_disparity_lr = history_inverse * current_factor
                history, history_valid = _masked_resize_scalar(
                    history_disparity_lr,
                    history_lr_valid,
                    gt.shape[-2:],
                )
                gt_warp = _warp_gt(batch)
                history_error = (history - gt).abs()
                matched_gt = gt_valid & gt_warp.valid_mask.bool()
                history_supported = matched_gt & history_valid.bool()
                opportunity = a4_error - history_error
                actual_gain = a4_error - a5_error
                hr_size = gt.shape[-2:]
                gate = F.interpolate(
                    temporal.learned_gate.float().mean(dim=1, keepdim=True),
                    size=hr_size,
                    mode="bilinear",
                    align_corners=False,
                )
                accepted = F.interpolate(
                    temporal.valid_mask.float(), size=hr_size, mode="nearest"
                ).bool()
                visible = F.interpolate(
                    temporal.zbuffer_visible_mask.float(), size=hr_size, mode="nearest"
                ).bool()
                collision = F.interpolate(
                    temporal.collision_mask.float(), size=hr_size, mode="nearest"
                ).bool()
                dynamic = batch["dynamic_mask_current"].bool()
                dynamic_available = batch["dynamic_mask_available"].bool().reshape(-1, 1, 1, 1)
                static = (~dynamic) & dynamic_available
                masks = {"matched": matched_gt, "history_supported": history_supported,
                         "static": matched_gt & static, "dynamic": matched_gt & dynamic,
                         "high_detail": matched_gt & _spring_masks(cpu_batch, dataset)["high_detail"].to(context.device)}
                for domain, domain_mask in masks.items():
                    _add(stats, f"{domain}_a4_epe_px", a4_error, domain_mask & a4_valid)
                    _add(stats, f"{domain}_a5_epe_px", a5_error, domain_mask & a5_valid)
                    _add(stats, f"{domain}_history_oracle_epe_px", history_error, domain_mask & history_valid)
                    _add(stats, f"{domain}_actual_gain_px", actual_gain, domain_mask & a5_valid)
                    _add(stats, f"{domain}_oracle_opportunity_px", opportunity, domain_mask & history_valid)
                    _add_rate(stats, f"{domain}_oracle_opportunity_rate", opportunity > args.opportunity_threshold_px, domain_mask & history_valid)
                    _add_rate(stats, f"{domain}_actual_gain_rate", actual_gain > args.opportunity_threshold_px, domain_mask & a5_valid)
                    _add_rate(stats, f"{domain}_oracle_but_unused_rate", (opportunity > args.opportunity_threshold_px) & (actual_gain <= args.opportunity_threshold_px), domain_mask & history_supported)
                    _add_rate(stats, f"{domain}_wrong_history_accept_rate", (history_error > a4_error + args.opportunity_threshold_px), domain_mask & accepted)
                    _add_rate(stats, f"{domain}_gate_accept_rate", accepted, domain_mask)
                    _add_rate(stats, f"{domain}_zbuffer_visible_rate", visible, domain_mask)
                    _add_rate(stats, f"{domain}_collision_rate", collision, domain_mask)
                    _add(stats, f"{domain}_gate_mean", gate, domain_mask)
                _add_rate(stats, "all_history_support_rate", history_supported, matched_gt)
                _add_rate(stats, "all_history_rejected_rate", history_supported & ~accepted, history_supported)
                _record_top(top_opportunities, opportunity, history_supported & (opportunity > args.opportunity_threshold_px), cpu_batch, a4_error=a4_error, history_error=history_error, a5_error=a5_error, gate=gate)
                _record_top(top_failures, a5_error - a4_error, history_supported & (opportunity > args.opportunity_threshold_px) & (actual_gain <= args.opportunity_threshold_px), cpu_batch, a4_error=a4_error, history_error=history_error, a5_error=a5_error, gate=gate, score_name="a5_degradation_px")
                samples_seen += 1
            batches_seen += 1
    _sync(stats, context)
    if context.world_size > 1:
        gathered: list[Any] = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered, {"opportunities": top_opportunities, "failures": top_failures})
        if context.primary:
            top_opportunities = sorted((row for payload in gathered for row in payload["opportunities"]), key=lambda item: item["oracle_opportunity_px"], reverse=True)[:10]
            top_failures = sorted((row for payload in gathered for row in payload["failures"]), key=lambda item: item.get("a5_degradation_px", 0.0), reverse=True)[:10]
    count_tensor = torch.tensor([samples_seen], dtype=torch.int64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    if context.primary:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "analysis": "independent_A4_vs_A5_temporal_opportunity",
            "a4_checkpoint": str(args.a4_checkpoint.expanduser().resolve()),
            "a5_checkpoint": str(args.a5_checkpoint.expanduser().resolve()),
            "samples": int(count_tensor.item()),
            "batches": batches_seen * context.world_size,
            "opportunity_threshold_px": float(args.opportunity_threshold_px),
            "definitions": {
                "oracle_opportunity": "E_A4 - abs(warp(A5_prev)-GT_current)",
                "actual_gain": "E_A4 - E_A5",
                "oracle_but_unused": "oracle_opportunity > threshold and actual_gain <= threshold on GT history-supported pixels",
                "wrong_history_accept": "abs(warp(A5_prev)-GT) > E_A4 + threshold where A5 gate accepts history",
            },
            "metrics": _finalize(stats),
            "top_oracle_opportunities": top_opportunities,
            "top_oracle_but_unused": top_failures,
        }
        (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (output_dir / "analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value", "numerator", "count", "valid"])
            writer.writeheader()
            for name, metric in report["metrics"].items():
                writer.writerow({"metric": name, **metric})
        print(json.dumps({"analysis": report["analysis"], "samples": report["samples"], "metrics": report["metrics"]}, sort_keys=True), flush=True)
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
