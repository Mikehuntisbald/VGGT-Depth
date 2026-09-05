#!/usr/bin/env python3
"""Render qualitative panels for the trained metric stereo-video model.

Every panel uses the same validation crop for all five shared-checkpoint
variants.  The layout is deliberately dense: input stereo, metric targets,
prediction/error maps, ablation predictions/errors, uncertainty, and causal
visibility diagnostics are shown together in one inspectable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.eval_metric_stereo_video import (  # noqa: E402
    VARIANTS,
    _load_model,
    _move_batch,
    _read_config,
    _set_variant,
)
from tools.train_metric_stereo_video import (  # noqa: E402
    _dataset,
    _distributed_context,
)
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples  # noqa: E402


VARIANT_LABELS = {
    "stereo_metric_prior_only": "Stereo prior",
    "stereo_plus_vggt_gauge": "Stereo + VGGT gauge",
    "stereo_plus_temporal_memory": "Stereo + temporal",
    "full_model": "Full model",
    "full_model_no_visibility_gating": "Full - visibility gate",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/metric_stereo_video/visualizations_final"),
    )
    parser.add_argument(
        "--samples-per-rank",
        type=int,
        default=1,
        help="number of fixed validation clips rendered by each GPU rank",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("diverse", "sequential"),
        default="diverse",
        help="choose one sequence per rank or use the distributed validation order",
    )
    return parser.parse_args()


def _map(value: torch.Tensor, index: int = 0) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[index, 0]
    elif tensor.ndim == 3:
        tensor = tensor[index]
    elif tensor.ndim == 2:
        pass
    else:
        raise ValueError(f"cannot render tensor with shape {tuple(tensor.shape)}")
    return tensor.numpy()


def _rgb(value: torch.Tensor, index: int = 0) -> np.ndarray:
    tensor = value.detach().float().cpu()[index].permute(1, 2, 0).numpy()
    return np.clip(tensor, 0.0, 1.0)


def _resize_map(value: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    resized = torch.nn.functional.interpolate(
        value.float(), size=size, mode="bilinear", align_corners=False
    )
    return resized[0, 0].detach().cpu().numpy()


def _finite_range(values: list[np.ndarray], *, positive: bool = False) -> tuple[float, float]:
    merged = np.concatenate([value[np.isfinite(value)] for value in values])
    if positive:
        merged = merged[merged > 0]
    if merged.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(merged, [0.02, 0.98]).astype(float)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(merged)), float(np.nanmax(merged))
    if high <= low:
        high = low + 1.0
    return low, high


def _imshow(
    axis: Any,
    image: np.ndarray,
    *,
    title: str,
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> None:
    shown = image.copy()
    if mask is not None:
        shown = np.where(mask, shown, np.nan)
    if shown.ndim == 3:
        axis.imshow(np.clip(shown, 0.0, 1.0))
    else:
        axis.imshow(shown, cmap=cmap or "turbo", norm=Normalize(*limits) if limits else None)
    axis.set_title(title, fontsize=8, pad=3, color="white")
    axis.axis("off")


def _sample_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> str:
    usable = valid & np.isfinite(pred) & np.isfinite(target) & (target > 0)
    if not usable.any():
        return "no valid GT"
    epe = float(np.abs(pred[usable] - target[usable]).mean())
    bad1 = float((np.abs(pred[usable] - target[usable]) > 1.0).mean() * 100.0)
    return f"EPE {epe:.3f}px | bad1 {bad1:.1f}%"


def _render_panel(
    *,
    output_dir: Path,
    rank: int,
    sample_index: int,
    batch: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> Path:
    height, width = batch["rgb"].shape[-2:]
    left = _rgb(batch["rgb"][:, -1, 0], 0)
    right = _rgb(batch["rgb"][:, -1, 1], 0)
    target_disp = _map(batch["disparity_gt_left_px"][:, -1])
    target_valid = _map(batch["valid_gt_left"][:, -1]).astype(bool)
    fx = float(batch["K"][0, -1, 0, 0, 0].item())
    baseline = float(batch["baseline_m"][0, -1].item())
    target_depth = np.where(target_valid, fx * baseline / np.maximum(target_disp, 1e-8), np.nan)
    dynamic = _map(batch["dynamic_mask_current"]).astype(bool)
    full_output = outputs["full_model"]
    full_disp = _map(full_output.disparity_left_px)
    full_depth = _map(full_output.depth_m)
    uncertainty = _map(full_output.endpoint.uncertainty)
    confidence = _map(full_output.confidence)
    visibility = _resize_map(full_output.endpoint.temporal.zbuffer_visible_mask.float(), (height, width))
    collision = _resize_map(full_output.endpoint.temporal.collision_mask.float(), (height, width))
    warp = _resize_map(
        full_output.endpoint.temporal.warped_inverse_depth_pre_consistency_m_inv,
        (height, width),
    )
    current_inverse = np.reciprocal(np.maximum(full_depth, 1e-8))
    warp_residual = np.abs(np.log(np.maximum(current_inverse, 1e-8) / np.maximum(warp, 1e-8)))
    prior = _resize_map(outputs["full_model"].stereo.disparity_left_hr_px_lr_grid[:, -1], (height, width))

    variant_disparities = {
        name: _map(output.disparity_left_px)
        for name, output in outputs.items()
    }
    disp_low, disp_high = _finite_range([target_disp, prior, *variant_disparities.values()], positive=True)
    depth_low, depth_high = _finite_range([target_depth, full_depth], positive=True)
    error_high = max(1.0, float(np.nanquantile(np.abs(full_disp - target_disp)[target_valid], 0.98)))
    log_error = np.abs(np.log(np.maximum(full_depth, 1e-8) / np.maximum(target_depth, 1e-8)))
    log_error_high = max(0.1, float(np.nanquantile(log_error[target_valid], 0.98)))

    figure, axes = plt.subplots(4, 6, figsize=(18, 12), dpi=150)
    figure.patch.set_facecolor("#10141c")
    for axis in axes.flat:
        axis.set_facecolor("#10141c")
    _imshow(axes[0, 0], left, title="Current left RGB")
    _imshow(axes[0, 1], right, title="Current right RGB")
    _imshow(axes[0, 2], target_disp, title="GT disparity", limits=(disp_low, disp_high), mask=target_valid)
    _imshow(axes[0, 3], prior, title="FFS metric prior", limits=(disp_low, disp_high), mask=prior > 0)
    _imshow(axes[0, 4], full_disp, title="Full disparity | " + _sample_metrics(full_disp, target_disp, target_valid), limits=(disp_low, disp_high), mask=target_valid)
    _imshow(axes[0, 5], target_depth, title="GT metric depth (m)", cmap="viridis", limits=(depth_low, depth_high), mask=target_valid)

    _imshow(axes[1, 0], full_depth, title="Full metric depth (m)", cmap="viridis", limits=(depth_low, depth_high), mask=full_depth > 0)
    _imshow(axes[1, 1], np.abs(full_disp - target_disp), title="Full |disp - GT|", cmap="magma", limits=(0.0, error_high), mask=target_valid)
    _imshow(axes[1, 2], log_error, title="Full log-depth error", cmap="magma", limits=(0.0, log_error_high), mask=target_valid)
    _imshow(axes[1, 3], uncertainty, title="Predicted uncertainty", cmap="magma", limits=_finite_range([uncertainty], positive=True), mask=confidence > 0)
    _imshow(axes[1, 4], confidence, title="Confidence", cmap="viridis", limits=(0.0, 1.0))
    _imshow(axes[1, 5], dynamic.astype(np.float32), title="Dynamic mask", cmap="coolwarm", limits=(0.0, 1.0))

    for column, name in enumerate(VARIANTS):
        prediction = variant_disparities[name]
        _imshow(
            axes[2, column],
            prediction,
            title=VARIANT_LABELS[name],
            limits=(disp_low, disp_high),
            mask=prediction > 0,
        )
        axes[2, column].text(
            0.02,
            0.04,
            _sample_metrics(prediction, target_disp, target_valid),
            transform=axes[2, column].transAxes,
            color="white",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        error = np.abs(prediction - target_disp)
        _imshow(
            axes[3, column],
            error,
            title="error vs GT",
            cmap="magma",
            limits=(0.0, error_high),
            mask=target_valid,
        )

    # The sixth column is reserved for causal memory diagnostics.
    _imshow(axes[2, 5], visibility, title="Temporal visibility", cmap="viridis", limits=(0.0, 1.0))
    _imshow(axes[3, 5], warp_residual, title="Temporal warp log residual", cmap="magma", limits=(0.0, max(0.1, float(np.nanquantile(warp_residual[np.isfinite(warp_residual)], 0.98)))) , mask=visibility > 0)
    axes[3, 5].text(
        0.02,
        0.04,
        f"collision {collision.mean():.3f}",
        transform=axes[3, 5].transAxes,
        color="white",
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
    )

    sequence = str(batch.get("sequence_id", ["unknown"])[0])
    frame_ids = batch["frame_ids"][0].detach().cpu().tolist()
    endpoint_frame = frame_ids[-1]
    figure.suptitle(
        f"Metric Stereo Video Geometry | seq {sequence} | endpoint frame {endpoint_frame} | rank {rank}",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Same validation crop and same trained checkpoint across all variants | black = invalid / unsupported",
        ha="center",
        color="#c7d0df",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    output_path = output_dir / f"rank_{rank:02d}_sample_{sample_index:03d}_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> int:
    args = _args()
    if args.samples_per_rank <= 0:
        raise ValueError("--samples-per-rank must be positive")
    context = _distributed_context()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else checkpoint.parents[1] / "resolved_config.yaml"
    config = _read_config(config_path)
    config["data"]["num_workers"] = int(args.num_workers)
    dataset = _dataset(config, training=False)
    model = _load_model(config, checkpoint, context)
    output_dir = args.output_dir.expanduser().resolve()
    rendered: list[str] = []
    if args.selection == "diverse":
        indices_by_sequence: dict[str, list[int]] = {}
        for index, candidate in enumerate(dataset.candidates):
            indices_by_sequence.setdefault(candidate.sequence_id, []).append(index)
        sequences = sorted(indices_by_sequence)
        if context.world_size > len(sequences):
            raise ValueError(
                "diverse visualization requires world_size no larger than the "
                f"number of validation sequences ({len(sequences)})"
            )
        selected = indices_by_sequence[sequences[context.rank]]
        if args.samples_per_rank == 1:
            local_indices = [selected[len(selected) // 2]]
        else:
            local_indices = [
                selected[round(position)]
                for position in np.linspace(
                    0, len(selected) - 1, args.samples_per_rank
                )
            ]
        cpu_batches = [
            collate_raw_stereo_video_samples([dataset[index]])
            for index in local_indices
        ]
    else:
        # A deterministic strided order preserves collective parity across
        # ranks while yielding adjacent validation endpoints in aggregate.
        local_indices = [
            context.rank + context.world_size * index
            for index in range(args.samples_per_rank)
        ]
        cpu_batches = [
            collate_raw_stereo_video_samples([dataset[index]])
            for index in local_indices
        ]
    with torch.inference_mode():
        for sample_index, cpu_batch in enumerate(cpu_batches):
            batch = _move_batch(cpu_batch, context.device)
            outputs: dict[str, Any] = {}
            for name, settings in VARIANTS.items():
                _set_variant(model, settings)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs[name] = model(batch)
            # Batch size is one for the formal config; retain the explicit
            # sample index in the filename if a caller changes it.
            for item_index in range(int(batch["rgb"].shape[0])):
                single_batch = {
                    key: (value[item_index : item_index + 1] if isinstance(value, torch.Tensor) and value.shape[0] == batch["rgb"].shape[0] else value)
                    for key, value in batch.items()
                }
                single_outputs = {
                    name: output
                    for name, output in outputs.items()
                }
                path = _render_panel(
                    output_dir=output_dir,
                    rank=context.rank,
                    sample_index=sample_index * int(batch["rgb"].shape[0]) + item_index,
                    batch=single_batch,
                    outputs=single_outputs,
                )
                rendered.append(str(path))
    if context.world_size > 1:
        # Publish the gallery only after every rank has finished its panel.
        dist.barrier()
    if context.primary:
        panel_paths = sorted(output_dir.glob("rank_*_comparison.png"))
        if panel_paths:
            thumbnails: list[Image.Image] = []
            for path in panel_paths:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((900, 560), Image.Resampling.LANCZOS)
                    tile = Image.new("RGB", (920, 610), "#10141c")
                    tile.paste(image, ((920 - image.width) // 2, 36))
                    draw = ImageDraw.Draw(tile)
                    draw.text((16, 10), path.stem, fill="white")
                    thumbnails.append(tile)
            columns = 2
            rows = (len(thumbnails) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * 920, rows * 610), "#10141c")
            for index, tile in enumerate(thumbnails):
                sheet.paste(tile, ((index % columns) * 920, (index // columns) * 610))
            sheet.save(output_dir / "00_contact_sheet.jpg", quality=92)
        (output_dir / "README.txt").write_text(
            "Panels show the same fixed validation crops across the five shared-checkpoint variants.\n"
            "Rows: input/GT/prior/full, full-model diagnostics, variant disparities, variant absolute errors.\n"
            "This is a qualitative visualization; no new training or checkpoint is produced.\n",
            encoding="utf-8",
        )
        print("\n".join(rendered), flush=True)
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
