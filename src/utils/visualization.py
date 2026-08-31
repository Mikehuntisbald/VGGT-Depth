"""Small dependency-light visualizers for geometry cache inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


_COLOR_ANCHORS = np.asarray(
    [
        [10, 20, 80],
        [20, 100, 210],
        [30, 200, 160],
        [245, 210, 40],
        [220, 40, 30],
    ],
    dtype=np.float32,
)


def _as_numpy_2d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D scalar map after squeeze, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def scalar_to_rgb_uint8(
    value: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Colorize a scalar HxW map; invalid or non-finite pixels become black."""

    array = _as_numpy_2d(value)
    valid = np.isfinite(array)
    if valid_mask is not None:
        valid &= _as_numpy_2d(valid_mask).astype(bool)
    if not np.any(valid):
        return np.zeros(array.shape + (3,), dtype=np.uint8)
    finite_values = array[valid]
    low = float(np.quantile(finite_values, lower_quantile)) if minimum is None else float(minimum)
    high = float(np.quantile(finite_values, upper_quantile)) if maximum is None else float(maximum)
    if high <= low:
        high = low + 1.0
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    position = normalized * (_COLOR_ANCHORS.shape[0] - 1)
    lower_index = np.floor(position).astype(np.int64)
    upper_index = np.minimum(lower_index + 1, _COLOR_ANCHORS.shape[0] - 1)
    fraction = (position - lower_index)[..., None]
    rgb = _COLOR_ANCHORS[lower_index] * (1.0 - fraction) + _COLOR_ANCHORS[upper_index] * fraction
    rgb[~valid] = 0
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def grayscale_to_rgb_uint8(
    value: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> np.ndarray:
    """Render a bounded scalar map as RGB grayscale."""

    array = _as_numpy_2d(value)
    valid = np.isfinite(array)
    if valid_mask is not None:
        valid &= _as_numpy_2d(valid_mask).astype(bool)
    scale = max(maximum - minimum, np.finfo(np.float32).eps)
    gray = np.rint(np.clip((array - minimum) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=-1)


def save_rgb_uint8(path: Path, image_rgb_uint8: np.ndarray) -> None:
    """Save an HxWx3 uint8 RGB image, creating parent directories."""

    if image_rgb_uint8.dtype != np.uint8 or image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 RGB image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb_uint8, mode="RGB").save(path)


def tensor_statistics(value: Any) -> dict[str, Any]:
    """Return JSON-safe shape/dtype/finite statistics for a tensor-like value."""

    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    result: dict[str, Any] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if tensor.numel() == 0:
        return result | {"numel": 0}
    if tensor.dtype == torch.bool:
        return result | {
            "numel": tensor.numel(),
            "true_fraction": float(tensor.float().mean().item()),
        }
    floating = tensor.float()
    finite = torch.isfinite(floating)
    result["numel"] = tensor.numel()
    result["finite_fraction"] = float(finite.float().mean().item())
    if bool(finite.any().item()):
        values = floating[finite]
        result |= {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }
    return result
