"""Frozen VGGT-Omega adapter with auditable image-coordinate transforms.

The upstream preprocessor crops extreme aspect ratios, resizes to a patch-grid
shape, and finally pads heterogeneous image shapes with white pixels. Its
public function returns only the image tensor. This module reproduces that
public algorithm without modifying ``third_party/vggt-omega`` and records the
complete original-image <-> model-image transform for every frame.

Transforms use continuous image coordinates with the project's no-half-pixel
intrinsics convention. Crop rectangles are PIL-style half-open
``(left, top, right, bottom)`` boxes; sizes are always ``(height, width)``.

VGGT-Omega predicts geometry only up to scale. Decoded extrinsics are OpenCV
camera-from-world matrices ``[R | t]``. Predicted intrinsics are diagnostic
only; calibrated intrinsics remain the geometry owner. Upstream ``depth_conf``
is ``1 + exp(logit)`` and is therefore an unbounded score, not a probability.
VGGT-Omega has no causal switch: causality comes only from caller input order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
import torch.nn.functional as F


Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
PoseDecoder = Callable[..., tuple[Tensor, Tensor | None]]


@dataclass(frozen=True, slots=True)
class ImagePreprocessMetadata:
    """Coordinate provenance for one VGGT-Omega model input.

    All sizes are ``(height, width)``. ``crop_xyxy`` is the half-open crop in
    original-image coordinates. ``padding_lrtb`` follows
    :func:`torch.nn.functional.pad` order.
    """

    source_path: str
    original_size_hw: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    cropped_size_hw: tuple[int, int]
    resized_size_hw: tuple[int, int]
    resize_scale_xy: tuple[float, float]
    padding_lrtb: tuple[int, int, int, int]
    model_size_hw: tuple[int, int]
    original_to_model_3x3: Matrix3x3
    model_to_original_3x3: Matrix3x3

    @property
    def original_size(self) -> tuple[int, int]:
        """Compatibility alias returning ``(height, width)``."""

        return self.original_size_hw

    @property
    def resized_size(self) -> tuple[int, int]:
        """Compatibility alias returning pre-padding ``(height, width)``."""

        return self.resized_size_hw

    @property
    def crop_rectangle(self) -> tuple[int, int, int, int]:
        return self.crop_xyxy

    @property
    def scale(self) -> tuple[float, float]:
        """Return exact ``(scale_x, scale_y)`` after patch-grid rounding."""

        return self.resize_scale_xy

    @property
    def original_to_model_transform(self) -> Matrix3x3:
        return self.original_to_model_3x3

    @property
    def model_to_original_transform(self) -> Matrix3x3:
        return self.model_to_original_3x3

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe cache metadata with explicit conventions."""

        return {
            "source_path": self.source_path,
            "size_order": "height,width",
            "crop_order": "left,top,right,bottom; right/bottom exclusive",
            "transform_coordinate_convention": "continuous pixels; no half-pixel offset",
            "original_size_hw": list(self.original_size_hw),
            "crop_xyxy": list(self.crop_xyxy),
            "cropped_size_hw": list(self.cropped_size_hw),
            "resized_size_hw": list(self.resized_size_hw),
            "resize_scale_xy": list(self.resize_scale_xy),
            "padding_lrtb": list(self.padding_lrtb),
            "model_size_hw": list(self.model_size_hw),
            "original_to_model_3x3": [list(row) for row in self.original_to_model_3x3],
            "model_to_original_3x3": [list(row) for row in self.model_to_original_3x3],
        }


@dataclass(frozen=True, slots=True)
class PreprocessedVGGTOmegaInput:
    """Upstream-compatible images and per-image transform metadata."""

    images: Tensor
    metadata: tuple[ImagePreprocessMetadata, ...]


@dataclass(frozen=True, slots=True)
class VGGTOmegaOutput:
    """Frozen VGGT-Omega predictions for one ordered context window.

    ``depth`` is ``[B,S,H,W,1]`` positive camera-z depth up to scale.
    ``depth_conf`` is ``[B,S,H,W]`` and is an unbounded score >= 1.
    ``extrinsics`` is ``[B,S,3,4]`` OpenCV camera-from-world. Predicted
    intrinsics are diagnostic only. Calibrated intrinsics, when supplied, are
    retained separately and never replaced by the prediction.
    """

    depth: Tensor
    depth_conf: Tensor
    pose_enc: Tensor
    extrinsics: Tensor
    intrinsics_pred: Tensor
    camera_tokens: Tensor
    register_tokens: Tensor
    preprocessing: tuple[ImagePreprocessMetadata, ...]
    intrinsics_calibrated_original: Tensor | None
    intrinsics_calibrated_model: Tensor | None
    metadata: Mapping[str, Any]

    @property
    def camera_and_register_tokens(self) -> Tensor:
        """Reconstruct the upstream combined token tensor."""

        return torch.cat((self.camera_tokens, self.register_tokens), dim=2)


def _validate_preprocess_options(
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> None:
    if mode not in {"balanced", "max_size"}:
        raise ValueError("mode must be either 'balanced' or 'max_size'")
    if isinstance(image_resolution, bool) or not isinstance(image_resolution, int):
        raise TypeError("image_resolution must be an integer")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int):
        raise TypeError("patch_size must be an integer")
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")


def center_crop_supported_aspect_ratio(
    width: int,
    height: int,
    *,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[int, int, int, int]:
    """Return the exact center crop used by upstream VGGT-Omega.

    Aspect ratio is ``height / width``. The box is PIL-style half-open
    ``(left, top, right, bottom)`` in original-image coordinates.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if (
        not math.isfinite(min_aspect_ratio)
        or not math.isfinite(max_aspect_ratio)
        or min_aspect_ratio <= 0
        or max_aspect_ratio <= min_aspect_ratio
    ):
        raise ValueError("aspect-ratio limits must satisfy 0 < min < max")

    aspect_ratio = height / width
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def balanced_target_shape(
    aspect_ratio: float,
    image_resolution: int = 512,
    patch_size: int = 16,
) -> tuple[int, int]:
    """Return upstream ``balanced`` target shape as ``(height, width)``."""

    _validate_preprocess_options("balanced", image_resolution, patch_size)
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError(f"aspect_ratio must be finite and positive, got {aspect_ratio}")
    token_number = (image_resolution // patch_size) ** 2
    width_patches = np.sqrt(token_number / aspect_ratio)
    height_patches = token_number / width_patches
    width_patches = max(1, int(np.round(width_patches)))
    height_patches = max(1, int(np.round(height_patches)))
    return height_patches * patch_size, width_patches * patch_size


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def max_size_target_shape(
    aspect_ratio: float,
    image_resolution: int = 512,
    patch_size: int = 16,
) -> tuple[int, int]:
    """Return upstream ``max_size`` target shape as ``(height, width)``."""

    _validate_preprocess_options("max_size", image_resolution, patch_size)
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError(f"aspect_ratio must be finite and positive, got {aspect_ratio}")
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def symmetric_padding_to_shape(
    height: int,
    width: int,
    target_height: int,
    target_width: int,
) -> tuple[int, int, int, int]:
    """Return symmetric ``(left,right,top,bottom)`` padding."""

    if min(height, width, target_height, target_width) <= 0:
        raise ValueError("all spatial sizes must be positive")
    if height > target_height or width > target_width:
        raise ValueError(
            f"source {(height, width)} exceeds target {(target_height, target_width)}"
        )
    height_padding = target_height - height
    width_padding = target_width - width
    return (
        width_padding // 2,
        width_padding - width_padding // 2,
        height_padding // 2,
        height_padding - height_padding // 2,
    )


def preprocess_coordinate_transforms(
    crop_xyxy: tuple[int, int, int, int],
    resized_size_hw: tuple[int, int],
    padding_lrtb: tuple[int, int, int, int],
) -> tuple[Matrix3x3, Matrix3x3]:
    """Compose crop/resize/pad transforms under project convention."""

    left, top, right, bottom = crop_xyxy
    resized_height, resized_width = resized_size_hw
    pad_left, _pad_right, pad_top, _pad_bottom = padding_lrtb
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"crop must be non-empty, got {crop_xyxy}")
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError(f"resized size must be positive, got {resized_size_hw}")
    if min(padding_lrtb) < 0:
        raise ValueError(f"padding must be non-negative, got {padding_lrtb}")

    scale_x = resized_width / crop_width
    scale_y = resized_height / crop_height
    original_to_model: Matrix3x3 = (
        (scale_x, 0.0, float(pad_left) - scale_x * left),
        (0.0, scale_y, float(pad_top) - scale_y * top),
        (0.0, 0.0, 1.0),
    )
    model_to_original: Matrix3x3 = (
        (1.0 / scale_x, 0.0, float(left) - pad_left / scale_x),
        (0.0, 1.0 / scale_y, float(top) - pad_top / scale_y),
        (0.0, 0.0, 1.0),
    )
    return original_to_model, model_to_original


def _load_rgb_image(path: str | PathLike[str]) -> Image.Image:
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _pil_rgb_to_tensor(image: Image.Image) -> Tensor:
    """Match ``torchvision.transforms.ToTensor`` for an RGB PIL image."""

    array = np.array(image, dtype=np.uint8, copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB HWC image, got {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def preprocess_vggt_omega_images(
    image_path_list: Sequence[str | PathLike[str]],
    *,
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
) -> PreprocessedVGGTOmegaInput:
    """Run official-equivalent preprocessing and retain every transform.

    Returns images ``[S,3,H_model,W_model]`` in RGB ``[0,1]``. Differently
    shaped resized images are symmetrically white-padded to the maximum group
    shape, exactly as upstream.
    """

    _validate_preprocess_options(mode, image_resolution, patch_size)
    if len(image_path_list) == 0:
        raise ValueError("at least one image is required")

    tensors: list[Tensor] = []
    partial_records: list[dict[str, Any]] = []
    shapes: set[tuple[int, int]] = set()
    for raw_path in image_path_list:
        path = Path(raw_path)
        image = _load_rgb_image(path)
        original_width, original_height = image.size
        crop_xyxy = center_crop_supported_aspect_ratio(
            original_width, original_height
        )
        image = image.crop(crop_xyxy)
        crop_width, crop_height = image.size
        aspect_ratio = crop_height / crop_width
        if mode == "balanced":
            target_height, target_width = balanced_target_shape(
                aspect_ratio, image_resolution, patch_size
            )
        else:
            target_height, target_width = max_size_target_shape(
                aspect_ratio, image_resolution, patch_size
            )

        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = _pil_rgb_to_tensor(image)
        shape = (target_height, target_width)
        shapes.add(shape)
        tensors.append(tensor)
        partial_records.append(
            {
                "source_path": str(path.resolve()),
                "original_size_hw": (original_height, original_width),
                "crop_xyxy": crop_xyxy,
                "cropped_size_hw": (crop_height, crop_width),
                "resized_size_hw": shape,
                "resize_scale_xy": (
                    target_width / crop_width,
                    target_height / crop_height,
                ),
            }
        )

    model_height = max(shape[0] for shape in shapes)
    model_width = max(shape[1] for shape in shapes)
    if len(shapes) > 1:
        warnings.warn(
            f"Found images with different shapes: {shapes}; padding to a common size.",
            stacklevel=2,
        )

    padded_tensors: list[Tensor] = []
    metadata: list[ImagePreprocessMetadata] = []
    for tensor, record in zip(tensors, partial_records, strict=True):
        resized_height, resized_width = record["resized_size_hw"]
        padding_lrtb = symmetric_padding_to_shape(
            resized_height,
            resized_width,
            model_height,
            model_width,
        )
        if any(padding_lrtb):
            tensor = F.pad(tensor, padding_lrtb, mode="constant", value=1.0)
        padded_tensors.append(tensor)
        original_to_model, model_to_original = preprocess_coordinate_transforms(
            record["crop_xyxy"],
            record["resized_size_hw"],
            padding_lrtb,
        )
        metadata.append(
            ImagePreprocessMetadata(
                **record,
                padding_lrtb=padding_lrtb,
                model_size_hw=(model_height, model_width),
                original_to_model_3x3=original_to_model,
                model_to_original_3x3=model_to_original,
            )
        )

    return PreprocessedVGGTOmegaInput(
        images=torch.stack(padded_tensors),
        metadata=tuple(metadata),
    )


def load_and_preprocess_images_with_metadata(
    image_path_list: Sequence[str | PathLike[str]],
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
) -> tuple[Tensor, tuple[ImagePreprocessMetadata, ...]]:
    """Tuple-returning compatibility wrapper around the typed preprocessor."""

    result = preprocess_vggt_omega_images(
        image_path_list,
        mode=mode,
        image_resolution=image_resolution,
        patch_size=patch_size,
    )
    return result.images, result.metadata


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _validate_finite(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or infinite values")


def _normalize_calibrated_intrinsics(
    intrinsics: Tensor | Sequence[Any],
    *,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    calibrated = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    if calibrated.ndim == 3:
        calibrated = calibrated.unsqueeze(0)
    expected_shape = (1, sequence_length, 3, 3)
    if tuple(calibrated.shape) != expected_shape:
        raise ValueError(
            "calibrated intrinsics must have shape [S,3,3] or [1,S,3,3], "
            f"expected {expected_shape}, got {tuple(calibrated.shape)}"
        )
    _validate_finite("intrinsics_calibrated_original", calibrated)
    if not bool((calibrated[..., 0, 0] > 0).all().item()) or not bool(
        (calibrated[..., 1, 1] > 0).all().item()
    ):
        raise ValueError("calibrated focal lengths must be positive")
    expected_last_row = calibrated.new_tensor((0.0, 0.0, 1.0))
    if not torch.allclose(
        calibrated[..., 2, :], expected_last_row.expand_as(calibrated[..., 2, :])
    ):
        raise ValueError("calibrated intrinsics must have homogeneous row [0,0,1]")
    return calibrated


class VGGTOmegaAdapter(nn.Module):
    """Frozen wrapper for an instantiated official ``VGGTOmega`` model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        pose_decoder: PoseDecoder | None = None,
        input_mode: str = "balanced",
        image_resolution: int = 512,
        patch_size: int = 16,
        context_pairs: int = 5,
    ) -> None:
        super().__init__()
        _validate_preprocess_options(input_mode, image_resolution, patch_size)
        if (
            isinstance(context_pairs, bool)
            or not isinstance(context_pairs, int)
            or context_pairs <= 0
        ):
            raise ValueError(
                f"context_pairs must be a positive integer, got {context_pairs!r}"
            )
        self.model = model
        self.pose_decoder = pose_decoder
        self.input_mode = input_mode
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.context_pairs = context_pairs
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "VGGTOmegaAdapter":
        """Keep this cache-only adapter and upstream model in eval mode."""

        super().train(False)
        self.model.eval()
        return self

    def _pose_decoder(self) -> PoseDecoder:
        if self.pose_decoder is not None:
            return self.pose_decoder
        try:
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "official vggt_omega package is unavailable; install the pinned "
                "third_party/vggt-omega package or inject pose_decoder"
            ) from exc
        return encoding_to_camera

    @torch.inference_mode()
    def forward(
        self,
        image_paths_ordered: Sequence[str | PathLike[str]],
        *,
        intrinsics_calibrated_original: Tensor | Sequence[Any] | None = None,
    ) -> VGGTOmegaOutput:
        """Infer one caller-ordered causal stereo context.

        With the default five pairs callers must supply exactly
        ``L[t-4],R[t-4],...,L[t],R[t]``. Upstream has no causal flag; this
        method records, but cannot independently prove, caller ordering.
        """

        expected_images = 2 * self.context_pairs
        if len(image_paths_ordered) != expected_images:
            raise ValueError(
                f"expected exactly {expected_images} images as ordered stereo pairs, "
                f"got {len(image_paths_ordered)}"
            )
        preprocessed = preprocess_vggt_omega_images(
            image_paths_ordered,
            mode=self.input_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        )
        images = preprocessed.images.to(_model_device(self.model), non_blocking=True)
        self.model.eval()
        predictions = self.model(images)
        if not isinstance(predictions, Mapping):
            raise TypeError(
                "VGGT-Omega must return a prediction mapping, "
                f"got {type(predictions)!r}"
            )
        required_keys = {
            "depth",
            "depth_conf",
            "pose_enc",
            "camera_and_register_tokens",
        }
        missing = sorted(required_keys - predictions.keys())
        if missing:
            raise KeyError(f"VGGT-Omega predictions missing required keys: {missing}")

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        pose_enc = predictions["pose_enc"]
        combined_tokens = predictions["camera_and_register_tokens"]
        named_tensors = {
            "depth": depth,
            "depth_conf": depth_conf,
            "pose_enc": pose_enc,
            "camera_and_register_tokens": combined_tokens,
        }
        for name, tensor in named_tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"prediction {name!r} must be a Tensor, got {type(tensor)!r}"
                )
            _validate_finite(name, tensor)

        sequence_length = len(image_paths_ordered)
        model_height, model_width = images.shape[-2:]
        if tuple(depth.shape) != (1, sequence_length, model_height, model_width, 1):
            raise ValueError(
                "depth must have shape [1,S,H,W,1], got "
                f"{tuple(depth.shape)} for S/H/W="
                f"{(sequence_length, model_height, model_width)}"
            )
        if tuple(depth_conf.shape) != (
            1,
            sequence_length,
            model_height,
            model_width,
        ):
            raise ValueError(
                "depth_conf must have shape [1,S,H,W], got "
                f"{tuple(depth_conf.shape)}"
            )
        if tuple(pose_enc.shape) != (1, sequence_length, 9):
            raise ValueError(
                f"pose_enc must have shape [1,S,9], got {tuple(pose_enc.shape)}"
            )
        if (
            combined_tokens.ndim != 4
            or tuple(combined_tokens.shape[:2]) != (1, sequence_length)
        ):
            raise ValueError(
                "camera/register tokens must have shape [1,S,1+R,C], got "
                f"{tuple(combined_tokens.shape)}"
            )
        if combined_tokens.shape[2] < 2:
            raise ValueError(
                "combined token output must contain one camera and >=1 register token"
            )
        if not bool((depth > 0).all().item()):
            raise ValueError("VGGT-Omega depth must be positive")
        if not bool((depth_conf >= 1.0).all().item()):
            raise ValueError(
                "VGGT-Omega depth_conf violated upstream 1 + exp(logit) >= 1 contract"
            )

        decoded = self._pose_decoder()(
            pose_enc,
            (model_height, model_width),
            build_intrinsics=True,
        )
        if not isinstance(decoded, (tuple, list)) or len(decoded) != 2:
            raise TypeError("pose decoder must return (extrinsics, intrinsics_pred)")
        extrinsics, intrinsics_pred = decoded
        if not isinstance(extrinsics, Tensor) or not isinstance(intrinsics_pred, Tensor):
            raise TypeError("pose decoder must return two Tensors")
        if tuple(extrinsics.shape) != (1, sequence_length, 3, 4):
            raise ValueError(
                "decoded extrinsics must have shape [1,S,3,4], got "
                f"{tuple(extrinsics.shape)}"
            )
        if tuple(intrinsics_pred.shape) != (1, sequence_length, 3, 3):
            raise ValueError(
                "decoded predicted intrinsics must have shape [1,S,3,3], got "
                f"{tuple(intrinsics_pred.shape)}"
            )
        _validate_finite("extrinsics", extrinsics)
        _validate_finite("intrinsics_pred", intrinsics_pred)

        calibrated_original: Tensor | None = None
        calibrated_model: Tensor | None = None
        if intrinsics_calibrated_original is not None:
            calibrated_original = _normalize_calibrated_intrinsics(
                intrinsics_calibrated_original,
                sequence_length=sequence_length,
                device=extrinsics.device,
            )
            transforms = torch.tensor(
                [item.original_to_model_3x3 for item in preprocessed.metadata],
                dtype=calibrated_original.dtype,
                device=calibrated_original.device,
            ).unsqueeze(0)
            calibrated_model = transforms @ calibrated_original

        camera_tokens = combined_tokens[:, :, :1].contiguous()
        register_tokens = combined_tokens[:, :, 1:].contiguous()
        metadata: dict[str, Any] = {
            "input_order_contract": "L[t-4],R[t-4],...,L[t],R[t] for context_pairs=5",
            "causality_enforcement": "caller input order only; upstream has no causal flag",
            "context_pairs": self.context_pairs,
            "input_mode": self.input_mode,
            "image_resolution": self.image_resolution,
            "patch_size": self.patch_size,
            "model_input_shape_schw": list(images.shape),
            "frozen": True,
            "inference_mode": True,
            "depth_unit": "VGGT-Omega arbitrary scale; positive camera-z",
            "depth_conf_semantics": "unbounded score 1 + exp(logit), >=1; not probability",
            "extrinsics_convention": "OpenCV camera-from-world [R|t]",
            "intrinsics_pred_usage": "diagnostic only; never replaces calibrated intrinsics",
            "camera_token_count": 1,
            "register_token_count": int(register_tokens.shape[2]),
        }
        return VGGTOmegaOutput(
            depth=depth,
            depth_conf=depth_conf,
            pose_enc=pose_enc,
            extrinsics=extrinsics,
            intrinsics_pred=intrinsics_pred,
            camera_tokens=camera_tokens,
            register_tokens=register_tokens,
            preprocessing=preprocessed.metadata,
            intrinsics_calibrated_original=calibrated_original,
            intrinsics_calibrated_model=calibrated_model,
            metadata=metadata,
        )


__all__ = [
    "ImagePreprocessMetadata",
    "PreprocessedVGGTOmegaInput",
    "VGGTOmegaAdapter",
    "VGGTOmegaOutput",
    "balanced_target_shape",
    "center_crop_supported_aspect_ratio",
    "load_and_preprocess_images_with_metadata",
    "max_size_target_shape",
    "preprocess_coordinate_transforms",
    "preprocess_vggt_omega_images",
    "symmetric_padding_to_shape",
]
