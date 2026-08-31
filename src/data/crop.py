"""Scale-aligned crop definitions and calibrated-intrinsics updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from geometry.camera import PinholeIntrinsics, crop_intrinsics


class IntegerGenerator(Protocol):
    """Minimal random-generator protocol needed for aligned crop sampling."""

    def integers(self, low: int, high: int | None = None) -> int: ...


def _require_plain_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def validate_crop_origin(
    crop_x_px: int,
    crop_y_px: int,
    spatial_scale: int,
) -> None:
    """Validate that an HR crop origin remains aligned to an LR grid.

    Both origins must be non-negative integer multiples of ``spatial_scale``.
    For x2 reconstruction this ensures an HR crop and its LR observation refer
    to exactly the same sample grid.
    """

    crop_x_px = _require_plain_int(crop_x_px, "crop_x_px")
    crop_y_px = _require_plain_int(crop_y_px, "crop_y_px")
    spatial_scale = _require_plain_int(spatial_scale, "spatial_scale")
    if crop_x_px < 0 or crop_y_px < 0:
        raise ValueError("crop origins must be non-negative")
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")
    if crop_x_px % spatial_scale or crop_y_px % spatial_scale:
        raise ValueError(
            "crop origin must be an integer multiple of spatial_scale: "
            f"origin=({crop_x_px}, {crop_y_px}), scale={spatial_scale}"
        )


@dataclass(frozen=True, slots=True)
class CropWindow:
    """One rectangular HR crop whose origin is aligned to the LR grid."""

    x_px: int
    y_px: int
    width_px: int
    height_px: int
    spatial_scale: int

    def __post_init__(self) -> None:
        width_px = _require_plain_int(self.width_px, "width_px")
        height_px = _require_plain_int(self.height_px, "height_px")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("crop width and height must be positive")
        validate_crop_origin(self.x_px, self.y_px, self.spatial_scale)
        if width_px % self.spatial_scale or height_px % self.spatial_scale:
            raise ValueError(
                "crop dimensions must be integer multiples of spatial_scale"
            )

    @property
    def x_stop_px(self) -> int:
        """Exclusive horizontal crop endpoint in HR pixels."""

        return self.x_px + self.width_px

    @property
    def y_stop_px(self) -> int:
        """Exclusive vertical crop endpoint in HR pixels."""

        return self.y_px + self.height_px

    @property
    def slices_hw(self) -> tuple[slice, slice]:
        """Return ``(height_slice, width_slice)`` for ``[..., H, W]`` data."""

        return (
            slice(self.y_px, self.y_stop_px),
            slice(self.x_px, self.x_stop_px),
        )

    @property
    def lr_size_hw(self) -> tuple[int, int]:
        """Return the exactly corresponding LR crop size as ``(H, W)``."""

        return (
            self.height_px // self.spatial_scale,
            self.width_px // self.spatial_scale,
        )

    def validate_within(self, image_height_px: int, image_width_px: int) -> None:
        """Raise if this crop falls outside an ``H x W`` source image."""

        image_height_px = _require_plain_int(image_height_px, "image_height_px")
        image_width_px = _require_plain_int(image_width_px, "image_width_px")
        if image_height_px <= 0 or image_width_px <= 0:
            raise ValueError("image dimensions must be positive")
        if self.x_stop_px > image_width_px or self.y_stop_px > image_height_px:
            raise ValueError(
                f"crop {self} exceeds image size "
                f"(height={image_height_px}, width={image_width_px})"
            )

    def crop_intrinsics(self, intrinsics_3x3: object) -> object:
        """Return matrix intrinsics expressed in this crop's HR coordinates."""

        return crop_intrinsics(intrinsics_3x3, self.x_px, self.y_px)

    def crop_pinhole(self, intrinsics: PinholeIntrinsics) -> PinholeIntrinsics:
        """Return scalar pinhole intrinsics for this crop."""

        if intrinsics.width_px is not None and intrinsics.height_px is not None:
            self.validate_within(intrinsics.height_px, intrinsics.width_px)
        return intrinsics.cropped(
            self.x_px,
            self.y_px,
            self.width_px,
            self.height_px,
        )


def sample_aligned_crop(
    image_height_px: int,
    image_width_px: int,
    crop_height_px: int,
    crop_width_px: int,
    spatial_scale: int,
    *,
    generator: IntegerGenerator | None = None,
) -> CropWindow:
    """Uniformly sample a valid crop over scale-aligned origin positions.

    A fresh default NumPy RNG is used if ``generator`` is omitted; training
    code should pass its seeded worker RNG for reproducibility.
    """

    image_height_px = _require_plain_int(image_height_px, "image_height_px")
    image_width_px = _require_plain_int(image_width_px, "image_width_px")
    crop_height_px = _require_plain_int(crop_height_px, "crop_height_px")
    crop_width_px = _require_plain_int(crop_width_px, "crop_width_px")
    spatial_scale = _require_plain_int(spatial_scale, "spatial_scale")
    if image_height_px <= 0 or image_width_px <= 0:
        raise ValueError("image dimensions must be positive")
    if crop_height_px <= 0 or crop_width_px <= 0:
        raise ValueError("crop dimensions must be positive")
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")
    if crop_height_px > image_height_px or crop_width_px > image_width_px:
        raise ValueError("crop dimensions exceed source image dimensions")
    if crop_height_px % spatial_scale or crop_width_px % spatial_scale:
        raise ValueError(
            "crop dimensions must be integer multiples of spatial_scale"
        )

    max_y_index = (image_height_px - crop_height_px) // spatial_scale
    max_x_index = (image_width_px - crop_width_px) // spatial_scale
    rng = generator if generator is not None else np.random.default_rng()
    y_index = int(rng.integers(0, max_y_index + 1))
    x_index = int(rng.integers(0, max_x_index + 1))
    crop = CropWindow(
        x_px=x_index * spatial_scale,
        y_px=y_index * spatial_scale,
        width_px=crop_width_px,
        height_px=crop_height_px,
        spatial_scale=spatial_scale,
    )
    crop.validate_within(image_height_px, image_width_px)
    return crop
