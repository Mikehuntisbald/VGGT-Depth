"""Pinhole-camera intrinsics with explicit pixel-coordinate operations.

The project uses the simple pixel scaling convention required by the data
contract: resizing an image by ``(scale_x, scale_y)`` multiplies the
corresponding focal length and principal point by the same factor. No
half-pixel offset is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

try:  # Torch is supplied by runtime environments, not the base package.
    import torch
except ImportError:  # pragma: no cover - only in minimal installs.
    torch = None  # type: ignore[assignment]


def _require_finite_real(value: Real, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value_float


def _require_positive_real(value: Real, name: str) -> float:
    value_float = _require_finite_real(value, name)
    if value_float <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value_float


def _copy_intrinsics_matrix(intrinsics_3x3: Any) -> Any:
    """Return a floating copy of one ``3 x 3`` intrinsics matrix."""

    if torch is not None and isinstance(intrinsics_3x3, torch.Tensor):
        if tuple(intrinsics_3x3.shape) != (3, 3):
            raise ValueError(
                "intrinsics_3x3 must have shape (3, 3), got "
                f"{tuple(intrinsics_3x3.shape)}"
            )
        matrix = (
            intrinsics_3x3.clone()
            if intrinsics_3x3.is_floating_point()
            else intrinsics_3x3.to(dtype=torch.float64)
        )
        if not bool(torch.isfinite(matrix).all().item()):
            raise ValueError("intrinsics_3x3 must contain only finite values")
        return matrix

    matrix = np.asarray(intrinsics_3x3)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"intrinsics_3x3 must have shape (3, 3), got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("intrinsics_3x3 must be numeric")
    matrix = matrix.astype(
        matrix.dtype if np.issubdtype(matrix.dtype, np.floating) else np.float64,
        copy=True,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("intrinsics_3x3 must contain only finite values")
    return matrix


def validate_intrinsics(intrinsics_3x3: Any, *, atol: float = 1e-8) -> None:
    """Validate one conventional pinhole intrinsics matrix.

    Args:
        intrinsics_3x3: NumPy array, Torch tensor, or nested sequence with shape
            ``(3, 3)``. Focal lengths are expressed in pixels.
        atol: Absolute tolerance used to check the homogeneous last row.

    Raises:
        ValueError: If the matrix has an invalid shape, non-finite entries,
            non-positive focal lengths, or a non-standard last row.
    """

    matrix = _copy_intrinsics_matrix(intrinsics_3x3)
    if torch is not None and isinstance(matrix, torch.Tensor):
        fx_px = float(matrix[0, 0].item())
        fy_px = float(matrix[1, 1].item())
        last_row = matrix[2].detach().cpu().numpy()
    else:
        fx_px = float(matrix[0, 0])
        fy_px = float(matrix[1, 1])
        last_row = np.asarray(matrix[2])
    _require_positive_real(fx_px, "fx_px")
    _require_positive_real(fy_px, "fy_px")
    if not np.allclose(last_row, (0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        raise ValueError(
            "intrinsics_3x3 must use homogeneous last row [0, 0, 1], "
            f"got {last_row.tolist()}"
        )


def crop_intrinsics(
    intrinsics_3x3: Any,
    crop_x_px: Real,
    crop_y_px: Real,
) -> Any:
    """Shift calibrated intrinsics into a cropped image coordinate system.

    Returns a new matrix of the same array/tensor family. The input is never
    modified. Its principal point obeys ``cx' = cx - crop_x_px`` and
    ``cy' = cy - crop_y_px``.
    """

    validate_intrinsics(intrinsics_3x3)
    crop_x_float = _require_finite_real(crop_x_px, "crop_x_px")
    crop_y_float = _require_finite_real(crop_y_px, "crop_y_px")
    if crop_x_float < 0.0 or crop_y_float < 0.0:
        raise ValueError("crop origins must be non-negative")
    cropped = _copy_intrinsics_matrix(intrinsics_3x3)
    cropped[0, 2] -= crop_x_float
    cropped[1, 2] -= crop_y_float
    return cropped


def resize_intrinsics(
    intrinsics_3x3: Any,
    scale_x: Real,
    scale_y: Real | None = None,
) -> Any:
    """Scale calibrated intrinsics for an image resize.

    Row zero is scaled by ``scale_x`` and row one by ``scale_y``. This includes
    focal lengths, principal points, and skew. The input matrix is unchanged.
    """

    validate_intrinsics(intrinsics_3x3)
    scale_x_float = _require_positive_real(scale_x, "scale_x")
    scale_y_float = (
        scale_x_float
        if scale_y is None
        else _require_positive_real(scale_y, "scale_y")
    )
    resized = _copy_intrinsics_matrix(intrinsics_3x3)
    resized[0, :] *= scale_x_float
    resized[1, :] *= scale_y_float
    return resized


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Scalar representation of calibrated pinhole intrinsics in pixels."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int | None = None
    height_px: int | None = None

    def __post_init__(self) -> None:
        _require_positive_real(self.fx_px, "fx_px")
        _require_positive_real(self.fy_px, "fy_px")
        _require_finite_real(self.cx_px, "cx_px")
        _require_finite_real(self.cy_px, "cy_px")
        for value, name in (
            (self.width_px, "width_px"),
            (self.height_px, "height_px"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")

    @classmethod
    def from_matrix(
        cls,
        intrinsics_3x3: Any,
        *,
        width_px: int | None = None,
        height_px: int | None = None,
    ) -> "PinholeIntrinsics":
        """Construct from one calibrated ``3 x 3`` matrix."""

        validate_intrinsics(intrinsics_3x3)
        matrix = _copy_intrinsics_matrix(intrinsics_3x3)
        if torch is not None and isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        return cls(
            fx_px=float(matrix[0, 0]),
            fy_px=float(matrix[1, 1]),
            cx_px=float(matrix[0, 2]),
            cy_px=float(matrix[1, 2]),
            width_px=width_px,
            height_px=height_px,
        )

    def as_matrix(self, *, dtype: Any = np.float64) -> np.ndarray:
        """Return a new NumPy ``3 x 3`` intrinsics matrix."""

        return np.asarray(
            (
                (self.fx_px, 0.0, self.cx_px),
                (0.0, self.fy_px, self.cy_px),
                (0.0, 0.0, 1.0),
            ),
            dtype=dtype,
        )

    def cropped(
        self,
        crop_x_px: int,
        crop_y_px: int,
        crop_width_px: int,
        crop_height_px: int,
    ) -> "PinholeIntrinsics":
        """Return intrinsics for a validated rectangular crop."""

        for value, name in (
            (crop_x_px, "crop_x_px"),
            (crop_y_px, "crop_y_px"),
            (crop_width_px, "crop_width_px"),
            (crop_height_px, "crop_height_px"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if crop_x_px < 0 or crop_y_px < 0:
            raise ValueError("crop origins must be non-negative")
        if crop_width_px <= 0 or crop_height_px <= 0:
            raise ValueError("crop dimensions must be positive")
        if self.width_px is not None and crop_x_px + crop_width_px > self.width_px:
            raise ValueError("crop exceeds calibrated image width")
        if self.height_px is not None and crop_y_px + crop_height_px > self.height_px:
            raise ValueError("crop exceeds calibrated image height")
        return PinholeIntrinsics(
            fx_px=self.fx_px,
            fy_px=self.fy_px,
            cx_px=self.cx_px - crop_x_px,
            cy_px=self.cy_px - crop_y_px,
            width_px=crop_width_px,
            height_px=crop_height_px,
        )

    def resized(
        self,
        scale_x: Real,
        scale_y: Real | None = None,
    ) -> "PinholeIntrinsics":
        """Return intrinsics expressed in resized-image pixel units."""

        scale_x_float = _require_positive_real(scale_x, "scale_x")
        scale_y_float = (
            scale_x_float
            if scale_y is None
            else _require_positive_real(scale_y, "scale_y")
        )
        width_px = (
            None
            if self.width_px is None
            else int(round(self.width_px * scale_x_float))
        )
        height_px = (
            None
            if self.height_px is None
            else int(round(self.height_px * scale_y_float))
        )
        if width_px == 0 or height_px == 0:
            raise ValueError("resize scale produces a zero-sized image")
        return PinholeIntrinsics(
            fx_px=self.fx_px * scale_x_float,
            fy_px=self.fy_px * scale_y_float,
            cx_px=self.cx_px * scale_x_float,
            cy_px=self.cy_px * scale_y_float,
            width_px=width_px,
            height_px=height_px,
        )
