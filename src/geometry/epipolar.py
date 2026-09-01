"""Explicit row-coordinate contract for local rectified stereo matching."""

from __future__ import annotations


EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION = "audited_same_row_rectified_pixels_v1"

EPIPOLAR_GEOMETRY_CONTRACT = {
    "version": EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION,
    "horizontal_correspondence": "u_right=u_left-disparity-delta",
    "vertical_correspondence": "v_right=v_left",
    "runtime_right_row_scale": 1.0,
    "runtime_right_row_offset_hr_px": 0.0,
    "coordinate_frame": "shared cropped HR rectified-image pixels",
    "evidence": "required pixel-level epipolar rectification audit receipt",
    "calibration_note": (
        "K_right/P_right are retained as diagnostics but are not used for row "
        "mapping when they conflict with audited stored-image correspondences"
    ),
}


__all__ = [
    "EPIPOLAR_GEOMETRY_CONTRACT",
    "EPIPOLAR_PIXEL_ROW_CONTRACT_VERSION",
]
