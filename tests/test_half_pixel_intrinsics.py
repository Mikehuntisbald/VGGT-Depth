from __future__ import annotations

import numpy as np
import pytest
import torch

import train
from geometry.camera import (
    crop_intrinsics,
    resize_intrinsics,
    resize_intrinsics_align_corners_false,
)


@pytest.mark.parametrize(
    ("scale", "expected_cx", "expected_cy", "legacy_error_px"),
    (
        (0.5, 319.75, 179.75, 0.25),
        (0.25, 159.625, 89.625, 0.375),
    ),
)
def test_align_corners_false_x2_x4_principal_point_contract(
    scale: float,
    expected_cx: float,
    expected_cy: float,
    legacy_error_px: float,
) -> None:
    K_hr = torch.tensor(
        [
            [[800.0, 2.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]],
            [[900.0, 0.0, 641.0], [0.0, 920.0, 361.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    original = K_hr.clone()
    corrected = resize_intrinsics_align_corners_false(K_hr, scale, scale)
    legacy = K_hr.clone()
    legacy[..., 0, :] *= scale
    legacy[..., 1, :] *= scale

    torch.testing.assert_close(K_hr, original, atol=0.0, rtol=0.0)
    assert corrected.dtype == K_hr.dtype and corrected.device == K_hr.device
    assert corrected[0, 0, 2].item() == pytest.approx(expected_cx)
    assert corrected[0, 1, 2].item() == pytest.approx(expected_cy)
    torch.testing.assert_close(corrected[..., 0, 0], K_hr[..., 0, 0] * scale)
    torch.testing.assert_close(corrected[..., 1, 1], K_hr[..., 1, 1] * scale)
    torch.testing.assert_close(corrected[..., 0, 1], K_hr[..., 0, 1] * scale)
    torch.testing.assert_close(
        legacy[..., :2, 2] - corrected[..., :2, 2],
        torch.full_like(corrected[..., :2, 2], legacy_error_px),
    )


def test_align_corners_false_crop_resize_commute_for_aligned_crop() -> None:
    K_hr = np.asarray(
        [[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    crop_x_hr, crop_y_hr = 100, 40
    scale_x, scale_y = 0.5, 0.25

    crop_then_resize = resize_intrinsics_align_corners_false(
        crop_intrinsics(K_hr, crop_x_hr, crop_y_hr), scale_x, scale_y
    )
    resize_then_crop = crop_intrinsics(
        resize_intrinsics_align_corners_false(K_hr, scale_x, scale_y),
        crop_x_hr * scale_x,
        crop_y_hr * scale_y,
    )
    np.testing.assert_allclose(crop_then_resize, resize_then_crop, atol=0.0)


def test_v3_lr_intrinsics_are_corrected_while_v2_is_byte_exact_legacy() -> None:
    K_hr = torch.tensor(
        [[[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    legacy = train._lr_intrinsics_from_hr(K_hr, scale=2)
    corrected = train._lr_intrinsics_from_hr(
        K_hr,
        scale=2,
        align_corners_false_pixel_centers=True,
    )
    expected_legacy = resize_intrinsics(K_hr[0], 0.5).unsqueeze(0)
    expected_corrected = resize_intrinsics_align_corners_false(K_hr, 0.5, 0.5)

    torch.testing.assert_close(legacy, expected_legacy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(corrected, expected_corrected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        legacy[..., :2, 2] - corrected[..., :2, 2],
        torch.full((1, 2), 0.25),
    )


def test_pixel_center_config_is_explicit_and_unknown_value_fails_closed() -> None:
    base = {
        "calibration_conditioning_v3": {
            "enabled": True,
            "protocol_version": train.CALIBRATION_CONDITIONING_V3_PROTOCOL,
            "use_rays": True,
            "use_stereo_pose": True,
            "use_temporal_pose": True,
        }
    }
    legacy = train.calibration_conditioning_v3_from_config(base)
    assert not legacy.align_corners_false_pixel_centers

    corrected_config = {
        "calibration_conditioning_v3": {
            **base["calibration_conditioning_v3"],
            "pixel_center_contract": (
                train.ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT
            ),
        }
    }
    corrected = train.calibration_conditioning_v3_from_config(corrected_config)
    assert corrected.align_corners_false_pixel_centers

    malformed = {
        "calibration_conditioning_v3": {
            **base["calibration_conditioning_v3"],
            "pixel_center_contract": "unknown",
        }
    }
    with pytest.raises(ValueError, match="pixel_center_contract"):
        train.calibration_conditioning_v3_from_config(malformed)


def test_align_corners_false_coordinate_and_ray_invariance() -> None:
    K_hr = torch.tensor(
        [[[20.0, 0.0, 5.5], [0.0, 24.0, 3.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    K_lr = resize_intrinsics_align_corners_false(K_hr, 0.25, 0.25)
    u_lr = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    v_lr = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    u_hr = (u_lr + 0.5) / 0.25 - 0.5
    v_hr = (v_lr + 0.5) / 0.25 - 0.5

    x_hr = (u_hr - K_hr[0, 0, 2]) / K_hr[0, 0, 0]
    y_hr = (v_hr - K_hr[0, 1, 2]) / K_hr[0, 1, 1]
    x_lr = (u_lr - K_lr[0, 0, 2]) / K_lr[0, 0, 0]
    y_lr = (v_lr - K_lr[0, 1, 2]) / K_lr[0, 1, 1]
    torch.testing.assert_close(x_lr, x_hr, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y_lr, y_hr, atol=1e-12, rtol=1e-12)
