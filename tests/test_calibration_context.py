import pytest
import torch

from geometry.calibration_context import (
    rectified_stereo_transform_4x4,
    temporal_conditioning_transforms,
)


def _window(batch: int = 1) -> torch.Tensor:
    poses = torch.zeros(batch, 10, 3, 4)
    poses[:, :, :3, :3] = torch.eye(3)
    # camera-from-world translations for left views t-4..t
    for pair in range(5):
        poses[:, 2 * pair, 0, 3] = -float(pair)
        poses[:, 2 * pair + 1, 0, 3] = -float(pair) - 0.1
    return poses


def test_temporal_conditioning_is_causal_and_age_ordered() -> None:
    poses = _window()
    transforms, valid = temporal_conditioning_transforms(
        poses, torch.tensor([True]), student_time_index=2
    )
    assert valid.tolist() == [[True, True]]
    torch.testing.assert_close(transforms[0, 0, :3, 3], torch.tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(transforms[0, 1, :3, 3], torch.tensor([-2.0, 0.0, 0.0]))


def test_temporal_conditioning_invalid_slots_are_identity() -> None:
    transforms, valid = temporal_conditioning_transforms(
        _window(2), torch.tensor([True, False]), student_time_index=1
    )
    assert valid.tolist() == [[True, False], [False, False]]
    identity = torch.eye(4)
    torch.testing.assert_close(transforms[0, 1], identity)
    torch.testing.assert_close(transforms[1, 0], identity)
    torch.testing.assert_close(transforms[1, 1], identity)


def test_temporal_conditioning_composition_stays_fp32_inside_bf16_autocast() -> None:
    poses = _window()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        transforms, valid = temporal_conditioning_transforms(
            poses, torch.tensor([True]), student_time_index=2
        )
    assert transforms.dtype == torch.float32
    assert valid.tolist() == [[True, True]]
    rotations = transforms[0, :, :3, :3]
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations,
        torch.eye(3).expand(2, -1, -1),
        atol=1e-6,
        rtol=0.0,
    )


def test_temporal_conditioning_canonicalizes_inverse_homogeneous_row() -> None:
    # A numerically valid FP32 rigid window can accumulate a ~2e-6 residue in
    # the last row after inversion.  The public contract requires the exact
    # homogeneous row before the calibration conditioner sees it.
    generator = torch.Generator().manual_seed(30)
    poses = torch.zeros(1, 10, 3, 4)
    for pair in range(5):
        matrix = torch.randn(3, 3, generator=generator)
        rotation, _ = torch.linalg.qr(matrix)
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        translation = torch.randn(3, generator=generator) * 100.0
        poses[0, 2 * pair, :, :3] = rotation
        poses[0, 2 * pair, :, 3] = translation
        poses[0, 2 * pair + 1, :, :3] = rotation
        poses[0, 2 * pair + 1, :, 3] = translation + torch.tensor([0.1, 0.0, 0.0])
    transforms, valid = temporal_conditioning_transforms(
        poses, torch.tensor([True]), student_time_index=2
    )
    assert valid.tolist() == [[True, True]]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, -1)
    torch.testing.assert_close(transforms[0, :, 3], expected, atol=0.0, rtol=0.0)


def test_temporal_conditioning_rejects_future_index() -> None:
    with pytest.raises(ValueError, match="0, 1, or 2"):
        temporal_conditioning_transforms(
            _window(), torch.tensor([True]), student_time_index=3
        )


def test_rectified_stereo_transform_strips_homogeneous_row() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = -0.1
    result = rectified_stereo_transform_4x4(transform)
    assert result.shape == (1, 4, 4)
    torch.testing.assert_close(result, transform)


def test_rectified_stereo_transform_rejects_bad_bottom_row() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous row"):
        rectified_stereo_transform_4x4(transform)


def test_rectified_right_camera_center_is_positive_baseline() -> None:
    baseline_m = 0.12
    transform_right_from_left = torch.eye(4)
    transform_right_from_left[0, 3] = -baseline_m
    right_camera_from_world = transform_right_from_left
    rotation = right_camera_from_world[:3, :3]
    translation = right_camera_from_world[:3, 3]
    center_world = -rotation.transpose(0, 1) @ translation
    torch.testing.assert_close(
        center_world, torch.tensor([baseline_m, 0.0, 0.0])
    )


def test_hard_stereo_relative_transform_is_world_gauge_invariant() -> None:
    transform_right_from_left = torch.eye(4)
    transform_right_from_left[0, 3] = -0.12
    left_camera_from_world = torch.eye(4)
    left_camera_from_world[:3, 3] = torch.tensor([0.3, -0.2, 0.1])
    world_old_from_world_new = torch.eye(4)
    world_old_from_world_new[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    world_old_from_world_new[:3, 3] = torch.tensor([2.0, 3.0, -1.0])

    for left in (
        left_camera_from_world,
        left_camera_from_world @ world_old_from_world_new,
    ):
        right = transform_right_from_left @ left
        recovered = right @ torch.linalg.inv(left)
        torch.testing.assert_close(recovered, transform_right_from_left)
