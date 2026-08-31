import pytest

torch = pytest.importorskip("torch")

from geometry.zbuffer_reproject import (
    relative_camera_transform,
    zbuffer_reproject,
)


def _intrinsics(fx_px: float = 2.0):
    return torch.tensor(
        [[fx_px, 0.0, 0.0], [0.0, fx_px, 0.0], [0.0, 0.0, 1.0]]
    )


def _camera_from_world_at_x(camera_center_world_x_m: float):
    extrinsics = torch.eye(4)
    # For R=I, camera-from-world t=-C_world.
    extrinsics[0, 3] = -camera_center_world_x_m
    return extrinsics


def test_positive_camera_x_motion_projects_history_to_the_left() -> None:
    disparity_hr_px = torch.zeros((1, 1, 1, 7))
    depth_m = torch.zeros_like(disparity_hr_px)
    confidence = torch.zeros_like(disparity_hr_px)
    disparity_hr_px[0, 0, 0, 3] = 2.0
    depth_m[0, 0, 0, 3] = 2.0
    confidence[0, 0, 0, 3] = 0.75

    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=4.0),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(0.5),
    )

    # u_current = 3 - fx * camera_translation / Z = 3 - 4*0.5/2 = 2.
    assert result.valid_mask[0, 0, 0].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert result.disparity_hr_px[0, 0, 0, 2].item() == pytest.approx(2.0)
    assert result.depth_m[0, 0, 0, 2].item() == pytest.approx(2.0)
    torch.testing.assert_close(
        result.projected_uv[0, :, 0, 2], torch.tensor([2.0, 0.0])
    )


def test_camera_from_world_relative_transform_direction_is_explicit() -> None:
    previous = _camera_from_world_at_x(1.0)
    current = _camera_from_world_at_x(0.0)

    transform_current_previous = relative_camera_transform(previous, current)

    # A previous-camera point X=0 is at world X=1, hence current-camera X=1.
    point_previous = torch.tensor([0.0, 0.0, 2.0, 1.0])
    point_current = transform_current_previous[0] @ point_previous
    torch.testing.assert_close(point_current, torch.tensor([1.0, 0.0, 2.0, 1.0]))

    disparity_hr_px = torch.zeros((1, 1, 1, 4))
    depth_m = torch.zeros_like(disparity_hr_px)
    confidence = torch.zeros_like(disparity_hr_px)
    disparity_hr_px[0, 0, 0, 1] = 1.0
    depth_m[0, 0, 0, 1] = 2.0
    confidence[0, 0, 0, 1] = 1.0
    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=2.0),
        previous,
        current,
    )
    # Returning from camera centre +1 m to centre 0 m shifts the point right.
    assert bool(result.valid_mask[0, 0, 0, 2])


def test_out_of_field_of_view_projection_is_invalid() -> None:
    disparity_hr_px = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    depth_m = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    confidence = torch.tensor([[[[1.0, 0.0, 0.0]]]])

    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=2.0),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(1.0),
    )

    assert not bool(result.valid_mask.any())
    assert not bool(result.visibility_mask.any())
    assert not bool(result.collision_mask.any())
    assert not bool(result.disparity_hr_px.any())


def test_zbuffer_collision_keeps_nearest_current_depth() -> None:
    # With fx=2 and camera motion +1 m:
    # source u=2,Z=1 -> target u=0; source u=1,Z=2 -> target u=0.
    disparity_hr_px = torch.zeros((1, 1, 1, 4))
    depth_m = torch.zeros_like(disparity_hr_px)
    confidence = torch.zeros_like(disparity_hr_px)
    disparity_hr_px[0, 0, 0, 1] = 1.0
    depth_m[0, 0, 0, 1] = 2.0
    confidence[0, 0, 0, 1] = 0.25
    disparity_hr_px[0, 0, 0, 2] = 2.0
    depth_m[0, 0, 0, 2] = 1.0
    confidence[0, 0, 0, 2] = 0.9

    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=2.0),
        _camera_from_world_at_x(0.0),
        _camera_from_world_at_x(1.0),
    )

    assert bool(result.valid_mask[0, 0, 0, 0])
    assert bool(result.visibility_mask[0, 0, 0, 0])
    assert bool(result.collision_mask[0, 0, 0, 0])
    assert result.depth_m[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert result.disparity_hr_px[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert result.confidence[0, 0, 0, 0].item() == pytest.approx(0.9)
