import pytest

torch = pytest.importorskip("torch")

from geometry.zbuffer_reproject import zbuffer_reproject


def _intrinsics(*, fx_px: float, fy_px: float, cx_px: float, cy_px: float):
    return torch.tensor(
        [[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def test_identity_pose_preserves_disparity_depth_and_confidence() -> None:
    disparity_hr_px = torch.tensor(
        [[[[2.0, 4.0, 8.0], [1.0, 3.0, 6.0]]]], dtype=torch.float64
    )
    depth_m = 12.0 / disparity_hr_px
    confidence = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]], dtype=torch.float64
    )
    identity = torch.eye(4, dtype=torch.float64)

    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=4.0, fy_px=5.0, cx_px=1.0, cy_px=0.5),
        identity,
        identity,
    )

    torch.testing.assert_close(result.disparity_hr_px, disparity_hr_px)
    torch.testing.assert_close(result.depth_m, depth_m)
    torch.testing.assert_close(result.confidence, confidence)
    assert bool(result.valid_mask.all())
    assert bool(result.visibility_mask.all())
    assert not bool(result.collision_mask.any())
    torch.testing.assert_close(
        result.fractional_offset, torch.zeros_like(result.fractional_offset)
    )
    grid_v, grid_u = torch.meshgrid(
        torch.arange(disparity_hr_px.shape[-2]),
        torch.arange(disparity_hr_px.shape[-1]),
        indexing="ij",
    )
    expected_source_uv = (
        torch.stack((grid_u, grid_v), dim=0).unsqueeze(0).to(torch.float64)
    )
    torch.testing.assert_close(result.source_uv, expected_source_uv)
    assert not result.disparity_hr_px.requires_grad


def test_invalid_and_out_of_view_sources_leave_explicit_zero_holes() -> None:
    disparity_hr_px = torch.tensor([[[[2.0, 0.0, float("nan")]]]])
    depth_m = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    confidence = torch.ones_like(depth_m)
    identity = torch.eye(4)

    result = zbuffer_reproject(
        disparity_hr_px,
        depth_m,
        confidence,
        _intrinsics(fx_px=2.0, fy_px=2.0, cx_px=0.0, cy_px=0.0).float(),
        identity,
        identity,
    )

    assert result.valid_mask.tolist() == [[[[True, False, False]]]]
    assert result.disparity_hr_px.tolist() == [[[[2.0, 0.0, 0.0]]]]
    assert bool(torch.isfinite(result.disparity_hr_px).all())
    assert bool(torch.isfinite(result.projected_uv).all())
