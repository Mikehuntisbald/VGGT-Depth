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


def test_dual_intrinsics_and_baseline_use_target_projection_and_disparity_unit() -> None:
    disparity_source_hr_px = torch.tensor([[[[0.1, 0.0, 0.0]]]], dtype=torch.float64)
    depth_source_m = torch.tensor([[[[2.0, 0.0, 0.0]]]], dtype=torch.float64)
    confidence = torch.ones_like(depth_source_m)
    source_k = _intrinsics(fx_px=2.0, fy_px=2.0, cx_px=0.0, cy_px=0.0)
    target_k = _intrinsics(fx_px=4.0, fy_px=3.0, cx_px=1.0, cy_px=0.0)
    identity = torch.eye(4, dtype=torch.float64)

    result = zbuffer_reproject(
        disparity_source_hr_px,
        depth_source_m,
        confidence,
        source_k,
        identity,
        identity,
        intrinsics_current_hr_3x3=target_k,
        baseline_previous_m=torch.tensor([0.1], dtype=torch.float64),
        baseline_current_m=torch.tensor([0.2], dtype=torch.float64),
    )

    # Source point (u=0,Z=2) is X=0. Target cx moves it to u=1 and target
    # disparity is fx_t*B_t/Z = 4*0.2/2 = 0.4 target-HR pixels.
    assert result.valid_mask.tolist() == [[[[False, True, False]]]]
    assert result.disparity_hr_px[0, 0, 0, 1].item() == pytest.approx(0.4)
    assert result.depth_m[0, 0, 0, 1].item() == pytest.approx(2.0)
    assert result.projected_uv[0, 0, 0, 1].item() == pytest.approx(1.0)


def test_dual_calibration_z_translation_recomputes_target_disparity() -> None:
    disparity_source_hr_px = torch.tensor([[[[0.1]]]], dtype=torch.float64)
    depth_source_m = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    confidence = torch.ones_like(depth_source_m)
    source_k = _intrinsics(fx_px=2.0, fy_px=2.0, cx_px=0.0, cy_px=0.0)
    target_k = _intrinsics(fx_px=4.0, fy_px=3.0, cx_px=0.0, cy_px=0.0)
    previous = torch.eye(4, dtype=torch.float64)
    current = torch.eye(4, dtype=torch.float64)
    current[2, 3] = 1.0

    result = zbuffer_reproject(
        disparity_source_hr_px,
        depth_source_m,
        confidence,
        source_k,
        previous,
        current,
        intrinsics_current_hr_3x3=target_k,
        baseline_previous_m=torch.tensor([0.1], dtype=torch.float64),
        baseline_current_m=torch.tensor([0.2], dtype=torch.float64),
    )

    assert bool(result.valid_mask.item())
    assert result.depth_m.item() == pytest.approx(3.0)
    assert result.disparity_hr_px.item() == pytest.approx(4.0 * 0.2 / 3.0)


def test_dual_calibration_rejects_inconsistent_source_depth() -> None:
    disparity = torch.tensor([[[[0.1]]]], dtype=torch.float64)
    wrong_depth = torch.tensor([[[[9.0]]]], dtype=torch.float64)
    calibration = _intrinsics(fx_px=2.0, fy_px=2.0, cx_px=0.0, cy_px=0.0)
    baseline = torch.tensor([0.1], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"fx\*baseline/disparity"):
        zbuffer_reproject(
            disparity,
            wrong_depth,
            torch.ones_like(disparity),
            calibration,
            torch.eye(4, dtype=torch.float64),
            torch.eye(4, dtype=torch.float64),
            intrinsics_current_hr_3x3=calibration,
            baseline_previous_m=baseline,
            baseline_current_m=baseline,
        )


def test_explicit_same_calibration_matches_legacy_zbuffer() -> None:
    disparity = torch.tensor([[[[2.0, 4.0, 8.0]]]], dtype=torch.float64)
    depth = 1.0 / disparity
    confidence = torch.tensor([[[[0.2, 0.5, 0.9]]]], dtype=torch.float64)
    calibration = _intrinsics(fx_px=5.0, fy_px=5.0, cx_px=1.0, cy_px=0.0)
    baseline = torch.tensor([0.2], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    legacy = zbuffer_reproject(
        disparity, depth, confidence, calibration, identity, identity
    )
    explicit = zbuffer_reproject(
        disparity,
        depth,
        confidence,
        calibration,
        identity,
        identity,
        intrinsics_current_hr_3x3=calibration,
        baseline_previous_m=baseline,
        baseline_current_m=baseline,
    )
    for name in (
        "disparity_hr_px",
        "depth_m",
        "confidence",
        "projected_uv",
        "fractional_offset",
        "source_uv",
        "valid_mask",
        "visibility_mask",
        "collision_mask",
    ):
        torch.testing.assert_close(getattr(explicit, name), getattr(legacy, name))
