import numpy as np
import pytest

from geometry.disparity import (
    depth_from_disparity,
    disparity_from_depth,
    disparity_hr_to_lr_units,
    disparity_lr_to_hr_units,
)


def test_lr_disparity_is_multiplied_when_expressed_in_hr_pixels() -> None:
    disparity_lr_px = np.asarray([1.5, 24.0, 96.0], dtype=np.float32)

    disparity_hr_px = disparity_lr_to_hr_units(disparity_lr_px, scale=2)

    np.testing.assert_allclose(disparity_hr_px, [3.0, 48.0, 192.0])
    np.testing.assert_allclose(
        disparity_hr_to_lr_units(disparity_hr_px, scale=2), disparity_lr_px
    )


def test_hr_and_lr_disparity_formulas_produce_identical_metric_depth() -> None:
    scale = 2
    focal_length_hr_px = 800.0
    focal_length_lr_px = focal_length_hr_px / scale
    baseline_m = 0.12
    disparity_lr_px = np.asarray([4.0, 16.0, 80.0], dtype=np.float64)
    disparity_hr_px = disparity_lr_to_hr_units(disparity_lr_px, scale)

    depth_from_hr_m = depth_from_disparity(
        disparity_hr_px, focal_length_hr_px, baseline_m
    )
    depth_from_lr_m = depth_from_disparity(
        disparity_lr_px, focal_length_lr_px, baseline_m
    )

    np.testing.assert_allclose(depth_from_hr_m, depth_from_lr_m, rtol=1e-12)


def test_depth_disparity_round_trip_and_invalid_values_are_explicit() -> None:
    disparity_hr_px = np.asarray([48.0, 0.0, -1.0, np.nan, np.inf])

    depth_m = depth_from_disparity(
        disparity_hr_px, focal_length_px=800.0, baseline_m=0.12
    )

    assert depth_m[0] == pytest.approx(2.0)
    assert np.isnan(depth_m[1:]).all()
    recovered_disparity_hr_px = disparity_from_depth(
        depth_m, focal_length_px=800.0, baseline_m=0.12
    )
    assert recovered_disparity_hr_px[0] == pytest.approx(48.0)
    assert np.isnan(recovered_disparity_hr_px[1:]).all()


def test_empty_disparity_array_is_supported() -> None:
    disparity_lr_px = np.empty((0, 4), dtype=np.float32)

    disparity_hr_px = disparity_lr_to_hr_units(disparity_lr_px, scale=2)
    depth_m = depth_from_disparity(disparity_hr_px, 800.0, 0.12)

    assert disparity_hr_px.shape == (0, 4)
    assert depth_m.shape == (0, 4)


@pytest.mark.parametrize("scale", [0, -2, float("nan"), float("inf")])
def test_invalid_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        disparity_lr_to_hr_units(np.asarray([1.0]), scale)


def test_torch_conversion_preserves_device_shape_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    disparity_lr_px = torch.tensor([4.0, 8.0], requires_grad=True)

    disparity_hr_px = disparity_lr_to_hr_units(disparity_lr_px, scale=2)
    depth_m = depth_from_disparity(disparity_hr_px, 800.0, 0.12)

    assert disparity_hr_px.device == disparity_lr_px.device
    assert disparity_hr_px.shape == disparity_lr_px.shape
    assert torch.allclose(depth_m, torch.tensor([12.0, 6.0]))
    depth_m.sum().backward()
    assert disparity_lr_px.grad is not None
