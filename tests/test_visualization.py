import numpy as np
import torch

from tools.inspect_cache import _is_renderable_scalar_map_shape
from utils.visualization import grayscale_to_rgb_uint8, scalar_to_rgb_uint8, tensor_statistics


def test_scalar_colorization_marks_invalid_black() -> None:
    value = torch.tensor([[0.0, 1.0], [2.0, float("nan")]])
    image = scalar_to_rgb_uint8(value)
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[1, 1], np.zeros(3, dtype=np.uint8))


def test_grayscale_is_bounded() -> None:
    image = grayscale_to_rgb_uint8(np.asarray([[-1.0, 0.5, 2.0]], dtype=np.float32))
    assert image[0, 0, 0] == 0
    assert image[0, 1, 0] in (127, 128)
    assert image[0, 2, 0] == 255


def test_tensor_statistics_handles_bool() -> None:
    stats = tensor_statistics(torch.tensor([[True, False]]))
    assert stats["true_fraction"] == 0.5
    assert stats["shape"] == [1, 2]


def test_inspector_skips_pose_and_token_tensors_but_keeps_dense_maps() -> None:
    assert _is_renderable_scalar_map_shape((1, 1, 400, 640))
    assert _is_renderable_scalar_map_shape((1, 400, 640))
    assert not _is_renderable_scalar_map_shape((10, 3, 4))
    assert not _is_renderable_scalar_map_shape((10, 16, 2048))
    assert not _is_renderable_scalar_map_shape((10, 9))
