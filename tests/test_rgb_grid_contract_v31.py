from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.rgb_encoder import ConvNormAct, RGBPyramidEncoder


def _identity_rgb_encoder(*, corrected: bool) -> RGBPyramidEncoder:
    encoder = RGBPyramidEncoder(
        (3, 3, 3),
        align_corners_false_pixel_centers=corrected,
    )
    encoder.hr_stage = nn.Identity()
    for block in encoder.lr_stage:
        assert isinstance(block, ConvNormAct)
        convolution = block[0]
        assert isinstance(convolution, nn.Conv2d)
        with torch.no_grad():
            convolution.weight.zero_()
            for channel in range(3):
                convolution.weight[channel, channel, 1, 1] = 1.0
        block[1] = nn.Identity()
        block[2] = nn.Identity()
    return encoder.eval()


def test_corrected_rgb_lr_grid_matches_align_corners_false_observation_centres() -> None:
    y, x = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(12, dtype=torch.float32),
        indexing="ij",
    )
    rgb = torch.stack((x, y, x + 10.0 * y), dim=0).unsqueeze(0)
    corrected = _identity_rgb_encoder(corrected=True)(rgb).feature_lr
    expected = F.avg_pool2d(rgb, kernel_size=2, stride=2)
    torch.testing.assert_close(corrected, expected, rtol=0, atol=1e-6)


def test_legacy_rgb_stride_two_grid_and_state_keys_remain_unchanged() -> None:
    rgb = torch.randn(1, 3, 8, 12)
    legacy = _identity_rgb_encoder(corrected=False)
    corrected = _identity_rgb_encoder(corrected=True)
    torch.testing.assert_close(
        legacy(rgb).feature_lr,
        rgb[..., ::2, ::2],
        rtol=0,
        atol=0,
    )
    assert tuple(legacy.state_dict()) == tuple(corrected.state_dict())


def test_corrected_rgb_grid_backpropagates_to_all_four_central_hr_phases() -> None:
    rgb = torch.zeros(1, 3, 4, 4, requires_grad=True)
    output = _identity_rgb_encoder(corrected=True)(rgb).feature_lr
    output[0, 0, 0, 0].backward()
    gradient = rgb.grad[0, 0]
    assert gradient[0:2, 0:2].gt(0).all()
    assert torch.count_nonzero(gradient) >= 4
