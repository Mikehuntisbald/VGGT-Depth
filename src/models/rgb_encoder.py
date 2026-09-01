"""Lightweight RGB pyramid used by the trainable TSR head."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor, nn


def _group_count(channels: int, preferred_groups: int = 8) -> int:
    """Return the largest useful GroupNorm divisor up to ``preferred_groups``."""

    for groups in range(min(channels, preferred_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """A 3x3 convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


@dataclass(frozen=True)
class RGBPyramidFeatures:
    """RGB features at the HR and x2-downsampled LR grids.

    Attributes:
        feature_hr: Tensor shaped ``[B, 32, 2H, 2W]``.
        feature_lr: Tensor shaped ``[B, 96, H, W]``.
    """

    feature_hr: Tensor
    feature_lr: Tensor


class RGBPyramidEncoder(nn.Module):
    """Encode current HR left RGB into HR guidance and a 96-channel LR feature.

    Only one stride-two operation is used, so this encoder is specifically the
    x2 MVP path. Input RGB values are expected to have already been normalized
    by the training data pipeline.
    """

    def __init__(
        self,
        channels: tuple[int, int, int] = (32, 64, 96),
        *,
        align_corners_false_pixel_centers: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError(f"channels must contain three positive values, got {channels}")
        channels_hr, channels_mid, channels_lr = channels
        self.channels = channels
        if not isinstance(align_corners_false_pixel_centers, bool):
            raise TypeError("align_corners_false_pixel_centers must be a bool")
        self.align_corners_false_pixel_centers = (
            align_corners_false_pixel_centers
        )
        self.hr_stage = nn.Sequential(
            ConvNormAct(3, channels_hr),
            ConvNormAct(channels_hr, channels_hr),
        )
        self.lr_stage = nn.Sequential(
            ConvNormAct(channels_hr, channels_mid, stride=2),
            ConvNormAct(channels_mid, channels_mid),
            ConvNormAct(channels_mid, channels_lr),
            ConvNormAct(channels_lr, channels_lr),
        )

    @property
    def hr_channels(self) -> int:
        return self.channels[0]

    @property
    def lr_channels(self) -> int:
        return self.channels[-1]

    def forward(self, rgb_hr: Tensor) -> RGBPyramidFeatures:
        """Encode ``rgb_hr`` shaped ``[B, 3, 2H, 2W]``."""

        if rgb_hr.ndim != 4 or rgb_hr.shape[1] != 3:
            raise ValueError(f"rgb_hr must have shape [B,3,2H,2W], got {rgb_hr.shape}")
        feature_hr = self.hr_stage(rgb_hr)
        if self.align_corners_false_pixel_centers:
            # FFS observations and calibrated LR rays live at continuous HR
            # centres ``(2j+0.5, 2i+0.5)``. A stride-2 k3/p1 convolution is
            # centred at integer HR coordinate ``(2j,2i)`` instead. Resample
            # the HR feature with the exact x2 bilinear centre-sampling
            # operator first, then
            # reuse the same stored convolution weights at stride one on the
            # physical LR grid. No parameter/state key changes are introduced.
            height_hr, width_hr = feature_hr.shape[-2:]
            if height_hr % 2 or width_hr % 2:
                raise ValueError(
                    "corrected x2 RGB grid requires even HR height and width"
                )
            # For an integer x2 reduction, a 2x2 average is exactly bilinear
            # point sampling at the align-corners-false centre.  The explicit
            # pooling kernel also has deterministic CUDA backward, unlike the
            # antialiased interpolate backward used by some PyTorch builds.
            feature_lr_input = functional.avg_pool2d(
                feature_hr, kernel_size=2, stride=2
            )
            downsample_block = self.lr_stage[0]
            assert isinstance(downsample_block, ConvNormAct)
            convolution = downsample_block[0]
            normalization = downsample_block[1]
            activation = downsample_block[2]
            assert isinstance(convolution, nn.Conv2d)
            feature_lr = functional.conv2d(
                feature_lr_input,
                convolution.weight,
                convolution.bias,
                stride=1,
                padding=convolution.padding,
                dilation=convolution.dilation,
                groups=convolution.groups,
            )
            feature_lr = activation(normalization(feature_lr))
            feature_lr = self.lr_stage[1:](feature_lr)
        else:
            feature_lr = self.lr_stage(feature_hr)
        return RGBPyramidFeatures(feature_hr=feature_hr, feature_lr=feature_lr)
