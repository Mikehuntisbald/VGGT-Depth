"""Lightweight RGB pyramid used by the trainable TSR head."""

from __future__ import annotations

from dataclasses import dataclass

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

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 96)) -> None:
        super().__init__()
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError(f"channels must contain three positive values, got {channels}")
        channels_hr, channels_mid, channels_lr = channels
        self.channels = channels
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
        feature_lr = self.lr_stage(feature_hr)
        return RGBPyramidFeatures(feature_hr=feature_hr, feature_lr=feature_lr)
