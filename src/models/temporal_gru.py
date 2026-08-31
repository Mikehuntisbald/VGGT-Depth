"""Causal convolutional recurrent fusion at the LR grid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConvGRUCell(nn.Module):
    """A spatial GRU cell preserving ``[H,W]`` resolution."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("input_channels and hidden_channels must be positive")
        combined_channels = input_channels + hidden_channels
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            combined_channels,
            2 * hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.candidate = nn.Conv2d(
            combined_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, feature_lr: Tensor, hidden_lr: Tensor | None = None) -> Tensor:
        """Advance one causal step on tensors shaped ``[B,C,H,W]``."""

        if feature_lr.ndim != 4 or feature_lr.shape[1] != self.input_channels:
            raise ValueError(
                "feature_lr must have shape "
                f"[B,{self.input_channels},H,W], got {feature_lr.shape}"
            )
        expected_hidden_shape = (
            feature_lr.shape[0],
            self.hidden_channels,
            feature_lr.shape[2],
            feature_lr.shape[3],
        )
        if hidden_lr is None:
            hidden_lr = feature_lr.new_zeros(expected_hidden_shape)
        elif hidden_lr.shape != expected_hidden_shape:
            raise ValueError(
                f"hidden_lr must have shape {expected_hidden_shape}, got {hidden_lr.shape}"
            )
        elif hidden_lr.device != feature_lr.device:
            raise ValueError("feature_lr and hidden_lr must be on the same device")
        else:
            hidden_lr = hidden_lr.to(dtype=feature_lr.dtype)

        reset_gate, update_gate = torch.sigmoid(
            self.gates(torch.cat((feature_lr, hidden_lr), dim=1))
        ).chunk(2, dim=1)
        candidate_lr = torch.tanh(
            self.candidate(torch.cat((feature_lr, reset_gate * hidden_lr), dim=1))
        )
        return (1.0 - update_gate) * hidden_lr + update_gate * candidate_lr


class StackedConvGRU(nn.Module):
    """A two-or-more-layer causal ConvGRU with explicit recurrent state."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 96,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        cells: list[ConvGRUCell] = []
        for layer_index in range(num_layers):
            layer_input_channels = input_channels if layer_index == 0 else hidden_channels
            cells.append(ConvGRUCell(layer_input_channels, hidden_channels))
        self.cells = nn.ModuleList(cells)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def forward(
        self,
        feature_lr: Tensor,
        hidden_state: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Advance all layers and return top feature plus per-layer state.

        ``hidden_state`` belongs to the immediately preceding time step. No
        future feature is accepted by this interface, making recurrence causal.
        """

        if hidden_state is None:
            prior_states: tuple[Tensor | None, ...] = (None,) * self.num_layers
        else:
            if len(hidden_state) != self.num_layers:
                raise ValueError(
                    f"hidden_state needs {self.num_layers} tensors, got {len(hidden_state)}"
                )
            prior_states = tuple(hidden_state)

        layer_feature = feature_lr
        next_states: list[Tensor] = []
        for cell, prior_state in zip(self.cells, prior_states, strict=True):
            layer_feature = cell(layer_feature, prior_state)
            next_states.append(layer_feature)
        return layer_feature, tuple(next_states)
