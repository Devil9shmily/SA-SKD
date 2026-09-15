"""Temporal residual units shared by teacher and student models."""

from __future__ import annotations

from torch import Tensor, nn


class TemporalConvUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(dilation, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(dilation, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.net(x) + x)
