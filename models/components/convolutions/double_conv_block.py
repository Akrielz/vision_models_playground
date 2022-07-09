from typing import Optional

from torch import nn


class DoubleConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            padding: int = 1,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
