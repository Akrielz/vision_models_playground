from typing import Literal

from einops.layers.torch import Rearrange
from torch import nn


class ConvProjection(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: int,
            method: Literal['dw_bn', 'avg', 'linear'],
            groups: int,
    ):
        super().__init__()

        if method == 'dw_bn':
            net = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                Rearrange("b c h w -> b (h w) c")
            )

        elif method == 'avg':
            net = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=True
                ),
                Rearrange("b c h w -> b (h w) c")
            )

        elif method == 'linear':
            net = nn.Identity()

        else:
            raise ValueError(f'Unknown method: {method}')

        self.net = net

    def forward(self, x):
        return self.net(x)
