from typing import Literal

import torch
from einops.layers.torch import Rearrange
from torch import nn


class ConvProjection(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1,
            method: Literal['conv', 'avg', 'linear'] = 'conv',
            bias: bool = False,
    ):
        super().__init__()

        # Create network according to the method
        if method == 'conv':
            net = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
            )

        elif method == 'avg':
            net = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=True
            )

        elif method == 'linear':
            net = nn.Identity()

        else:
            raise ValueError(f'Unknown method: {method}')

        # Create projections layer
        projection = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.Linear(in_channels, out_channels, bias=bias)
        )

        self.net = nn.Sequential(
            net, projection
        )

    def forward(self, x):
        return self.net(x)


def main():
    block = ConvProjection(in_channels=3, out_channels=60, kernel_size=3, padding=1, stride=1, method='conv')

    x = torch.randn(1, 3, 32, 32)
    out = block(x)  # [1, 32*32, 60]
    print(out.shape)


if __name__ == "__main__":
    main()