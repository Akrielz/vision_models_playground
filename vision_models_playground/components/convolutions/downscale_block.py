from typing import Literal

import torch
from torch import nn

from vision_models_playground.components.convolutions.double_conv_block import DoubleConvBlock


class DownscaleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale: int = 2,
            pooling_type: Literal["avg", "max"] = "max",
            conv_kernel_size: int = 3,
            conv_padding: int = 1,
    ):
        super().__init__()

        pool_class = nn.MaxPool2d if pooling_type == "max" else nn.AvgPool2d

        self.net = nn.Sequential(
            pool_class(kernel_size=scale),
            DoubleConvBlock(in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding),
        )

    def forward(self, x):
        return self.net(x)


def main():
    block = DownscaleBlock(in_channels=3, out_channels=64, scale=2, pooling_type="max", conv_kernel_size=3, conv_padding=1)
    x = torch.randn(1, 3, 32, 32)
    print(block(x).shape)


if __name__ == "__main__":
    main()