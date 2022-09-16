from typing import Literal

import torch
from torch import nn

from vision_models_playground.components.convolutions.double_conv_block import DoubleConvBlock


class UpscaleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale: int = 2,
            conv_kernel_size: int = 3,
            conv_padding: int = 1,
            method: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "conv"] = "conv",
    ):
        super().__init__()

        upscale = nn.Upsample(scale_factor=scale, mode=method, align_corners=True) if method != "conv" \
            else nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale, stride=scale)

        conv = DoubleConvBlock(in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding)

        self.net = nn.Sequential(upscale, conv)

    def forward(self, x):
        return self.net(x)


def main():
    block = UpscaleBlock(in_channels=16, out_channels=32, scale=2, conv_kernel_size=3, conv_padding=1, mode="bicubic")
    x = torch.randn(1, 16, 32, 32)
    print(block(x).shape)


if __name__ == "__main__":
    main()
