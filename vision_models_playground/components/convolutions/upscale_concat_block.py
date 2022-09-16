from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vision_models_playground.components.convolutions.double_conv_block import DoubleConvBlock


class UpscaleConcatBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale: int = 2,
            conv_kernel_size: int = 3,
            conv_padding: int = 1,
            method: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "conv"] = "conv",
            crop: bool = False
    ):
        super().__init__()

        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=method, align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=scale+1, padding=scale-1)
        ) if method != "conv" else \
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale, stride=scale)

        self.conv = DoubleConvBlock(in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding)

        self.crop = crop

    def forward(self, x1, x2):
        # Upscale x1
        x1 = self.upscale(x1)

        # pad the first image to match the second shape
        diff_y = x2.shape[2] - x1.shape[2]
        left_y = diff_y // 2
        right_y = diff_y - left_y

        diff_x = x2.shape[3] - x1.shape[3]
        left_x = diff_x // 2
        right_x = diff_x - left_x

        x1 = F.pad(
            x1,
            [left_y, right_y,
             left_x, right_x]
        )

        # add the two images together
        x = torch.cat([x1, x2], dim=1)

        # pass the combined image through the convolutional layer
        x = self.conv(x)

        if self.crop:
            x = x[:, :, left_y: -right_y, left_x: -right_x]

        return x


def main():
    block = UpscaleConcatBlock(in_channels=1024, out_channels=512, scale=2, conv_kernel_size=3, conv_padding=1, method="conv", crop=True)
    x1 = torch.randn(1, 1024, 28, 28)
    x2 = torch.randn(1, 512, 64, 64)
    print(block(x1, x2).shape)


if __name__ == "__main__":
    main()