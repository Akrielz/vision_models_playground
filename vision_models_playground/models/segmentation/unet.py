from typing import Literal, List

import torch
from torch import nn

from vision_models_playground.components.convolutions import UpscaleConcatBlock, DoubleConvBlock, DownscaleBlock


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: List[int],
            pooling_type: Literal["avg", "max"] = "max",
            scale: int = 2,
            conv_kernel_size: int = 3,
            conv_padding: int = 1,
            method: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "conv"] = "conv",
            crop: bool = False
    ):
        super(UNet, self).__init__()

        # Make sure channels are given
        assert channels is not None, "channels must be specified"

        # Extract the number of layers from the list of channels
        num_layers = len(channels)

        # Prepare the layers vector
        self.downscale_layers = nn.ModuleList([])
        self.upscale_layers = nn.ModuleList([])

        # Create the first two convolutions
        self.input_conv = DoubleConvBlock(in_channels, channels[0], kernel_size=conv_kernel_size, padding=conv_padding)

        # Create the downscale layers
        for i in range(num_layers - 1):
            self.downscale_layers.append(
                DownscaleBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    scale=scale,
                    pooling_type=pooling_type,
                    conv_kernel_size=conv_kernel_size,
                    conv_padding=conv_padding,
                )
            )

        # Create the upscale layers
        for i in range(num_layers - 1):
            self.upscale_layers.append(
                UpscaleConcatBlock(
                    in_channels=channels[-i - 1],
                    out_channels=channels[-i - 2],
                    scale=scale,
                    conv_kernel_size=conv_kernel_size,
                    conv_padding=conv_padding,
                    method=method,
                    crop=crop
                )
            )

        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Pass the input through the input convolutional layer
        x = self.input_conv(x)

        # Pass the input through the downscale layers
        identity = []
        for downscale_layer in self.downscale_layers:
            identity.append(x)
            x = downscale_layer(x)

        # Pass the input through the upscale layers
        for i, upscale_layer in enumerate(self.upscale_layers):
            x = upscale_layer(x, identity[-i - 1])

        # Pass the input through the output convolutional layer
        x = self.output_conv(x)

        return x


def main():
    unet = UNet(
        in_channels=1,
        out_channels=2,
        channels=[64, 128, 256, 512, 1024],
        pooling_type="max",
        scale=2,
        conv_kernel_size=3,
        conv_padding=0,
        method="conv",
        crop=True
    )
    x = torch.randn(1, 1, 572, 572)
    print(unet(x).shape)


if __name__ == "__main__":
    main()
