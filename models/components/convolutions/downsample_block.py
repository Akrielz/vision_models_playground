from typing import Optional, Literal

import torch
from torch import nn
from torch.nn import functional as F

from models.components.convolutions.residual_block import ResidualBlock


class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            kernel_size: int = 2,
            pooling_type: Literal["avg", "max", "min"] = "avg",
    ):
        super(DownsampleBlock, self).__init__()

        self.conv = ResidualBlock(in_channels, out_channels, stride=1)

        self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride) if pooling_type == "avg" else \
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        out = self.pooling(out)
        return out
