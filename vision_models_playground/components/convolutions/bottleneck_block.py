from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from vision_models_playground.components.convolutions.conv_block import ConvBlock


class BottleneckBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            hidden_channels: Optional[int] = None,
    ):
        if hidden_channels is None:
            hidden_channels = out_channels

        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels, stride=stride, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, stride=1)
        self.conv3 = ConvBlock(hidden_channels, out_channels, stride=1, apply_relu=False)

        self.identity_reshape = ConvBlock(
            in_channels, out_channels, stride=stride, apply_relu=False, kernel_size=1, padding=0
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.identity_reshape(identity)
        out += identity
        out = F.relu(out)

        return out


def main():
    block = BottleneckBlock(in_channels=64, out_channels=128, hidden_channels=32, stride=1)

    x = torch.randn(1, 64, 32, 32)
    out = block(x)
    print(out.shape)


if __name__ == "__main__":
    main()
