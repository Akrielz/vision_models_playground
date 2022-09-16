import torch
from torch import nn
from torch.nn import functional as F

from vision_models_playground.components.convolutions.conv_block import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, apply_relu=False)

        self.identity_reshape = ConvBlock(
            in_channels, out_channels, stride=stride, apply_relu=False, kernel_size=1, padding=0
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.identity_reshape(identity)
        out = F.relu(out)

        return out


def main():
    block = ResidualBlock(in_channels=3, out_channels=128, stride=1)

    x = torch.randn(1, 3, 300, 300)
    out = block(x)
    print(out.shape)


if __name__ == "__main__":
    main()
