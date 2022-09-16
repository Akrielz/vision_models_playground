from torch import nn


class ConvTransposedBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            apply_relu: bool = True,
            kernel_size: int = 3,
            padding: int = 1,
            bias: bool = False
    ):
        super(ConvTransposedBlock, self).__init__()

        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if apply_relu else nn.Identity()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out
