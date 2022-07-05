from einops import rearrange
from torch import nn


class ConvEmbedding(nn.Module):
    """
    Image to Conv Embedding
    """

    def __init__(
            self,
            patch_size: int = 7,
            in_channels: int = 3,
            out_channels: int = 64,
            stride: int = 4,
            padding: int = 2,
            apply_norm: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        self.norm = nn.LayerNorm(out_channels) if apply_norm else None

    def forward(self, x):
        x = self.proj(x)

        b, c, h, w = x.shape

        if self.norm is not None:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x
