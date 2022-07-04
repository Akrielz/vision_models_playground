from einops import rearrange
from torch import nn


class ConvEmbedding(nn.Module):
    """
    Image to Conv Embedding
    """

    def __init__(
            self,
            patch_size: int = 7,
            dim_in: int = 3,
            dim_out: int = 64,
            stride: int = 4,
            padding: int = 2,
            apply_norm: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            dim_in, dim_out,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        self.norm = nn.LayerNorm(dim_out) if apply_norm else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape

        if self.norm is not None:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x
