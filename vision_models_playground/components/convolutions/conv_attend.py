from typing import Optional, Callable

import torch
from einops import rearrange
from torch import nn

from vision_models_playground.components.attention import PreNorm, FeedForward
from vision_models_playground.components.convolutions.conv_attention import ConvAttention
from vision_models_playground.components.dropout import DropPath


class ConvAttend(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            dim_per_head: int,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            activation: Optional[Callable] = None,
            ff_hidden_dim: int = 256,
            output_2d: bool = False,
            **kwargs
    ):
        super().__init__()

        self.output_2d = output_2d

        self.norm = nn.LayerNorm(in_channels)

        self.attention = ConvAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dim_per_head=dim_per_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            output_drop=drop,
            **kwargs
        )

        self.mlp = PreNorm(
            dim=out_channels,
            fn=FeedForward(
                dim=out_channels,
                hidden_dim=ff_hidden_dim,
                dropout=drop,
                activation=activation,
                output_dim=out_channels
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.identity_reshape = nn.Identity() if in_channels == out_channels \
            else nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")
        res = x
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)

        x = self.identity_reshape(res) + self.drop_path(self.attention(x))
        x = x + self.drop_path(self.mlp(x))

        if self.output_2d:
            x = rearrange(x, "b (h w) c -> b c h w", h=h)

        return x


def main():
    block = ConvAttend(
        in_channels=64,
        out_channels=128,
        num_heads=4,
        dim_per_head=32,
        qkv_bias=False,
        ff_hidden_dim=256,
    )

    x = torch.randn(1, 64, 32, 32)
    out = block(x)

    print(out.shape)


if __name__ == "__main__":
    main()
