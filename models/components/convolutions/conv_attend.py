from typing import Optional, Callable

from torch import nn

from models.components import PreNorm, FeedForward
from models.components.convolutions.conv_attention import ConvAttention
from models.components.dropout import DropPath


class ConvAttend(nn.Module):

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            num_heads: int,
            dim_per_head: int,
            qkv_bias: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            activation: Optional[Callable] = None,
            ff_hidden_dim: int = 256,
            **kwargs
    ):
        super().__init__()

        self.attention = PreNorm(
            dim=dim_in,
            fn=ConvAttention(
                in_channels=dim_in,
                out_channels=dim_out,
                num_heads=num_heads,
                dim_per_head=dim_per_head,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                output_drop=drop,
                **kwargs
            )
        )

        self.mlp = PreNorm(
            dim=dim_out,
            fn=FeedForward(
                dim=dim_out,
                hidden_dim=ff_hidden_dim,
                dropout=drop,
                activation=activation,
                output_dim=dim_out
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attention(x))
        x = x + self.drop_path(self.mlp(x))
        return x