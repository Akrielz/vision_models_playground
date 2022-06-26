from typing import Optional, Callable

import torch
from torch import nn

from models.components.activations.relu_squared import ReluSquared
from models.components.attention.attention import Attention
from models.components.attention.feed_forward import FeedForward
from models.components.attention.pre_norm import PreNorm


class Attend(nn.Module):
    def __init__(
            self,
            query_dim: int,
            context_dim: Optional[int] = None,
            attn_heads: int = 8,
            attn_dim_head: int = 64,
            attn_dropout: float = 0.0,
            apply_rotary_emb: bool = False,
            hidden_dim: int = 256,
            ff_dropout: float = 0.0,
            activation: Callable = None
    ):
        super(Attend, self).__init__()

        self.attention = PreNorm(
            dim=query_dim,
            fn=Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                apply_rotary_emb=apply_rotary_emb
            ),
            context_dim=context_dim
        )

        self.mlp = PreNorm(
            dim=query_dim,
            fn=FeedForward(
                dim=query_dim,
                hidden_dim=hidden_dim,
                dropout=ff_dropout,
                activation=activation
            )
        )

    def forward(
            self,
            queries: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.attention(x=queries, context=context, mask=mask) + queries
        x = self.mlp(x=x) + x

        return x
