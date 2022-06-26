from typing import Optional, Callable

import torch
from torch import nn

from models.components.attention.attention import Attention
from models.components.attention.feed_forward import FeedForward
from models.components.attention.pre_norm import PreNorm


class Attend(nn.Module):
    def __init__(
            self,
            query_dim: int,
            context_dim: Optional[int] = None,
            num_heads: int = 8,
            head_dim: int = 64,
            attention_dropout: float = 0.0,
            apply_rotary_emb: bool = False,
            ff_hidden_dim: int = 256,
            ff_dropout: float = 0.0,
            activation: Callable = None
    ):
        super(Attend, self).__init__()

        self.attention = PreNorm(
            dim=query_dim,
            fn=Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=num_heads,
                dim_head=head_dim,
                dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb
            ),
            context_dim=context_dim
        )

        self.mlp = PreNorm(
            dim=query_dim,
            fn=FeedForward(
                dim=query_dim,
                hidden_dim=ff_hidden_dim,
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
