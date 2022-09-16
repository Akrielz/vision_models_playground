from typing import Optional, Callable

import torch
from torch import nn

from vision_models_playground.components.attention.attention import Attention
from vision_models_playground.components.attention.feed_forward import FeedForward
from vision_models_playground.components.attention.pre_norm import PreNorm
from vision_models_playground.components.dropout import DropPath


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
            activation: Callable = None,
            drop_path: float = 0.0,
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            queries: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = queries + self.drop_path(self.attention(x=queries, context=context, mask=mask))
        x = x + self.drop_path(self.mlp(x=x))

        return x
