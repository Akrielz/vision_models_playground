from typing import Optional, Callable, Literal

import torch
from torch import nn

from vision_models_playground.components.attention.feed_forward import FeedForward
from vision_models_playground.components.attention.attention import Attention
from vision_models_playground.components.attention.pre_norm import PreNorm
from vision_models_playground.components.attention.post_norm import PostNorm
from vision_models_playground.components.dropout import DropPath


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            head_dim: int,
            mlp_dim: int,
            mlp_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            apply_rotary_emb: bool = False,
            activation: Optional[Callable] = None,
            drop_path: float = 0.0,
            norm_type: Literal['pre_norm', 'post_norm'] = 'pre_norm',
    ):
        super().__init__()

        norm_layer = PreNorm if norm_type == 'pre_norm' else PostNorm

        self.self_attention = norm_layer(
            dim=dim,
            fn=Attention(
                query_dim=dim,
                heads=heads,
                dim_head=head_dim,
                dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb
            )
        )

        self.mlp = norm_layer(
            dim=dim,
            fn=FeedForward(
                dim=dim,
                hidden_dim=mlp_dim,
                dropout=mlp_dropout,
                activation=activation
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len, seq_len]
        """

        x = x + self.drop_path(self.self_attention(x=x, mask=mask))
        x = x + self.drop_path(self.mlp(x=x))

        return x
