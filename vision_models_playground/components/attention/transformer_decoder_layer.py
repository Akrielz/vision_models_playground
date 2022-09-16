from typing import Optional, Callable, Literal

import torch
from torch import nn

from vision_models_playground.components.attention.feed_forward import FeedForward
from vision_models_playground.components.attention.attention import Attention
from vision_models_playground.components.attention.pre_norm import PreNorm
from vision_models_playground.components.attention.post_norm import PostNorm
from vision_models_playground.components.dropout import DropPath


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            target_dim: int,
            context_dim: int,
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
            dim=target_dim,
            fn=Attention(
                query_dim=target_dim,
                heads=heads,
                dim_head=head_dim,
                dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb
            )
        )

        self.cross_attention = norm_layer(
            dim=target_dim,
            fn=Attention(
                query_dim=target_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=head_dim,
                dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb
            ),
            context_dim=context_dim,
        )

        self.mlp = norm_layer(
            dim=target_dim,
            fn=FeedForward(
                dim=target_dim,
                hidden_dim=mlp_dim,
                dropout=mlp_dropout,
                activation=activation
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            target: torch.Tensor,
            context: Optional = None,
            target_mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            target: [batch_size, seq_len_1, dim_1]
            context: [batch_size, seq_len_2, dim_2]
            target_mask: [batch_size, seq_len_1, seq_len_1]
            context_mask: [batch_size, seq_len_1, seq_len_2]
        """

        target = target + self.drop_path(self.self_attention(x=target, mask=target_mask))
        target = target + self.drop_path(self.cross_attention(x=target, context=context, mask=context_mask))
        target = target + self.drop_path(self.mlp(x=target))

        return target
