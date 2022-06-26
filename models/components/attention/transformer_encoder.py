from typing import Optional, Callable

import torch
from torch import nn

from models.components.attention.attend import Attend


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            head_dim: int,
            mlp_dim: int,
            mlp_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            apply_rotary_emb: bool = False,
            activation: Optional[Callable] = None
    ):
        super().__init__()

        def get_self_attend() -> nn.Module:
            return Attend(
                query_dim=dim,
                context_dim=None,
                num_heads=heads,
                head_dim=head_dim,
                attention_dropout=attention_dropout,
                ff_hidden_dim=mlp_dim,
                ff_dropout=mlp_dropout,
                apply_rotary_emb=apply_rotary_emb,
                activation=activation
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(get_self_attend())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attend in self.layers:
            x = attend(queries=x)

        return x
