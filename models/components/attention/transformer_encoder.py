import torch
from torch import nn

from models.components.attention.attend import Attend


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            dropout: float = 0.0,
            apply_rotary_emb: bool = False
    ):
        super().__init__()

        def get_self_attend() -> nn.Module:
            return Attend(
                query_dim=dim,
                context_dim=None,
                attn_heads=heads,
                attn_dim_head=dim_head,
                attn_dropout=dropout,
                hidden_dim=mlp_dim,
                ff_dropout=dropout,
                apply_rotary_emb=apply_rotary_emb,
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(get_self_attend())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attend in self.layers:
            x = attend(queries=x)

        return x
