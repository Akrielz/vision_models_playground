from typing import Optional

import torch
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, einsum

from vision_models_playground.utility.functions import default, exists


class Attention(nn.Module):
    def __init__(
            self,
            query_dim: int,
            context_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            apply_rotary_emb: bool = False
    ):
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.rotary_emb = None
        if apply_rotary_emb:
            assert dim_head % 2 == 0, "Rotary embedding cannot be applied on odd dimensions"
            self.rotary_emb = RotaryEmbedding(dim=dim_head // 2)

    def forward(
            self,
            queries: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get useful info
        h = self.heads
        context = default(context, queries)

        # Get query, key and value
        q = self.to_q(queries)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape query, key and value according to heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), [q, k, v])

        # Apply rotary embedding if necessary
        if exists(self.rotary_emb):
            q, k = map(lambda t: self.rotary_emb.rotate_queries_or_keys(t), [q, k])

        # Apply attention
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Mask attention if necessary
        if exists(mask):
            mask = repeat(mask, "b ... -> b h ...", h=h)

            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention sim matrix
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project to output channels
        return self.to_out(out)
