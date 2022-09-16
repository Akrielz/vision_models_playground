from typing import Callable, Optional

import torch
from torch import nn

from vision_models_playground.utility.functions import exists


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, context_dim: Optional[int] = None):
        super().__init__()
        self.fn = fn
        self.norm_queries = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm_queries(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)
