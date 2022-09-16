from typing import Callable

import torch
from torch import nn


class PostNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, **kwargs):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.norm(self.fn(x, **kwargs))
