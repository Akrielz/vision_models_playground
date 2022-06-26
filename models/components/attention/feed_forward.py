from typing import Callable

import torch
from torch import nn

from models.components.activations.geglu import GEGLU
from models.components.activations.relu_squared import ReluSquared


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            activation: Callable = None
    ):
        super(FeedForward, self).__init__()

        if activation is None:
            activation = ReluSquared()

        if isinstance(activation, GEGLU):
            assert hidden_dim % 2 == 0, "hidden_dim must be even when using GEGLU"

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
