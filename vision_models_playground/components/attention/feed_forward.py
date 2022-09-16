from typing import Callable

import torch
from torch import nn

from vision_models_playground.components.activations.geglu import GEGLU
from vision_models_playground.components.activations.relu_squared import ReluSquared


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            activation: Callable = None,
            output_dim: int = None
    ):
        super(FeedForward, self).__init__()

        if activation is None:
            activation = ReluSquared()

        activation_dim = hidden_dim
        if isinstance(activation, GEGLU):
            assert hidden_dim % 2 == 0, "hidden_dim must be even when using GEGLU"
            activation_dim = hidden_dim // 2

        if output_dim is None:
            output_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(activation_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
