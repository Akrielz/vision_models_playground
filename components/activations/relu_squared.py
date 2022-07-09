import torch
from torch import nn
import torch.nn.functional as F


class ReluSquared(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2
