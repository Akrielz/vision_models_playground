import torch
from torch import nn
import torch.nn.functional as F


class ReluSquared(nn.Module):
    def __init__(self):
        super(ReluSquared, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2
