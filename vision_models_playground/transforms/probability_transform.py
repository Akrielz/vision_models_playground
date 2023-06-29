import torch
from torch import nn


class Prob(nn.Module):
    def __init__(self, transform: nn.Module, p: float = 0.5):
        super().__init__()
        self.transform = transform
        self.p = p

        # We want to make sure that the transform is always applied
        if hasattr(self.transform, "p"):
            self.transform.p = 1.0

    def forward(self, *args, **kwargs):
        if torch.rand(1) < self.p:
            return self.transform(*args, **kwargs)

        return args
