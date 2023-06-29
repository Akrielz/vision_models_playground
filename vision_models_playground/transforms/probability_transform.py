import torch
from torch import nn


class ProbTransform(nn.Module):
    def __init__(self, transform, prob):
        super().__init__()
        self.transform = transform
        self.prob = prob

        # We want to make sure that the transform is always applied
        if hasattr(self.transform, "p"):
            self.transform.p = 1.0

    def forward(self, *args, **kwargs):
        if torch.rand(1) < self.prob:
            return self.transform(*args, **kwargs)

        return args
