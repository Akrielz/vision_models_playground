from typing import List

import torch
from einops import rearrange
from torch import nn


class UnNormalize(nn.Module):
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.inplace = inplace

        # Rearrange the mean and std to fall over channels
        self.mean = rearrange(self.mean, 'c -> c 1 1')
        self.std = rearrange(self.std, 'c -> c 1 1')

    def forward(self, image: torch.Tensor):

        if not self.inplace:
            image = image.clone()

        return self.std * image + self.mean


class Normalize(nn.Module):
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.inplace = inplace

        # Rearrange the mean and std to fall over channels
        self.mean = rearrange(self.mean, 'c -> c 1 1')
        self.std = rearrange(self.std, 'c -> c 1 1')

    def forward(self, image: torch.Tensor):

        if not self.inplace:
            image = image.clone()

        return (image - self.mean) / self.std

