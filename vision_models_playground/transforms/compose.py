from typing import Optional, Tuple, List

import torch
from torch import nn


class ComposeGeneral(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for transform in self.transforms:
            args = transform(*args)

        return args


class ComposeRandomOrder(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # get a permutation of the transforms
        transforms_len = len(self.transforms)
        permutation = torch.randperm(transforms_len)

        for i in permutation:
            transform = self.transforms[i]
            args = transform(*args)

        return args