from typing import List, Tuple, Optional

import torch
from torch import nn


class ChooseOne(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # choose only one transform
        index = torch.randint(len(self.transforms), (1,))
        transform = self.transforms[index]

        args = transform(*args)

        return args