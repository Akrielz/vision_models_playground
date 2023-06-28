from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class TransformWithCoordsModule(nn.Module, ABC):
    """
    Base class for transforms that take coords as input. The coords are optional.
    """

    @abstractmethod
    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
