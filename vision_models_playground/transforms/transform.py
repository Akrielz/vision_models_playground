from typing import Optional, Tuple

import torch

from vision_models_playground.transforms.base import TransformWithCoordsModule


class WithCoords(TransformWithCoordsModule):
    """
    This is a generic module that allows to apply a transform to an image,
    considering that the coords will not be transformed.

    Such a module is useful when we want to write a pipeline of multiple
    transforms for an image with coords, but we want to apply an image
    transformation that does not change the coords (for example a normalization,
    a color jitter, etc.).
    """

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, coords = self.transform(image), coords
        return image, coords