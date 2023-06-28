from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def horizontal_flip_coords(
        coords: torch.Tensor,
        width: int
) -> torch.Tensor:
    """
    Flip the given coords horizontally.
    """

    # Flip the coords
    coords[..., 0] = width - coords[..., 0]

    return coords


class RandomHorizontalFlipWithCoords(TransformWithCoordsModule):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        width = image.shape[-1]

        if torch.rand(1) > self.p:
            return image, coords

        # Flip the image
        flipped_image = F.hflip(image)

        # Flip the coords
        flipped_coords = None
        if coords is not None:
            flipped_coords = horizontal_flip_coords(coords, width)

        return flipped_image, flipped_coords
