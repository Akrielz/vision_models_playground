from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def vertical_flip_coords(
        coords: torch.Tensor,
        height: int
) -> torch.Tensor:
    """
    Flip the given coords vertically.
    """

    # Flip the coords
    coords[..., 1] = height - coords[..., 1]

    return coords


class RandomVerticalFlipWithCoords(TransformWithCoordsModule):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        height = image.shape[-2]

        if torch.rand(1) > self.p:
            return image, coords

        # Flip the image
        flipped_image = F.vflip(image)

        flipped_coords = None
        if coords is not None:
            flipped_coords = vertical_flip_coords(coords, height)

        return flipped_image, flipped_coords
