from typing import Optional, Tuple

import torch

from vision_models_playground.transforms.base import TransformWithCoordsModule


def clamp_coords(
        coords: torch.Tensor,
        original_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Clamp the given coords to the original size.
    """

    height, width = original_size

    # Clamp the coords
    coords[..., 0] = torch.clamp(coords[..., 0], min=0, max=width-1)
    coords[..., 1] = torch.clamp(coords[..., 1], min=0, max=height-1)

    return coords


class ClampWithCoords(TransformWithCoordsModule):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        original_size = image.shape[-2:]

        clamped_coords = None
        if coords is not None:
            clamped_coords = clamp_coords(coords, original_size)

        return image, clamped_coords
