from typing import Tuple, Optional, Union

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def resize_coords(
        coords: torch.Tensor,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize the given coords to the given size.
    """
    # Convert the coords to float
    coords = coords.to(torch.float32)

    original_height, original_width = original_size
    target_height, target_width = target_size

    # Compute the scale
    scale = (target_width / original_width, target_height / original_height)

    # Scale the coords
    coords = coords * torch.tensor(scale)

    return coords


class ResizeWithCoords(TransformWithCoordsModule):
    """
    Resize the input image or coords to the given size.
    """

    def __init__(
            self,
            size: Tuple[int, int],
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            antialias: Optional[Union[str, bool]] = True
    ):
        """
        Arguments
        ---------

        size : Tuple[int, int]
            The target size. (height, width)

        interpolation : InterpolationMode
            The interpolation mode to use.

        antialias : Optional[Union[str, bool]]
            Whether to use antialiasing. If "warn", it will warn if antialiasing is not available.
        """

        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, image: torch.Tensor, coords: Optional[torch.Tensor] = None):
        """
        Resize the input image or coords to the given size.

        Arguments
        ---------
        image : torch.Tensor
            The image to resize. Shape: [num_channels, height, width]

        coords : Optional[torch.Tensor]
            The coords to resize. Shape: [num_coords, 2]  # (x, y)

        Returns
        -------
        (image_resized, coords_resized) : Tuple[torch.Tensor, torch.Tensor]
            The resized image and coords.
        """

        original_size = image.shape[-2:]

        # Resize the image
        image_resized = F.resize(image, list(self.size), self.interpolation, None, self.antialias)

        # Resize the coords
        coords_resized = None
        if coords is not None:
            coords_resized = resize_coords(coords, original_size, self.size)

        return image_resized, coords_resized