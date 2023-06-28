from typing import Tuple, Optional

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def rotate_coords(
        coords: torch.Tensor,
        angle: float,
        original_size: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Rotate the given coords by angle degrees and return the rotated coords.

    Arguments
    ---------

    coords : torch.Tensor
        The coords to rotate. The shape must be (num_coords, 2). [x, y]

    angle : float
        The angle in degrees to rotate the coords.

    original_size : Tuple[int, int]
        The original size of the image.

    center : Optional[Tuple[int, int]]
        The center of rotation. If None, the center of the image is used.
    """

    # Instantiate the center
    original_height, original_width = original_size

    if center is None:
        center = [original_width / 2, original_height / 2]

    print(center)

    center = torch.tensor(center, dtype=torch.float32)

    # Convert the coords to float
    coords = coords.to(torch.float32)

    # Convert the angle to radians
    angle = np.deg2rad(angle)

    # Compute the rotation matrix
    rotation_matrix = torch.tensor([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ], dtype=torch.float32)

    # Compute the new coords
    translated_coords = coords - center
    rotated_coords = torch.einsum('i j, n j -> n i', rotation_matrix, translated_coords)
    rotated_coords = rotated_coords + center

    return rotated_coords


class RandomRotationWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            degrees: Tuple[float, float],
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            center: Optional[Tuple[int, int]] = None,
            fill: float = 0.0,
    ):
        super().__init__()
        self.degrees = [float(degrees[0]), float(degrees[1])]
        self.interpolation = interpolation
        self.center = center
        self.fill = fill
        self.expand = False

    def _sample_angle(self):
        # sample angle uniformly from the given range using numpy
        angle = np.random.uniform(self.degrees[0], self.degrees[1], size=(1,)).item()
        return angle

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        channels, height, width = image.shape[-3:]
        original_size = (height, width)

        fill = [float(self.fill)] * channels
        angle = self._sample_angle()

        # Rotate the image
        rotated_image = F.rotate(image, angle, self.interpolation, self.expand, self.center, fill)

        # Rotate the coords
        rotated_coords = None
        if coords is not None:
            rotated_coords = rotate_coords(coords, angle, original_size, self.center)

        return rotated_image, rotated_coords
