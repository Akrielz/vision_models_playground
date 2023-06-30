from typing import Tuple, Optional

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, RandomRotation
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def rotate_coords(
        coords: torch.Tensor,
        angle: float,
        original_size: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None,
        expand: bool = False,
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

    expand : bool
        Whether to expand the image to fit the whole rotated image. This means
        the coordinates will be translated to fit the expanded image.
    """

    # Instantiate the center
    original_height, original_width = original_size

    if center is None:
        center = [original_width / 2, original_height / 2]

    center = torch.tensor(center, dtype=torch.float32)

    # Convert the coords to float
    coords = coords.to(torch.float32)

    if expand:
        # Add the four corners in the coords
        corners = torch.tensor(
            [[0, 0], [0, original_height - 1], [original_width - 1, 0], [original_width - 1, original_height - 1]], dtype=torch.float32
        )
        coords = torch.cat([coords, corners], dim=0)

    # Convert the angle to radians
    angle = np.deg2rad(angle)

    # Compute the rotation matrix
    # This is the clockwise rotation matrix, because that's applied to the image too
    rotation_matrix = torch.tensor([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ], dtype=torch.float32)

    # Compute the new coords
    translated_coords = coords - center
    rotated_coords = torch.einsum('i j, n j -> n i', rotation_matrix, translated_coords)
    rotated_coords = rotated_coords + center

    if expand:
        # remove the four corners in the coords
        corners = rotated_coords[-4:]
        rotated_coords = rotated_coords[:-4]

        # compute how much to shift the coords
        min_x = torch.min(corners[:, 0])
        min_y = torch.min(corners[:, 1])

        # translate the coords to fit the expanded image
        rotated_coords[:, 0] = rotated_coords[:, 0] - min_x
        rotated_coords[:, 1] = rotated_coords[:, 1] - min_y

    return rotated_coords


class RandomRotationWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            degrees: Tuple[float, float],
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            expand: bool = False,
            center: Optional[Tuple[int, int]] = None,
            fill: float = 0.0,
    ):
        super().__init__()
        self.degrees = [float(degrees[0]), float(degrees[1])]
        self.interpolation = interpolation
        self.center = center
        self.fill = fill
        self.expand = expand

    def _sample_angle(self):
        # sample angle uniformly from the given range using numpy
        angle = RandomRotation.get_params(self.degrees)
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
            rotated_coords = rotate_coords(coords, angle, original_size, self.center, self.expand)

        return rotated_image, rotated_coords
