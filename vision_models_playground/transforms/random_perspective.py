from typing import Optional, Tuple, List

import torch
from torchvision.transforms import InterpolationMode, RandomPerspective
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def perspective_coords(
        coords: torch.Tensor,
        start_points: List[Tuple[int, int]],
        end_points: List[Tuple[int, int]],
):
    """
    Apply a perspective transformation to the given coords.

    Arguments
    ---------
    coords : torch.Tensor
        The coords to transform. The shape must be (num_coords, 2). [x, y]

    start_points : List[Tuple[int, int]]
        The starting points of the transformation. The shape must be (4, 2). [x, y]

    end_points : List[Tuple[int, int]]
        The ending points of the transformation. The shape must be (4, 2). [x, y]

    Returns
    -------
    torch.Tensor
        The transformed coords.
    """

    # Convert the coords to float
    coords = coords.to(torch.float32)

    # Get the coefficients
    coeffs = F._get_perspective_coeffs(end_points, start_points)
    a, b, c, d, e, f, g, h = coeffs

    # Apply the transformation
    coords[:, 0] = (a * coords[:, 0] + b * coords[:, 1] + c) / (g * coords[:, 0] + h * coords[:, 1] + 1)
    coords[:, 1] = (d * coords[:, 0] + e * coords[:, 1] + f) / (g * coords[:, 0] + h * coords[:, 1] + 1)

    return coords


class RandomPerspectiveWithCoords(TransformWithCoordsModule):
    """
    Random perspective transformation of the input image and coords.
    """

    def __init__(
            self,
            distortion_scale: float,
            p: float = 0.5,
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            fill: int = 0,
    ):
        """
        Arguments
        ---------

        distortion_scale : float
            The scale of the distortion.

        p : float
            The probability of applying the transform.

        interpolation : InterpolationMode
            The interpolation mode to use.

        fill : int
            The value to fill the area outside the transform in the image.


        """

        super().__init__()
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.fill = fill
        self.p = p

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if torch.rand(1) > self.p:
            return image, coords

        channels, height, width = image.shape[-3:]
        fill = [float(self.fill)] * channels

        # Get the start reference points
        start_points, end_points = RandomPerspective.get_params(width, height, self.distortion_scale)

        distorted_image = F.perspective(image, start_points, end_points, self.interpolation, fill)

        # Apply the transform to the coords
        distorted_coords = None
        if coords is not None:
            distorted_coords = perspective_coords(coords, start_points, end_points)

        return distorted_image, distorted_coords