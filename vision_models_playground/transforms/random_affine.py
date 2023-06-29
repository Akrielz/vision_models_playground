import numbers
from typing import Tuple, Optional, List

import torch
from torchvision.transforms import InterpolationMode, RandomAffine
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def _get_inverse_affine_matrix(
        angle: float,
        translate: List[int],
        scale: float,
        shear: List[float],
        original_size: Tuple[int, int],  # height, width
        center: Optional[List[int]] = None,
):
    """
    Affine matrix is : M = T * C * RotateScaleShear * C^-1

    where T is translation matrix:
          [[1, 0, tx],
           [0, 1, ty],
           [0, 0, 1 ]]
    C is translation matrix to keep center:
          [[1, 0, cx]
           [0, 1, cy]
           [0, 0, 1 ]]
    RotateScaleShear is rotation with scale and shear matrix

    RotateScaleShear(a, s, (sx, sy)) =
          = R(a) * S(s) * SHy(sy) * SHx(sx)
          = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
            [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
            [ 0                    , 0                                        , 1 ]
    where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
             [0, 1      ]              [-tan(s), 1]
    """

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    center_f = [0.0, 0.0]
    if center is not None:
        height, width = original_size
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = F._get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear, inverted=False)

    theta = torch.tensor(matrix, dtype=torch.float).reshape(2, 3)
    last_row = torch.tensor([[0, 0, 1]], dtype=torch.float)

    theta = torch.cat([theta, last_row], dim=0)
    return theta


def affine_coords(
        coords: torch.Tensor,
        angle: float,
        translations: Tuple[int, int],
        scale: float,
        shear: Tuple[float, float],
        original_size: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:

    # Extract the height and width
    original_height, original_width = original_size

    affine_matrix = _get_inverse_affine_matrix(
        angle,
        list(translations),
        scale,
        list(shear),
        original_size,
        center,
    )  # shape [3, 3]

    # Cast the coords to float
    coords = coords.to(torch.float32)

    # Add a column of ones to the coords
    ones = torch.ones(coords.shape[:-1] + (1,), dtype=torch.float32, device=coords.device)
    coords = torch.cat([coords, ones], dim=-1)

    # Instantiate the center
    if center is None:
        center = [original_width / 2, original_height / 2, 0]

    center = torch.tensor(center, dtype=torch.float32)

    # Apply the affine matrix
    coords = coords - center  # This step is relevant, because the affine matrix treats the coords as relative to the center
    coords = torch.matmul(coords, affine_matrix.t())
    coords = coords + center

    # Remove the last column
    coords = coords[..., :-1]

    return coords


class RandomAffineWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            degrees: Tuple[float, float] = (0, 0),
            translate: Optional[Tuple[float, float]] = None,
            scale: Optional[Tuple[float, float]] = None,
            shear: Tuple[float, float] = (0, 0),
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: int = 0,
            center: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        channels, height, width = image.shape[-3:]
        original_size = (height, width)

        fill = [float(self.fill)] * channels

        original_size_flipped = [width, height]
        angle, translations, scale, shear = RandomAffine.get_params(list(self.degrees), self.translate, self.scale, list(self.shear), original_size_flipped)
        ret = [angle, translations, scale, shear]

        affined_image = F.affine(image, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

        affined_coords = None
        if coords is not None:
            affined_coords = affine_coords(coords, angle, translations, scale, shear, original_size, self.center)

        return affined_image, affined_coords
