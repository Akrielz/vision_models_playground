from typing import Optional, Tuple, List, Union

import torch
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as F

from vision_models_playground.transforms.base import TransformWithCoordsModule


def resized_crop(
        coords: torch.Tensor,
        i: int,
        j: int,
        h: int,
        w: int,
        target_size: Union[int, Tuple[int, int], List[int]],
) -> torch.Tensor:
    """
    Crop the given coords to the given size.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Convert the coords to float
    coords = coords.to(torch.float32)
    target_height, target_width = target_size

    # Compute the scale
    scale = (target_width / w, target_height / h)

    # Scale the coords
    coords = coords - torch.tensor([j, i])
    coords = coords * torch.tensor(scale)

    return coords


class RandomResizedCropWithCoords(TransformWithCoordsModule):
    """
    This is a wrapper around torchvision.transforms.RandomResizedCrop
    that also applies the same transformation to the coords.
    """

    def __init__(
            self,
            size: Tuple[int, int],
            scale: Tuple[float, float] = (0.08, 1.0),
            ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            antialias: bool = True
    ):
        super().__init__()
        self.size = list(size)
        self.scale = list(scale)
        self.ratio = list(ratio)
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def get_area_to_crop(
            image: torch.Tensor,
            scale: List[float],
            ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        return RandomResizedCrop.get_params(image, scale, ratio)

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        left, top, height, width = self.get_area_to_crop(image, self.scale, self.ratio)

        # Apply the transform to the image
        cropped_image = F.resized_crop(image, left, top, height, width, self.size, self.interpolation, antialias=self.antialias)

        # Apply the same transform to the coords
        cropped_coords = None
        if coords is not None:
            cropped_coords = resized_crop(coords, left, top, height, width, self.size)

        return cropped_image, cropped_coords
