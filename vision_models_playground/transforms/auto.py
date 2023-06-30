from typing import Tuple, Optional

import numpy as np
import torch
from torchvision.transforms import ColorJitter, GaussianBlur, RandomPosterize, RandomAdjustSharpness, RandomAutocontrast, \
    RandomEqualize, RandomErasing

from vision_models_playground.transforms.base import TransformWithCoordsModule
from vision_models_playground.transforms.choose_one import ChooseOne
from vision_models_playground.transforms.clamp import ClampWithCoords
from vision_models_playground.transforms.compose import ComposeGeneral, ComposeRandomOrder
from vision_models_playground.transforms.conversions import StandardToUnitWrapper
from vision_models_playground.transforms.probability_transform import Prob
from vision_models_playground.transforms.random_affine import RandomAffineWithCoords
from vision_models_playground.transforms.random_horizontal_flip import RandomHorizontalFlipWithCoords
from vision_models_playground.transforms.random_perspective import RandomPerspectiveWithCoords
from vision_models_playground.transforms.random_resized_crop import RandomResizedCropWithCoords
from vision_models_playground.transforms.random_vertical_flip import RandomVerticalFlipWithCoords
from vision_models_playground.transforms.resize import ResizeWithCoords
from vision_models_playground.transforms.transform import WithCoords


class AutoTransformWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            size: Tuple[int, int],
            prob_photometric: float = 0.2,
            prob_geometric: float = 0.1,
    ):
        super().__init__()

        self.size = size
        self.prob_photometric = prob_photometric
        self.prob_geometric = prob_geometric

        self.photometric_transforms = AutoPhotometricWithCoords(p=prob_photometric)
        self.geometric_transforms = AutoGeometricWithCoords(p=prob_geometric, size=size)

        self.transform = ComposeGeneral([
            self.photometric_transforms,
            self.geometric_transforms,
        ])

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, coords = self.transform(image, coords)
        return image, coords


class AutoPhotometricWithCoords(TransformWithCoordsModule):
    def __init__(self, p: float = 0.1):
        super().__init__()

        self.p = p

        self.transform = ComposeRandomOrder([
            # Add color jitter
            Prob(WithCoords(ColorJitter(brightness=0.8)), p=p),
            Prob(WithCoords(ColorJitter(contrast=0.5)), p=p),
            Prob(WithCoords(ColorJitter(saturation=0.8)), p=p),
            Prob(WithCoords(ColorJitter(hue=0.2)), p=p),

            # Add gaussian blur
            ChooseOne([
                Prob(WithCoords(GaussianBlur(kernel_size=k)), p=p)
                for k in range(3, 15, 2)
            ]),

            # Add random posterize
            ChooseOne([
                Prob(WithCoords(StandardToUnitWrapper(RandomPosterize(bits=i))), p=p)
                for i in range(4, 8)
            ]),

            # Add Sharpness
            ChooseOne([
                Prob(WithCoords(RandomAdjustSharpness(sharpness_factor=i)), p=p)
                for i in range(1, 4)
            ]),

            # Add Contrast
            Prob(WithCoords(RandomAutocontrast()), p=p),

            # Add Equalize
            Prob(WithCoords(StandardToUnitWrapper(RandomEqualize())), p=p),
        ])

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, coords = self.transform(image, coords)
        return image, coords


class AutoGeometricWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            size: Tuple[int, int],
            p: float = 0.1
    ):
        super().__init__()

        self.p = p

        self.geometric_transforms = ComposeRandomOrder([
            # Add horizontal flip
            Prob(RandomHorizontalFlipWithCoords(), p=p),

            # Add vertical flip
            Prob(RandomVerticalFlipWithCoords(), p=p),

            # Add crop
            Prob(RandomResizedCropWithCoords(size=size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)), p=p),

            # Add affine
            Prob(RandomAffineWithCoords(degrees=(-30, 30)), p=p),
            Prob(RandomAffineWithCoords(translate=(0.0, 0.5)), p=p),
            Prob(RandomAffineWithCoords(scale=(0.7, 1.1)), p=p),
            Prob(RandomAffineWithCoords(shear=(-10, 10)), p=p),

            # Add perspective
            Prob(RandomPerspectiveWithCoords(distortion_scale=0.6), p=p),

            # Add erasing
            Prob(WithCoords(RandomErasing()), p=p),
        ])

        self.transform = ComposeGeneral([
            self.geometric_transforms,
            ResizeWithCoords(size=size),
            ClampWithCoords()
        ])

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image, coords = self.transform(image, coords)
        return image, coords
