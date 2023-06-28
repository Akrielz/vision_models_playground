from typing import Optional, Tuple

import torch

from vision_models_playground.transforms.base import TransformWithCoordsModule


class ComposeWithCoords(TransformWithCoordsModule):
    def __init__(self, transforms: list[TransformWithCoordsModule]):
        super().__init__()
        self.transforms = transforms

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for transform in self.transforms:
            image, coords = transform(image, coords)

        return image, coords

