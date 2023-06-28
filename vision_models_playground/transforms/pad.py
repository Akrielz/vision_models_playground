from typing import Union, Tuple, Optional

import torch

from vision_models_playground.transforms.base import TransformWithCoordsModule


def pad_coords(
        coords: torch.Tensor,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]
):
    if isinstance(padding, int):  # side
        padding = (padding, padding, padding, padding)

    if len(padding) == 2:  # (left, right) and (top, bottom)
        padding = (padding[0], padding[1], padding[0], padding[1])

    coords = coords + torch.tensor([padding[0], padding[2]])
    return coords


class PadWithCoords(TransformWithCoordsModule):
    def __init__(
            self,
            padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
            fill: int = 0,
            padding_mode: str = 'constant'
    ):
        """
        Pad the given image on all sides with the given "pad" value.

        Arguments
        ---------
        padding:
            Padding on each border.
            If a single int is provided this is used to pad all borders.
            If a sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.

        fill:
            Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.

        padding_mode:
            Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        """

        super().__init__()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(
            self,
            image: torch.Tensor,
            coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pad the given image on all sides with the given "pad" value.

        Arguments
        ---------
        image:
            Image to be padded.

        coords:
            Coordinates of the objects in the image.

        Returns
        -------
        Tuple of padded image and padded coordinates.
        """

        # Pad the image
        padded_image = torch.nn.functional.pad(
            input=image,
            pad=self.padding,
            mode=self.padding_mode,
            value=self.fill
        )

        # Pad the coords
        padded_coords = None
        if coords is not None:
            padded_coords = pad_coords(coords, self.padding)

        return padded_image, padded_coords