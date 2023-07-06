from einops.layers.torch import Rearrange
from torch import nn


class YoloV1Head(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_bounding_boxes: int = 2,
            grid_size: int = 7,
            mlp_size: int = 4096,
            negative_slope: float = 0.1,
            internal_size: int = 1024
    ):
        """
        This is the head of the YoloV1 model according to this:
        https://arxiv.org/pdf/1506.02640.pdf
        It takes the output of the backbone and outputs the bounding boxes for each grid cell.

        The paper encodes the predictions as follows:
        S x S x (B * 5 + C)
        where S is the grid size, B is the number of bounding boxes and C is the number of classes.

        Arguments
        ---------
        in_channels : int
            The number of input channels of the backbone.

        num_classes : int
            The number of classes to predict.

        num_bounding_boxes : int
            The number of bounding boxes to predict per grid cell.

        grid_size : int
            The size of the grid.

        mlp_size : int
            The size of the internal layers of the MLP.

        negative_slope : float
            The negative slope of the leaky relu activation function.

        internal_size : int
            The size of the internal layers.
        """

        super().__init__()

        self.output_channels = grid_size * grid_size * (num_bounding_boxes * 5 + num_classes)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, internal_size, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope),

            nn.Conv2d(internal_size, internal_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope),

            nn.Conv2d(internal_size, internal_size, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope),

            nn.Conv2d(internal_size, internal_size, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope)
        )

        self.predicting_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(internal_size * grid_size * grid_size, mlp_size),
            nn.LeakyReLU(negative_slope),
            nn.Linear(mlp_size, self.output_channels),
            Rearrange('b (h w c) -> b h w c', w=grid_size, h=grid_size)
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.predicting_head(out)

        return out
