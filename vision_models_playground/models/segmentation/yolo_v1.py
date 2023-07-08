from typing import List

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from vision_models_playground.components.convolutions.yolo_v1_head import YoloV1Head
from vision_models_playground.utility.config import config_wrapper


class YoloV1(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,

            dims: List[List[int]],
            kernel_size: List[List[int]],
            stride: List[List[int]],
            max_pools: List[bool],

            num_bounding_boxes: int = 2,
            grid_size: int = 7,
            negative_slope: float = 0.1,
            mlp_size: int = 4096
    ):
        super().__init__()

        # save the parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_bounding_boxes = num_bounding_boxes
        self.grid_size = grid_size
        self.negative_slope = negative_slope

        # save the layer parameters
        self.dims = dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pools = max_pools
        self.mlp_size = mlp_size

        self.layers = nn.ModuleList([
            self._build_layer(i) for i in range(len(dims))
        ])

        self.head = YoloV1Head(
            in_channels=self.dims[-1][-1],
            num_classes=self.num_classes,
            num_bounding_boxes=self.num_bounding_boxes,
            grid_size=self.grid_size,
            mlp_size=self.mlp_size,
            negative_slope=self.negative_slope,
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x

    def _get_init_dim_layer(self, layer_index: int):
        if layer_index == 0:
            return self.in_channels

        return self.dims[layer_index - 1][-1]

    def _build_layer(self, layer_index: int):
        layer_dim = self.dims[layer_index]
        layer_kernel_size = self.kernel_size[layer_index]
        layer_stride = self.stride[layer_index]

        layers = []
        for i in range(len(layer_dim)):
            dim_in = self._get_init_dim_layer(layer_index) if i == 0 else layer_dim[i - 1]
            dim_out = layer_dim[i]

            kernel_size = layer_kernel_size[i]
            padding = kernel_size // 2
            stride = layer_stride[i]

            layers.extend([
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.LeakyReLU(self.negative_slope)
            ])

        if self.max_pools[layer_index]:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)


class ResNetYoloV1(nn.Module):
    """
    This is a YOLOv1 model with a ResNet50 backbone. The main idea is to be able to use the pretrained weights
    from the ResNet50 model and then train the YOLOv1 head from scratch.
    """

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_bounding_boxes: int = 2,
            grid_size: int = 7,
            mlp_size: int = 1024,
            negative_slope: float = 0.1,
            internal_size: int = 1024,
    ):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)

        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        backbone_channels = 2048

        yolo_head = YoloV1Head(
            in_channels=backbone_channels,
            num_classes=num_classes,
            num_bounding_boxes=num_bounding_boxes,
            grid_size=grid_size,
            mlp_size=mlp_size,
            negative_slope=negative_slope,
            internal_size=internal_size
        )

        self.model = nn.Sequential(
            backbone,
            Rearrange('b (h w c) -> b c h w', h=14, w=14),  # TODO: Make this dynamic
            yolo_head
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


def build_yolo_v1(
        in_channels: int,
        num_classes: int,
        num_bounding_boxes: int = 2,
        grid_size: int = 7,
        mlp_size: int = 4096
):
    dims = [[64], [192], [128, 256, 256, 512], [256, 512, 256, 512, 256, 512, 256, 512, 512, 1024], [512, 1024, 512, 1024]]
    kernel_size = [[7], [3], [1, 3, 1, 3], [1, 3, 1, 3, 1, 3, 1, 3, 1, 3], [1, 3, 1, 3]]
    stride = [[2], [1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1]]
    max_pools = [True, True, True, True, False]

    YoloV1WithConfig = config_wrapper(YoloV1)

    return YoloV1WithConfig(
        dims=dims,
        kernel_size=kernel_size,
        stride=stride,
        max_pools=max_pools,

        in_channels=in_channels,
        num_classes=num_classes,
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,

        mlp_size=mlp_size
    )