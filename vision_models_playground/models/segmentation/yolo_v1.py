from typing import List

import torch
from torch import nn

from vision_models_playground.components.convolutions.yolo_v1_head import YoloV1Head
from vision_models_playground.datasets.datasets import get_voc_detection_dataset_yolo_aug, get_voc_detection_dataset_yolo
from vision_models_playground.train.train_yolo import train_yolo_v1


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
            hidden_size: int = 4096
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
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList([
            self._build_layer(i) for i in range(len(dims))
        ])

        self.head = YoloV1Head(
            in_channels=self.dims[-1][-1],
            num_classes=self.num_classes,
            num_bounding_boxes=self.num_bounding_boxes,
            grid_size=self.grid_size,
            hidden_size=self.hidden_size,
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


def build_yolo_v1(
        in_channels: int,
        num_classes: int,
        num_bounding_boxes: int = 2,
        grid_size: int = 7,
        hidden_size: int = 4096
):
    dims = [[64], [192], [128, 256, 256, 512], [256, 512, 256, 512, 256, 512, 256, 512, 512, 1024], [512, 1024, 512, 1024]]
    kernel_size = [[7], [3], [1, 3, 1, 3], [1, 3, 1, 3, 1, 3, 1, 3, 1, 3], [1, 3, 1, 3]]
    stride = [[2], [1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1]]
    max_pools = [True, True, True, True, False]

    return YoloV1(
        dims=dims,
        kernel_size=kernel_size,
        stride=stride,
        max_pools=max_pools,

        in_channels=in_channels,
        num_classes=num_classes,
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,

        hidden_size=hidden_size
    )


def main():
    in_channels = 3
    num_bounding_boxes = 2
    grid_size = 7

    num_epochs = 130
    batch_size = 1

    train_dataset = get_voc_detection_dataset_yolo_aug(
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size
    )[0]
    test_dataset = get_voc_detection_dataset_yolo(
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size
    )[1]

    num_classes = len(train_dataset.classes)

    model = build_yolo_v1(
        in_channels=in_channels,
        num_classes=num_classes,
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,
        hidden_size=1024
    )

    train_yolo_v1(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_bounding_boxes=num_bounding_boxes,
    )


if __name__ == '__main__':
    main()