from typing import Optional, Callable, List

import torch
import torchmetrics
from torch import nn
from torchmetrics import Accuracy, AveragePrecision, AUROC, Dice, F1Score

from vision_models_playground.datasets.datasets import get_voc_detection_dataset_yolo, get_voc_detection_dataset_yolo_aug
from vision_models_playground.losses.yolo_v1_loss import YoloV1Loss
from vision_models_playground.metrics.wrapper import YoloV1ClassMetricWrapper, YoloV1MeanAveragePrecision
from vision_models_playground.models.segmentation.yolo_v1 import ResNetYoloV1
from vision_models_playground.train.train_models import train_model
from vision_models_playground.utility.config import config_wrapper


def train_yolo_v1(
        model: nn.Module,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        num_bounding_boxes: int = 2,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
):
    if train_dataset is None:
        train_dataset = get_voc_detection_dataset_yolo()[0]

    if valid_dataset is None:
        valid_dataset = get_voc_detection_dataset_yolo()[1]

    num_classes = len(train_dataset.classes)

    # init loss function
    if loss_fn is None:
        loss_fn = YoloV1Loss(num_classes=num_classes, num_bounding_boxes=num_bounding_boxes)

    classification_metrics_kwargs = {
        'num_classes': num_classes,
        'task': 'multiclass'
    }

    if metrics is None:
        classification_metrics = [
            Accuracy(**classification_metrics_kwargs).cuda(),
            AveragePrecision(**classification_metrics_kwargs).cuda(),
            AUROC(**classification_metrics_kwargs).cuda(),
            Dice(**classification_metrics_kwargs).cuda(),
            F1Score(**classification_metrics_kwargs).cuda()
        ]

        metrics: List[torchmetrics.Metric] = [
            YoloV1ClassMetricWrapper(metric, num_bounding_boxes=num_bounding_boxes, num_classes=num_classes)
            for metric in classification_metrics
        ]

    train_model(
        model,
        train_dataset,
        valid_dataset,
        loss_fn,
        optimizer,
        num_epochs,
        batch_size,
        print_every_x_steps,
        metrics,
        device=device,
        num_workers=num_workers
    )


def main():
    in_channels = 3
    num_bounding_boxes = 2
    grid_size = 7

    num_epochs = 130
    batch_size = 16

    train_dataset = get_voc_detection_dataset_yolo_aug(
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,
        download=False
    )[0]
    valid_dataset = get_voc_detection_dataset_yolo(
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,
        download=False
    )[1]

    num_classes = len(train_dataset.classes)

    ResNetYoloV1WithConfig = config_wrapper(ResNetYoloV1)
    model = ResNetYoloV1WithConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        num_bounding_boxes=num_bounding_boxes,
        grid_size=grid_size,
        mlp_size=1024,
        internal_size=1024
    )

    train_yolo_v1(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_bounding_boxes=num_bounding_boxes,
    )


if __name__ == '__main__':
    main()