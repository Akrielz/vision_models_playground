from typing import Optional, Callable, List

import torch
import torchmetrics
from torch import nn
from torchmetrics import Accuracy, AveragePrecision, AUROC, Dice, F1Score

from vision_models_playground.datasets.datasets import get_voc_detection_dataset_yolo
from vision_models_playground.losses.yolo_v1_loss import YoloV1Loss
from vision_models_playground.metrics.wrapper import YoloV1ClassMetricWrapper
from vision_models_playground.train.train_models import train_model


def train_yolo_v1(
        model: nn.Module,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        num_bounding_boxes: int = 2,
):
    if train_dataset is None:
        train_dataset = get_voc_detection_dataset_yolo()[0]

    if test_dataset is None:
        test_dataset = get_voc_detection_dataset_yolo()[1]

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

        metrics = [
            YoloV1ClassMetricWrapper(metric, num_bounding_boxes=num_bounding_boxes, num_classes=num_classes).cuda()
            for metric in classification_metrics
        ]

    train_model(model, train_dataset, test_dataset, loss_fn, optimizer, num_epochs, batch_size, print_every_x_steps, metrics)
