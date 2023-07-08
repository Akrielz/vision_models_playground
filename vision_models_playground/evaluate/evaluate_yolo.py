from typing import Optional, Callable, List

import torch
import torchmetrics
from torch import nn
from torchmetrics import Accuracy, AveragePrecision, AUROC, Dice, F1Score

from vision_models_playground.datasets.datasets import get_voc_detection_dataset_yolo
from vision_models_playground.evaluate.evaluate_models import evaluate_model
from vision_models_playground.losses.yolo_v1_loss import YoloV1Loss
from vision_models_playground.metrics.wrapper import YoloV1ClassMetricWrapper, YoloV1MeanAveragePrecision
from vision_models_playground.utility.load_models import load_best_model


def evaluate_yolo_v1(
        model: nn.Module,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[Callable] = None,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        num_bounding_boxes: int = 2,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
):
    if test_dataset is None:
        test_dataset = get_voc_detection_dataset_yolo()[1]

    num_classes = len(test_dataset.classes)

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

    evaluate_model(
        model,
        test_dataset,
        loss_fn,
        batch_size,
        print_every_x_steps,
        metrics,
        device=device,
        num_workers=num_workers,
    )


def main():
    model = load_best_model("models/train/ResNetYoloV1/2023-07-06_14-37-23")
    evaluate_yolo_v1(model)


if __name__ == '__main__':
    main()