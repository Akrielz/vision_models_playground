from typing import Optional, Callable, List

import torch
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, AveragePrecision, AUROC, Dice, F1Score, MatthewsCorrCoef

from vision_models_playground.train.train_models import train_model


def train_model_classifier(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        loss_fn: Optional[Callable] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        **kwargs
):
    num_classes = len(train_dataset.classes)

    # init loss function
    if loss_fn is None:
        loss_fn = CrossEntropyLoss()

    metrics_kwargs = {
        'num_classes': num_classes,
        'task': 'multiclass'
    }

    if metrics is None:
        metrics = [
            Accuracy(**metrics_kwargs),
            F1Score(**metrics_kwargs),
            MatthewsCorrCoef(**metrics_kwargs),
        ]

    train_model(
        model,
        train_dataset,
        valid_dataset,
        loss_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        metrics=metrics,
        device=None,
        monitor_metric_name='MulticlassAccuracy',
        monitor_metric_mode='max',
        **kwargs
    )
