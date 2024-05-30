from typing import Optional, Callable, List

import torch
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef

from vision_models_playground.evaluate import evaluate_model


def evaluate_model_classifier(
        model: nn.Module,
        test_dataset: torch.utils.data.Dataset,
        loss_fn: Optional[Callable] = None,
        batch_size: int = 64,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        **kwargs
):
    num_classes = len(test_dataset.classes)

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

    evaluate_model(
        model=model,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        batch_size=batch_size,
        metrics=metrics,
        **kwargs
    )
