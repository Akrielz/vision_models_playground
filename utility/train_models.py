from copy import deepcopy
from typing import Optional, Callable, List

import torch
import torchmetrics
from colorama import Fore
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from torchmetrics import Accuracy, AUROC, AveragePrecision, Dice, F1Score
from tqdm import tqdm


def train_model_classifier(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None
):
    num_classes = len(train_dataset.classes)

    # init loss function
    if loss_fn is None:
        loss_fn = CrossEntropyLoss()

    if metrics is None:
        metrics = [
            Accuracy(num_classes=num_classes).cuda(),
            AveragePrecision(num_classes=num_classes).cuda(),
            AUROC(num_classes=num_classes).cuda(),
            Dice(num_classes=num_classes).cuda(),
            F1Score(num_classes=num_classes).cuda()
        ]

    train_model(model, train_dataset, test_dataset, loss_fn, optimizer, num_epochs, batch_size, print_every_x_steps, metrics)


def train_model(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        loss_fn: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None
):
    # init optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # init loss function
    assert loss_fn is not None, "loss_fn must not be None"

    if metrics is None:
        metrics = []

    # get train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_metrics = metrics

    # get test loader
    test_loader = None
    test_metrics = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_metrics = deepcopy(metrics)

    # train model
    for epoch in range(num_epochs):
        __train_epoch(epoch, model, optimizer, print_every_x_steps, loss_fn, train_metrics, train_loader)

        if test_loader is not None:
            __test_epoch(epoch, model, print_every_x_steps, loss_fn, test_metrics, test_loader)


def __epoch(epoch, model, optimizer, print_every_x_steps, loss_fn, metrics, loader, color, mode):
    progress_bar = tqdm(loader)
    for i, (data, target) in enumerate(progress_bar):
        # Forward
        data, target = data.cuda(), target.cuda()
        output = model(data)

        # Compute loss
        loss = loss_fn(output, target)

        # Apply optimizer
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics
        for metric in metrics:
            metric.update(output, target)

        # Print progress
        if i % print_every_x_steps == 0:
            # prepare metric log
            metric_log = ''
            for metric in metrics:
                metric_log += f'{metric.__repr__()[:-2]}: {metric.compute():.4f} | '

            loss_name = "Loss" if len(loss_fn.__repr__()) > 30 else loss_fn.__repr__()[:-2]
            loss_log = f'{loss_name}: {loss.item():.4f} '

            description = color + f"{mode} Epoch: {epoch}, Step: {i} | {loss_log} | {metric_log}"
            progress_bar.set_description_str(description, refresh=False)


def __train_epoch(epoch, model, optimizer, print_every_x_steps, loss_fn, metrics, train_loader):
    __epoch(epoch, model, optimizer, print_every_x_steps, loss_fn, metrics, train_loader, Fore.CYAN, 'Train')


@torch.no_grad()
def __test_epoch(epoch, model, print_every_x_steps, loss_fn, metrics, test_loader):
    __epoch(epoch, model, None, print_every_x_steps, loss_fn, metrics, test_loader, Fore.YELLOW, 'Test')
