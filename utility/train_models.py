from typing import Optional

import torch
from colorama import Fore
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from tqdm import tqdm


def train_model(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
):
    # get train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_acc = Accuracy().cuda()

    # get test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_acc = Accuracy().cuda()

    # init optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    for epoch in range(num_epochs):
        __train_epoch(epoch, model, optimizer, print_every_x_steps, train_acc, train_loader)

        if test_loader is not None:
            __test_epoch(epoch, model, print_every_x_steps, test_acc, test_loader)


def __train_epoch(epoch, model, optimizer, print_every_x_steps, train_acc, train_loader):
    progress_bar = tqdm(train_loader)
    for i, (data, target) in enumerate(progress_bar):
        # Forward
        data, target = data.cuda(), target.cuda()
        output = model(data)

        # Compute loss
        loss = F.cross_entropy(output, target)

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        train_acc.update(output, target)

        # Print progress
        if i % print_every_x_steps == 0:
            description = Fore.CYAN + f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc.compute():.4f}"
            progress_bar.set_description_str(description, refresh=False)


@torch.no_grad()
def __test_epoch(epoch, model, print_every_x_steps, test_acc, test_loader):
    progress_bar = tqdm(test_loader)
    for i, (data, target) in enumerate(progress_bar):
        # Forward
        data, target = data.cuda(), target.cuda()
        output = model(data)

        # Compute loss metric
        loss = F.cross_entropy(output, target)

        # Update metrics
        test_acc.update(output, target)

        # Print progress
        if i % print_every_x_steps == 0:
            description = Fore.YELLOW + f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc.compute():.4f}"
            progress_bar.set_description_str(description, refresh=False)
