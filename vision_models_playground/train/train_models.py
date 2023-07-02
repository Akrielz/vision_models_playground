import os
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Literal, Union

import torch
import torchmetrics
from colorama import Fore
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision_models_playground.metrics.loss_tracker import LossTracker


def train_model(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        loss_fn: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        save_dir: Optional[str] = None,
        monitor_metric_name: str = 'loss',
        monitor_metric_mode: Literal['min', 'max'] = 'min',
        device: Optional[torch.device] = None,
):
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=batch_size,
        print_every_x_steps=print_every_x_steps,
        metrics=metrics,
        save_dir=save_dir,
        monitor_metric_name=monitor_metric_name,
        monitor_metric_mode=monitor_metric_mode,
        device=device,
    )

    trainer.train(num_epochs)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            loss_fn: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            batch_size: int = 64,
            print_every_x_steps: int = 1,
            metrics: Optional[List[torchmetrics.Metric]] = None,
            save_dir: Optional[str] = None,
            device: Optional[torch.device] = None,
            monitor_metric_name: str = 'loss',
            monitor_metric_mode: Literal['min', 'max'] = 'min',
    ):
        """
        Arguments
        ---------
        model: nn.Module
            The model to train

        train_dataset: torch.utils.data.Dataset
            The dataset to train on

        test_dataset: torch.utils.data.Dataset
            The dataset to test on

        loss_fn: nn.Module
            The loss function to use

        optimizer: Optional[torch.optim.Optimizer]
            The optimizer to use. If None, use Adam with lr=0.001

        batch_size: int
            The batch size to use

        print_every_x_steps: int
            Print the metrics every x steps

        metrics: Optional[List[torchmetrics.Metric]]
            The metrics to use

        save_dir: Optional[str]
            The directory to save the model in

        device: Optional[torch.device]
            The device to use. If None, use cuda if available

        monitor_metric_name: str
            The name of the metric to monitor

        monitor_metric_mode: Literal['min', 'max']
            The mode of the metric to monitor. Either 'min' or 'max'
        """

        # Init optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Init metrics
        if metrics is None:
            metrics = []

        # Get train loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_metrics = metrics

        # Get test loader
        test_loader = None
        test_metrics = None
        if test_dataset is not None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_metrics = deepcopy(metrics)

        # If save_dir is None, save in current directory
        if save_dir is None:
            save_dir = '.'

        current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save_dir = f'{save_dir}/models/{model.__class__.__name__}/{current_date}'
        save_dir = os.path.normpath(save_dir)

        # Create save_dir if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # If device is None, use cuda if available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Save the data
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.print_every_x_steps = print_every_x_steps
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.train_metrics = train_metrics
        self.test_loader = test_loader
        self.test_metrics = test_metrics
        self.save_dir = save_dir
        self.device = device
        self.monitor_metric_name = monitor_metric_name
        self.monitor_metric_mode = monitor_metric_mode
        self.monitor_value = None

        # Add summary writer
        self.writer = SummaryWriter(log_dir=save_dir)

        # Add a loss tracker
        self.train_loss_tracker = LossTracker(loss_fn)
        self.test_loss_tracker = LossTracker(loss_fn)

        # Move to device all the necessary things
        self.__move_to_device()

        # Set up the monitor metric
        self._monitor_metric = self.__setup_monitor_metric()
        self._monitor_comparator = lambda x, y: x < y if self.monitor_metric_mode == 'min' else x > y

    def __setup_monitor_metric(self):
        if self.monitor_metric_name == 'loss':
            return self.test_loss_tracker

        for metric in self.test_metrics:
            if metric.__repr__()[:-2] == self.monitor_metric_name:
                return metric

        raise ValueError(f'Could not find metric with name {self.monitor_metric_name}')

    def __move_to_device(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for metric in self.train_metrics:
            metric.to(self.device)

        for metric in self.test_metrics:
            metric.to(self.device)

        self.train_loss_tracker.to(self.device)
        self.test_loss_tracker.to(self.device)

    def train(self, num_epochs: int):
        # train model
        for epoch in range(num_epochs):
            self.__train_epoch(epoch)

            if self.test_loader is None:
                continue

            self.__test_epoch(epoch)

    def __epoch(
            self,
            epoch: int,
            metrics: List[torchmetrics.Metric],
            loader: torch.utils.data.DataLoader,
            color: Union[int, str],
            phase: Literal['Train', 'Test'],
            loss_tracker: LossTracker,
    ):
        self.__init_phase(metrics, phase)

        progress_bar = tqdm(loader)
        for i, (data, target) in enumerate(progress_bar):
            # Forward
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            # Compute loss
            loss = self.loss_fn(output, target)

            # Apply optimizer
            if phase == 'Train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics
            for metric in metrics:
                metric.update(output, target)

            # Update the loss tracker
            loss_tracker.update(loss.detach().cpu().item(), num_samples=data.shape[0])

            # Print progress
            if i % self.print_every_x_steps == 0:
                self.__update_description(epoch, i, metrics, loss, color, phase, progress_bar, loss_tracker)

        # Save model
        if phase == 'Test':
            self.__save_model()

    def __init_phase(self, metrics: List[torchmetrics.Metric], phase: Literal['Train', 'Test']):
        # Set the model to train or eval mode
        if phase == 'Train':
            self.model.train()
        else:
            self.model.eval()

        # Reset all metrics for test phase to make sure the final computed value is correct
        if phase == 'Test':
            for metric in metrics:
                metric.reset()

    def __prepare_metric_log(
            self,
            metrics: List[torchmetrics.Metric],
            phase: Literal['Train', 'Test'],
            step: int,
    ) -> str:

        metric_log = ''

        for metric in metrics:
            metric_computed = metric.compute()

            if isinstance(metric_computed, dict):
                for key, value in metric_computed.items():
                    metric_name = f'{metric.__repr__()[:-2]}_{key}'
                    metric_log += f'{metric_name}: {value:.4f} | '

                    self.writer.add_scalar(f'{phase}/{metric_name}', value, step)

            else:
                metric_name = f'{metric.__repr__()[:-2]}'
                metric_log += f'{metric_name}: {metric_computed:.4f} | '

                self.writer.add_scalar(f'{phase}/{metric_name}', metric_computed, step)

        return metric_log

    def __prepare_loss_log(
            self,
            loss: torch.Tensor,
            loss_tracker: LossTracker,
            phase: Literal['Train', 'Test'],
            step,
    ) -> str:

        loss_item = loss.detach().cpu().item()

        # Process the loss
        loss_name = "Loss" if len(self.loss_fn.__repr__()) > 30 else self.loss_fn.__repr__()[:-2]
        loss_log = f'{loss_name}: {loss_item:.4f} | '
        self.writer.add_scalar(f'{phase}/{loss_name}', loss_item, step)

        # Process the tracked loss
        loss_tracked = loss_tracker.compute()
        loss_name = "LossTracked"
        loss_log += f'{loss_name}: {loss_tracked:.4f}'
        self.writer.add_scalar(f'{phase}/{loss_name}', loss_tracked, step)

        return loss_log

    def __update_description(
            self,
            epoch: int,
            i: int,
            metrics: List[torchmetrics.Metric],
            loss: torch.Tensor,
            color: Union[int, str],
            phase: Literal['Train', 'Test'],
            progress_bar: tqdm,
            loss_tracker: LossTracker,
    ):
        step = epoch * len(progress_bar) + i

        # Prepare metric log
        metric_log = self.__prepare_metric_log(metrics, phase, step)
        loss_log = self.__prepare_loss_log(loss, loss_tracker, phase, step)

        description = color + f"{phase} Epoch: {epoch}, Step: {i} | {loss_log} | {metric_log}"
        progress_bar.set_description_str(description, refresh=False)

    def __save_model(self):
        # Save last checkpoint
        torch.save(self.model.state_dict(), f'{self.save_dir}/last.pt')

        # Save best checkpoint based on monitor_metric
        current_metric = self._monitor_metric.compute()
        if self.monitor_value is None or self._monitor_comparator(current_metric, self.monitor_value):
            self.monitor_value = current_metric
            torch.save(self.model.state_dict(), f'{self.save_dir}/best.pt')

    def __train_epoch(self, epoch: int):
        self.__epoch(epoch, self.train_metrics, self.train_loader, Fore.GREEN, 'Train', self.train_loss_tracker)

    @torch.no_grad()
    def __test_epoch(self, epoch: int):
        self.__epoch(epoch, self.test_metrics, self.test_loader, Fore.YELLOW, 'Test', self.test_loss_tracker)
