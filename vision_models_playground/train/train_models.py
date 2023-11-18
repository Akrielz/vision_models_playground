import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Literal, Union, Dict

import torch
import torchmetrics
from colorama import Fore
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision_models_playground.data_structures.tensor_structure_operations import anything_to_device
from vision_models_playground.metrics.loss_tracker import LossTracker


def train_model(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
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
        num_workers: Optional[int] = None,
):
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=batch_size,
        print_every_x_steps=print_every_x_steps,
        metrics=metrics,
        save_dir=save_dir,
        monitor_metric_name=monitor_metric_name,
        monitor_metric_mode=monitor_metric_mode,
        device=device,
        num_workers=num_workers
    )

    trainer.train(num_epochs)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: torch.utils.data.Dataset,
            valid_dataset: torch.utils.data.Dataset,
            loss_fn: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            batch_size: int = 64,
            print_every_x_steps: int = 1,
            metrics: Optional[List[torchmetrics.Metric]] = None,
            save_dir: Optional[str] = None,
            device: Optional[torch.device] = None,
            monitor_metric_name: str = 'loss',
            monitor_metric_mode: Literal['min', 'max'] = 'min',
            num_workers: Optional[int] = None,
            lr_scheduler_params: Optional[Dict] = None,
    ):
        self.__setup(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=batch_size,
            print_every_x_steps=print_every_x_steps,
            metrics=metrics,
            save_dir=save_dir,
            device=device,
            monitor_metric_name=monitor_metric_name,
            monitor_metric_mode=monitor_metric_mode,
            num_workers=num_workers,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.__move_to_device()

    def __setup(
            self,
            model: nn.Module,
            train_dataset: torch.utils.data.Dataset,
            valid_dataset: torch.utils.data.Dataset,
            loss_fn: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            batch_size: int = 64,
            print_every_x_steps: int = 1,
            metrics: Optional[List[torchmetrics.Metric]] = None,
            save_dir: Optional[str] = None,
            device: Optional[torch.device] = None,
            monitor_metric_name: str = 'loss',
            monitor_metric_mode: Literal['min', 'max'] = 'min',
            num_workers: Optional[int] = None,
            lr_scheduler_params: Optional[Dict] = None,
    ):
        """
        Arguments
        ---------
        model: nn.Module
            The model to train

        train_dataset: torch.utils.data.Dataset
            The dataset to train on

        valid_dataset: torch.utils.data.Dataset
            The dataset to validate on

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

        num_workers: Optional[int]
            The number of workers to use for the dataloader. If None, use half of the available cores

        lr_scheduler_params: Optional[dict]
            The parameters to pass to the lr scheduler. Those should be the params of the ReduceLROnPlateau scheduler
        """

        # Init lr scheduler params
        if lr_scheduler_params is None:
            lr_scheduler_params = {
                'factor': 0.33,
                'patience': 10,
                'threshold': 1e-4,
                'cooldown': 0,
                'min_lr': 1e-7,
            }

        # Init optimizer
        if optimizer is None:
            optimizer = Adam(model.parameters(), lr=1e-4)

        # Init metrics
        if metrics is None:
            metrics = []

        # Get half of the available cores
        if num_workers is None:
            num_workers = os.cpu_count() // 2

        # Get collate_fn
        self.collate_fn = getattr(model, 'collate_fn', None)

        # Get train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
        train_metrics = metrics

        # Get valid loader
        valid_loader = None
        valid_metrics = []
        if valid_dataset is not None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fn
            )
            valid_metrics = deepcopy(metrics)

        # If save_dir is None, save in current directory
        if save_dir is None:
            save_dir = '.'

        current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save_dir = f'{save_dir}/models/train/{model.__class__.__name__}/{current_date}'
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
        self.valid_loader = valid_loader
        self.valid_metrics = valid_metrics
        self.save_dir = save_dir
        self.device = device
        self.monitor_metric_name = monitor_metric_name
        self.monitor_metric_mode = monitor_metric_mode
        self.monitor_value = None
        self.num_workers = num_workers

        # Add summary writer
        self.writer = SummaryWriter(log_dir=save_dir)

        # Add a loss tracker
        self.train_metrics.append(LossTracker(loss_fn))
        self.valid_metrics.append(LossTracker(loss_fn))

        # Set up the monitor metric
        self._monitor_metric = self._setup_monitor_metric()
        self._monitor_comparator = lambda x, y: x < y if self.monitor_metric_mode == 'min' else x > y

        # Set up the lr scheduler
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode=monitor_metric_mode, **lr_scheduler_params)

        # Save the config if it exists
        config_path = f'{save_dir}/config.json'
        config = getattr(model, '_config', None)
        if config is not None:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

    def _setup_monitor_metric(self):
        if self.monitor_metric_name == 'loss':
            return deepcopy(self.valid_metrics[-1])

        for metric in self.valid_metrics:
            if metric.__repr__()[:-2] == self.monitor_metric_name:
                return deepcopy(metric)

        raise ValueError(f'Could not find metric with name {self.monitor_metric_name}')

    def __move_to_device(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for metric in self.train_metrics:
            metric.to(self.device)

        for metric in self.valid_metrics:
            metric.to(self.device)

        self._monitor_metric.to(self.device)

    def train(self, num_epochs: int):
        # train model
        for epoch in range(num_epochs):
            self._train_epoch(epoch)

            if self.valid_loader is None:
                continue

            self._valid_epoch(epoch)

    def _epoch(
            self,
            epoch: int,
            metrics: List[torchmetrics.Metric],
            loader: DataLoader,
            color: Union[int, str],
            phase: Literal['Train', 'Valid'],
    ):
        self._init_phase(phase)

        progress_bar = tqdm(loader)
        for i, (data, target) in enumerate(progress_bar):
            # Forward
            data = anything_to_device(data, self.device)
            target = anything_to_device(target, self.device)
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

            # Update the monitor metric
            if phase == 'Valid':
                self._monitor_metric.update(output, target)

            # Print progress
            if i % self.print_every_x_steps == 0:
                self._update_description(epoch, i, metrics, color, phase, progress_bar)

    def _init_phase(self, phase: Literal['Train', 'valid']):
        # Set the model to train or eval mode
        if phase == 'Train':
            self.model.train()
        else:
            self.model.eval()

        # Reset the monitor metrics, to make sure at the end of the epoch we have the correct value
        if phase == 'Valid':
            self._monitor_metric.reset()

    def _prepare_metric_log(
            self,
            metrics: List[torchmetrics.Metric],
            phase: Literal['Train', 'Valid'],
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

    def _prepare_loss_log(
            self,
            loss: torch.Tensor,
            phase: Literal['Train', 'valid'],
            step,
    ) -> str:

        loss_item = loss.detach().cpu().item()

        # Process the loss
        loss_name = "Loss" if len(self.loss_fn.__repr__()[:-2]) > 30 else self.loss_fn.__repr__()[:-2]
        loss_log = f'{loss_name}: {loss_item:.4f}'
        self.writer.add_scalar(f'{phase}/{loss_name}', loss_item, step)

        return loss_log

    def _update_description(
            self,
            epoch: int,
            i: int,
            metrics: List[torchmetrics.Metric],
            color: Union[int, str],
            phase: Literal['Train', 'Valid'],
            progress_bar: tqdm,
    ):
        step = epoch * len(progress_bar) + i

        # Prepare metric log
        metric_log = self._prepare_metric_log(metrics, phase, step)

        # Pad the step
        step_str = "Step: {:>7}".format(i)

        description = color + f"{phase} Epoch: {epoch}, {step_str} | {metric_log}"
        progress_bar.set_description_str(description, refresh=False)

    def _save_state(self):
        current_state = self.model.state_dict()

        # Save last checkpoint
        torch.save(current_state, f'{self.save_dir}/last.pt')

        # Save best checkpoint based on monitor_metric
        current_metric = self._monitor_metric.compute()
        if self.monitor_value is None or self._monitor_comparator(current_metric, self.monitor_value):
            self.monitor_value = current_metric
            torch.save(current_state, f'{self.save_dir}/best.pt')

        # Update the lr scheduler
        self.lr_scheduler.step(current_metric)

    def _train_epoch(self, epoch: int):
        self._epoch(epoch, self.train_metrics, self.train_loader, Fore.GREEN, 'Train')

    @torch.no_grad()
    def _valid_epoch(self, epoch: int):
        self._epoch(epoch, self.valid_metrics, self.valid_loader, Fore.YELLOW, 'Valid')
        self._save_state()