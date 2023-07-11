import os
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Literal, Union

import torch
import torchmetrics
from colorama import Fore
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision_models_playground.metrics.loss_tracker import LossTracker


def evaluate_model(
        model: nn.Module,
        test_dataset: torch.utils.data.Dataset,
        loss_fn: nn.Module,
        batch_size: int = 64,
        print_every_x_steps: int = 1,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        save_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
):
    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        batch_size=batch_size,
        print_every_x_steps=print_every_x_steps,
        metrics=metrics,
        save_dir=save_dir,
        device=device,
        num_workers=num_workers
    )

    evaluator.evaluate()


class Evaluator:
    def __init__(
            self,
            model: nn.Module,
            test_dataset: torch.utils.data.Dataset,
            loss_fn: nn.Module,
            batch_size: int = 64,
            print_every_x_steps: int = 1,
            metrics: Optional[List[torchmetrics.Metric]] = None,
            save_dir: Optional[str] = None,
            device: Optional[torch.device] = None,
            num_workers: Optional[int] = None,
    ):
        """
        Arguments
        ---------
        model: nn.Module
            The model to train

        test_dataset: torch.utils.data.Dataset
            The dataset to test on

        loss_fn: nn.Module
            The loss function to use

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
        """

        # Init metrics
        if metrics is None:
            metrics = []

        # Get half of the available cores
        if num_workers is None:
            num_workers = os.cpu_count() // 2

        # Get test loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_metrics = deepcopy(metrics)

        # If save_dir is None, save in current directory
        if save_dir is None:
            save_dir = '.'

        current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save_dir = f'{save_dir}/models/eval/{model.__class__.__name__}/{current_date}'
        save_dir = os.path.normpath(save_dir)

        # Create save_dir if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # If device is None, use cuda if available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Save the data
        self.model = model
        self.batch_size = batch_size
        self.print_every_x_steps = print_every_x_steps
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.test_metrics = test_metrics
        self.save_dir = save_dir
        self.device = device
        self.monitor_value = None

        self.test_dataset_name = test_dataset.__class__.__name__
        self.test_dataset_module = test_dataset.__class__.__module__

        # Add summary writer
        self.writer = SummaryWriter(log_dir=save_dir)

        self.test_metrics.append(LossTracker(loss_fn))

        # Move to device all the necessary things
        self.__move_to_device()

    def __move_to_device(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for metric in self.test_metrics:
            metric.to(self.device)

    @torch.no_grad()
    def __epoch(
            self,
            metrics: List[torchmetrics.Metric],
            loader: DataLoader,
            color: Union[int, str],
    ):
        self.model.eval()

        progress_bar = tqdm(loader)
        for i, (data, target) in enumerate(progress_bar):
            # Forward
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            # Compute loss
            loss = self.loss_fn(output, target)

            # Update metrics
            for metric in metrics:
                metric.update(output, target)

            # Print progress
            if i % self.print_every_x_steps == 0:
                self.__update_description(i, metrics, loss, color, 'Test', progress_bar)

    def __prepare_metric_log(
            self,
            metrics: List[torchmetrics.Metric],
            phase: Literal['Test'],
            step: int,
            delimiter: str = ' | ',
            apply_writer: bool = True,
    ) -> str:

        metric_log = ''

        for metric in metrics:
            metric_computed = metric.compute()

            if isinstance(metric_computed, dict):
                for key, value in metric_computed.items():
                    metric_name = f'{metric.__repr__()[:-2]}_{key}'
                    metric_log += f'{metric_name}: {value:.4f}{delimiter}'

                    if not apply_writer:
                        continue

                    self.writer.add_scalar(f'{phase}/{metric_name}', value, step)

            else:
                metric_name = f'{metric.__repr__()[:-2]}'
                metric_log += f'{metric_name}: {metric_computed:.4f}{delimiter}'

                if not apply_writer:
                    continue

                self.writer.add_scalar(f'{phase}/{metric_name}', metric_computed, step)

        return metric_log

    def __prepare_loss_log(
            self,
            loss: torch.Tensor,
            phase: Literal['Test'],
            step,
    ) -> str:

        loss_item = loss.detach().cpu().item()

        # Process the loss
        loss_name = "Loss" if len(self.loss_fn.__repr__()[:-2]) > 30 else self.loss_fn.__repr__()[:-2]
        loss_log = f'{loss_name}: {loss_item:.4f}'
        self.writer.add_scalar(f'{phase}/{loss_name}', loss_item, step)

        return loss_log

    def __update_description(
            self,
            i: int,
            metrics: List[torchmetrics.Metric],
            loss: torch.Tensor,
            color: Union[int, str],
            phase: Literal['Train', 'Test'],
            progress_bar: tqdm,
    ):
        step = i

        # Prepare metric log
        metric_log = self.__prepare_metric_log(metrics, phase, step)
        loss_log = self.__prepare_loss_log(loss, phase, step)

        phase_padded = phase
        if phase_padded == 'Test':
            phase_padded = f'{phase_padded} '

        description = color + f"{phase_padded} Step: {i} | {loss_log} | {metric_log}"
        progress_bar.set_description_str(description, refresh=False)

    def create_report(self):
        header = '## **Metrics**'

        dataset_info = f'The model was evaluated on the following dataset: **{self.test_dataset_name}** from ```{self.test_dataset_module}```'

        metric_log = self.__prepare_metric_log(self.test_metrics, 'Test', 0, delimiter='  \n', apply_writer=False)
        metric_log_formatted = ''.join([f"\n- {line}" for line in metric_log.split('\n') if len(line) > 0])
        metrics_info = "These are the results of the evaluation:  \n\n" + metric_log_formatted

        full_log = f"{header}\n\n{dataset_info}\n\n{metrics_info}"

        # Write in file
        with open(f'{self.save_dir}/report.md', 'w') as f:
            f.write(full_log)

    @torch.no_grad()
    def __test_epoch(self):
        self.__epoch(self.test_metrics, self.test_loader, Fore.YELLOW)

    def evaluate(self):
        self.__test_epoch()
        self.writer.close()
        self.create_report()
