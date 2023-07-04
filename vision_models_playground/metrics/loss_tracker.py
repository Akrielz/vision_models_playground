import torch
from torch import nn
from torchmetrics import Metric


class LossTracker(Metric):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()

        self.loss_fn = loss_fn

        reduction = loss_fn.reduction
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', \
            f'Expected reduction to be "mean", "sum", but got {reduction}'

        # Add the reduction to the state_dict, so that it can be restored correctly
        self.reduction = reduction
        self.add_state('loss', torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # Calculate the loss
        with torch.no_grad():
            loss = self.loss_fn(pred, target)
            loss = loss.item()

            num_samples = pred.shape[0]

            if self.reduction == 'mean':
                loss *= num_samples

        # Update the state
        self.loss += loss
        self.num_samples += num_samples

    def compute(self):
        return self.loss / self.num_samples

    def reset(self):
        self.loss = torch.tensor(0.0)
        self.num_samples = torch.tensor(0)

    def __repr__(self):
        return 'LossTracker()'
