from typing import Iterator, Tuple, Callable, Optional

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    This is an implementation of the Lion (Evolved Sign Momentum) optimizer. It is based on the paper:
    https://arxiv.org/abs/2302.06675
    """

    def __init__(
            self,
            params: Iterator[torch.Tensor],
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0
    ):
        """
        Initialize the hyperparameters.

        Arguments
        ---------
        params : Iterator[torch.Tensor]
            The parameters of the model.

        lr : float
            The learning rate.

        betas : Tuple[float, float]
            The beta parameters for the exponential moving average of the gradient.

        weight_decay : float
            The weight decay.
        """

        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Arguments
        ---------
        closure: Optional[Callable]:
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        The loss of the model.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
