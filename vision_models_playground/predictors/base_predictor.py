from copy import deepcopy
from typing import Callable

from torch import nn


class Predictor:
    def __init__(
            self, model: nn.Module,
            collate_in: Callable = None,
            collate_out: Callable = None
    ):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self._collate_in = collate_in
        self._collate_out = collate_out

    def predict(self, x):
        initial_x = deepcopy(x)

        if self._collate_in is not None:
            x = self._collate_in(x)

        y = self.model(x)

        if self._collate_out is not None:
            y = self._collate_out(initial_x, y)

        return y

    def __call__(self, x):
        return self.predict(x)