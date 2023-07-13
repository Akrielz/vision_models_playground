from copy import deepcopy
from typing import Callable, Optional, Any, List

import torch
from torch import nn


class Pipeline:
    def __init__(
            self,
            model: nn.Module,
            collate_in: Callable = None,
            collate_out: Callable = None,
            *,
            device: Optional[torch.device] = None
    ):
        if device is None:
            device = torch.device('cpu')

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self._collate_in = collate_in
        self._collate_out = collate_out

        self.device = device
        self.to(device)

    def to(self, device):
        self.model.to(device)
        self.device = device

        return self

    def predict(self, x: Any):
        initial_x = deepcopy(x)

        if self._collate_in is not None:
            x = self._collate_in(x)

        y = self.model(x)

        if self._collate_out is not None:
            y = self._collate_out(initial_x, y)

        return y

    def __call__(self, x: Any):
        return self.predict(x)

    def input_type(self):
        if self._collate_in is None:
            return Any

        annotations = list(self._collate_in.__annotations__.keys())
        input = [annotation for annotation in annotations if annotation != 'return'][0]
        type = self._collate_in.__annotations__[input]

        return type

    def output_type(self):
        if self._collate_out is None:
            return Any

        return self._collate_out.__annotations__['return']