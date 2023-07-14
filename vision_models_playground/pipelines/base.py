from copy import deepcopy
from typing import Callable, Optional, Any, List

import torch
from torch import nn


class Pipeline:
    def __init__(
            self,
            model: Optional[nn.Module] = None,
            collate_in: Callable = None,
            collate_out: Callable = None,
            *,
            device: Optional[torch.device] = None
    ):
        if device is None:
            device = torch.device('cpu')

        self.set_model(model)

        self._collate_in = collate_in
        self._collate_out = collate_out

        self.device = device
        self.to(device)

    def to(self, device):
        if self.model is not None:
            self.model = self.model.to(device)

        self.device = device

        return self

    def set_model(self, model: Optional[nn.Module]):
        self.model = model

        if model is None:
            return self

        self.model.eval()
        self.model.requires_grad_(False)

        return self

    def predict(self, x: Any):
        if self.model is None:
            raise ValueError('No model set, please set a model using the self.set_model(model).')

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