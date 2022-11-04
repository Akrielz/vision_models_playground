import math
from typing import Literal

import torch
from torch import nn


class GELU(nn.Module):
    """
    GELU is a smooth approximation to the ReLU function.
    This can work with either the standard GELU formula or the GPT-2 GELU formula.
    """

    def __init__(self, formula_type: Literal["standard", "gpt"] = "standard"):
        super().__init__()
        self.formula_type = formula_type

    def forward(self, x: torch.Tensor):
        if self.formula_type == "standard":
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        elif self.formula_type == "gpt":
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        else:
            raise ValueError(f"Unknown formula type {self.formula_type}")