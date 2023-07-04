from typing import Any

from torch import nn


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, default_val: Any) -> Any:
    return val if exists(val) else default_val


def get_number_of_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def format_number_of_parameters(number_of_parameters: int) -> str:
    return f'{number_of_parameters:,}'


def get_number_of_parameters_formatted(model: nn.Module) -> str:
    return format_number_of_parameters(get_number_of_parameters(model))