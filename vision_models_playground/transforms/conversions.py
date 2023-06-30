import torch
from torch import nn


def to_float(image: torch.Tensor) -> torch.Tensor:
    return image.float()


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    return image.byte()


def from_standard_to_unit(image: torch.Tensor) -> torch.Tensor:
    return to_float(image / 255.0)


def from_unit_to_standard(image: torch.Tensor) -> torch.Tensor:
    return to_uint8(image * 255.0)


def from_general_to_unit(image: torch.Tensor) -> torch.Tensor:
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)


def from_general_to_standard(image: torch.Tensor) -> torch.Tensor:
    unit = from_general_to_unit(image)
    return from_unit_to_standard(unit)


class ToFloat(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return to_float(image)


class ToUint8(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return to_uint8(image)


class StandardToUnit(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return from_standard_to_unit(image)


class UnitToStandard(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return from_unit_to_standard(image)


class GeneralToUnit(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return from_general_to_unit(image)


class GeneralToStandard(nn.Module):
    @staticmethod
    def forward(image: torch.Tensor):
        return from_general_to_standard(image)


class StandardToUnitWrapper(nn.Module):
    """
    Wrapper for transforms that work on the standard format (0-255) and return
    a tensor in the unit format (0-1).

    Example: RandomPosterize is a transform that works on the standard format
    """

    def __init__(self, transform: nn.Module):
        super().__init__()
        self.transform = transform

    def forward(self, image: torch.Tensor):
        image = from_unit_to_standard(image)
        image = self.transform(image)
        return from_standard_to_unit(image)
