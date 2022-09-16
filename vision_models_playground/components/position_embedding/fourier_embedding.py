from math import pi
from typing import List, Union

import torch
from einops import rearrange, repeat
from torch import nn


def _fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class FourierEmbedding(nn.Module):
    """
    Fourier Embedding will append the fourier positional encoding to the given
    input.

    The fourier positional encoding is calculated on sines and cosines of the
    original input.

    @Params:
    num_freq_bands: int
        - determines how many sines and cosines to append

    max_freq: int
        - determines the frequency domain for sines and cosines

    constant_mapping: bool = False
        - determines if the mapping is constant or dynamic

    max_position: Union[List[int], int] = 1000
        - in case of constant mapping, this will be the constant for the linear space
        - it can be a list, having a maximum boundary for each dimension, excepting batch and last dim
        - it can be an int, and it will apply the same maximum for all dimensions

    For a given torch tensor with shape [... c]
    it will return a torch tensor of shape [... f]
    where f = c + input_axis * ((num_freq_bands * 2) + 1)
    """

    def __init__(
            self,
            max_freq: int,
            num_freq_bands: int,
            constant_mapping: bool = False,
            max_position: Union[List[int], int] = 1600
    ):
        """Initialize the class."""
        super(FourierEmbedding, self).__init__()
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.constant_mapping = constant_mapping
        self.max_position = max_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input to append num_freq_bands sines and cosines to it.
        """

        # Take important information about input
        b, *axis, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Map the positions to a linear space in the interval [-1, 1], agnostic
        # of batch_size
        if self.constant_mapping:

            # Transform max_position into list
            if not isinstance(self.max_position, list):
                max_position = [self.max_position for _ in axis]
            else:
                assert len(self.max_position) == len(axis), \
                    f"Wrong number of dimensions were given to max_position. \n" \
                    f"Expected {len(axis)}, got {len(self.max_position)}."
                max_position = self.max_position

            # Make sure we have the right amount of space
            assert all(size <= max_pos for size, max_pos in zip(axis, max_position)), \
                f"More positions are necessary than what is configured.\n" \
                f"{axis} > {self.max_position}"

            axis_pos = [
                torch.linspace(-1.0, 1.0, steps=max_pos, device=device, dtype=dtype)[:size]
                for size, max_pos in zip(axis, max_position)
            ]
        else:
            axis_pos = list(map(
                lambda size: torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype),
                axis
            ))

        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)

        # Apply the fourier encoding to calculate the sines and cosines
        enc_pos = _fourier_encode(pos, self.max_freq, self.num_freq_bands)

        # Reshape the content to be exactly same shape as input
        # (excepting last channel)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')

        # Repeat it for each element inside the batch
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        # Concatenate the encoded positionals with the original channels
        return torch.cat([x, enc_pos], dim=-1)
