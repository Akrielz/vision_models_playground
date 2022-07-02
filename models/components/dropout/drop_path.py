import torch
from torch import nn


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """
        :param drop_prob: probability of dropping a path
        :param scale_by_keep: scale by the number of paths to drop
        """

        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """

        # Check if the dropout must happen
        if self.drop_prob == 0. or not self.training:
            return x

        # Get a tensor resembling the number of paths to drop
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        # Scale by the number of paths to drop
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob

        # Apply the dropout
        return x * random_tensor
