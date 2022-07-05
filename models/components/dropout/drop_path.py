import torch
from torch import nn


class DropPath(nn.Module):
    """
    Drop paths (entire batches) (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """
        :param drop_prob: probability of dropping a path
        :param scale_by_keep: scale by the number of paths to drop
        """

        assert 0.0 <= drop_prob <= 1.0, "drop_prob must be between 0.0 and 1.0"

        super(DropPath, self).__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """

        # Check if the dropout must happen
        if self.drop_prob == 0.0 or not self.training:
            return x

        # Get a tensor resembling the number of paths to drop
        keep_prob = 1.0 - self.drop_prob
        batch_size = x.shape[0]
        num_dim = x.dim()
        shape = [batch_size] + [1] * (num_dim - 1)  # create [B, 1, 1, ..., 1]
        random_tensor = torch.zeros(shape, dtype=x.dtype).to(x.device).bernoulli_(keep_prob)

        # Scale the not dropped paths by the average number of paths dropped
        if self.scale_by_keep:
            random_tensor = random_tensor / keep_prob

        # Apply the dropout
        return x * random_tensor


def main():
    drop_path = DropPath(drop_prob=0.5, scale_by_keep=True)

    x = torch.randn(4, 2, 3, 3)
    out = drop_path(x)


if __name__ == "__main__":
    main()
