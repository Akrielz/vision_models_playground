import torch
from torch import nn


class Swish(nn.Module):
    """
    Swish - Described in: https://arxiv.org/abs/1710.05941
    """

    def forward(self, x):
        return x * x.sigmoid()


def main():
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 100)
    y = Swish()(x)

    plt.plot(x.numpy(), y.numpy(), label="Swish")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()