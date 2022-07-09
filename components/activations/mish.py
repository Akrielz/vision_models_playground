import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    """
    Swish - Described in: https://arxiv.org/abs/1908.08681
    """
    def forward(self, x):
        return x * F.softplus(x).tanh()


def main():
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 100)
    y = Mish()(x)

    plt.plot(x.numpy(), y.numpy(), label="Swish")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()