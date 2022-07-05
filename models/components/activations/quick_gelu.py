import torch
from torch import nn


class QuickGELU(nn.Module):
    """
    This is a quick GELU implementation that is faster than the official
    implementation.

    This implementation is based on the following paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def main():
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 100)

    # Compute the GELU values
    y_gelu = torch.nn.GELU()(x)

    # Compute the QuickGELU values
    y_quick_gelu = QuickGELU()(x)

    # Plot the results
    plt.plot(x.numpy(), y_gelu.numpy(), label="GELU")
    plt.plot(x.numpy(), y_quick_gelu.numpy(), label="QuickGELU")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
