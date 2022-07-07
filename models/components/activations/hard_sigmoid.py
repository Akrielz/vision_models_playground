import torch
from torch import nn
from torch.nn import functional as F


class HardSigmoid(nn.Module):
    def forward(self, x):
        return x * (x > 0).float() * (x < 1).float()

    def forward2(self, x):
        return F.relu6(x + 3) / 6


def main():
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 100)
    y = HardSigmoid().forward2(x)

    plt.plot(x.numpy(), y.numpy(), label="Swish")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()