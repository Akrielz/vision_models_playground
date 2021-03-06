from typing import List

import torch
from einops import rearrange
from torch import nn

from utility.datasets import get_mnist_dataset, get_cifar10_dataset
from utility.train_models import train_model, train_model_classifier


class FeedForwardClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int = 28 * 28,
            hidden_dims: List[int] = None,
            num_classes: int = 10,
    ):
        super(FeedForwardClassifier, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        net_dims = [input_dim] + hidden_dims + [num_classes]

        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            if i != len(net_dims) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = rearrange(x, "b ... -> b (...)")
        return self.net(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def predict_prob(self, x):
        with torch.no_grad():
            return self.forward(x)


def main():
    model = FeedForwardClassifier(
        input_dim=32 * 32 * 3,
        hidden_dims=[256, 128],
        num_classes=10,
    ).cuda()
    train_dataset, test_dataset = get_cifar10_dataset()
    train_model_classifier(model, train_dataset, test_dataset, num_epochs=100)


if __name__ == '__main__':
    main()
