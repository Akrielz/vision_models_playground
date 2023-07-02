import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from vision_models_playground.datasets.datasets import get_cifar10_dataset
from vision_models_playground.train.train_classifier import train_model_classifier


class ConvolutionalClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            channels: int = 1
    ):
        # create a sequential model that has convolutions and batch normalization
        super(ConvolutionalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # create batch normalizations
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # create two linear components
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, num_classes)

        # create a dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # pass the input through the convolutional components
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        # pass the input through the linear layer
        x = rearrange(x, "b ... -> b (...)")
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def predict_prob(self, x):
        with torch.no_grad():
            return self.forward(x)


def main():
    model = ConvolutionalClassifier(channels=3)
    train_dataset, test_dataset = get_cifar10_dataset()
    train_model_classifier(model, train_dataset, test_dataset, num_epochs=100)


if __name__ == '__main__':
    main()
