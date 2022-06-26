import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim=100, channels=1):
        """
        Initializes the Generator.
        :param noise_dim: the dimension of the noise vector
        :param channels: the number of channels in the image
        """

        super(Generator, self).__init__()

        self.noise_dim = noise_dim

        self.tconv1 = nn.ConvTranspose2d(noise_dim, 512, 4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.tconv4 = nn.ConvTranspose2d(128, channels, 4, stride=2, padding=3)

    def forward(self, x):
        """
        Generates an image from a noise vector.
        :param x: the noise vector
        :return: image
        """

        x = x.view(-1, 100, 1, 1)
        x = F.relu(self.tconv1(x))
        x = self.bn1(x)
        x = F.relu(self.tconv2(x))
        x = self.bn2(x)
        x = F.relu(self.tconv3(x))
        x = self.bn3(x)
        x = F.relu(self.tconv4(x))
        x = torch.tanh(x)

        return x
