from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self):
        """
        Initialize the Discriminator
        """

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Discriminator forward pass
        :param x: The input image
        :return: The probability that the input is real
        """

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)

        # flatten the image using rearrange from einops
        x = rearrange(x, 'b ... -> b (...)')

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x