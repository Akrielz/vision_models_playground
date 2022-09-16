from typing import List

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from vision_models_playground.utility.datasets import get_mnist_dataset, to_autoencoder_dataset
from vision_models_playground.utility.train_models import train_model


class FeedForwardAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
    ):

        super(FeedForwardAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        net_dims = [input_dim] + hidden_dims

        encoder_layers = nn.ModuleList([])
        for i in range(len(net_dims) - 1):
            encoder_layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = nn.ModuleList([])
        for i in range(len(net_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(net_dims[i], net_dims[i - 1]))
            if i != 1:
                continue
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.float()
        initial_shape = x.shape
        x = rearrange(x, "b ... -> b (...)")
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(initial_shape)

        return x

    def decode(self, x):
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)


def main():
    model = FeedForwardAutoencoder(
        input_dim=28 * 28 * 1,
        hidden_dims=[256, 128],
    ).cuda()
    train_dataset, test_dataset = get_mnist_dataset()
    train_dataset, test_dataset = map(lambda d: to_autoencoder_dataset(d), [train_dataset, test_dataset])

    # use train_dataset to train the model
    train_model(model, train_dataset, test_dataset, num_epochs=100, loss_fn=F.mse_loss)


if __name__ == '__main__':
    main()