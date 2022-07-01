from typing import Union, List, Optional, Callable

from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from models.classifiers.perceiver import Perceiver
from models.components.position_embedding.fourrier_embedding import FourierEmbedding
from utility.datasets import get_mnist_dataset, get_cifar10_dataset
from utility.train_models import train_model


class VisionPerceiver(nn.Module):
    def __init__(
            self,
            *,
            image_size: int,
            patch_size: int,
            projection_dim: int,

            num_classes: int = 1000,

            apply_rotary_emb: bool = True,

            apply_fourier_encoding: bool = True,
            max_freq: int = 10,
            num_freq_bands: int = 6,
            constant_mapping: bool = False,
            max_position: Union[List[int], int] = 1600,

            num_layers: int = 2,

            num_latents: int = 128,
            latent_dim: int = 512,

            cross_num_heads: int = 8,
            cross_head_dim: int = 64,

            self_attend_heads: int = 8,
            self_attend_dim: int = 64,

            transformer_depth: int = 2,

            attention_dropout: float = 0.0,
            ff_hidden_dim: int = 512,
            ff_dropout: float = 0.0,
            activation: Optional[Callable] = None,
    ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        self.patch_height, self.patch_width = patch_height, patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        # Create fourier encoder
        self.pos_encoder = FourierEmbedding(max_freq, num_freq_bands, constant_mapping, max_position) \
            if apply_fourier_encoding else nn.Identity()

        self.to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.to_projection = nn.LazyLinear(projection_dim)

        # build perceiver
        self.perceiver = Perceiver(
            input_dim=projection_dim,
            input_axis=1,
            final_classifier_head=True,
            num_classes=num_classes,
            apply_rotary_emb=apply_rotary_emb,
            apply_fourier_encoding=False,
            num_layers=num_layers,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_num_heads=cross_num_heads,
            cross_head_dim=cross_head_dim,
            self_attend_heads=self_attend_heads,
            self_attend_dim=self_attend_dim,
            transformer_depth=transformer_depth,
            attention_dropout=attention_dropout,
            ff_hidden_dim=ff_hidden_dim,
            ff_dropout=ff_dropout,
            activation=activation,
        )

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.pos_encoder(x)
        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        x = self.to_projection(x)
        x = self.perceiver(x)

        return x


def main():
    model = VisionPerceiver(
        image_size=28,
        patch_size=4,
        projection_dim=128,
        num_classes=10,
        apply_rotary_emb=True,
        apply_fourier_encoding=True,
        max_freq=10,
        num_freq_bands=6,
        constant_mapping=False,
        max_position=1600,
        num_layers=2,
        num_latents=16,
        latent_dim=32,
        cross_num_heads=8,
        cross_head_dim=64,
        self_attend_heads=8,
        self_attend_dim=64,
        transformer_depth=2,
        attention_dropout=0.0,
        ff_hidden_dim=512,
        ff_dropout=0.0,
        activation=None,
    ).cuda()

    train_dataset, test_dataset = get_mnist_dataset()
    train_model(model, train_dataset, test_dataset, num_epochs=100)


if __name__ == "__main__":
    main()
