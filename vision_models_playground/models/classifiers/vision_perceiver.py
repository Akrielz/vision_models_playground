from typing import Union, List, Optional, Callable

from torch import nn

from einops import rearrange

from vision_models_playground.components.position_embedding import FourierEmbedding
from vision_models_playground.models.classifiers.perceiver import Perceiver
from vision_models_playground.datasets.datasets import get_cifar10_dataset
from vision_models_playground.train.train_classifier import train_model_classifier


class VisionPerceiver(nn.Module):
    def __init__(
            self,
            *,
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
        patch_height, patch_width = patch_size, patch_size

        self.patch_height, self.patch_width = patch_height, patch_width

        # Create fourier encoder
        self.pos_encoder = FourierEmbedding(max_freq, num_freq_bands, constant_mapping, max_position) \
            if apply_fourier_encoding else nn.Identity()

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
        patch_size=4,
        projection_dim=1024,
        num_classes=10,
        apply_rotary_emb=True,
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
    )

    train_dataset, test_dataset = get_cifar10_dataset()
    train_model_classifier(model, train_dataset, test_dataset, batch_size=128)


if __name__ == "__main__":
    main()
