from typing import Union, List, Callable, Optional

import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
from torch import nn

from vision_models_playground.components.activations.geglu import GEGLU
from vision_models_playground.components.attention import TransformerEncoder, FeedForward
from vision_models_playground.components.attention.attend import Attend
from vision_models_playground.components.position_embedding import FourierEmbedding
from vision_models_playground.datasets.datasets import get_cifar10_dataset
from vision_models_playground.train.train_classifier import train_model_classifier


class Perceiver(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_axis: int,

            final_classifier_head: bool = True,
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
        """
        Initialize the Perceiver
        """

        super().__init__()

        # Default activation is GEGLU
        if activation is None:
            activation = GEGLU()

        # Create latent space
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.latent_dim = latent_dim

        # Create fourier encoder
        self.pos_encoder = FourierEmbedding(max_freq, num_freq_bands, constant_mapping, max_position) \
            if apply_fourier_encoding else nn.Identity()

        # Compute the pos_encoder output dim
        self.pos_encoder_dim = input_dim + input_axis * ((num_freq_bands * 2) + 1) if apply_fourier_encoding else input_dim

        def create_cross_attend():
            return Attend(
                query_dim=latent_dim,
                context_dim=self.pos_encoder_dim,
                num_heads=cross_num_heads,
                head_dim=cross_head_dim,
                attention_dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb,
                ff_hidden_dim=ff_hidden_dim,
                ff_dropout=ff_dropout,
                activation=activation
            )

        def create_latent_transformer():
            return TransformerEncoder(
                dim=latent_dim,
                depth=transformer_depth,
                heads=self_attend_heads,
                head_dim=self_attend_dim,
                mlp_dim=ff_hidden_dim,
                mlp_dropout=ff_dropout,
                attention_dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb
            )

        # Build components
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            layer = nn.ModuleList([])

            layer.append(create_cross_attend())
            layer.append(create_latent_transformer())

            self.layers.append(layer)

        # Create output layer
        self.classifier = nn.Sequential(
            Reduce("b l d -> b d", "mean"),
            nn.LayerNorm(latent_dim),
            FeedForward(
                dim=latent_dim,
                hidden_dim=ff_hidden_dim,
                dropout=0.0,
                activation=activation,
                output_dim=num_classes
            )
        ) if final_classifier_head else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
            latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Take input about the shape of the input
        batch_size = x.shape[0]

        # Initialize Latents
        latents = self.latents if latents is None else latents
        assert latents.shape[-1] == self.latent_dim

        # Repeat latents for each batch
        if len(latents.shape) == 2:
            latents = repeat(latents, "l d -> b l d", b=batch_size)

        # Apply position encoding
        x = self.pos_encoder(x)

        # Flatten the input
        x = rearrange(x, "b ... d -> b (...) d")

        for cross_attend, transformer in self.layers:
            latents = cross_attend(latents, x, mask)
            latents = transformer(latents)

        output = self.classifier(latents)
        return output


def main():
    model = Perceiver(
        input_dim=1,
        input_axis=2,
        final_classifier_head=True,
        num_classes=10,
        apply_rotary_emb=True,
        apply_fourier_encoding=True,
        max_freq=10,
        num_freq_bands=6,
        constant_mapping=False,
        max_position=1600,
        num_layers=4,
        num_latents=16,
        latent_dim=32,
        cross_num_heads=4,
        cross_head_dim=32,
        self_attend_heads=4,
        self_attend_dim=32,
        transformer_depth=2,
        attention_dropout=0.,
        ff_hidden_dim=64,
        ff_dropout=0.,
        activation=None,
    )

    model = nn.Sequential(
        Rearrange("b c h w -> b h w c"),
        model,
    )

    dataset_train, dataset_test = get_cifar10_dataset()
    train_model_classifier(model, dataset_train, dataset_test, batch_size=128)


if __name__ == "__main__":
    main()
