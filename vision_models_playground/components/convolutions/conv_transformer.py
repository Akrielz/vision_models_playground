from typing import Callable, Optional

import torch
from torch import nn

from vision_models_playground.components.convolutions.conv_attend import ConvAttend
from vision_models_playground.components.convolutions.conv_embedding import ConvEmbedding


class ConvTransformer(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 4,
            patch_stride: int = 4,
            patch_padding: int = 0,
            embedding_dim: int = 512,
            depth: int = 4,
            num_heads: int = 8,
            ff_hidden_dim: int = 2048,
            qkv_bias: bool = False,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            activation: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_features = embedding_dim

        self.patch_embedding = ConvEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            stride=patch_stride,
            padding=patch_padding,
            out_channels=embedding_dim,
            apply_norm=True
        )

        self.positional_drop = nn.Dropout2d(drop_rate)
        self.drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(
                ConvAttend(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    num_heads=num_heads,
                    dim_per_head=embedding_dim // num_heads,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=self.drop_path_rate[i],
                    activation=activation,
                    ff_hidden_dim=ff_hidden_dim,
                    output_2d=True,
                    **kwargs
                )
            )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x


def main():
    conv_transformer = ConvTransformer(
        patch_size=4,
        patch_stride=4,
        patch_padding=0,
        in_channels=3,
        embedding_dim=32,
        num_heads=8,
        ff_hidden_dim=256,
    )

    x = torch.randn(1, 3, 64, 64)
    y = conv_transformer(x)
    print(y.shape)


if __name__ == "__main__":
    main()
