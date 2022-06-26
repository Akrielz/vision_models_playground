from typing import Literal

import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from models.components.attention.transformer_encoder import TransformerEncoder
from utility.datasets import get_mnist_dataset, get_cifar10_dataset
from utility.train_models import train_model


class VisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            image_size: int,
            patch_size: int,
            num_classes: int,
            dim: int,
            depth: int,
            heads: int,
            mlp_dim: int,
            pool: Literal["cls", "mean"] = 'mean',
            dim_head: int = 64,
            dropout: float = 0.0,
            emb_dropout: float = 0.0,
            apply_rotary_emb: bool = False
    ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LazyLinear(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(
            dim, depth, heads, dim_head, mlp_dim, dropout, dropout, apply_rotary_emb
        )

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img: torch.Tensor):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)

        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


def main():
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        emb_dropout=0.2,
        apply_rotary_emb=True,
        pool="cls",
    ).cuda()
    train_dataset, test_dataset = get_cifar10_dataset()
    train_model(model, train_dataset, test_dataset, num_epochs=100)


if __name__ == '__main__':
    main()
