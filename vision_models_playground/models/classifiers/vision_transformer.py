from typing import Literal

import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from vision_models_playground.components.attention import TransformerEncoder
from vision_models_playground.datasets.datasets import get_cifar10_dataset
from vision_models_playground.train.train_classifier import train_model_classifier


class VisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            image_size: int,
            patch_size: int,
            num_classes: int,
            projection_dim: int,
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
            nn.LazyLinear(projection_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, projection_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(
            dim=projection_dim,
            depth=depth,
            heads=heads,
            head_dim=dim_head,
            mlp_dim=mlp_dim,
            mlp_dropout=dropout,
            attention_dropout=dropout,
            apply_rotary_emb=apply_rotary_emb
        )

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, num_classes)
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
        projection_dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        emb_dropout=0.2,
        apply_rotary_emb=True,
        pool="cls",
    )
    train_dataset, test_dataset = get_cifar10_dataset()
    train_model_classifier(model, train_dataset, test_dataset, num_epochs=100)


if __name__ == '__main__':
    main()
