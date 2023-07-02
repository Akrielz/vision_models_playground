from typing import Callable, List, Literal

import torch
from einops.layers.torch import Reduce, Rearrange
from torch import nn

from vision_models_playground.components.activations import QuickGELU
from vision_models_playground.components.convolutions.conv_transformer import ConvTransformer
from vision_models_playground.datasets.datasets import get_cifar10_dataset
from vision_models_playground.train.train_classifier import train_model_classifier
from vision_models_playground.utility.functions import get_number_of_parameters


class ConvVisionTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1000,
            activation: Callable = None,
            final_classifier_head: bool = True,

            patch_size: List[int] = None,
            patch_stride: List[int] = None,
            patch_padding: List[int] = None,
            embedding_dim: List[int] = None,
            depth: List[int] = None,
            num_heads: List[int] = None,
            ff_hidden_dim: List[int] = None,
            qkv_bias: List[bool] = None,
            drop_rate: List[float] = None,
            attn_drop_rate: List[float] = None,
            drop_path_rate: List[float] = None,
            kernel_size: List[int] = None,
            stride_kv: List[int] = None,
            stride_q: List[int] = None,
            padding_kv: List[int] = None,
            padding_q: List[int] = None,
            method: List[Literal['conv', 'avg', 'linear']] = None,
    ):
        super().__init__()

        if activation is None:
            activation = QuickGELU()

        # Ensure no param is None
        assert all(x is not None for x in [
            patch_size, patch_stride, patch_padding, embedding_dim, depth, num_heads, ff_hidden_dim,
            qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, kernel_size, stride_kv, stride_q,
            padding_kv, padding_q, method,
        ])

        # Ensure all lists have the same length
        assert len(patch_size) == len(patch_stride) == len(patch_padding) == len(embedding_dim) == \
               len(depth) == len(num_heads) == len(ff_hidden_dim) == len(qkv_bias) == len(drop_rate) == \
               len(attn_drop_rate) == len(drop_path_rate) == len(kernel_size) == len(stride_kv) == \
               len(stride_q) == len(padding_kv) == len(padding_q) == len(method)

        self.num_transformers = len(patch_size)

        self.transformers = nn.ModuleList([])

        current_channels = in_channels
        for i in range(self.num_transformers):
            self.transformers.append(
                ConvTransformer(
                    in_channels=current_channels,
                    patch_size=patch_size[i],
                    patch_stride=patch_stride[i],
                    patch_padding=patch_padding[i],
                    embedding_dim=embedding_dim[i],
                    depth=depth[i],
                    num_heads=num_heads[i],
                    ff_hidden_dim=ff_hidden_dim[i],
                    qkv_bias=qkv_bias[i],
                    drop_rate=drop_rate[i],
                    attn_drop_rate=attn_drop_rate[i],
                    drop_path_rate=drop_path_rate[i],
                    kernel_size=kernel_size[i],
                    stride_kv=stride_kv[i],
                    stride_q=stride_q[i],
                    padding_kv=padding_kv[i],
                    padding_q=padding_q[i],
                    method=method[i],
                    activation=activation,
                )
            )

            current_channels = embedding_dim[i]

        self.head = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            Reduce("b l d -> b d", "mean"),
            nn.LayerNorm(current_channels),
            nn.Linear(current_channels, num_classes),
        ) if final_classifier_head else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transformer in self.transformers:
            x = transformer(x)

        return self.head(x)


def build_cvt_13(num_classes: int = 10, in_channels: int = 3):
    return ConvVisionTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embedding_dim=[64, 192, 384],
        depth=[1, 2, 10],
        num_heads=[1, 3, 6],
        ff_hidden_dim=[256, 768, 1536],
        qkv_bias=[True, True, True],
        drop_rate=[0.0, 0.0, 0.0],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        kernel_size=[3, 3, 3],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        padding_q=[1, 1, 1],
        method=['conv', 'conv', 'conv'],
    )


def build_cvt_21(num_classes: int = 10, in_channels: int = 3):
    return ConvVisionTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embedding_dim=[64, 192, 384],
        depth=[1, 4, 16],
        num_heads=[1, 3, 6],
        ff_hidden_dim=[256, 768, 1536],
        qkv_bias=[True, True, True],
        drop_rate=[0.0, 0.0, 0.0],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        kernel_size=[3, 3, 3],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        padding_q=[1, 1, 1],
        method=['conv', 'conv', 'conv'],
    )


def build_cvt_w24(num_classes: int = 10, in_channels: int = 3):
    return ConvVisionTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embedding_dim=[64, 768, 1024],
        depth=[2, 2, 20],
        num_heads=[3, 12, 16],
        ff_hidden_dim=[256, 3072, 4096],
        qkv_bias=[True, True, True],
        drop_rate=[0.0, 0.0, 0.0],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.3],
        kernel_size=[3, 3, 3],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        padding_q=[1, 1, 1],
        method=['conv', 'conv', 'conv'],
        final_classifier_head=True,
    )


if __name__ == '__main__':
    model = ConvVisionTransformer(
        in_channels=3,
        num_classes=10,
        patch_size=[7, 3],
        patch_stride=[4, 2],
        patch_padding=[2, 1],
        embedding_dim=[64, 192],
        depth=[2, 8],
        num_heads=[1, 3],
        ff_hidden_dim=[256, 768],
        qkv_bias=[True, True],
        drop_rate=[0.0, 0.0],
        attn_drop_rate=[0.0, 0.0],
        drop_path_rate=[0.0, 0.1],
        kernel_size=[3, 3],
        stride_kv=[2, 2],
        stride_q=[1, 1],
        padding_kv=[1, 1],
        padding_q=[1, 1],
        method=['conv', 'conv'],
    )

    # print number of params of the model
    print(f"Number of params: {get_number_of_parameters(model) / (1024 ** 2):.3f} M")

    dataset_train, dataset_test = get_cifar10_dataset()
    train_model_classifier(model, dataset_train, dataset_test)
