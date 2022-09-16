from typing import Literal

import torch
from einops import rearrange
from torch import nn, einsum

from vision_models_playground.components.convolutions.conv_projection import ConvProjection


class ConvAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            dim_per_head: int = 64,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            output_drop: float = 0.0,
            kernel_size: int = 3,
            stride_kv: int = 1,
            stride_q: int = 1,
            padding_kv: int = 1,
            padding_q: int = 1,
            method: Literal['conv', 'avg', 'linear'] = 'conv',
    ):
        super().__init__()

        # Save params
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5
        self.inner_dim = dim_per_head * num_heads

        method_q = 'linear' if method == 'avg' else method
        method_kw = method

        # Get convolution projections components for query, key and value
        self.to_q = ConvProjection(
            in_channels, self.inner_dim, kernel_size, padding_q, stride_q, method_q, qkv_bias,
        )
        self.to_k = ConvProjection(
            in_channels, self.inner_dim, kernel_size, padding_kv, stride_kv, method_kw, qkv_bias
        )
        self.to_v = ConvProjection(
            in_channels, self.inner_dim, kernel_size, padding_kv, stride_kv, method_kw, qkv_bias
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, out_channels)
        self.proj_drop = nn.Dropout(output_drop)

    def forward(self, x):
        # Get query, key and value
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Reshape query, key and value according to heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), [q, k, v])

        # Calculate attention sim matrix
        attn_sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = attn_sim.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project to output channels
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


def main():
    block = ConvAttention(in_channels=3, out_channels=60, num_heads=8)
    x = torch.randn(1, 3, 64, 64)
    print(block(x).shape)


if __name__ == "__main__":
    main()