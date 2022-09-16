from typing import Optional

import torch
from torch import nn

from vision_models_playground.components.attention.attend import Attend


class Compressor(nn.Module):
    """
    Compression that uses attention mechanism with information from input
    """

    def __init__(
            self,
            input_dim: int,
            output_len: int,
            output_dim: Optional[int] = None,
            attn_heads: int = 8,
            attn_dim_head: int = 64,
            attn_dropout: float = 0.0,
            hidden_dim: int = 256,
            ff_dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Dimension of the input.
            output_len: Length of the output.
            output_dim: Dimension of the output.
            attn_heads: Number of attention heads.
            attn_dim_head: Dimension of the attention heads.
            attn_dropout: Dropout probability for the attention.
            hidden_dim: Dimension of the hidden components.
            ff_dropout: Dropout probability for the feed-forward components.
        """

        super(Compressor, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.cross_attend = Attend(
            query_dim=input_dim,
            context_dim=input_dim,
            num_heads=attn_heads,
            head_dim=attn_dim_head,
            attention_dropout=attn_dropout,
            apply_rotary_emb=False,
            ff_hidden_dim=hidden_dim,
            ff_dropout=ff_dropout
        )

        self.channel_reshape = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        ) if output_dim != input_dim else nn.Identity()

        self.output_len = output_len

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
        Returns:
            Output tensor of shape (batch_size, output_len, dim).
        """

        assert x.shape[1] >= self.output_len, "Input sequence length must be >= output length"

        queries = x[:, :self.output_len, :]
        compressed = self.cross_attend(queries=queries, context=x)
        compressed = self.channel_reshape(compressed)

        return compressed


def main():
    x = torch.randn(1, 64, 16)

    compressor = Compressor(
        input_dim=16,
        output_len=32,
    )

    out = compressor(x)
    print(out.shape)


if __name__ == "__main__":
    main()
