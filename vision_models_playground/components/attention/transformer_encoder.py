from typing import Optional, Callable, Literal, List, Union

import torch
from einops import repeat
from torch import nn

from vision_models_playground.components.attention.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            head_dim: int,
            mlp_dim: int,
            mlp_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            apply_rotary_emb: bool = False,
            activation: Optional[Callable] = None,
            drop_path: float = 0.0,
            norm_type: Literal['pre_norm', 'post_norm'] = 'pre_norm',
    ):
        """
        Args:
            dim:
                Dimension of the input tensor.

            depth:
                Number of layers in the encoder.

            heads:
                Number of attention heads.

            head_dim:
                Dimension of each attention head.

            mlp_dim:
                Dimension of the MLP.

            mlp_dropout:
                Dropout probability for the MLP.

            attention_dropout:
                Dropout probability for the attention.

            apply_rotary_emb:
                Whether to apply rotary embedding to the queries.

            activation:
                Activation function for the MLP.

            drop_path:
                Dropout probability for the drop path.

            norm_type:
                Type of normalization to use.
            """

        super().__init__()

        def get_transformer_encoder_layer() -> nn.Module:
            return TransformerEncoderLayer(
                dim=dim,
                heads=heads,
                head_dim=head_dim,
                mlp_dim=mlp_dim,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                apply_rotary_emb=apply_rotary_emb,
                activation=activation,
                drop_path=drop_path,
                norm_type=norm_type,
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(get_transformer_encoder_layer())

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x:
                Input tensor of shape [batch_size, seq_len, dim].

            mask:
                Mask tensor of shape [batch_size, seq_len] or [batch_size, seq_len, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """

        # Compute the mask for the input.
        mask = self.compute_2d_mask(mask)

        # Forward pass
        for transformer_encoder_layer in self.layers:
            x = transformer_encoder_layer(x=x, mask=mask)

        return x

    @staticmethod
    def compute_2d_mask(mask: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if mask is not None and mask.dim() == 2:
            if target_len is None:
                target_len = mask.size(1)
            mask = repeat(mask, "b n -> b l n", l=target_len)

        return mask


def main():
    encoder = TransformerEncoder(
        dim=512,
        depth=6,
        heads=8,
        head_dim=64,
        mlp_dim=512,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        apply_rotary_emb=True,
        activation=nn.ReLU(),
        norm_type='post_norm',
    )

    x = torch.randn(4, 10, 512)
    mask = torch.ones(4, 10).bool()
    mask[:, 5:] = 0

    print(encoder(x, mask).shape)


if __name__ == '__main__':
    main()