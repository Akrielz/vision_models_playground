from typing import Optional, Callable, Literal

import torch
from torch import nn

from vision_models_playground.components.attention.transformer_decoder import TransformerDecoder
from vision_models_playground.components.attention.transformer_encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(
            self,
            target_dim: int,
            context_dim: int,
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
            target_dim:
                Dimension of the input tensor.

            context_dim:
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

        self.encoder = TransformerEncoder(
            dim=context_dim,
            depth=depth,
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

        self.decoder = TransformerDecoder(
            target_dim=target_dim,
            context_dim=context_dim,
            depth=depth,
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

    def forward(
            self,
            target: torch.Tensor,
            context: torch.Tensor,
            target_mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            return_context_too: bool = False,
            causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            target:
                Target tensor. [batch_size, seq_len_1, dim_1]

            context:
                Context tensor. [batch_size, seq_len_2, dim_2]

            target_mask:
                Target mask. [batch_size, seq_len_1]

            context_mask:
                Context mask. [batch_size, seq_len_2]

            return_context_too:
                Whether to return the context.

            causal:
                Whether to use causal attention to the target.

        Returns:
            Output tensor. [batch_size, seq_len_1, dim_1] if return_context_too is False,
            otherwise Tuple[Tensor[batch_size, seq_len_1, seq_len_1, dim_1], Tensor[batch_size, seq_len_2, dim_2]]
        """

        # Compute the masks
        encoder_self_attention_mask = self.encoder.compute_2d_mask(context_mask)

        decoder_self_attention_mask = self.decoder.compute_2d_mask(target_mask)
        decoder_self_attention_mask = self.decoder.compute_causal_mask(decoder_self_attention_mask, causal)

        decoder_cross_attention_mask = self.decoder.compute_2d_mask(context_mask, target.shape[1])

        # Forward pass
        for encoder_layer, decoder_layer in zip(self.encoder.layers, self.decoder.layers):
            context = encoder_layer(context, encoder_self_attention_mask)
            target = decoder_layer(target, context, decoder_self_attention_mask, decoder_cross_attention_mask)

        return target if not return_context_too else (target, context)


def main():
    model = Transformer(
        target_dim=64,
        context_dim=128,
        depth=6,
        heads=8,
        head_dim=64,
        mlp_dim=2048,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        apply_rotary_emb=True,
        activation=nn.ReLU(),
        drop_path=0.1,
        norm_type='pre_norm',
    )

    target = torch.randn(2, 6, 64)
    context = torch.randn(2, 10, 128)

    target_mask = torch.ones(2, 6).bool()
    target_mask[:, 3:] = 0
    context_mask = torch.ones(2, 10).bool()
    context_mask[:, 8:] = 0

    output = model(target, context, target_mask, context_mask)
    print(output.shape)


if __name__ == '__main__':
    main()
