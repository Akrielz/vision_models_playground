from typing import Optional, Callable, Literal, List, Union

import torch
from einops import repeat
from torch import nn

from vision_models_playground.components.attention.transformer_decoder_layer import TransformerDecoderLayer
from vision_models_playground.utility.masks import create_triangular_mask


class TransformerDecoder(nn.Module):
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

        def get_transformer_decoder_layer() -> nn.Module:
            return TransformerDecoderLayer(
                target_dim=target_dim,
                context_dim=context_dim,
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
            self.layers.append(get_transformer_decoder_layer())

    def forward(
            self,
            target: torch.Tensor,
            context: Union[torch.Tensor, List[torch.Tensor]],
            target_mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            target:
                Target tensor. [batch_size, seq_len_1, dim_1]

            context:
                Context tensor. [batch_size, seq_len_2, dim_2]
                or a list [batch_size, seq_len_i, dim_i], where len(context) = depth.

            target_mask:
                Mask for the target tensor of shape [batch_size, seq_len_1]
                or [batch_size, seq_len_1, seq_len_1].

            context_mask:
                Mask for context tensor of shape [batch_size, seq_len_2]
                or [batch_size, seq_len_1, seq_len_2].

            causal:
                Whether to use causal attention to the target.
        """

        # Assert default values.
        context_list = context
        if isinstance(context, torch.Tensor):
            context_list = [context for _ in range(len(self.layers))]

        # Make sure the context_list was correctly provided.
        assert len(context_list) == len(self.layers)

        # Compute the mask for the target.
        target_mask = self.compute_2d_mask(target_mask)
        target_mask = self.compute_causal_mask(target_mask, causal)

        # Compute the mask for the context.
        context_mask = self.compute_2d_mask(context_mask, target.shape[1])

        # Forward pass.
        for transformer_decoder_layer, context in zip(self.layers, context_list):
            target = transformer_decoder_layer(target, context, target_mask, context_mask)

        return target

    @staticmethod
    def compute_causal_mask(target_mask, causal):
        if causal and target_mask is not None:
            triangular_mask = create_triangular_mask(target_mask.shape[0], target_mask.shape[1], device=target_mask.device)
            target_mask = target_mask & triangular_mask

        return target_mask

    @staticmethod
    def compute_2d_mask(mask: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if mask is not None and mask.dim() == 2:
            if target_len is None:
                target_len = mask.size(1)
            mask = repeat(mask, "b n -> b l n", l=target_len)

        return mask


def main():
    decoder = TransformerDecoder(
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

    output = decoder(target, context, target_mask, context_mask, causal=True)
    print(output.shape)


if __name__ == '__main__':
    main()
