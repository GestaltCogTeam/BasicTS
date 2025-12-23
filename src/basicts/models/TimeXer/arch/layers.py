from typing import Callable, Tuple

import torch
from torch import nn

from basicts.modules.transformer import Seq2SeqDecoderLayer


class FlattenHead(nn.Module):
    """
    Flatten head for TimeXer.
    """
    def __init__(self, nf, target_window, dropout = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class TimeXerEncoderLayer(Seq2SeqDecoderLayer):
    """
        TimeXer encoder layer.
    """
    def __init__(self,
                 self_attn: nn.Module,
                 cross_attn: nn.Module,
                 ffn_layer: nn.Module,
                 layer_norm: Callable | Tuple[Callable, dict]):
        super().__init__(self_attn, cross_attn, ffn_layer, layer_norm=layer_norm, norm_position="post")

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs # pylint: disable=unused-argument
    ):

        batch_size, _, hidden_size = key_value_states.shape

        residual = hidden_states

        # Self-attention
        attn_output, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        global_token = hidden_states[:, -1:, :]
        residual = global_token

        cross_output, cross_attn_weights, _ = self.cross_attn(
            global_token.reshape(batch_size, -1, hidden_size),
            key_value_states=key_value_states,
            output_attentions=output_attentions
        )
        cross_output = cross_output.reshape(-1, 1, hidden_size)
        global_token = residual + cross_output

        if self.post_cross_norm is not None:
            global_token = self.post_cross_norm(global_token)

        hidden_states = torch.cat([hidden_states[:, :-1, :], global_token], dim=1)

        residual = hidden_states

        # FFN
        ffn_output = self.ffn_layer(hidden_states)

        # Residual connection
        hidden_states = residual + ffn_output

        # Post-LN
        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        return hidden_states, self_attn_weights, cross_attn_weights
