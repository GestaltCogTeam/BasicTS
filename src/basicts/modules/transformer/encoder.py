from typing import Any, Callable, List, Literal, Optional, Tuple

import torch
from torch import nn

from .utils import build_layer_norm


class EncoderLayer(nn.Module):

    """
    BasicTS Transformer block.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Callable | Tuple[Callable, Any],
        norm_position: Literal["pre", "post", "sandwich"] = "pre"
    ):
        super().__init__()
        self.self_attn = self_attn
        self.ffn_layer = ffn_layer

        self.pre_attn_norm = None if norm_position == "post" \
            else build_layer_norm(layer_norm)
        self.pre_ffn_norm = None if norm_position == "post" \
            else build_layer_norm(layer_norm)
        self.post_attn_norm = None if norm_position == "pre" \
            else build_layer_norm(layer_norm)
        self.post_ffn_norm = None if norm_position == "pre" \
            else build_layer_norm(layer_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        if not output_attentions:
            attn_weights = None

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        residual = hidden_states

        # Pre-LN
        if self.pre_ffn_norm is not None:
            hidden_states = self.pre_ffn_norm(hidden_states)

        # FFN
        ffn_output = self.ffn_layer(hidden_states)

        # Residual connection
        hidden_states = residual + ffn_output

        # Post-LN
        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states, attn_weights


class Encoder(nn.Module):

    """
    BasicTS Transformer encoder.
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = encoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:

        attn_weights = []
        for layer in self.layers:
            hidden_states, attns = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attn_weights.append(attns)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            attn_weights = None
        return hidden_states, attn_weights
