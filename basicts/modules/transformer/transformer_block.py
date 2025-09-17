from typing import Callable, Literal, Optional, Tuple

import torch
from torch import nn


class TransformerBlock(nn.Module):

    """
    BasicTS Transformer block.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Callable | Tuple[Callable, dict],
        cross_attn: Optional[nn.Module] = None,
        norm_position: Literal["pre", "post", "sandwich"] = "pre"
    ):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn_layer = ffn_layer
        norm_fn = layer_norm[0] if isinstance(layer_norm, tuple) else layer_norm
        norm_kwargs = layer_norm[1] if isinstance(layer_norm, tuple) else {}

        self.pre_attn_norm = None if norm_position == "post" else norm_fn(**norm_kwargs)
        self.pre_ffn_norm = None if norm_position == "post" else norm_fn(**norm_kwargs)

        self.post_attn_norm = None if norm_position == "pre" else norm_fn(**norm_kwargs)
        self.post_ffn_norm = None if norm_position == "pre" else norm_fn(**norm_kwargs)

        if self.cross_attn is not None:
            self.pre_cross_norm = None if norm_position == "post" else norm_fn(**norm_kwargs)
            self.post_cross_norm = None if norm_position == "pre" else norm_fn(**norm_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        self_past_kv = past_key_value[:2] if past_key_value is not None else None
        cross_past_kv = past_key_value[2:] if past_key_value is not None else None

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, self_attn_weights, self_present_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=self_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        if self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            if self.pre_cross_norm is not None:
                hidden_states = self.pre_cross_norm(hidden_states)
            cross_output, cross_attn_weights, cross_present_kv = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + cross_output
            if self.post_cross_norm is not None:
                hidden_states = self.post_cross_norm(hidden_states)
        else:
            cross_attn_weights = None
            cross_present_kv = None

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

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        present_key_value = None
        if use_cache:
            self_present_kv = self_present_kv if self_present_kv is not None else ()
            cross_present_kv = cross_present_kv if cross_present_kv is not None else ()
            present_key_value = self_present_kv + cross_present_kv

        return hidden_states, self_attn_weights, cross_attn_weights, present_key_value
