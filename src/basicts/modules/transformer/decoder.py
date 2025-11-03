from typing import Callable, Literal, Optional, Tuple, Union

import torch
from torch import nn

from .kv_cache import KVCache
from .utils import build_layer


class DecoderOnlyLayer(nn.Module):

    """
    BasicTS Transformer decoder layer for decoder-only architecture.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, dict]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre",
    ):
        super().__init__()
        self.self_attn = self_attn
        self.ffn_layer = ffn_layer

        self.pre_attn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.pre_ffn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.post_attn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)
        self.post_ffn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx
        )

        if not output_attentions:
            attn_weights = None
        if not use_cache:
            present_key_value = None

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

        return hidden_states, attn_weights, present_key_value


class AutoRegressiveDecoder(nn.Module):

    """
    Auto-regressive decoder for decoder-only architecture.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = decoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[KVCache]]:

        if use_cache and kv_cache is None:
            kv_cache = KVCache()
        seq_len = hidden_states.size(1)
        past_kv_len = 0
        if use_cache:
            past_kv_len = kv_cache.get_seq_length()

        if position_ids is None:
            position_ids = torch.arange(
                past_kv_len, seq_len + past_kv_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        attn_weights = ()
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attns, kv_cache = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=kv_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
            )
            if output_attentions:
                attn_weights += (attns,)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            attn_weights = None
        return hidden_states, attn_weights, kv_cache


class Seq2SeqDecoderLayer(nn.Module):

    """
    BasicTS Transformer decoder layer for encoder-decoder architecture.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, dict]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre",
    ):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn_layer = ffn_layer

        self.pre_attn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.pre_ffn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.pre_cross_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.post_attn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)
        self.post_ffn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)
        self.post_cross_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[KVCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[KVCache]]]:

        self_past_kv, cross_past_kv = past_key_value[0], past_key_value[1] \
            if past_key_value is not None else (None, None)

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, self_attn_weights, self_present_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        if self.cross_attn is not None and key_value_states is not None:
            residual = hidden_states
            if self.pre_cross_norm is not None:
                hidden_states = self.pre_cross_norm(hidden_states)
            cross_output, cross_attn_weights, cross_present_kv = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
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

        present_key_value = (self_present_kv, cross_present_kv) if use_cache else None

        return hidden_states, self_attn_weights, cross_attn_weights, present_key_value


class Seq2SeqDecoder(nn.Module):
    """
    Seq2Seq decoder for encoder-decoder architecture.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = decoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[KVCache]] = None,
    ) -> Tuple[torch.Tensor,
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[KVCache]]]:

        if use_cache and kv_cache is None:
            kv_cache = (KVCache(), KVCache())

        self_attn_weights, cross_attn_weights = (), ()
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, self_attns, cross_attns, kv_cache = layer(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=kv_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
            )
            if output_attentions:
                self_attn_weights += (self_attns,)
                cross_attn_weights += (cross_attns,)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            self_attn_weights = cross_attn_weights = None
        if not use_cache:
            kv_cache = None
        return hidden_states, self_attn_weights, cross_attn_weights, kv_cache
