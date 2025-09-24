from typing import Optional, Tuple

import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import CenteredLayerNorm
from basicts.modules.transformer import AutoCorrelation

from ..config.autoformer_config import AutoformerConfig


class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer decoder block with the progressive decomposition architecture
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.self_attn = AutoCorrelation(config.hidden_size,
                                         config.n_heads,
                                         config.dropout,
                                         config.factor)
        self.cross_attn = AutoCorrelation(config.hidden_size,
                                         config.n_heads,
                                         config.dropout,
                                         config.factor)
        self.ffn_layer = MLPLayer(config.hidden_size,
                                  config.intermediate_size,
                                  hidden_act=config.hidden_act,
                                  dropout=config.dropout)
        self.self_attn_decomp = MovingAverageDecomposition(config.moving_avg)
        self.cross_attn_decomp = MovingAverageDecomposition(config.moving_avg)
        self.ffn_decomp = MovingAverageDecomposition(config.moving_avg)
        self.projection = nn.Conv1d(config.hidden_size, config.num_features, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if past_key_value is not None:
            self_past_kv, cross_past_kv = past_key_value[:2], past_key_value[2:]
        else:
            self_past_kv, cross_past_kv = None, None

        residual = hidden_states

        # Self-attention
        self_attn_output, self_attn_weights, self_present_kv = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Residual connection
        hidden_states = residual + self_attn_output

        # Decomposition
        hidden_states, self_trend = self.self_attn_decomp(hidden_states)

        residual = hidden_states

        cross_attn_output, cross_attn_weights, cross_present_kv = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=cross_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Residual connection
        hidden_states = residual + cross_attn_output

        # Decomposition
        hidden_states, cross_trend = self.cross_attn_decomp(hidden_states)

        residual = hidden_states

        # FFN
        ffn_output = self.ffn_layer(hidden_states)

        # Residual connection
        hidden_states = residual + ffn_output

        # Decomposition
        hidden_states, ffn_trend = self.ffn_decomp(hidden_states)

        trend = self_trend + cross_trend + ffn_trend
        trend = self.projection(trend.permute(0, 2, 1)).transpose(1, 2)

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        present_key_value = None
        if use_cache:
            present_key_value = self_present_kv + cross_present_kv

        return hidden_states, trend, self_attn_weights, cross_attn_weights, present_key_value


class AutoformerDecoder(nn.Module):
    """
    Autoformer decoder with the progressive decomposition architecture
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.decoder = nn.ModuleList([AutoformerDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.decoder_post_norm = CenteredLayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        trend: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        past_key_value = None # TODO: update kv cache
        dec_self_attn_weights, dec_cross_attn_weights = [], []

        for layer in self.decoder:
            hidden_states, res_trend, self_attns, cross_attns, past_key_value = layer(
                hidden_states,
                key_value_states,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            trend += res_trend
            if output_attentions:
                dec_self_attn_weights.append(self_attns)
                dec_cross_attn_weights.append(cross_attns)
        dec_output = self.decoder_post_norm(hidden_states)
        return dec_output, trend, dec_self_attn_weights, dec_cross_attn_weights
