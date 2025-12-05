from typing import Dict, Union

import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.embed import FeatureEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import CenteredLayerNorm
from basicts.modules.transformer import AutoCorrelation, Encoder

from ..config.autoformer_config import AutoformerConfig
from .layers import (AutoformerDecoder, AutoformerDecoderLayer,
                     AutoformerEncoderLayer)


class Autoformer(nn.Module):
    """
    Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Link: https://arxiv.org/abs/2106.13008
    Ref Official Code: https://github.com/thuml/Autoformer
    Venue: NeurIPS 2021
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.label_len = int(config.label_len)
        self.output_attentions = config.output_attentions

        # Decomp
        self.decomp = MovingAverageDecomposition(config.moving_avg)

        # Embedding
        self.enc_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            dropout=config.dropout)
        self.dec_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            dropout=config.dropout)

        self.encoder = Encoder(
            nn.ModuleList([
                AutoformerEncoderLayer(
                    AutoCorrelation(config.hidden_size, config.n_heads, config.dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    (MovingAverageDecomposition, config.moving_avg)
                ) for _ in range(config.num_encoder_layers)
            ]),
            layer_norm=CenteredLayerNorm(config.hidden_size),
        )

        self.decoder = AutoformerDecoder(
            nn.ModuleList([
                AutoformerDecoderLayer(
                    AutoCorrelation(config.hidden_size, config.n_heads, config.dropout),
                    AutoCorrelation(config.hidden_size, config.n_heads, config.dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    (MovingAverageDecomposition, config.moving_avg),
                    config=config
                ) for _ in range(config.num_decoder_layers)
            ]),
            layer_norm=CenteredLayerNorm(config.hidden_size),
        )

        self.projection = nn.Linear(config.hidden_size, config.num_features)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            inputs_timestamps: torch.Tensor = None,
            targets_timestamps: torch.Tensor = None
            ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Feed forward of Autoformer.

        Args:
            inputs: Input data with shape: [batch_size, input_len, num_features]
            targets: Future data with shape: [batch_size, output_len, num_features]
            inputs_timestamps: Input timestamps with shape: [batch_size, input_len, num_timestamps]
            targets_timestamps: Future timestamps with shape: [batch_size, output_len, num_timestamps]

        Returns:
            Output data with shape: [batch_size, output_len, num_features]
            Attention weights if output_attentions is True, otherwise None.
        """

        # decomp init
        mean = torch.mean(inputs, dim=1, keepdim=True).repeat(1, self.output_len, 1)
        zeros = torch.zeros_like(targets)
        seasonal, trend = self.decomp(inputs)
        # decoder input
        trend = torch.cat([trend[:, -self.label_len:, :], mean], dim=1)
        seasonal = torch.cat([seasonal[:, -self.label_len:, :], zeros], dim=1)

        # encoder
        enc_hidden_states = self.enc_embedding(inputs, inputs_timestamps)

        enc_output, enc_attn_weights = self.encoder(
            enc_hidden_states,
            output_attentions=self.output_attentions
            )

        # decoder
        targets_timestamps = torch.cat([inputs_timestamps[:, -self.label_len:, :], targets_timestamps], dim=1)
        dec_hidden_states = self.dec_embedding(seasonal, targets_timestamps)

        dec_output, trend, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            dec_hidden_states,
            enc_output,
            trend,
            output_attentions=self.output_attentions)

        seasonal = self.projection(dec_output)

        prediction = (trend + seasonal)[:, -self.output_len:, :]

        if self.output_attentions:
            attn_weights = {"enc_attn_weights": enc_attn_weights,
                            "dec_self_attn_weights": dec_self_attn_weights,
                            "dec_cross_attn_weights": dec_cross_attn_weights}
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
