from typing import Optional

import torch
from torch import nn

from basicts.modules.embed import FeatureEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.transformer import (EncoderLayer, MultiHeadAttention,
                                         ProbAttention, Seq2SeqDecoder,
                                         Seq2SeqDecoderLayer,
                                         prepare_causal_attention_mask)

from ..config.informer_config import InformerConfig
from .conv import ConvLayer
from .encoder import InformerEncoder


class Informer(nn.Module):
    """
    Paper: Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting
    Link: https://arxiv.org/abs/2012.07436
    Official Code: https://github.com/zhouhaoyi/Informer2020
    Venue: AAAI 2021
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.output_attentions = config.output_attentions

        # Embedding
        self.enc_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            use_pe=True,
            dropout=config.dropout)
        self.dec_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            use_pe=True,
            dropout=config.dropout)

        # Encoder
        self.encoder = InformerEncoder(
            nn.ModuleList(
                [
                    EncoderLayer(
                        ProbAttention(config.hidden_size, config.n_heads, config.factor, config.dropout)
                        if config.prob_attn else MultiHeadAttention(
                            config.hidden_size, config.n_heads, config.dropout),
                        MLPLayer(
                            config.hidden_size,
                            config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dropout=config.dropout
                            ),
                        layer_norm=(nn.LayerNorm, config.hidden_size),
                        norm_position="post"
                ) for _ in range(config.num_encoder_layers)
            ]),
            nn.ModuleList(
                [
                    ConvLayer(config.hidden_size, hidden_act=config.hidden_act)
                    for _ in range(config.num_encoder_layers - 1)
                ] if config.distill else None),
            layer_norm=nn.LayerNorm(config.hidden_size)
        )
        # Decoder
        self.decoder = Seq2SeqDecoder(
            nn.ModuleList(
                [
                    Seq2SeqDecoderLayer(
                        ProbAttention(config.hidden_size, config.n_heads, config.factor, config.dropout)
                        if config.prob_attn else MultiHeadAttention(
                            config.hidden_size, config.n_heads, config.dropout),
                        MultiHeadAttention(config.hidden_size, config.n_heads, config.dropout),
                        MLPLayer(
                            config.hidden_size,
                            config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dropout=config.dropout
                            ),
                        layer_norm=(nn.LayerNorm, config.hidden_size),
                        norm_position="post"
                        )
                    for _ in range(config.num_decoder_layers)
                ]),
            layer_norm=torch.nn.LayerNorm(config.hidden_size)
        )
        self.projection = nn.Linear(config.hidden_size, config.num_features, bias=True)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None,
            targets_timestamps: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        """Feed forward of Informer.

        Args:
            inputs: Input data with shape: [batch_size, input_len, num_features]
            targets: Future data with shape: [batch_size, output_len, num_features]
            inputs_timestamps: Input timestamps with shape: [batch_size, input_len, num_timestamps]
            targets_timestamps: Future timestamps with shape: [batch_size, output_len, num_timestamps]

        Returns:
            Output data with shape: [batch_size, output_len, num_features]
        """

        enc_hidden_states = self.enc_embedding(inputs, inputs_timestamps)
        enc_hidden_states, enc_attn_weights = self.encoder(enc_hidden_states, output_attentions=self.output_attentions)

        dec_hidden_states = self.dec_embedding(targets, targets_timestamps)
        attention_mask = prepare_causal_attention_mask(
            (targets.shape[0], targets.shape[1]), dec_hidden_states)
        dec_hidden_states, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            dec_hidden_states, enc_hidden_states, attention_mask, output_attentions=self.output_attentions)
        prediction = self.projection(dec_hidden_states)[:, -self.output_len:, :]

        if self.output_attentions:
            attn_weights = {"enc_attn_weights": enc_attn_weights,
                            "dec_self_attn_weights": dec_self_attn_weights,
                            "dec_cross_attn_weights": dec_cross_attn_weights}
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
