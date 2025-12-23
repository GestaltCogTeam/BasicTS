from typing import List, Optional, Tuple

import torch
from torch import nn

from basicts.modules.activations import ACT2FN
from basicts.modules.embed import FeatureEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.transformer import Encoder, Seq2SeqDecoder

from ..config.ns_transformer_config import NonstationaryTransformerConfig
from .ns_transformer_layers import (DSAttention,
                                    NonstationaryTransformerDecoderLayer,
                                    NonstationaryTransformerEncoderLayer,
                                    Projector)


class NonstationaryTransformerBackbone(nn.Module):
    """
    Paper: Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
    Official Code: https://github.com/thuml/Nonstationary_Transformers
    Link: https://arxiv.org/abs/2205.14415
    Venue: NeurIPS 2022
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: NonstationaryTransformerConfig):
        super().__init__()
        self.threshold = config.threshold

        # Embedding
        self.enc_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            use_pe=True,
            dropout=config.dropout)

        # Encoder
        self.encoder = Encoder(
            nn.ModuleList(
                [
                    NonstationaryTransformerEncoderLayer(
                        DSAttention(
                            config.hidden_size,
                            config.n_heads,
                            config.dropout
                        ),
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
            layer_norm=nn.LayerNorm(config.hidden_size)
        )

        self.tau_learner = Projector(
            config.num_features,
            config.input_len,
            config.proj_hidden_size,
            config.num_proj_layers,
            output_size=1)
        self.delta_learner = Projector(
            config.num_features,
            config.input_len,
            config.proj_hidden_size,
            config.num_proj_layers,
            output_size=config.input_len)


    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None,
            inputs_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_std (Tensor): Input standard deviation with shape: [batch_size, 1, num_features]
            inputs_mean (Tensor): Input mean with shape: [batch_size, 1, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        inputs_raw = inputs.clone().detach()
        # Normalization
        if inputs_mask is None:
            inputs_mask = torch.ones_like(inputs)
        valid_count = inputs_mask.sum(dim=1, keepdim=True)
        self.mean = inputs.sum(dim=1, keepdim=True) / valid_count
        inputs = (inputs - self.mean) * inputs_mask
        self.std = torch.sqrt(
            (inputs ** 2).sum(dim=1, keepdim=True) / valid_count + 1e-5)
        inputs /= self.std

        tau = self.tau_learner(inputs_raw, self.std)
        tau_clamped = torch.clamp(tau, max=self.threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        delta = self.delta_learner(inputs_raw, self.mean)
        hidden_states = self.enc_embedding(inputs, inputs_timestamps)
        hidden_states, attns = self.encoder(hidden_states, tau=tau, delta=delta)
        return hidden_states, attns, tau, delta


class NonstationaryTransformerForForecasting(nn.Module):
    """
    NonstationaryTransformer for time series forecasting.
    """
    def __init__(self, config: NonstationaryTransformerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.label_len = config.label_len
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features

        self.backbone = NonstationaryTransformerBackbone(config)

        self.dec_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            use_pe=True,
            dropout=config.dropout)

        self.decoder = Seq2SeqDecoder(
            nn.ModuleList(
                [
                    NonstationaryTransformerDecoderLayer(
                        DSAttention(
                            config.hidden_size,
                            config.n_heads,
                            config.dropout
                        ),
                        DSAttention(
                            config.hidden_size,
                            config.n_heads,
                            config.dropout
                        ),
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
                layer_norm=nn.LayerNorm(config.hidden_size)
            )
        self.prediction_head = nn.Linear(config.hidden_size, config.num_features)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor = None,
            inputs_timestamps: Optional[torch.Tensor] = None,
            targets_timestamps: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        enc_hidden_states, enc_attn_weights, tau, delta = self.backbone(inputs, inputs_timestamps)
        dec_hidden_states = torch.cat([inputs[:, -self.label_len:, :], torch.zeros_like(targets)], dim=1)
        targets_timestamps = torch.cat([inputs_timestamps[:, -self.label_len:, :], targets_timestamps], dim=1)
        dec_hidden_states = self.dec_embedding(dec_hidden_states, targets_timestamps)
        dec_hidden_states, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            dec_hidden_states, enc_hidden_states, tau=tau, delta=delta)
        prediction = self.prediction_head(dec_hidden_states)[:, -self.output_len:, :]
        prediction = prediction * self.backbone.std + self.backbone.mean

        if self.output_attentions:
            attn_weights = {"enc_attn_weights": enc_attn_weights,
                            "dec_self_attn_weights": dec_self_attn_weights,
                            "dec_cross_attn_weights": dec_cross_attn_weights}
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class NonstationaryTransformerForClassification(nn.Module):
    """
    NonstationaryTransformer for time series classification.
    """
    def __init__(self, config: NonstationaryTransformerConfig):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features

        self.backbone = NonstationaryTransformerBackbone(config)
        self.act = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)
        self.classification_head = nn.Linear(config.hidden_size * config.num_features, config.num_classes)

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        batch_size = inputs.size(0)
        hidden_states, attn_weights, _, _ = self.backbone(inputs, inputs_timestamps)
        hidden_states = self.dropout(self.act(hidden_states))
        hidden_states = hidden_states.reshape(batch_size, -1)  # [batch_size, num_features * hidden_size]
        prediction = self.classification_head(hidden_states)  # [batch_size, num_classes]
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class NonstationaryTransformerForReconstruction(nn.Module):
    """
    NonstationaryTransformer for time series reconstruction.
    """
    def __init__(self, config: NonstationaryTransformerConfig):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features

        self.backbone = NonstationaryTransformerBackbone(config)
        self.act = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)
        self.reconstruction_head = nn.Linear(config.hidden_size, config.num_features)

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None,
            inputs_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the NonstationaryTransformerForReconstruction model.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        hidden_states, attn_weights, _, _ = self.backbone(inputs, inputs_timestamps, inputs_mask)
        prediction = self.reconstruction_head(hidden_states)  # [batch_size, input_len, num_features]
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
