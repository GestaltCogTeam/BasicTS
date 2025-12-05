from typing import List, Optional, Tuple

import torch
from torch import nn

from basicts.modules.activations import ACT2FN
from basicts.modules.embed import SequenceEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import (Encoder, EncoderLayer,
                                         MultiHeadAttention)

from ..config.itransformer_config import iTransformerConfig


class iTransformerBackbone(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting, Time Series Classification
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.num_features = config.num_features
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(self.num_features, affine=False)

        # Embedding
        self.enc_embedding = SequenceEmbedding(config.input_len, config.hidden_size, config.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    MultiHeadAttention(config.hidden_size, config.n_heads, config.dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    layer_norm=(nn.LayerNorm, config.hidden_size),
                    norm_position="post"
                ) for _ in range(config.num_layers)
            ]),
            layer_norm=nn.LayerNorm(config.hidden_size),
        )
        self.output_attentions = config.output_attentions

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        hidden_states = self.enc_embedding(inputs, inputs_timestamps)
        hidden_states, attn_weights= self.encoder(hidden_states, output_attentions=self.output_attentions)
        return hidden_states, attn_weights


class iTransformerForForecasting(nn.Module):
    """
    iTransformer for time series forecasting.
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(self.num_features, affine=False)

        self.backbone = iTransformerBackbone(config)
        self.forecasting_head = nn.Linear(config.hidden_size, config.output_len)

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of iTransformerForForecasting.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        hidden_states, attn_weights = self.backbone(inputs, inputs_timestamps)
        prediction = self.forecasting_head(hidden_states).transpose(1, 2)[..., :self.num_features]
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class iTransformerForClassification(nn.Module):
    """
    iTransformer for time series classification.
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features

        self.backbone = iTransformerBackbone(config)
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
        hidden_states, attn_weights = self.backbone(inputs, inputs_timestamps)
        hidden_states = self.dropout(self.act(hidden_states))
        hidden_states = hidden_states.reshape(batch_size, -1)  # [batch_size, num_features * hidden_size]
        prediction = self.classification_head(hidden_states)  # [batch_size, num_classes]
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class iTransformerForReconstruction(nn.Module):
    """
    iTransformer for time series reconstruction.
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_features = config.num_features
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(self.num_features, affine=False)

        self.backbone = iTransformerBackbone(config)
        self.reconstruction_head = nn.Linear(config.hidden_size, config.input_len)

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

        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        hidden_states, attn_weights = self.backbone(inputs, inputs_timestamps)
        prediction = self.reconstruction_head(hidden_states).transpose(1, 2)[..., :self.num_features]
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
