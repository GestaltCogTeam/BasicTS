from typing import Optional, Tuple, List

import torch
from torch import nn

from basicts.modules.embed import SequenceEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer, MultiHeadAttention

from ..config.itransformer_config import iTransformerConfig


class iTransformer(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting, Time Series Classification
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.output_attention = config.output_attention
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
        self.projection = nn.Linear(config.hidden_size, config.output_len)

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

        hidden_states = self.enc_embedding(inputs, inputs_timestamps)

        hidden_states, attn_weights= self.encoder(hidden_states, output_attentions=self.output_attention)

        prediction = self.projection(hidden_states).transpose(1, 2)
        prediction = prediction[..., :self.num_features]

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        if self.output_attention:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
