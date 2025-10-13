import torch
from torch import nn

from basicts.modules import MLPLayer
from basicts.modules.embed import PatchEmbedding
from basicts.modules.norm import RevIN
from basicts.modules.transformer import (AutoRegressiveDecoder,
                                         DecoderOnlyLayer, MultiHeadAttention,
                                         prepare_causal_attention_mask)

from ..config import TimerConfig


class Timer(nn.Module):

    """
    Paper: Timer: Generative Pre-trained Transformers Are Large Time Series Models
    Link: https://arxiv.org/abs/2402.02368
    Repo: https://github.com/thuml/Large-Time-Series-Model
    Venue: ICML 2024
    Task: Forecasting, imputation, anomaly detection
    """

    def __init__(self, config: TimerConfig):
        super().__init__()

        self.embedding = PatchEmbedding(
            config.hidden_size,
            config.patch_len,
            config.patch_len,
            dropout=config.dropout)

        self.decoder = AutoRegressiveDecoder(
            nn.ModuleList(
                [
                    DecoderOnlyLayer(
                        MultiHeadAttention(
                            config.hidden_size,
                            config.n_heads,
                            dropout=config.dropout
                        ),
                        MLPLayer(
                            config.hidden_size,
                            config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dropout=config.dropout
                        ),
                        layer_norm=(nn.LayerNorm, config.hidden_size),
                        norm_position="post"
                    ) for _ in range(config.num_layers)
                ]
            ),
            layer_norm=nn.LayerNorm(config.hidden_size)
        )

        self.head = nn.Linear(config.hidden_size, config.patch_len)
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(affine=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Timer model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features].
        """

        batch_size, _,  num_features = inputs.shape

        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # [batch_size * num_features, num_patches, hidden_size]
        hidden_states = self.embedding(inputs)

        # decoder
        hidden_states, _, _ = self.decoder(
            hidden_states,
            attention_mask=prepare_causal_attention_mask(
                (batch_size, hidden_states.shape[1]),
                hidden_states
            )
        )

        # [batch_size * num_features, num_patches, hidden_size]
        prediction = self.head(hidden_states)
        prediction = prediction.reshape(batch_size, num_features, -1).transpose(1, 2)

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        return prediction
