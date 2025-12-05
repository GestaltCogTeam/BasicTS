import torch
from torch import nn

from basicts.modules import MLPLayer
from basicts.modules.embed import PositionEmbedding, SequenceEmbedding
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, MultiHeadAttention

from ..config.leddam_config import LeddamConfig
from .leddam_layers import (AutoAttention, LearnableDecomposition,
                            LeddamEncoderLayer)


class Leddam(nn.Module):
    """
    Paper: Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with
    Inter-Series Dependencies and Intra-Series Variations Modeling

    Official Code: https://github.com/Levi-Ackman/Leddam
    
    Link: https://openreview.net/forum?id=87CYNyCGOo
    
    Venue: ICML 2024
    
    Task: Time Series Forecasting
    """
    def __init__(self, config: LeddamConfig):

        super().__init__()
        self.output_attentions = config.output_attentions
        self.denoising_layer = LearnableDecomposition(config.kernel_size)
        self.channel_encoder = Encoder(
            nn.ModuleList([
                LeddamEncoderLayer(
                    MultiHeadAttention(config.hidden_size, config.n_heads, config.dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    layer_norm=(nn.LayerNorm, config.hidden_size),
                    attn_norm=nn.BatchNorm1d(config.num_features)
                ) for _ in range(config.num_layers)])
            )
        self.auto_encoder = Encoder(
            nn.ModuleList([
                LeddamEncoderLayer(
                    AutoAttention(config.hidden_size, config.period, config.dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    layer_norm=(nn.LayerNorm, config.hidden_size),
                    attn_norm=nn.BatchNorm1d(config.num_features)
                ) for _ in range(config.num_layers)])
            )
        self.sequence_embedding = SequenceEmbedding(
            config.input_len, config.hidden_size, config.dropout)
        self.position_embedding = PositionEmbedding(config.hidden_size)

        self.linear_main = nn.Linear(config.hidden_size, config.output_len)
        self.linear_res = nn.Linear(config.hidden_size, config.output_len)

        self.linear_main.weight = nn.Parameter(
                (1 / config.hidden_size) * torch.ones([config.output_len, config.hidden_size]))
        self.linear_res.weight = nn.Parameter(
                (1 / config.hidden_size) * torch.ones([config.output_len, config.hidden_size]))

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(affine=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of the Leddam model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features].
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # [batch_size, num_features, hidden_size]
        hidden_states = self.sequence_embedding(inputs) + \
            self.position_embedding(inputs.transpose(1, 2))

        main = self.denoising_layer(hidden_states)
        res = hidden_states - main

        # channel attention blocks and auto attention blocks
        channel_out, channel_attn_weights = self.channel_encoder(
            res, output_attentions=self.output_attentions)
        auto_out, auto_attn_weights = self.auto_encoder(
            res, output_attentions=self.output_attentions)
        res = channel_out + auto_out

        main_out = self.linear_main(main)
        res_out = self.linear_res(res)
        prediction = main_out + res_out
        prediction = prediction.transpose(1, 2)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if self.output_attentions:
            attn_weights = {
                "channel_attn_weights": channel_attn_weights,
                "auto_attn_weights": auto_attn_weights
            }
            return {"prediction": prediction, "attn_weights": attn_weights}
        return prediction
