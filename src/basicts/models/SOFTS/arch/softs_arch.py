import torch
from torch import nn

from basicts.modules.embed import SequenceEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer

from ..config.softs_config import SOFTSConfig
from .star import STAR


class SOFTS(nn.Module):
    '''
    Paper: SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion
    Official Code: https://github.com/Secilia-Cxy/SOFTS
    Link: https://arxiv.org/pdf/2404.14197
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    '''
    def __init__(self, config: SOFTSConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.enc_embedding = SequenceEmbedding(config.input_len, config.hidden_size, config.dropout)
        # Encoder
        self.encoder = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    STAR(config.hidden_size, config.core_size, config.hidden_act),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout),
                    layer_norm=(nn.LayerNorm, config.hidden_size),
                    norm_position="post"
                ) for _ in range(config.num_layers)
            ]),
            layer_norm=nn.LayerNorm(config.hidden_size)
        )

        # Decoder
        self.projection = nn.Linear(config.hidden_size, config.output_len)

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(affine=False)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SOFTS model.

        Args:
            inputs (`torch.Tensor`): input tensor of shape [batch_size, seq_len, num_features]
            inputs_timestamps (`torch.Tensor`): timestamp tensor of shape [batch_size, seq_len, num_timestamps]
        Returns:
            `torch.Tensor`: prediction tensor of shape [batch_size, output_len, num_features]
        """

        # RevIN
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        num_features = inputs.size(-1)
        hidden_states = self.enc_embedding(inputs, inputs_timestamps)
        hidden_states, _ = self.encoder(hidden_states)
        prediction = self.projection(hidden_states).transpose(1, 2)[:, :, :num_features]

        # RevIN
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        return prediction
