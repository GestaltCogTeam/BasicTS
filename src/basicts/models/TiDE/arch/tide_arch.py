from typing import Optional

import torch
from torch import nn

from basicts.modules.norm import RevIN

from ..config.tide_config import TiDEConfig


class ResBlock(nn.Module):
    """
    This is the MLP-based Residual Block
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            bias: bool = True):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.fc3 = nn.Linear(input_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(output_size, bias=bias)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs + self.fc3(inputs)
        outputs = self.ln(outputs)
        return outputs


class TiDE(nn.Module):
    """
    Paper: Long-term Forecasting with TiDE: Time-series Dense Encoder
    Official Code: https://github.com/lich99/TiDE
    Link: https://arxiv.org/abs/2304.08424
    Venue: TMLR 2023
    Task: Long-term Time Series Forecasting
    """

    def __init__(self,  config: TiDEConfig):
        super().__init__()

        self.input_len = config.input_len
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.hidden_size = config.hidden_size
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers =  config.num_decoder_layers
        self.intermediate_size = config.intermediate_size
        self.num_timestamps = config.num_timestamps
        self.timestamps_encode_size = config.timestamps_encode_size
        self.dropout = config.dropout
        self.bias = config.bias
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(affine=False)

        flatten_dim = self.input_len + (self.input_len + self.output_len) * self.timestamps_encode_size

        self.feature_encoder = ResBlock(
            self.num_timestamps, self.hidden_size, self.timestamps_encode_size, self.dropout, self.bias)
        self.dense_encoders = nn.Sequential(
            ResBlock(flatten_dim, self.hidden_size, self.hidden_size, self.dropout, self.bias),
            *([ResBlock(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, self.bias)
               for _ in range(self.num_encoder_layers - 1)]))

        self.dense_decoders = nn.Sequential(
            *([ResBlock(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, self.bias)
               for _ in range(self.num_decoder_layers - 1)]),
               ResBlock(self.hidden_size, self.hidden_size, self.num_features * self.output_len,
                        self.dropout, self.bias))
        self.temporal_decoder = ResBlock(
            self.num_features + self.timestamps_encode_size, self.intermediate_size, 1,
            self.dropout, self.bias)
        self.residual_proj = nn.Linear(self.input_len, self.output_len, bias=self.bias)

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None,
            targets_timestamps: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        """Feed forward of TiDE.

        Args:
            inputs: Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps: Input timestamps with shape: [batch_size, input_len, num_timestamps]
            targets_timestamps: Future timestamps with shape: [batch_size, output_len, num_timestamps]

        Returns:
            Output data with shape: [batch_size, output_len, num_features]
        """
        # Normalization
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # Timestamps
        if targets_timestamps is None:
            timestamps = torch.zeros((inputs.shape[0], self.input_len + self.output_len, self.num_timestamps)).to(inputs.device).detach()
        else:
            timestamps = torch.concat([inputs_timestamps, targets_timestamps[:, -self.output_len:, :]], dim=1)

        # Backbone
        prediction = []
        for i in range(self.num_features):
            x_enc = inputs[:,:,i]
            feature = self.feature_encoder(timestamps)
            hidden = self.dense_encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
            decoded = self.dense_decoders(hidden).reshape(hidden.shape[0], self.output_len, self.num_features)
            dec_out = self.temporal_decoder(torch.cat([feature[:, self.output_len:], decoded], dim=-1)).squeeze(
                -1) + self.residual_proj(x_enc)
            prediction.append(dec_out.unsqueeze(-1))

        # De-Normalization
        prediction = torch.cat(prediction, dim=-1)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        return prediction
