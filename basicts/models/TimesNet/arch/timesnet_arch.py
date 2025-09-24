import torch
from torch import nn

from basicts.modules.embed import FeatureEmbedding
from basicts.modules.norm import RevIN

from ..config.timesnet_config import TimesNetConfig
from .times_block import TimesBlock


class TimesNetBackbone(nn.Module):
    """
    Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis, ICLR 2023.
    Link: https://openreview.net/pdf?id=ju_Uqw384Oq
    Official Code: https://github.com/thuml/TimesNet
    Venue: ICLR 2023
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.output_len = config.output_len
        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.num_layers)])
        self.enc_embedding = FeatureEmbedding(
            config.num_features,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            dropout=config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if config.output_len is not None:
            self.predict_linear = nn.Linear(config.input_len, config.input_len + config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimesNetBackbone.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].
            inputs_timestamps (torch.Tensor): Input tensor of timestamps of shape [batch_size, input_len, num_timestamps].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, input_len + output_len, hidden_size].
        """

        # embedding
        hidden_states = self.enc_embedding(inputs, inputs_timestamps)  # [batch_size,input_len, hidden_size]
        # for forecasting task, align the temporal dimension -> [batch_size,input_len + output_len, hidden_size]
        if self.output_len is not None:
            hidden_states = self.predict_linear(hidden_states.transpose(1, 2)).transpose(1, 2)
        # TimesNet
        for layer in self.model:
            hidden_states = self.layer_norm(layer(hidden_states))
        return hidden_states


class TimesNetForForecasting(nn.Module):
    """
    Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis, ICLR 2023.
    Link: https://openreview.net/pdf?id=ju_Uqw384Oq
    Official Code: https://github.com/thuml/TimesNet
    Venue: ICLR 2023
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.output_len = config.output_len
        self.backbone = TimesNetBackbone(config)
        self.projection = nn.Linear(config.hidden_size, config.num_features)
        self.revin = RevIN(affine=False)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimesNetForForecasting.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].
            inputs_timestamps (torch.Tensor): Input tensor of timestamps of shape [batch_size, input_len, num_timestamps].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features].
        """

        inputs = self.revin(inputs, "norm")
        hidden_states = self.backbone(inputs, inputs_timestamps)
        prediction = self.projection(hidden_states)
        prediction = self.revin(prediction[:, -self.output_len:, :], "denorm")
        return prediction
