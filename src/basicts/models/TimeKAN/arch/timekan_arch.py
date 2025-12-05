import torch
from torch import nn

from basicts.modules.embed import FeatureEmbedding
from basicts.modules.norm import RevIN

from ..config.timekan_config import TimeKANConfig
from .timekan_layers import FrequencyDecompLayer, FrequencyMixingLayer


class TimeKAN(nn.Module):

    """
    Paper: TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting
    Official Code: https://github.com/huangst21/TimeKAN
    Link: https://arxiv.org/abs/2502.06910
    Venue: ICLR 2025
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: TimeKANConfig):
        super().__init__()
        self.down_sampling_layers = config.down_sampling_layers
        self.down_pooling = nn.AvgPool1d(config.down_sampling_window)
        self.freq_decomp_layers = nn.ModuleList(
            [FrequencyDecompLayer(config) for _ in range(config.num_layers)])
        self.freq_mixing_layers = nn.ModuleList(
            [FrequencyMixingLayer(config) for _ in range(config.num_layers)])
        self.enc_embedding = FeatureEmbedding(1, config.hidden_size, dropout=config.dropout)
        self.norm_layers = nn.ModuleList([RevIN(config.num_features, affine=True)
                                                for _ in range(self.down_sampling_layers + 1)])
        self.num_layers = config.num_layers
        self.predict_layer =nn. Linear(config.input_len, config.output_len)
        self.projection_layer = nn.Linear(config.hidden_size, 1)

    def _prepare_multi_scale_inputs(self, inputs: torch.Tensor) -> list[torch.Tensor]:

        multi_scale_inputs = [inputs]
        sample = inputs.permute(0, 2, 1) # [batch_size, num_features, seq_len]

        for _ in range(self.down_sampling_layers):
            down_sampled = self.down_pooling(sample)
            multi_scale_inputs.append(down_sampled.permute(0, 2, 1))
            sample = down_sampled

        return multi_scale_inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of TimeKAN model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_len, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_len, num_features).
        """

        batch_size, _, num_features = inputs.size()
        x_list = self._prepare_multi_scale_inputs(inputs)

        hidden_states_list = []
        for i, x in enumerate(x_list):
            input_len = x.size(1)
            x = self.norm_layers[i](x, "norm")
            x = x.transpose(1, 2).reshape(-1, input_len, 1)
            hidden_states = self.enc_embedding(x)
            hidden_states_list.append(hidden_states)

        for i in range(self.num_layers):
            hidden_states_list = self.freq_decomp_layers[i](hidden_states_list)
            hidden_states_list = self.freq_mixing_layers[i](hidden_states_list)

        # [batch_size, output_len, hidden_size]
        prediction = self.predict_layer(hidden_states_list[0].permute(0, 2, 1)).permute(0, 2, 1)
        prediction = self.projection_layer(prediction).reshape(batch_size, num_features, -1).transpose(1, 2)
        prediction = self.norm_layers[0](prediction, "denorm")
        return prediction
