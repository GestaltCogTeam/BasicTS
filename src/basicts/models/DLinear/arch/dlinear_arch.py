import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition

from ..config.dlinear_config import DLinearConfig


class DLinear(nn.Module):
    """
        Paper: Are Transformers Effective for Time Series Forecasting?
        Link: https://arxiv.org/abs/2205.13504
        Official Code: https://github.com/cure-lab/DLinear
        Venue: AAAI 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: DLinearConfig):
        super().__init__()

        self.decompsition = MovingAverageDecomposition(config.moving_avg, config.stride)
        self.individual = config.individual
        self.num_features = config.num_features

        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(config.input_len, config.output_len)
                for _ in range(self.num_features)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(config.input_len, config.output_len)
                for _ in range(self.num_features)
            ])
        else:
            self.linear_seasonal = nn.Linear(config.input_len, config.output_len)
            self.linear_trend = nn.Linear(config.input_len, config.output_len)

    def forward(self, inputs: torch.Tensor = None) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            inputs (torch.Tensor): inputs data with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: prediction with shape [batch_size, output_len, num_features]
        """

        seasonal, trend = self.decompsition(inputs)
        seasonal, trend = seasonal.transpose(1, 2), trend.transpose(1, 2)
        if self.individual:
            seasonal_output = torch.stack(
                [linear(seasonal[:, i, :]) for i, linear in enumerate(self.linear_seasonal)], dim=1)
            trend_output = torch.stack(
                [linear(trend[:, i, :]) for i, linear in enumerate(self.linear_trend)], dim=1)
        else:
            seasonal_output = self.linear_seasonal(seasonal)
            trend_output = self.linear_trend(trend)

        prediction = seasonal_output + trend_output # [batch_size, num_features, output_len]
        return prediction.permute(0, 2, 1)
