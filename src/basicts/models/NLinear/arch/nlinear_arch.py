import torch
from torch import nn

from basicts.models.NLinear.config.nlinear_config import NLinearConfig


class NLinear(nn.Module):
    """
    Paper: Are Transformers Effective for Time Series Forecasting?
    Link: https://arxiv.org/abs/2205.13504
    Official Code: https://github.com/cure-lab/DLinear
    Venue: AAAI 2023
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: NLinearConfig):
        super().__init__()
        self.linear = nn.Linear(config.input_len, config.output_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feed forward of NLinear.

        Args:
            inputs (torch.Tensor): input data with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: prediction with shape [batch_size, output_len, num_features]
        """

        last_value = inputs[:,-1:,:].detach()
        inputs = inputs - last_value
        inputs = self.linear(inputs.permute(0,2,1)).permute(0,2,1)
        prediction = inputs + last_value
        return prediction
