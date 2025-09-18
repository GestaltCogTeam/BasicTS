from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    
    Args:
        kernel_size (int): kernel size of moving average.
        stride (int): stride of moving average.
    """

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride)
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the moving average block.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, num_features]
        """
        # padding on the both ends of time series
        inputs = F.pad(inputs.transpose(1, 2), (self.pad_left, self.pad_right), mode='replicate')
        outputs = self.avg(inputs)
        return outputs.transpose(1, 2)


class MovingAverageDecomposition(nn.Module):
    """Time series decomposition block by moving average."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the moving average decomposition block.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            seasonal (torch.Tensor): Seasonal tensor of shape [batch_size, seq_len, num_features]
            trend (torch.Tensor): Trend tensor of shape [batch_size, seq_len, num_features]
        """
        trend = self.moving_avg(inputs)
        seasonal = inputs - trend
        return seasonal, trend
