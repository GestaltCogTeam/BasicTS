# pylint: disable=not-callable
from typing import Sequence, Tuple

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
    """
    Time series decomposition layer by moving average.
    """

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


class MultiMovingAverageDecomposition(nn.Module):
    """
    Time series decomposition layer using multiple moving averages.
    """
    def __init__(self, kernel_size: Sequence[int], stride: int = 1):
        super().__init__()
        self.moving_avg = [MovingAverage(kernel, stride) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the series decomposition block.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            seasonal (torch.Tensor): Seasonal tensor of shape [batch_size, seq_len, num_features]
            trend (torch.Tensor): Trend tensor of shape [batch_size, seq_len, num_features]
        """
        trend=[]
        for func in self.moving_avg:
            moving_avg = func(inputs)
            trend.append(moving_avg.unsqueeze(-1))
        trend=torch.cat(trend,dim=-1)
        trend = torch.sum(trend*nn.Softmax(-1)(self.layer(inputs.unsqueeze(-1))),dim=-1)
        seasonal = inputs - trend
        return seasonal, trend


class DFTDecomposition(nn.Module):
    """
    Time series decomposition layer by discrete Fourier transform.
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DFT decomposition layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            seasonal (torch.Tensor): Seasonal tensor of shape [batch_size, seq_len, num_features]
            trend (torch.Tensor): Trend tensor of shape [batch_size, seq_len, num_features]
        """
        inputs_freq = torch.fft.rfft(inputs).abs()
        freq = inputs_freq.abs()
        freq[0] = 0 # remove the DC component
        top_k_amps, _ = torch.topk(freq, self.top_k)
        inputs_freq[freq <= top_k_amps.min()] = 0
        seasonal = torch.fft.irfft(inputs_freq)
        trend = inputs - seasonal
        return seasonal, trend
