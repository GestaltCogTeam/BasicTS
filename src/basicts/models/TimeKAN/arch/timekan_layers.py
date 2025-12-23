# pylint: disable=not-callable
from typing import Callable, List

import torch
from torch import nn

from ..config.timekan_config import TimeKANConfig


class ChebyKANLinear(nn.Module):

    """
    Kolmogorov-Arnold Network layer using Chebyshev polynomials instead of splines coefficients.
    """

    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        # x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        x = x* self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        return y


class FrequencyBaseLayer(nn.Module):
    """
    Base class for frequency processing modules.
    Provides common functionality for frequency decomposition and mixing.
    """
    def __init__(self, config: TimeKANConfig):
        super().__init__()
        self.down_sampling_window = config.down_sampling_window
        self.down_sampling_layers = config.down_sampling_layers
        self.input_len = config.input_len

    def _process_levels(self,
                       level_list: List[torch.Tensor],
                       process_func: Callable) -> List[torch.Tensor]:
        """
        Common processing pipeline for frequency levels.
        
        Args:
            level_list: List of frequency levels (from high to low)
            process_func: Function to process each level pair
            
        Returns:
            Processed levels (from low to high)
        """
        reversed_levels = level_list.copy()
        reversed_levels.reverse()

        out_low = reversed_levels[0]
        out_high = reversed_levels[1]
        processed_levels = [out_low]

        for i in range(len(reversed_levels) - 1):
            # Calculate current and target lengths for interpolation
            current_len = self.input_len // (self.down_sampling_window ** (self.down_sampling_layers - i))
            target_len = self.input_len // (self.down_sampling_window ** (self.down_sampling_layers - i - 1))

            # Process the current level pair
            processed_high = process_func(out_low, out_high, current_len, target_len, i)

            # Update tracking variables
            out_low = out_high
            if i + 2 <= len(reversed_levels) - 1:
                out_high = reversed_levels[i + 2]

            processed_levels.append(processed_high)

        processed_levels.reverse()
        return processed_levels


class FrequencyDecompLayer(FrequencyBaseLayer):
    """
    Frequency decomposition layer that separates high and low frequency components.
    """

    # pylint: disable=unused-argument
    def _decompose_levels(self,
                          low_freq: torch.Tensor,
                          high_freq: torch.Tensor,
                          current_len: int,
                          target_len: int,
                          block_idx: int) -> torch.Tensor:
        """
        Decompose frequency components by subtracting the interpolated low frequency.
        """
        high_res = frequency_interpolation(
            low_freq.transpose(1, 2), current_len, target_len).transpose(1, 2)
        return high_freq - high_res

    def forward(self, level_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._process_levels(level_list, self._decompose_levels)


class FrequencyMixingLayer(FrequencyBaseLayer):

    """
    Frequency mixing layer that combines low and high frequency components.
    """

    def __init__(self, config: TimeKANConfig):
        super().__init__(config)
        self.front_block = TimeKANLayer(config.hidden_size, degree=config.begin_order)
        self.front_blocks = nn.ModuleList([
            TimeKANLayer(config.hidden_size, degree=config.begin_order + i + 1)
            for i in range(config.down_sampling_layers)
        ])

    def _mix_levels(self,
                   low_freq: torch.Tensor,
                   high_freq: torch.Tensor,
                   current_len: int,
                   target_len: int,
                   block_idx: int
                   ) -> torch.Tensor:

        transformed_high = self.front_blocks[block_idx](high_freq)
        high_res = frequency_interpolation(
            low_freq.transpose(1, 2),
            current_len,
            target_len
        ).transpose(1, 2)
        return transformed_high + high_res

    def forward(self, level_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._process_levels(level_list, self._mix_levels)


def frequency_interpolation(x, seq_len: int, target_len: int) -> torch.Tensor:
    len_ratio = seq_len / target_len
    x_fft = torch.fft.rfft(x, dim=2)
    out_fft = torch.zeros(
        [x_fft.size(0), x_fft.size(1), target_len // 2 + 1],
        dtype=x_fft.dtype,
        device=x_fft.device)
    out_fft[:, :, :seq_len // 2 + 1] = x_fft
    out = torch.fft.irfft(out_fft, dim=2)
    return out * len_ratio


class TimeKANLayer(nn.Module):

    """
    M_KAN layer.
    """

    def __init__(self, hidden_size: int, degree: int):
        super().__init__()
        self.channel_mixer = ChebyKANLinear(hidden_size, hidden_size, degree)
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the M_KAN layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        batch_size, _, hidden_size = x.shape

        x_kan = self.channel_mixer(x.reshape(-1, hidden_size))
        x_kan = x_kan.reshape(batch_size, -1, hidden_size)

        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)

        return x_kan + x_conv
