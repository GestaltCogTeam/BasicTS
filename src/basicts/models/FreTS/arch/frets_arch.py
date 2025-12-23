# pylint: disable=not-callable
import torch
from torch import nn
from torch.nn import functional as F

from basicts.modules import MLPLayer

from ..config.frets_config import FreTSConfig


class FreTS(nn.Module):
    """
    Paper: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting
    Official Code: https://github.com/aikunyi/FreTS
    Link: https://arxiv.org/abs/2311.06184
    Venue: NeurIPS 2023
    Task: Time Series Forecasting
    """

    def __init__(self, config: FreTSConfig):
        super().__init__()
        self.channel_independence = config.channel_independence
        self.embedding = nn.Parameter(torch.randn(1, config.embed_size))
        self.temporal_mlp = FreMLP(config.embed_size, config.scale, config.sparsity_threshold)
        if not self.channel_independence:
            self.channel_mlp = FreMLP(config.embed_size, config.scale, config.sparsity_threshold)

        self.fc = MLPLayer(
            config.input_len * config.embed_size,
            config.hidden_size,
            output_size=config.output_len,
            hidden_act=config.hidden_act,
            dropout=config.dropout
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FreTS model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features].
        """
        B, L, N = inputs.shape
        inputs = inputs.transpose(1, 2).unsqueeze(-1) # [B, L, N] -> [B, N, L, 1]
        hidden_states = inputs * self.embedding # [B, N, L, 1] -> [B, N, L, D]
        residual = hidden_states
        if not self.channel_independence:
            # FFT on N dimension: [B, L, N, D] -> [B, L, N//2+1, D]
            hidden_states = torch.fft.rfft(hidden_states.transpose(1, 2), dim=2, norm="ortho")
            hidden_states = self.channel_mlp(hidden_states)
            # IFFT on N dimension: [B, L, N//2+1, D] -> [B, L, N, D]
            hidden_states = torch.fft.irfft(hidden_states, n=N, dim=2, norm="ortho")
            hidden_states = hidden_states.transpose(1, 2) # [B, N, L, D]
        # FFT on L dimension: [B, N, L, D] -> [B, N, L//2+1, D]
        hidden_states = torch.fft.rfft(hidden_states, dim=2, norm="ortho")
        hidden_states = self.temporal_mlp(hidden_states)
        # IFFT on L dimension: [B, N, L//2+1, D] -> [B, N, L, D]
        hidden_states = torch.fft.irfft(hidden_states, n=L, dim=2, norm="ortho")
        hidden_states = hidden_states + residual
        prediction = self.fc(hidden_states.reshape(B, N, -1)).permute(0, 2, 1)
        return prediction


class FreMLP(nn.Module):
    """
    frequency-domain MLPs

    Attributes:
        real (nn.Parameter): the real part of weights
        imag (nn.Parameter): the imaginary part of weights
        real_bias (nn.Parameter): the real part of bias
        imag_bias (nn.Parameter): the imaginary part of bias
        sparsity_threshold (float): the threshold for sparsity
    """

    def __init__(
            self,
            embed_size: int,
            scale: float = 0.02,
            sparsity_threshold: float = 0.01):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.real = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.imag = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.real_bias = nn.Parameter(scale * torch.randn(embed_size))
        self.imag_bias = nn.Parameter(scale * torch.randn(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        o1_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, self.real) - \
            torch.einsum("bijd,dd->bijd", x.imag, self.imag) + \
            self.real_bias
        )

        o1_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, self.real) + \
            torch.einsum("bijd,dd->bijd", x.real, self.imag) + \
            self.imag_bias
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y
